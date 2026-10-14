"""Leaf: should the conversation be handed to a human now? (是否转人工)

Two sources:

- ABCD: real customer-service chats (crowdworkers role-playing customers and trained agents who
  followed written guidelines) in which the agent escalated the case to a manager. Each chat is cut
  right before the agent starts the escalation (positive) and at earlier points where the agent was
  still identifying the customer, before any policy answer or mention of a manager (negatives).
- Scope cards: human intent labels of real chatbot queries (HINT3) and customer-service requests
  (NatCS) paired with a runtime "scope card" that lists what the assistant handles. The fixed rule
  (transfer if the user asks for a human or the request is not on the card) and the human label
  determine the answer; every request is asked once with a card that covers it and once with a
  card that does not.
"""

from __future__ import annotations

import csv
import gzip
import json
import re

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "dialog/human_handoff"

# ---------- ABCD: escalation to a manager ----------

ABCD_URL = "https://raw.githubusercontent.com/asappresearch/abcd/master/data/abcd_v1.1.json.gz"
# The two escalation instructions of the ABCD agent guidelines (data/guidelines.json), verbatim:
# Subscription Inquiry > Manage Extension, and Product Defect > Refund Status.
ABCD_GUIDELINE_EXTENSION = (
    "(Option B) If they do not get an extension and escalate to the manager: Ask for their phone number so they have a "
    "number you can call them on; Put the phone number into [Enter Details]; Then let them know, the manager will call "
    "them back later; Enter 'manager' into the [Notify Internal Team] action"
)
ABCD_GUIDELINE_REFUND = (
    "If the customer is not satisfied with your answer, and would like to change their refund: If they do not like the "
    "status: > Enter 'manager' into [Notify Internal Team] > Explain that you have escalated the issue to the manager"
)
ABCD_QUESTION = (
    "A customer is chatting with an agent of AcmeBrands, an online clothing store. The agent follows the company's "
    "written guidelines: identify the customer and the order, check what the store policy and the customer's membership "
    "level allow, and then resolve the request or explain why it cannot be granted. The guidelines say when to escalate "
    "a case to a manager, who calls the customer back, for example: \"" + ABCD_GUIDELINE_EXTENSION + "\" and \""
    + ABCD_GUIDELINE_REFUND + "\". In short: escalate once the customer asks for a manager, or once the customer's "
    "request has been refused or cannot be met under the policy; do not escalate while the agent can still handle the "
    "request itself, e.g. while it is still identifying the customer or has not yet checked the policy and given its "
    "answer. Lines starting with 'Action:' are the agent's system actions. Based on the conversation so far, should the "
    "agent escalate the case to a manager now?"
)

# Escalation words in the agent's messages, and the logged notification of the manager.
ABCD_ESCALATION = re.compile(r"\bmanag(er|ers|ement)\b|\bmanger\b|\bmanafer\b|supervisor|escalat|higher up|someone higher|superior", re.I)
# Once the trigger has happened, these start the escalation too: taking a phone number for the call back, notifying, forwarding.
ABCD_START = re.compile(
    r"phone|your number|note your number|call you|call back|contact you|reach (out|you)|get in touch|forward|transfer you|"
    r"proper team|notify|put in a request|refer (this|you|it)|let (my|the|a|our) \w+ know|someone (higher|else)|"
    r"one of our team|member of our team|internal team",
    re.I,
)
ABCD_MANAGER_ACTION = re.compile(r"has been notified", re.I)
ABCD_MANAGER_VALUE = re.compile(r"manag|manger|escalate issue", re.I)
# The customer asks for a manager or someone with more authority.
ABCD_ASK = re.compile(r"\bmanag(er|ers|ement)\b|supervisor|escalat|someone higher|higher up|superior|anyone else|someone else|overrule", re.I)
# The agent refuses, or says it cannot help, under the policy.
ABCD_REFUSAL = re.compile(
    r"unfor\w*nate|cannot|can't|can not|\bunable\b|not (be )?able|not eligible|n't eligible|n't qualify|not qualify|"
    r"not authori[sz]ed|n't have the (authority|authorization|ability)|not have (the|that) (authority|authorization|ability)|"
    r"no authority|not possible|nothing (else |more )?(that )?(i|we) can do|nothing i am able|hands are tied|"
    r"only (able|authorized|allowed) to|only (give|offer|do)\b|not (allowed|permitted)|n't allow|do not allow|"
    r"not (grant|offer|extend)|n't (grant|offer|extend)|no refund|billed correctly|only billed once|not been paid twice|"
    r"not a mistake|was not a mistake|too long ago|out of the return|outside of the|past the normal|do not see an issue|"
    r"does not look like you|there is not, sorry|i'm afraid|i am afraid",
    re.I,
)
# An agent message that offers or announces an escalation (not just any mention of a manager).
ABCD_OFFER = re.compile(r"escalat|call you|contact you|reach (out|you)|get in touch|speak|talk|notify|let (my|the|a|our) \w+ know|"
                        r"put in a request|refer|forward|have (a|my|the|our|one of our) \w+ (call|contact)|pass", re.I)
# Anything that ends the identification phase: a policy check or answer, a membership level, a manager.
ABCD_LEVEL = re.compile(r"\b(gold|silver|bronze|guest)\b|member ?level|membership level|level of membership|what level", re.I)
ABCD_IDENTIFICATION_ACTIONS = ("account has been pulled up", "identity verification in prog", "purchase validation in progress")


def abcd_lines(turns: list[list[str]]) -> str:
    names = {"agent": "Agent", "customer": "Customer", "action": "Action"}
    return "\n".join(f"{names[speaker]}: {' '.join(text.split())}" for speaker, text in turns)


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def abcd_starts_escalation(text: str) -> bool:
    return bool(ABCD_ESCALATION.search(text) or ABCD_START.search(text))


def abcd_escalation_point(turns: list[list[str]]) -> tuple[int, str] | None:
    """Where the agent starts the escalation: (index of the turn, kept prefix of that turn's text or "").

    The escalation starts at the first agent message after the trigger (the customer asked for a manager, or the
    agent refused) that mentions a manager or escalating, asks for a phone number for the call back, or announces
    notifying or forwarding; or at the logged manager notification if that comes first. When the escalating message
    opens with the refusal, the sentences (or clauses) before the escalation are kept and shown; a bare
    acknowledgement ("Of course.") is not. Returns None when there is no rule-based escalation point."""
    notify = next((i for i, (s, t) in enumerate(turns) if s == "action" and ABCD_MANAGER_ACTION.search(t) and ABCD_MANAGER_VALUE.search(t)), None)
    first_customer = next((i for i, (s, _) in enumerate(turns) if s == "customer"), None)
    if notify is None or first_customer is None:
        return None
    for i in range(first_customer + 1, notify + 1):
        speaker, text = turns[i]
        if i == notify:
            return (i, "") if abcd_triggered(turns[:i]) else None
        if speaker != "agent" or not abcd_starts_escalation(text):
            continue
        prefix = []
        for sentence in split_sentences(text):
            if abcd_starts_escalation(sentence):
                # the refusal may open the escalating sentence: "Sorry, I can't do anything for you, but I can escalate it"
                lead = []
                for clause in re.split(r"(?<=,)\s+", sentence):
                    if abcd_starts_escalation(clause):
                        break
                    lead.append(clause)
                if lead and ABCD_REFUSAL.search(" ".join(lead)):
                    prefix += lead
                break
            prefix.append(sentence)
        kept = " ".join(prefix) if ABCD_REFUSAL.search(" ".join(prefix)) else ""
        if abcd_triggered(turns[:i] + ([["agent", kept]] if kept else [])):
            return i, kept
        if ABCD_ESCALATION.search(text) and ABCD_OFFER.search(text):
            return None  # an escalation offered before any refusal or request: the escalation point is not rule-based
        # otherwise a mention of a manager that is not an escalation ("my manager would also like to know ..."), or an
        # identification step such as asking for a phone number
    return None


def abcd_triggered(before: list[list[str]]) -> bool:
    """The guideline's trigger has happened: the customer asked for a manager, or the agent refused (recently)."""
    asked = any(s == "customer" and ABCD_ASK.search(t) for s, t in before)
    refused = any(s == "agent" and ABCD_REFUSAL.search(t) for s, t in before[-8:])
    return asked or refused


def abcd_negative_cuts(turns: list[list[str]], limit: int) -> dict[str, list[int]]:
    """Cut points (number of turns shown) during identification, by who spoke last: after the customer has stated
    the request (a message of four or more words) and before any policy check, policy answer, membership level,
    refusal or mention of a manager; each cut is where a speaker finished their turn."""
    cuts: dict[str, list[int]] = {"customer": [], "agent": []}
    stated = False
    for i, (speaker, text) in enumerate(turns[:limit]):
        low = text.lower()
        if speaker == "action" and not low.startswith(ABCD_IDENTIFICATION_ACTIONS):
            break
        if speaker != "action" and (ABCD_LEVEL.search(text) or ABCD_ESCALATION.search(text) or ABCD_ASK.search(text)):
            break
        if speaker == "agent" and ABCD_REFUSAL.search(text):
            break
        if speaker == "customer" and len(text.split()) >= 4:
            stated = True
        if stated and speaker != "action" and i + 1 < limit and turns[i + 1][0] != speaker:
            cuts[speaker].append(i + 1)
    return cuts


@register_task
class ABCDEscalation(TaskAdapter):
    name = f"{LEAF}/abcd"
    leaf = LEAF
    source = "ABCD, Action-Based Conversations Dataset: customer-service chats escalated to a manager, all splits (Chen et al., 2021)"
    url = "https://github.com/asappresearch/abcd"
    license = "MIT"
    tier = "B"
    criteria_source = ("The two escalation instructions of the ABCD agent guidelines (Manage Extension, Option B; Refund Status), "
                       "verbatim, with a one-sentence summary of when to escalate written for the task")
    items_per_record = 3
    notes = ("ABCD has 88 chats in which the agent notified a manager (train, dev and test together). A chat is used when, before "
             "the escalation, the customer asked for a manager or the agent refused the request under the policy (rule-based "
             "check); this drops chats where the manager was notified for another reason (a product suggestion, a note on the "
             "account, a question the agent could have answered, or a logging error). Positive: the chat up to the agent's "
             "first escalating message (or the logged notification, if earlier); when that message opens with the refusal, "
             "the sentences before the escalation are kept. Negatives: two earlier cuts (one after a customer turn and one "
             "after an agent turn, where available) while the agent is still identifying the customer: after the request "
             "is stated and before any policy check, policy answer, membership level, refusal or mention of a manager. "
             "Agent system actions are shown as 'Action:' lines. Customer names, IDs and phone numbers come from ABCD's "
             "synthetic scenarios. 75 chats qualify, so the source has 75 positives and 145 negatives.")

    def load(self):
        data = json.load(gzip.open(fetch(ABCD_URL, "abcd_v1.1.json.gz")))
        records = []
        for split in ("train", "dev", "test"):
            for convo in data[split]:
                values = [t["targets"][3] for t in convo["delexed"] if t["targets"][2] == "notify-team"]
                if not any(v and re.search(r"manag|manger|escalat|supervis", v[0], re.I) for v in values):
                    continue
                point = abcd_escalation_point(convo["original"])
                if point is None:
                    continue
                index, kept = point
                records.append({
                    "split": split, "convo_id": convo["convo_id"], "subflow": convo["scenario"]["subflow"],
                    "turns": convo["original"], "escalation_index": index, "escalation_prefix": kept,
                    "negative_cuts": abcd_negative_cuts(convo["original"], index),
                })
        return records

    def label_of(self, record):
        return record["subflow"] if record["subflow"] in ("manage_extension", "manage_dispute_bill") else "other"

    def to_items(self, record, rng):
        question = Question(type="boolean", instructions=ABCD_QUESTION)
        turns = record["turns"]
        shown = turns[:record["escalation_index"]] + ([["agent", record["escalation_prefix"]]] if record["escalation_prefix"] else [])
        meta = {"convo_id": record["convo_id"], "abcd_split": record["split"], "subflow": record["subflow"]}
        items = [self.item({"conversation": abcd_lines(shown)}, {"escalate": question}, {"escalate": Gold(label="true")},
                           cut="escalation", turns_shown=len(shown), cut_inside_message=bool(record["escalation_prefix"]), **meta)]
        # two negatives: one right after the customer and one right after the agent, where both exist
        by_speaker = {k: sorted(v) for k, v in record["negative_cuts"].items()}
        chosen = [rng.choice(v) for v in by_speaker.values() if v]
        rest = sorted(set(c for v in by_speaker.values() for c in v) - set(chosen))
        rng.shuffle(rest)
        chosen = (chosen + rest)[:2]
        for cut in sorted(chosen):
            items.append(self.item({"conversation": abcd_lines(turns[:cut])}, {"escalate": question}, {"escalate": Gold(label="false")},
                                   cut="identification", turns_shown=cut, cut_inside_message=False, **meta))
        rng.shuffle(items)
        return items


# ---------- Scope cards over human intent labels (HINT3, NatCS) ----------

HINT3_URL = "https://raw.githubusercontent.com/hellohaptik/HINT3/master/dataset/v2/test/{bot}_test.csv"
NATCS_URL = "https://raw.githubusercontent.com/amazon-science/dstc11-track2-intent-induction/main/dstc11/{part}/test-utterances.jsonl"

# Per source: who the assistant is, and every intent that can appear on a card as (topic family, card entry).
# An out-of-scope card never lists an intent of the request's own family, so a request never has a near-synonym on
# the card that is supposed not to cover it. Families group intents whose queries overlap in the data.
SCOPE_SOURCES: dict[str, dict] = {
    "hint3_sofmattress": {
        "assistant": "the website chat assistant of SOF Mattress, an Indian mattress brand",
        "intents": {
            "ABOUT_SOF_MATTRESS": ("products", "General information about SOF Mattress and its mattresses"),
            "PRODUCT_VARIANTS": ("products", "Which mattress models are available, and help choosing one"),
            "COMPARISON": ("products", "Comparing the SOF Ergo and SOF Ortho mattresses"),
            "ERGO_FEATURES": ("products", "Features of the SOF Ergo mattress"),
            "ORTHO_FEATURES": ("products", "Features of the SOF Ortho (orthopaedic) mattress"),
            "PILLOWS": ("products", "Buying pillows"),
            "WHAT_SIZE_TO_ORDER": ("products", "Working out which mattress size to order for a bed"),
            "SIZE_CUSTOMIZATION": ("products", "Ordering a mattress in a custom size"),
            "MATTRESS_COST": ("products", "Mattress prices"),
            "OFFERS": ("products", "Current offers and discounts"),
            "EMI": ("products", "Paying in EMI instalments"),
            "COD": ("products", "Cash on delivery"),
            "ORDER_STATUS": ("orders", "The status of an order that has been placed"),
            "DELAY_IN_DELIVERY": ("orders", "An order that is late or has not arrived yet"),
            "CANCEL_ORDER": ("orders", "Cancelling an order"),
            "RETURN_EXCHANGE": ("orders", "Returns, exchanges and refunds"),
            "100_NIGHT_TRIAL_OFFER": ("orders", "The 100-night free trial"),
            "CHECK_PINCODE": ("orders", "Whether delivery is available to a PIN code or town"),
            "DISTRIBUTORS": ("orders", "Showrooms, stores and distributors in a city"),
        },
        "human_labels": ("LEAD_GEN",),
    },
    "hint3_curekart": {
        "assistant": "the chat assistant of CureKart, an Indian online store for health supplements and fitness products",
        "intents": {  # ORDER_TAKING is not used: its queries straddle product requests and delivery questions
            "RECOMMEND_PRODUCT": ("shopping", "Finding and recommending products"),
            "SIDE_EFFECT": ("shopping", "Side effects of products"),
            "CONSULT_START": ("shopping", "Diet and fitness advice, such as diet plans"),
            "USER_GOAL_FORM": ("shopping", "Filling in or retaking the fitness goal assessment"),
            "ORDER_STATUS": ("orders", "Tracking an order and its delivery date"),
            "DELAY_IN_PARCEL": ("orders", "Late or missing deliveries"),
            "RESUME_DELIVERY": ("orders", "Delivery times and whether deliveries are running"),
            "CHECK_PINCODE": ("orders", "Whether delivery is available to a PIN code or town"),
            "MODIFY_ADDRESS": ("orders", "Changing the delivery address of an order"),
            "CANCEL_ORDER": ("orders", "Cancelling an order"),
            "REFUNDS_RETURNS_REPLACEMENTS": ("orders", "Refunds, returns and replacements"),
            "PAYMENT_AND_BILL": ("orders", "Payments and invoices"),
            "ORIGINAL_PRODUCT": ("orders", "Whether products are genuine"),
            "EXPIRY_DATE": ("orders", "Expiry dates of products"),
            "SIGN_UP": ("company", "Signing up as a new user"),
            "FRANCHISE": ("company", "Becoming a seller or franchise partner"),
            "STORE_INFORMATION": ("company", "CureKart's physical stores"),
        },
        "human_labels": ("CHAT_WITH_AGENT", "CALL_CENTER"),
    },
    "hint3_powerplay11": {
        "assistant": "the chat assistant of PowerPlay11, an Indian fantasy sports app",
        "intents": {
            "ACCOUNT_BALANCE_DEDUCTED": ("money", "How the account balance is deducted when joining a contest"),
            "CHECK_DEPOSIT_STATUS": ("money", "Deposit status and deposit history"),
            "CHECK_WALLET_BALANCE": ("money", "Wallet balance"),
            "DEDUCTED_AMOUNT_NOT_RECEIVED": ("money", "Money debited from the bank but not added to the wallet"),
            "REFUND_OF_ADDED_CASH": ("money", "Refunds of added cash"),
            "REFUND_OF_WRONG_AMOUNT": ("money", "Refunds of an amount added by mistake"),
            "UNUTILIZED_MONEY": ("money", "Unutilized money in the wallet"),
            "WINNINGS": ("money", "Winnings not received"),
            "LESS_WINNINGS_AMOUNT": ("money", "Winnings lower than expected"),
            "TAXES_ON_WINNINGS": ("money", "Taxes on winnings"),
            "INSTANT_WITHDRAWAL": ("money", "Instant withdrawal"),
            "WITHDRAWAL_INTRO": ("money", "How withdrawals work and withdrawal limits"),
            "WITHDRAWAL_STATUS": ("money", "Status of a withdrawal"),
            "WITHDRAWAL_TIME": ("money", "How long withdrawals take"),
            "WITHDRAW_CASH_BONUS": ("money", "Withdrawing a cash bonus"),
            "CASH_BONUS": ("money", "Cash bonus"),
            "CASH_BONUS_EXPIRY": ("money", "When a cash bonus expires"),
            "SIGNUP_BONUS": ("money", "Sign-up bonus"),
            "TYPES_BONUS": ("money", "The types of bonuses"),
            "OFFERS_AND_REFERRALS": ("money", "Offers and referral programmes"),
            "MATCH_ABANDONED": ("money", "What happens when a match is abandoned"),
            "CANNOT_SEE_JOINED_CONTESTS": ("money", "Joined contests that do not show up"),
            "HOW_POINTS_CALCULATED": ("money", "How fantasy points are calculated"),
            "POINTS_NOT_UPDATED": ("money", "Player points that have not been updated"),
            "WRONG_SCORES": ("money", "Wrong scores or points"),
            "HOW_TO_PLAY": ("money", "How to play"),
            "JOIN_CONTEST": ("money", "Joining a contest"),
            "NEW_TEAM_PATTERN": ("money", "Team composition rules"),
            "TEAM_DEADLINE": ("money", "Team deadlines, safe play and regular play"),
            "TYPES_CONTESTS": ("money", "The types of contests"),
            "WHAT_IF_THERES_A_TIE": ("money", "What happens in a tie"),
            "FAIRPLAY_VIOLATIONS": ("money", "Fair play violations"),
            "FAKE_TEAMS": ("money", "Complaints about fake teams or fake users"),
            "ACCOUNT_NOT_VERIFIED": ("account", "Verifying the account"),
            "ACCOUNT_RESET": ("account", "Resetting the account"),
            "BANK_VERIFICATION_DETAILS": ("account", "Bank account verification"),
            "CHANGE_BANK_ACCOUNT": ("account", "Changing the bank account"),
            "CHANGE_MOBILE_NUMBER": ("account", "Changing the mobile number"),
            "CHANGE_PROFILE_TEAM_DETAILS": ("account", "Changing profile or team details"),
            "DELETE_PAN_CARD": ("account", "Removing or changing the PAN card"),
            "PAN_VERIFICATION_FAILED": ("account", "PAN card verification errors"),
            "VERIFY_EMAIL": ("account", "Verifying the email address"),
            "VERIFY_PAN": ("account", "Verifying the PAN card"),
            "NO_EMAIL_CONFIRMATION": ("account", "Email confirmation that did not arrive"),
            "DOWNLOAD_POWERPLAY11": ("app", "Downloading and installing the app"),
            "UPDATE_APP": ("app", "Updating the app"),
        },
        "human_labels": ("CHAT_WITH_AN_AGENT", "CONTACT_NUMBER"),
    },
    "natcs_insurance": {
        "assistant": "the customer-service chat assistant of an insurance company",
        "intents": {
            "GetQuote": ("policies", "Getting an insurance quote"),
            "EnrollInPlan": ("policies", "Enrolling in a new insurance plan"),
            "ChangePlan": ("policies", "Changing to a different insurance plan"),
            "CancelPlan": ("policies", "Cancelling an insurance plan"),
            "AddDependent": ("policies", "Adding a dependent to a plan"),
            "RemoveDependent": ("policies", "Removing a dependent from a plan"),
            "RequestProofOfInsurance": ("policies", "Getting proof of insurance"),
            "GetPolicyNumber": ("policies", "Looking up a policy number"),
            "FileClaim": ("claims", "Filing an insurance claim"),
            "ReportAutomobileAccident": ("claims", "Reporting a car accident"),
            "PayBill": ("billing", "Paying a bill"),
            "ReportBillingIssue": ("billing", "Reporting a problem with a bill"),
            "UpdatePaymentPreference": ("billing", "Changing the billing preference"),
            "CheckPaymentStatus": ("billing", "Checking whether a payment went through"),
            "CancelAutomaticBilling": ("billing", "Cancelling automatic billing"),
            "UpdateBillingFrequency": ("billing", "Changing how often you are billed"),
            "CheckAccountBalance": ("billing", "Checking the account balance"),
            "CreateAccount": ("account", "Creating an account"),
            "ResetPassword": ("account", "Resetting the account password"),
            "ChangeSecurityQuestion": ("account", "Changing the account security question"),
            "ChangeAddress": ("account", "Changing your address"),
        },
        "human_labels": (),
    },
    "natcs_banking": {
        "assistant": "the customer-service chat assistant of a bank",
        "intents": {
            "CheckAccountBalance": ("accounts", "Checking an account balance"),
            "CheckTransactionHistory": ("accounts", "Viewing recent transactions and charges"),
            "DisputeCharge": ("accounts", "Disputing a charge or reporting a fraudulent transaction"),
            "ReportLostStolenCard": ("accounts", "Reporting a lost or stolen card"),
            "GetWithdrawalLimit": ("accounts", "Daily withdrawal limits"),
            "OpenBankingAccount": ("accounts", "Opening a new bank account"),
            "CloseBankAccount": ("accounts", "Closing a bank account"),
            "SetUpOnlineBanking": ("accounts", "Setting up online banking"),
            "InternalFundsTransfer": ("accounts", "Transferring money between accounts at this bank"),
            "ExternalWireTransfer": ("accounts", "Sending a wire transfer to another bank"),
            "AskAboutTransferTime": ("accounts", "How long transfers and payments take to go through"),
            "FindATM": ("locations", "Finding the nearest ATM"),
            "AskAboutCashDeposits": ("locations", "Depositing cash at an ATM"),
            "FindBranch": ("locations", "Finding the nearest branch"),
            "GetBranchHours": ("locations", "Branch opening hours"),
            "UpdatePhoneNumber": ("contact", "Updating your phone number"),
            "UpdateEmail": ("contact", "Updating your email address"),
            "UpdateStreetAddress": ("contact", "Updating your street address"),
        },
        "human_labels": (),
    },
    "natcs_finance": {
        "assistant": "the customer-service chat assistant of a bank for personal and small-business customers",
        "intents": {
            "ApplyLoan": ("banking", "Applying for a loan"),
            "PayLoan": ("banking", "Making or scheduling a loan payment"),
            "CheckLoanBalance": ("banking", "Checking a loan balance"),
            "GetLoanInfo": ("banking", "Information about the loans offered"),
            "ApplyCreditCard": ("banking", "Applying for a credit card or a line of credit"),
            "MakeCreditCardPayment": ("banking", "Making or scheduling a credit card payment"),
            "GetCreditCardInfo": ("banking", "Information about the credit cards offered"),
            "CheckCreditCardBalance": ("banking", "Checking the amount due on a credit card"),
            "RequestNewCard": ("banking", "Requesting a replacement card"),
            "ChangePin": ("banking", "Changing a card PIN"),
            "GetCreditReport": ("banking", "Ordering a credit report"),
            "SetAutoPayment": ("banking", "Setting up automatic payments"),
            "OpenAccount": ("banking", "Opening a checking or savings account"),
            "CloseAccount": ("banking", "Closing an account"),
            "CheckAccountBalance": ("banking", "Checking an account balance"),
            "GetBankStatement": ("banking", "Getting a bank or loan statement"),
            "ChangeStatementDelivery": ("banking", "Changing how statements are delivered"),
            "AddUserToAccount": ("banking", "Adding a user to an account"),
            "MakeTransfer": ("banking", "Making a wire or bank transfer"),
            "OrderCheck": ("banking", "Ordering checks"),
            "CancelCheck": ("banking", "Cancelling a check"),
            "OnlineBankingInfo": ("banking", "Online banking and the banking app"),
            "AskConsumerPriceIndex": ("markets", "Consumer price index figures"),
            "GetExchangeRate": ("markets", "Currency exchange rates"),
            "GetTreasuryBondYield": ("markets", "Treasury bond yields"),
            "AskLiquidityRatio": ("markets", "A business's liquidity ratio"),
            "GetDebtIncomeRatio": ("markets", "Debt-to-income ratio"),
            "NetIncome": ("markets", "Net income figures"),
            "PurchaseStocks": ("markets", "Buying stocks"),
            "GetStockQuote": ("markets", "Stock quotes"),
            "GetInvestmentReport": ("markets", "Investment reports"),
            "GetBranchHours": ("branches", "Branch and office opening hours"),
            "FindBranch": ("branches", "Finding a branch or office"),
            "UpdateEmail": ("contact", "Updating your email address"),
            "UpdatePhoneNumber": ("contact", "Updating your phone number"),
            "UpdateStreetAddress": ("contact", "Updating your street address"),
        },
        "human_labels": (),
    },
}
NATCS_PARTS = {"natcs_insurance": "development", "natcs_banking": "test-banking", "natcs_finance": "test-finance"}
# The user asks for a human: an agent, representative or customer care, or a call back.
HUMAN_REQUEST = re.compile(
    r"\b(agent|agrnt|human|representative|executive|expert|customer ?care|customer caee|customer service|coustmer|helpline|"
    r"call|contact me|contact number|contect number|ph number|phone number|talk to|talk with|chat with|"
    r"connect to|connect an|speak to|speak with|someone|in person)\b",
    re.I,
)
SCOPE_QUESTION = (
    "A customer-service chat assistant works from a scope card (given in the input) that lists the only requests it can "
    "handle itself. Hand-off rule: transfer the conversation to a human agent if the user asks for a human (an agent, a "
    "representative, customer care or a call back), or if the user's request is not covered by any entry on the scope "
    "card; otherwise the assistant handles the request itself. Given the user's message, should the conversation be "
    "transferred to a human agent now?"
)


def scope_card_text(source: str, intents: list[str]) -> str:
    spec = SCOPE_SOURCES[source]
    lines = [f"Assistant: {spec['assistant']}.", "It can handle only these requests:"]
    return "\n".join(lines + [f"- {spec['intents'][i][1]}" for i in intents])


@register_task
class ScopeCardHandoff(TaskAdapter):
    name = f"{LEAF}/scope_cards"
    leaf = LEAF
    source = ("Scope cards over human intent labels: HINT3 v2 test sets, real queries to three chatbots (Arora et al., 2020), "
              "and NatCS customer-service requests in insurance, banking and finance (Gung et al., 2023)")
    url = "https://github.com/hellohaptik/HINT3"
    license = "HINT3: ODbL 1.0 (database) and DbCL 1.0 (contents); NatCS (DSTC11 track 2 data): CDLA-Permissive-2.0"
    tier = "C"
    criteria_source = "Written: hand-off rule, card format and card entries (from the intent names and their training examples)"
    items_per_record = 2
    sampling = "balanced"
    notes = ("Correct by construction from the human intent labels (HINT3: first messages real users sent to three live "
             "chatbots, labelled by several annotators with 73-80% agreement and conflicts resolved by domain experts; NatCS: "
             "labelled by annotators trained in linguistics). Each request is asked twice: with a card of four to "
             "seven entries that includes its labelled intent (answer: no transfer) and with an equally long card drawn "
             "only from other topic families of the same assistant (answer: transfer); families group intents whose "
             "queries overlap, so a card never lists a near-synonym of the request's intent. Requests labelled with a "
             "hand-off intent (HINT3 LEAD_GEN, CHAT_WITH_AGENT, CALL_CENTER, CHAT_WITH_AN_AGENT, CONTACT_NUMBER) whose text "
             "explicitly asks for an agent, customer care or a call back are asked with two random cards (answer: transfer "
             "both times); other requests that mention a human are left out, as are out-of-scope, greeting and thanks "
             "labels, and intents whose requests ask for a person (NatCS FindAgent, ScheduleAppointment) or a confirmation "
             "email (RequestEmail). Exact duplicate texts are merged; texts with two labels are dropped. Sampled evenly "
             "across the six assistants and the hand-off requests. NatCS data is CDLA-Permissive-2.0, which asks that "
             "shared copies come with the agreement text (dstc11/LICENSE.md in the source repository).")

    def load(self):
        rows = []  # (source, text, label)
        for bot in ("sofmattress", "curekart", "powerplay11"):
            path = fetch(HINT3_URL.format(bot=bot), f"hint3_v2_{bot}_test.csv")
            with path.open(encoding="utf-8") as f:
                rows += [(f"hint3_{bot}", r["sentence"], r["label"]) for r in csv.DictReader(f)]
        for source, part in NATCS_PARTS.items():
            path = fetch(NATCS_URL.format(part=part), f"dstc11_{part}_test-utterances.jsonl")
            with path.open(encoding="utf-8") as f:
                rows += [(source, r["utterance"], r["intent"]) for r in map(json.loads, filter(str.strip, f))]
        labels: dict[tuple[str, str], set[str]] = {}
        first: dict[tuple[str, str], str] = {}
        for source, text, label in rows:
            key = (source, " ".join(text.lower().split()))
            labels.setdefault(key, set()).add(label)
            first.setdefault(key, " ".join(text.split()))
        records = []
        for key, label_set in labels.items():
            if len(label_set) != 1:
                continue
            source, label, text = key[0], next(iter(label_set)), first[key]
            spec = SCOPE_SOURCES[source]
            if label in spec["human_labels"] and HUMAN_REQUEST.search(text):
                kind = "human"
            elif label in spec["intents"] and not HUMAN_REQUEST.search(text):
                kind = "pair"
            else:
                continue
            records.append({"source": source, "utterance": text, "label": label, "kind": kind})
        records.sort(key=lambda r: (r["source"], r["utterance"]))
        return records

    def label_of(self, record):
        return "hint3_handoff_requests" if record["kind"] == "human" else record["source"]

    def to_items(self, record, rng):
        source, label = record["source"], record["label"]
        intents = sorted(SCOPE_SOURCES[source]["intents"])
        families = {i: SCOPE_SOURCES[source]["intents"][i][0] for i in intents}
        size = rng.randint(4, 7)
        if record["kind"] == "pair":
            others = [i for i in intents if families[i] != families[label]]
            size = min(size, len(others))
            covering = [label] + rng.sample([i for i in intents if i != label], size - 1)
            not_covering = rng.sample(others, size)
            cards = [(covering, False, "covers"), (not_covering, True, "does_not_cover")]
        else:
            cards = [(rng.sample(intents, size), True, "handoff_request"), (rng.sample(intents, size), True, "handoff_request")]
        question = Question(type="boolean", instructions=SCOPE_QUESTION)
        items = []
        for card, transfer, kind in cards:
            rng.shuffle(card)
            state = {"scope_card": scope_card_text(source, card), "user_message": record["utterance"]}
            items.append(self.item(state, {"transfer": question}, {"transfer": Gold(label="true" if transfer else "false")},
                                   dataset=source, intent=label, card=kind, card_intents=card))
        rng.shuffle(items)
        return items
