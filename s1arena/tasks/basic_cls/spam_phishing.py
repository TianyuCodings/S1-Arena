"""Leaf: spam and phishing (垃圾信息 / 钓鱼)."""

from __future__ import annotations

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows

LEAF = "basic_cls/spam_phishing"

# TREC 2005 Spam Track overview (Cormack and Lynam, 2005), verbatim; the SMS task adapts "email" to "message".
TREC_SPAM = (
    "unsolicited, unwanted email that was sent indiscriminately, directly or indirectly, by a sender having no current "
    "relationship with the recipient"
)


@register_task
class SMSSpam(TaskAdapter):
    name = f"{LEAF}/sms_spam"
    leaf = LEAF
    source = "SMS Spam Collection (Almeida et al., 2011)"
    url = "https://huggingface.co/datasets/ucirvine/sms_spam"
    license = "CC BY 4.0 (UCI)"
    tier = "B"
    criteria_source = "TREC 2005 spam definition, adapted from email to text messages"
    notes = "Exact duplicate messages removed before sampling (the collection has no official test split)."

    def load(self):
        seen, rows = set(), []
        for r in hf_rows("ucirvine/sms_spam", "plain_text", "train"):
            key = r["sms"].strip()
            if key and key not in seen:
                seen.add(key)
                rows.append(r)
        return rows

    def label_of(self, record):
        return record["label_name"]

    def to_item(self, record, rng):
        question = Question(
            type="boolean",
            instructions="Is this text message spam? Spam is " + TREC_SPAM.replace("email", "messages") + ".",
        )
        gold = Gold(label="true" if record["label_name"] == "spam" else "false")
        return self.item({"message": record["sms"].strip()}, {"spam": question}, {"spam": gold})


@register_task
class EnronSpam(TaskAdapter):
    name = f"{LEAF}/enron_spam"
    leaf = LEAF
    source = "Enron-Spam (Metsis et al., 2006)"
    url = "https://huggingface.co/datasets/SetFit/enron_spam"
    license = "Not stated upstream"
    redistributable = False
    tier = "B"
    criteria_source = "TREC 2005 spam definition"
    notes = "Test texts that also appear in the training split, and duplicates, are removed."

    def load(self):
        train = {r["text"].strip() for r in hf_rows("SetFit/enron_spam", None, "train")}
        seen, rows = set(), []
        for r in hf_rows("SetFit/enron_spam", None, "test"):
            key = r["text"].strip()
            if key and key not in train and key not in seen:
                seen.add(key)
                rows.append(r)
        return rows

    def label_of(self, record):
        return record["label_text"]

    def to_item(self, record, rng):
        question = Question(type="boolean", instructions="Is this email spam? Spam is " + TREC_SPAM + ".")
        gold = Gold(label="true" if record["label_text"] == "spam" else "false")
        state = {"subject": (record["subject"] or "").strip(), "body": (record["message"] or "").strip()}
        return self.item(state, {"spam": question}, {"spam": gold}, original_id=record["message_id"])
