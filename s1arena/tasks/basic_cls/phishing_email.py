"""Leaf: spam and phishing (垃圾信息 / 钓鱼) -- three-way email source: ham / spam / phishing.

Raw emails from the SpamAssassin public corpus (ham and spam) and Jose Nazario's phishing
corpus (phishing), labelled by corpus membership. Each message is parsed into subject, sender
domain and body text: MIME parts are decoded, HTML is converted to text (link targets kept in
parentheses), attachments are dropped. Texts that carry an answer (a spam filter's verdict, the
SpamAssassin "sightings" list footer, spam tags in subjects) are removed or cleaned, and exact
and near-duplicate messages are collapsed; a message whose text occurs in more than one class is
dropped entirely.
"""

from __future__ import annotations

import email
import email.header
import email.policy
import email.utils
import html.parser
import re
import tarfile
import zlib
from collections import defaultdict

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "basic_cls/spam_phishing"

SA_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SA_PARTS = (  # latest version of each part (the 20021010 files are older copies of the same messages)
    ("20030228_easy_ham", "ham"),
    ("20030228_easy_ham_2", "ham"),
    ("20030228_hard_ham", "ham"),
    ("20030228_spam", "spam"),
    ("20050311_spam_2", "spam"),
)
NAZARIO_URL = "https://monkey.org/~jose/phishing/"
# The 2004-2006 mailboxes, closest in time to the SpamAssassin corpus (2002-2005). phishing3.mbox
# (2006-2007) is left out: the malware scanner on the build machine deletes it on download.
NAZARIO_PARTS = ("phishing0", "phishing1", "phishing2", "20051114")

LONG_BODY = 20_000  # characters; longer bodies are cut to their first KEEP_CHARS characters
KEEP_CHARS = 6_000
TRUNCATION_MARK = "\n[... message truncated]"
MIN_BODY = 30  # characters of text needed to classify a message (image- or attachment-only mails are dropped)

# TREC 2005 Spam Track overview (Cormack and Lynam, 2005), verbatim.
TREC_SPAM = (
    "unsolicited, unwanted email that was sent indiscriminately, directly or indirectly, by a sender having no current "
    "relationship with the recipient"
)
# APWG Phishing Activity Trends Reports, "Phishing Defined" (e.g. 4th Quarter 2016), first two sentences verbatim.
APWG_PHISHING = (
    "Phishing is a criminal mechanism employing both social engineering and technical subterfuge to steal consumers' "
    "personal identity data and financial account credentials. Social engineering schemes use spoofed e-mails purporting "
    "to be from legitimate businesses and agencies, designed to lead consumers to counterfeit websites that trick "
    "recipients into divulging financial data such as usernames and passwords."
)
OPTIONS = {
    "ham": "Ham: a legitimate email, neither spam nor phishing (for example personal or work correspondence, a mailing-list "
    "post, or a newsletter the recipient subscribed to).",
    "spam": "Spam that is not phishing. Spam is " + TREC_SPAM + ".",
    "phishing": "Phishing, as defined by the Anti-Phishing Working Group (APWG): \"" + APWG_PHISHING + "\"",
}

# Placeholders the corpus editors substituted for real addresses; they say nothing about the sender.
ANONYMIZED_DOMAINS = {"spamassassin.taint.org", "taint.org", "example.com", "example.net", "example.org"}
# Tags that a recipient's spam filter added to subjects, e.g. "[*** POSIBLE SPAM***]" or "****SPAM****".
SUBJECT_TAG = re.compile(r"\[\s*\*+\s*POSS?IBLE\s+SPAM\s*\*+\s*\]|\*{2,}\s*SPAM\s*\*{2,}|\[\s*SPAM\s*\]", re.I)
# Texts that reveal the answer: a spam filter's verdict, or the list to which SpamAssassin users forwarded spam samples.
REVEALING = re.compile(
    r"identified this incoming email as possible spam|content analysis details|x-spam-(flag|status):\s*yes|"
    r"spamassassin-sightings|deathtospamdeathtospam",
    re.I,
)
# Forwarded messages ("Fwd: ...", "[sender@host: subject]"): the forwarder's note often states the verdict ("more phishing").
FORWARDED = re.compile(r"^\s*(\[\s*)?(fwd?|fw)\s*:|^\s*\[[^\]\s]+@[^\]\s]+:", re.I)
# Mail addressed to the phishing collector by his registered name (genuine eBay newsletters) or written by him.
COLLECTOR = re.compile(r"nazario", re.I)
# Control characters (other than tab and newline) mean the text was not decoded properly.
GARBLED = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ufffd]")
# Mail-program bookkeeping stored in a mailbox (e.g. Pine's "FOLDER INTERNAL DATA" message), not a received email.
MAILBOX_ARTIFACT = re.compile(r"part of the internal format of your mail folder", re.I)


def mojibake(text: str) -> bool:
    """East Asian text in an undeclared charset decoded as Latin-1 / Windows-1252: mostly accented Latin symbols."""
    chars = [c for c in text if not c.isspace()]
    odd = sum(1 for c in chars if "\u0080" <= c <= "\u02ff" or "\u2010" <= c <= "\u2122")
    return len(chars) > 0 and odd > 0.3 * len(chars)


class _HTMLText(html.parser.HTMLParser):
    """HTML to plain text: block elements become line breaks, link targets follow the link text in parentheses."""

    BLOCK = {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "hr", "blockquote", "pre",
             "td", "th", "center", "form"}
    SKIP = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.links: list[tuple[str, int]] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag == "a":
            self.links.append(((dict(attrs).get("href") or "").strip(), len(self.out)))

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag == "a" and self.links:
            href, start = self.links.pop()
            shown = "".join(self.out[start:])
            if href and not href.lower().startswith(("mailto:", "#", "javascript:")) and href not in shown:
                self.out.append(f" ({href})")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(markup: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # malformed markup: keep what was parsed
        pass
    return "".join(parser.out)


def looks_like_html(text: str) -> bool:
    return len(re.findall(r"<\s*(html|body|table|font|div|p|br|td|tr|a\s+href|img)\b", text, re.I)) >= 3


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[\ufeff\u200b\u200c\u200d]", "", text)  # byte-order marks and zero-width characters
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_bytes(data: bytes, charset: str | None) -> str:
    """Declared charset first, then UTF-8, then Windows-1252; Latin-1 never fails."""
    for cs in (charset, "utf-8", "cp1252"):
        if not cs:
            continue
        try:
            return data.decode(cs)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("latin-1")


def decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    out = []
    try:
        pieces = email.header.decode_header(value)
    except Exception:
        pieces = [(value, None)]
    for piece, charset in pieces:
        if isinstance(piece, bytes):
            out.append(decode_bytes(piece, charset))
        else:  # raw 8-bit bytes arrive as surrogate escapes
            out.append(decode_bytes(piece.encode("utf-8", "surrogateescape"), None))
    return " ".join("".join(out).split())


def text_parts(part, found: list[tuple[str, str]]) -> None:
    """Collect (subtype, text) of the inline text parts; attachments (and forwarded messages attached as files) are skipped."""
    disposition = (part.get("Content-Disposition") or "").lower()
    if "attachment" in disposition:
        return
    if part.is_multipart():
        for sub in part.get_payload():
            text_parts(sub, found)
        return
    if part.get_content_maintype() != "text" or part.get_content_subtype() not in ("plain", "html"):
        return
    payload = part.get_payload(decode=True)
    if payload:
        found.append((part.get_content_subtype(), decode_bytes(payload, part.get_content_charset())))


def parse_email(raw: bytes) -> dict:
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    subject = SUBJECT_TAG.sub(" ", decode_header_value(msg.get("Subject")))
    subject = " ".join(subject.split())
    address = email.utils.parseaddr(decode_header_value(msg.get("From")))[1]
    domain = address.rsplit("@", 1)[1].strip(" <>.").lower() if "@" in address else ""
    if domain in ANONYMIZED_DOMAINS or not re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", domain):
        domain = ""
    found: list[tuple[str, str]] = []
    text_parts(msg, found)
    plain = clean_text("\n\n".join(html_to_text(t) if looks_like_html(t) else t for kind, t in found if kind == "plain"))
    rendered = clean_text("\n\n".join(html_to_text(t) for kind, t in found if kind == "html"))
    # Plain and HTML parts are usually alternatives of one text; the plain one is sometimes only a stub
    # ("your e-mail program does not support HTML"), so the longer rendering is used.
    body = max(plain, rendered, key=len)
    return {"subject": subject, "sender_domain": domain, "body": body}


def norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def shingles(text: str) -> set[str]:
    words = norm_words(text)
    return {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def near_duplicate_groups(texts: list[str], threshold: float = 0.8) -> list[int]:
    """Group index per text: texts whose word 5-shingle sets have Jaccard similarity >= threshold share a group
    (MinHash with 8 bands of 4 rows proposes candidate pairs, the exact Jaccard similarity decides)."""
    import numpy as np

    prime = (1 << 61) - 1
    rng = np.random.default_rng(20260924)
    a = rng.integers(1, 1 << 29, size=32, dtype=np.int64)  # crc32 (< 2**32) * a stays below 2**61: no overflow
    b = rng.integers(0, 1 << 31, size=32, dtype=np.int64)
    shingle_sets = [shingles(text) for text in texts]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, shingle_set in enumerate(shingle_sets):
        hashes = np.array([zlib.crc32(s.encode()) for s in shingle_set], dtype=np.int64)
        signature = ((np.outer(hashes, a) + b) % prime).min(axis=0)
        for band in range(8):
            buckets[(band, *signature[band * 4:(band + 1) * 4].tolist())].append(i)
    for members in buckets.values():
        for k, i in enumerate(members):
            for j in members[k + 1:]:
                if find(i) == find(j):
                    continue
                x, y = shingle_sets[i], shingle_sets[j]
                if len(x & y) >= threshold * len(x | y):
                    parent[max(find(i), find(j))] = min(find(i), find(j))  # the group is named after its first text
    return [find(i) for i in range(len(texts))]


def cross_label_conflicts(texts: list[str], labels: list[str], threshold: float = 0.5, max_df: int = 100) -> set[int]:
    """Indices of texts that have a Jaccard similarity >= threshold (word 5-shingles) with a text of another label.
    Candidate pairs share a shingle that occurs in at most max_df texts."""
    sets = [shingles(text) for text in texts]
    df: dict[str, int] = defaultdict(int)
    for s in sets:
        for shingle in s:
            df[shingle] += 1
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for shingle in s:
            if df[shingle] <= max_df:
                index[shingle].append(i)
    candidates = set()
    for ids in index.values():
        if len({labels[i] for i in ids}) > 1:
            candidates |= {(x, y) for k, x in enumerate(ids) for y in ids[k + 1:] if labels[x] != labels[y]}
    conflicts = set()
    for x, y in candidates:
        if len(sets[x] & sets[y]) >= threshold * len(sets[x] | sets[y]):
            conflicts |= {x, y}
    return conflicts


@register_task
class SpamAssassinNazario(TaskAdapter):
    name = f"{LEAF}/spamassassin_nazario"
    leaf = LEAF
    source = ("SpamAssassin public corpus, ham and spam (2002-2005), and Jose Nazario's phishing corpus, 2004-2006 mailboxes; "
              "labels by corpus membership")
    url = "https://monkey.org/~jose/phishing/"
    license = ("SpamAssassin corpus: not stated (readme: copyright for the text remains with the original senders); "
               "Nazario phishing corpus: CC BY 4.0")
    tier = "B"
    criteria_source = ("Spam: TREC 2005 Spam Track definition, verbatim; phishing: APWG 'Phishing Defined' (Phishing Activity "
                       "Trends Reports), first two sentences verbatim; ham option and precedence written")
    sampling = "balanced"
    notes = ("Ham = SpamAssassin easy_ham, easy_ham_2 and hard_ham; spam = SpamAssassin spam and spam_2 (20050311); phishing = "
             "Nazario's phishing0-2 and 20051114 mailboxes. The corpora were sorted by hand by their collectors, so some messages "
             "are in the wrong corpus: in a check of the 55 phishing items, 5 look like ordinary spam or genuine marketing. Each "
             "message becomes subject, sender domain (from the From header; placeholders the corpus editors substituted, such "
             "as spamassassin.taint.org, are shown as unknown) and body: the longer of the text/plain parts and the HTML parts "
             "converted to text (link targets kept in parentheses); attachments are dropped. Spam-filter tags are removed from "
             "subjects. Dropped: messages that contain a spam filter's verdict or the footer of the list where SpamAssassin users "
             "posted spam sightings, forwarded messages (the forwarder's note may name the verdict), mail addressed to the "
             "phishing collector by name (genuine service mail), image- or attachment-only and undecodable messages, exact and "
             "near duplicates (word 5-shingle Jaccard >= 0.8, first copy kept) and messages whose text occurs in two classes "
             "(Jaccard >= 0.5). Bodies over 20,000 characters (about 0.4% of messages) would be cut to their first 6,000 "
             "characters and marked as truncated; none was sampled. Sampled evenly across the three classes.")

    def raw_messages(self):
        """(corpus part, file name, label, raw bytes) in a fixed order."""
        for part, label in SA_PARTS:
            path = fetch(SA_URL + f"{part}.tar.bz2", f"spamassassin_{part}.tar.bz2")
            with tarfile.open(path) as archive:
                members = sorted((m for m in archive.getmembers() if m.isfile() and not m.name.endswith("/cmds")), key=lambda m: m.name)
                for member in members:
                    yield part, member.name, label, archive.extractfile(member).read()
        for part in NAZARIO_PARTS:
            data = fetch(NAZARIO_URL + f"{part}.mbox", f"nazario_{part}.mbox").read_bytes()
            messages = [m for m in re.split(rb"(?m)^From [^\n]*\n", data) if m.strip()]
            for i, raw in enumerate(messages):
                yield part, f"{part}.mbox#{i}", "phishing", re.sub(rb"(?m)^>(>*From )", rb"\1", raw)

    def load(self):
        records = []
        for part, file, label, raw in self.raw_messages():
            parsed = parse_email(raw)
            text = parsed["subject"] + "\n" + parsed["body"]
            if len(parsed["body"]) < MIN_BODY or GARBLED.search(text) or mojibake(text) or MAILBOX_ARTIFACT.search(text):
                continue  # nothing readable to classify
            if REVEALING.search(text) or COLLECTOR.search(text) or FORWARDED.search(parsed["subject"]):
                continue  # the text carries the answer, or is not a received email of its corpus
            records.append({"corpus": part, "file": file, "label": label, **parsed})
        # Near duplicates (Jaccard >= 0.8) form one group; a group spanning two classes is dropped, otherwise its first copy is kept.
        groups = near_duplicate_groups([r["subject"] + " " + r["body"] for r in records])
        labels_of_group: dict[int, set[str]] = defaultdict(set)
        for record, group in zip(records, groups):
            labels_of_group[group].add(record["label"])
        firsts, seen = [], set()
        for record, group in zip(records, groups):
            if len(labels_of_group[group]) == 1 and group not in seen:
                seen.add(group)
                firsts.append(record)
        # A looser check across classes (Jaccard >= 0.5): the same text should never appear with two labels.
        conflicts = cross_label_conflicts([r["subject"] + " " + r["body"] for r in firsts], [r["label"] for r in firsts])
        kept = []
        for i, record in enumerate(firsts):
            if i in conflicts:
                continue
            record["truncated"] = len(record["body"]) > LONG_BODY
            if record["truncated"]:
                record["body"] = record["body"][:KEEP_CHARS].rstrip() + TRUNCATION_MARK
            kept.append(record)
        self.stats = {"parsed": len(records), "after_dedup": len(firsts), "cross_class_conflicts": len(conflicts), "kept": len(kept)}
        return kept

    def label_of(self, record):
        return record["label"]

    def to_item(self, record, rng):
        question = Question(
            type="choice",
            instructions="Classify this email as ham, spam or phishing. Phishing emails are unsolicited as well; choose "
            "phishing whenever the email is phishing.",
            criteria=shuffled_options(OPTIONS, rng),
        )
        state = {"subject": record["subject"], "sender_domain": record["sender_domain"] or "unknown", "body": record["body"]}
        return self.item(state, {"email_type": question}, {"email_type": Gold(label=record["label"])},
                         corpus=record["corpus"], original_file=record["file"], truncated=record["truncated"])
