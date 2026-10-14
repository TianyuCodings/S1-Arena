"""Leaf: security operations (安全运营): vulnerability scoring and classification, alert triage.

NVD analysts' CVSS v3.1 base metrics and CWE assignments for CVEs of 2025, and the
attack category of network connection records (NSL-KDD). Log anomalies are left to
struct/log_anomaly (Loghub).
"""

from __future__ import annotations

import csv
import functools
import gzip
import json
import re
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch

LEAF = "pro/secops"

NVD_FEED = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-2025.json.gz"
NVD_CUTOFF = "2026-09-01"  # records published or modified later are ignored, so a later feed can only lose records
NVD_LICENSE = "CVE Terms of Use (descriptions, © MITRE); NVD metrics and CWE mappings are US government data"


@functools.lru_cache(maxsize=1)
def nvd_2025() -> tuple[dict, ...]:
    """Analysed CVE-2025 records: English description, NVD's own CVSS v3.1 metrics and CWE mappings."""
    data = json.load(gzip.open(fetch(NVD_FEED), "rt", encoding="utf-8"))
    out = []
    for vuln in data["vulnerabilities"]:
        cve = vuln["cve"]
        if cve.get("vulnStatus") not in ("Analyzed", "Modified") or not (cve["published"] < NVD_CUTOFF and cve["lastModified"] < NVD_CUTOFF):
            continue
        text = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "").strip()
        metrics = [m for m in cve.get("metrics", {}).get("cvssMetricV31", []) if m.get("type") == "Primary" and m.get("source") == "nvd@nist.gov"]
        cwes = sorted({d["value"] for w in cve.get("weaknesses", []) if w.get("type") == "Primary" and w.get("source") == "nvd@nist.gov"
                       for d in w["description"]})
        out.append({"id": cve["id"], "description": text, "cvss": metrics[0]["cvssData"] if metrics else None, "cwes": cwes})
    return tuple(out)


# ---------- NVD: CVSS v3.1 base metrics ----------

# FIRST CVSS v3.1 specification (Sections 2 and 5), verbatim. CVSS is owned by FIRST and used by permission.
CVSS_METRICS = {
    "attack_vector": ("attackVector", "Attack Vector (AV): This metric reflects the context by which vulnerability exploitation is "
        "possible. This metric value (and consequently the Base Score) will be larger the more remote (logically, and physically) "
        "an attacker can be in order to exploit the vulnerable component. Scoring Guidance: When deciding between Network and "
        "Adjacent, if an attack can be launched over a wide area network or from outside the logically adjacent administrative "
        "network domain, use Network. Network should be used even if the attacker is required to be on the same intranet to "
        "exploit the vulnerable system (e.g., the attacker can only exploit the vulnerability from inside a corporate network).", {
        "network": ("NETWORK", "Network (N): The vulnerable component is bound to the network stack and the set of possible attackers "
            "extends beyond the other options listed below, up to and including the entire Internet. Such a vulnerability is often "
            "termed “remotely exploitable” and can be thought of as an attack being exploitable at the protocol level one or more "
            "network hops away (e.g., across one or more routers). An example of a network attack is an attacker causing a denial "
            "of service (DoS) by sending a specially crafted TCP packet across a wide area network (e.g., CVE‑2004‑0230)."),
        "adjacent": ("ADJACENT_NETWORK", "Adjacent (A): The vulnerable component is bound to the network stack, but the attack is "
            "limited at the protocol level to a logically adjacent topology. This can mean an attack must be launched from the same "
            "shared physical (e.g., Bluetooth or IEEE 802.11) or logical (e.g., local IP subnet) network, or from within a secure or "
            "otherwise limited administrative domain (e.g., MPLS, secure VPN to an administrative network zone). One example of an "
            "Adjacent attack would be an ARP (IPv4) or neighbor discovery (IPv6) flood leading to a denial of service on the local "
            "LAN segment (e.g., CVE‑2013‑6014)."),
        "local": ("LOCAL", "Local (L): The vulnerable component is not bound to the network stack and the attacker’s path is via "
            "read/write/execute capabilities. Either: the attacker exploits the vulnerability by accessing the target system locally "
            "(e.g., keyboard, console), or remotely (e.g., SSH); or the attacker relies on User Interaction by another person to "
            "perform actions required to exploit the vulnerability (e.g., using social engineering techniques to trick a legitimate "
            "user into opening a malicious document)."),
        "physical": ("PHYSICAL", "Physical (P): The attack requires the attacker to physically touch or manipulate the vulnerable "
            "component. Physical interaction may be brief (e.g., evil maid attack) or persistent. An example of such an attack is a "
            "cold boot attack in which an attacker gains access to disk encryption keys after physically accessing the target "
            "system. Other examples include peripheral attacks via FireWire/USB Direct Memory Access (DMA)."),
    }),
    "attack_complexity": ("attackComplexity", "Attack Complexity (AC): This metric describes the conditions beyond the attacker’s "
        "control that must exist in order to exploit the vulnerability. As described below, such conditions may require the "
        "collection of more information about the target, or computational exceptions. Importantly, the assessment of this metric "
        "excludes any requirements for user interaction in order to exploit the vulnerability (such conditions are captured in the "
        "User Interaction metric). If a specific configuration is required for an attack to succeed, the Base metrics should be "
        "scored assuming the vulnerable component is in that configuration.", {
        "low": ("LOW", "Low (L): Specialized access conditions or extenuating circumstances do not exist. An attacker can expect "
            "repeatable success when attacking the vulnerable component."),
        "high": ("HIGH", "High (H): A successful attack depends on conditions beyond the attacker's control. That is, a successful "
            "attack cannot be accomplished at will, but requires the attacker to invest in some measurable amount of effort in "
            "preparation or execution against the vulnerable component before a successful attack can be expected. For example, a "
            "successful attack may depend on an attacker overcoming any of the following conditions: The attacker must gather "
            "knowledge about the environment in which the vulnerable target/component exists. For example, a requirement to collect "
            "details on target configuration settings, sequence numbers, or shared secrets. The attacker must prepare the target "
            "environment to improve exploit reliability. For example, repeated exploitation to win a race condition, or overcoming "
            "advanced exploit mitigation techniques. The attacker must inject themselves into the logical network path between the "
            "target and the resource requested by the victim in order to read and/or modify network communications (e.g., a man in "
            "the middle attack)."),
    }),
    "privileges_required": ("privilegesRequired", "Privileges Required (PR): This metric describes the level of privileges an "
        "attacker must possess before successfully exploiting the vulnerability. Scoring Guidance: Privileges Required is usually "
        "None for hard-coded credential vulnerabilities or vulnerabilities requiring social engineering (e.g., reflected cross-site "
        "scripting, cross-site request forgery, or file parsing vulnerability in a PDF reader).", {
        "none": ("NONE", "None (N): The attacker is unauthorized prior to attack, and therefore does not require any access to "
            "settings or files of the vulnerable system to carry out an attack."),
        "low": ("LOW", "Low (L): The attacker requires privileges that provide basic user capabilities that could normally affect "
            "only settings and files owned by a user. Alternatively, an attacker with Low privileges has the ability to access only "
            "non-sensitive resources."),
        "high": ("HIGH", "High (H): The attacker requires privileges that provide significant (e.g., administrative) control over "
            "the vulnerable component allowing access to component-wide settings and files."),
    }),
    "user_interaction": ("userInteraction", "User Interaction (UI): This metric captures the requirement for a human user, other "
        "than the attacker, to participate in the successful compromise of the vulnerable component. This metric determines whether "
        "the vulnerability can be exploited solely at the will of the attacker, or whether a separate user (or user-initiated "
        "process) must participate in some manner.", {
        "none": ("NONE", "None (N): The vulnerable system can be exploited without interaction from any user."),
        "required": ("REQUIRED", "Required (R): Successful exploitation of this vulnerability requires a user to take some action "
            "before the vulnerability can be exploited. For example, a successful exploit may only be possible during the "
            "installation of an application by a system administrator."),
    }),
    "scope": ("scope", "Scope (S): The Scope metric captures whether a vulnerability in one vulnerable component impacts resources "
        "in components beyond its security scope. Intuitively, whenever the impact of a vulnerability breaches a security/trust "
        "boundary and impacts components outside the security scope in which vulnerable component resides, a Scope change occurs.", {
        "unchanged": ("UNCHANGED", "Unchanged (U): An exploited vulnerability can only affect resources managed by the same security "
            "authority. In this case, the vulnerable component and the impacted component are either the same, or both are managed "
            "by the same security authority."),
        "changed": ("CHANGED", "Changed (C): An exploited vulnerability can affect resources beyond the security scope managed by the "
            "security authority of the vulnerable component. In this case, the vulnerable component and the impacted component are "
            "different and managed by different security authorities."),
    }),
    "confidentiality": ("confidentialityImpact", "Confidentiality (C): This metric measures the impact to the confidentiality of the "
        "information resources managed by a software component due to a successfully exploited vulnerability. Confidentiality "
        "refers to limiting information access and disclosure to only authorized users, as well as preventing access by, or "
        "disclosure to, unauthorized ones.", {
        "high": ("HIGH", "High (H): There is a total loss of confidentiality, resulting in all resources within the impacted "
            "component being divulged to the attacker. Alternatively, access to only some restricted information is obtained, but "
            "the disclosed information presents a direct, serious impact. For example, an attacker steals the administrator's "
            "password, or private encryption keys of a web server."),
        "low": ("LOW", "Low (L): There is some loss of confidentiality. Access to some restricted information is obtained, but the "
            "attacker does not have control over what information is obtained, or the amount or kind of loss is limited. The "
            "information disclosure does not cause a direct, serious loss to the impacted component."),
        "none": ("NONE", "None (N): There is no loss of confidentiality within the impacted component."),
    }),
    "integrity": ("integrityImpact", "Integrity (I): This metric measures the impact to integrity of a successfully exploited "
        "vulnerability. Integrity refers to the trustworthiness and veracity of information.", {
        "high": ("HIGH", "High (H): There is a total loss of integrity, or a complete loss of protection. For example, the attacker "
            "is able to modify any/all files protected by the impacted component. Alternatively, only some files can be modified, "
            "but malicious modification would present a direct, serious consequence to the impacted component."),
        "low": ("LOW", "Low (L): Modification of data is possible, but the attacker does not have control over the consequence of a "
            "modification, or the amount of modification is limited. The data modification does not have a direct, serious impact "
            "on the impacted component."),
        "none": ("NONE", "None (N): There is no loss of integrity within the impacted component."),
    }),
    "availability": ("availabilityImpact", "Availability (A): This metric measures the impact to the availability of the impacted "
        "component resulting from a successfully exploited vulnerability. While the Confidentiality and Integrity impact metrics "
        "apply to the loss of confidentiality or integrity of data (e.g., information, files) used by the impacted component, this "
        "metric refers to the loss of availability of the impacted component itself, such as a networked service (e.g., web, "
        "database, email).", {
        "high": ("HIGH", "High (H): There is a total loss of availability, resulting in the attacker being able to fully deny access "
            "to resources in the impacted component; this loss is either sustained (while the attacker continues to deliver the "
            "attack) or persistent (the condition persists even after the attack has completed). Alternatively, the attacker has the "
            "ability to deny some availability, but the loss of availability presents a direct, serious consequence to the impacted "
            "component (e.g., the attacker cannot disrupt existing connections, but can prevent new connections; the attacker can "
            "repeatedly exploit a vulnerability that, in each instance of a successful attack, leaks a only small amount of memory, "
            "but after repeated exploitation causes a service to become completely unavailable)."),
        "low": ("LOW", "Low (L): Performance is reduced or there are interruptions in resource availability. Even if repeated "
            "exploitation of the vulnerability is possible, the attacker does not have the ability to completely deny service to "
            "legitimate users. The resources in the impacted component are either partially available all of the time, or fully "
            "available only some of the time, but overall there is no direct, serious consequence to the impacted component."),
        "none": ("NONE", "None (N): There is no impact to availability within the impacted component."),
    }),
}
SEVERITY_LEVELS = ["None: CVSS score 0.0", "Low: CVSS score 0.1 - 3.9", "Medium: CVSS score 4.0 - 6.9", "High: CVSS score 7.0 - 8.9",
                   "Critical: CVSS score 9.0 - 10.0"]
SEVERITY = {"NONE": "0", "LOW": "1", "MEDIUM": "2", "HIGH": "3", "CRITICAL": "4"}
# Descriptions that state a score, a vector or a rating (e.g. VulDB's "classified as critical") would leak the answer.
CVSS_LEAK = re.compile(r"cvss|base score|\bAV:[NALP]\b|severity|classified as|rated as|declared as|"
                       r"\b(critical|high|medium|moderate|low)[- ](risk|severity|impact)\b", re.I)


@register_task
class NVDCVSS(TaskAdapter):
    name = f"{LEAF}/nvd_cvss31"
    leaf = LEAF
    source = "NVD CVSS v3.1 base metrics assigned by NVD analysts, CVE-2025 records (NIST National Vulnerability Database)"
    url = "https://nvd.nist.gov/vuln/data-feeds"
    license = NVD_LICENSE
    tier = "A"
    criteria_source = ("Metric descriptions, value definitions, scoring guidance and the qualitative severity scale verbatim from "
                       "the FIRST CVSS v3.1 specification (CVSS is owned by FIRST and used by permission)")
    notes = ("One request per CVE: the eight base metrics as choice questions and the severity rating as a score. Only NVD's own "
             "(Primary) v3.1 vector is used, for analysed CVE-2025 records; descriptions that mention CVSS, a vector, a severity or a "
             "CNA rating ('classified as critical') are dropped. The NVD feed is regenerated daily; only records published and last "
             "modified before 2026-09-01 are used, so a rebuild from a later feed can only lose records NVD touched since, never "
             "gain records or see changed labels.")

    def load(self):
        rows = [r for r in nvd_2025() if r["cvss"] and r["description"] and not CVSS_LEAK.search(r["description"])]
        return [{"id": r["id"], "description": r["description"], "cvss": r["cvss"]} for r in rows]

    def label_of(self, record):
        return record["cvss"]["baseSeverity"]

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for key, (field, text, values) in CVSS_METRICS.items():
            options = {value_id: desc for value_id, (_, desc) in values.items()}
            questions[key] = Question(type="choice", instructions=text + " Which value applies to the vulnerability described?",
                                      criteria=shuffled_options(options, rng))
            gold[key] = Gold(label=next(v for v, (nvd, _) in values.items() if nvd == record["cvss"][field]))
        questions["severity"] = Question(type="score", instructions="Which CVSS v3.1 qualitative severity rating does the Base Score "
                                         "of this vulnerability have?", criteria=SEVERITY_LEVELS)
        gold["severity"] = Gold(label=SEVERITY[record["cvss"]["baseSeverity"]])
        return self.item({"vulnerability_description": record["description"]}, questions, gold, original_id=record["id"],
                         vector=record["cvss"]["vectorString"])


# ---------- NVD: CWE weakness type ----------

# The 32 CWEs that NVD assigned (as the single Primary CWE) to at least 15 analysed CVE-2025 records.
CWE_IDS = [89, 787, 79, 476, 125, 120, 416, 434, 401, 667, 77, 78, 362, 352, 908, 862, 22, 94, 502, 863, 190, 415, 639, 770,
           306, 369, 617, 129, 835, 918, 522, 798]
CWE_XML = "https://cwe.mitre.org/data/xml/cwec_v4.20.xml.zip"


@register_task
class NVDCWE(TaskAdapter):
    name = f"{LEAF}/nvd_cwe"
    leaf = LEAF
    source = "NVD CWE mappings assigned by NVD analysts, CVE-2025 records (NIST National Vulnerability Database)"
    url = "https://nvd.nist.gov/vuln/categories"
    license = NVD_LICENSE + "; CWE names and descriptions © MITRE (CWE Terms of Use)"
    tier = "B"
    criteria_source = "CWE names and descriptions verbatim from CWE v4.20 (MITRE); question written"
    sampling = "balanced"
    notes = ("Options are 32 frequent CWEs, including overlapping parent/child entries (e.g. CWE-77/78, CWE-119 family, CWE-862/863/639) "
             "as in real mappings. Only CVEs whose single NVD Primary CWE is one of them; descriptions naming a CWE are dropped. "
             "Sampled evenly across CWEs because the stream is dominated by template-generated SQL injection reports (24%). Many "
             "VulDB-style descriptions name the weakness in words ('leads to cross site scripting', 'hard-coded credentials'), so "
             "part of the pool is easy. Same 2026-09-01 publication/modification cutoff as nvd_cvss31.")

    def load(self):
        with zipfile.ZipFile(fetch(CWE_XML)) as zf:
            text = zf.read(zf.namelist()[0]).decode("utf-8")
        self.options = {}
        for cwe_id in CWE_IDS:
            match = re.search(rf'<Weakness ID="{cwe_id}" Name="([^"]+)".*?<Description>(.*?)</Description>', text, re.S)
            name, desc = (html_unescape(m) for m in match.groups())
            self.options[f"CWE-{cwe_id}"] = f"CWE-{cwe_id}: {name}. {' '.join(desc.split())}"
        rows = [r for r in nvd_2025() if len(r["cwes"]) == 1 and r["cwes"][0] in self.options and r["description"]
                and "CWE-" not in r["description"].upper()]
        return [{"id": r["id"], "description": r["description"], "cwe": r["cwes"][0]} for r in rows]

    def label_of(self, record):
        return record["cwe"]

    def to_item(self, record, rng):
        question = Question(type="choice", instructions="Which CWE weakness is the root cause of the vulnerability described? Choose "
                            "the most specific weakness that applies.", criteria=shuffled_options(self.options, rng))
        return self.item({"vulnerability_description": record["description"]}, {"weakness": question}, {"weakness": Gold(label=record["cwe"])},
                         original_id=record["id"])


def html_unescape(text: str) -> str:
    import html

    return html.unescape(text)


# ---------- NSL-KDD: connection records ----------

KDD_TEST = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
# KDD Cup 1999 feature names; descriptions verbatim from the KDD-CUP-99 task description (tables 1-3), host-based ones written.
KDD_FEATURES = [
    ("duration", "length (number of seconds) of the connection"),
    ("protocol_type", "type of the protocol, e.g. tcp, udp, etc."),
    ("service", "network service on the destination, e.g., http, telnet, etc."),
    ("flag", "normal or error status of the connection"),
    ("src_bytes", "number of data bytes from source to destination"),
    ("dst_bytes", "number of data bytes from destination to source"),
    ("land", "1 if connection is from/to the same host/port; 0 otherwise"),
    ("wrong_fragment", "number of ``wrong'' fragments"),
    ("urgent", "number of urgent packets"),
    ("hot", "number of ``hot'' indicators"),
    ("num_failed_logins", "number of failed login attempts"),
    ("logged_in", "1 if successfully logged in; 0 otherwise"),
    ("num_compromised", "number of ``compromised'' conditions"),
    ("root_shell", "1 if root shell is obtained; 0 otherwise"),
    ("su_attempted", "1 if ``su root'' command attempted; 0 otherwise"),
    ("num_root", "number of ``root'' accesses"),
    ("num_file_creations", "number of file creation operations"),
    ("num_shells", "number of shell prompts"),
    ("num_access_files", "number of operations on access control files"),
    ("num_outbound_cmds", "number of outbound commands in an ftp session"),
    ("is_host_login", "1 if the login belongs to the ``hot'' list; 0 otherwise"),
    ("is_guest_login", "1 if the login is a ``guest''login; 0 otherwise"),
    ("count", "number of connections to the same host as the current connection in the past two seconds"),
    ("srv_count", "number of connections to the same service as the current connection in the past two seconds"),
    ("serror_rate", "% of connections that have ``SYN'' errors (same-host connections)"),
    ("srv_serror_rate", "% of connections that have ``SYN'' errors (same-service connections)"),
    ("rerror_rate", "% of connections that have ``REJ'' errors (same-host connections)"),
    ("srv_rerror_rate", "% of connections that have ``REJ'' errors (same-service connections)"),
    ("same_srv_rate", "% of connections to the same service (same-host connections)"),
    ("diff_srv_rate", "% of connections to different services (same-host connections)"),
    ("srv_diff_host_rate", "% of connections to different hosts (same-service connections)"),
    ("dst_host_count", "number of recent connections having the same destination host (host-based window, capped at 255)"),
    ("dst_host_srv_count", "number of recent connections having the same destination service (host-based window, capped at 255)"),
    ("dst_host_same_srv_rate", "% of the connections counted in dst_host_count that were to the same service"),
    ("dst_host_diff_srv_rate", "% of the connections counted in dst_host_count that were to different services"),
    ("dst_host_same_src_port_rate", "% of the connections counted in dst_host_srv_count that came from the same source port"),
    ("dst_host_srv_diff_host_rate", "% of the connections counted in dst_host_srv_count that were to different destination hosts"),
    ("dst_host_serror_rate", "% of the connections counted in dst_host_count that have ``SYN'' errors"),
    ("dst_host_srv_serror_rate", "% of the connections counted in dst_host_srv_count that have ``SYN'' errors"),
    ("dst_host_rerror_rate", "% of the connections counted in dst_host_count that have ``REJ'' errors"),
    ("dst_host_srv_rerror_rate", "% of the connections counted in dst_host_srv_count that have ``REJ'' errors"),
]
# KDD Cup 1999 training_attack_types for the 22 training attacks; the test-only attacks follow the DARPA 1999 categories.
KDD_CATEGORY = {
    "normal": "normal",
    **dict.fromkeys(["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb", "processtable", "udpstorm"], "dos"),
    **dict.fromkeys(["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"], "probe"),
    **dict.fromkeys(["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "named", "sendmail",
                     "snmpguess", "xlock", "xsnoop"], "r2l"),
    **dict.fromkeys(["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"], "u2r"),
}
KDD_OPTIONS = {  # KDD-CUP-99 task description, verbatim ("normal" written)
    "normal": "normal: a normal connection, not an attack",
    "dos": "DOS: denial-of-service, e.g. syn flood",
    "r2l": "R2L: unauthorized access from a remote machine, e.g. guessing password",
    "u2r": "U2R: unauthorized access to local superuser (root) privileges, e.g., various ``buffer overflow'' attacks",
    "probe": "probing: surveillance and other probing, e.g., port scanning",
}


@register_task
class NSLKDD(TaskAdapter):
    name = f"{LEAF}/nsl_kdd"
    leaf = LEAF
    source = "NSL-KDD, KDDTest+ (Tavallaee et al., 2009; from the 1998 DARPA / KDD Cup 1999 data)"
    url = "https://www.unb.ca/cic/datasets/nsl.html"
    license = "Redistribution permitted with citation (UNB Canadian Institute for Cybersecurity)"
    tier = "B"
    criteria_source = ("Task wording, attack-category definitions and feature descriptions from the KDD-CUP-99 task description; "
                       "host-based feature descriptions and the 'normal' option written")
    notes = ("Each connection record is shown as its 41 features with descriptions; gold is the attack category of the record's "
             "attack label. The official download is gone; KDDTest+.txt comes from the defcom17/NSL_KDD GitHub mirror (UNB allows "
             "mirroring; checksum identical to a second mirror). snmpgetattack (records identical to normal traffic) and attacks "
             "whose category is disputed (httptunnel, worm) are excluded. Stratified by attack type, so every type appears. "
             "Simulated 1998 US Air Force LAN traffic.")

    def load(self):
        with fetch(KDD_TEST, "nsl_kdd_KDDTest+.txt").open(encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if len(r) == 43]
        return [{"features": r[:41], "attack": r[41], "difficulty": r[42]} for r in rows if r[41] in KDD_CATEGORY]

    def label_of(self, record):
        return record["attack"]

    def to_item(self, record, rng):
        lines = [f"{name} ({desc}): {value}" for (name, desc), value in zip(KDD_FEATURES, record["features"])]
        question = Question(
            type="choice",
            instructions="An intrusion detector must distinguish between ``bad'' connections, called intrusions or attacks, and "
            "``good'' normal connections. Is this network connection normal, or which main category of attack is it? "
            "Rates (%) are given as fractions between 0 and 1.",
            criteria=shuffled_options(KDD_OPTIONS, rng),
        )
        gold = Gold(label=KDD_CATEGORY[record["attack"]])
        return self.item({"connection_record": "\n".join(lines)}, {"category": question}, {"category": gold}, attack=record["attack"],
                         difficulty=record["difficulty"])
