"""Leaf: medical decisions (医疗).

USMLE-style clinical questions (MedQA), yes/no/maybe questions answered from a study
abstract (PubMedQA) and emergency-department triage levels assigned by experts (KTAS).
ICD coding is not included: the standard sources (MIMIC) need credentialed access.
"""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter, shuffled_options
from ..common import fetch, hf_rows

LEAF = "pro/medical"


# ---------- MedQA: USMLE questions ----------

@register_task
class MedQA(TaskAdapter):
    name = f"{LEAF}/medqa_usmle"
    leaf = LEAF
    source = "MedQA, US (USMLE) questions with 4 options, test split (Jin et al., 2021)"
    url = "https://github.com/jind11/MedQA"
    license = "MIT"
    tier = "C"
    criteria_source = "Written; options are the original answer texts in their original order"
    notes = "Read from the 4-option copy GBaker/MedQA-USMLE-4-options. Widely used for LLM evaluation, so contamination is likely."

    def load(self):
        return hf_rows("GBaker/MedQA-USMLE-4-options", None, "test")

    def label_of(self, record):
        return f"{record['meta_info']}|{record['answer_idx']}"

    def to_item(self, record, rng):
        options = {k: str(v) for k, v in sorted(record["options"].items())}
        if record["answer_idx"] not in options or options[record["answer_idx"]] != record["answer"]:
            return None
        question = Question(type="choice", instructions="This is a question from the United States Medical Licensing Examination (USMLE). Which option is the correct answer?", criteria=options)
        return self.item({"question": record["question"].strip()}, {"answer": question}, {"answer": Gold(label=record["answer_idx"])}, step=record["meta_info"])


# ---------- PubMedQA: yes / no / maybe from the abstract ----------

@register_task
class PubMedQA(TaskAdapter):
    name = f"{LEAF}/pubmedqa"
    leaf = LEAF
    source = "PubMedQA, expert-labelled PQA-L, official 500-question test set, reasoning-required setting (Jin et al., 2019)"
    url = "https://github.com/pubmedqa/pubmedqa"
    license = "MIT"
    tier = "B"
    criteria_source = "Option definitions adapted from the PubMedQA annotation criteria (Jin et al., 2019, Appendix C)"
    notes = "The abstract's conclusion (the long answer) is withheld, as in the paper's reasoning-required setting; the test PMIDs come from data/test_ground_truth.json of the official repository."
    OPTIONS = {
        "yes": "yes: the experiments and results of this study indicate the answer is yes (for example, a significant difference in an outcome between the experimental and control group).",
        "no": "no: the experiments and results of this study indicate the answer is no (for example, no significant difference).",
        "maybe": "maybe: the study describes conditions where the answer is yes and conditions where it is no, or the question asks about several interventions or observations and the answer is yes for some but no for others.",
    }

    def load(self):
        path = fetch("https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json", "pubmedqa_test_ground_truth.json")
        test = json.loads(path.read_text(encoding="utf-8"))
        rows = [r for r in hf_rows("qiaojin/PubMedQA", "pqa_labeled", "train") if str(r["pubid"]) in test]
        return [r for r in rows if r["final_decision"] == test[str(r["pubid"])]]

    def label_of(self, record):
        return record["final_decision"]

    def to_item(self, record, rng):
        sections = zip(record["context"]["labels"], record["context"]["contexts"])
        context = "\n".join(f"{label.upper()}: {text.strip()}" for label, text in sections)
        question = Question(
            type="choice",
            instructions="Answer the research question with yes, no or maybe, using the abstract of the study (its conclusion is not shown). The answer is context-dependent: it is what the experiments and results of this study indicate.",
            criteria=shuffled_options(self.OPTIONS, rng),
        )
        state = {"question": record["question"].strip(), "abstract": context}
        return self.item(state, {"answer": question}, {"answer": Gold(label=record["final_decision"])}, pubid=str(record["pubid"]))


# ---------- KTAS: emergency-department triage level ----------

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path) -> list[dict[str, str]]:
    """Rows of the first worksheet as dicts of strings (header row = keys); stdlib only."""
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    table = []
    for row in sheet.find("m:sheetData", NS).findall("m:row", NS):
        cells = {}
        for c in row.findall("m:c", NS):
            column = re.match(r"[A-Z]+", c.get("r")).group(0)
            kind, v = c.get("t"), c.find("m:v", NS)
            if kind == "s":
                value = shared[int(v.text)]
            elif kind == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
            elif v is None or kind == "e":  # empty or an error value such as #NULL!
                value = ""
            else:
                value = v.text or ""
                try:  # shortest form of numbers: '71.0' -> '71', '36.600000000000001' -> '36.6'
                    number = float(value)
                    value = str(int(number)) if number.is_integer() else repr(number)
                except ValueError:
                    pass
            cells[column] = value.strip()
        table.append(cells)
    header = table[0]
    return [{header[col]: cells.get(col, "") for col in header} for cells in table[1:]]


@register_task
class KTASTriage(TaskAdapter):
    name = f"{LEAF}/ktas_triage"
    leaf = LEAF
    source = "Emergency department visits with expert-reviewed KTAS triage levels (Moon et al., 2019, PLOS ONE, S1 data)"
    url = "https://doi.org/10.1371/journal.pone.0216972"
    license = "CC BY 4.0"
    tier = "B"
    criteria_source = "Level descriptions verbatim from the Canadian Triage and Acuity Scale (CTAS participant manual v2.5b), on which KTAS is based; KTAS level names from the KTAS literature"
    notes = ("1,267 visits to two Korean EDs (2016-2017); gold is the KTAS level assigned by three triage experts reviewing the record "
             "(KTAS_expert). The nurses' triage level, diagnosis, disposition, length of stay and mistriage fields are withheld. "
             "A few chief complaints are in Korean, as recorded.")
    LEVELS = [
        "Level 1 (resuscitation): conditions that are threats to life or limb (or imminent risk of deterioration) requiring aggressive interventions",
        "Level 2 (emergency): conditions that are a potential threat to life, limb or function, requiring rapid medical intervention",
        "Level 3 (urgent): conditions that could potentially progress to a serious problem requiring emergency intervention; may be associated with significant discomfort or affect ability to function at work or in activities of daily living",
        "Level 4 (less urgent): conditions that relate to patient age, distress, or potential for deterioration that would benefit from intervention or reassurance within one or two hours",
        "Level 5 (non-urgent): conditions that may be acute but non-urgent as well as conditions which may be part of a chronic problem with or without evidence of deterioration; investigation or interventions could be delayed or even referred to other areas of the hospital or health care system",
    ]
    CODES = {  # from the S2 coding guideline
        "Sex": {"1": "female", "2": "male"},
        "Arrival mode": {"1": "walking", "2": "public ambulance (119)", "3": "private car", "4": "private ambulance", "5": "public transportation (police etc.)", "6": "wheelchair", "7": "other"},
        "Injury": {"1": "no", "2": "yes"},
        "Mental": {"1": "alert", "2": "responds to verbal stimuli", "3": "responds to pain", "4": "unconscious"},
        "Pain": {"1": "yes", "0": "no", "2": "no"},
    }
    VITALS = (("SBP", "systolic_blood_pressure", " mmHg"), ("DBP", "diastolic_blood_pressure", " mmHg"), ("HR", "heart_rate", " beats/min"),
              ("RR", "respiratory_rate", " breaths/min"), ("BT", "body_temperature", " °C"), ("Saturation", "oxygen_saturation", "%"))

    def load(self):
        path = fetch("https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0216972.s001", "ktas_moon2019_s1.xlsx")
        rows = [dict(r, sheet_row=i + 2) for i, r in enumerate(read_xlsx(path))]  # spreadsheet row number, for provenance
        return [r for r in rows if r["KTAS_expert"] in ("1", "2", "3", "4", "5")]

    def label_of(self, record):
        return record["KTAS_expert"]

    @staticmethod
    def _measure(value: str, unit: str) -> str:
        if not value:
            return "not recorded"
        if value == "측불":  # Korean: could not be measured
            return "could not be measured"
        return value.rstrip(".") + unit  # a few values are typed as '36.'

    def to_item(self, record, rng):
        code = lambda field: self.CODES[field].get(record[field], "not recorded")
        state = {
            "sex": code("Sex"),
            "age": f"{record['Age']} years",
            "arrival_mode": code("Arrival mode"),
            "injury": code("Injury"),
            "chief_complaint": record["Chief_complain"],
            "mental_state": code("Mental"),
            "pain": code("Pain"),
            "pain_score_nrs_0_10": record["NRS_pain"] or ("not applicable (no pain)" if code("Pain") == "no" else "not recorded"),
        }
        for column, key, unit in self.VITALS:
            state[key] = self._measure(record[column], unit)
        question = Question(type="score", instructions="Assign the Korean Triage and Acuity Scale (KTAS) level for this emergency department patient at triage.", criteria=self.LEVELS)
        gold = Gold(label=str(int(record["KTAS_expert"]) - 1))
        return self.item(state, {"ktas_level": question}, {"ktas_level": gold}, sheet_row=record["sheet_row"])
