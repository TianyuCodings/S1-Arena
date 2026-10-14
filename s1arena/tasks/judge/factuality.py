"""Leaf: factuality of atomic claims (事实性，原子声明).

A response has already been split into atomic facts (FActScore, split by annotators) or
segments (FELM); each fact or segment is one boolean question in the same request.
No evidence is given: the judge must rely on its own knowledge. This is what separates
this leaf from judge/rag_faithfulness, where the context is provided.
"""

from __future__ import annotations

import json
import zipfile

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch

LEAF = "judge/atomic_factuality"

# FActScore (Min et al., 2023, Section 3): an atomic fact is a short sentence conveying one piece of information;
# it is Supported when the knowledge source (the person's English Wikipedia article) supports it.
FACTSCORE_QUESTION = (
    "The sentence was split by annotators into atomic facts (short statements that each convey one piece of information). "
    "Is this atomic fact true, i.e. supported by the English Wikipedia article about {topic}? Atomic fact: \"{fact}\""
)


@register_task
class FActScore(TaskAdapter):
    name = f"{LEAF}/factscore"
    leaf = LEAF
    source = "FActScore human-annotated biographies by InstructGPT, ChatGPT and PerplexityAI (Min et al., 2023)"
    url = "https://github.com/shmsw25/FActScore"
    license = "MIT"
    tier = "B"
    criteria_source = "Adapted from the FActScore definitions of atomic facts and of Supported (Min et al., 2023, Section 3)"
    notes = ("One item per response sentence; each of its human-written atomic facts is a boolean question (Supported = true, "
             "Not-supported = false; facts marked Irrelevant are left out). The annotators judged against the person's English "
             "Wikipedia article, which is not shown, so the judge must rely on its own knowledge; entities range from very "
             "frequent to very rare. Data downloaded from the authors' Google Drive link.")
    MODELS = ("InstructGPT", "ChatGPT", "PerplexityAI")

    def load(self):
        path = fetch("https://drive.usercontent.google.com/download?id=155exEdKs7R21gZF4G-x54-XN3qswBcPo&export=download&confirm=t", "factscore_data.zip")
        out = []
        with zipfile.ZipFile(path) as zf:
            for model in self.MODELS:
                with zf.open(f"data/labeled/{model}.jsonl") as f:
                    for g, line in enumerate(f):
                        r = json.loads(line)
                        for s, sentence in enumerate(r.get("annotations") or []):
                            facts = [(x["text"].strip(), x["label"]) for x in sentence.get("human-atomic-facts") or [] if x.get("label") in ("S", "NS")]
                            if facts:
                                out.append({"model": model, "generation": g, "index": s, "topic": r["topic"], "prompt": r["input"],
                                            "response": r["output"], "sentence": sentence["text"], "facts": facts})
        return out

    def label_of(self, record):
        return f"{record['model']}|{any(label == 'NS' for _, label in record['facts'])}"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for i, (fact, label) in enumerate(record["facts"], 1):
            questions[f"fact_{i}"] = Question(type="boolean", instructions=FACTSCORE_QUESTION.format(topic=record["topic"], fact=fact))
            gold[f"fact_{i}"] = Gold(label="true" if label == "S" else "false")
        state = {"prompt": record["prompt"], "response": record["response"], "sentence": record["sentence"]}
        return self.item(state, questions, gold, topic=record["topic"], response_model=record["model"])


# FELM annotation guidelines (Chen et al., 2023, Section 3.2): whether each segment contains a factual error, with
# four error types; adapted into a question.
FELM_QUESTION = (
    "Is this segment of the response free of factual errors? A segment has a factual error if it states hallucinated or "
    "inaccurate information (knowledge error), uses flawed reasoning or faulty logic, including wrong calculations "
    "(reasoning error), is unrelated to the prompt (irrelevant), or fails to recognize a falsehood or joke in the prompt "
    "(fooled). Segment {i}: \"{segment}\""
)


@register_task
class FELM(TaskAdapter):
    name = f"{LEAF}/felm"
    leaf = LEAF
    source = "FELM, segment-level factuality of ChatGPT responses (Chen et al., 2023)"
    url = "https://huggingface.co/datasets/hkust-nlp/felm"
    license = "CC BY-NC-SA 4.0"
    tier = "B"
    criteria_source = "Adapted from the FELM annotation guidelines (factual error per segment and its four error types, Section 3.2)"
    notes = ("World knowledge, science/technology, writing/recommendation, reasoning and math prompts; every segment was "
             "labelled by two expert annotators and reviewed. Reference links are not shown.")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        path = hf_hub_download("hkust-nlp/felm", "all.jsonl", repo_type="dataset")
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                segments, labels = r["segmented_response"], r["labels"]
                if not segments or len(segments) != len(labels) or not all(isinstance(x, bool) for x in labels):
                    continue
                response = r["response"].replace("\\n", "\n") if isinstance(r["response"], str) else " ".join(segments)
                out.append({"index": r["index"], "domain": r["domain"], "prompt": r["prompt"], "response": response, "segments": segments, "labels": labels})
        return out

    def label_of(self, record):
        return f"{record['domain']}|{all(record['labels'])}"

    def to_item(self, record, rng):
        questions, gold = {}, {}
        for i, (segment, correct) in enumerate(zip(record["segments"], record["labels"]), 1):
            questions[f"segment_{i}"] = Question(type="boolean", instructions=FELM_QUESTION.format(i=i, segment=segment.strip()))
            gold[f"segment_{i}"] = Gold(label="true" if correct else "false")
        return self.item({"prompt": record["prompt"], "response": record["response"]}, questions, gold, domain=record["domain"], original_id=record["index"])
