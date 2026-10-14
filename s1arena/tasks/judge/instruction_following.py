"""Leaf: instruction-following verification (指令遵循核验).

Each item is a prompt with verifiable constraints and one model response; there is one
boolean question per constraint. Gold comes from the benchmarks' own checker programs
(strict mode), as recorded by the dataset releases: IFEval results in PPE, IFBench
results in shisa-ai/eval-IFBench-results. Constraint wording is rendered from the
benchmarks' own description templates.
"""

from __future__ import annotations

import json

from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import hf_rows, slug

LEAF = "judge/instruction_following"
CHECK = "Does the response follow this instruction from the prompt? Instruction: \"{desc}\" Judge strictly and literally, the way an automatic checker would."


def number(value):
    """Parquet copies store integer arguments as floats (5.0); render them as integers."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def clean_kwargs(kwargs: dict | None) -> dict:
    return {k: number(v) for k, v in (kwargs or {}).items() if v is not None}


# ---------- IFEval (Zhou et al., 2023) ----------

# Description templates from IFEval instructions.py (build_description), verbatim.
IFEVAL_TEMPLATES = {
    "keywords:existence": "Include keywords {keywords} in the response.",
    "keywords:frequency": "In your response, the word {keyword} should appear {relation} {frequency} times.",
    "keywords:forbidden_words": "Do not include keywords {forbidden_words} in the response.",
    "keywords:letter_frequency": "In your response, the letter {letter} should appear {let_relation} {let_frequency} times.",
    "language:response_language": "Your ENTIRE response should be in {language} language, no other language is allowed.",
    "length_constraints:number_sentences": "Your response should contain {relation} {num_sentences} sentences.",
    "length_constraints:number_paragraphs": "There should be {num_paragraphs} paragraphs. Paragraphs are separated with the markdown divider: ***",
    "length_constraints:number_words": "Answer with {relation} {num_words} words.",
    "length_constraints:nth_paragraph_first_word": (
        "There should be {num_paragraphs} paragraphs. Paragraphs and only paragraphs are separated with each other by two "
        "new lines as if it was '\\n\\n' in python. Paragraph {nth_paragraph} must start with word {first_word}."),
    "detectable_content:number_placeholders": "The response must contain at least {num_placeholders} placeholders represented by square brackets, such as [address].",
    "detectable_content:postscript": "At the end of your response, please explicitly add a postscript starting with {postscript_marker}",
    "detectable_format:number_bullet_lists": "Your answer must contain exactly {num_bullets} bullet points. Use the markdown bullet points such as:\n* This is point 1. \n* This is point 2",
    "detectable_format:constrained_response": "Answer with one of the following options: My answer is yes. / My answer is no. / My answer is maybe.",
    "detectable_format:number_highlighted_sections": "Highlight at least {num_highlights} sections in your answer with markdown, i.e. *highlighted section*.",
    "detectable_format:multiple_sections": (
        "Your response must have {num_sections} sections. Mark the beginning of each section with {section_spliter} X, such as:\n"
        "{section_spliter} 1\n[content of section 1]\n{section_spliter} 2\n[content of section 2]"),
    "detectable_format:json_format": "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```.",
    "detectable_format:title": "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.",
    "combination:two_responses": "Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: ******.",
    "combination:repeat_prompt": (
        "First repeat the request word for word without change, then give your answer (1. do not say any words or characters "
        "before repeating the request; 2. the request you need to repeat does not include this sentence). The request to repeat: {prompt_to_repeat}"),
    "startend:end_checker": "Finish your response with this exact phrase {end_phrase}. No other words should follow this phrase.",
    "change_case:capital_word_frequency": "In your response, words with all capital letters should appear {capital_relation} {capital_frequency} times.",
    "change_case:english_capital": "Your entire response should be in English, and in all capital letters.",
    "change_case:english_lowercase": "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed.",
    "punctuation:no_comma": "In your entire response, refrain from the use of any commas.",
    "startend:quotation": "Wrap your entire response with double quotation marks.",
}
# instructions_util.LANGUAGE_CODES of IFEval.
IFEVAL_LANGUAGES = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi", "fr": "French", "ru": "Russian",
    "de": "German", "ja": "Japanese", "it": "Italian", "bn": "Bengali", "uk": "Ukrainian", "th": "Thai", "ur": "Urdu",
    "ta": "Tamil", "te": "Telugu", "bg": "Bulgarian", "ko": "Korean", "pl": "Polish", "he": "Hebrew", "fa": "Persian",
    "vi": "Vietnamese", "ne": "Nepali", "sw": "Swahili", "kn": "Kannada", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ml": "Malayalam", "fi": "Finnish",
}


def ifeval_description(instruction_id: str, kwargs: dict) -> str:
    kwargs = clean_kwargs(kwargs)
    for key in ("keywords", "forbidden_words"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = ", ".join(str(k) for k in kwargs[key])
    if "language" in kwargs:
        kwargs["language"] = IFEVAL_LANGUAGES.get(kwargs["language"], kwargs["language"])
    return IFEVAL_TEMPLATES[instruction_id].format(**kwargs)


class ConstraintTask(TaskAdapter):
    leaf = LEAF
    tier = "B"

    def constraint_item(self, prompt: str, response: str, constraints: list[tuple[str, str, bool]], **meta):
        """constraints: (instruction id, description, followed) in prompt order."""
        questions, gold = {}, {}
        for i, (instruction_id, desc, followed) in enumerate(constraints, 1):
            key = f"c{i}_{slug(instruction_id.split(':')[-1])}"
            questions[key] = Question(type="boolean", instructions=CHECK.format(desc=desc))
            gold[key] = Gold(label="true" if followed else "false")
        return self.item({"prompt": prompt, "response": response}, questions, gold, **meta)


@register_task
class PPEIFEval(ConstraintTask):
    name = f"{LEAF}/ifeval_ppe"
    source = "IFEval prompts (Zhou et al., 2023) with sampled responses and strict checker results from PPE (Frick et al., 2025)"
    url = "https://huggingface.co/datasets/lmarena-ai/PPE-IFEval-Best-of-K"
    license = "Apache-2.0 (prompts); model outputs under the providers' terms of use (PPE dataset card)"
    criteria_source = "Constraint descriptions rendered from the IFEval description templates (instructions.py) with each prompt's arguments"
    notes = ("128 IFEval prompts x 4 models (Gemma-2-9B-it, GPT-4o-mini, Llama-3-8B-Instruct, Claude-3-Haiku) x 32 samples; "
             "gold is IFEval's strict instruction-level result as released by PPE. The prompt shown is the original IFEval "
             "prompt (PPE prefixed it with a sentence asking to follow all instructions exactly).")

    @staticmethod
    def checkable(instruction_id: str, kwargs: dict) -> bool:
        """IFEval replaces a letter argument that is not a-z (e.g. '#') by a random letter at check time, so its result is arbitrary."""
        if instruction_id == "keywords:letter_frequency":
            letter = str(kwargs.get("letter") or "")
            return len(letter) == 1 and letter.lower() in "abcdefghijklmnopqrstuvwxyz"
        return instruction_id in IFEVAL_TEMPLATES

    def load(self):
        out, seen = [], set()
        for r in hf_rows("lmarena-ai/PPE-IFEval-Best-of-K", None, "train"):
            ids, kwargs = r["instruction_id_list"], r["kwargs"]
            for k, score in enumerate(r["score_data"], 1):
                response = r.get(f"response_{k}")
                results = score["inst_level_strict_acc"]
                if not response or not response.strip() or len(results) != len(ids) or (r["key"], response) in seen:
                    continue
                seen.add((r["key"], response))
                keep = [(i, kw, ok) for i, kw, ok in zip(ids, kwargs, results) if self.checkable(i, kw)]
                if keep:
                    out.append({"key": r["key"], "prompt": r["ifeval_prompt"], "model": r["model_name"], "sample": k,
                                "constraints": keep, "all": all(ok for _, _, ok in keep), "response": response})
        return out

    def label_of(self, record):
        return f"{record['all']}|{record['model']}"

    def to_item(self, record, rng):
        constraints = [(i, ifeval_description(i, kw), ok) for i, kw, ok in record["constraints"]]
        return self.constraint_item(record["prompt"], record["response"], constraints, original_id=record["key"], response_model=record["model"])


# ---------- IFBench (Pyatkin et al., 2025) ----------

PERSON_NAMES = (
    "Emma, Liam, Sophia, Jackson, Olivia, Noah, Ava, Lucas, Isabella, Mason, Mia, Ethan, Charlotte, Alexander, Amelia, "
    "Benjamin, Harper, Leo, Zoe, Daniel, Chloe, Samuel, Lily, Matthew, Grace, Owen, Abigail, Gabriel, Ella, Jacob, "
    "Scarlett, Nathan, Victoria, Elijah, Layla, Nicholas, Audrey, David, Hannah, Christopher, Penelope, Thomas, Nora, "
    "Andrew, Aria, Joseph, Claire, Ryan, Stella, Jonathan"
)
# Description templates (_description_pattern) from IFBench ifbench/instructions.py, verbatim.
IFBENCH_TEMPLATES = {
    "count:conjunctions": "Use at least {small_n} different coordinating conjunctions in the response.",
    "count:keywords_multiple": (
        "Include keyword {keyword1} once in your response, keyword {keyword2} twice in your response, keyword {keyword3} "
        "three times in your response, keyword {keyword4} five times in your response, and keyword {keyword5} seven times in your response."),
    "count:numbers": "Include exactly {N} numbers in the response.",
    "count:person_names": "Mention at least {N} different person names in the response, from this list of person names: " + PERSON_NAMES + " .",
    "count:pronouns": "The response should include at least {N} pronouns.",
    "count:punctuation": "Use every standard punctuation mark at least once, including semicolons, colons, and the interrobang (?!).",
    "count:unique_word_count": "Use at least {N} unique words in the response.",
    "count:word_count_range": "The response must contain between {min_words} and {max_words} words.",
    "count:words_japanese": "Every {N}th word of your response must be in Japanese.",
    "custom:character_reverse": "What animal is the national symbol of the US? Respond to this query, but make your sentence in reverse order of what it should be, per letter.",
    "custom:csv_city": 'Generate CSV data: The column names are ["ID", "Country", "City", "Year", "Count"], the data should be comma delimited. Please generate 7 rows.',
    "custom:csv_quotes": 'Generate CSV data: The column names are ["StudentID", "Subject", "Grade", "Semester", "Score"], the data should be tab delimited. Please generate 3 rows and enclose each single field in double quotes.',
    "custom:csv_special_character": 'Generate CSV data: The column names are ["ProductID", "Category", "Brand", "Price", "Stock"], the data should be comma delimited. Please generate 14 rows. Add one field which contains a special character and enclose it in double quotes.',
    "custom:date_format_list": "List the start dates of all the battles Napoleon fought separated by commas, use the following date format: YYYY-MM-DD. Do not provide an explanation.",
    "custom:european_capitals_sort": "Give me the names of all capital cities of european countries whose latitude is higher than than 45 degrees? List the capital cities without country names, separated by commas, sorted by latitude, from highest to lowest.",
    "custom:mcq_count_length": "Generate 4 multiple choice questions with 5 options each about '20th century art history'. Each question should start with the label \"Question\". The questions should get progressively longer. Do not provide an explanation.",
    "custom:multiples": "Count from 10 to 50 but only print multiples of 7.",
    "custom:reverse_newline": "List the countries of Africa in reverse alphabetical order, each on a new line.",
    "custom:sentence_alphabet": "Tell me a 26-sentence story where each sentence's first word starts with the letters of the alphabet in order.",
    "custom:word_reverse": "What animal is the national symbol of the US? Respond to this query, but make your sentence in reverse order of what it should be, per word.",
    "format:emoji": "Please use an emoji at the end of every sentence.",
    "format:line_indent": "Create stairs by incrementally indenting each new line.",
    "format:list": "Answer with a list of items, instead of bullet points use {sep}.",
    "format:newline": "Write each word on a new line.",
    "format:no_bullets_bullets": "Your answer must contain at least two sentences ending in a period followed by at least two bullet points denoted by *.",
    "format:no_whitespace": "The output should not contain any whitespace.",
    "format:options": "Answer with one of the following options: {options}. Do not give any explanation.",
    "format:output_template": "Use this exact template for your response: My Answer: [answer] My Conclusion: [conclusion] Future Outlook: [outlook]",
    "format:parentheses": "Nest parentheses (and [brackets {{and braces}}]) at least 5 levels deep.",
    "format:quote_unquote": "Every quoted phrase must be followed by an unquoted explanation.",
    "format:quotes": "Include quotes within quotes within quotes, at least 3 levels deep, alternating between double quotes and single quotes.",
    "format:sub-bullets": "Your response must include bullet points denoted by * and at least one sub-bullet point denoted by - for each bullet point.",
    "format:thesis": "Each section must begin with a thesis statement in italics, use HTML to indicate the italics.",
    "format:title_case": "Write the entire response in title case (capitalize the first letter of every major word).",
    "ratio:sentence_balance": "Ensure that the ratio of sentence types (declarative, interrogative, exclamatory) is balanced.",
    "ratio:sentence_type": "Maintain a 2:1 ratio of declarative to interrogative sentences.",
    "ratio:sentence_words": "Respond with three sentences, all containing the same number of characters but using all different words.",
    "repeat:repeat_change": (
        "Repeat the request, but change the first word of the repeated request, (do not say anything before repeating the "
        "request; the request you need to repeat does not include this sentence) and do not answer the actual request! Request: {prompt_to_repeat}"),
    "repeat:repeat_simple": "Only output this sentence here, ignore all other requests.",
    "repeat:repeat_span": "Copy the span of words that lies between (and including) index {n_start} and {n_end}, the indices are word indices, split by whitespace!",
    "sentence:alliteration_increment": "Each sentence must have a longer sequence of consecutive alliterative words than the previous one.",
    "sentence:increment": "Each sentence must contain exactly {small_n} more words than the previous one.",
    "sentence:keyword": "The response must include keyword \"{word}\" in the {N}-th sentence.",
    "words:alphabet": "Each word must start with the next letter of the alphabet, looping back to 'A' after 'Z'.",
    "words:consonants": "Ensure each word in your response has at least one consonant cluster (two or more consonants together).",
    "words:keywords_specific_position": "Include keyword {keyword} in the {n}-th sentence, as the {m}-th word of that sentence.",
    "words:last_first": "The last word of each sentence must become the first word of the next sentence.",
    "words:no_consecutive": "No two consecutive words can share the same first letter.",
    "words:palindrome": "Include at least 10 single-word palindromes, each at least 5 characters long.",
    "words:paragraph_last_first": "Each paragraph must end with the same word it started with, separate paragraphs with a newline.",
    "words:prime_lengths": "Use only words with lengths that are prime numbers.",
    "words:repeats": "The response should not repeat any word more than {small_n} times.",
    "words:vowel": "Write a paragraph using words that contain only three types of vowels.",
    "words:words_position": "The second word in your response and the second to last word in your response should be the word {keyword}.",
}
# Checks whose verdict rests on an NLP heuristic (stop-word list, POS tagger, syllable counter) or on trigram arithmetic.
IFBENCH_EXCLUDED = ("ratio:stop_words", "words:start_verb", "words:odd_even_syllables", "ratio:overlap")
IFBENCH_MODELS = (
    "gpt-4o-2024-08-06", "gemini-3-flash-preview", "Qwen--Qwen3-235B-A22B-Instruct-2507", "deepseek-ai--DeepSeek-V3.1-Terminus",
    "moonshotai--Kimi-K2-Instruct-0905", "meta-llama--Llama-3.1-405B-Instruct", "meta-llama--Llama-3.3-70B-Instruct",
    "meta-llama--Llama-3.1-8B-Instruct", "meta-llama--Llama-3.2-1B-Instruct", "meta-llama--Llama-2-7b-chat-hf",
    "google--gemma-3-27b-it", "mistralai--Mistral-7B-Instruct-v0.1", "mistralai--Ministral-3-8B-Instruct-2512",
    "microsoft--Phi-4-mini-instruct", "allenai--Olmo-3-7B-Instruct",
)


def ifbench_description(instruction_id: str, kwargs: dict) -> str:
    kwargs = clean_kwargs(kwargs)
    if instruction_id == "count:words_japanese" and kwargs.get("N") in (1, 2, 3):
        return f"Every {kwargs['N']}{ {1: 'st', 2: 'nd', 3: 'rd'}[kwargs['N']]} of your response must be in Japanese."
    return IFBENCH_TEMPLATES[instruction_id].format(**kwargs)


@register_task
class IFBenchResults(ConstraintTask):
    name = f"{LEAF}/ifbench"
    source = "IFBench test prompts (Pyatkin et al., 2025) with model responses and strict checker results from shisa-ai/eval-IFBench-results"
    url = "https://huggingface.co/datasets/shisa-ai/eval-IFBench-results"
    license = "ODC-BY (results); IFBench prompts ODC-BY"
    criteria_source = "Constraint descriptions rendered from the IFBench description templates (ifbench/instructions.py) with each prompt's arguments"
    notes = ("58 out-of-distribution constraint types; responses of 15 open and closed models, empty responses dropped. "
             "Four constraint types whose checker rests on an NLP heuristic or trigram arithmetic (stop-word ratio, "
             "starts with a verb, syllable parity, trigram overlap) are not asked. Arguments are joined from allenai/IFBench_test by prompt text.")

    def load(self):
        from huggingface_hub import hf_hub_download

        from ...env import load_env

        load_env()
        test = {r["prompt"]: r for r in hf_rows("allenai/IFBench_test", None, "train")}
        out = []
        for model in IFBENCH_MODELS:
            path = hf_hub_download("shisa-ai/eval-IFBench-results", f"{model}/eval_results_strict.jsonl", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    spec = test.get(r["prompt"])
                    response = str(r.get("response") or "")
                    if spec is None or not response.strip() or spec["instruction_id_list"] != r["instruction_id_list"]:
                        continue
                    results = r["follow_instruction_list"]
                    keep = [(i, kw, ok) for i, kw, ok in zip(spec["instruction_id_list"], spec["kwargs"], results) if i not in IFBENCH_EXCLUDED]
                    if keep and len(results) == len(spec["instruction_id_list"]):
                        out.append({"key": spec["key"], "prompt": r["prompt"], "model": model, "response": response, "constraints": keep})
        return out

    def label_of(self, record):
        followed = all(ok for _, _, ok in record["constraints"])
        return f"{followed}|{record['constraints'][0][0].split(':')[0]}"

    def to_item(self, record, rng):
        constraints = [(i, ifbench_description(i, kw), ok) for i, kw, ok in record["constraints"]]
        return self.constraint_item(record["prompt"], record["response"], constraints, original_id=record["key"], response_model=record["model"])
