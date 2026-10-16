"""Write the Hugging Face dataset of S1-Arena into a local folder (nothing is uploaded).

Usage (from the repository root):

    uv run python scripts/export_hf_dataset.py [--data data] [--out ../S1-Arena-hf]
    uv run python scripts/export_hf_dataset.py --restricted [--data data_local] [--out ../S1-Arena-restricted-hf]

The folder gets one config per domain (Parquet, split "test"; one row per request), a dataset card (README.md) with the
license of every source, and environments.json, the closed-loop environments with their seeds. Only sources that allow
redistribution are written. With --restricted, the folder gets the other sources (from data_local/) for a private
dataset that only invited people can read; everybody else rebuilds them with `s1arena build --local`. Upload a folder
yourself, for example with `hf upload <owner>/<name> <folder> --repo-type dataset --private`, after you check the card.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from s1arena import dataio, hub
from s1arena.env import DATA, DATA_LOCAL, HF_DATASET, HF_DATASET_RESTRICTED, ROOT
from s1arena.registry import TASKS, discover, envs_under

DOMAIN_NAMES = {
    "basic_cls": "Basic text classification", "know": "Knowledge and reasoning", "social": "Subjective and social judgments",
    "pro": "Professional domains", "ret": "Retrieval, ranking and matching", "struct": "Structured data, tables and logs",
    "code": "Code", "fore": "Forecasting", "dialog": "Dialogue and interaction", "batch": "Batch semantic processing",
    "judge": "LLM-as-judge and evaluators", "safety": "Safety and moderation", "route": "Routing and orchestration",
    "step": "Agent single steps (static)", "game": "Games (static)", "embodied": "Embodied and robotics (static)",
}
SCHEMA = pa.schema([(c, pa.string()) for c in hub.COLUMNS])
CODE = "https://github.com/TianyuCodings/S1-Arena"
SITE = "https://tianyucodings.github.io/S1-Arena"
CITATION = ["@article{s1arena2026,", "  title  = {System1-Arena: Which Decisions Can Be Left to System 1?},",
            "  author = {Chen, Tianyu and Liu, Xinyang and An, Ruyi and Wang, Zhendong and Zhou, Mingyuan},",
            "  journal = {arXiv preprint arXiv:2610.21244},", "  year   = {2026}", "}"]


def card_text(configs: dict[str, int], sources: list[dict], local: list[dict], envs: list[dict]) -> str:
    yaml = ["---", "license: other", "license_name: per-source", "license_link: LICENSE.md", "pretty_name: S1-Arena",
            "language:", "- en", "task_categories:", "- text-classification", "- question-answering",
            "tags:", "- benchmark", "- decision-making", "- calibration", "- llm-evaluation",
            "size_categories:", "- 10K<n<100K", "configs:"]
    for config in configs:
        yaml += [f"- config_name: {config}", "  data_files:", "  - split: test", f"    path: {config}/test-*.parquet"]
    yaml.append("---")
    total = sum(configs.values())
    lines = [
        "# S1-Arena",
        "",
        "S1-Arena asks which decisions a fast System-1 model can make, and which still need a generative LLM. Every "
        "request holds an input (`state`) and one or more typed questions: `choice` (one of 2 to 255 options), "
        "`boolean` (the probability that a statement is true) or `score` (2 to 10 ordered levels). A model answers "
        "each question with a probability distribution over its options.",
        "",
        f"This dataset holds {total:,} built requests from {len(sources)} sources in {len(configs)} domains (one config "
        f"per domain, split `test`). Code: [{CODE}]({CODE}) · leaderboard: [{SITE}]({SITE}).",
        "",
        "## Use",
        "",
        "```bash",
        f"git clone {CODE} && cd S1-Arena && uv sync",
        "uv run s1arena download        # writes data/<task>.jsonl",
        "uv run s1arena build --local   # rebuilds the sources that do not allow redistribution",
        "```",
        "",
        "```python",
        "from datasets import load_dataset",
        f"rows = load_dataset(\"{HF_DATASET}\", \"basic_cls\", split=\"test\")",
        "```",
        "",
        "## Columns",
        "",
        "| Column | Content |",
        "|---|---|",
        "| `id` | content hash of the request; any change to the request changes it |",
        "| `task` | `<domain>/<leaf>/<source>` |",
        "| `leaf` | `<domain>/<leaf>`: the kind of decision; a leaf's score is the mean over its sources |",
        "| `request` | JSON: `{\"state\": {...}, \"questions\": {name: {\"type\", \"instructions\", \"criteria\"}}}`, all a model may see |",
        "| `gold` | JSON: `{name: {\"label\": ..., \"dist\": ...}}`, a hard label, a soft label distribution, or both |",
        "| `source`, `license`, `tier`, `url` | the source as cited, its license, the tier of its instructions (A verbatim, B adapted, C newly written), its page |",
        "| `meta` | JSON: every metadata field of the request |",
        "",
        "## Domains",
        "",
        "| Config | Domain | Requests |",
        "|---|---|---:|",
    ]
    lines += [f"| `{c}` | {DOMAIN_NAMES.get(c, c)} | {n:,} |" for c, n in configs.items()]
    lines += [
        "",
        "## Closed-loop environments",
        "",
        f"`environments.json` lists the {len(envs)} closed-loop environments (games, web and text agents, embodied control, "
        "realtime control) with their seeds. They run in simulators, not from this dataset: `s1arena play`.",
        "",
        "## Licenses",
        "",
        "Each source keeps its own license; check it before you use a source. The code of S1-Arena is Apache-2.0. "
        "Requests that S1-Arena generates are licensed CC BY 4.0.",
        "",
        "| Task | Source | License | Tier | Requests |",
        "|---|---|---|---|---:|",
    ]
    lines += [f"| `{s['task']}` | [{s['source']}]({s['url']}) | {s['license']} | {s['tier']} | {s['requests']:,} |".replace("\n", " ") for s in sources]
    if local:
        lines += ["", f"### Not included ({len(local)} sources)", "",
                  "These sources do not allow redistribution. `s1arena build --local` rebuilds their requests from the original "
                  f"data (people invited to the private dataset `{HF_DATASET_RESTRICTED}` get them with `s1arena download`).", "",
                  "| Task | Source | License |", "|---|---|---|"]
        lines += [f"| `{s['task']}` | [{s['source']}]({s['url']}) | {s['license']} |" for s in local]
    lines += ["", "## Citation", "", "```bibtex", *CITATION, "```", ""]
    return "\n".join(yaml) + "\n" + "\n".join(lines)


def restricted_card_text(configs: dict[str, int], sources: list[dict]) -> str:
    yaml = ["---", "license: other", "license_name: per-source", "pretty_name: S1-Arena (restricted sources)",
            "language:", "- en", "tags:", "- benchmark", "configs:"]
    for config in configs:
        yaml += [f"- config_name: {config}", "  data_files:", "  - split: test", f"    path: {config}/test-*.parquet"]
    yaml.append("---")
    lines = [
        "# S1-Arena: restricted sources",
        "",
        f"This private dataset holds the {sum(configs.values()):,} built requests of the {len(sources)} S1-Arena sources "
        f"whose terms do not allow redistribution. The other sources are in the public dataset "
        f"[{HF_DATASET}](https://huggingface.co/datasets/{HF_DATASET}), with the same columns and one config per domain.",
        "",
        "**Keep this dataset private, and do not share its files.** Each source's terms (below) apply. People without "
        "access rebuild the same requests from the original data with `s1arena build --local`.",
        "",
        "`s1arena download` reads this dataset into `data_local/` when your Hugging Face token has access to it.",
        "",
        "| Task | Source | License | Tier | Requests |",
        "|---|---|---|---|---:|",
    ]
    lines += [f"| `{s['task']}` | [{s['source']}]({s['url']}) | {s['license']} | {s['tier']} | {s['requests']:,} |".replace("\n", " ") for s in sources]
    lines += ["", "## Citation", "", "```bibtex", *CITATION, "```", ""]
    return "\n".join(yaml) + "\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--restricted", action="store_true", help="write the sources that do not allow redistribution instead")
    parser.add_argument("--data", default=None, help="folder with the built requests (default: data/, or data_local/ with --restricted)")
    parser.add_argument("--out", default=None, help="output folder outside the repository (default: ../S1-Arena-hf or ../S1-Arena-restricted-hf)")
    args = parser.parse_args()
    discover()
    data = Path(args.data or (DATA_LOCAL if args.restricted else DATA)).resolve()
    out = Path(args.out or ROOT.parent / ("S1-Arena-restricted-hf" if args.restricted else "S1-Arena-hf")).expanduser().resolve()
    rows: dict[str, list[dict]] = defaultdict(list)
    sources, skipped = [], Counter()
    for path in sorted(data.rglob("*.jsonl")):
        task = path.relative_to(data).with_suffix("").as_posix()
        cls = TASKS.get(task)
        if cls is None:
            skipped["not registered"] += 1
            continue
        card = cls().card()
        if card["redistributable"] == args.restricted:
            skipped["redistributable" if args.restricted else "not redistributable"] += 1
            continue
        items = dataio.read_items(path)
        for item in items:
            row = hub.item_to_row(item, card)
            rows[task.split("/")[0]].append(row)
        sources.append({"task": task, "source": card["source"], "url": card["url"], "license": card["license"], "tier": card["tier"], "requests": len(items)})
    order = list(DOMAIN_NAMES)
    configs = {}
    for domain in sorted(rows, key=lambda d: order.index(d) if d in order else len(order)):
        folder = out / domain
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.parquet"):
            old.unlink()
        table = pa.Table.from_pylist(rows[domain], schema=SCHEMA)
        pq.write_table(table, folder / "test-00000-of-00001.parquet", compression="zstd")
        configs[domain] = table.num_rows
    if args.restricted:
        (out / "README.md").write_text(restricted_card_text(configs, sources), encoding="utf-8")
        for config, n in configs.items():
            print(f"{config:<10} {n:>7,} requests")
        print(f"total      {sum(configs.values()):>7,} requests from {len(sources)} sources; skipped files: {dict(skipped) or 'none'}")
        print(f"wrote {out}")
        return
    local = [{"task": n, **{k: c().card()[k] for k in ("source", "url", "license")}} for n, c in sorted(TASKS.items()) if not c.redistributable]
    from s1arena.envs.loop.text_games import ALFWorld  # registered only with the `alfworld` extra; listed in any case

    classes = sorted({*envs_under(""), ALFWorld}, key=lambda cls: cls.name)
    envs = [{**{k: v for k, v in cls().card().items() if k not in ("criteria_source",)}, "suite": cls.leaf, "seeds": list(range(cls.episodes))}
            for cls in classes]
    (out / "environments.json").write_text(json.dumps(envs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "README.md").write_text(card_text(configs, sources, local, envs), encoding="utf-8")
    (out / "LICENSE.md").write_text(
        "# Licenses\n\nEach source of S1-Arena keeps its own license, listed in README.md. Requests that S1-Arena generates are "
        "licensed CC BY 4.0. The S1-Arena code is licensed Apache-2.0.\n", encoding="utf-8")
    for config, n in configs.items():
        print(f"{config:<10} {n:>7,} requests")
    print(f"total      {sum(configs.values()):>7,} requests from {len(sources)} sources; {len(envs)} environments; skipped files: {dict(skipped) or 'none'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
