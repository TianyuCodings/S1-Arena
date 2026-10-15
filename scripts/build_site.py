"""Write the leaderboard website (site/index.html) from leaderboard/results.json.

Usage (from the repository root; standard library only):

    python scripts/build_site.py                       # writes site/index.html
    python scripts/build_site.py --out _site/index.html

The page is one file: site/template.html with the results and the question examples (leaderboard/examples.json,
site/media/closed_loop.json) embedded as JSON, so it needs no server and no fetch; only the closed-loop video
(site/media/) sits next to it. Edit the template, not index.html. The output depends only on the two inputs, so a second run writes the
same file. `--fragment` writes the page without the document skeleton (for hosts that add their own).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "leaderboard" / "results.json"
TEMPLATE = ROOT / "site" / "template.html"
EXAMPLES = ROOT / "leaderboard" / "examples.json"  # scripts/build_examples.py
VIDEO = ROOT / "site" / "media" / "closed_loop.json"  # scripts/build_video.py; the video itself is site/media/*.mp4
OUT = ROOT / "site" / "index.html"
DESCRIPTION = "S1-Arena leaderboard: which decisions can be left to System 1? Accuracy, calibration, latency and cost of System-1 models and LLMs."


def embed(obj, element_id: str) -> str:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")  # safe inside <script>
    return f'<script type="application/json" id="{element_id}">{data}</script>'


def page(results: dict, template: str, fragment: bool = False, examples: dict | None = None) -> str:
    head, body = template.split("<!--BODY-->\n", 1)
    body = body.replace("<!--DATA-->", embed(results, "s1-data") + "\n" + embed(examples or {}, "s1-examples"), 1)
    if fragment:
        return head + body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<meta name=\"description\" content=\"{DESCRIPTION}\">\n"
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--fragment", action="store_true", help="omit <!doctype>, <html>, <head> and <body>")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8")) if EXAMPLES.exists() else {}
    if examples and VIDEO.exists():
        examples["video"] = json.loads(VIDEO.read_text(encoding="utf-8"))
    html = page(results, TEMPLATE.read_text(encoding="utf-8"), args.fragment, examples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(results.get('models', []))} models, {len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
