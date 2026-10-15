"""Write the leaderboard website (site/index.html) from leaderboard/results.json.

Usage (from the repository root; standard library only):

    python scripts/build_site.py                       # writes site/index.html
    python scripts/build_site.py --out _site/index.html

The page is one self-contained file: site/template.html with the results embedded as JSON, so it needs no server and
no fetch. Edit the template, not index.html. The output depends only on the two inputs, so a second run writes the
same file. `--fragment` writes the page without the document skeleton (for hosts that add their own).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "leaderboard" / "results.json"
TEMPLATE = ROOT / "site" / "template.html"
OUT = ROOT / "site" / "index.html"
DESCRIPTION = "S1-Arena leaderboard: which decisions can be left to System 1? Accuracy, calibration, latency and cost of System-1 models and LLMs."


def page(results: dict, template: str, fragment: bool = False) -> str:
    head, body = template.split("<!--BODY-->\n", 1)
    data = json.dumps(results, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")  # safe inside <script>
    body = body.replace("<!--DATA-->", f'<script type="application/json" id="s1-data">{data}</script>', 1)
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
    html = page(results, TEMPLATE.read_text(encoding="utf-8"), args.fragment)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(results.get('models', []))} models, {len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
