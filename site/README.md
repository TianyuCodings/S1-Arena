# Leaderboard website

One static page: `index.html` is `template.html` with `../leaderboard/results.json` embedded, so it opens from disk
and needs no server. Do not edit `index.html`; edit the template or the results, then rebuild:

```bash
python scripts/build_site.py          # writes site/index.html (standard library only)
```

The page has one tab per category (overall, the three non-agentic groups with their domains, and the agentic group).
Each tab shows its latency-accuracy frontier and the ranked table; family chips, search and column sorting apply to
both. The URL keeps the tab (for example `#applied/code`).

`.github/workflows/pages.yml` rebuilds the page and deploys it to GitHub Pages on every push to `main`.
