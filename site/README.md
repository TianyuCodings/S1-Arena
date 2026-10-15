# Leaderboard website

One static page: `index.html` is `template.html` with `../leaderboard/results.json` embedded, so it opens from disk
and needs no server. Do not edit `index.html`; edit the template or the results, then rebuild:

```bash
python scripts/build_site.py          # writes site/index.html (standard library only)
```

The page has one tab per category (overall, the three non-agentic groups with their domains, and the agentic group).
Each tab shows its latency-accuracy frontier and the ranked table; family chips, search and column sorting apply to
both. The URL keeps the tab (for example `#applied/code`).

Each tab also shows real requests of its domains with their gold answers (`leaderboard/examples.json`, picked
deterministically from `data/` by `scripts/build_examples.py`), and the closed-loop tab plays one recorded episode
(`media/`, made by `scripts/build_video.py`, which replays the episode and checks that it reproduces the recorded
outcome). Both scripts need the data or the episodes, so their outputs are committed; rebuild them only when those
change:

```bash
uv run --with highway-env==1.12.1 --with gymnasium==1.3.0 --with imageio==2.37.0 --with imageio-ffmpeg==0.6.0 \
    --with pillow python scripts/build_video.py --episodes path/to/results/episodes
uv run --with highway-env==1.12.1 --with gymnasium==1.3.0 python scripts/build_examples.py
python scripts/build_site.py
```

`.github/workflows/pages.yml` rebuilds the page and deploys it, with `media/`, to GitHub Pages on every push to `main`.
