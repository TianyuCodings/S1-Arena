# Setting up the environments that need more than `uv sync`

Most environments install with the `envs` extra (and ALFWorld with `alfworld`). The environments added on
2026-09-25 have heavier requirements; they were built and run on Amazon Linux 2023 (x86_64, no root) under
`~/s1arena_env/<name>/`. The scripts here are the ones used there.

- **WebShop** (`loop/web_shopping`): WebShop pins old packages (Python 3.8, pyserini 0.17 with a JDK 11, spaCy 3.3),
  so it runs as a separate server, `s1arena/envs/loop/web_shopping_server.py`, in its own Python 3.8 venv
  (`webshop/requirements_webshop.txt`, WebShop at commit 64fa2a5) with a Temurin JDK 11 unpacked in user space. The
  product data comes from the Hugging Face mirror `YWZBrandon/webshop-data` (`webshop/download_data.sh`; the original
  Google Drive links are gone); the search index is built with WebShop's own scripts (about 8 minutes, 3.4 GB). Start
  and stop the server with `webshop/start_webshop_server.sh` and `stop_webshop_server.sh` (about 2 minutes and 17 GB of
  RAM); the environment talks to it at `$S1ARENA_WEBSHOP_URL` (default `http://127.0.0.1:3000`).
- **BALROG games** (`game/open_world`, `game/rule_puzzles`, MiniHack and NetHack in `game/roguelike`):
  `balrog/setup_balrog_env.sh` builds a Python 3.12 venv with crafter, baba-is-ai, BALROG's MiniHack fork and
  balrog-nle (built from source with a user-space conda-forge toolchain, since its wheels stop at Python 3.11).
  `CLOCKPIN=1` applies `balrog/nle_clockpin.patch.py`, which lets NetHack read its clock from `NLE_FIXED_TIME`; the
  NetHack environment pins it, so games do not change on full-moon, new-moon or Friday-the-13th days.
- **Jericho and ScienceWorld** (`loop/text_games/jericho_*`, `loop/science_experiments`): jericho 3.3.1 (source only;
  needs gcc and make) with spaCy's `en_core_web_sm`, scienceworld at commit e8216d6 (1.3.0; PyPI 1.2.3 is not
  deterministic across JVM launches) and a Java runtime (Temurin JRE 17). Game files are downloaded at run time from
  BYU-PCCL/z-machine-games and checked by SHA-256. `textsci/env.sh` sets the paths used on the build machine.
- **Coding controller** (`loop/coding_controller`): Docker with the SWE-bench evaluation images of the 40 instances
  (`swebench/sweb.eval.x86_64.*`) and `swebench==5.0.2`. The GPT-4.1 mini variant calls the AI Gateway for proposals and
  caches them under `.cache/coding_controller/proposals/`.
