# S1-Arena

**Which decisions can be left to System 1?**

Many applications ask a language model for a discrete decision: is this e-mail phishing, which tool comes next, did
the agent finish the task. S1-Arena measures how well models make such decisions and how good their probabilities
are. Every request holds an input and one or more typed questions, and every model answers each question with a
probability distribution over its options. Fast System-1 models (one forward pass, native probabilities) and
generative LLMs (verbalized probabilities or label-token probabilities) answer the same requests and get the same
scores.

[Leaderboard](https://tianyucodings.github.io/S1-Arena) · [Paper (arXiv)](https://arxiv.org/abs/2610.21244) · [Dataset](https://huggingface.co/datasets/C-Tianyu/S1-Arena)

## What is in it

| Part | Size |
|---|---|
| Static tasks | 268 sources in 105 kinds of decision ("leaves") and 16 domains; 52,199 requests |
| Non-agentic domains | 13 domains, 220 sources, 43,226 requests, 78,362 questions |
| Agentic static tasks | agent single steps, game positions, embodied decisions: 48 sources, 8,973 requests |
| Closed-loop environments | 138 games, web and text agents, coding, embodied control and realtime control |
| Leaderboard | 30 models: 10 System-1 models, 7 closed LLMs, 13 open LLMs |

Domains of the static tasks (task cards in `docs/`, licenses in [docs/sources.md](docs/sources.md)):

| Domain | Prefix | Leaves | Sources | Requests |
|---|---|---:|---:|---:|
| Basic text classification | `basic_cls` | 10 | 26 | 5,000 |
| Knowledge and reasoning | `know` | 12 | 30 | 6,000 |
| Subjective and social judgments | `social` | 4 | 14 | 1,999 |
| Professional domains | `pro` | 9 | 26 | 4,500 |
| Retrieval, ranking and matching | `ret` | 5 | 13 | 2,499 |
| Structured data, tables and logs | `struct` | 5 | 13 | 2,500 |
| Code | `code` | 8 | 17 | 3,886 |
| Forecasting | `fore` | 3 | 7 | 1,420 |
| Dialogue and interaction | `dialog` | 5 | 10 | 2,464 |
| Batch semantic processing | `batch` | 3 | 6 | 1,500 |
| LLM-as-judge and evaluators | `judge` | 10 | 24 | 4,998 |
| Safety and moderation | `safety` | 6 | 16 | 2,998 |
| Routing and orchestration | `route` | 7 | 18 | 3,462 |
| Agent single steps | `step` | 7 | 18 | 3,481 |
| Games (static) | `game` | 5 | 13 | 2,500 |
| Embodied and robotics (static) | `embodied` | 6 | 17 | 2,992 |

The closed-loop environments are listed in [docs/envs.md](docs/envs.md).

## Leaderboard (top of the non-agentic tasks)

| Rank | Model | Family | Accuracy | Brier | Latency p50 | Cost / 1,000 requests |
|---:|---|---|---:|---:|---:|---:|
| 1 | Claude Opus 5.5 | closed LLM | 81.1% | 0.232 | 2,293 ms | $10.38 |
| 2 | GPT-6 Astra | closed LLM | 80.1% | 0.282 | 2,189 ms | $15.05 |
| 6 | Gemma-4-31B | open LLM | 73.9% | 0.474 | 60 ms | 142 GPU-s |
| 7 | Decision-2.0-Vega-27B | System-1 | 73.0% | 0.343 | 228 ms | 316 GPU-s |
| 8 | Jev 1.13 | System-1 | 72.6% | 0.349 | 321 ms | $0.05 |
| 10 | GPT-6 Luna | closed LLM | 71.2% | 0.461 | 1,276 ms | $0.14 |

All 30 models, per-domain scores and the agentic results: [leaderboard/results.json](leaderboard/results.json) and
the [website](https://tianyucodings.github.io/S1-Arena). GPU-seconds are H100-equivalent.

## Install

S1-Arena needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TianyuCodings/S1-Arena.git
cd S1-Arena
uv sync                       # add `--extra envs` for the simulators of the closed-loop environments
uv run pytest -q
```

## Quickstart

```bash
uv run s1arena download                                    # the built requests, from Hugging Face, into data/
uv run s1arena build --local                               # the 14 sources that do not allow redistribution, into data_local/
uv run s1arena run --model random --leaf basic_cls --limit 20
uv run s1arena score --models random --leaf basic_cls --limit 20
```

A hosted LLM through the Vercel AI Gateway (put `AI_GATEWAY_API_KEY=...` in `.env`; estimate the cost first, no calls
are made):

```bash
uv run s1arena estimate --model llm:openai/gpt-4.1-mini --leaf basic_cls
uv run s1arena run --model llm:openai/gpt-4.1-mini --leaf basic_cls --workers 8
uv run s1arena compare --models random llm:openai/gpt-4.1-mini --leaf basic_cls
```

A closed-loop environment:

```bash
uv sync --extra envs
uv run s1arena play --model random --env embodied/mujoco_control --episodes 2 --max-decisions 30
uv run s1arena env-score --models random --env embodied/mujoco_control --episodes 2 --max-decisions 30
```

Everything for many models at once is a sweep plan (`sweeps/smoke.toml`, `sweeps/full.toml`):

```bash
uv run s1arena sweep sweeps/smoke.toml          # 3 requests per task, 1 short episode per environment; no keys
uv run s1arena sweep-status sweeps/full.toml    # coverage per model and domain, no calls
uv run s1arena sweep-report sweeps/full.toml    # result tables, no calls
```

Runs are cached per (model, request, sample) in `results/runs/` and episodes per (model, environment, seed) in
`results/episodes/`, so a stopped run continues where it stopped. Clusters, containers and AWS:
[docs/deploy.md](docs/deploy.md).

## Requests

Every task produces the same format (`s1arena/schema.py`):

```json
{
  "task": "basic_cls/intent/clinc150",
  "request": {
    "state": {"request": "..."},
    "questions": {
      "intent": {"type": "choice", "instructions": "...", "criteria": {"option_id": "description"}}
    }
  },
  "gold": {"intent": {"label": "option_id", "dist": null}},
  "meta": {"source": "...", "license": "...", "tier": "A"}
}
```

- `request` is everything a model may see. A model never sees `gold` or `meta`.
- Question types: `choice` (one of 2 to 255 options), `boolean` (the probability that a statement is true) and `score`
  (2 to 10 ordered levels). A multi-label task uses several boolean questions.
- `gold` holds a hard label, a soft label distribution (from several annotators), or both.
- A request id is a hash of its content, so any change to a request changes its id.

## Metrics

All models are scored by the same code (`s1arena/metrics.py`), from their distributions only:

- **Accuracy** (avg@k, and maj@k over k samples). A domain's accuracy is the mean over its tasks.
- **Brier score** against the one-hot or soft target, and the **expected calibration error** (top option, 10
  equal-mass bins; tied confidences share their mean correctness).
- **Latency** (median of single requests) and **cost** per 1,000 requests.
- **NDCG** for ranking tasks and **pair accuracy** for requests asked in both orders, where a task declares them.
- In closed-loop environments: the **success rate** and the raw score of the simulator.

**A question without an answer scores 0**: a failed call, a reply that cannot be used, or an input the model refuses
(for example because it is longer than its context). There is no shared input budget; each model is limited only by
its own context. `s1arena sweep-report` lists the questions each model left without an answer.

## Add a benchmark or a model

A task, an environment and a model adapter are each one module with one decorated class; discovery finds them. Copy a
template from [templates/](templates), then check it:

```bash
uv run s1arena check-task <task name>     # validates 5 items, prints one request as an LLM sees it
uv run s1arena check-model <model spec>   # three toy requests: choice, boolean and score
```

The recipes are in [CONTRIBUTING.md](CONTRIBUTING.md); the adapters and their needs are in
[docs/models.md](docs/models.md).

## Data and licenses

The code is licensed under Apache-2.0 ([LICENSE](LICENSE)). Each source keeps its own license; check it before you
use a source ([docs/sources.md](docs/sources.md)). Requests that S1-Arena generates are licensed CC BY 4.0. The
Hugging Face dataset holds the requests of every source that allows redistribution; `s1arena build --local` rebuilds
the other 14 sources from the original data (or `s1arena download` reads them from a private dataset, if you have
access to it). Some tasks contain offensive or disturbing text by design (toxicity,
safety and moderation sources).

## Citation

```bibtex
@article{s1arena2026,
  title   = {System1-Arena: Which Decisions Can Be Left to System 1?},
  author  = {Chen, Tianyu and Liu, Xinyang and An, Ruyi and Wang, Zhendong and Zhou, Mingyuan},
  journal = {arXiv preprint arXiv:2610.21244},
  year    = {2026}
}
```
