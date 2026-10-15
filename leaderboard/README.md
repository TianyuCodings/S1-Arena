# Leaderboard data

`results.json` holds the numbers of the leaderboard website: one entry per model, aggregate scores only (no model
responses). `scripts/build_leaderboard.py` writes it from the results summary; do not edit it by hand.

Per model:

| Field | Content |
|---|---|
| `rank` | rank by overall accuracy on the non-agentic tasks (`null` for a model with agentic results only) |
| `name`, `family` | display name; System-1 model, closed LLM or open LLM |
| `probabilities`, `reasons` | how the probabilities are obtained; whether the model writes a reasoning text first |
| `ids`, `link` | run ids of the results (non-agentic, agentic); Hugging Face page of open weights |
| `accuracy`, `brier` | over the 13 non-agentic domains (mean over tasks); a question without an answer is wrong |
| `groups`, `domains` | accuracy per group and per domain (`non_agentic.domains` names them) |
| `latency_ms`, `latency_where` | median time of one request; local GPU or network service |
| `cost_per_1k_requests` | `value` and `unit`: USD for hosted models, H100-equivalent GPU-seconds for local models |
| `output_tokens` | mean tokens the model writes per request (0 for a label read) |
| `agentic` | `step_accuracy` (static agent steps), `loop_success` (closed-loop environments), shares without an answer |

## Leaderboard website

After a change to `results.json`, run `python scripts/build_site.py` to rebuild `site/index.html` (see
`site/README.md`). The GitHub Pages workflow also rebuilds it on every push to `main`.
