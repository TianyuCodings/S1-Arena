# Models

A model enters S1-Arena through an adapter: a `ModelAdapter` subclass in `s1arena/models/` that turns a request into
one probability distribution per question. A spec selects the adapter and the model: `<kind>[:<argument>]`, for
example `random`, `jev`, `llm:openai/gpt-4.1-mini` or `lm:Qwen/Qwen3-8B`. Every adapter reports how it obtains its
probabilities (`probability_source`), and the leaderboard shows it. To add a model, see
[CONTRIBUTING.md](../CONTRIBUTING.md); check a new adapter with `uv run s1arena check-model <spec>`.

## Adapters

| Kind | Example spec | What it runs | Probabilities | Needs |
|---|---|---|---|---|
| `random` | `random` | random distributions, to test the pipeline | random | nothing |
| `llm` | `llm:openai/gpt-4.1-mini` | any chat model on the Vercel AI Gateway | verbalized (JSON reply) | `AI_GATEWAY_API_KEY` |
| `copilot` | `copilot:gpt-6-luna@effort=none` | chat models through a GitHub Copilot subscription | verbalized | `copilot` extra, `copilot login` |
| `bedrock` | `bedrock:us.openai.gpt-6-luna@effort=none` | OpenAI models on Amazon Bedrock (Converse API) | verbalized | `AWS_API_KEY`, `BEDROCK_REGION` |
| `jev` | `jev` | TypeSafe's Jev through the AI Gateway | native | `AI_GATEWAY_API_KEY` |
| `decisions` | `decisions` | OpenAI's Decisions API (GPT-6 Luna) | native | `OPENAI_API_KEY` |
| `ordecisions` | `ordecisions` | OpenRouter's Decisions API (Microsoft-Decision-1) | native | `OPENROUTER_API_KEY` |
| `decision2` | `decision2:vllm-sr/Decision-2.0-Kai-0.6B` | Decision 2.0 models through their own code | native (candidate logits) | torch, transformers ≥ 5.17, GPU |
| `laya` | `laya:convaiinnovations/laya` | Laya through its library | native (option logits) | torch, `pip install --no-deps laya==0.3.27` |
| `clm` | `clm:Contrastive-LM/CLM-v0.1-8B` | CLM projection heads on a frozen encoder | native (state-option similarity) | torch, `pip install --no-deps contrastive-lm==0.1.0`, an embeddings endpoint (`CLM_EMB_URL`) |
| `lm` | `lm:Qwen/Qwen3-8B` | any autoregressive model on vLLM | label-token log-probabilities at the answer position | vLLM, GPU |
| `thinklm` | `thinklm:Qwen/Qwen3-8B@think=1024` | `lm` after a thought of at most `think` tokens | label-token log-probabilities after the thought | vLLM, GPU |
| `dgemma` | `dgemma:google/diffusiongemma-26B-A4B-it` | DiffusionGemma, one read of an answer slot | answer-slot log-probabilities | vLLM with structured reads, GPU |

Put keys in `.env` (never committed). `uv run python -m s1arena.models.copilot_llm` lists the Copilot model ids, their
prices and reasoning efforts. Runs of one model through different services are stored and labeled separately
(`copilot/<model>`, `bedrock/<id>`), because the services sample differently. The docstring of each adapter module
gives its options and details.

## Shared rules

- **One prompt.** Every chat LLM gets the same two messages (`s1arena/render.py`) and states a probability for each
  option in a JSON object. The parser takes the last JSON object of the reply that names a question. A reply without
  one is a failed call. `s1arena reparse` reads stored replies again after a parser change, without new calls.
- **No answer scores 0.** A failed call, a reply that cannot be used, a question the reply leaves out, or an input
  the model refuses (for example because it is longer than its context) counts as wrong. For the Brier score and
  calibration, such a question counts as a uniform distribution.
- **No shared input budget.** Long inputs are kept. Each model is limited only by its own context. Two System-1 models
  cut long inputs as released: Laya's library keeps 512 tokens, and the CLM adapter keeps the last 2,048 tokens of a
  text. With `strict=1`, both cut nothing and refuse an input that does not fit. Decision 2.0 refuses such an input in
  its own runtime.
- **Latency and cost.** Every response records `latency_ms`, and `cost_usd` where the service has a price. Local
  models record the GPU time of their batches.

## Evaluated models

The leaderboard (`leaderboard/results.json`) lists the models of the first release with the run id of each result:
10 System-1 models (Decision 2.0 Vega, Lux, Nox, Sol and Kai; Jev; the OpenAI Decisions API; Microsoft-Decision-1;
Laya; CLM), 7 closed LLMs (Claude Opus 5.5, GPT-6 Astra, Grok 4.7, GPT-6 Sol and Luna, GPT-5.6 Sol and Luna) and
13 open LLMs (Gemma 4, Qwen3, Qwen3.5, Qwen3.8, Llama 3, gpt-oss and DiffusionGemma). Closed LLMs answer without
reasoning where they can. Claude Opus 5.5, GPT-6 Astra, Grok 4.7 and gpt-oss cannot turn reasoning off; they run at
their lowest effort.
