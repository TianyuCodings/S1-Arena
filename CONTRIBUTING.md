# Contributing

S1-Arena grows by three kinds of plug-ins: a **task** (one static benchmark source), a **closed-loop environment**
and a **model adapter**. Each is one Python module with one decorated class. Discovery imports every module under
`s1arena/tasks/`, `s1arena/envs/` and `s1arena/models/`, so nothing else changes. Copy a template from `templates/`,
rename it, and replace its body.

Set up once:

```bash
uv sync
uv run pytest -q
uv run s1arena download          # the built requests, from Hugging Face
```

## Add a task (a static benchmark source)

1. Copy `templates/task_template.py` to `s1arena/tasks/<domain>/<module>.py`. Use an existing domain
   (`uv run s1arena tasks | cut -d/ -f1 | sort -u`); a new leaf is fine.
2. Set the class attributes: `name` (`<domain>/<leaf>/<source>`), `leaf`, `source` (as you cite it), `url`, `license`,
   `tier` (A: the source's instructions verbatim; B: adapted; C: newly written) and `criteria_source`. Set
   `redistributable = False` if the license does not allow sharing; its requests then go to `data_local/` and stay
   out of the Hugging Face dataset.
3. Write `load()` (the raw records of the evaluation split; `hf_rows()` and `fetch()` in `s1arena/tasks/common.py`
   download and cache), `label_of()` (the key for stratified sampling) and `to_item()` (one request: the `state` the
   model reads, the typed questions, and the gold answers). Build the item with `self.item(...)`.
4. Check and build it:

   ```bash
   uv run s1arena check-task <name>                    # validates 5 items and prints one request as an LLM sees it
   uv run s1arena build --leaf <name> --n 500
   uv run s1arena run --model random --leaf <name> --limit 20 && uv run s1arena score --models random --leaf <name> --limit 20
   ```

Rules for a good task:

- The state holds only what a model may see. Put the answer in `gold`, provenance in `meta` (extra keyword arguments
  of `self.item`).
- Question types: `choice` (2 to 255 options, a dict from option id to description), `boolean` (a statement; no
  criteria) or `score` (2 to 10 ordered levels, a list). A multi-label task uses several boolean questions.
- Give soft labels (`Gold(dist=...)`) when the source has several annotators.
- Present choice options in a random but reproducible order: `shuffled_options(options, rng)`.
- Pairs asked in both orders: override `to_items()` and set `items_per_record = 2`; pair accuracy is then reported.
  Ranking tasks set `rank_key`; NDCG is then reported.

## Add a closed-loop environment

1. Copy `templates/env_template.py` to `s1arena/envs/<domain>/<module>.py` (`embodied`, `game`, `loop` or `realtime`).
2. Implement `reset(seed)`, `observe()` (the state and the action questions as a `DecisionRequest`), `act(choice)`,
   `done()` and `outcome()` (`{"success": bool or None, "score": float}`). Import simulator packages inside the
   methods, so discovery works on machines without them.
3. Choose the timing: `mode = "paused"` (the world waits) or `"realtime"` (set `control_dt` and implement `idle()`;
   the runner charges each model its fixed latency and keeps its previous action running meanwhile).
4. Play it:

   ```bash
   uv run s1arena envs --env <name>
   uv run s1arena play --model random --env <name> --episodes 2 --max-decisions 30
   uv run s1arena env-score --models random --env <name> --episodes 2 --max-decisions 30
   ```

Raise `version` when the requests or the dynamics change, so cached episodes of the old harness stop counting. Check
that a random policy rarely succeeds and that a scripted policy can (see `scripts/realtime_oracle.py`).

## Add a model

1. Copy `templates/model_template.py` to `s1arena/models/<module>.py` and set `kind` (the spec prefix).
2. Implement `decide(request, sample)`. Return one distribution per question over `question.option_ids()`. Leave a
   question out, or set `error`, when the model cannot answer it; the question then scores 0. Record `latency_ms` and,
   if the service has a price, `cost_usd`.
3. Set `native_probs` and `probability_source`, which the leaderboard shows. A model that answers in free text can
   implement `reparse()`, so stored replies can be read again after a parser change.
4. Local models can implement `decide_many(requests)` to answer a batch at once.
5. Check and run it:

   ```bash
   uv run s1arena check-model <spec>                   # three toy requests: choice, boolean and score
   uv run s1arena estimate --model <spec> --leaf basic_cls
   uv run s1arena sweep sweeps/smoke.toml --models <spec>   # add the spec to a copy of the plan first
   ```

## Submit results to the leaderboard

Run the full plan (`sweeps/full.toml`, with your model) and the report (`s1arena sweep-report`). Open a pull request
with the adapter, the plan and the report files from `results/summary/`, and say on which hardware you measured the
latency. We check the run and add the model to `leaderboard/results.json`.

## Style

- Code: match the surrounding code; type hints; short docstrings that say what and why.
- Docs: short sentences in the active voice, one term for one thing.
- Run `uv run pytest -q` before you open a pull request.
