"""Command line: s1arena <command>. `s1arena --help` lists the commands."""

from __future__ import annotations

import argparse
import json
import sys

from . import dataio, metrics, runner
from .env import DATA, DATA_LOCAL, HF_DATASET, HF_DATASET_RESTRICTED, SUMMARY, load_env
from .registry import envs_under, make_model, tasks_under


def cmd_tasks(args: argparse.Namespace) -> None:
    for task in tasks_under(args.leaf):
        c = task.card()
        print(f"{c['name']:<48} tier {c['tier']}  {c['license']:<18} {c['source']}")


def cmd_build(args: argparse.Namespace) -> None:
    load_env()
    manifest = []
    tasks = tasks_under(args.leaf)
    if args.local:  # only the sources that do not allow redistribution (they are not in the Hugging Face dataset)
        tasks = [t for t in tasks if not t.redistributable]
    sizes = {t.name: args.n for t in tasks}
    if args.per_leaf:  # split one budget per leaf evenly across all its sources, also when rebuilding one
        for t in tasks:
            names = sorted(other.name for other in tasks_under(t.leaf))
            i = names.index(t.name)
            sizes[t.name] = args.per_leaf // len(names) + (1 if i < args.per_leaf % len(names) else 0)
    failed = []
    for task in tasks:
        try:
            items = task.build(n=sizes[task.name], seed=args.seed)
        except Exception as error:  # e.g. a gated dataset whose terms were not accepted; the others still build
            failed.append(task.name)
            print(f"{task.name}: FAILED, skipped ({type(error).__name__}: {str(error)[:200]})")
            continue
        path = dataio.task_path(task.name, task.redistributable)
        dataio.write_items(path, items)
        labels: dict[str, int] = {}
        for item in items:
            for gold in item.gold.values():
                labels[gold.hard()] = labels.get(gold.hard(), 0) + 1
        excluded = getattr(task, "excluded", 0)
        manifest.append({**task.card(), "items": len(items), "excluded_over_budget": excluded, "duplicates_replaced": getattr(task, "duplicates", 0),
                         "path": str(path.relative_to(DATA.parent)), "label_counts": labels})
        print(f"{task.name}: {len(items)} items -> {path.relative_to(DATA.parent)}" + (f" ({excluded} over the token budget excluded)" if excluded else ""))
    for domain in sorted({row["name"].split("/")[0] for row in manifest}):  # one manifest per domain
        rows = [row for row in manifest if row["name"].startswith(domain + "/")]
        dataio.write_manifest(DATA / domain, rows, known={t.name for t in tasks_under(domain)})
    if failed:
        print(f"{len(failed)} task(s) not built: {', '.join(failed)}")


def cmd_download(args: argparse.Namespace) -> None:
    """Fetch the built requests of every redistributable source from the Hugging Face dataset into data/, and the
    others from the private restricted dataset into data_local/ when the token has access to it."""
    from . import hub

    counts = hub.download(args.repo, args.domains or None, args.revision)
    print(f"wrote {sum(counts.values()):,} requests of {len(counts)} tasks to {DATA}")
    if not args.restricted_repo:
        return
    try:
        counts = hub.download(args.restricted_repo, args.domains or None, None, restricted=True)
    except Exception as error:  # no access (private dataset) or no token: the loaders rebuild these sources
        counts, reason = {}, type(error).__name__
    else:
        reason = "no requests"
    if counts:
        print(f"wrote {sum(counts.values()):,} requests of {len(counts)} tasks to {DATA_LOCAL}")
    else:
        print(f"{args.restricted_repo}: not available ({reason}); "
              "build the sources that do not allow redistribution with `s1arena build --local`.")


def cmd_check_task(args: argparse.Namespace) -> None:
    """Load a few records of one task, validate the items and show one rendered request."""
    from . import checks

    task, items, problems = checks.check_task(args.name, args.n)
    print(f"{task.name}: {len(items)} items checked (source: {task.source}; license: {task.license}; tier {task.tier})")
    if items:
        item = items[0]
        print("\n" + checks.rendered(item.request))
        print("\ngold:", json.dumps({k: v.model_dump(exclude_none=True) for k, v in item.gold.items()}, ensure_ascii=False))
    for problem in problems:
        print("PROBLEM:", problem)
    if problems:
        raise SystemExit(1)
    print("OK")


def cmd_check_model(args: argparse.Namespace) -> None:
    """Run a model on three toy requests (choice, boolean, score) and check its distributions."""
    from . import checks

    load_env()
    rows = checks.check_model(args.spec)
    for row in rows:
        latency = f"{row['latency_ms']:.0f} ms" if row["latency_ms"] is not None else "-"
        answers = "; ".join(f"{q}: " + ", ".join(f"{o}={p:.2f}" for o, p in probs.items()) for q, probs in row["answers"].items())
        print(f"{row['kind']:<8} {latency:>9}  {answers}")
        for problem in row["problems"]:
            print("  PROBLEM:", problem)
    if any(row["problems"] for row in rows):
        raise SystemExit(1)
    print("OK")


def cmd_run(args: argparse.Namespace) -> None:
    model = make_model(args.model)
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        if not items:
            print(f"{task.name}: not built, skipping")
            continue
        calls = runner.run_task(model, task.name, items, k=args.k, workers=args.workers)
        print(f"{task.name}: {calls} new calls")


def cmd_reparse(args: argparse.Namespace) -> None:
    model = make_model(args.model)
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        if items:
            print(f"{task.name}: {runner.reparse_task(model, task.name, items)} replies re-read successfully")


def cmd_score(args: argparse.Namespace) -> None:
    table = []
    for spec in args.models:
        model = make_model(spec)
        for task in tasks_under(args.leaf):
            items = dataio.load_task(task)[: args.limit or None]
            keep = {i.id for i in items}
            rows = [r for r in runner.latest_rows(model.slug, task.name) if r["item_id"] in keep]
            ids = {r["item_id"] for r in rows}
            result = metrics.task_metrics(task, [i for i in items if i.id in ids], rows)
            if result.get("n_questions"):
                table.append({"model": model.model_id, "task": task.name, "leaf": task.leaf, **result})
    for row in table:
        cells = [f"{row['model']:<28}", f"{row['task']:<44}", f"n={row['n_questions']:<4}"]
        cells += [f"acc={row['accuracy']:.3f}±{row['accuracy_se']:.3f}", f"maj={row['maj']:.3f}", f"brier={row['brier']:.3f}", f"ece={row['ece']:.3f}"]
        if row.get("latency_p50_ms") is not None:
            cells.append(f"p50={row['latency_p50_ms']:.0f}ms")
        if row.get("cost_per_1k_usd") is not None:
            cells.append(f"${row['cost_per_1k_usd']:.4f}/1k")
        if row.get("ndcg") is not None:
            cells.append(f"ndcg={row['ndcg']:.3f}")
        if row.get("pair_accuracy") is not None:
            cells.append(f"pair={row['pair_accuracy']:.3f}")
        cells.append(f"err={row['error_rate']:.1%}")
        print("  ".join(cells))
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        path = SUMMARY / args.out
        path.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")


REDACTED = "[not shown: this source does not allow redistribution; run the loader locally to see the text]"


def cmd_examples(args: argparse.Namespace) -> None:
    """One full example item per task, for the website."""
    out = {}
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)
        if not items:
            continue
        short = [i for i in items if len(json.dumps(i.request.state, ensure_ascii=False)) <= args.max_chars]
        pool = short or items
        item = json.loads(pool[args.index % len(pool)].model_dump_json())
        if not task.redistributable:  # show the question structure, not the licensed text
            item["request"]["state"] = {k: REDACTED for k in item["request"]["state"]}
        out[task.name] = {"card": task.card(), "item": item}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {len(out)} examples to {args.out}")
    else:
        sys.stdout.write(text + "\n")


def cmd_cards(args: argparse.Namespace) -> None:
    """Markdown table of task cards plus item and label counts, for docs/."""
    lines = [f"# Tasks under `{args.leaf or 'all'}`", "", "Generated by `s1arena cards`. Tier A: original instructions verbatim; B: adapted; C: newly written.", ""]
    lines += ["| Task | Source | License | Tier | Items | Labels | Criteria source | Notes |", "|---|---|---|---|---|---|---|---|"]
    for task in tasks_under(args.leaf):
        c = task.card()
        items = dataio.load_task(task)
        labels = {g.hard() for i in items for g in i.gold.values()}
        note = c["notes"] + ("" if c["redistributable"] else " Loader only: items are not redistributed.")
        cells = [f"`{c['name']}`", f"[{c['source']}]({c['url']})", c["license"], c["tier"], str(len(items)), str(len(labels)), c["criteria_source"], note.strip()]
        lines.append("| " + " | ".join(x.replace("|", "/") for x in cells) + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate the cost of running a model on built items (no model calls are made)."""
    from . import estimate

    model = make_model(args.model)
    rows = []
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        if items:
            rows.append(estimate.estimate_task(model, task.name, items, args.k))
    for r in rows:
        measured = f"${r['measured_usd']:.2f} (from {r['measured_from_calls']} calls)" if r["measured_usd"] is not None else "-"
        price = f"${r['price_based_usd']:.2f}" if r["price_based_usd"] is not None else "-"
        print(f"{r['task']:<48} calls={r['calls']:<6} in={r['input_tokens']:>10,} tok  price-based={price:<9} measured={measured}")
    total_price = sum(r["price_based_usd"] or 0 for r in rows)
    measured = [r for r in rows if r["measured_usd"] is not None]
    print(f"TOTAL calls={sum(r['calls'] for r in rows):,}  price-based=${total_price:.2f}" + (f"  measured=${sum(r['measured_usd'] for r in measured):.2f} over {len(measured)}/{len(rows)} tasks" if measured else ""))


METRICS = ("accuracy", "maj", "brier", "ece", "latency_p50_ms", "cost_per_1k_usd", "ndcg", "pair_accuracy")


def cmd_compare(args: argparse.Namespace) -> None:
    """Per-leaf pivot of several models on the same items.

    A leaf's score is the mean over its tasks; the overall score is the simple mean over
    leaves (median for latency), so leaves with more sources do not weigh more.
    """
    import numpy as np

    models = [make_model(spec) for spec in args.models]
    names = [m.model_id for m in models]
    by_leaf: dict[str, dict[str, dict[str, list[float]]]] = {}  # leaf -> model -> metric -> task values
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        keep = {i.id for i in items}
        for m in models:
            rows = [r for r in runner.latest_rows(m.slug, task.name) if r["item_id"] in keep and r["sample"] < (args.k or 10**9)]
            if not rows:
                continue
            res = metrics.task_metrics(task, [i for i in items if i.id in {r["item_id"] for r in rows}], rows)
            cell = by_leaf.setdefault(task.leaf, {}).setdefault(m.model_id, {k: [] for k in METRICS})
            for key in METRICS:
                if res.get(key) is not None:
                    cell[key].append(res[key])

    def agg(key: str, values: list[float]) -> float:
        return float(np.median(values)) if key == "latency_p50_ms" else float(np.mean(values))

    per_leaf = {leaf: {n: {k: agg(k, v) for k, v in cell.items() if v} for n, cell in d.items()} for leaf, d in by_leaf.items()}
    print(f"{'leaf':<36}" + "".join(f"{n.split('/')[-1][:18]:>20}" for n in names))
    for leaf in sorted(per_leaf):
        cells = [f"{per_leaf[leaf][n]['accuracy']:.3f}" if n in per_leaf[leaf] else "-" for n in names]
        print(f"{leaf.split('/', 1)[-1]:<36}" + "".join(f"{c:>20}" for c in cells))
    print()
    summary = {}
    for n in names:
        leaves = [d[n] for d in per_leaf.values() if n in d]
        if not leaves:
            continue
        summary[n] = {k: agg(k, [l[k] for l in leaves if k in l]) for k in METRICS if any(k in l for l in leaves)}
        summary[n]["leaves"] = len(leaves)
        s = summary[n]
        extra = "".join(f" {k}={s[k]:.3f}" for k in ("ndcg", "pair_accuracy") if k in s)
        print(f"{n:<32} acc={s['accuracy']:.3f} maj={s.get('maj', float('nan')):.3f} brier={s['brier']:.3f} ece={s['ece']:.3f} "
              f"p50={s.get('latency_p50_ms', float('nan')):.0f}ms cost/1k=${s.get('cost_per_1k_usd', float('nan')):.3f}{extra}  ({len(leaves)} leaves)")
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / args.out).write_text(json.dumps({"per_leaf": per_leaf, "overall": summary}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SUMMARY / args.out}")


def cmd_envs(args: argparse.Namespace) -> None:
    for cls in envs_under(args.env):
        print(f"{cls.name:<52} {cls.mode:<8} episodes {cls.episodes:<3} max decisions {cls.max_decisions:<5} {cls.source}")


def cmd_play(args: argparse.Namespace) -> None:
    """Play the benchmark episodes of closed-loop environments (cached per model, environment, seed, mode)."""
    from .envs import runner as env_runner

    load_env()
    model = make_model(args.model)
    for cls in envs_under(args.env):
        mode = "realtime" if args.realtime or cls.mode == "realtime" else "paused"
        latency = args.latency_ms if args.latency_ms is not None else env_runner.model_latency_ms(model) if mode == "realtime" else 0.0
        seeds = list(range(args.episodes or cls.episodes))
        played = env_runner.run_env(model, cls, seeds, mode, latency, workers=args.workers, max_decisions=args.max_decisions or None)
        print(f"{cls.name}: {played} episodes played ({mode}" + (f", {latency:.0f} ms per decision" if mode == "realtime" else "") + ")")


def cmd_env_score(args: argparse.Namespace) -> None:
    from .envs import runner as env_runner

    table = []
    for spec in args.models:
        model = make_model(spec)
        for cls in envs_under(args.env):
            seeds = set(range(args.episodes or cls.episodes))
            for mode in ("paused", "realtime"):
                cap = args.max_decisions or cls.max_decisions  # smoke runs with a lower cap are scored separately
                latency = None
                if mode == "realtime":  # the model's fixed latency unless another one is asked for
                    try:
                        latency = args.latency_ms if args.latency_ms is not None else env_runner.model_latency_ms(model)
                    except ValueError:  # no static runs to fix a latency from: accept any realtime episodes
                        latency = None
                rows = [r for r in env_runner.latest_episodes(model.slug, cls.name, mode, cap, latency, cls.version) if r["seed"] in seeds]
                if rows:
                    table.append({"model": model.model_id, "env": cls.name, "leaf": cls.leaf, "mode": mode, "score_name": cls.score_name, **env_runner.summarize(rows)})
    for row in table:
        success = f"success={row['success_rate']:.2f}" if row.get("success_rate") is not None else "success=-"
        score = f"{row['score_name']}={row['score_mean']:.2f}" + (f"±{row['score_se']:.2f}" if row.get("score_se") else "")
        extra = f"p50={row['latency_p50_ms']:.0f}ms" if row.get("latency_p50_ms") else ""
        print(f"{row['model']:<28} {row['env']:<52} {row['mode']:<8} n={row['episodes']:<3} {success}  {score}  decisions={row['decisions_mean']:.0f}  failed={row['failed_decision_rate'] or 0:.1%}  {extra}")
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / args.out).write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {SUMMARY / args.out}")


def cmd_env_examples(args: argparse.Namespace) -> None:
    """The first decision request of seed 0 for each environment, in the same shape as task examples."""
    out = {}
    classes = envs_under(args.env)
    if args.per_leaf:  # a spread of at most N environments per leaf, e.g. for the site
        by_leaf: dict[str, list] = {}
        for cls in classes:
            by_leaf.setdefault(cls.leaf, []).append(cls)
        classes = [group[round(i * (len(group) - 1) / max(1, args.per_leaf - 1))] for group in by_leaf.values() for i in range(min(args.per_leaf, len(group)))]
        classes = list(dict.fromkeys(classes))
    for cls in classes:
        env = cls()
        try:
            env.reset(0)
            request = env.observe()
        except Exception as error:  # e.g. an environment whose simulator server is not running here
            print(f"skipped {cls.name}: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
            continue
        finally:
            env.close()
        meta = {"source": cls.source, "license": cls.license, "tier": cls.tier, "closed_loop": True}
        out[cls.name] = {"card": env.card(), "item": {"id": cls.name, "leaf": cls.leaf, "task": cls.name, "request": json.loads(request.model_dump_json()), "gold": {}, "meta": meta}}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {len(out)} environment examples to {args.out}")
    else:
        sys.stdout.write(text + "\n")


def cmd_env_cards(args: argparse.Namespace) -> None:
    """Markdown table of the closed-loop environments, for docs/."""
    lines = [f"# Closed-loop environments under `{args.env or 'all'}`", "",
             "Generated by `s1arena env-cards`. Mode: paused (the world waits) or realtime (it keeps moving while the model decides). "
             "Scores are raw; success is shown where the task defines it.", "",
             "| Environment | Source | License | Tier | Mode | Episodes | Max decisions | Score | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for cls in envs_under(args.env):
        mode = cls.mode + (f" ({cls.control_dt:.3g} s/step)" if cls.mode == "realtime" and cls.control_dt else "")
        cells = [f"`{cls.name}`", f"[{cls.source}]({cls.url})" if cls.url else cls.source, cls.license, cls.tier, mode, str(cls.episodes), str(cls.max_decisions), cls.score_name, cls.notes]
        lines.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in cells) + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run every model of a plan on its tasks and environments; cached, so it resumes where it stopped."""
    from . import sweep

    plan = sweep.load_plan(args.plan)
    if args.models:
        plan.models = [m for m in plan.models if m.spec in args.models]
    if args.envs is not None:
        plan.envs = args.envs
    for m in plan.models:
        m.static = m.static and args.only != "envs"
        m.envs = m.envs and args.only != "static"
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit(f"--shard {args.shard}: need 0 <= index < count")
        plan.shard = (index, count)
    sweep.run(plan, parallel_models=args.parallel_models)
    sweep.print_status(table := sweep.status(plan))
    if plan.shard == (0, 1):  # shards would overwrite each other's file; run sweep-status once they are done
        print(f"wrote {sweep.write_status(plan, table)}")


def cmd_sweep_docker(args: argparse.Namespace) -> None:
    from . import docker_sweep, sweep

    if args.list_images:
        plan = sweep.load_plan(args.plan)
        for cls in docker_sweep.docker_envs(plan):
            for seed in plan.seeds(cls):
                print(f"{cls.name}\t{seed}\t{cls.image(seed)}")
        return
    docker_sweep.run(args.plan, args.models or None)
    plan = sweep.load_plan(args.plan)
    sweep.print_status(table := sweep.status(plan))
    print(f"wrote {sweep.write_status(plan, table)}")


def cmd_sweep_status(args: argparse.Namespace) -> None:
    from . import sweep

    plan = sweep.load_plan(args.plan)
    sweep.print_status(table := sweep.status(plan))
    print(f"wrote {sweep.write_status(plan, table)}")


def cmd_sweep_report(args: argparse.Namespace) -> None:
    from . import sweep

    plan = sweep.load_plan(args.plan)
    rep = sweep.report(plan)
    sys.stdout.write(sweep.report_markdown(rep))
    print("wrote " + ", ".join(map(str, sweep.write_report(plan, rep))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="s1arena", description="S1-Arena: typed decisions with probabilities, for System-1 models and LLMs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tasks", help="list registered tasks")
    p.add_argument("--leaf", default="")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("build", help="build items from source datasets")
    p.add_argument("--leaf", default="")
    p.add_argument("--n", type=int, default=500, help="items per source dataset")
    p.add_argument("--per-leaf", type=int, default=0, help="items per leaf, split evenly across its sources (overrides --n)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local", action="store_true", help="only the sources that do not allow redistribution (into data_local/)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("download", help="fetch the built requests from the Hugging Face dataset into data/")
    p.add_argument("--repo", default=HF_DATASET, help="Hugging Face dataset id (default: $S1ARENA_HF_DATASET or %(default)s)")
    p.add_argument("--domains", nargs="*", default=[], help="only these domains (dataset configs), e.g. basic_cls know")
    p.add_argument("--revision", default=None, help="a dataset revision (branch, tag or commit)")
    p.add_argument("--restricted-repo", default=HF_DATASET_RESTRICTED,
                   help="private dataset with the sources that do not allow redistribution, read into data_local/ if the "
                        "token has access ('' to skip; default: $S1ARENA_HF_DATASET_RESTRICTED or %(default)s)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("check-task", help="validate a few items of one task and show a rendered request")
    p.add_argument("name", help="task name, e.g. basic_cls/intent/clinc150")
    p.add_argument("--n", type=int, default=5, help="items to check")
    p.set_defaults(func=cmd_check_task)

    p = sub.add_parser("check-model", help="run a model on three toy requests and check its distributions")
    p.add_argument("spec", help="model spec, e.g. random, jev, llm:openai/gpt-4.1-mini")
    p.set_defaults(func=cmd_check_model)

    p = sub.add_parser("run", help="query a model on built items")
    p.add_argument("--model", required=True, help="e.g. jev, llm:openai/gpt-4.1-mini, random")
    p.add_argument("--leaf", default="")
    p.add_argument("--k", type=int, default=1, help="samples per item")
    p.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("reparse", help="re-read stored replies of failed calls with the current parser (no new calls)")
    p.add_argument("--model", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    p.set_defaults(func=cmd_reparse)

    p = sub.add_parser("envs", help="list closed-loop environments")
    p.add_argument("--env", default="")
    p.set_defaults(func=cmd_envs)

    p = sub.add_parser("play", help="play closed-loop episodes with a model")
    p.add_argument("--model", required=True)
    p.add_argument("--env", default="")
    p.add_argument("--episodes", type=int, default=0, help="seeds 0..N-1 (0 = the environment's default)")
    p.add_argument("--max-decisions", type=int, default=0, help="cap decisions per episode, e.g. for smoke tests (0 = default)")
    p.add_argument("--realtime", action="store_true", help="run paused environments in realtime mode too")
    p.add_argument("--latency-ms", type=float, default=None, help="fixed latency per decision in realtime mode (default: the model's median)")
    p.add_argument("--workers", type=int, default=4, help="episodes played in parallel")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("env-score", help="summarize cached episodes")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--env", default="")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--max-decisions", type=int, default=0)
    p.add_argument("--latency-ms", type=float, default=None, help="realtime episodes played at this latency (default: the model's fixed latency)")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env_score)

    p = sub.add_parser("env-examples", help="export each environment's first decision request")
    p.add_argument("--env", default="")
    p.add_argument("--per-leaf", type=int, default=0, help="at most N environments per leaf, spread over the list (0 = all)")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env_examples)

    p = sub.add_parser("env-cards", help="write a markdown table of the closed-loop environments")
    p.add_argument("--env", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env_cards)

    p = sub.add_parser("score", help="score cached runs")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--limit", type=int, default=0, help="score only the first N items per task (0 = all)")
    p.add_argument("--out", default="", help="file name under results/summary/")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("examples", help="export one example per task")
    p.add_argument("--leaf", default="")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=1200, help="prefer items whose input is at most this long")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_examples)

    p = sub.add_parser("cards", help="write a markdown table of task cards")
    p.add_argument("--leaf", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_cards)

    p = sub.add_parser("estimate", help="estimate the cost of a run before making any calls")
    p.add_argument("--model", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("compare", help="per-leaf comparison of several models on the same items")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    p.add_argument("--k", type=int, default=0, help="use only the first k samples (0 = all)")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="run all models of a plan on all its tasks and environments")
    p.add_argument("plan", help="TOML plan, e.g. sweeps/full.toml")
    p.add_argument("--models", nargs="*", default=[], help="only these model specs from the plan")
    p.add_argument("--parallel-models", type=int, default=0, help="models run at once (0 = all)")
    p.add_argument("--shard", default="", help="I/N: this worker takes every N-th task and environment from the I-th (0-based)")
    p.add_argument("--only", choices=["static", "envs"], help="only the static tasks or only the environments")
    p.add_argument("--envs", nargs="*", help="environment prefixes instead of the plan's (e.g. game/ for the BALROG image)")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("sweep-status", help="coverage of a plan per model and domain (no calls)")
    p.add_argument("plan")
    p.set_defaults(func=cmd_sweep_status)

    p = sub.add_parser("sweep-report", help="result tables of a plan (markdown + JSON in results/summary/, no calls)")
    p.add_argument("plan")
    p.set_defaults(func=cmd_sweep_report)

    p = sub.add_parser("sweep-docker", help="play a plan's Docker environments (one image per seed) for all its models")
    p.add_argument("plan", help="TOML plan; optional [docker] table: parallel_seeds, prefetch, keep_images, min_free_gb")
    p.add_argument("--models", nargs="*", default=[], help="only these model specs from the plan")
    p.add_argument("--list-images", action="store_true", help="only print the images the plan needs")
    p.set_defaults(func=cmd_sweep_docker)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
