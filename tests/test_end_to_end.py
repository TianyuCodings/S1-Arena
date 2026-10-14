"""A toy task, a toy environment and the random and template models, end to end without network."""

import json

from s1arena import checks, dataio, hub, metrics, runner
from s1arena.envs import runner as env_runner
from s1arena.registry import make_model


def test_task_template_build_run_score(template):
    module = template("task_template")
    task = module.MyDataset()
    items = task.build(n=6, seed=0)
    assert len(items) == 6 and len({i.id for i in items}) == 6
    assert items == task.build(n=6, seed=0)  # building is deterministic
    dataio.write_items(dataio.task_path(task.name), items)
    assert dataio.load_task(task) == items

    model = make_model("random")
    assert runner.run_task(model, task.name, items, k=2, workers=2) == 12
    assert runner.run_task(model, task.name, items, k=2, workers=2) == 0  # cached
    rows = runner.latest_rows(model.slug, task.name)
    result = metrics.task_metrics(task, items, rows)
    assert result["n_questions"] == 6 and result["samples"] == 2
    assert 0.0 <= result["accuracy"] <= 1.0 and 0.0 <= result["brier"] <= 2.0


def test_check_task_on_the_template(template):
    template("task_template")
    task, items, problems = checks.check_task("basic_cls/example/my_dataset")
    assert problems == [] and len(items) == 5


def test_check_model_on_random_and_template(template):
    template("model_template")
    for spec in ("random", "example", "example:other"):
        rows = checks.check_model(spec)
        assert [r["kind"] for r in rows] == ["choice", "boolean", "score"]
        assert all(r["problems"] == [] for r in rows), rows


def test_env_template_plays_an_episode(template):
    module = template("env_template")
    env = module.NumberLine()
    episode = env_runner.play_episode(make_model("random"), env, seed=0, mode="paused", latency_ms=0.0)
    assert episode["decisions"] <= env.max_decisions
    assert set(episode["outcome"]) >= {"success", "score"}


def test_hub_rows_round_trip(template):
    task = template("task_template").MyDataset()
    for item in task.build(n=3):
        row = hub.item_to_row(item, task.card())
        assert set(row) == set(hub.COLUMNS) and all(isinstance(v, str) for v in row.values())
        assert json.loads(row["meta"])["url"] == task.url
        back = hub.row_to_item(row)
        assert back.id == item.id and back.request == item.request and back.gold == item.gold
