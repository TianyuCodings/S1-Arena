import pytest

from s1arena.registry import ENVS, MODELS, TASKS, discover, envs_under, make_model, tasks_under


def test_discovery_finds_everything():
    discover()
    assert len(TASKS) > 200 and len(ENVS) > 100
    assert {"random", "llm", "jev", "lm"} <= set(MODELS)
    domains = {name.split("/")[0] for name in TASKS}
    assert {"basic_cls", "know", "judge", "safety", "step"} <= domains


def test_every_task_card_is_complete():
    discover()
    for name, cls in TASKS.items():
        card = cls().card()
        assert card["leaf"] and name.startswith(card["leaf"] + "/"), name
        assert card["source"] and card["license"] and card["tier"] in ("A", "B", "C"), name


def test_prefix_selection():
    assert all(t.name.startswith("basic_cls/intent/") for t in tasks_under("basic_cls/intent"))
    assert all(c.name.startswith("realtime/") for c in envs_under("realtime"))


def test_make_model():
    assert make_model("random").model_id == "test/random"
    with pytest.raises(KeyError):
        make_model("no-such-kind:x")
