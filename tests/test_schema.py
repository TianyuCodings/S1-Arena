import pytest
from pydantic import ValidationError

from s1arena.schema import DecisionRequest, Gold, Item, Question, normalize


def choice(options=("a", "b", "c")):
    return Question(type="choice", instructions="Pick one.", criteria={o: f"option {o}" for o in options})


def item(question, gold, name="q"):
    return Item(leaf="basic_cls/test", task="basic_cls/test/toy", request=DecisionRequest(state={"text": "x"}, questions={name: question}),
                gold={name: gold})


def test_question_types_and_option_ids():
    assert choice().option_ids() == ["a", "b", "c"]
    assert Question(type="boolean", instructions="True?").option_ids() == ["true", "false"]
    assert Question(type="score", instructions="How much?", criteria=["low", "mid", "high"]).option_ids() == ["0", "1", "2"]


@pytest.mark.parametrize("kwargs", [
    {"type": "choice", "criteria": {"only": "one option"}},  # a choice needs at least 2 options
    {"type": "choice", "criteria": ["a", "b"]},  # options are a dict
    {"type": "score", "criteria": [str(i) for i in range(11)]},  # at most 10 levels
    {"type": "boolean", "criteria": {"true": "t", "false": "f"}},  # a boolean takes no criteria
])
def test_invalid_questions(kwargs):
    with pytest.raises(ValidationError):
        Question(instructions="x", **kwargs)


def test_gold_rules():
    with pytest.raises(ValidationError):
        Gold()
    with pytest.raises(ValidationError):
        Gold(dist={"a": 0.5, "b": 0.4})
    with pytest.raises(ValidationError):
        item(choice(), Gold(label="z"))  # not an option
    assert Gold(dist={"a": 0.2, "b": 0.8}).hard() == "b"
    assert Gold(label="a").target(["a", "b"]) == {"a": 1.0, "b": 0.0}


def test_content_id_follows_content():
    first = item(choice(), Gold(label="a"))
    assert first.id == item(choice(), Gold(label="a")).id
    assert first.id.startswith("basic_cls.test.toy:")
    assert first.id != item(choice(("c", "b", "a")), Gold(label="a")).id  # another option order is another request


def test_normalize():
    assert normalize({"a": 2, "b": 2, "x": 5}, ["a", "b"]) == {"a": 0.5, "b": 0.5}
    assert normalize({}, ["a", "b", "c", "d"]) == {o: 0.25 for o in "abcd"}
