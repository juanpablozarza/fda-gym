import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


def _load(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


REGISTRY = Registry().with_resources(
    (s["$id"], Resource.from_contents(s))
    for s in (
        _load(n)
        for n in (
            "answer.schema.json",
            "task.schema.json",
            "trajectory.schema.json",
            "teacher_trajectory.schema.json",
        )
    )
)


def validator(name: str) -> Draft202012Validator:
    schema = _load(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=REGISTRY)


@pytest.mark.parametrize(
    ("schema", "example"),
    [
        ("task.schema.json", "task.example.json"),
        ("trajectory.schema.json", "trajectory.example.json"),
        ("teacher_trajectory.schema.json", "teacher_trajectory.example.json"),
    ],
)
def test_example_validates(schema, example):
    validator(schema).validate(_load(f"examples/{example}"))


def test_answer_rejects_list_style_answers():
    v = validator("answer.schema.json")
    assert not v.is_valid({"answer": ["NAUSEA", "DIARRHOEA"], "citations": []})


def test_answer_rejects_unknown_endpoint():
    v = validator("answer.schema.json")
    bad = {"answer": "x", "citations": [{"endpoint": "drug/enforcement", "id": "1", "field": "f", "quote": "q"}]}
    assert not v.is_valid(bad)
