"""Format reward: 1 when submit_answer validates against the answer schema, else 0."""

from jsonschema import Draft202012Validator

from fdagym.schemas import load as load_schema

_SCHEMA = load_schema("answer.schema.json")
answer_validator = Draft202012Validator(_SCHEMA)


def score_format(final_answer: object) -> tuple[float, list[str]]:
    if answer_validator.is_valid(final_answer):
        return 1.0, ["format: answer matches the schema"]
    return 0.0, ["format: submit_answer is missing or does not match the schema"]
