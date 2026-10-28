"""Keep only the fields tasks, tools, and citations use. A raw label is often 50 to 200 KB.

A spec maps field name to `True` (keep as is) or to a nested spec. Lists are trimmed element by element.
Count responses ({"term", "count"} rows) are small and pass through untouched.
"""

from typing import Any

from fdagym.snapshot import Endpoint

Spec = dict[str, "bool | Spec"]

LABEL_SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
    "dosage_and_administration",
)

SPECS: dict[Endpoint, Spec] = {
    "drug/label": {
        "id": True,
        "set_id": True,
        "version": True,
        "effective_time": True,
        "openfda": {
            "generic_name": True,
            "brand_name": True,
            "manufacturer_name": True,
            "application_number": True,
            "product_type": True,
        },
        **{section: True for section in LABEL_SECTIONS},
    },
    "drug/event": {
        "safetyreportid": True,
        "receivedate": True,
        "serious": True,
        "seriousnessdeath": True,
        "patient": {
            "reaction": {"reactionmeddrapt": True, "reactionoutcome": True},
            "drug": {
                "medicinalproduct": True,
                "drugcharacterization": True,
                "openfda": {"generic_name": True, "brand_name": True},
            },
        },
    },
    "drug/drugsfda": {
        "application_number": True,
        "sponsor_name": True,
        "openfda": {"generic_name": True, "brand_name": True, "manufacturer_name": True},
        "submissions": {
            "submission_type": True,
            "submission_number": True,
            "submission_status": True,
            "submission_status_date": True,
            "submission_class_code": True,
            "submission_class_code_description": True,
            "review_priority": True,
        },
        "products": {
            "product_number": True,
            "brand_name": True,
            "active_ingredients": True,
            "dosage_form": True,
            "route": True,
            "marketing_status": True,
        },
    },
}


def pick(value: Any, spec: Spec) -> Any:
    if isinstance(value, list):
        return [pick(item, spec) for item in value]
    if not isinstance(value, dict):
        return value
    kept = {}
    for name, sub in spec.items():
        if name in value:
            kept[name] = value[name] if sub is True else pick(value[name], sub)
    return kept


def trim_response(endpoint: Endpoint, body: dict, *, is_count: bool) -> dict:
    if is_count:
        return body
    return {**body, "results": pick(body.get("results", []), SPECS[endpoint])}
