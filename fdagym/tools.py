"""The five agent tools. Arguments stay constrained so every call maps to a query in fdagym.queries."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from fdagym import queries
from fdagym.approvals import find_applications, parse_application_number
from fdagym.queries import Query
from fdagym.snapshot import Snapshot, SnapshotMiss

PAGE_SIZE = 10
SECTION_LIMIT = 1500
# Rollout clips the observation JSON at this many characters. Approval pages stay inside it.
APPROVAL_VISIBLE_CHARS = 3500
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 +-]{0,79}$")
_SECTION = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SKIP_KEYS = {"set_id", "id", "effective_time", "version", "openfda", "brand_names"}
_SUBMISSION_KEYS = (
    "submission_type",
    "submission_number",
    "submission_status",
    "submission_status_date",
    "submission_class_code_description",
)

_ANSWER = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "answer.schema.json").read_text())


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": required}


def _function(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


_NAME_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 80, "pattern": _NAME.pattern}
_SECTION_SCHEMA = {"type": "string", "pattern": _SECTION.pattern}

TOOLS: list[dict] = [
    _function(
        "search_drug_label",
        "List drug-label summaries for one generic name (set_id, brand, manufacturer, effective_time, sections). "
        f"At most {PAGE_SIZE} rows per page. has_section keeps labels that contain that section. "
        "Section text is not included; call get_label_section for it.",
        _object(
            {
                "generic_name": _NAME_SCHEMA,
                "has_section": _SECTION_SCHEMA,
                "page": {"type": "integer", "minimum": 1},
            },
            ["generic_name"],
        ),
    ),
    _function(
        "get_label_section",
        "Return one section of a label whose set_id was already returned by search_drug_label in this episode. "
        f"Text longer than {SECTION_LIMIT} characters is cut, and the output says so.",
        _object({"set_id": {"type": "string", "minLength": 1}, "section": _SECTION_SCHEMA}, ["set_id", "section"]),
    ),
    _function(
        "query_adverse_events",
        "FAERS reports for one generic name. mode top_reactions returns the top MedDRA terms; "
        "mode count returns how many reports match. serious may only be true (omit it for all reports). "
        f"year, when set, must be from {min(queries.FAERS_YEARS)} through {max(queries.FAERS_YEARS)}.",
        _object(
            {
                "generic_name": _NAME_SCHEMA,
                "mode": {"type": "string", "enum": ["top_reactions", "count"]},
                "serious": {"type": "boolean", "enum": [True]},
                "year": {"type": "integer"},
            },
            ["generic_name", "mode"],
        ),
    ),
    _function(
        "search_approvals",
        "Drugs@FDA applications. Pass exactly one of generic_name, application_number, or sponsor_name. "
        "An application number (NDA020977, NDA 020977, 020977, or BLA/ANDA) fetches that application. "
        "sponsor_name, copied from it, lists the snapshot drugs that share that sponsor. "
        "An application result includes sponsor_name, brand, products, and the original approval. "
        "A sponsor result includes generic_name and application_numbers. "
        "page starts at 1. An unknown application number returns an empty list.",
        _object(
            {
                "generic_name": _NAME_SCHEMA,
                "application_number": {"type": "string", "minLength": 4, "maxLength": 24},
                "sponsor_name": {"type": "string", "minLength": 1, "maxLength": 120},
                "page": {"type": "integer", "minimum": 1},
            },
            [],
        ),
    ),
    _function(
        "submit_answer",
        "Finish the episode. answer is one string, number, or boolean, or the string NOT_FOUND. "
        "citations quote a field you were shown. At most 8 citations.",
        _object(
            {
                "answer": _ANSWER["properties"]["answer"],
                "citations": {"type": "array", "maxItems": 8, "items": _ANSWER["$defs"]["citation"]},
            },
            ["answer", "citations"],
        ),
    ),
]

_VALIDATORS = {tool["function"]["name"]: Draft202012Validator(tool["function"]["parameters"]) for tool in TOOLS}


@dataclass
class EpisodeState:
    snapshot: Snapshot
    labels: dict[str, dict] = field(default_factory=dict)


def execute(state: EpisodeState, name: str, arguments: dict, *, step_index: int) -> tuple[dict, str | None]:
    if name not in _VALIDATORS:
        return _fail(f"unknown tool {name}")
    error = _validate(name, arguments)
    if error:
        return _fail(f"invalid arguments: {error}")
    if name == "search_drug_label":
        return _search_labels(state, arguments, step_index)
    if name == "get_label_section":
        return _label_section(state, arguments)
    if name == "query_adverse_events":
        return _adverse_events(state, arguments, step_index)
    if name == "search_approvals":
        return _approvals(state, arguments, step_index)
    return _fail("submit_answer is handled by the environment")


def faers_modes() -> tuple[str, ...]:
    tool = next(item for item in TOOLS if item["function"]["name"] == "query_adverse_events")
    return tuple(tool["function"]["parameters"]["properties"]["mode"]["enum"])


def _faers_mode_hint(exc: ValidationError) -> str | None:
    """Name the legal modes. Other argument errors keep the schema message."""
    modes = ", ".join(faers_modes())
    if exc.validator == "required" and exc.message == "'mode' is a required property":
        return f"mode is required; valid values are {modes}"
    if exc.validator == "enum" and list(exc.path) == ["mode"]:
        return f"mode must be one of {modes}"
    return None


def _validate(name: str, arguments: object) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    try:
        _VALIDATORS[name].validate(arguments)
    except ValidationError as exc:
        if name == "query_adverse_events":
            hint = _faers_mode_hint(exc)
            if hint:
                return hint
        return exc.message
    if name == "query_adverse_events" and "year" in arguments and arguments["year"] not in queries.FAERS_YEARS:
        return f"year must be in {min(queries.FAERS_YEARS)}-{max(queries.FAERS_YEARS)}"
    if name == "search_approvals":
        chosen = [
            key
            for key in ("generic_name", "application_number", "sponsor_name")
            if isinstance(arguments.get(key), str) and arguments[key].strip()
        ]
        if len(chosen) != 1:
            return "pass exactly one of generic_name, application_number, or sponsor_name"
        if "application_number" in chosen and parse_application_number(arguments["application_number"]) is None:
            return "application_number must look like NDA020977, NDA 020977, 020977, or a BLA/ANDA number"
    return None


def _search_labels(state: EpisodeState, arguments: dict, step_index: int) -> tuple[dict, str | None]:
    query = queries.label_search(arguments["generic_name"])
    body = _read(state, query)
    if isinstance(body, str):
        return _fail(body)
    records = [row for row in body.get("results") or [] if isinstance(row, dict)]
    section = arguments.get("has_section")
    if section:
        records = [row for row in records if _section_text(row, section) is not None]
    page = int(arguments.get("page", 1))
    start = (page - 1) * PAGE_SIZE
    shown = records[start : start + PAGE_SIZE]
    for record in shown:
        set_id = record.get("set_id")
        if isinstance(set_id, str):
            state.labels[set_id] = record
    total = len(records) if section else _api_total(body)
    if not isinstance(total, int):
        total = len(records)
    results = [_label_summary(record) for record in shown]
    return _aggregate(step_index, results, total, page=page), None


def _label_section(state: EpisodeState, arguments: dict) -> tuple[dict, str | None]:
    set_id = arguments["set_id"]
    section = arguments["section"]
    record = state.labels.get(set_id)
    if record is None:
        return _fail(f"set_id {set_id} was not returned by search_drug_label in this episode")
    text = _section_text(record, section)
    if text is None:
        return _fail(f"section {section} is not on label {set_id}")
    output: dict = {"set_id": set_id, section: text}
    if len(text) > SECTION_LIMIT:
        output[section] = text[:SECTION_LIMIT]
        output["truncated"] = True
        output["note"] = f"Section text cut at {SECTION_LIMIT} characters."
    return output, None


def _adverse_events(state: EpisodeState, arguments: dict, step_index: int) -> tuple[dict, str | None]:
    name = arguments["generic_name"]
    serious = True if arguments.get("serious") is True else None
    year = arguments.get("year")
    if arguments["mode"] == "top_reactions":
        query = queries.faers_top_reactions(name, serious=serious, year=year)
    else:
        query = queries.faers_report_count(name, serious=serious, year=year)
    body = _read(state, query)
    if isinstance(body, str):
        return _fail(body)
    if arguments["mode"] == "count":
        total = _api_total(body)
        if total is None:
            return _fail("count response has no meta.results.total")
        return _aggregate(step_index, [], total), None
    rows = []
    for row in body.get("results") or []:
        if isinstance(row, dict) and "term" in row:
            rows.append({"term": row["term"], "count": row["count"]})
    total = _api_total(body)
    return _aggregate(step_index, rows, len(rows) if total is None else total), None


def _approvals(state: EpisodeState, arguments: dict, step_index: int) -> tuple[dict, str | None]:
    sponsor = arguments.get("sponsor_name")
    if isinstance(sponsor, str) and sponsor.strip():
        from fdagym.approvals import sponsor_drugs

        rows = sponsor_drugs(state.snapshot, sponsor)
        return _aggregate(step_index, rows, len(rows)), None
    if "application_number" in arguments:
        found = find_applications(state.snapshot, arguments["application_number"])
        if found is None:
            return _fail("application_number must look like NDA020977, NDA 020977, 020977, or a BLA/ANDA number")
        records = found
    else:
        query = queries.approvals_search(arguments["generic_name"])
        body = _read(state, query)
        if isinstance(body, str):
            return _fail(body)
        records = [record for record in body.get("results") or [] if isinstance(record, dict)]
    total = len(records)
    compact = [_compact_approval(record) for record in _sort_approvals(records)]
    pages = _approval_pages(compact, total)
    page = int(arguments.get("page", 1))
    shown = pages[page - 1] if 1 <= page <= len(pages) else []
    output = _aggregate(step_index, shown, total, page=page, page_size=len(shown))
    output["meta"]["results"]["n_pages"] = len(pages)
    return output, None


def _sort_approvals(records: list[dict]) -> list[dict]:
    """NDA, then BLA, then ANDA. Within a kind, the earliest original approval comes first."""
    return sorted(records, key=lambda record: (_kind_rank(record), _orig_sort_date(record), _app_number(record)))


def _kind_rank(record: dict) -> int:
    number = _app_number(record)
    if number.startswith("NDA"):
        return 0
    if number.startswith("BLA"):
        return 1
    if number.startswith("ANDA"):
        return 2
    return 3


def _orig_sort_date(record: dict) -> str:
    dates = []
    for sub in record.get("submissions") or []:
        if not isinstance(sub, dict) or str(sub.get("submission_type", "")).upper() != "ORIG":
            continue
        if str(sub.get("submission_status", "")).upper() not in {"AP", "APPROVED"}:
            continue
        date = sub.get("submission_status_date")
        if isinstance(date, str) and date:
            dates.append(date)
    return min(dates) if dates else "99999999"


def _app_number(record: dict) -> str:
    number = record.get("application_number")
    return number.upper() if isinstance(number, str) else ""


def _compact_approval(record: dict) -> dict:
    """One application line: citeable sponsor, products, and original submissions only."""
    originals = []
    supplements = 0
    for sub in record.get("submissions") or []:
        if not isinstance(sub, dict):
            continue
        if str(sub.get("submission_type", "")).upper() == "ORIG":
            originals.append({key: sub[key] for key in _SUBMISSION_KEYS if key in sub})
        else:
            supplements += 1
    return {
        "application_number": record.get("application_number"),
        "sponsor_name": record.get("sponsor_name"),
        "brand": _application_brands(record),
        "products": _compact_products(record),
        "submissions": originals,
        "n_supplements": supplements,
    }


def _compact_products(record: dict) -> list[dict]:
    """Drop strength duplicates. Brand, dosage form, and route stay so those answers can be cited."""
    seen: list[str] = []
    rows = []
    for product in record.get("products") or []:
        if not isinstance(product, dict):
            continue
        names: list[str] = []
        for ingredient in product.get("active_ingredients") or []:
            name = ingredient.get("name") if isinstance(ingredient, dict) else None
            if isinstance(name, str) and name and name not in names:
                names.append(name)
        row = {key: product[key] for key in ("brand_name", "dosage_form", "route") if key in product}
        if names:
            row["active_ingredients"] = [{"name": name} for name in names]
        if not row:
            continue
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.append(key)
        rows.append(row)
    return rows


def _approval_pages(rows: list[dict], total: int) -> list[list[dict]]:
    """Pack compact rows so the observation the model sees stays within 3,500 characters."""
    fitted = [_fit_approval_row(row, total) for row in rows]
    pages: list[list[dict]] = []
    current: list[dict] = []
    for row in fitted:
        page_no = len(pages) + 1
        trial = current + [row]
        if current and _approval_visible_len(trial, total, page_no) > APPROVAL_VISIBLE_CHARS:
            pages.append(current)
            current = [row]
        else:
            current = trial
    if current or not pages:
        pages.append(current)
    return pages


def _fit_approval_row(row: dict, total: int) -> dict:
    """A single application has to fit on a page even when it has many products.

    The fit is measured as page 999 so a later page number cannot push the line over the clip.
    """
    if _approval_visible_len([row], total, 999) <= APPROVAL_VISIBLE_CHARS:
        return row
    slim = dict(row)
    products = []
    for product in slim.get("products") or []:
        kept = {key: product[key] for key in ("brand_name", "dosage_form", "route") if key in product}
        if kept:
            products.append(kept)
    best = products[:1]
    lo, hi = 1, len(products)
    while lo <= hi:
        mid = max((lo + hi) // 2, 1)
        slim["products"] = products[:mid]
        if _approval_visible_len([slim], total, 999) <= APPROVAL_VISIBLE_CHARS:
            best = products[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    slim["products"] = best
    if _approval_visible_len([slim], total, 999) > APPROVAL_VISIBLE_CHARS:
        slim["brand"] = list(slim.get("brand") or [])[:1]
        slim["submissions"] = list(slim.get("submissions") or [])[:1]
    return slim


def _approval_visible_len(results: list, total: int, page: int) -> int:
    """Upper bound on the observation JSON the rollout clips at 3,500 characters.

    ``n_pages`` is padded to three digits so the packed page stays inside the clip
    once the real page count is written.
    """
    output = _aggregate(1, results, total, page=page, page_size=len(results))
    output["meta"]["results"]["n_pages"] = 999
    body = {"tool": "search_approvals", "output": output, "error": None, "tool_calls_remaining": 5}
    return len(json.dumps(body, ensure_ascii=False))


def _aggregate(
    step_index: int, results: list, total: int, *, page: int | None = None, page_size: int | None = None
) -> dict:
    meta: dict = {"total": total}
    if page is not None:
        meta["page"] = page
        meta["page_size"] = PAGE_SIZE if page_size is None else page_size
    return {"id": f"step:{step_index}", "results": results, "meta": {"results": meta}}


def _read(state: EpisodeState, query: Query) -> dict | str:
    try:
        body = state.snapshot.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)
    except SnapshotMiss:
        return f"no snapshot data for {query.endpoint} {query.search}"
    if not isinstance(body, dict):
        return f"snapshot returned {type(body).__name__} for {query.endpoint}"
    return body


def _fail(message: str) -> tuple[dict, str]:
    return {"error": message}, message


def _api_total(body: dict) -> int | None:
    meta = body.get("meta")
    if not isinstance(meta, dict):
        return None
    results = meta.get("results")
    if not isinstance(results, dict):
        return None
    total = results.get("total")
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    return total


def _label_summary(record: dict) -> dict:
    openfda = record.get("openfda") if isinstance(record.get("openfda"), dict) else {}
    return {
        "set_id": record.get("set_id"),
        "brand": _first(openfda.get("brand_name")),
        "manufacturer": _first(openfda.get("manufacturer_name")),
        "effective_time": record.get("effective_time"),
        "sections": [key for key in record if key not in _SKIP_KEYS and _section_text(record, key) is not None],
    }


def _section_text(record: dict, section: str) -> str | None:
    value = record.get(section)
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        text = "\n".join(value).strip()
        return text or None
    return None


def _application_brands(record: dict) -> list[str]:
    brands = _names(record.get("brand_names"))
    for product in record.get("products") or []:
        if isinstance(product, dict):
            brands.extend(_names(product.get("brand_name")))
    openfda = record.get("openfda") if isinstance(record.get("openfda"), dict) else {}
    brands.extend(_names(openfda.get("brand_name")))
    seen: list[str] = []
    for brand in brands:
        if brand not in seen:
            seen.append(brand)
    return seen


def _names(value: object) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str) and item]
    return []


def _first(value: object) -> str | None:
    names = _names(value)
    return names[0] if names else None
