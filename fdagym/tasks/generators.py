"""Build tasks from a Snapshot. Every gold value is read out of a recorded response."""

import re
from collections import defaultdict
from collections.abc import Iterable

from fdagym.drugnames import names_match
from fdagym.drugnames import product_matches as _product_is_single
from fdagym.queries import FAERS_YEARS, Query, faers_report_count, faers_top_reactions
from fdagym.snapshot import Snapshot, SnapshotMiss
from fdagym.tasks.config import GenConfig
from fdagym.tasks.difficulty import features, tier
from fdagym.tasks.meddra import aliases
from fdagym.tasks.plan import absence_year, allow_unanswerable, load_universe, planned_queries, probe_year
from fdagym.tasks.split import assign_splits, balance_boxed_yes_no, select_for_shares
from fdagym.tasks.validate import TaskRejected, require_nonempty, require_oracle_calls, unique_top_term
from fdagym.tools import PAGE_SIZE
from fdagym.tools import _section_text as tool_section_text

_STOP = {
    "warning",
    "patients",
    "patient",
    "including",
    "increased",
    "increase",
    "risk",
    "risks",
    "serious",
    "fatal",
    "death",
    "deaths",
    "cause",
    "causes",
    "caused",
    "associated",
    "treatment",
    "treated",
    "prescribing",
    "information",
    "healthcare",
    "provider",
    "providers",
    "should",
    "which",
    "their",
    "there",
    "these",
    "those",
    "about",
    "after",
    "before",
    "during",
    "within",
    "without",
    "other",
    "others",
    "using",
    "monitor",
    "clinical",
    "studies",
    "reported",
    "reactions",
    "reaction",
    "adverse",
    "events",
    "event",
    "label",
    "labels",
    "boxed",
    "warnings",
    "black",
    "product",
    "products",
    "therapy",
    "therapeutic",
    "contraindicated",
    "contraindications",
    "history",
    "known",
    "hypersensitivity",
    "complete",
    "detected",
    "discontinue",
    "sodium",
    "major",
    "drugs",
    "dietary",
    "changes",
    "perform",
    "regular",
    "monitoring",
    "dosage",
    "administration",
    "precautions",
    "interactions",
    "developing",
    "angiotensin",
    "system",
    "injury",
    "possible",
    "directly",
    "rodents",
}

_PRODUCT_SCOPE = {
    "HUMAN PRESCRIPTION DRUG": "prescription",
    "HUMAN OTC DRUG": "over-the-counter",
}


def generate(
    snapshot: Snapshot,
    cfg: GenConfig | None = None,
    drugs: Iterable[str] | None = None,
) -> list[dict]:
    cfg = cfg or GenConfig()
    universe = tuple(drugs) if drugs is not None else load_universe()
    by_drug = {drug: list(_candidates(snapshot, drug, cfg)) for drug in universe}
    by_drug = {drug: tasks for drug, tasks in by_drug.items() if tasks}
    splits = assign_splits(by_drug, cfg)
    tagged = [{**task, "split": splits[drug]} for drug, tasks in by_drug.items() for task in tasks]
    return _assign_ids(select_for_shares(balance_boxed_yes_no(tagged), cfg))


def _assign_ids(tasks: list[dict]) -> list[dict]:
    ordered = sorted(tasks, key=lambda t: (t["split"], t["type"], t["entity"]["generic_name"], t["question"]))
    return [{**task, "id": f"FG-{i:04d}"} for i, task in enumerate(ordered, 1)]


def _candidates(snapshot: Snapshot, drug: str, cfg: GenConfig) -> Iterable[dict]:
    loaded = {}
    for query in planned_queries(drug):
        body = _optional(snapshot, query)
        if body is not None:
            loaded[query] = body
    labels = _results(loaded[planned_queries(drug)[0]]) if planned_queries(drug)[0] in loaded else []
    approvals = _results(loaded[planned_queries(drug)[1]]) if planned_queries(drug)[1] in loaded else []
    year = probe_year(drug)
    gap = absence_year(drug)
    brand = _dominant_brand(labels, drug)
    version = snapshot.version
    seen: set[str] = set()

    def emit(task: dict) -> dict | None:
        if task["question"] in seen:
            return None
        seen.add(task["question"])
        return task

    builders = []
    if approvals:
        builders.append(lambda: _approval(drug, brand, approvals, version, cfg))
    if labels:
        builders.append(lambda: _boxed_tasks(drug, brand, labels, version))
        builders.append(lambda: [_label_open(drug, brand, labels, version)])
    top_all = loaded.get(_query(drug, "top", serious=None, year=None))
    top_year = loaded.get(_query(drug, "top", serious=True, year=year))
    top_gap = loaded.get(_query(drug, "top", serious=True, year=gap))
    count_year = loaded.get(_query(drug, "count", serious=True, year=year))
    if top_all is not None:
        builders.append(lambda: [_faers_top(drug, brand, top_all, version, year=None, serious=False)])
    if top_year is not None:
        builders.append(lambda: [_faers_top(drug, brand, top_year, version, year=year, serious=True)])
    if top_gap is not None:
        builders.append(lambda: [_faers_top(drug, brand, top_gap, version, year=gap, serious=True)])
    if count_year is not None:
        builders.append(lambda: [_faers_count(drug, brand, count_year, version, year=year, serious=True)])
        if labels:
            builders.append(lambda: [_composite(drug, brand, count_year, labels, version, year)])
    for build in builders:
        try:
            produced = build()
        except TaskRejected:
            continue
        for task in produced:
            kept = emit(task)
            if kept is not None:
                yield kept
    # Honest empty windows the snapshot actually recorded, beyond the two planned years.
    for faers_year in FAERS_YEARS:
        for serious in (None, True):
            query = faers_top_reactions(drug, serious=serious, year=faers_year)
            body = _optional(snapshot, query)
            if body is None or _results(body):
                continue
            try:
                kept = emit(_faers_top(drug, brand, body, version, year=faers_year, serious=bool(serious)))
            except TaskRejected:
                continue
            if kept is not None:
                yield kept


def _query(drug: str, kind: str, *, serious: bool | None, year: int | None) -> Query:
    build = faers_top_reactions if kind == "top" else faers_report_count
    return build(drug, serious=serious, year=year)


def _run(snapshot: Snapshot, query: Query) -> dict:
    return snapshot.get(query.endpoint, search=query.search, count=query.count, limit=query.limit)


def _optional(snapshot: Snapshot, query: Query) -> dict | None:
    try:
        return _run(snapshot, query)
    except SnapshotMiss:
        return None


def _results(body: dict) -> list:
    return list(body.get("results") or [])


def _approval(drug: str, brand: str | None, records: list[dict], version: str, cfg: GenConfig) -> list[dict]:
    """Earliest ORIG year for one brand. Comparing several NDAs counts as aggregation."""
    del brand
    by_brand: dict[str, list[tuple[dict, int, str]]] = defaultdict(list)
    for record in records:
        number = str(record.get("application_number", "")).upper()
        if not number.startswith(("NDA", "BLA")):
            continue
        products = [p for p in record.get("products") or [] if _product_is_single(p, drug)]
        if not products:
            continue
        year = _orig_year(record)
        if year is None:
            continue
        for product in products:
            raw = str(product.get("brand_name") or "").strip()
            if not raw or _combo_name(raw):
                continue
            by_brand[raw.lower()].append((record, year, raw))
    usable = []
    for hits in by_brand.values():
        year = min(hit[1] for hit in hits)
        winners = [hit for hit in hits if hit[1] == year]
        apps = {hit[0]["application_number"] for hit in hits}
        usable.append((year, winners[0][2], winners, len(apps) > 1))
    if not usable:
        raise TaskRejected("empty approval")
    year, chosen, winners, across_records = min(usable, key=lambda hit: (hit[0], hit[1].lower()))
    app_ids = []
    for hit in winners:
        app_id = hit[0]["application_number"]
        if app_id not in app_ids:
            app_ids.append(app_id)
    kind = "BLA" if app_ids and all(app.upper().startswith("BLA") for app in app_ids) else "NDA"
    tasks = [
        _make(
            type="approval",
            question=f"In what year did the FDA approve the original {kind} for {chosen} ({drug})?",
            sources=["drug/drugsfda"],
            drug=drug,
            brand=chosen,
            version=version,
            answerable=True,
            gold={"kind": "number", "value": year, "tolerance": 0},
            citations=[
                {
                    "endpoint": "drug/drugsfda",
                    "ids": app_ids,
                    "fields": ["submissions"],
                }
            ],
            oracle_calls=[_approval_call(drug)],
            source_count=1,
            needs_disambiguation=across_records,
            needs_aggregation=across_records or cfg.single_record_earliest_is_aggregation,
            needs_normalization=False,
            has_constraint=False,
        )
    ]
    seen_apps: set[str] = set()
    for record, _year, _name in winners:
        app_id = record["application_number"]
        if app_id in seen_apps:
            continue
        seen_apps.add(app_id)
        tasks.extend(_application_facts(drug, version, record))
    return tasks


def _application_facts(drug: str, version: str, record: dict) -> list[dict]:
    """One named application: sponsor, brand, and dosage form or route. Each is one record."""
    app = str(record["application_number"])
    products = [p for p in record.get("products") or [] if _product_is_single(p, drug)]
    tasks = []
    sponsor = str(record.get("sponsor_name") or "").strip()
    if sponsor:
        tasks.append(_fact(drug, version, app, f"Who is the sponsor of application {app}?", sponsor, "sponsor_name"))
    brands = sorted(
        {
            str(product.get("brand_name") or "").strip()
            for product in products
            if str(product.get("brand_name") or "").strip() and not _combo_name(str(product.get("brand_name")))
        }
    )
    if len(brands) == 1:
        tasks.append(
            _fact(
                drug,
                version,
                app,
                f"What brand name is marketed under application {app}?",
                brands[0],
                "products",
                brand=brands[0],
            )
        )
    forms = {str(product.get("dosage_form") or "").strip() for product in products}
    forms.discard("")
    routes = {str(product.get("route") or "").strip() for product in products}
    routes.discard("")
    if len(forms) == 1:
        form = next(iter(forms))
        tasks.append(
            _fact(
                drug,
                version,
                app,
                f"What is the dosage form of the product under application {app}?",
                form,
                "products",
            )
        )
    elif len(routes) == 1:
        route = next(iter(routes))
        tasks.append(
            _fact(drug, version, app, f"What is the route of the product under application {app}?", route, "products")
        )
    return tasks


def _fact(drug, version, app, question, value, field, brand=None) -> dict:
    return _make(
        type="approval",
        question=question,
        sources=["drug/drugsfda"],
        drug=drug,
        brand=brand,
        version=version,
        answerable=True,
        gold={"kind": "exact", "value": value},
        citations=[{"endpoint": "drug/drugsfda", "ids": [app], "fields": [field]}],
        oracle_calls=[_approval_call(drug)],
        source_count=1,
        needs_disambiguation=False,
        needs_aggregation=False,
        needs_normalization=False,
        has_constraint=False,
    )


def _orig_year(record: dict) -> int | None:
    years = []
    for sub in record.get("submissions") or []:
        if str(sub.get("submission_type", "")).upper() != "ORIG":
            continue
        if str(sub.get("submission_status", "")).upper() not in {"AP", "APPROVED"}:
            continue
        year = _year_of(sub.get("submission_status_date"))
        if year is not None:
            years.append(year)
    return min(years) if years else None


def _boxed_tasks(drug: str, brand: str | None, labels: list[dict], version: str) -> list[dict]:
    singles = [label for label in labels if _label_is_single(label, drug)]
    require_nonempty(singles, "labels")
    groups = _boxed_groups(singles)
    tasks = []
    if len(groups) == 1 and groups[0][0] is None:
        _scope, group, present = groups[0]
        tasks.append(_easy_boxed(drug, brand, version, group, present, labels))
        if not present:
            return tasks
    for scope, group, present in groups:
        try:
            tasks.append(_one_boxed(drug, brand, version, scope, group, present, labels))
        except TaskRejected:
            continue
    if not tasks:
        raise TaskRejected("boxed warning")
    return tasks


def _easy_boxed(drug, brand, version, group, present, labels) -> dict:
    """Yes/no only. Every label agrees, so the agent does not choose among records."""
    ids = _set_ids(group)
    return _make(
        type="boxed_warning",
        question=f"Does {drug}'s label carry a boxed warning?",
        sources=["drug/label"],
        drug=drug,
        brand=brand,
        version=version,
        answerable=True,
        gold={"kind": "boolean", "value": present},
        citations=[{"endpoint": "drug/label", "ids": ids, "fields": ["sections"]}],
        oracle_calls=_label_calls(drug, labels, ids, section=None),
        source_count=1,
        needs_disambiguation=False,
        needs_aggregation=False,
        needs_normalization=False,
        has_constraint=False,
    )


def _one_boxed(drug, brand, version, scope, group, present, labels) -> dict:
    texts = [_section_text(label.get("boxed_warning")) for label in group]
    gold: dict = {"kind": "boolean", "value": present}
    ids = _set_ids(group)
    # A missing section is not on the label. The search summary's `sections` list is the evidence.
    section = "boxed_warning" if present else None
    if present:
        words = _keywords(texts, drug)
        require_nonempty(words, "boxed-warning keywords")
        gold["keywords"] = words
    scope_phrase = _PRODUCT_SCOPE.get(scope or "")
    if scope_phrase:
        question = f"Does the {scope_phrase} FDA label for {drug} carry a boxed warning?"
    else:
        question = f"Does the FDA label for {drug} carry a boxed warning?"
    if present:
        question += " If it does, name the risk."
    return _make(
        type="boxed_warning",
        question=question,
        sources=["drug/label"],
        drug=drug,
        brand=brand,
        version=version,
        answerable=True,
        gold=gold,
        citations=[{"endpoint": "drug/label", "ids": ids, "fields": ["boxed_warning" if present else "sections"]}],
        oracle_calls=_label_calls(drug, labels, ids, section=section),
        source_count=1,
        needs_disambiguation=len(group) > 1,
        needs_aggregation=False,
        needs_normalization=False,
        has_constraint=scope_phrase is not None,
    )


def _boxed_groups(labels: list[dict]) -> list[tuple[str | None, list[dict], bool]]:
    flags = {_has_warning(label) for label in labels}
    if len(flags) == 1:
        return [(None, labels, flags.pop())]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for label in labels:
        by_type[_product_type(label)].append(label)
    groups = []
    for product_type, group in sorted(by_type.items()):
        group_flags = {_has_warning(label) for label in group}
        if len(group_flags) == 1:
            groups.append((product_type, group, group_flags.pop()))
    if not groups:
        raise TaskRejected("multi-value boxed warning")
    return groups


def _faers_top(drug, brand, body, version, *, year: int | None, serious: bool) -> dict:
    results = _results(body)
    constrained = year is not None or serious
    if not results:
        if not allow_unanswerable(drug, year):
            raise TaskRejected("unsafe empty faers window")
        return _make(
            type="faers_top_reaction",
            question=_top_question(drug, year, serious),
            sources=["drug/event"],
            drug=drug,
            brand=brand,
            version=version,
            answerable=False,
            gold={"kind": "exact", "value": "NOT_FOUND"},
            citations=[{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}],
            oracle_calls=[_event_call(drug, "top_reactions", year, serious)],
            source_count=1,
            needs_disambiguation=False,
            needs_aggregation=True,
            needs_normalization=True,
            has_constraint=constrained,
        )
    term = unique_top_term(results)
    gold: dict = {"kind": "exact", "value": term}
    also = aliases(term)
    if also:
        gold["aliases"] = also
    return _make(
        type="faers_top_reaction",
        question=_top_question(drug, year, serious),
        sources=["drug/event"],
        drug=drug,
        brand=brand,
        version=version,
        answerable=True,
        gold=gold,
        citations=[{"endpoint": "drug/event", "ids": [], "fields": ["results"]}],
        oracle_calls=[_event_call(drug, "top_reactions", year, serious)],
        source_count=1,
        needs_disambiguation=False,
        needs_aggregation=True,
        needs_normalization=True,
        has_constraint=constrained,
    )


def _top_question(drug: str, year: int | None, serious: bool) -> str:
    if year is None and not serious:
        return f"What is the most-reported adverse reaction in FAERS for {drug}?"
    if year is not None and serious:
        return f"What is the most-reported adverse reaction in FAERS for {drug} among serious reports received in {year}?"
    if year is not None:
        return f"What is the most-reported adverse reaction in FAERS for {drug} among reports received in {year}?"
    return f"What is the most-reported adverse reaction in FAERS for {drug} among serious reports?"


def _faers_count(drug, brand, body, version, *, year: int | None, serious: bool) -> dict:
    total = _report_total(body)
    if year is not None and serious:
        question = f"How many serious FAERS reports for {drug} were received in {year}?"
    elif year is not None:
        question = f"How many FAERS reports for {drug} were received in {year}?"
    elif serious:
        question = f"How many serious FAERS reports for {drug} are in FAERS?"
    else:
        question = f"How many FAERS reports for {drug} are in FAERS?"
    return _make(
        type="faers_count",
        question=question,
        sources=["drug/event"],
        drug=drug,
        brand=brand,
        version=version,
        answerable=True,
        gold={"kind": "number", "value": total, "tolerance": 0},
        citations=[{"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]}],
        oracle_calls=[_event_call(drug, "count", year, serious)],
        source_count=1,
        needs_disambiguation=False,
        needs_aggregation=True,
        needs_normalization=False,
        has_constraint=year is not None or serious,
    )


def _label_open(drug: str, brand: str | None, labels: list[dict], version: str) -> dict:
    del brand
    singles = [label for label in labels if _label_is_single(label, drug)]
    chosen, group = _best_consistent_brand(singles)
    text = _section_text(group[0].get("contraindications"))
    require_nonempty(text, "contraindications")
    ids = _set_ids(group)
    excerpt = re.sub(r"\s+", " ", text).strip()[:500]
    if "pregnan" in excerpt.lower():
        first = "Mentions pregnancy or teratogenicity as a contraindication named in the label."
    else:
        first = "States a contraindication that appears in the cited contraindications text."
    return _make(
        type="label_open",
        question=f"Summarize the contraindications on the {chosen} label for {drug}.",
        sources=["drug/label"],
        drug=drug,
        brand=chosen,
        version=version,
        answerable=True,
        gold={
            "kind": "judge",
            "value": excerpt,
            "rubric": [
                first,
                "Does not add a contraindication absent from the cited contraindications section.",
                "Cites the contraindications section.",
            ],
        },
        citations=[{"endpoint": "drug/label", "ids": ids, "fields": ["contraindications"]}],
        oracle_calls=_label_calls(drug, labels, ids, section="contraindications"),
        source_count=1,
        needs_disambiguation=len(singles) > 1,
        needs_aggregation=False,
        needs_normalization=False,
        has_constraint=False,
    )


def _best_consistent_brand(labels: list[dict]) -> tuple[str, list[dict]]:
    """A brand whose single-ingredient labels share one contraindications text."""
    groups: dict[str, list[dict]] = defaultdict(list)
    display: dict[str, str] = {}
    for label in labels:
        for name in _brand_names_raw(label):
            if _combo_name(name):
                continue
            key = name.lower()
            display.setdefault(key, name)
            if label not in groups[key]:
                groups[key].append(label)
    best: tuple[tuple, str, list[dict]] | None = None
    for key, group in groups.items():
        clusters = _text_clusters(group)
        if len(clusters) != 1:
            continue
        text = next(iter(clusters))
        if not text:
            continue
        rank = (len(group), len(text), key)
        if best is None or rank > best[0]:
            best = (rank, display[key], group)
    if best is None:
        raise TaskRejected("multi-value contraindications")
    return best[1], best[2]


def _composite(drug, brand, count_body, labels, version, year: int) -> dict:
    singles = [label for label in labels if _label_is_single(label, drug)]
    groups = _boxed_groups(singles)
    unscoped = [group for scope, group, _present in groups if scope is None]
    if not unscoped:
        raise TaskRejected("multi-value boxed warning")
    present = _has_warning(unscoped[0][0])
    total = _report_total(count_body)
    if total == 0:
        raise TaskRejected("empty count")
    answer = "yes" if present else "no"
    ids = _set_ids(unscoped[0])
    section = "boxed_warning" if present else None
    return _make(
        type="composite",
        question=(
            f"How many serious FAERS reports for {drug} were received in {year}, "
            f"and does its FDA label carry a boxed warning? "
            f"Reply with the count, a semicolon, then yes or no."
        ),
        sources=["drug/event", "drug/label"],
        drug=drug,
        brand=brand,
        version=version,
        answerable=True,
        gold={"kind": "exact", "value": f"{total}; {answer}"},
        citations=[
            {"endpoint": "drug/event", "ids": [], "fields": ["meta.results.total"]},
            {"endpoint": "drug/label", "ids": ids, "fields": ["boxed_warning" if present else "sections"]},
        ],
        oracle_calls=[_event_call(drug, "count", year, True), *_label_calls(drug, labels, ids, section=section)],
        source_count=2,
        needs_disambiguation=len(unscoped[0]) > 1,
        needs_aggregation=True,
        needs_normalization=False,
        has_constraint=True,
    )


def _make(
    *,
    type: str,
    question: str,
    sources: list[str],
    drug: str,
    brand: str | None,
    version: str,
    answerable: bool,
    gold: dict,
    citations: list[dict],
    oracle_calls: list[dict],
    source_count: int,
    needs_disambiguation: bool,
    needs_aggregation: bool,
    needs_normalization: bool,
    has_constraint: bool,
) -> dict:
    if "suspect" in question.lower():
        raise TaskRejected("suspect phrasing")
    feats = features(
        sources=source_count,
        min_tool_calls=len(oracle_calls),
        needs_disambiguation=needs_disambiguation,
        needs_aggregation=needs_aggregation,
        needs_normalization=needs_normalization,
        has_constraint=has_constraint,
    )
    entity = {"generic_name": drug}
    if brand and brand.strip().lower() != drug.strip().lower():
        entity["brand_name"] = brand
    task = {
        "type": type,
        "question": question,
        "sources": sources,
        "entity": entity,
        "answerable": answerable,
        "difficulty": tier(feats),
        "difficulty_features": dict(feats),
        "base_pass_rate": None,
        "gold": gold,
        "gold_citations": citations,
        "oracle_calls": oracle_calls,
        "snapshot_version": version,
    }
    require_oracle_calls(task)
    return task


def _approval_call(drug: str) -> dict:
    return {"tool": "search_approvals", "arguments": {"generic_name": drug}}


def _event_call(drug: str, mode: str, year: int | None, serious: bool) -> dict:
    arguments: dict = {"generic_name": drug, "mode": mode}
    if serious:
        arguments["serious"] = True
    if year is not None:
        arguments["year"] = year
    return {"tool": "query_adverse_events", "arguments": arguments}


def _label_calls(drug: str, labels: list[dict], wanted: list[str], *, section: str | None) -> list[dict]:
    """Pages of search_drug_label until a cited set_id is on the page, then the section read when the text is the evidence."""
    wanted_ids = set(wanted)
    visible = [
        label
        for label in labels
        if section is None or tool_section_text(label, section) is not None
    ]
    index = next((i for i, label in enumerate(visible) if label.get("set_id") in wanted_ids), None)
    if index is None:
        raise TaskRejected("cited set_id is not in the label search")
    page = index // PAGE_SIZE + 1
    calls = []
    for number in range(1, page + 1):
        arguments: dict = {"generic_name": drug}
        if section is not None:
            arguments["has_section"] = section
        if number > 1:
            arguments["page"] = number
        calls.append({"tool": "search_drug_label", "arguments": arguments})
    if section is not None:
        calls.append(
            {
                "tool": "get_label_section",
                "arguments": {"set_id": visible[index]["set_id"], "section": section},
            }
        )
    if len(calls) > 6:
        raise TaskRejected("oracle needs more than 6 tool calls")
    return calls


def _text_clusters(labels: list[dict]) -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for label in labels:
        text = re.sub(r"\s+", " ", _section_text(label.get("contraindications"))).strip().lower()
        if text:
            clusters[text].append(label)
    return clusters


def _keywords(texts: list[str], generic: str, limit: int = 3) -> list[str]:
    """Words from the warning headline that every cited label shares."""
    ban = _STOP | {generic.strip().lower()}
    sets = []
    for text in texts:
        head = text[:160].lower()
        words = re.findall(r"[a-z]{5,}", head)
        sets.append({word for word in words if word not in ban})
    if not sets or any(not group for group in sets):
        return []
    common = set.intersection(*sets)
    if not common:
        return []

    def rank(word: str) -> tuple:
        position = min(text.lower().find(word) for text in texts)
        return (position if position >= 0 else 10**6, -len(word), word)

    return sorted(common, key=rank)[:limit]


def _section_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(part) for part in value).strip()
    return str(value).strip()


def _has_warning(label: dict) -> bool:
    return bool(_section_text(label.get("boxed_warning")))


def _product_type(label: dict) -> str:
    raw = (label.get("openfda") or {}).get("product_type")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return str(raw or "UNKNOWN")


def _set_ids(labels: list[dict]) -> list[str]:
    ids = []
    for label in labels:
        set_id = label.get("set_id")
        if set_id and set_id not in ids:
            ids.append(set_id)
    require_nonempty(ids, "set_ids")
    return ids


def _report_total(body: dict) -> int:
    total = (body.get("meta") or {}).get("results", {}).get("total")
    if total is None:
        raise TaskRejected("missing report total")
    return int(total)


def _year_of(value) -> int | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) < 4:
        return None
    year = int(digits[:4])
    if 1900 <= year <= 2100:
        return year
    return None


def _label_is_single(label: dict, generic: str) -> bool:
    names = (label.get("openfda") or {}).get("generic_name") or []
    if isinstance(names, str):
        names = [names]
    return len(names) == 1 and names_match(str(names[0]), generic)


def _brand_names_raw(label: dict) -> list[str]:
    raw = (label.get("openfda") or {}).get("brand_name") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(name).strip() for name in raw if str(name).strip()]


def _combo_name(name: str) -> bool:
    folded = name.lower()
    return any(token in folded for token in ("/", "+", ";", " and ", ","))


def _dominant_brand(labels: list[dict], generic: str) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    display: dict[str, str] = {}
    for label in labels:
        if not _label_is_single(label, generic):
            continue
        raw = (label.get("openfda") or {}).get("brand_name") or []
        if isinstance(raw, str):
            raw = [raw]
        for name in raw:
            text = str(name).strip()
            if not text:
                continue
            key = text.lower()
            counts[key] += 1
            display.setdefault(key, text)
    if not counts:
        return None
    winner = min(counts, key=lambda key: (-counts[key], key))
    return display[winner]
