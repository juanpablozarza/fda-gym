"""Salt-aware product names for hop credit. Strict eval does not use this matcher."""

from pathlib import Path

import pytest

from fdagym.drugnames import names_match, product_matches
from fdagym.rewards import score
from fdagym.rewards.combine import WEIGHTS
from fdagym.rewards.hops import HOP_CAP, hop_score
from fdagym.tasks.generators import _label_is_single, _product_is_single
from fdagym.tasks.hard_solver import question_earliest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15" / "manifest.jsonl"

SALT_PAIRS = [
    ("divalproex", "DIVALPROEX SODIUM"),
    ("metoprolol", "metoprolol succinate"),
    ("metoprolol", "METOPROLOL-SUCCINATE"),
    ("metoprolol", "metoprolol HCl"),
    ("metoprolol", "metoprolol hydrochloride"),
    ("metoprolol", "metoprolol hydrochloride monohydrate"),
    ("metoprolol", "metoprolol tartrate"),
    ("metoprolol", "metoprolol (succinate)"),
    ("warfarin", "Warfarin Sodium"),
    ("quetiapine", "QUETIAPINE FUMARATE"),
    ("lithium", "lithium carbonate"),
    ("tofacitinib", "TOFACITINIB CITRATE"),
    ("paroxetine", "paroxetine mesylate"),
    ("amlodipine", "amlodipine besylate"),
    ("promethazine", "promethazine maleate"),
    ("clindamycin", "clindamycin phosphate"),
    ("megestrol", "megestrol acetate"),
    ("dextromethorphan", "dextromethorphan hydrobromide"),
    ("morphine", "morphine sulfate"),
    ("fingolimod", "fingolimod lauryl sulfate"),
    ("oxycodone", "oxycodone terephthalate"),
    ("exenatide", "EXENATIDE SYNTHETIC"),
    ("insulin glargine", "INSULIN GLARGINE RECOMBINANT"),
    ("cetirizine", "cetirizine hcl"),
    ("natalizumab", "NATALIZUMAB-SZTN"),
    ("adalimumab", "adalimumab-atto"),
    ("metformin", "METFORMIN"),
    ("metformin", "  Metformin Hydrochloride  "),
]


@pytest.mark.parametrize("generic,raw", SALT_PAIRS)
def test_salt_ester_and_biosimilar_suffixes_match(generic, raw):
    assert names_match(raw, generic)
    assert product_matches({"active_ingredients": [{"name": raw}]}, generic)
    assert _product_is_single is product_matches
    assert _label_is_single({"openfda": {"generic_name": [raw]}}, generic)


FALSE_PAIRS = [
    ("metformin", "metformin and sitagliptin"),
    ("metformin", "metformin / sitagliptin"),
    ("metformin", "METFORMIN HYDROCHLORIDE AND SITAGLIPTIN PHOSPHATE"),
    ("metformin", "metformin-sitagliptin"),
    ("metformin", "metformin with sitagliptin"),
    ("amlodipine", "amlodipine besylate valsartan hydrochlorothiazide"),
    ("diclofenac", "diclofenac sodium diphenhydramine hcl"),
    ("insulin", "insulin glargine"),
    ("insulin", "insulin lispro"),
    ("insulin", "insulin aspart"),
    ("insulin", "insulin detemir"),
    ("insulin", "insulin degludec"),
    ("insulin", "insulin glulisine"),
    ("insulin", "INSULIN GLARGINE RECOMBINANT"),
    ("insulin", "insulin glargine-yfgn"),
    ("met", "metformin"),
    ("met", "metformin hydrochloride"),
    ("ace", "acetaminophen"),
    ("tin", "tofacitinib citrate"),
    ("ox", "oxycodone hydrochloride"),
    ("sodium", "divalproex sodium"),
    ("divalproex", ""),
    ("", "divalproex sodium"),
]


@pytest.mark.parametrize("generic,raw", FALSE_PAIRS)
def test_partial_match_rejects_combos_analogs_and_short_substrings(generic, raw):
    assert not names_match(raw, generic)
    assert not product_matches({"active_ingredients": [{"name": raw}]}, generic)


def test_two_ingredients_are_a_combination_even_when_one_is_the_salt():
    product = {"active_ingredients": [{"name": "METFORMIN HYDROCHLORIDE"}, {"name": "SITAGLIPTIN PHOSPHATE"}]}
    assert not product_matches(product, "metformin")
    assert not product_matches(product, "sitagliptin")
    blank = {"active_ingredients": [{"name": "METFORMIN HYDROCHLORIDE"}, {"name": ""}]}
    assert not product_matches(blank, "metformin")


def _approval(drug, ingredient, *, year="19830310", app="NDA018723"):
    return {
        "tool": "search_approvals",
        "arguments": {"generic_name": drug},
        "output": {
            "results": [
                {
                    "application_number": app,
                    "products": [{"active_ingredients": [{"name": ingredient}]}],
                    "submissions": [
                        {
                            "submission_type": "ORIG",
                            "submission_status": "AP",
                            "submission_status_date": year,
                        }
                    ],
                }
            ]
        },
    }


def _earliest_task():
    return {
        "type": "compare",
        "question": question_earliest(["divalproex", "metformin", "lithium"]),
        "difficulty_features": {"min_tool_calls": 3},
    }


def test_salt_form_lookup_earns_the_year_hop_and_strict_score_does_not():
    steps = [
        _approval("divalproex", "DIVALPROEX SODIUM", year="19830310"),
        _approval("metformin", "METFORMIN HYDROCHLORIDE", year="19950303", app="NDA020357"),
        _approval("lithium", "LITHIUM CARBONATE", year="19700101", app="NDA016084"),
    ]
    task = _earliest_task()
    hops = hop_score(steps, task)
    assert hops.applies
    for drug in ("divalproex", "metformin", "lithium"):
        assert f"hop year:{drug}: hit" in hops.notes
    final = {"answer": "nope", "citations": []}
    strict = score({"steps": steps, "final_answer": final}, task)
    dense = score({"steps": steps, "final_answer": final}, task, dense=True)
    assert "hops" not in strict
    assert strict["correctness"] == 0.0
    assert strict["total"] == 0.1
    assert dense["hops"] == hops.rate
    assert dense["correctness"] == pytest.approx(HOP_CAP * hops.rate)
    assert dense["total"] == pytest.approx(WEIGHTS["correctness"] * dense["correctness"] + WEIGHTS["format"])
    assert HOP_CAP == 0.5
    assert WEIGHTS == {"correctness": 0.60, "citation": 0.25, "format": 0.10, "efficiency": 0.05}


def test_insulin_analog_lookup_does_not_earn_the_year_hop():
    steps = [_approval("insulin", "INSULIN GLARGINE", year="20000420", app="BLA021081")]
    task = {
        "type": "compare",
        "question": question_earliest(["insulin", "metformin", "lithium"]),
    }
    hops = hop_score(steps, task)
    assert "hop year:insulin: miss" in hops.notes


def _exact_product(product, wanted):
    names = [
        str(item.get("name") or "").strip().casefold()
        for item in product.get("active_ingredients") or []
        if isinstance(item, dict)
    ]
    return [name for name in names if name] == [wanted.strip().casefold()]


def _year_hits(steps, task) -> int:
    scored = hop_score(steps, task)
    return sum(1 for note in scored.notes if note.startswith("hop year:") and note.endswith(": hit")), scored.rate


def test_hard_train_correct_lookups_gain_the_missed_year_hops():
    """Exact ingredient equality missed salt forms. The shared matcher credits them.

    An earlier pass estimated about 179 of the hard_train questions. Counted
    on the frozen file, with the approval page the reward actually scores, 63
    questions gain a year hop that was a miss, and 63 have a higher hop rate.
    """
    if not SNAPSHOT.exists():
        pytest.skip("pilot snapshot absent")
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot
    from fdagym.rewards import hops as hops_mod
    from fdagym.tasks.hard_solver import parse_question
    from fdagym.tools import EpisodeState, execute

    tasks = load_split("hard_train")
    assert len(tasks) == 319
    snapshot = DiskSnapshot("2026-10-15")
    cache: dict[str, dict] = {}

    def approvals(drug: str) -> dict:
        if drug not in cache:
            output, err = execute(
                EpisodeState(snapshot),
                "search_approvals",
                {"generic_name": drug},
                step_index=1,
            )
            cache[drug] = {
                "tool": "search_approvals",
                "arguments": {"generic_name": drug},
                "output": output,
                "error": err,
            }
        return cache[drug]

    gained_year = []
    gained_rate = []
    lost_year = []
    saved = hops_mod.product_matches
    try:
        for task in tasks:
            spec = parse_question(task["question"])
            if spec["variant"] not in {"earliest_boxed", "earliest_reaction"}:
                continue
            steps = [approvals(drug) for drug in spec["drugs"]]
            hops_mod.product_matches = _exact_product
            before_hits, before_rate = _year_hits(steps, task)
            hops_mod.product_matches = product_matches
            after_hits, after_rate = _year_hits(steps, task)
            if after_hits > before_hits:
                gained_year.append(task["id"])
            if after_hits < before_hits:
                lost_year.append(task["id"])
            if after_rate > before_rate:
                gained_rate.append(task["id"])
    finally:
        hops_mod.product_matches = saved

    assert lost_year == []
    assert len(gained_year) == 63
    assert len(gained_rate) == 63
    assert set(gained_year) <= set(gained_rate)
