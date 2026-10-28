"""Boxed and composite trajectories must not depend on the answer."""

import copy
import json
from pathlib import Path

import pytest

from fdagym.queries import label_search
from fdagym.train.boxed_oracle import LABEL_TOOLS, evidence_oracle, question_scope, tool_actions
from fdagym.train.rollout_dry import scripted_actions
from fdagym.train.sft_tools import teacher_forced_tool_match
from tests.env.support import EXAMPLE_TASK, MapSnapshot, task_nausea

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "snapshots" / "2026-10-15" / "manifest.jsonl"


def _label(set_id: str, generic, *, warning: str | None, product_type: str) -> dict:
    record = {
        "set_id": set_id,
        "openfda": {
            "generic_name": generic if isinstance(generic, list) else [generic],
            "brand_name": [set_id],
            "product_type": [product_type],
        },
        "indications_and_usage": ["used for the labeled indication"],
    }
    if warning:
        record["boxed_warning"] = [warning]
    return record


def _snapshot(*records: dict) -> MapSnapshot:
    query = label_search("metformin")
    body = {"meta": {"results": {"total": len(records)}}, "results": list(records)}
    return MapSnapshot({(query.endpoint, query.search, query.count, query.limit): body})


def _boxed(task_id: str, value: bool, oracle_calls: list[dict], *, question: str | None = None) -> dict:
    return {
        "id": task_id,
        "type": "boxed_warning",
        "question": question or "Does metformin's label carry a boxed warning?",
        "sources": ["drug/label"],
        "entity": {"generic_name": "metformin"},
        "split": "train",
        "answerable": True,
        "difficulty": "easy",
        "difficulty_features": {
            "sources": 1,
            "min_tool_calls": len(oracle_calls),
            "needs_disambiguation": False,
            "needs_aggregation": False,
            "needs_normalization": False,
            "has_constraint": False,
        },
        "gold": {"kind": "boolean", "value": value, **({"keywords": ["lactic"]} if value else {})},
        "gold_citations": [
            {
                "endpoint": "drug/label",
                "ids": ["single"],
                "fields": ["boxed_warning" if value else "sections"],
            }
        ],
        "oracle_calls": oracle_calls,
        "snapshot_version": "test",
    }


def _search(has_section: bool) -> dict:
    arguments = {"generic_name": "metformin"}
    if has_section:
        arguments["has_section"] = "boxed_warning"
    return {"tool": "search_drug_label", "arguments": arguments}


def _section() -> dict:
    return {"tool": "get_label_section", "arguments": {"set_id": "single", "section": "boxed_warning"}}


def _tools(actions: list[dict]) -> list[dict]:
    return [action for action in actions if action["tool"] != "submit_answer"]


def test_yes_and_no_variants_share_the_tool_calls_up_to_the_evidence():
    """Stored oracle_calls leak the answer. The script must ignore that and the flipped answer."""
    present = _label("single", "metformin", warning="WARNING: LACTIC ACIDOSIS", product_type="HUMAN PRESCRIPTION DRUG")
    combo = _label(
        "combo",
        ["metformin", "sitagliptin"],
        warning="WARNING: LACTIC ACIDOSIS",
        product_type="HUMAN PRESCRIPTION DRUG",
    )
    snapshot = _snapshot(combo, present)
    yes = _boxed("yes", True, [_search(True), _section()])
    no = _boxed("no", False, [_search(False)])
    assert yes["oracle_calls"] != no["oracle_calls"]
    assert "has_section" in yes["oracle_calls"][0]["arguments"]
    yes_actions, _env = tool_actions(snapshot, [yes], yes)
    no_actions, _env = tool_actions(snapshot, [no], no)
    assert yes_actions == no_actions
    assert yes_actions[0]["arguments"] == {"generic_name": "metformin"}
    assert "has_section" not in yes_actions[0]["arguments"]
    assert [action["tool"] for action in yes_actions] == ["search_drug_label", "get_label_section"]
    assert yes_actions[1]["arguments"]["set_id"] == "single"


def test_a_combination_on_the_page_does_not_count_as_the_drugs_warning():
    single = _label("single", "metformin", warning=None, product_type="HUMAN PRESCRIPTION DRUG")
    combo = _label(
        "combo",
        ["metformin", "sitagliptin"],
        warning="WARNING: LACTIC ACIDOSIS",
        product_type="HUMAN PRESCRIPTION DRUG",
    )
    snapshot = _snapshot(combo, single)
    yes = _boxed("yes", True, [_search(True), _section()])
    no = _boxed("no", False, [_search(False)])
    yes_actions, _env = tool_actions(snapshot, [yes], yes)
    no_actions, _env = tool_actions(snapshot, [no], no)
    assert yes_actions == no_actions
    assert [action["tool"] for action in yes_actions] == ["search_drug_label"]


def test_product_scope_in_the_question_selects_the_label():
    rx = _label("rx", "metformin", warning=None, product_type="HUMAN PRESCRIPTION DRUG")
    otc = _label("otc", "metformin", warning="WARNING: LACTIC ACIDOSIS", product_type="HUMAN OTC DRUG")
    snapshot = _snapshot(otc, rx)
    question = "Does the prescription FDA label for metformin carry a boxed warning?"
    task = _boxed("rx", False, [_search(False)], question=question)
    actions, _env = tool_actions(snapshot, [task], task)
    assert question_scope(question) == "HUMAN PRESCRIPTION DRUG"
    assert [action["tool"] for action in actions] == ["search_drug_label"]
    flipped = copy.deepcopy(task)
    flipped["id"] = "rx-yes"
    flipped["gold"]["value"] = True
    flipped["oracle_calls"] = [
        _search(True),
        {"tool": "get_label_section", "arguments": {"set_id": "otc", "section": "boxed_warning"}},
    ]
    flipped_actions, _env = tool_actions(snapshot, [flipped], flipped)
    assert _tools(flipped_actions) == _tools(actions)


def test_decision_module_does_not_read_the_answer():
    source = (ROOT / "fdagym" / "train" / "boxed_oracle.py").read_text().lower()
    assert "gold" not in source


def test_other_task_types_still_replay_oracle_calls():
    from tests.env.support import JsonSnapshot

    task = task_nausea()
    assert not evidence_oracle(task)
    actions = scripted_actions(JsonSnapshot(), [task], task)
    assert _tools(actions) == task["oracle_calls"]
    assert actions[-1]["tool"] == "submit_answer"


def test_train_boxed_and_composite_scripts_ignore_stored_label_calls():
    if not SNAPSHOT.exists():
        pytest.skip("pilot snapshot absent")
    from fdagym.agents.baseline import load_split
    from fdagym.openfda import DiskSnapshot

    snapshot = DiskSnapshot("2026-10-15")
    tasks = [task for task in load_split("train") if task["type"] in {"boxed_warning", "composite"}]
    assert tasks
    leaked = [
        task for task in tasks if any("has_section" in call.get("arguments", {}) for call in task["oracle_calls"])
    ]
    plain = [task for task in tasks if task not in leaked]
    assert leaked and plain
    for task in tasks:
        actions, _env = tool_actions(snapshot, [task], task)
        searches = [action for action in actions if action["tool"] == "search_drug_label"]
        assert searches
        assert all("has_section" not in action["arguments"] for action in searches)
        mutated = _opposite_script(task)
        mutated_actions, _env = tool_actions(snapshot, [mutated], mutated)
        assert mutated_actions == actions


def _opposite_script(task: dict) -> dict:
    """Flip the stored answer and the stored label calls. An evidence script must not care."""
    mutated = copy.deepcopy(task)
    mutated["id"] = task["id"] + "-mut"
    value = mutated["gold"].get("value")
    if isinstance(value, bool):
        mutated["gold"]["value"] = not value
    elif isinstance(value, str) and value.endswith("yes"):
        mutated["gold"]["value"] = value[:-3] + "no"
    elif isinstance(value, str) and value.endswith("no"):
        mutated["gold"]["value"] = value[:-2] + "yes"
    calls = []
    had_filter = any("has_section" in call.get("arguments", {}) for call in task["oracle_calls"])
    for call in task["oracle_calls"]:
        if call["tool"] not in LABEL_TOOLS:
            calls.append(copy.deepcopy(call))
    search = {"tool": "search_drug_label", "arguments": {"generic_name": task["entity"]["generic_name"]}}
    if not had_filter:
        search["arguments"]["has_section"] = "boxed_warning"
        calls.append(search)
        calls.append(
            {"tool": "get_label_section", "arguments": {"set_id": "not-the-evidence", "section": "boxed_warning"}}
        )
    else:
        calls.append(search)
    mutated["oracle_calls"] = calls
    return mutated


def test_teacher_forced_match_is_the_first_supervised_span():
    labels = [-100, -100, 5, 6, 7, -100, 8, 9]
    preds = [0, 5, 6, 7, 0, 8, 1]
    match = teacher_forced_tool_match(preds, labels)
    assert match == {"first_tool_exact": True, "tool_seq_exact": False}
    preds[1] = 4
    assert teacher_forced_tool_match(preds, labels)["first_tool_exact"] is False
    assert teacher_forced_tool_match([], [-100, -100]) == {"first_tool_exact": False, "tool_seq_exact": False}


def test_fixture_boxed_task_still_scores_one_without_the_section_filter():
    from fdagym.train.rollout_dry import CharTokenizer
    from fdagym.train.sft_tools import build_sft_rows
    from tests.env.support import ALL_TASKS, JsonSnapshot

    built = build_sft_rows(JsonSnapshot(), ALL_TASKS, ["FG-0001"], CharTokenizer(), max_length=8192)
    assert built["n_kept"] == 1
    rollout = built["rollouts"][0]
    assert rollout.reward["total"] == 1.0
    model = "".join(span.text for span in rollout.spans if span.role == "model")
    assert "has_section" not in model
    assert "get_label_section" in model
    assert EXAMPLE_TASK["oracle_calls"][0]["arguments"]["has_section"] == "boxed_warning"
    stored = json.dumps(EXAMPLE_TASK["oracle_calls"])
    assert "has_section" in stored
