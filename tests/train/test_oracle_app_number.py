from fdagym.train.rollout_dry import by_application_number


def _task(question, generic="abacavir"):
    return {
        "type": "approval",
        "question": question,
        "entity": {"generic_name": generic},
        "oracle_calls": [{"tool": "search_approvals", "arguments": {"generic_name": generic}}],
    }


def test_number_only_question_searches_by_number():
    task = by_application_number(_task("Who is the sponsor of NDA020977?"))
    assert task["oracle_calls"] == [{"tool": "search_approvals", "arguments": {"application_number": "NDA020977"}}]


def test_question_naming_the_drug_keeps_generic_search():
    original = _task("In what year was the original NDA for ZIAGEN (abacavir) approved?")
    assert by_application_number(original) is original


def test_other_types_unchanged():
    task = {**_task("NDA020977"), "type": "boxed_warning"}
    assert by_application_number(task) is task
