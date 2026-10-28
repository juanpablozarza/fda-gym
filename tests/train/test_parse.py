from fdagym.train.parse import parse_answer


def test_raw_json():
    parsed = parse_answer('{"answer": "NAUSEA", "citations": []}')
    assert parsed == {"answer": "NAUSEA", "citations": []}


def test_fenced_json_with_prose():
    text = """Here you go
```json
{"answer": true, "citations": [{"endpoint": "drug/label", "id": "abc", "field": "boxed_warning", "quote": "bleeding"}]}
```
thanks
"""
    parsed = parse_answer(text)
    assert parsed["answer"] is True
    assert parsed["citations"][0]["id"] == "abc"


def test_chat_completion_and_junk():
    parsed = parse_answer([{"role": "assistant", "content": '{"answer": "NOT_FOUND", "citations": []}'}])
    assert parsed["answer"] == "NOT_FOUND"
    assert parse_answer("I think it causes bleeding") is None
    assert parse_answer("") is None
