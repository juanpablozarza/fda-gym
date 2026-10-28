"""Single-turn prompt: question plus retrieved documents. No tool-calling loop."""

import json

SYSTEM = """\
You answer FDA-Gym questions using only the documents below. Some documents may be irrelevant.
Cite only text that appears in a document from this prompt.

Write a single JSON object and nothing else:
{"answer": <string|number|boolean>, "citations": [{"endpoint": "drug/label"|"drug/event"|"drug/drugsfda", "id": "<id or step:n>", "field": "<path>", "quote": "<verbatim span>"}]}

Rules:
- id is a set_id, application_number, or the "id" field on a document (often "step:<n>" for FAERS aggregates).
- quote is a verbatim span from that field, not a paraphrase. Quotes shorter than 4 characters never count.
- MedDRA reaction terms are uppercase with British spelling; copy them.
- NOT_FOUND is only correct when a document shows an empty search, and you must cite that search.
- Do not call tools. Do not wrap the JSON in markdown unless you also include the raw object.
"""


def render_documents(packs: list[dict]) -> str:
    blocks = []
    for index, pack in enumerate(packs, 1):
        payload = json.dumps(
            {"tool": pack["tool"], "arguments": pack["arguments"], "output": pack["output"]},
            ensure_ascii=False,
            default=str,
        )
        blocks.append(f"### Document {index}\n{payload}")
    return "\n\n".join(blocks) if blocks else "(no documents)"


def render_user(task: dict, packs: list[dict]) -> str:
    return f"Question:\n{task['question']}\n\nDocuments:\n{render_documents(packs)}"


def render_messages(task: dict, packs: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render_user(task, packs)},
    ]
