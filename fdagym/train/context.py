"""Run oracle_calls through FDAEnv so citation ids exist in this episode's steps."""

import json
import re
from random import Random

from fdagym.env import FDAEnv
from fdagym.snapshot import Snapshot
from fdagym.train.prompt import render_messages

MAX_PACK_CHARS = 6000
_APP_NUMBER = re.compile(r"\b(?:NDA|ANDA|BLA)\d+\b", re.IGNORECASE)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)


def play_oracle(env: FDAEnv, task: dict) -> list[dict]:
    """Reset, execute every oracle tool call, leave the env ready for submit_answer."""
    env.reset(task["id"])
    packs = []
    for call in task["oracle_calls"]:
        obs, _reward, done, _info = env.step({"tool": call["tool"], "arguments": call["arguments"]})
        if done:
            raise RuntimeError(f"{task['id']} oracle ended the episode on {call['tool']}")
        if obs["error"]:
            raise RuntimeError(f"{task['id']} {call['tool']}: {obs['error']}")
        packs.append(
            {
                "tool": call["tool"],
                "arguments": call["arguments"],
                "output": obs["output"],
                "step_index": len(env.steps),
                "task_id": task["id"],
            }
        )
    return packs


def pack_oracle(snapshot: Snapshot, tasks: list[dict], task: dict, *, max_chars: int = MAX_PACK_CHARS) -> list[dict]:
    env = FDAEnv(snapshot, tasks, run_id="pack")
    return [_clip(pack, max_chars, task) for pack in play_oracle(env, task)]


def pack_distractors(
    snapshot: Snapshot,
    tasks: list[dict],
    task: dict,
    *,
    k: int = 2,
    max_chars: int = MAX_PACK_CHARS,
    rng=None,
) -> list[dict]:
    """Documents from other drugs, packed in a throwaway env so they never enter the scoring episode."""
    drug = (task.get("entity") or {}).get("generic_name")
    others = [
        other
        for other in tasks
        if other["id"] != task["id"]
        and other.get("type") != "label_open"
        and (other.get("entity") or {}).get("generic_name") != drug
    ]
    if rng is not None:
        # Sort first so the same pool yields the same decoys regardless of file order.
        others = sorted(others, key=lambda other: other["id"])
        rng.shuffle(others)
    packs = []
    for other in others:
        if len(packs) >= k:
            break
        try:
            docs = pack_oracle(snapshot, tasks, other, max_chars=max_chars)
        except RuntimeError:
            continue
        if docs:
            packs.append(docs[0])
    return packs


def assemble_documents(
    snapshot: Snapshot,
    tasks: list[dict],
    task: dict,
    *,
    n_distractors: int = 2,
    max_chars: int = MAX_PACK_CHARS,
    seed: int = 0,
) -> list[dict]:
    """Oracle docs plus decoys. Seeded by task id so train and eval render the same prompt."""
    rng = Random(f"{seed}:{task['id']}")
    packs = pack_oracle(snapshot, tasks, task, max_chars=max_chars)
    extra = pack_distractors(
        snapshot,
        tasks,
        task,
        k=n_distractors,
        max_chars=max_chars,
        rng=rng,
    )
    docs = list(packs) + extra
    rng.shuffle(docs)
    return docs


def render_task_prompt(
    snapshot: Snapshot,
    tasks: list[dict],
    task: dict,
    *,
    n_distractors: int = 2,
    max_chars: int = MAX_PACK_CHARS,
    seed: int = 0,
) -> list[dict]:
    """The single prompt both GRPO rows and Plan A eval send to the model."""
    return render_messages(
        task,
        assemble_documents(
            snapshot,
            tasks,
            task,
            n_distractors=n_distractors,
            max_chars=max_chars,
            seed=seed,
        ),
    )


def _clip(pack: dict, max_chars: int, task: dict) -> dict:
    """Keep whole result records. Rank decides who survives a cut; order stays original."""
    output = pack["output"]
    if not isinstance(output, dict):
        return _clip_raw(pack, max_chars)
    fitted = _fit_output(output, max_chars, task)
    if fitted is output:
        return pack
    clipped = dict(pack)
    clipped["output"] = fitted
    return clipped


def _fit_output(output: dict, max_chars: int, task: dict) -> dict:
    results = output.get("results")
    if not isinstance(results, list):
        if len(_dumps(output)) <= max_chars:
            return output
        return _shrink_record(output, max_chars, task)
    if len(_dumps(output)) <= max_chars:
        return output
    ids, drugs = _needles(task)
    base = {key: value for key, value in output.items() if key != "results"}
    ranked = sorted(range(len(results)), key=lambda index: _rank_key(results[index], ids, drugs, orig=False, index=index))
    chosen: list[int] = []
    replacements: dict[int, object] = {}
    saw_id_match = False
    for index in ranked:
        trial_ids = chosen + [index]
        trial = _wrapped(
            base,
            [replacements.get(i, results[i]) for i in sorted(trial_ids)],
            omitted=len(results) - len(trial_ids),
        )
        if len(_dumps(trial)) <= max_chars:
            chosen.append(index)
            if _matches_id(results[index], ids):
                saw_id_match = True
            continue
        needs_shrink = not chosen or (_matches_id(results[index], ids) and not saw_id_match)
        if needs_shrink:
            fitted = _shrink_into(
                base,
                [replacements.get(i, results[i]) for i in sorted(chosen)],
                results[index],
                len(results),
                max_chars,
                task,
            )
            if fitted is not None:
                replacements[index] = fitted
                chosen.append(index)
                saw_id_match = saw_id_match or _matches_id(results[index], ids)
    if not chosen and results:
        best = ranked[0]
        stub = _shrink_record(results[best], max(32, max_chars // 2), task)
        return _wrapped(base, [stub], omitted=len(results) - 1, truncated=True)
    chosen.sort()
    kept = [replacements.get(index, results[index]) for index in chosen]
    return _wrapped(base, kept, omitted=len(results) - len(chosen), truncated=True)


def _shrink_into(base: dict, kept: list, record, n_results: int, max_chars: int, task: dict):
    """Shrink one record until the wrapped document fits. None if even a stub does not."""
    omitted = n_results - len(kept) - 1
    shrunk = record
    for _ in range(6):
        trial = _wrapped(base, kept + [shrunk], omitted=omitted, truncated=True)
        overflow = len(_dumps(trial)) - max_chars
        if overflow <= 0:
            return shrunk
        target = len(_dumps(shrunk)) - overflow - 32
        if target < 24:
            break
        nxt = _shrink_record(record, target, task)
        if len(_dumps(nxt)) >= len(_dumps(shrunk)):
            break
        shrunk = nxt
    trial = _wrapped(base, kept + [shrunk], omitted=omitted, truncated=True)
    if len(_dumps(trial)) <= max_chars:
        return shrunk
    return None


def _shrink_record(record, budget: int, task: dict):
    """Drop list items at element boundaries. Survivors keep their original order."""
    if not isinstance(record, dict):
        text = _dumps(record)
        if len(text) <= budget:
            return record
        return {"truncated": True, "preview": text[: max(0, budget - 40)]}
    if len(_dumps(record)) <= budget:
        return record
    ids, drugs = _needles(task)
    scalars = {key: value for key, value in record.items() if not isinstance(value, list)}
    lists = {key: value for key, value in record.items() if isinstance(value, list)}
    candidates = []
    for key, items in lists.items():
        for index, item in enumerate(items):
            orig = isinstance(item, dict) and item.get("submission_type") == "ORIG"
            candidates.append((key, index, item, _rank_key(item, ids, drugs, orig=orig, index=index)))
    candidates.sort(key=lambda row: row[3])
    selected = {key: [] for key in lists}

    def materialize():
        body = dict(scalars)
        for key, items in lists.items():
            body[key] = [items[index] for index in sorted(selected[key])]
        return body

    if len(_dumps(materialize())) > budget:
        return _shrink_scalars(scalars, budget)
    for key, index, _item, _rank in candidates:
        selected[key].append(index)
        if len(_dumps(materialize())) > budget:
            selected[key].pop()
    return materialize()


def _shrink_scalars(scalars: dict, budget: int) -> dict:
    kept = {}
    for key, value in scalars.items():
        trial = dict(kept)
        trial[key] = value
        if len(_dumps(trial)) <= budget:
            kept[key] = value
    if kept:
        return kept
    # Last resort: the shortest scalar, trimmed, so an application number can still survive.
    if not scalars:
        return {"truncated": True}
    key, value = min(scalars.items(), key=lambda item: len(_dumps(item[1])))
    text = value if isinstance(value, str) else _dumps(value)
    room = max(0, budget - len(key) - 16)
    return {key: text[:room], "truncated": True}


def _wrapped(base: dict, results: list, *, omitted: int, truncated: bool = False) -> dict:
    out = dict(base)
    out["results"] = results
    if omitted > 0 or truncated:
        out["truncated"] = True
    if omitted > 0:
        out["omitted"] = omitted
    return out


def _needles(task: dict) -> tuple[set[str], set[str]]:
    """Ids and names that appear in the question. Gold fields are not readable here."""
    question = str(task.get("question") or "")
    ids = {match.lower() for match in _APP_NUMBER.findall(question)}
    ids.update(match.lower() for match in _UUID.findall(question))
    drugs: set[str] = set()
    entity = task.get("entity") or {}
    if isinstance(entity, dict):
        folded = question.lower()
        for key in ("generic_name", "brand_name"):
            value = entity.get(key)
            if isinstance(value, str) and _mentioned(folded, value):
                drugs.add(value.strip().lower())
    return ids, drugs


def _mentioned(question_lower: str, name: str) -> bool:
    cleaned = name.strip().lower()
    if len(cleaned) < 3:
        return False
    return _drug_in(question_lower, cleaned)


def _matches_id(item, ids: set[str]) -> bool:
    if not ids:
        return False
    blob = _dumps(item).lower()
    return any(cid in blob for cid in ids)


def _rank_key(item, ids: set[str], drugs: set[str], *, orig: bool, index: int) -> tuple:
    """Question needles first. ORIG only breaks ties. The last key is original position."""
    blob = _dumps(item).lower()
    id_hit = 0 if any(cid in blob for cid in ids) else 1
    drug_hits = sum(1 for drug in drugs if _drug_in(blob, drug))
    return (id_hit, -drug_hits, 0 if orig else 1, index)


def _drug_in(blob: str, drug: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(drug)}(?![a-z0-9])", blob) is not None


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _clip_raw(pack: dict, max_chars: int) -> dict:
    text = _dumps(pack["output"])
    if len(text) <= max_chars:
        return pack
    clipped = dict(pack)
    clipped["output"] = {"truncated": True, "preview": text[: max(0, max_chars - 40)]}
    return clipped
