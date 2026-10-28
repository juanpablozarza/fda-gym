"""Weighted reward. Components stay in [0, 1]; total is their weighted sum.

Strict score (the eval baseline, including the 0.0185 Plan A number):

    R = 0.60*correctness + 0.25*citation + 0.10*format + 0.05*efficiency

``dense=True`` keeps those weights. Format stays 0.10 so a schema-only answer
cannot dominate a partial citation (0.25 * 0.50 = 0.125) or a near-miss
correctness. Partial rates live in ``fdagym/rewards/dense.py`` and
``citation.ID_ONLY``. On compare / sponsor_hop, a wrong answer still earns
``HOP_CAP * hop_rate`` of the correctness slot from tool observations
(``fdagym/rewards/hops.py``). Efficiency stays gated on the pre-hop
correctness. Training turns dense on; eval still stores the strict score on
the trajectory and reports the dense score beside it.

JP's rule, dense sponsor-hop only: citation credit (a valid citation or
id-only partial credit) is withheld unless a tool call looked up the
hop-target drug (``hop_target_searched``). Citing the anchor application
after the first hop is not paid.

A ``sponsor_hop`` task pays the answer's correctness, in strict eval and in
dense training, only when the trace proves the sponsor join. The three unnamed
questions need every hop. An older question needs the application, the sponsor
match, and FAERS when that question has it. The boxed-warning keyword is not
required once the match has left one drug. Dense training still shapes a
partial hop into the correctness slot after that gate, and efficiency stays on
the pre-shape value.
"""

from fdagym.rewards.citation import CitationScore, score_citations
from fdagym.rewards.correctness import Judge, UnimplementedJudge, score_correctness
from fdagym.rewards.dense import dense_correctness
from fdagym.rewards.efficiency import score_efficiency
from fdagym.rewards.format import score_format
from fdagym.rewards.hops import HOP_CAP, hop_score, hop_target_searched
from fdagym.rewards.version import DATA_VERSION, GRADER_VERSION

WEIGHTS = {"correctness": 0.60, "citation": 0.25, "format": 0.10, "efficiency": 0.05}


def score(trajectory: dict, task: dict, *, judge: Judge | None = None, dense: bool = False, snapshot=None) -> dict:
    final = trajectory.get("final_answer")
    format_score, format_notes = score_format(final)
    if format_score == 1.0 and isinstance(final, dict):
        citations = final.get("citations") or []
        answer = final.get("answer")
    else:
        citations = []
        answer = None

    steps = trajectory.get("steps") or []
    cited = score_citations(steps, task, citations, dense=dense)
    # JP's rule: on sponsor-hop, dense citation credit requires a lookup of the
    # hop-target drug. A one-hop citation of the anchor application (valid quote
    # or id-only partial) is worth 0, so it also cannot open the correctness
    # gate. Hop credit, format, and efficiency are separate terms and stay.
    # Strict eval does not pass dense=True, so this does not change it.
    if (
        dense
        and task.get("type") == "sponsor_hop"
        and not hop_target_searched(steps, task)
        and (cited.score > 0 or cited.valid or cited.id_hits)
    ):
        cited = CitationScore(
            score=0.0,
            valid=0,
            total=cited.total,
            empty_grounded=False,
            notes=[
                *cited.notes,
                "JP's rule: no sponsor-hop citation credit without a lookup of the hop-target drug",
            ],
            id_hits=0,
        )
    hop = hop_score(steps, task) if dense or task.get("type") == "sponsor_hop" else None
    if dense:
        correctness, correctness_notes = dense_correctness(
            answer,
            task,
            valid_citations=cited.valid,
            id_hits=cited.id_hits,
            format_ok=format_score == 1.0,
            empty_grounded=cited.empty_grounded,
            judge=judge or UnimplementedJudge(),
            snapshot=snapshot,
        )
        answer_correctness = correctness
        correctness, correctness_notes, answer_correctness = _apply_sponsor_proof(
            task, correctness, correctness_notes, answer_correctness, hop
        )
        if correctness <= 0 and hop is not None and hop.applies:
            correctness = HOP_CAP * hop.rate
            correctness_notes = [
                *correctness_notes,
                *hop.notes,
                f"dense hops: {hop.hits}/{hop.total} * {HOP_CAP:g} = {correctness:.4f}",
            ]
    else:
        correctness, correctness_notes = score_correctness(
            answer,
            task,
            valid_citations=cited.valid,
            format_ok=format_score == 1.0,
            empty_grounded=cited.empty_grounded,
            judge=judge,
        )
        answer_correctness = correctness
        correctness, correctness_notes, answer_correctness = _apply_sponsor_proof(
            task, correctness, correctness_notes, answer_correctness, hop
        )
    efficiency, efficiency_notes = score_efficiency(steps, task, answer_correctness)
    total = round(
        WEIGHTS["correctness"] * correctness
        + WEIGHTS["citation"] * cited.score
        + WEIGHTS["format"] * format_score
        + WEIGHTS["efficiency"] * efficiency,
        10,
    )
    notes = [
        *format_notes,
        *cited.notes,
        *correctness_notes,
        *efficiency_notes,
        (
            f"total {total:.4f} = 0.60*{correctness:.4f} + 0.25*{cited.score:.4f}"
            f" + 0.10*{format_score:.4f} + 0.05*{efficiency:.4f}" + (" dense" if dense else "")
        ),
    ]
    result = {
        "total": total,
        "correctness": correctness,
        "citation": cited.score,
        "format": format_score,
        "efficiency": efficiency,
        "citations_valid": cited.valid,
        "citations_total": cited.total,
        "grader_version": GRADER_VERSION,
        "data_version": DATA_VERSION,
        "notes": notes,
    }
    if dense and hop is not None:
        result["hops"] = hop.rate
        result["answer_correctness"] = answer_correctness
    return result


def _apply_sponsor_proof(task, correctness, notes, answer_correctness, hop):
    if task.get("type") != "sponsor_hop" or correctness <= 0 or hop is None or hop.proves():
        return correctness, notes, answer_correctness
    return (
        0.0,
        [*notes, "correctness needs the sponsor proof", *hop.notes],
        0.0,
    )
