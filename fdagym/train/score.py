"""Score a Plan A completion: oracle_calls in the env, then submit_answer from the text."""

from fdagym.env import FDAEnv
from fdagym.snapshot import Snapshot
from fdagym.train.context import play_oracle
from fdagym.train.parse import parse_answer


def run_completion(
    snapshot: Snapshot,
    tasks: list[dict],
    task: dict,
    completion,
    *,
    run_id: str = "grpo",
    seed: int = 0,
    model: dict | None = None,
) -> dict:
    """Play oracle_calls, parse the completion, submit. Returns the trajectory.

    Distractors belong in the prompt only. Putting them in env.steps would burn efficiency
    and would make their ids valid citations.
    """
    env = FDAEnv(
        snapshot,
        tasks,
        run_id=run_id,
        seed=seed,
        model=model or {"name": "grpo", "kind": "trained", "checkpoint_step": None},
    )
    play_oracle(env, task)
    parsed = parse_answer(completion)
    arguments = parsed if parsed is not None else {}
    _obs, _total, _done, info = env.step({"tool": "submit_answer", "arguments": arguments})
    return info["trajectory"]


def score_completion(
    snapshot: Snapshot,
    tasks: list[dict],
    task: dict,
    completion,
    *,
    run_id: str = "grpo",
    seed: int = 0,
    model: dict | None = None,
    dense: bool = False,
) -> dict:
    """Same as ``run_completion`` but returns only the reward dict.

    ``dense=False`` is the strict score stored on trajectories. ``dense=True`` is
    the training reward: partial citation and near-miss correctness, same weights.
    """
    trajectory = run_completion(
        snapshot,
        tasks,
        task,
        completion,
        run_id=run_id,
        seed=seed,
        model=model,
    )
    if not dense:
        return trajectory["reward"]
    from fdagym.rewards import score

    return score(
        {"steps": trajectory["steps"], "final_answer": trajectory["final_answer"]},
        task,
        dense=True,
        snapshot=snapshot,
    )

