# Scripts directory

Scripts are command-line entry points for repeatable project workflows. Keep each command thin: parse arguments, validate the request, call the implementation in `fdagym/`, and print or save its result. Put reusable logic in the owning package module rather than duplicating it across scripts.

Group closely related hard-split audits under `audit_hard.py` (`disjointness`, `reachability`, `memorization`) and hard-split generation under `generate_hard.py` (`eval`, `train`). Use a preset or subcommand when command variants share a purpose; keep distinct training and evaluation protocols as distinct entry points. When changing a command, update the runbook, lessons, tests, and dashboard documentation in the same change.

Keep modules import-safe. Put remote-service imports and side effects inside `main()` or a function called by it. GPU and paid operations must require an explicit command. Snapshot, evaluation, and task audit commands should use the frozen local snapshot and remain reproducible offline where possible.

Use `uv add` for dependencies. Avoid introducing dependencies for small CLI tasks. Do not change schemas as part of script cleanup without updating the owning track and reporting the contract change.

## Command map

| Workflow | Entry point | Implementation |
|---|---|---|
| Record or inspect the frozen openFDA snapshot | `snapshot.py` | `fdagym.openfda`, `fdagym.queries` |
| Generate standard tasks | `python -m fdagym.tasks` | `fdagym.tasks` |
| Generate hard evaluation/train splits | `generate_hard.py eval\|train` | `fdagym.tasks.hard`, `fdagym.tasks.hard_train` |
| Audit hard splits | `audit_hard.py disjointness\|reachability\|memorization` | `fdagym.tasks`, `fdagym.swapped` |
| Run OpenRouter baseline | `run_baseline.py` | `fdagym.agents` |
| Score one model on a split via Chat Completions | `eval_chat.py` | `fdagym.agents.chat` |
| Train/evaluate locally | `grpo_singleturn.py`, `grpo_tools.py`, `sft_*.py`, `eval_local.py`, `eval_tools.py`, `run_gpu.py` | `fdagym.train` |
| Spawn a remote training job | `launch_pipeline.py`, `spawn_modal.py` | `fdagym.train.pipeline`, `fdagym.train.spawn_plan` |
| Compare or chart runs | `compare_runs.py`, `wandb_chart.py` | `fdagym.train` |

The old separate hard audit and hard generation commands were consolidated into the subcommands above. Use the command map and project docs rather than recreating those one-off files.
