# LiteLLM supply-chain incident and guard

On October 21, 2026, LiteLLM's release pipeline published a malicious `litellm==1.107.0` package for about two hours. The official incident report says it stole credentials and provided remote-code execution; it also identifies affected Docker tags `v1.107.0`, `latest`, `main-latest`, and `main-stable`. The maintainer's remediation was to rebuild environments and pin `1.106.2` or another explicit pre-incident release. See the [incident report](https://docs.litellm.ai/blog/security-incident-oct-21-2026) and the [canonical attack-chain update](https://docs.litellm.ai/blog/security-incident-oct-21-2026-attack-chain).

## This repository's exposure check

The repo uses LiteLLM as a Python SDK through `scripts/run_baseline.py`; it has no LiteLLM proxy container or Docker service definition. The checked-in lockfile resolved to `1.105.1`, and the local project virtualenv metadata also reported `1.105.1`, both below the affected release. The current project dependency is pinned to the maintainer-recommended `1.106.2` release, with its PyPI artifact hashes locked.

If an environment actually ran `1.107.0` during the incident window, treat its reachable secrets as exposed and rebuild it. The maintainer says not to start Python in an affected environment before removing the incident's startup payload; follow the [incident remediation instructions](https://docs.litellm.ai/blog/security-incident-oct-21-2026#what-to-do-if-you-are-affected).

## Run the guard

```bash
python3 scripts/check_litellm_security.py
```

`.github/workflows/litellm-supply-chain.yml` runs the same guard on dependency-related pull requests and pushes, daily, and on manual dispatch. It checks the exact project pin, the locked version and all approved PyPI hashes, and rejects known affected or floating LiteLLM Docker tags. When adopting a newer release, verify it against the maintainer's current advisory, then update the exact pin and approved artifact hashes in both `pyproject.toml`, `uv.lock`, and the guard script in one reviewed change.
