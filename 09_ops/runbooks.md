# Runbooks

Operational runbooks for the current single-landscape `habconn`
workflow. There is no production app yet; "operations" here means
reproducible local and HPC execution.

## Notebook Walkthrough (Workstation)

For a guided, human-friendly tour of the existing workflow, use the
notebooks under `08_pkg/habconn/notebooks/`:

| Notebook | Purpose |
| --- | --- |
| `00_environment_check.ipynb` | Verify Python interpreter, `habconn` import, dependencies, `graphab.jar`, bundled `small_vector_001` data, and the landscape registry. |
| `01_problem_and_environment_walkthrough.ipynb` | Load the vector problem, build a `VectorHabitatEnv`, reset, and inspect the v2 observation + action masks. |
| `02_training_smoke_run.ipynb` | Run a tiny `ExperimentConfig` end-to-end via `run_experiment(...)`. |
| `03_outputs_evaluation_deployment_inspection.ipynb` | Open the artifacts produced by a prior run (config, summary, evaluation comparison, model selection, deployment summary, inspection schema, action trace). |

Run all notebooks from `08_pkg/habconn/`. Notebooks are a
use/exploration environment, not scientific evidence.

## HPC Compatibility Gate

The HPC compatibility check is in [HPC_HABCONN_SMOKE_TEST.md](../HPC_HABCONN_SMOKE_TEST.md).
It proves that the existing single-landscape MaskablePPO + Graphab
loop can run on a compute node and write the contract artifacts. It
uses two primitive entry points:

- `python scripts/train_small_vector.py` (fixed configuration), or
- an inline `run_experiment(...)` heredoc with scratch-safe
  `work_root` and `output_root` (sections 13 and 14 of the smoke
  runbook).

These are intentionally minimal. They are not a general training CLI.

## Training Rehearsal

The single-landscape training rehearsal runner is live:

- CLI: `08_pkg/habconn/scripts/run_experiment.py`
  (presets: `smoke` / `rehearsal`; TensorBoard via `--tensorboard`).
- Slurm template:
  [`06_infra/templates/habconn_training_rehearsal.slurm`](../06_infra/templates/habconn_training_rehearsal.slurm).
- Runbook: [`06_infra/training_rehearsal.md`](../06_infra/training_rehearsal.md).

The earlier design-of-record is in
[`06_infra/hpc_training.md`](../06_infra/hpc_training.md). The smoke
runbook ([`HPC_HABCONN_SMOKE_TEST.md`](../HPC_HABCONN_SMOKE_TEST.md))
remains the HPC compatibility gate; the rehearsal runner is the
next step after that gate has passed.

## Publish Repo Sync

The standalone `habconn` publish repo lives at
`C:/Users/dev/work/tum/habconn`. Update it from this dev repo via
the manifest-driven sync tool:

- Tool: `scripts/python/sync_habconn_publish_repo.py`.
- Manifest: `scripts/python/publish_habconn_sync_manifest.toml`.
- Runbook: [`06_infra/publish_repo_sync.md`](../06_infra/publish_repo_sync.md).
- Colleague handoff (in the publish repo):
  `docs/PUBLISH_REPO_HANDOFF.md`.

Standard preview + apply cycle:

```bash
cd C:/Users/dev/work/tum/drl-sp
python scripts/python/sync_habconn_publish_repo.py --check
python scripts/python/sync_habconn_publish_repo.py --apply
```

The tool refuses to run against a dirty target unless
`--allow-dirty-target` is passed, refuses to write outside the
target repo root, and does not commit or push automatically.

## Pointers

- Package README: `08_pkg/habconn/README.md`.
- Roadmap: `docs/HABCONN_ROADMAP.md`.
- Architecture (current vs. target): `06_infra/architecture.md`.
- Deployment (status: not active): `06_infra/deployment.md`.
- Smoke runbook: `HPC_HABCONN_SMOKE_TEST.md`.
- HPC training design: `06_infra/hpc_training.md`.
- Training rehearsal runbook: `06_infra/training_rehearsal.md`.
- Publish repo sync runbook: `06_infra/publish_repo_sync.md`.
