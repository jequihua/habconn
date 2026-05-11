# Single-Landscape Training Rehearsal Runbook

This runbook explains how to run a longer, artifact-rich
single-landscape training rehearsal on `small_vector_001` — locally
or under Slurm.

**Status:** runtime is single-landscape. Transfer learning, additional
landscapes, `SubprocVecEnv`, checkpoint resume, hyperparameter
search, and reward normalization remain deferred.

## Smoke test vs. rehearsal

| | Smoke test | Rehearsal |
| --- | --- | --- |
| Purpose | HPC compatibility gate — proves the runtime can run | Learning rehearsal — produces a longer, reviewable run |
| Entry point | `scripts/train_small_vector.py` (fixed) or `scripts/run_experiment.py --preset smoke` | `scripts/run_experiment.py --preset rehearsal` |
| Timesteps | 50 | 1024 (preset default) |
| Checkpoint cadence | every 16 callbacks | every 64 callbacks |
| Eval episodes | 1 | 3 |
| TensorBoard | optional | recommended (`--tensorboard`) |
| Where it lands | `tmp/experiments/baseline_small_vector_001/` (fixed) or whatever `--output-root` resolves to | typically `tmp/rehearsals/<run_name>/` locally, `$SCRATCH/.../<run_name>/` on HPC |

The compatibility smoke test still lives in
[`HPC_HABCONN_SMOKE_TEST.md`](../HPC_HABCONN_SMOKE_TEST.md). It is
intentionally minimal and is the first check on a fresh HPC node.
This rehearsal runbook is for the *next* step.

## Local rehearsal

From the package root:

```bash
cd 08_pkg/habconn
.venv/Scripts/python scripts/run_experiment.py \
  --preset rehearsal \
  --run-name rehearsal_001 \
  --output-root tmp/rehearsals \
  --work-root tmp/rehearsal_runs \
  --tensorboard
```

Expected wall time on a typical workstation: roughly an hour, since
each environment step calls Graphab CLI exact evaluation
(~3-5 s per step). Use `--total-timesteps` to shorten or extend.

Outputs land in
`tmp/rehearsals/rehearsal_001/`:

```text
config.json
metadata.json
history.jsonl
baseline_summary.json
evaluation/comparison.{json,csv}
checkpoints/checkpoint_NNNNNN_steps.zip
selection/{checkpoint_evaluations.json, model_selection.json}
deployment/{deployment_summary.json,
            selected_planning_units.gpkg,
            selected_planning_units.csv}
inspection/{observation_schema.json, feature_summary.json,
            deployment_action_trace.json,
            deployment_action_trace.csv}
models/{final_model.zip, best_model.zip}
tensorboard/        # only when --tensorboard is passed
```

`tensorboard/` only appears when TensorBoard logging was requested.
Other paths are always produced.

## Starting TensorBoard

After (or during) a rehearsal run with `--tensorboard`:

```bash
.venv/Scripts/python -m tensorboard.main \
  --logdir tmp/rehearsals/rehearsal_001/tensorboard
```

Or with the installed launcher:

```bash
tensorboard --logdir tmp/rehearsals/rehearsal_001/tensorboard
```

Open the printed URL in a browser. Per-rollout SB3 scalars (rollout
length, policy loss, value loss, etc.) appear under the `ppo`
sub-run (or whatever `--tb-log-name` you passed).

`--tensorboard-log <path>` overrides the default log directory and
also enables TensorBoard. Use it when the run directory and log
directory must live on different filesystems (for example, run
artifacts under `$SCRATCH` and TensorBoard logs under a
home-relative path that a remote TensorBoard process can read).

## HPC rehearsal (Slurm)

Use the template:

```text
06_infra/templates/habconn_training_rehearsal.slurm
```

Edit the placeholders (repo path, venv activation, module loads,
job resources), then submit:

```bash
sbatch 06_infra/templates/habconn_training_rehearsal.slurm
```

The template runs the new CLI; it does **not** call an inline
training loop.

To monitor TensorBoard from a compute job:

- ensure `--tensorboard` is in the CLI call (already is in the
  template),
- copy or mount the resulting `$SCRATCH/.../rehearsal_001/tensorboard/`
  directory to where you can run TensorBoard,
- or run TensorBoard on a login node against the `$SCRATCH` path if
  the cluster allows it.

## What to inspect after a rehearsal

- `baseline_summary.json` — top-level metrics, selected candidate,
  deployment selected planning units, every artifact path.
- `selection/model_selection.json` — which candidate won and why
  (metric, mode, tie-break rule).
- `evaluation/comparison.csv` — trained policy vs.
  `random_valid` / `lowest_cost` / `largest_area` on the same
  landscape.
- `deployment/selected_planning_units.csv` — the selected PUs in
  policy-selection order.
- `inspection/observation_schema.json` — the v2 observation contract
  and which keys the flat extractor actually consumes.
- `inspection/deployment_action_trace.csv` — one row per
  (step × candidate slot), with `chosen` / `valid` / costs / areas
  / running PC.
- `tensorboard/` — per-rollout SB3 scalars when enabled.

For a human-friendly walkthrough of these artifacts, see the notebook
[`03_outputs_evaluation_deployment_inspection.ipynb`](../08_pkg/habconn/notebooks/03_outputs_evaluation_deployment_inspection.ipynb).

## What remains deferred

- Transfer-learning training and evaluation (deferred — only
  `small_vector_001` is registered).
- Additional bundled / synthesized landscapes.
- `SubprocVecEnv` (genuine multi-core parallelism).
- Checkpoint resume across crashed jobs.
- Hyperparameter search.
- Reward normalization experiments.
- A production app / orchestration layer.

This rehearsal runner is the runtime entry point for the existing
contract; it is not a transfer-learning experiment.
