# HPC Training Readiness — Design

Status: **superseded — implemented**. The configurable
single-landscape training rehearsal runner now ships as
`08_pkg/habconn/scripts/run_experiment.py` (presets `smoke` and
`rehearsal`, TensorBoard via `--tensorboard`), with the Slurm
template at
`06_infra/templates/habconn_training_rehearsal.slurm` and the
runbook at `06_infra/training_rehearsal.md`.

This document remains as the **design-of-record** for the knobs,
TensorBoard placement, and scratch-safe HPC conventions. Update it
alongside changes to the CLI or the trainer's TensorBoard wiring.

## Context

Stage 4 (single-landscape DRL on `small_vector_001`) is complete
end-to-end. Stage 5 milestone 1 (landscape registry + split contract)
is in place. Transfer-learning milestones are archived/deferred. The
active direction is **human-friendly use and HPC training readiness**
for the existing single-landscape workflow.

## Current primitive run surface

Two ways to run a training experiment exist today:

1. **Fixed smoke script**:

   ```bash
   cd 08_pkg/habconn
   .venv/Scripts/python scripts/train_small_vector.py
   ```

   - Uses package-root-relative paths.
   - Writes to `tmp/experiments/baseline_small_vector_001/`.
   - Hard-codes `run_name`, `budget=3`, `k=10`,
     `total_timesteps=50`, `n_eval_episodes=1`.
   - Useful for the HPC compatibility gate; not configurable from the
     command line.

2. **Inline `run_experiment` snippets** in
   `HPC_HABCONN_SMOKE_TEST.md` (sections 13 and 14):

   ```python
   from habconn.training.experiment import ExperimentConfig, run_experiment

   config = ExperimentConfig(
       run_name='hpc_smoke',
       data_dir=pkg / 'data' / 'examples' / 'small_vector_001',
       graphab_jar=pkg / 'tools' / 'graphab.jar',
       work_root=scratch / 'training_runs',
       output_root=scratch / 'experiments',
       budget=3,
       k=10,
       seed=42,
       total_timesteps=50,
       n_eval_episodes=1,
   )
   summary = run_experiment(config)
   ```

   - Scratch-safe (`work_root` / `output_root` point under
     `$SCRATCH`).
   - Allows per-job configuration without editing the script.
   - Heredoc-based; not a CLI.

These two entry points remain the HPC compatibility gate. The
configurable training rehearsal runner described below is the next
step after that gate.

## Training rehearsal runner — implemented

The configurable single-landscape training rehearsal runner is live:

- CLI: `08_pkg/habconn/scripts/run_experiment.py` (thin argparse
  wrapper around `ExperimentConfig` / `run_experiment`).
- Presets: `--preset smoke` (50 timesteps, matches the compatibility
  gate) and `--preset rehearsal` (1024 timesteps, longer learning
  run); the default is `smoke`.
- Explicit CLI flags for every `ExperimentConfig` field; flags
  override preset values.
- Opt-in TensorBoard logging via `--tensorboard` (default
  placement) or `--tensorboard-log <path>` (override + implies
  enabling). When disabled, no `tensorboard/` directory appears.
- Slurm template: `06_infra/templates/habconn_training_rehearsal.slurm`
  (calls the CLI directly, no inline training loop).
- Operator runbook: `06_infra/training_rehearsal.md` (smoke vs.
  rehearsal, local command, TensorBoard launch, HPC/Slurm usage,
  artifact inspection, deferred list).

This document remains the **design-of-record** for the
`ExperimentConfig` knob table, TensorBoard placement, scratch-safe
HPC conventions, and Slurm template requirements. Update it
alongside changes to the CLI or trainer wiring.

The rehearsal runner still uses only `small_vector_001`. It does
not add transfer learning, new landscapes, `SubprocVecEnv`,
checkpoint resume, hyperparameter search, or reward normalization.

## Configurable entry point — knob table

The CLI exposes every `ExperimentConfig` knob without duplicating
trainer logic. It:

- parses CLI flags,
- constructs an `ExperimentConfig`,
- calls `run_experiment(config)`,
- prints the run directory + key metrics on completion.

### Config knobs

All fields below already exist on `ExperimentConfig` (see
`08_pkg/habconn/src/habconn/training/experiment.py`):

| Knob | Default | Notes |
| --- | --- | --- |
| `run_name` | `"baseline"` | Slug; validated against `[A-Za-z0-9_][A-Za-z0-9_.-]*`. |
| `seed` | `42` | Used for env reset, vectorized worker seeds, baseline RNG. |
| `data_dir` | `data/examples/small_vector_001` | Bundled fixture; only `small_vector_001` is registered. |
| `graphab_jar` | `tools/graphab.jar` | Required for exact CLI evaluation. |
| `work_root` | `tmp/training_runs` | Scratch for Graphab work; should point under `$SCRATCH` on HPC. |
| `output_root` | `tmp/experiments` | Run directory parent; should also point under `$SCRATCH` (or a project-writable path). |
| `budget` | `3` | Number of planning units selected per episode. |
| `k` | `10` | Candidate slot count per step. |
| `total_timesteps` | `50` | PPO timesteps. |
| `learning_rate` | `3e-4` | PPO. |
| `n_steps` | `8` | PPO rollout length. |
| `batch_size` | `4` | PPO. |
| `n_epochs` | `2` | PPO. |
| `gamma` | `0.99` | PPO. |
| `n_eval_episodes` | `1` | Deterministic post-training eval. |
| `n_envs` | `1` | `DummyVecEnv` size (no `SubprocVecEnv` yet). |
| `checkpoint_freq` | `16` | Callback invocations between checkpoints. |
| `selection_metric` | `mean_final_pc` | Whitelist: `mean_final_pc` / `mean_delta_pc` / `mean_return`. |
| `selection_mode` | `max` | Currently only `max`. |

### TensorBoard knobs (implemented)

- `enable_tensorboard: bool = False` — top-level on/off switch.
- `tensorboard_log: Optional[Path] = None` — explicit override of
  the default log directory. When `None` and
  `enable_tensorboard=True`, the trainer uses
  `ExperimentPaths.tensorboard_dir = output_root/run_name/tensorboard/`.
- `tb_log_name: str = "ppo"` — SB3 sub-run name passed to
  `model.learn(...)`.

`MaskablePPO(..., tensorboard_log=...)` is constructed lazily — the
SB3 argument is `None` when TensorBoard is disabled, so no event
file is created.

### Command shape

`08_pkg/habconn/scripts/run_experiment.py` accepts every
`ExperimentConfig` knob as a CLI flag and calls
`run_experiment(config)`. Example invocation:

```bash
cd 08_pkg/habconn
python scripts/run_experiment.py \
  --run-name hpc_smoke_a \
  --seed 42 \
  --data-dir data/examples/small_vector_001 \
  --graphab-jar tools/graphab.jar \
  --work-root "$SCRATCH/habconn/training_runs" \
  --output-root "$SCRATCH/habconn/experiments" \
  --budget 3 --k 10 \
  --total-timesteps 50 \
  --n-envs 1 \
  --checkpoint-freq 16 \
  --selection-metric mean_final_pc
```

See `scripts/run_experiment.py --help` for the canonical flag list.

## TensorBoard placement (implemented)

- Default log directory: `output_root/run_name/tensorboard/`,
  exposed as `ExperimentPaths.tensorboard_dir`.
- `ExperimentPaths.ensure()` deliberately does **not** create the
  directory. The trainer creates it lazily only when
  `tensorboard_log` is set, so the directory appears on disk only
  for runs that actually requested TensorBoard.
- The resolved path is surfaced in the returned run summary, in
  `baseline_summary.json` (top-level and inside the embedded
  `config` block), and in `config.json` (which records the user's
  override or `null`).
- TensorBoard logging is **off by default**. Enable it via
  `--tensorboard` (default placement) or `--tensorboard-log <path>`
  (explicit override; also implies enabling). The
  `tensorboard>=2.13` dependency is declared in the `train` and
  `dev` extras of `08_pkg/habconn/pyproject.toml`.

## Scratch-safe HPC conventions

- Set both `work_root` and `output_root` under `$SCRATCH` (or the
  cluster equivalent), not under the repository.
- Use one Slurm job per training run. Different runs should use
  different `run_name` values to avoid overwriting artifacts inside
  the same `output_root`.
- The `n_envs > 1` path writes per-worker Graphab scratch directories
  under `work_root/worker_NNN/`. This isolation is already verified
  end-to-end (see `tests/integration/`).
- A crashed run cannot currently resume; checkpoint-resume is
  deferred. Re-running from scratch is the only option today.

## Slurm template requirements

A Slurm template for a single training run should:

- module-load Python 3.10+ and Java 17,
- activate the `habconn` venv/conda env,
- set `HABCONN_PKG=<repo>/08_pkg/habconn`,
- set `HABCONN_SCRATCH=$SCRATCH/habconn_<job-id>`,
- export `work_root=$HABCONN_SCRATCH/training_runs` and
  `output_root=$HABCONN_SCRATCH/experiments`,
- call the rehearsal CLI directly
  (`python scripts/run_experiment.py --preset rehearsal ...`); do
  not copy the trainer's internals into the job script,
- collect output artifacts under `output_root/run_name/`.

The shipped template at
`06_infra/templates/habconn_training_rehearsal.slurm` implements
this contract. `HPC_HABCONN_SMOKE_TEST.md` section 14 retains its
inline-heredoc example as historical context for the compatibility
gate; rehearsal-scale runs should go through the CLI instead.

## Notebooks vs. HPC training

The notebooks under `08_pkg/habconn/notebooks/` are for **human use
and exploration** on a workstation. They are not Slurm-friendly: they
run a small interactive loop and write to
`tmp/notebooks/experiments/`. They should not be required by any
unit or integration test.

The HPC training entry point is for **non-interactive runs** on a
compute node. It writes to `$SCRATCH`-relative paths and is meant to
be called from a Slurm script.

## What remains deferred

- Checkpoint-resume across crashed jobs.
- `SubprocVecEnv` (genuine multi-core parallelism).
- Multi-landscape data and transfer-learning training.
- Hyperparameter search.
- Reward normalization experiments.
- A production app / orchestration layer.

This document is the foundation for the next implementation pass. It
does not commit code.
