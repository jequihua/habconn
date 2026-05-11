# habconn Publish Repo Handoff

This document is for a colleague pulling the standalone `habconn`
publish repo for the first time.

This repo is a **curated projection** of the active development repo
(`drl-sp`). It contains the package, tests, notebooks, example data,
Graphab tooling, and the operator handoff docs needed to install
`habconn`, smoke-test it, and run a single-landscape training
rehearsal on `small_vector_001`.

It is **not** a transfer-learning project. Only one bundled landscape
is registered.

## Layout

```text
README.md                           # package README
pyproject.toml                      # package metadata + dependencies
pytest.ini / conftest.py            # pytest configuration
.gitignore
HPC_HABCONN_SMOKE_TEST.md           # HPC compatibility runbook
graphab_command_line.md             # Graphab CLI notes
continue_development_instructions   # short dev hand-off
src/                                # the habconn package
tests/                              # unit + integration tests
scripts/                            # CLIs (incl. run_experiment.py)
configs/                            # bundled training/graphab configs
data/examples/small_vector_001/     # the only bundled landscape
tools/                              # graphab.jar + bootstrap script
notebooks/                          # human walkthrough notebooks
06_infra/
    hpc_training.md                 # design-of-record (knobs, TB, Slurm)
    training_rehearsal.md           # operator runbook for rehearsal runs
    templates/
        habconn_training_rehearsal.slurm
09_ops/runbooks.md                  # operational pointers
docs/
    HABCONN_RL_APPROACH.md
    HABCONN_SINGLE_LANDSCAPE_DRL_WORKFLOW.md
    HABCONN_ROADMAP.md
    PUBLISH_REPO_HANDOFF.md         # this file
03_experiments/run_summary.md       # latest run summary
```

## Install

Java 17 and Python 3.10+ are required. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # (Unix)
.\.venv\Scripts\Activate.ps1       # (Windows PowerShell)
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

The `dev` extra includes `pytest`, `stable-baselines3`, `sb3-contrib`,
`torch`, and `tensorboard>=2.13`.

Verify:

```bash
python -c "import habconn; print(habconn.__file__)"
python -c "import torch, stable_baselines3, sb3_contrib, tensorboard; print('ok')"
java -version
ls tools/graphab.jar
ls data/examples/small_vector_001
```

If geospatial dependencies fail to install via pip, see the
"Option B: Conda/Mamba" route in `HPC_HABCONN_SMOKE_TEST.md` section 7B.

## Run tests

```bash
python -m pytest tests/unit -q
```

Expected: all unit tests pass. The training-related integration tests
under `tests/integration/` call Graphab and take longer — run them
when verifying a fresh environment:

```bash
python -m pytest tests/ -q
```

## Notebook walkthrough

Open the notebooks from the repo root so default relative paths
resolve:

```bash
jupyter notebook notebooks/
```

Run them in order:

1. `00_environment_check.ipynb` — quick environment + data + registry
   sanity (no Graphab, no training).
2. `01_problem_and_environment_walkthrough.ipynb` — load
   `VectorConnectivityProblem`, build `VectorHabitatEnv`, inspect the
   v2 observation contract and action masks; takes one env step.
3. `02_training_smoke_run.ipynb` — runs a 50-timestep MaskablePPO
   smoke through `run_experiment(...)`. Expect a few minutes (Graphab
   CLI is ~3–5 s per env step). Outputs land under
   `tmp/notebooks/experiments/notebook_smoke/`.
4. `03_outputs_evaluation_deployment_inspection.ipynb` — opens the
   artifacts produced by the smoke run (config, summary, evaluation
   comparison, selected model, deployment summary, observation
   schema, action trace).

## HPC compatibility gate

`HPC_HABCONN_SMOKE_TEST.md` is the first check on a fresh HPC node.
It proves the runtime can run; it is intentionally minimal. The
canonical sequence on a compute node:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration/test_training_smoke.py::TestEnvFactory::test_make_env_creates_valid_env -q
python scripts/train_small_vector.py
```

## Training rehearsal CLI

The rehearsal runner lives at `scripts/run_experiment.py`. It is a
thin argparse wrapper around `ExperimentConfig` / `run_experiment`.

### Local smoke

```bash
python scripts/run_experiment.py \
  --preset smoke \
  --run-name cli_smoke
```

Compatibility-scale: 50 timesteps. Same magnitude as the existing
`train_small_vector.py`, but configurable.

### Local rehearsal with TensorBoard

```bash
python scripts/run_experiment.py \
  --preset rehearsal \
  --run-name rehearsal_001 \
  --output-root tmp/rehearsals \
  --work-root tmp/rehearsal_runs \
  --tensorboard
```

Roughly an hour on a typical workstation (Graphab CLI ~3–5 s per env
step). Adjust `--total-timesteps` to shorten.

`tensorboard --logdir tmp/rehearsals/rehearsal_001/tensorboard`
opens the live training scalars.

### HPC / Slurm

Edit `06_infra/templates/habconn_training_rehearsal.slurm` for your
cluster (module loads, venv activation, scratch path), then submit:

```bash
sbatch 06_infra/templates/habconn_training_rehearsal.slurm
```

The template calls the CLI; it does **not** embed the training loop.

See `06_infra/training_rehearsal.md` for a full operator walkthrough
(smoke vs. rehearsal, TensorBoard launch, post-run inspection,
deferred-work list).

## Where artifacts land

Each rehearsal run writes to `<output_root>/<run_name>/`:

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
tensorboard/                    # only when --tensorboard was passed
```

Inspect the artifacts with notebook
`03_outputs_evaluation_deployment_inspection.ipynb`.

## What is intentionally deferred

This publish repo does **not** support, and the code does not
implement:

- transfer-learning training or evaluation (only one bundled
  landscape; `development_split()` is explicitly not transfer-ready),
- second / synthesized landscapes,
- `SubprocVecEnv` (genuine multi-core parallelism beyond
  `DummyVecEnv`),
- checkpoint resume across crashed runs,
- reward normalization experiments,
- hyperparameter search,
- a production app or orchestration layer,
- graph/set encoders that consume the unused node-level observation
  arrays.

These are all live items in the dev roadmap (`docs/HABCONN_ROADMAP.md`);
they are not present in this publish repo yet.

## Where to look next

- `06_infra/training_rehearsal.md` — operator runbook.
- `06_infra/hpc_training.md` — design-of-record for the rehearsal
  contract (knob table, TensorBoard placement, Slurm requirements).
- `09_ops/runbooks.md` — operational pointers.
- `docs/HABCONN_SINGLE_LANDSCAPE_DRL_WORKFLOW.md` — narrative
  description of the workflow.
- `docs/HABCONN_RL_APPROACH.md` — the RL approach.
- `docs/HABCONN_ROADMAP.md` — current stages + deferred work.
