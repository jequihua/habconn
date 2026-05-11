# habconn HPC Installation And Smoke-Test Runbook

This document is a step-by-step guide for installing `habconn` on an HPC system and running a small smoke test of the current package.

It is written for a colleague who may not know the project history.

The goal is **not** to run a serious training experiment yet. The goal is to confirm that the current single-landscape, Graphab-backed training loop can run on an HPC node from a clean environment and write the expected artifacts.

---

## 1. What This Smoke Test Proves

The current package can run a small end-to-end loop:

```text
small_vector_001 data
-> VectorConnectivityProblem
-> VectorHabitatEnv
-> Graphab CLI exact PC evaluation
-> MaskablePPO training
-> single-landscape evaluation comparison
-> config.json + metadata.json + history.jsonl + baseline_summary.json
   + evaluation/comparison.{json,csv} + models/final_model.zip
```

If the smoke test succeeds, it proves:

- Python dependencies can be installed on the HPC.
- Java is available and can run Graphab.
- `graphab.jar` is visible at the expected path.
- The package imports from the intended checkout.
- The bundled example data can be read by GeoPandas/Rasterio.
- Graphab can run in the HPC filesystem.
- A tiny masked-action PPO baseline can complete and write artifacts.

It does **not** prove:

- transfer learning,
- scientific optimality,
- multi-landscape generalization,
- HPC scalability,
- safe multi-worker Graphab orchestration,
- GPU training performance,
- or readiness for production-scale experiments.

Treat this as an HPC compatibility check.

---

## 2. Current Package State

The active package is:

```text
08_pkg/habconn/
```

The current verified local state is:

- backend milestone: closed
- environment milestone: closed
- minimal trainable baseline milestone: closed
- richer feature baseline/evaluation milestone: closed
- single-landscape experiment contract milestone: closed (Stage 4 milestone 1)
- worker-safe vectorized env training milestone: closed (Stage 4 milestone 2)
  — `DummyVecEnv` only; `SubprocVecEnv` deferred
- single-landscape evaluation suite milestone: closed (Stage 4 milestone 3)
  — trained policy compared to `random_valid`, `lowest_cost`, and
  `largest_area` baselines; comparison written to
  `<run_dir>/evaluation/comparison.{json,csv}`
- checkpointing + best-model selection milestone: closed (Stage 4 milestone 4)
  — periodic checkpoints under `<run_dir>/checkpoints/`, selected
  candidate copied to `<run_dir>/models/best_model.zip`, selection
  rule recorded in `<run_dir>/selection/{checkpoint_evaluations.json,
  model_selection.json}`
- deployment export milestone: closed (Stage 4 milestone 5)
  — `best_model.zip` is loaded and deployed on the dedicated single-env
  eval; selected planning units exported to
  `<run_dir>/deployment/{deployment_summary.json,
  selected_planning_units.gpkg, selected_planning_units.csv}`
- feature inspection + deployment action trace milestone: closed
  (Stage 4 milestone 6) — v2 observation schema + initial-observation
  feature summary + per-step action trace written to
  `<run_dir>/inspection/{observation_schema.json,
  feature_summary.json, deployment_action_trace.json,
  deployment_action_trace.csv}`
- landscape registry and split contract milestone: implemented
  (Stage 5 milestone 1) — `experiments/landscape_registry.py`
  ships `LandscapeSpec`, `LandscapeSplit`, validators, the built-in
  registry (only `small_vector_001`), and the development split
  (explicitly not transfer-ready). No new bundled landscapes; no
  transfer-learning training or evaluation.
- tests: `287 passed` locally in the package-local venv
- bundled data: only one landscape, `small_vector_001`
- training path: configurable `n_envs` (default 1, single-env legacy path);
  `n_envs > 1` uses `DummyVecEnv` with isolated `<work_root>/worker_NNN/`
  Graphab scratch directories per worker

Important current limitations:

- Only one bundled landscape exists.
- Training is slow because each environment step calls exact Graphab evaluation.
- `DummyVecEnv` runs sub-environments serially in the same process; there is
  no `SubprocVecEnv` parallelism yet.
- The current model is a flat MLP extractor, not a graph or set encoder.
- Node-level arrays exist in the observation but are not consumed by the current extractor.
- Checkpointing is in place but there is no checkpoint-resume; a crashed
  run still restarts from scratch.
- No HPC-specific vectorized environment orchestration exists yet.

---

## 3. Expected Repository Layout

After cloning or copying the repository to the HPC, these files/directories should exist:

```text
<repo-root>/
    08_pkg/
        habconn/
            pyproject.toml
            README.md
            data/
                examples/
                    small_vector_001/
                        candidates.shp
                        habitat.tif
                        resistance.tif
            scripts/
                train_small_vector.py
            src/
                habconn/
            tests/
            tools/
                graphab.jar
```

The most important paths are:

```text
08_pkg/habconn/pyproject.toml
08_pkg/habconn/tools/graphab.jar
08_pkg/habconn/data/examples/small_vector_001/
```

If `tools/graphab.jar` is missing, the Graphab-backed smoke test cannot run until the jar is copied into place or the training configuration is adjusted to point to the jar elsewhere.

---

## 4. HPC Prerequisites

The HPC needs:

- Linux shell access.
- Python 3.10 or newer, preferably Python 3.11.
- Java runtime compatible with Graphab. Use Java 17 if available.
- Ability to install Python packages into a user environment.
- Enough local or scratch disk space for temporary Graphab runs.
- Access to the repository checkout and bundled example data.

Recommended resources for the first smoke test:

- 1 node
- 1 task
- 1 CPU core is enough, but 2-4 CPUs are fine
- 4-8 GB RAM
- 20-30 minutes wall time
- no GPU required

The smoke test is CPU-bound and Graphab-bound. Requesting a GPU is unnecessary.

---

## 5. Login Node Versus Compute Node

Most clusters distinguish between:

- **login nodes**, used for editing, environment setup, and lightweight checks;
- **compute nodes**, used for actual jobs.

Recommended policy:

- Create/install the Python environment on the login node if the cluster allows it.
- Do not run the full training smoke test on the login node.
- Run Graphab-backed integration or training commands inside an allocated compute job.

Lightweight checks that are usually okay on a login node:

```bash
python --version
java -version
python -c "import sys; print(sys.executable)"
```

Commands that should usually run on a compute node:

```bash
python -m pytest tests/integration/test_training_smoke.py -q
python scripts/train_small_vector.py
```

---

## 6. Load System Modules

On many HPC systems, Python and Java are provided through environment modules.

Start from the repository root:

```bash
cd /path/to/drl-sp
```

Inspect available modules:

```bash
module avail python
module avail java
module avail gdal
```

Load suitable modules. The exact names vary by cluster. Examples:

```bash
module load python/3.11
module load java/17
```

Some clusters also require geospatial libraries:

```bash
module load gdal
module load geos
module load proj
```

If the cluster uses Conda/Mamba instead of modules, skip to the Conda section below.

Verify versions:

```bash
python --version
java -version
```

Expected:

- Python should be `3.10+`.
- Java should ideally be `17`.

If Java is missing or too old, stop and ask the HPC administrator which Java module should be used for Graphab.

---

## 7. Create A Python Environment

There are two recommended installation routes:

- **Option A: venv + pip**, simplest if binary wheels work on the cluster.
- **Option B: conda/mamba**, safer for geospatial dependencies if `rasterio` or `geopandas` installation fails under pip.

Use only one option.

---

## 7A. Option A: Create A venv With pip

*Important note: at NHR@FAU HPC it is discouraged to work with venv but to work with conda*

From the repository root:

```bash
cd /path/to/drl-sp
python -m venv .venv_hpc_habconn
source .venv_hpc_habconn/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Move into the package root:

```bash
cd 08_pkg/habconn
```

Install the package in editable mode with development/training dependencies:

```bash
python -m pip install -e ".[dev]"
```

What this installs:

- core package dependencies: `numpy`, `pandas`, `geopandas`, `rasterio`, `shapely`, `gymnasium`
- test dependency: `pytest`
- training dependencies: `torch`, `stable-baselines3`, `sb3-contrib`

If this succeeds, continue to section 8.

If installation fails on `geopandas`, `rasterio`, `fiona`, `pyproj`, or `shapely`, use the Conda/Mamba option instead. Geospatial Python packages often depend on native GDAL/PROJ/GEOS libraries, and Conda is usually more reliable on HPC systems.

---

## 7B. Option B: Create A Conda/Mamba Environment

Use this if the cluster supports Conda/Mamba or if the pip install path fails.

From the repository root:

```bash
cd /path/to/drl-sp
```

Create the environment:

```bash
mamba create -n habconn-hpc -c conda-forge python=3.11 numpy pandas geopandas rasterio shapely pytest pip
```

If `mamba` is not available, use `conda`:

```bash
conda create -n habconn-hpc -c conda-forge python=3.11 numpy pandas geopandas rasterio shapely pytest pip
```

Activate it:

```bash
conda activate habconn-hpc
```

Install training packages and the local package:

```bash
cd 08_pkg/habconn
python -m pip install -e ".[dev]"
```

This may still install `torch`, `stable-baselines3`, and `sb3-contrib` from pip. For CPU-only smoke testing, that is acceptable.

If the HPC has a preferred PyTorch install command, follow the site policy first, then run:

```bash
python -m pip install -e ".[dev]"
```

---

## 8. Verify The Installation

All commands in this section should be run from:

```bash
cd /path/to/drl-sp/08_pkg/habconn
```

Check Python executable:

```bash
python -c "import sys; print(sys.executable)"
```

Check package import:

```bash
python -c "import habconn; print(habconn.__file__)"
```

Expected output should point inside this checkout:

```text
/path/to/drl-sp/08_pkg/habconn/src/habconn/__init__.py
```

If it points somewhere else, the wrong checkout is installed. Re-run:

```bash
python -m pip install -e ".[dev]"
python -c "import habconn; print(habconn.__file__)"
```

Check training dependencies:

```bash
python -c "import gymnasium, torch, stable_baselines3, sb3_contrib; print('training deps ok')"
```

Check geospatial dependencies:

```bash
python -c "import geopandas, rasterio, shapely; print('geospatial deps ok')"
```

Check Java:

```bash
java -version
```

Check Graphab jar:

```bash
ls -lh tools/graphab.jar
```

Expected: the file exists and is roughly tens of MB. In the current repo it is about 56 MB.

Check example data:

```bash
ls -lh data/examples/small_vector_001
```

Expected: shapefile sidecars and rasters are present, including:

```text
candidates.shp
habitat.tif
resistance.tif
```

Note: a shapefile requires several sidecar files, commonly `.shp`, `.shx`, `.dbf`, `.prj`. Do not copy only `candidates.shp` by itself.

---

## 9. Run Fast Unit Tests First

Before submitting a Graphab-backed training job, run the unit tests.

From:

```bash
cd /path/to/drl-sp/08_pkg/habconn
```

Run:

```bash
python -m pytest tests/unit -q
```

Expected:

- unit tests pass;
- no Graphab-heavy training loop is started.

If unit tests fail, stop and fix the Python environment before running the HPC smoke test.

Common unit-test failure causes:

- package imported from the wrong checkout;
- missing `torch` / `stable-baselines3` / `sb3-contrib`;
- missing geospatial libraries;
- environment was not activated.

---

## 10. Run A Minimal Graphab Import/Data Check

This check confirms the example data can be loaded.

From:

```bash
cd /path/to/drl-sp/08_pkg/habconn
```

Run:

```bash
python - <<'PY'
from pathlib import Path
from habconn.problems.vector_problem import VectorConnectivityProblem

root = Path.cwd()
data_dir = root / "data" / "examples" / "small_vector_001"

problem = VectorConnectivityProblem.from_files(
    name="small_vector_001",
    vector_path=data_dir / "candidates.shp",
    habitat_raster_path=data_dir / "habitat.tif",
    resistance_raster_path=data_dir / "resistance.tif",
    id_column="lyr_1",
    area_column="area",
    uniform_cost=1.0,
)

print(problem.summary())
PY
```

Expected:

- command exits successfully;
- printed summary includes `name: small_vector_001`;
- `n_planning_units` is positive.

This does not run Graphab yet. It only verifies Python GIS loading.

---

## 11. Recommended Smoke Test Strategy

Use three levels:

1. **Unit tests**: verify Python package logic.
2. **Small selected integration test**: verify environment/training path can touch Graphab.
3. **Tiny baseline script**: verify the end-to-end training script writes artifacts.

Do not start with the full test suite on a new HPC environment. The full suite is currently valid, but it repeatedly exercises Graphab and takes several minutes locally.

Recommended first HPC sequence:

```bash
cd /path/to/drl-sp/08_pkg/habconn
python -m pytest tests/unit -q
python -m pytest tests/integration/test_training_smoke.py::TestEnvFactory::test_make_env_creates_valid_env -q
python scripts/train_small_vector.py
```

If these pass, the HPC smoke test is successful.

Optional later validation:

```bash
python -m pytest tests -q
```

Expected full-suite result in the current package state:

```text
287 passed
```

---

## 12. Running The Built-In Baseline Script

The simplest smoke test is:

```bash
cd /path/to/drl-sp/08_pkg/habconn
python scripts/train_small_vector.py
```

This script uses paths relative to the package root:

```text
data/examples/small_vector_001
tools/graphab.jar
tmp/training_runs
tmp/experiments/baseline_small_vector_001
```

The script routes through the experiment contract in
`habconn.training.experiment.run_experiment`, so each run is a
self-contained directory:

```text
tmp/experiments/baseline_small_vector_001/
    config.json
    metadata.json
    history.jsonl
    baseline_summary.json
    models/
        final_model.zip
```

Expected console output ends with:

```text
=== Baseline Training Complete ===
  Total timesteps       : 50
  Training episodes     : ...
  Eval episodes         : 1
  Eval mean return      : ...
  Eval mean final PC    : ...
  Eval mean delta PC    : ...
  Eval mean steps       : 3.0
  Ep0 baseline PC       : ...
  Ep0 final PC          : ...
  Ep0 selected PUs      : [...]
  Model path            : ...
  History path          : ...
  Summary path          : ...
```

Expected artifacts:

```text
tmp/experiments/baseline_small_vector_001/config.json
tmp/experiments/baseline_small_vector_001/metadata.json
tmp/experiments/baseline_small_vector_001/history.jsonl
tmp/experiments/baseline_small_vector_001/baseline_summary.json
tmp/experiments/baseline_small_vector_001/evaluation/comparison.json
tmp/experiments/baseline_small_vector_001/evaluation/comparison.csv
tmp/experiments/baseline_small_vector_001/checkpoints/checkpoint_NNNNNN_steps.zip
tmp/experiments/baseline_small_vector_001/selection/checkpoint_evaluations.json
tmp/experiments/baseline_small_vector_001/selection/model_selection.json
tmp/experiments/baseline_small_vector_001/deployment/deployment_summary.json
tmp/experiments/baseline_small_vector_001/deployment/selected_planning_units.gpkg
tmp/experiments/baseline_small_vector_001/deployment/selected_planning_units.csv
tmp/experiments/baseline_small_vector_001/inspection/observation_schema.json
tmp/experiments/baseline_small_vector_001/inspection/feature_summary.json
tmp/experiments/baseline_small_vector_001/inspection/deployment_action_trace.json
tmp/experiments/baseline_small_vector_001/inspection/deployment_action_trace.csv
tmp/experiments/baseline_small_vector_001/models/final_model.zip
tmp/experiments/baseline_small_vector_001/models/best_model.zip
```

Representative local result from the current package:

```text
Training episodes     : 18
Eval mean return      : 4.792e-06
Eval mean final PC    : 2.652311e-05
Eval mean delta PC    : 4.792e-06
Ep0 selected PUs      : [3, 4, 5]
```

Small numerical differences are acceptable across machines, as long as:

- the run completes;
- artifacts are written;
- final PC is finite;
- at least one planning unit is selected;
- no invalid-action failure occurs.

---

## 13. Scratch-Safe Smoke Test Alternative

Some HPC systems discourage writing temporary files under the repository directory. In that case, use a scratch-safe inline run instead of `scripts/train_small_vector.py`.

Set environment variables first:

```bash
export HABCONN_REPO=/path/to/drl-sp
export HABCONN_PKG=$HABCONN_REPO/08_pkg/habconn
export HABCONN_SCRATCH=${SCRATCH:-$HABCONN_PKG/tmp}/habconn_smoke_${SLURM_JOB_ID:-manual}
mkdir -p "$HABCONN_SCRATCH"
```

Run:

```bash
cd "$HABCONN_PKG"

python - <<'PY'
import os
from pathlib import Path

from habconn.training.experiment import ExperimentConfig, run_experiment

pkg = Path(os.environ["HABCONN_PKG"]).resolve()
scratch = Path(os.environ["HABCONN_SCRATCH"]).resolve()

config = ExperimentConfig(
    run_name="hpc_smoke",
    data_dir=pkg / "data" / "examples" / "small_vector_001",
    graphab_jar=pkg / "tools" / "graphab.jar",
    work_root=scratch / "training_runs",
    output_root=scratch / "experiments",
    budget=3,
    k=10,
    seed=42,
    total_timesteps=50,
    n_eval_episodes=1,
)

summary = run_experiment(config)

eval_summary = summary["evaluation"]
ep0 = eval_summary["episodes"][0]

print("\n=== HPC Scratch Smoke Test Complete ===")
print(f"run_dir              : {summary['run_dir']}")
print(f"total_timesteps      : {summary['total_timesteps']}")
print(f"training_episodes    : {summary['n_training_episodes_logged']}")
print(f"eval_mean_return     : {eval_summary['mean_return']:.3e}")
print(f"eval_mean_final_pc   : {eval_summary['mean_final_pc']:.6e}")
print(f"eval_mean_delta_pc   : {eval_summary['mean_delta_pc']:.3e}")
print(f"ep0_selected_pus     : {ep0['selected_pu_ids']}")
print(f"config_path          : {summary['config_path']}")
print(f"metadata_path        : {summary['metadata_path']}")
print(f"history_path         : {summary['history_path']}")
print(f"summary_path         : {summary['summary_path']}")
print(f"model_path           : {summary['model_path']}")
PY
```

This writes artifacts to:

```text
$HABCONN_SCRATCH/experiments/hpc_smoke/
```

This is preferable for cluster jobs because Graphab work directories and baseline outputs are isolated per job.

---

## 14. Example Slurm Job Script

If the cluster uses Slurm, create a file such as:

```text
run_habconn_smoke.slurm
```

Example - it does not fully work on the used HPC:

```bash
#!/bin/bash
#SBATCH --job-name=habconn-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=habconn-smoke-%j.out
#SBATCH --error=habconn-smoke-%j.err

set -euo pipefail

# Adjust these module names for the local HPC.
module purge
module load python/3.11
module load java/17

# Repository path on the HPC.
export HABCONN_REPO=/path/to/drl-sp
export HABCONN_PKG=$HABCONN_REPO/08_pkg/habconn

# Use job-local scratch if available, otherwise fall back to package tmp.
export HABCONN_SCRATCH=${SCRATCH:-$HABCONN_PKG/tmp}/habconn_smoke_${SLURM_JOB_ID}
mkdir -p "$HABCONN_SCRATCH"

cd "$HABCONN_REPO"

# Activate the environment created during setup.
source .venv_hpc_habconn/bin/activate

cd "$HABCONN_PKG"

echo "=== Runtime information ==="
hostname
date
python --version
python -c "import sys; print(sys.executable)"
python -c "import habconn; print(habconn.__file__)"
java -version
ls -lh tools/graphab.jar

echo "=== Unit tests ==="
python -m pytest tests/unit -q

echo "=== Minimal env factory integration test ==="
python -m pytest tests/integration/test_training_smoke.py::TestEnvFactory::test_make_env_creates_valid_env -q

echo "=== Scratch-safe baseline smoke run ==="
python - <<'PY'
import os
from pathlib import Path

from habconn.training.experiment import ExperimentConfig, run_experiment

pkg = Path(os.environ["HABCONN_PKG"]).resolve()
scratch = Path(os.environ["HABCONN_SCRATCH"]).resolve()

config = ExperimentConfig(
    run_name="hpc_smoke",
    data_dir=pkg / "data" / "examples" / "small_vector_001",
    graphab_jar=pkg / "tools" / "graphab.jar",
    work_root=scratch / "training_runs",
    output_root=scratch / "experiments",
    budget=3,
    k=10,
    seed=42,
    total_timesteps=50,
    n_eval_episodes=1,
)

summary = run_experiment(config)
eval_summary = summary["evaluation"]
ep0 = eval_summary["episodes"][0]

print("\n=== HPC Smoke Test Complete ===")
print(f"run_dir              : {summary['run_dir']}")
print(f"total_timesteps      : {summary['total_timesteps']}")
print(f"training_episodes    : {summary['n_training_episodes_logged']}")
print(f"eval_mean_return     : {eval_summary['mean_return']:.3e}")
print(f"eval_mean_final_pc   : {eval_summary['mean_final_pc']:.6e}")
print(f"eval_mean_delta_pc   : {eval_summary['mean_delta_pc']:.3e}")
print(f"ep0_selected_pus     : {ep0['selected_pu_ids']}")
print(f"config_path          : {summary['config_path']}")
print(f"metadata_path        : {summary['metadata_path']}")
print(f"history_path         : {summary['history_path']}")
print(f"summary_path         : {summary['summary_path']}")
print(f"model_path           : {summary['model_path']}")
PY

echo "=== Output artifacts ==="
find "$HABCONN_SCRATCH" -maxdepth 3 -type f -print

echo "=== Done ==="
date
```

This is the SLURM script that worked:



```bash
#!/bin/bash -l
#SBATCH --job-name=habconn-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=habconn-smoke-%j.out
#SBATCH --error=habconn-smoke-%j.err
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

module purge
module load java/jdk8u345-b01-hotspot
module load python

conda activate habconn-py311

export HABCONN_PKG="$HOME/habconn/habconn-main"
export HABCONN_SCRATCH="${TMPDIR:-${SCRATCH:-$HABCONN_PKG/tmp}}/habconn_smoke_${SLURM_JOB_ID}"
mkdir -p "$HABCONN_SCRATCH"

cd "$HABCONN_PKG"

echo "=== Runtime information ==="
hostname
date
pwd
python --version
which python
python -c "import sys; print(sys.executable)"
python -c "import habconn; print(habconn.__file__)"
java -version
ls -lh tools/graphab.jar

echo "=== Unit tests ==="
python -m pytest tests/unit -q

echo "=== Minimal env factory integration test ==="
python -m pytest tests/integration/test_training_smoke.py::TestEnvFactory::test_make_env_creates_valid_env -q

echo "=== Scratch-safe baseline smoke run ==="
python - <<'PY'
import os
from pathlib import Path

from habconn.training.trainer import BaselineConfig, train_baseline

pkg = Path(os.environ["HABCONN_PKG"]).resolve()
scratch = Path(os.environ["HABCONN_SCRATCH"]).resolve()

config = BaselineConfig(
    data_dir=pkg / "data" / "examples" / "small_vector_001",
    graphab_jar=pkg / "tools" / "graphab.jar",
    work_root=scratch / "training_runs",
    output_dir=scratch / "baseline_output",
    budget=3,
    k=10,
    seed=42,
    total_timesteps=50,
    n_eval_episodes=1,
)

summary = train_baseline(config)
eval_summary = summary["evaluation"]
ep0 = eval_summary["episodes"][0]

print("\n=== HPC Smoke Test Complete ===")
print(f"scratch              : {scratch}")
print(f"total_timesteps      : {summary['total_timesteps']}")
print(f"training_episodes    : {summary['n_training_episodes_logged']}")
print(f"eval_mean_return     : {eval_summary['mean_return']:.3e}")
print(f"eval_mean_final_pc   : {eval_summary['mean_final_pc']:.6e}")
print(f"eval_mean_delta_pc   : {eval_summary['mean_delta_pc']:.3e}")
print(f"ep0_selected_pus     : {ep0['selected_pu_ids']}")
print(f"model_path           : {summary['model_path']}")
print(f"history_path         : {summary['history_path']}")
print(f"summary_path         : {config.output_dir / 'baseline_summary.json'}")
PY

echo "=== Output artifacts ==="
find "$HABCONN_SCRATCH" -maxdepth 4 -type f -print | sort

echo "=== Done ==="
date
```



Submit:

```bash
sbatch run_habconn_smoke.sh
```

Watch:

```bash
squeue -u "$USER"
```

After it finishes, inspect:

```bash
cat habconn-smoke-<jobid>.out
cat habconn-smoke-<jobid>.err
```

Expected final output includes:

```text
=== HPC Smoke Test Complete ===
```

and lists:

```text
config.json
metadata.json
history.jsonl
baseline_summary.json
evaluation/comparison.json
evaluation/comparison.csv
checkpoints/checkpoint_NNNNNN_steps.zip
selection/checkpoint_evaluations.json
selection/model_selection.json
deployment/deployment_summary.json
deployment/selected_planning_units.gpkg
deployment/selected_planning_units.csv
inspection/observation_schema.json
inspection/feature_summary.json
inspection/deployment_action_trace.json
inspection/deployment_action_trace.csv
models/final_model.zip
models/best_model.zip
```

---

## 15. Checking The Output Artifacts

After a successful run, inspect the summary JSON:

```bash
python - <<'PY'
import json
from pathlib import Path
import os

scratch = Path(os.environ.get("HABCONN_SCRATCH", "tmp")).resolve()
summary_path = scratch / "experiments" / "hpc_smoke" / "baseline_summary.json"

if not summary_path.exists():
    summary_path = Path("tmp/experiments/baseline_small_vector_001/baseline_summary.json")

print(summary_path)
summary = json.loads(summary_path.read_text())
print(json.dumps(summary["evaluation"], indent=2))
PY
```

Check history:

```bash
head -n 5 "$HABCONN_SCRATCH/experiments/hpc_smoke/history.jsonl"
wc -l "$HABCONN_SCRATCH/experiments/hpc_smoke/history.jsonl"
```

Expected:

- `history.jsonl` has at least one JSON line;
- `baseline_summary.json` has an `evaluation` section;
- `evaluation.episodes[0].step_rewards` is present;
- `evaluation.episodes[0].step_pc_values` is present;
- `evaluation.episodes[0].selected_pu_ids` is non-empty.

---

## 16. Optional Full Test Suite

Only run the full suite after the smaller smoke checks pass.

```bash
cd /path/to/drl-sp/08_pkg/habconn
python -m pytest tests -q
```

Expected current result:

```text
287 passed
```

Local reference runtime was about 4-5 minutes. HPC runtime may vary depending on filesystem speed, Java startup overhead, and node load.

If the full suite is too slow for the queue policy, this is acceptable for the first HPC smoke test. The minimum useful HPC validation is:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration/test_training_smoke.py::TestEnvFactory::test_make_env_creates_valid_env -q
python scripts/train_small_vector.py
```

or the scratch-safe equivalent.

---

## 17. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'habconn'`

Cause:

- package was not installed into the active environment;
- wrong environment is active.

Fix:

```bash
cd /path/to/drl-sp/08_pkg/habconn
python -m pip install -e ".[dev]"
python -c "import habconn; print(habconn.__file__)"
```

Expected path:

```text
/path/to/drl-sp/08_pkg/habconn/src/habconn/__init__.py
```

---

### Problem: `ModuleNotFoundError` for `torch`, `stable_baselines3`, or `sb3_contrib`

Cause:

- training dependencies were not installed.

Fix:

```bash
cd /path/to/drl-sp/08_pkg/habconn
python -m pip install -e ".[dev]"
```

Verify:

```bash
python -c "import torch, stable_baselines3, sb3_contrib; print('ok')"
```

---

### Problem: `ImportError` or native library errors from `rasterio`, `geopandas`, `fiona`, `pyproj`, or `shapely`

Cause:

- geospatial native dependencies are missing or incompatible.

Fix:

- Prefer the Conda/Mamba installation route using `conda-forge`.
- Ask the HPC team whether GDAL/PROJ/GEOS modules must be loaded.

Example:

```bash
mamba create -n habconn-hpc -c conda-forge python=3.11 numpy pandas geopandas rasterio shapely pytest pip
conda activate habconn-hpc
cd /path/to/drl-sp/08_pkg/habconn
python -m pip install -e ".[dev]"
```

---

### Problem: `java: command not found`

Cause:

- Java module is not loaded or Java is not installed.

Fix:

```bash
module avail java
module load java/17
java -version
```

If no Java module is available, ask the HPC administrator.

---

### Problem: Graphab jar missing

Symptom:

- errors mention `tools/graphab.jar` not found.

Check:

```bash
cd /path/to/drl-sp/08_pkg/habconn
ls -lh tools/graphab.jar
```

Fix:

- copy `graphab.jar` into `08_pkg/habconn/tools/`;
- or adjust `BaselineConfig(graphab_jar=...)` in the scratch-safe smoke snippet to point to the actual jar location.

---

### Problem: Shapefile cannot be read

Cause:

- only `candidates.shp` was copied without sidecar files.

Fix:

- ensure all shapefile sidecars are copied together:

```text
candidates.shp
candidates.shx
candidates.dbf
candidates.prj
```

There may also be `.cpg` or other sidecars. Copy the whole `small_vector_001` directory.

---

### Problem: Permission denied writing temporary files

Cause:

- the repo directory is read-only or not appropriate for job scratch.

Fix:

- use the scratch-safe smoke test in section 13;
- set `HABCONN_SCRATCH` to a writable job-local directory.

Example:

```bash
export HABCONN_SCRATCH=$SCRATCH/habconn_smoke_$SLURM_JOB_ID
mkdir -p "$HABCONN_SCRATCH"
```

---

### Problem: Training appears slow or `fps` is zero

This is expected for the current exact Graphab-backed training path.

The smoke test uses exact Graphab CLI calls inside environment steps. It is not optimized for speed yet.

Success criterion:

- the 50-timestep run completes and writes artifacts.

Do not interpret low `fps` as a failure unless the job times out or Graphab crashes.

---

### Problem: Full test suite takes too long

This is acceptable for the first HPC smoke test.

Use the smaller sequence:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration/test_training_smoke.py::TestEnvFactory::test_make_env_creates_valid_env -q
python scripts/train_small_vector.py
```

---

## 18. What To Report Back

After running the smoke test, report:

- HPC cluster name.
- Node type / partition.
- Python version.
- Java version.
- Whether venv or Conda was used.
- Exact install command used.
- Whether `import habconn` points to `08_pkg/habconn/src/habconn`.
- Unit test result.
- Selected integration test result.
- Baseline script or scratch-safe run result.
- Wall-clock runtime.
- Location of output artifacts.
- Any errors from Graphab, Java, GeoPandas, Rasterio, or filesystem permissions.

Suggested report template:

```text
habconn HPC smoke test report

Cluster:
Partition:
Node:
Date:
Python:
Java:
Environment type: venv / conda
Package path from import:

Unit tests:
Integration smoke:
Baseline smoke:

Runtime:
Output directory:

Evaluation mean return:
Evaluation mean final PC:
Selected PUs:
Training episodes logged:

Issues encountered:
Conclusion:
```

---

## 19. Pass/Fail Criteria

The HPC smoke test passes if:

- package imports from this checkout;
- unit tests pass;
- Java and Graphab jar are available;
- at least one Graphab-backed integration check passes;
- the tiny baseline training run completes;
- the experiment-contract artifacts are written under the run directory:
  `config.json`, `metadata.json`, `history.jsonl`,
  `baseline_summary.json`, `evaluation/comparison.json`,
  `evaluation/comparison.csv`, at least one
  `checkpoints/checkpoint_NNNNNN_steps.zip`,
  `selection/checkpoint_evaluations.json`,
  `selection/model_selection.json`,
  `deployment/deployment_summary.json`,
  `deployment/selected_planning_units.gpkg`,
  `deployment/selected_planning_units.csv`,
  `inspection/observation_schema.json`,
  `inspection/feature_summary.json`,
  `inspection/deployment_action_trace.json`,
  `inspection/deployment_action_trace.csv`,
  `models/final_model.zip`, and `models/best_model.zip`;
- `baseline_summary.json` contains a finite final PC and non-empty selected planning units.

The smoke test fails if:

- the package cannot be installed;
- geospatial dependencies cannot import;
- Java or Graphab cannot run;
- the example data cannot be read;
- the baseline run crashes;
- artifacts are not written;
- `habconn` imports from the wrong checkout.

---

## 20. Recommended Next Step After A Successful Smoke Test

<<<<<<< HEAD
If this smoke test passes, the next step is **not** transfer
learning or large-scale HPC training. It is a longer learning
rehearsal on the same single landscape.

Use the implemented training rehearsal runner:

- CLI: `08_pkg/habconn/scripts/run_experiment.py`
  (`--preset smoke` matches this compatibility gate;
  `--preset rehearsal` runs a longer learning run; opt into
  TensorBoard with `--tensorboard`).
- Operator runbook: [`06_infra/training_rehearsal.md`](06_infra/training_rehearsal.md).
- HPC/Slurm template:
  [`06_infra/templates/habconn_training_rehearsal.slurm`](06_infra/templates/habconn_training_rehearsal.slurm).
- Design-of-record (knobs, conventions):
  [`06_infra/hpc_training.md`](06_infra/hpc_training.md).

For a human-friendly workstation walkthrough of the produced
artifacts, see the notebook scaffold under
`08_pkg/habconn/notebooks/`.

**This smoke runbook remains the HPC compatibility gate.** Keep it
short and fast — do not slow it down into a rehearsal. The
rehearsal runner is the second step after this gate has passed.
=======
If this smoke test passes, the next project milestone should not immediately be large-scale HPC training.

The recommended next milestone is:

```text
Stage 4: multi-landscape data and transfer-learning-ready evaluation
```

That means:

- add at least one additional real example landscape;
- define landscape-level metadata and split artifacts;
- make evaluation run across multiple landscapes;
- preserve the single-landscape smoke test as a stable baseline;
- only then consider vectorized or HPC-scaled training.

The HPC smoke test is a compatibility gate. It should come before, not replace, the multi-landscape milestone.
>>>>>>> 9515ef820bae372f0f77fdc80998d735bd0ed182
