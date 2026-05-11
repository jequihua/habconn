#!/bin/bash
#SBATCH --job-name=habconn-rehearsal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=habconn-rehearsal-%j.out
#SBATCH --error=habconn-rehearsal-%j.err

# Single-landscape training rehearsal for the habconn package.
#
# Scope:
# - This is NOT a transfer-learning run.
# - Uses only the bundled small_vector_001 landscape.
# - Calls scripts/run_experiment.py (no inline training loop).
# - Optionally enables TensorBoard logging.
#
# Edit the placeholders below for your cluster.

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster modules. Names vary by site.
# ---------------------------------------------------------------------------
module purge
module load python/3.11   # or your cluster's Python module name
module load java/17        # required for Graphab CLI

# ---------------------------------------------------------------------------
# Paths. Adjust these for your environment.
# ---------------------------------------------------------------------------
export HABCONN_REPO=/path/to/drl-sp
export HABCONN_PKG=$HABCONN_REPO/08_pkg/habconn

# Job-local scratch. Falls back to repo tmp/ if $SCRATCH is unset.
export HABCONN_SCRATCH=${SCRATCH:-$HABCONN_PKG/tmp}/habconn_rehearsal_${SLURM_JOB_ID:-manual}
mkdir -p "$HABCONN_SCRATCH"

# Run identity. One Slurm job per run_name; change run_name per submission
# to keep artifact trees isolated under the same output_root.
RUN_NAME="rehearsal_${SLURM_JOB_ID:-manual}"

cd "$HABCONN_REPO"

# ---------------------------------------------------------------------------
# Activate the habconn environment. Edit for venv / conda.
# ---------------------------------------------------------------------------
# Example for a repo-root venv:
source .venv_hpc_habconn/bin/activate
# Example for a Conda/Mamba env:
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate habconn-hpc

cd "$HABCONN_PKG"

echo "=== Runtime information ==="
hostname
date
python --version
python -c "import sys; print(sys.executable)"
python -c "import habconn; print(habconn.__file__)"
java -version

# ---------------------------------------------------------------------------
# Run the rehearsal.
#
# --preset rehearsal selects the longer learning preset. Override any
# config knob with an explicit flag (see scripts/run_experiment.py --help).
# --tensorboard turns on TensorBoard logging at
# <output_root>/<run_name>/tensorboard/.
# ---------------------------------------------------------------------------
python scripts/run_experiment.py \
    --preset rehearsal \
    --run-name "$RUN_NAME" \
    --output-root "$HABCONN_SCRATCH/experiments" \
    --work-root "$HABCONN_SCRATCH/training_runs" \
    --tensorboard

echo "=== Output tree ==="
find "$HABCONN_SCRATCH/experiments/$RUN_NAME" -maxdepth 2 -type f -print

echo "=== Done ==="
date
