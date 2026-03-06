Param(
    [string]$ProjectName = "habconn"
)

$ErrorActionPreference = "Stop"

$root = Join-Path (Get-Location) $ProjectName

if (Test-Path $root) {
    Write-Host "Directory already exists: $root" -ForegroundColor Yellow
} else {
    New-Item -ItemType Directory -Path $root | Out-Null
    Write-Host "Created project root: $root" -ForegroundColor Green
}

# ---------------------------------------
# Directories
# ---------------------------------------
$dirs = @(
    ".github",
    ".github/workflows",
    "configs",
    "configs/train",
    "configs/env",
    "configs/graphab",
    "data",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/examples",
    "data/examples/small_vector_001",
    "docs",
    "docs/architecture",
    "docs/design_notes",
    "docs/experiments",
    "notebooks",
    "scripts",
    "src",
    "src/habconn",
    "src/habconn/types",
    "src/habconn/config",
    "src/habconn/io",
    "src/habconn/problems",
    "src/habconn/state",
    "src/habconn/features",
    "src/habconn/evaluators",
    "src/habconn/envs",
    "src/habconn/models",
    "src/habconn/models/extractors",
    "src/habconn/models/policies",
    "src/habconn/training",
    "src/habconn/experiments",
    "src/habconn/utils",
    "tests",
    "tests/fixtures",
    "tests/integration",
    "tests/unit",
    "tools"
)

foreach ($dir in $dirs) {
    $path = Join-Path $root $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

# ---------------------------------------
# Files with starter content
# ---------------------------------------
$files = @{
    ".gitignore" = @"
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
.eggs/
dist/
build/
pip-wheel-metadata/

# Virtual environments
.venv/
venv/
env/

# Testing / coverage
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Project data
data/interim/
data/processed/

# Runtime / scratch
tmp/
scratch/
logs/
"@

    "pyproject.toml" = @"
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "habconn"
version = "0.1.0"
description = "Habitat connectivity optimization with reinforcement learning and Graphab"
readme = "README.md"
requires-python = ">=3.10"
authors = [
  { name = "Project Team" }
]
dependencies = []

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
"@

    "src/habconn/__init__.py" = @"
__all__ = []
__version__ = "0.1.0"
"@

    "src/habconn/types/__init__.py" = ""
    "src/habconn/types/aliases.py" = @"
\"\"\"Type aliases used across the package.\"\"\"
"@
    "src/habconn/types/dataclasses.py" = @"
\"\"\"Shared dataclasses and small structured containers.\"\"\"
"@

    "src/habconn/config/__init__.py" = ""
    "src/habconn/config/schema.py" = @"
\"\"\"Configuration schemas for the project.\"\"\"
"@
    "src/habconn/config/loaders.py" = @"
\"\"\"Utilities to load and validate config files.\"\"\"
"@

    "src/habconn/io/__init__.py" = ""
    "src/habconn/io/vectors.py" = @"
\"\"\"Load and validate vector planning-unit data.\"\"\"
"@
    "src/habconn/io/rasters.py" = @"
\"\"\"Load and validate habitat and resistance rasters.\"\"\"
"@
    "src/habconn/io/graphab_inputs.py" = @"
\"\"\"Prepare files and structures needed by Graphab.\"\"\"
"@
    "src/habconn/io/paths.py" = @"
\"\"\"Filesystem path helpers for local and HPC execution.\"\"\"
"@

    "src/habconn/problems/__init__.py" = ""
    "src/habconn/problems/base.py" = @"
\"\"\"Core optimization problem interfaces.\"\"\"
"@
    "src/habconn/problems/vector_problem.py" = @"
\"\"\"Vector-first habitat connectivity problem definition.\"\"\"
"@
    "src/habconn/problems/raster_problem.py" = @"
\"\"\"Placeholder for future raster-action problem definitions.\"\"\"
"@
    "src/habconn/problems/candidate_generation.py" = @"
\"\"\"Candidate generation for Option B fixed-K action selection.\"\"\"
"@
    "src/habconn/problems/budgeting.py" = @"
\"\"\"Budget and cost accounting utilities.\"\"\"
"@
    "src/habconn/problems/normalization.py" = @"
\"\"\"Feature normalization and scaling utilities.\"\"\"
"@

    "src/habconn/state/__init__.py" = ""
    "src/habconn/state/landscape_state.py" = @"
\"\"\"Dynamic state of one optimization episode.\"\"\"
"@
    "src/habconn/state/transitions.py" = @"
\"\"\"State transition logic after applying one action.\"\"\"
"@
    "src/habconn/state/action_mapping.py" = @"
\"\"\"Map policy action slots to real planning units.\"\"\"
"@
    "src/habconn/state/masks.py" = @"
\"\"\"Action and node masking logic.\"\"\"
"@
    "src/habconn/state/termination.py" = @"
\"\"\"Episode termination conditions.\"\"\"
"@

    "src/habconn/features/__init__.py" = ""
    "src/habconn/features/node_features.py" = @"
\"\"\"Per-planning-unit feature builders.\"\"\"
"@
    "src/habconn/features/global_features.py" = @"
\"\"\"Global landscape and episode feature builders.\"\"\"
"@
    "src/habconn/features/candidate_features.py" = @"
\"\"\"Feature builders for candidate action slots.\"\"\"
"@
    "src/habconn/features/topology_features.py" = @"
\"\"\"Topology and neighborhood summary features.\"\"\"
"@
    "src/habconn/features/packing.py" = @"
\"\"\"Padding, packing, and mask creation for observations.\"\"\"
"@

    "src/habconn/evaluators/__init__.py" = ""
    "src/habconn/evaluators/base.py" = @"
\"\"\"Abstract interfaces for connectivity evaluators.\"\"\"
"@
    "src/habconn/evaluators/graphab_runner.py" = @"
\"\"\"Low-level Graphab command runner and process management.\"\"\"
"@
    "src/habconn/evaluators/graphab_evaluator.py" = @"
\"\"\"High-level evaluator that computes connectivity through Graphab.\"\"\"
"@
    "src/habconn/evaluators/cached_evaluator.py" = @"
\"\"\"Caching wrapper for expensive connectivity evaluations.\"\"\"
"@
    "src/habconn/evaluators/surrogate_evaluator.py" = @"
\"\"\"Placeholder for future approximate or learned evaluators.\"\"\"
"@
    "src/habconn/evaluators/reward.py" = @"
\"\"\"Reward shaping and objective-delta computation.\"\"\"
"@

    "src/habconn/envs/__init__.py" = ""
    "src/habconn/envs/vector_env.py" = @"
\"\"\"Gymnasium environment for vector-action habitat restoration.\"\"\"
"@
    "src/habconn/envs/wrappers.py" = @"
\"\"\"Environment wrappers, including masking helpers.\"\"\"
"@
    "src/habconn/envs/observation_space.py" = @"
\"\"\"Observation space definitions for Gymnasium.\"\"\"
"@
    "src/habconn/envs/action_space.py" = @"
\"\"\"Action space definitions and validation.\"\"\"
"@

    "src/habconn/models/__init__.py" = ""
    "src/habconn/models/extractors/__init__.py" = ""
    "src/habconn/models/extractors/base.py" = @"
\"\"\"Base feature extractor interfaces.\"\"\"
"@
    "src/habconn/models/extractors/padded_mlp.py" = @"
\"\"\"Initial padded-and-masked feature extractor for variable-size inputs.\"\"\"
"@
    "src/habconn/models/extractors/set_encoder.py" = @"
\"\"\"Placeholder for future set/attention encoder.\"\"\"
"@
    "src/habconn/models/extractors/gnn_encoder.py" = @"
\"\"\"Placeholder for future graph neural encoder.\"\"\"
"@
    "src/habconn/models/policies/__init__.py" = ""
    "src/habconn/models/policies/masked_policy.py" = @"
\"\"\"Policy definitions for masked candidate-action selection.\"\"\"
"@

    "src/habconn/training/__init__.py" = ""
    "src/habconn/training/make_env.py" = @"
\"\"\"Factory functions to create training and evaluation environments.\"\"\"
"@
    "src/habconn/training/callbacks.py" = @"
\"\"\"SB3 callbacks for logging, evaluation, and checkpointing.\"\"\"
"@
    "src/habconn/training/trainer.py" = @"
\"\"\"Training entry points and orchestration utilities.\"\"\"
"@
    "src/habconn/training/evaluation.py" = @"
\"\"\"Policy evaluation utilities.\"\"\"
"@
    "src/habconn/training/vecenv.py" = @"
\"\"\"Vectorized environment creation and worker-safe setup.\"\"\"
"@

    "src/habconn/experiments/__init__.py" = ""
    "src/habconn/experiments/small_vector_benchmark.py" = @"
\"\"\"Small benchmark experiment definitions.\"\"\"
"@

    "src/habconn/utils/__init__.py" = ""
    "src/habconn/utils/logging.py" = @"
\"\"\"Logging setup utilities.\"\"\"
"@
    "src/habconn/utils/seeding.py" = @"
\"\"\"Random seed utilities for reproducible experiments.\"\"\"
"@
    "src/habconn/utils/tempdirs.py" = @"
\"\"\"Temporary directory and scratch-space helpers.\"\"\"
"@
    "src/habconn/utils/profiling.py" = @"
\"\"\"Profiling and timing helpers.\"\"\"
"@

    "configs/train/ppo_small_vector.yaml" = @"
# Placeholder training config
experiment_name: ppo_small_vector
"@

    "configs/env/small_vector_debug.yaml" = @"
# Placeholder environment config
env_name: small_vector_debug
"@

    "configs/graphab/local.yaml" = @"
# Placeholder local Graphab config
mode: local
"@

    "configs/graphab/hpc.yaml" = @"
# Placeholder HPC Graphab config
mode: hpc
"@

    "scripts/prepare_small_dataset.py" = @"
\"\"\"Prepare tiny example datasets for early development.\"\"\"
"@
    "scripts/run_baseline_graphab.py" = @"
\"\"\"Run baseline Graphab evaluation on a small landscape.\"\"\"
"@
    "scripts/train_small_vector.py" = @"
\"\"\"Train a first small vector-action RL model.\"\"\"
"@
    "scripts/evaluate_policy.py" = @"
\"\"\"Evaluate a trained policy.\"\"\"
"@
    "scripts/deploy_policy.py" = @"
\"\"\"Deploy a trained policy to produce a restoration plan.\"\"\"
"@

    "tests/unit/test_candidate_generation.py" = @"
def test_placeholder():
    assert True
"@
    "tests/unit/test_action_masks.py" = @"
def test_placeholder():
    assert True
"@
    "tests/unit/test_budgeting.py" = @"
def test_placeholder():
    assert True
"@
    "tests/unit/test_state_transitions.py" = @"
def test_placeholder():
    assert True
"@
    "tests/unit/test_feature_packing.py" = @"
def test_placeholder():
    assert True
"@
    "tests/unit/test_graphab_runner.py" = @"
def test_placeholder():
    assert True
"@
    "tests/integration/test_vector_env_smoke.py" = @"
def test_placeholder():
    assert True
"@

    "tools/bootstrap_habconn.ps1" = "# Placeholder for bootstrap script copy"
    "README.md" = "# habconn`r`n`r`nProject README to be replaced with the detailed version."
}

foreach ($relativePath in $files.Keys) {
    $fullPath = Join-Path $root $relativePath
    $parent = Split-Path $fullPath -Parent
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    if (-not (Test-Path $fullPath)) {
        Set-Content -Path $fullPath -Value $files[$relativePath] -Encoding UTF8
    }
}

Write-Host ""
Write-Host "Project structure created successfully at:" -ForegroundColor Green
Write-Host "  $root" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Replace README.md with the detailed version"
Write-Host "  2. Create a virtual environment"
Write-Host "  3. Start implementing:"
Write-Host "     - src/habconn/problems/vector_problem.py"
Write-Host "     - src/habconn/state/landscape_state.py"
Write-Host "     - src/habconn/evaluators/graphab_runner.py"
Write-Host "     - src/habconn/evaluators/graphab_evaluator.py"