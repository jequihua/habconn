# habconn Notebooks

Human-friendly walkthrough notebooks for the current single-landscape
`habconn` workflow. These are a **use/testing environment**, not
scientific evidence and not training infrastructure.

## Scope

All reusable logic lives in the `habconn` package. Notebooks import the
package APIs — they do **not** redefine builders, environments, or
trainers. Cells are intentionally small so a human can step through them
one at a time.

The notebooks exercise the **existing** Stage 4 single-landscape
workflow on `small_vector_001`. They do not run transfer-learning
experiments and they do not register a second landscape.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `00_environment_check.ipynb` | Verify the Python interpreter, that `habconn` imports from this checkout, that key dependencies are present, that `graphab.jar` is on disk, and that the bundled `small_vector_001` data resolves. |
| `01_problem_and_environment_walkthrough.ipynb` | Load `VectorConnectivityProblem`, inspect planning units, build a `VectorHabitatEnv` via `make_env`, reset it, and inspect observation keys + action masks. |
| `02_training_smoke_run.ipynb` | Build a tiny `ExperimentConfig`, call `run_experiment(...)`, and print the run directory + key metrics. Runs MaskablePPO end-to-end against Graphab — expect minutes, not seconds. |
| `03_outputs_evaluation_deployment_inspection.ipynb` | Open the artifacts produced by `02_*` (or any prior run) — `config.json`, `baseline_summary.json`, `evaluation/comparison.{json,csv}`, `selection/model_selection.json`, `deployment/deployment_summary.json`, `deployment/selected_planning_units.csv`, `inspection/observation_schema.json`, `inspection/deployment_action_trace.csv`. |

## Expected run cost

- `00_*` and `01_*` are cheap (no training, no Graphab evaluation
  beyond what env construction needs).
- `02_*` runs a 50-timestep MaskablePPO baseline. Each environment step
  calls Graphab CLI exact evaluation (~3-5 s). Expect a few minutes per
  run.
- `03_*` is offline (reads JSON/CSV only).

## Conventions

- Run all notebooks from the package root (`08_pkg/habconn/`) so the
  default relative paths resolve. The notebooks accept an optional
  `HABCONN_RUN_DIR` env var (`03_*`) for inspecting an external run.
- The training notebook writes to `tmp/notebooks/<run_name>/` — outside
  the canonical `tmp/experiments/` so notebook runs do not collide with
  the script smoke run.
- Notebooks must not be required by any unit or integration test. A
  cheap presence test confirms the files exist and parse as JSON
  notebooks.

## Non-goals

- No transfer-learning training or evaluation.
- No new landscape data; only `small_vector_001`.
- No graph/set encoders, no `SubprocVecEnv`, no reward normalization,
  no hyperparameter search.
- No scientific optimality claims.

## Where to look next

- `HPC_HABCONN_SMOKE_TEST.md` — compatibility gate for HPC clusters.
- `06_infra/hpc_training.md` — design notes for a configurable
  local/HPC training entry point + TensorBoard placement.
- `scripts/train_small_vector.py` — the existing primitive run surface.
