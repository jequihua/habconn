# Single-Landscape DRL Workflow Plan

## Purpose
This plan supersedes the immediate transfer-learning push. Transfer learning
remains a long-term goal, but the active development target is now simpler:

Build a complete single-landscape deep reinforcement learning workflow for
`small_vector_001`.

The workflow should train a masked-action PPO policy, evaluate it against
simple baselines, select a final model, and deploy that model to produce a
restoration selection artifact.

## Current Starting Point
The package already has:

- `VectorConnectivityProblem` for loading the bundled vector/raster fixture
- `VectorHabitatEnv` with candidate-slot actions, action masks, delta-PC reward,
  and terminal-consistent observations
- a v2 observation contract with candidate, node, and global fields
- `FlatObsExtractor` consuming candidate/global fields
- `MaskablePPO` integration through `env.action_masks()`
- `train_baseline()` and `scripts/train_small_vector.py`
- `evaluate_policy()` with per-step reward and PC traces
- `EpisodeHistoryCallback` writing `history.jsonl`
- final model saving and `baseline_summary.json`

The current run is still a smoke test: 50 timesteps, one synchronous env, no
checkpoint selection, no vectorized training, no deployment artifact.

## Active Non-Goal
Do not treat the next phase as transfer learning.

Do not claim:

- multi-landscape generalization
- train/eval landscape splits
- unseen-landscape performance
- production-scale HPC readiness

Those can return later after the single-landscape workflow is strong.

## Target End-to-End Workflow
The target command sequence should eventually support:

```text
configure experiment
-> train MaskablePPO with masked actions, optionally vectorized
-> log training history
-> checkpoint models
-> evaluate checkpoints/final model against simple baselines
-> select best model by explicit metric
-> deploy selected model on small_vector_001
-> export selected planning units, metrics, and map/GIS-friendly artifacts
```

## Milestone 1: Experiment Contract And Config
Goal: create a real experiment contract before adding more moving parts.

Deliverables:

- an experiment config object or simple JSON/YAML schema
- explicit output directory layout
- saved `config.json`
- saved environment/package metadata
- clear run naming and seed handling
- a command or function that runs the existing single-env baseline through this
  new output contract

Keep this milestone small. It should not introduce vectorized envs yet.

## Milestone 2: Vectorized Environment Training
Goal: train with more than one environment while keeping Graphab scratch safe.

Deliverables:

- `training/vecenv.py` creates worker-safe vectorized envs
- each worker gets isolated Graphab work directories
- `MaskablePPO` receives action masks correctly
- `n_envs` is configurable and recorded in artifacts
- tests prove at least `n_envs=2` works on the bundled fixture or through a
  cheap reliable path

## Milestone 3: Evaluation Suite
Goal: make model performance interpretable by comparing it with simple
baselines.

Deliverables:

- trained policy deterministic evaluation
- trained policy stochastic evaluation if useful
- random-valid baseline
- simple greedy baselines, such as lowest cost and largest area
- comparison table with final PC, delta-PC, selected PUs, total cost, and step
  count
- JSON/CSV artifacts for cross-run comparison

## Milestone 4: Checkpointing And Best Model Selection
Goal: stop treating the latest model as automatically best.

Deliverables:

- periodic checkpoints
- periodic evaluation results
- explicit model-selection metric, initially final PC or delta-PC
- `best_model.zip`
- `model_selection.json` explaining why the model was selected
- ability to resume or at least recover useful state after interruption

## Milestone 5: Deployment And Final Restoration Plan
Goal: use the selected policy to produce a human-reviewable restoration plan.

Deliverables:

- load `best_model.zip`
- run deterministic deployment on `small_vector_001`
- export selected planning-unit IDs
- export selected planning-unit table with order, cost, area, and PC trace
- export selected geometries as GeoPackage or shapefile
- write a compact deployment summary
- optionally produce a simple static map image

## Milestone 6: Feature Inspection And Reporting
Goal: make the current observation understandable to humans.

Deliverables:

- feature inspection script or command
- observation key, dtype, shape, min/max, and finite-value report
- reset and post-action example observations
- candidate feature summaries
- node feature summaries
- artifact such as `feature_report.json` or `feature_report.md`

## Milestone 7: Useful Single-Case Enhancements
Only after the core workflow is stable, consider:

- budget sensitivity runs
- candidate strategy comparisons
- reward normalization experiments
- node-level aggregate features for the flat extractor
- simple plots of PC improvement and training history
- richer deployment reports

## Review Standard
Each milestone must be reviewed for:

- masked-action correctness
- Graphab scratch isolation
- honest metrics
- artifact completeness
- reproducibility from saved config
- no transfer-learning overclaims

