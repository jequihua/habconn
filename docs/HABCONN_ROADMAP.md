# habconn Roadmap

## Stage 0: Framework Integration and Stabilization
- populate the template around the real package state
- encode milestone boundaries
- clarify current assumptions, risks, and review expectations

## Stage 1: Graphab Backend Acceleration and Validation
- introduce a backend abstraction for Graphab interaction
- keep the current CLI evaluator as the canonical exact reference path
- implement a Java service backend for exact incremental patch addition
- validate service outputs against the CLI path on the bundled example landscape
- explicitly defer incremental resistance updates to a later stage

## Stage 2: Minimal Vector Habitat Environment
- implement the first real Gymnasium-compatible environment
- define canonical v1 observation and reward contracts
- replace placeholder environment tests with real tests

## Stage 3: Minimal Trainable Baseline
- add feature packing needed for training
- connect the environment to a first masked-action training loop
- run a small reproducible baseline

## Stage 3.5: Richer Features and Evaluation (within single-landscape scope)
- replace placeholder feature builders with real logic
- enrich the observation contract without destabilizing the env
- upgrade the baseline extractor to consume richer features
- produce richer training/evaluation artifacts (history JSONL, structured eval summary)

## Stage 4: Single-Landscape DRL Restoration Planner
- pause transfer-learning work as the immediate target
- make `small_vector_001` a complete real DRL training example
- define experiment config and output layout
- add worker-safe vectorized environment training
- evaluate trained policies against simple baselines
- checkpoint models and select the best model by an explicit metric
- deploy the selected model to export a final restoration plan

## Stage 5: Transfer Learning, Scaling, and Sophistication
- move beyond a single bundled landscape
- define proper landscape-level splits
- evaluate transfer to unseen landscapes
- improve config and runtime portability
- prepare for HPC-safe orchestration
- add caching, profiling, and richer encoders (set/graph encoders consuming node-level arrays)

## Usability And HPC Training Readiness Track
- make the completed single-landscape workflow easy for a human to inspect
- provide notebooks that exercise current functionality one piece at a time
- turn the existing fixed smoke-test script / inline HPC snippets into a user-facing training entry point
- add TensorBoard logging for longer training runs
- provide HPC run templates that keep scratch, outputs, and logs explicit
- keep transfer learning archived until real multi-landscape data work resumes

## Current active stage
Stages 0-3.5 are complete. Stage 4 (Single-Landscape DRL Restoration Planner) is complete end-to-end on `small_vector_001`. Stage 5 milestone 1 (landscape registry and split contract) is complete as foundation work only:

- **Stage 5 milestone 1 (this milestone): landscape registry and split contract** —
  `habconn.experiments.landscape_registry` ships `LandscapeSpec`,
  `LandscapeSplit`, `builtin_landscape_specs()`, `development_split()`,
  and validators. The built-in registry contains only
  `small_vector_001` (development fixture); the development split is
  explicitly **not** transfer-learning evidence.

Stage 4 sub-milestones 1–6 are complete:

1. experiment contract / config foundation,
2. worker-safe vectorized environment training (`DummyVecEnv` only),
3. single-landscape evaluation suite,
4. periodic checkpointing with explicit best-model selection,
5. deployment export of selected planning units (`deployment_summary.json`, GeoPackage, CSV),
6. feature inspection + deployment action trace (`observation_schema.json`, `feature_summary.json`, `deployment_action_trace.{json,csv}`).

Transfer-learning training and evaluation are archived/deferred for now. The active work is the **Human-Friendly Use And HPC Training Readiness** track.

The notebook/runbook scaffold is in place. The **single-landscape training rehearsal runner** is now implemented:

- CLI: `08_pkg/habconn/scripts/run_experiment.py` (thin argparse wrapper around `ExperimentConfig` / `run_experiment`; presets `smoke` and `rehearsal`; explicit CLI flags override preset values; defaults to `smoke`).
- TensorBoard logging: off by default; enabled via `--tensorboard` (or `--tensorboard-log <path>`); default placement `output_root/run_name/tensorboard/`; surfaced in `baseline_summary.json` and `config.json`.
- Slurm template: `06_infra/templates/habconn_training_rehearsal.slurm` (calls the CLI; no inline training loop).
- Runbook: `06_infra/training_rehearsal.md` (smoke vs. rehearsal, local command, TensorBoard launch, HPC/Slurm usage, what to inspect, deferred-work list).
- Design-of-record: `06_infra/hpc_training.md` (now marked *superseded — implemented*).

The old HPC smoke-test runbook (`HPC_HABCONN_SMOKE_TEST.md`) remains the HPC compatibility gate; the rehearsal runner is the next step after that gate has passed.

The **publish-repo sync and handoff bundle** is now implemented:

- Sync tool: `scripts/python/sync_habconn_publish_repo.py`
  (manifest-driven, `--check` / `--apply` modes, refuses to write
  outside the target root or run against a dirty target without
  `--allow-dirty-target`).
- Manifest: `scripts/python/publish_habconn_sync_manifest.toml`
  (publishes the package, tests, scripts, configs,
  `data/examples/small_vector_001/`, `tools/`, notebooks, HPC smoke
  runbook, `06_infra/` rehearsal docs + Slurm template,
  `09_ops/runbooks.md`, key `docs/*.md`, and the new
  `docs/PUBLISH_REPO_HANDOFF.md`).
- Operator runbook: `06_infra/publish_repo_sync.md`.
- Tests: `tests/test_habconn_publish_sync.py` (21 cheap unit tests
  using fake dev/target git repos; no real publish-repo dependency).
- Status: `--check` previewed cleanly against
  `C:/Users/dev/work/tum/habconn`; `--apply` is pending explicit
  operator approval.

Still deferred on this track: checkpoint resume, `SubprocVecEnv`, hyperparameter search, reward normalization, multi-landscape data, transfer learning.
