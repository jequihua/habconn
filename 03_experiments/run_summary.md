# Run Summary

## Configuration
- Active package: `08_pkg/habconn`
- Development fixture: `small_vector_001` (single landscape)
- Backend: `CliExactBackend` (canonical exact reference)
- Environment: `VectorHabitatEnv` (v2 observation + delta-PC reward)
- Training: `MaskablePPO` from sb3-contrib with `FlatObsExtractor` (v2)

## Baseline training results

| Parameter | Value |
|-----------|-------|
| Total timesteps | 50 |
| Training episodes logged | 18 |
| Budget | 3 |
| K (candidate set size) | 10 |
| Seed | 42 |
| Eval episode return | 4.79e-06 |
| Eval episode steps | 3 |
| Eval baseline PC | 2.173e-05 |
| Eval final PC | 2.652e-05 |
| Eval delta PC | 4.79e-06 |
| Eval selected PUs | [3, 4, 5] |

## V2 observation contract (14 keys)

### Action-level (K,)
| Key | dtype | Description |
|-----|-------|-------------|
| action_mask | bool | Valid candidate slots |
| candidate_ids | int32 | Planning-unit ID per slot (-1 = pad) |
| candidate_costs | float32 | Cost per slot (0 = pad) |
| candidate_areas | float32 | Area per slot (0 = pad) |

### Node-level (N_max,)
| Key | dtype | Description |
|-----|-------|-------------|
| selected_mask | bool | Already-selected units |
| node_mask | bool | Real vs padded units |
| node_costs | float32 | Cost per unit (0 = pad) |
| node_areas | float32 | Area per unit (0 = pad) |
| eligibility_mask | bool | Eligible units |

### Global (1,)
| Key | dtype | Description |
|-----|-------|-------------|
| remaining_budget | float32 | Budget left |
| budget_fraction | float32 | remaining / initial |
| step_count | int32 | Current step |
| selected_fraction | float32 | n_selected / n_planning_units |
| current_pc | float32 | Most recent PC value |

## V2 extractor
- `FlatObsExtractor` concatenates action-level and global features only: 3K + 5 features
- Scalings: PC × 1e5, candidate_areas × 1e-4
- Node-level arrays are present in the observation but not consumed by this extractor (reserved for future encoders)

## Experiment contract (Stage 4 milestone 1)
Every run writes a self-contained directory under `output_root/run_name/`:

```
tmp/experiments/<run_name>/
    config.json              — resolved ExperimentConfig
    metadata.json            — timestamp, Python version, platform,
                                habconn version + path, git commit,
                                cheap dependency versions, key paths
    history.jsonl            — one record per completed training episode
    baseline_summary.json    — training + structured evaluation summary
    models/
        final_model.zip      — final SB3 model
```

`scripts/train_small_vector.py` runs under `run_name="baseline_small_vector_001"`
and writes to `08_pkg/habconn/tmp/experiments/baseline_small_vector_001/`.
The legacy `tmp/baseline_output/` layout still works when `BaselineConfig` is
used directly without explicit path overrides.

## Vectorized training (Stage 4 milestone 2)
- `n_envs` is part of `ExperimentConfig` and `BaselineConfig` (default 1, validated as a positive int).
- For `n_envs == 1` the trainer keeps the legacy single-env path so existing artifacts and metrics are unchanged.
- For `n_envs > 1` the trainer builds a `DummyVecEnv` of independent
  `VectorHabitatEnv` workers via `training/vecenv.py::make_vector_envs`.
  Each worker gets:
  - its own Graphab scratch directory at `<work_root>/worker_NNN/`,
  - a deterministic per-worker seed `(base_seed + worker_index) & 0x7FFFFFFF`,
  - a `Monitor` wrapper so `info["episode"]` flows into `EpisodeHistoryCallback`.
- Evaluation always runs on a separate single-env instance at
  `<work_root>/eval/` so per-step rewards and PC traces are unambiguous.
- `config.json` and `baseline_summary.json` now include `n_envs`,
  `vec_env_type` (`"DummyVecEnv"` for `n_envs > 1`, `null` for
  `n_envs == 1`), and `worker_work_roots`.
- `SubprocVecEnv` is intentionally deferred: `DummyVecEnv` is the safest
  first vectorized path on Windows and with a Graphab CLI subprocess.

## Evaluation suite (Stage 4 milestone 3)
- New module `training/baselines.py` provides three reproducible
  baselines that respect `env.action_masks()`:
  - `random_valid` — uniformly random valid slot, seeded per episode
    via `np.random.default_rng(base_seed + episode_index)`,
  - `lowest_cost` — argmin over `candidate_costs` masked by validity,
    tie-break to first valid slot,
  - `largest_area` — argmax over `candidate_areas` masked by validity,
    tie-break to first valid slot.
- `run_evaluation_comparison` reuses the existing trained-policy
  `EvalSummary` (no re-evaluation) and produces:
  - `<run_dir>/evaluation/comparison.json` — full per-method
    `EvalSummary` payloads + per-method means + run metadata
    (run_name/seed/budget/k/n_eval_episodes),
  - `<run_dir>/evaluation/comparison.csv` — compact method-level
    table: `method,n_episodes,mean_return,mean_final_pc,mean_delta_pc,mean_steps`.
- The trainer surfaces `evaluation_dir`, `comparison_json_path`,
  `comparison_csv_path`, and a small `comparison_method_means` block in
  the run summary; `run_experiment` mirrors the comparison paths at
  the experiment level so callers can locate every artifact from the
  returned summary.
- This is a **single-landscape** comparison. It is not an optimality
  proof, not a transfer-learning result, and not a replacement for
  scientific validation.

## Checkpointing and best-model selection (Stage 4 milestone 4)
- Periodic checkpointing is wired through a small `CheckpointCallback`
  in `training/checkpointing.py`. Checkpoints land under
  `<run_dir>/checkpoints/checkpoint_{num_timesteps:06d}_steps.zip`.
  The save cadence is controlled by `ExperimentConfig.checkpoint_freq`
  (default 16); the value counts SB3 callback invocations, so under
  a `DummyVecEnv` with `n_envs > 1` each invocation advances
  `num_timesteps` by `n_envs`. The saved filenames always reflect the
  actual `num_timesteps` at save time.
- After training, `run_checkpoint_selection` re-evaluates every
  checkpoint **and the final model** on the dedicated single-env eval
  scratch. Candidate evaluation is intentionally separate from the
  trained-policy summary in `baseline_summary.json` and from the
  baseline-comparison artifacts under `<run_dir>/evaluation/`; those
  remain reused / unchanged.
- The selected candidate is picked by `selection_metric` (whitelisted:
  `mean_final_pc`, `mean_delta_pc`, `mean_return`) under
  `selection_mode = "max"`. Tie-breaking is deterministic: among
  candidates with equal metric value, the larger `timestep` wins; if
  timesteps also tie, `candidate_type == "final"` wins over
  `"checkpoint"`. The rule is recorded verbatim in
  `selection/model_selection.json` as `tie_break_rule`.
- The selected model is copied to
  `<run_dir>/models/best_model.zip`. The earlier
  `<run_dir>/models/final_model.zip` artifact is preserved alongside.
- New artifacts written under the run directory:
  - `<run_dir>/checkpoints/checkpoint_NNNNNN_steps.zip` (one per save)
  - `<run_dir>/models/best_model.zip`
  - `<run_dir>/selection/checkpoint_evaluations.json` — one record per
    candidate (`candidate_id`, `model_path`, `candidate_type`,
    `timestep`, `evaluation`, `selected`)
  - `<run_dir>/selection/model_selection.json` —
    `selection_metric`, `selection_mode`, `tie_break_rule`,
    `selected_candidate_id`, `selected_model_path`, `best_model_path`,
    `selected_evaluation`, `all_candidate_ids`
- Run summary now also includes
  `selected_candidate_id` / `selected_candidate_type` /
  `selected_candidate_timestep` / `selected_evaluation` and
  `n_checkpoints_saved` so a reviewer can see selection state at a
  glance.
- This is **not** a deployment of the selected model. Loading
  `best_model.zip` to export selected planning units / GeoPackage
  artifacts remains the next milestone.

## Deployment export (Stage 4 milestone 5)
- `training/deployment.py::run_deployment_export` loads the selected
  `<run_dir>/models/best_model.zip`, runs one deterministic masked
  deployment episode against the dedicated single-env eval, and
  writes three artifacts under `<run_dir>/deployment/`:
  - `deployment_summary.json` — model path, model_selection_path,
    `selected_pu_ids` (in selection order), `n_selected`,
    `deployment_return`, `baseline_pc`, `final_pc`, `delta_pc_total`,
    `episode_steps`, `step_rewards`, `step_pc_values`, plus the
    artifact paths.
  - `selected_planning_units.gpkg` — GeoPackage of selected
    geometries with a `selection_order` column; empty selections
    fall back to an empty placeholder file.
  - `selected_planning_units.csv` — review-friendly per-row table
    (no geometry blob) with columns `internal_id_column`,
    `id_column` (when distinct), `cost_column`, `selection_order`.
- The deployment env is the dedicated single-env eval used by the
  evaluation comparison and the checkpoint selection step. It is
  reset before the deployment episode so prior eval state cannot
  leak in.
- The deployment path is explicitly tied to `best_model.zip` (the
  selected candidate). `final_model.zip` is preserved alongside; it
  is deployed only if it was also the selected best model.
- The trainer surfaces `deployment_dir`, `deployment_summary_path`,
  `selected_planning_units_gpkg_path`,
  `selected_planning_units_csv_path`,
  `deployment_selected_pu_ids`, `deployment_final_pc`, and
  `deployment_delta_pc_total` in the run summary; `run_experiment`
  mirrors the same paths at the experiment level.
- This is **single-landscape** deployment for `small_vector_001`.
  It is not transfer-learning evidence and not a scientific
  optimality proof.

## Feature inspection + deployment action trace (Stage 4 milestone 6)
- New module `training/inspection.py` produces four artifacts under
  `<run_dir>/inspection/` after deployment:
  - `observation_schema.json` — v2 observation contract (14 keys)
    with `group`, `shape`, `dtype`, `consumed_by_flat_extractor`,
    per-key notes, and a warning recording that node-level arrays
    are present but unused by the current `FlatObsExtractor`.
  - `feature_summary.json` — per-key stats over the initial
    deployment observation (`true_count`/`false_count` for bools,
    `finite_count`/`nan_count`/`min`/`max`/`mean` for numeric).
  - `deployment_action_trace.json` — one record per deployment
    step with `chosen_slot` / `chosen_pu_id` / `valid_action_count`
    / `remaining_budget_before` / `current_pc_before` /
    `reward_after` / `pc_after` / `selected_pu_ids_after` plus a
    `candidate_slots` list of per-slot
    `{slot, pu_id, valid, chosen, candidate_cost, candidate_area}`.
  - `deployment_action_trace.csv` — one row per (step, slot) for
    diff-friendly cross-run comparison.
- The trace is captured during the existing deployment episode, not
  in a second policy replay. `run_deployment_export` exposes the
  inspection side channel as underscore-prefixed keys
  (`_initial_observation`, `_trace_steps`); the on-disk
  `deployment_summary.json` is unchanged.
- The trainer surfaces `inspection_dir`,
  `observation_schema_path`, `feature_summary_path`,
  `deployment_action_trace_json_path`,
  `deployment_action_trace_csv_path`, plus
  `n_deployment_trace_steps` / `n_deployment_trace_rows` in the
  run summary; `run_experiment` mirrors the paths at the
  experiment level.
- This is **inspection**, not feature attribution. No SHAP, no
  permutation importance, no per-feature gradient analysis. The
  trace records what the policy saw and chose, not why.

## Reproducibility
- 287 tests pass in both repo-level `.venv` and local `08_pkg/habconn/.venv`
- `scripts/train_small_vector.py` runs in both venvs and produces identical
  evaluation metrics (mean return 4.792e-6, final PC 2.652311e-5,
  selected PUs [3, 4, 5]) before and after the contract refactor
- conftest.py guards test imports

## Decision
Richer-feature baseline milestone complete.

Course change: transfer-learning-ready work is paused as the immediate next
target. The active next phase is a complete single-landscape DRL workflow for
`small_vector_001`: experiment config, vectorized training, evaluation against
simple baselines, checkpoint/best-model selection, and deployment export of a
final restoration plan.

Stage 4 milestones 1–6 are complete: experiment contract, vectorized
env training, evaluation suite, checkpointing + best-model
selection, deployment export, and feature inspection + deployment
action trace. The single-landscape DRL workflow on `small_vector_001`
is end-to-end. Stage 5 (transfer learning + scaling + richer
encoders) is the next stage but is not yet active.
