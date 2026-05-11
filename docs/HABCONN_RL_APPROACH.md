# habconn Reinforcement Learning Approach

This document describes the current reinforcement learning formulation used by
`habconn` for the bundled `small_vector_001` landscape.

It reflects the current implementation, not an idealized future design. The
current workflow is a complete single-landscape masked-action DRL example:
load data, build a Gymnasium environment, train MaskablePPO, evaluate, select
the best checkpoint/final model, deploy it, and write reviewable artifacts.

## 1. Current Scope

The current RL system solves a **single-landscape habitat restoration planning
problem**.

The bundled landscape is:

```text
08_pkg/habconn/data/examples/small_vector_001/
    candidates.shp
    habitat.tif
    resistance.tif
```

The agent chooses restoration planning units under a fixed budget. Each chosen
planning unit is burned into the habitat/resistance representation and evaluated
with Graphab's Probability of Connectivity (PC) metric.

Current non-goals:

- no transfer-learning evidence,
- no second registered landscape,
- no graph/set encoder,
- no reward normalization,
- no `SubprocVecEnv`,
- no checkpoint resume,
- no scientific optimality claim.

## 2. Input Data

### Vector Planning Units

The action candidates come from `candidates.shp`.

For `small_vector_001`, the shapefile currently contains:

- 79 candidate polygon features,
- source identifier column: `lyr_1`,
- area column: `area`,
- additional patch/connectivity attributes that are present but not yet part of
  the canonical feature contract.

During problem loading, `VectorConnectivityProblem.from_files(...)` adds:

- `pu_id`: internal sequential planning-unit identifier,
- `cost`: restoration cost,
- `eligible`: boolean eligibility flag.

In the current factory path (`make_env`), costs are uniform:

```text
uniform_cost = 1.0
```

So with `budget=3`, the deployed policy can usually select three planning
units unless feasibility ends earlier.

### Raster Inputs

The raster inputs are:

- `habitat.tif`: baseline habitat raster,
- `resistance.tif`: baseline resistance raster.

The loader verifies that habitat and resistance rasters have matching:

- CRS,
- transform,
- width/height.

For `small_vector_001`, both rasters are currently:

```text
shape: 165 x 210
dtype: int16
nodata: -32768
```

If vector CRS differs from raster CRS, the vector candidates are reprojected to
the raster CRS during loading.

### Restoration Semantics

Selecting a planning unit means:

- the selected polygon becomes habitat,
- the selected polygon receives a restored resistance value,
- the new landscape state is evaluated through Graphab.

The current training path uses the CLI exact backend as the scientific
reference evaluator.

## 3. Static Problem Object

The static problem is represented by:

```python
habconn.problems.vector_problem.VectorConnectivityProblem
```

It owns:

- input paths,
- planning-unit GeoDataFrame,
- ID/cost/eligibility columns,
- raster metadata,
- restored resistance semantics,
- helper methods such as `get_cost(...)`, `get_planning_unit_row(...)`, and
  `selected_geodataframe(...)`.

The problem object is loaded once when the environment is created. Dynamic
episode state lives separately in `LandscapeState`.

## 4. Dynamic Episode State

The current state object is:

```python
habconn.state.landscape_state.LandscapeState
```

It tracks:

- `remaining_budget`,
- `selected_pu_ids`,
- `step_count`,
- `cached_pc_value`,
- `done`,
- misc `info`.

A planning unit is selectable only if:

- the episode is not already done,
- it has not already been selected,
- it is eligible,
- its cost is less than or equal to remaining budget.

After a valid action, the state:

- appends the selected `pu_id`,
- subtracts its cost from `remaining_budget`,
- increments `step_count`,
- marks `done=True` if no feasible actions remain.

## 5. Environment

The Gymnasium environment is:

```python
habconn.envs.vector_env.VectorHabitatEnv
```

The canonical factory is:

```python
habconn.training.make_env.make_env
```

The environment connects four layers:

- `problems/`: static landscape definition,
- `state/`: selected units, budget, feasibility,
- `evaluators/`: Graphab PC evaluation,
- `features/`: observation packing.

## 6. Action Space

The action space is:

```python
gymnasium.spaces.Discrete(K)
```

With the default small workflow:

```text
K = 10
```

The action is **not** a global planning-unit ID. It is a slot index into the
current candidate set:

```text
action 0 -> current candidate slot 0
action 1 -> current candidate slot 1
...
action K-1 -> current candidate slot K-1
```

The mapping changes at every step because selected or unaffordable planning
units leave the feasible set.

### Why A Fixed-K Candidate Space?

The number of planning units can vary by landscape. A policy head with one
logit per planning unit would be tied to a fixed landscape size. The current
design instead gives the policy a fixed `Discrete(K)` action space and proposes
up to K feasible candidates each step.

This keeps the first trainable baseline simple:

- fixed policy output size,
- compatible with MaskablePPO,
- easy to pad invalid slots,
- leaves room for future candidate generators or graph/set encoders.

### Candidate Generation

Candidate generation currently:

1. finds feasible planning units,
2. ranks them,
3. keeps the first K,
4. pads to length K with `-1`.

Supported ranking strategies include:

- `BY_PU_ID` (current default),
- `LOWEST_COST_FIRST`,
- `HIGHEST_AREA_FIRST`,
- `RANDOM`.

The default training path uses `BY_PU_ID`, so candidate order is deterministic.

### Action Masks

The environment exposes:

```python
env.action_masks()
```

It returns a boolean array of shape `(K,)`:

```text
True  = valid selectable slot
False = padded or invalid slot
```

`MaskablePPO` uses this mask before action selection so the policy should not
choose padded/invalid slots during training, evaluation, or deployment.

If an invalid slot is nevertheless passed to `env.step(...)`, the environment
fails fast:

- returns zero reward,
- returns `terminated=True`,
- marks internal state as done,
- returns an all-padded terminal observation.

This is intentional development safety. Invalid actions should be visible
immediately, not silently ignored.

## 7. Observation Space

The observation is a Gymnasium `Dict` with 14 keys. The current contract is
called the **v2 observation**.

It has three groups:

- action-level arrays of shape `(K,)`,
- node-level arrays of shape `(N_max,)`,
- global scalar arrays of shape `(1,)`.

For `small_vector_001`, `N_max` defaults to the number of planning units,
currently 79.

### Observation Keys

| Key | Group | Shape | Dtype | Meaning | Consumed by current `FlatObsExtractor` |
| --- | --- | --- | --- | --- | --- |
| `action_mask` | action | `(K,)` | bool | valid candidate slots | yes |
| `candidate_ids` | action | `(K,)` | int32 | `pu_id` per slot, `-1` for padding | no |
| `candidate_costs` | action | `(K,)` | float32 | restoration cost per slot, `0` for padding | yes |
| `candidate_areas` | action | `(K,)` | float32 | polygon area per slot, `0` for padding | yes |
| `selected_mask` | node | `(N_max,)` | bool | already selected planning units | no |
| `node_mask` | node | `(N_max,)` | bool | real planning-unit slots vs padding | no |
| `node_costs` | node | `(N_max,)` | float32 | cost per planning unit | no |
| `node_areas` | node | `(N_max,)` | float32 | area per planning unit | no |
| `eligibility_mask` | node | `(N_max,)` | bool | eligible planning units | no |
| `remaining_budget` | global | `(1,)` | float32 | budget after prior actions | yes |
| `budget_fraction` | global | `(1,)` | float32 | remaining budget / initial budget | yes |
| `step_count` | global | `(1,)` | int32 | number of valid actions taken | yes |
| `selected_fraction` | global | `(1,)` | float32 | selected units / total units | yes |
| `current_pc` | global | `(1,)` | float32 | latest Graphab PC value | yes |

### Why Include Node-Level Arrays If The Extractor Does Not Use Them?

The current policy uses a flat MLP extractor. It consumes only action-level and
global features. The node-level arrays are present to make the observation
contract future-proof for set/graph encoders, but they are not used by the
current model.

This is recorded honestly in the inspection artifact:

```text
inspection/observation_schema.json
```

## 8. Current Feature Extractor

The current extractor is:

```python
habconn.models.extractors.padded_mlp.FlatObsExtractor
```

It concatenates:

```text
action_mask        (K)
candidate_costs    (K)
candidate_areas    (K), scaled by 1e-4
remaining_budget   (1)
budget_fraction    (1)
step_count         (1)
selected_fraction  (1)
current_pc         (1), scaled by 1e5
```

Total feature dimension:

```text
3K + 5
```

With `K=10`, this is:

```text
35 features
```

Scaling choices:

- PC is typically around `1e-5`, so `current_pc * 1e5` puts it near order 1.
- Polygon areas are around `1e4`, so `candidate_areas * 1e-4` puts them near
  order 1.

This is a deliberately simple baseline extractor. It is not a final ecological
or spatial representation.

## 9. Reset Semantics

Calling:

```python
obs, info = env.reset(seed=...)
```

does the following:

1. initializes `LandscapeState` with no selected planning units,
2. sets the restoration budget,
3. evaluates the baseline landscape with Graphab,
4. stores the baseline/current PC value,
5. generates the first fixed-K candidate set,
6. returns the v2 observation and reset info.

Reset `info` includes:

- `pc_value`,
- `selected_pu_ids`,
- `step_count`,
- `remaining_budget`,
- `n_feasible`,
- `backend_type`.

The reset evaluation is important: reward is defined as a delta from the most
recent PC value, so the environment needs a baseline PC before the first action.

## 10. Step Semantics

Calling:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

does the following for a valid action:

1. maps the action slot to a planning-unit ID,
2. applies the restoration action to state,
3. evaluates the new landscape with Graphab,
4. computes raw delta-PC reward,
5. checks whether the episode is done,
6. generates the next candidate set or terminal padded candidates,
7. returns the next observation.

Step `info` includes:

- `pc_value`,
- `pc_before`,
- `delta_pc`,
- `selected_pu_ids`,
- `last_pu_id`,
- `step_count`,
- `remaining_budget`,
- `backend_type`,
- `action_type`,
- `n_feasible`.

The environment currently never sets `truncated=True`; time-limit truncation is
not part of the current contract.

## 11. Reward

The reward is raw delta-PC:

```text
reward = pc_after - pc_before
```

where PC is Graphab's Probability of Connectivity metric.

This choice is intentional:

- it keeps the RL objective tied directly to the ecological metric,
- it avoids premature reward shaping,
- it preserves interpretability in evaluation and deployment traces.

The magnitude is small, often around `1e-6` per step on the bundled example.
The current MaskablePPO baseline relies on PPO advantage normalization rather
than changing the environment reward.

Reward normalization experiments are explicitly deferred.

## 12. Termination Criteria

An episode terminates when one of these happens:

1. **No feasible planning units remain.**
   This can happen because the budget is exhausted, all affordable units are
   selected, or remaining units are ineligible.
2. **An invalid/padded action is passed to `step`.**
   This is a fail-fast development path and should not happen when masks are
   respected.

There is no separate max-step truncation in the current environment. In
practice, with uniform cost 1.0 and `budget=3`, the standard small run usually
ends after three valid selections.

## 13. Training Algorithm

Training uses:

```python
sb3_contrib.MaskablePPO
```

with:

```text
policy: MultiInputPolicy
features extractor: FlatObsExtractor
```

The action mask comes from `env.action_masks()`.

The default tiny training settings are intentionally small:

```text
total_timesteps = 50
learning_rate = 3e-4
n_steps = 8
batch_size = 4
n_epochs = 2
gamma = 0.99
n_eval_episodes = 1
```

These settings are for smoke testing and demonstration, not a tuned training
recipe.

## 14. Vectorized Training

`n_envs` is configurable.

- `n_envs == 1`: uses the legacy single environment path.
- `n_envs > 1`: uses `DummyVecEnv`.

For vectorized runs, each worker gets:

```text
<work_root>/worker_NNN/
```

and a deterministic per-worker seed:

```text
(base_seed + worker_index) & 0x7FFFFFFF
```

Evaluation remains on a separate single environment under:

```text
<work_root>/eval/
```

This keeps evaluation traces unambiguous.

`SubprocVecEnv` is not implemented yet. The current vectorized mode is
worker-safe but still in-process and serial.

## 15. Evaluation, Selection, Deployment, Inspection

After training, the workflow writes a full run directory:

```text
output_root/run_name/
    config.json
    metadata.json
    history.jsonl
    baseline_summary.json
    evaluation/
        comparison.json
        comparison.csv
    checkpoints/
        checkpoint_NNNNNN_steps.zip
    selection/
        checkpoint_evaluations.json
        model_selection.json
    models/
        final_model.zip
        best_model.zip
    deployment/
        deployment_summary.json
        selected_planning_units.gpkg
        selected_planning_units.csv
    inspection/
        observation_schema.json
        feature_summary.json
        deployment_action_trace.json
        deployment_action_trace.csv
```

Evaluation compares the trained policy to:

- `random_valid`,
- `lowest_cost`,
- `largest_area`.

Checkpoint selection evaluates checkpoints plus the final model and copies the
selected candidate to:

```text
models/best_model.zip
```

Deployment loads `best_model.zip`, runs one deterministic masked episode, and
exports selected planning units.

Inspection writes:

- observation schema,
- feature summary,
- deployment action trace.

## 16. Current Limitations

The current RL approach is useful and end-to-end, but intentionally modest.

Known limitations:

- only `small_vector_001` is bundled,
- no transfer-learning evaluation,
- no held-out landscape split,
- exact Graphab CLI evaluation is slow,
- vectorized training uses `DummyVecEnv`, not true multi-process parallelism,
- no checkpoint resume,
- no reward normalization,
- no hyperparameter search,
- no graph/set encoder,
- node-level observation arrays are not consumed by the current policy,
- no scientific optimality claim.

## 17. Practical Mental Model

The current environment can be understood as:

```text
state = selected planning units + remaining budget + current PC

candidate generator:
    state -> up to K feasible planning units

policy:
    observation -> masked slot choice in Discrete(K)

environment:
    slot -> planning unit -> Graphab PC evaluation -> delta-PC reward
```

This gives a real DRL training loop while keeping the first implementation
small, inspectable, and honest about what it does and does not prove.
