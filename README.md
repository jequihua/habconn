# habconn

**habconn** is a Python package for **habitat connectivity optimization on landscapes using reinforcement learning**, with an explicit focus on **restoration planning under constrained budgets** and integration with **Graphab** as an external ecological connectivity engine.

The package is being designed from the ground up to support:

- **vector-based restoration planning first**
- future support for **raster-based action spaces**
- **transfer learning** across many landscapes of different sizes
- **Gymnasium** as the environment API
- **stable-baselines3** as the initial reinforcement learning framework
- **Graphab** as the exact external evaluator of connectivity
- clean migration from a simple first implementation toward more advanced graph/set-based learning architectures

This repository is intended to be the **clean refoundation** of a prior research codebase that became difficult to maintain due to many overlapping scripts, duplicated experiments, hard-coded paths, mixed responsibilities, and ad hoc environment variants.

---

# Table of contents

1. [Project summary](#project-summary)
2. [Problem setting](#problem-setting)
3. [Scientific objective](#scientific-objective)
4. [Optimization formulation](#optimization-formulation)
5. [Why this repository exists](#why-this-repository-exists)
6. [Design principles](#design-principles)
7. [Current scope of version 1](#current-scope-of-version-1)
8. [Planned future scope](#planned-future-scope)
9. [Core modeling choice: Option B candidate-based actions](#core-modeling-choice-option-b-candidate-based-actions)
10. [Transfer learning strategy](#transfer-learning-strategy)
11. [Observation design](#observation-design)
12. [Action design](#action-design)
13. [Why not use a simple fixed action space over all polygons](#why-not-use-a-simple-fixed-action-space-over-all-polygons)
14. [Why not use image-only observations as the main vector representation](#why-not-use-image-only-observations-as-the-main-vector-representation)
15. [Graphab integration strategy](#graphab-integration-strategy)
16. [HPC and parallel training strategy](#hpc-and-parallel-training-strategy)
17. [Package architecture](#package-architecture)
18. [Directory structure](#directory-structure)
19. [Key abstractions](#key-abstractions)
20. [Development roadmap](#development-roadmap)
21. [Immediate milestones](#immediate-milestones)
22. [Testing philosophy](#testing-philosophy)
23. [Configuration philosophy](#configuration-philosophy)
24. [Data expectations](#data-expectations)
25. [How a typical experiment is expected to work](#how-a-typical-experiment-is-expected-to-work)
26. [Planned model evolution](#planned-model-evolution)
27. [Important non-goals for v1](#important-non-goals-for-v1)
28. [How to continue this project in a new session](#how-to-continue-this-project-in-a-new-session)
29. [Implementation order](#implementation-order)
30. [Repository status](#repository-status)

---

# Project summary

This project aims to build a rigorous, extensible Python package for solving **landscape restoration planning problems** where the goal is to maximize a **habitat connectivity metric** under a **limited restoration budget**.

The package is centered on the following idea:

- a landscape has a set of **candidate restoration planning units**
- each planning unit has a **cost**
- selecting a planning unit modifies the landscape
- the modified landscape is evaluated with a connectivity engine, initially **Graphab**
- the reinforcement learning agent learns a policy for selecting restoration actions that improve connectivity efficiently

The initial implementation targets a setting in which:

- the **actions occur on vector planning units** such as polygons
- the landscape also includes a **habitat raster**
- the landscape also includes an **animal movement resistance raster**
- many training landscapes may exist
- each landscape is small in spatial extent, but different landscapes may contain different numbers of polygons and patches
- transfer learning across landscapes is a major objective

---

# Problem setting

The original research context behind this package is landscape ecology and ecological restoration planning.

A typical instance contains:

- a vector layer of **candidate polygons** that may be restored
- a habitat raster describing existing habitat or habitat suitability
- a resistance raster describing movement resistance for an animal species of interest
- a restoration budget
- a connectivity objective, often derived from Graphab outputs

A restoration policy should choose which candidate planning units to restore, subject to budget constraints, in order to maximize the ecological connectivity objective.

The original codebase contained many experiment-specific scripts, variants of custom Gymnasium environments, Graphab wrappers, training scripts, deployment scripts, and utility modules. The current repository is a redesign intended to preserve the scientific goal while rebuilding the software architecture.

---

# Scientific objective

The scientific objective is to support ecological decision-making where restoration resources are limited.

Examples:

- restore only 5 polygons out of 30 candidate polygons
- restore only those units whose total cost does not exceed a given budget
- prioritize units that most improve overall connectivity
- support repeated experiments across many landscapes
- learn policies that can generalize to unseen landscapes

This is not just a one-instance optimization tool. It is being designed explicitly to support **learning across many small landscapes**.

---

# Optimization formulation

Conceptually, the optimization problem is:

- **input**:
  - landscape data
  - candidate planning units
  - costs
  - habitat and resistance representation
  - connectivity settings
  - budget constraint

- **state**:
  - current selected/restored planning units
  - remaining budget
  - step count
  - cached objective values
  - candidate action set

- **action**:
  - choose one planning unit from a bounded candidate set

- **transition**:
  - update the selected set
  - update budget
  - update eligibility and masks
  - compute the new connectivity score

- **reward**:
  - typically a function of connectivity improvement, optionally cost-aware and shaped carefully

- **termination**:
  - budget exhausted
  - no valid candidate actions remain
  - maximum step limit reached
  - optional early stop conditions

---

# Why this repository exists

The previous research repository was functional but had become structurally difficult to extend and maintain. It contained:

- multiple environment variants for different experiments
- overlapping training and deployment scripts
- mixed concerns between environment logic, GIS processing, plotting, file writing, and Graphab calls
- hard-coded paths
- unclear separation between reusable code and experiment code
- inconsistent masking and action handling
- code specialized to one project or one dataset layout

This repository exists to replace that approach with a package that is:

- modular
- testable
- scalable
- easier to reason about
- suitable for local development and HPC execution
- ready for transfer learning research

---

# Design principles

This project follows these principles:

## 1. Separate concerns aggressively
Do not mix:

- data loading
- state transitions
- connectivity evaluation
- Gymnasium environment logic
- neural network modeling
- experiment scripts

Each concern should live in its own module layer.

## 2. Vector-first design
The first version targets restoration actions on vector planning units.

Raster-action optimization is planned for later, but placeholders should exist now so the future extension is natural rather than a hack.

## 3. Transfer learning is a first-class requirement
The package must support landscapes with different numbers of candidate polygons and habitat patches.

This rules out a naive action design in which the action space is directly tied to the raw number of polygons in a single landscape.

## 4. Swappable model encoders
Version 1 should use a simple and stable representation based on:

- padding
- masking
- fixed-shape observations

But the code should be written so a future graph or set encoder can be dropped in later.

## 5. Graphab is an external evaluator, not the core architecture
Graphab should be wrapped behind a clean interface.

The rest of the package should not depend on Graphab-specific command-line details.

## 6. HPC readiness from the beginning
Even if early experiments are local, the package should be designed so later training on a single HPC node with many cores is safe and efficient.

## 7. Start simple, but do not block future sophistication
The first version should not attempt to solve every future need immediately. It should build clean seams for later upgrades.

---

# Current scope of version 1

Version 1 is intentionally narrow.

It is intended to support:

- vector-based restoration actions
- small landscapes
- many landscape instances
- habitat raster input
- resistance raster input
- exact connectivity evaluation through Graphab
- candidate-based action selection
- action masking
- padded observations
- Gymnasium environment
- stable-baselines3 training with masked actions

The point of version 1 is **not** to be the final most sophisticated system. The point is to build a stable and correct foundation.

---

# Planned future scope

After version 1 is stable, future work may include:

- raster-action optimization
- superpixel or patch proposal systems for raster actions
- set encoders
- graph neural networks
- attention-based policies
- surrogate connectivity evaluators
- multi-fidelity reward computation
- improved Graphab caching
- richer experiment management
- larger training corpora
- more advanced deployment pipelines

---

# Core modeling choice: Option B candidate-based actions

A central design choice in this package is the use of **candidate-based fixed-size action spaces**.

This was chosen because one landscape may have 10 planning units, another may have 20, another may have 200, and yet another may have far more. A reinforcement learning policy with a raw `Discrete(n_units)` head tied to each landscape would not support transfer learning well.

Instead, the package uses the following idea:

1. at each step, the environment constructs a **candidate set** of up to `K` planning units
2. the policy action space is `Discrete(K)`
3. each action slot refers to one candidate in the current candidate set
4. if fewer than `K` candidates are available, the remaining slots are masked out

Important clarification:

- `K` is **not** the total number of polygons in the landscape
- `K` is the maximum number of candidates shown to the policy at one step

This means the same policy can be applied to landscapes with different numbers of polygons.

---

# Transfer learning strategy

Transfer learning is a major reason for the architecture of this package.

The package is designed so the agent learns over **landscape states represented as collections of planning-unit features**, not as a fixed landscape-specific ID system.

The transfer strategy has several components:

## 1. Bounded candidate actions
The action head remains fixed-size across landscapes.

## 2. Feature-based representation
Planning units are represented through features such as cost, local topology, habitat adjacency, status flags, and connectivity proxies.

## 3. Padding and masking
Landscapes with different numbers of planning units are represented in fixed-shape tensors by padding unused rows and masking them out.

## 4. Normalized features
Where possible, features should be represented in normalized or relative terms rather than absolute IDs or arbitrary file-order information.

## 5. Encoder modularity
The observation contract should support a future transition from a simple padded MLP encoder to a graph/set encoder.

---

# Observation design

The observation design for version 1 is expected to be a **dictionary observation** rather than one giant flat array.

Conceptually, the observation may contain:

- `node_features`
- `node_mask`
- `candidate_features`
- `candidate_mask`
- `global_features`
- optional topology or adjacency summaries

## Why use a dictionary observation
A dictionary preserves semantics and makes it easier to later swap in more sophisticated feature extractors.

For example:

- version 1 may use only node features and global features
- later versions may use adjacency structure more explicitly
- the rest of the package should not need to change just because the encoder becomes more advanced

## Variable-size landscapes
Landscapes may have different numbers of planning units or habitat patches.

To handle this in SB3-compatible training, observations will be packed into fixed-shape tensors using padding. Masks will indicate which rows are real.

This solves the shape problem while preserving extensibility.

---

# Action design

The action design in v1 is based on **Option B**.

## Candidate generation
At each step, a candidate generator produces up to `K` feasible planning units.

The candidate generator may use simple heuristics initially, such as:

- low cost
- adjacency to current habitat
- local topology summaries
- connectivity proxy values
- diversity among candidates
- a small number of random exploratory slots

The candidate generator is not the final optimizer. It simply determines which actions are currently presented to the policy.

## Action mask
If only `m < K` candidates are feasible, then:

- the first `m` candidate slots are valid
- the remaining `K - m` slots are invalid and masked

## Why this is useful
This provides a fixed-size policy interface while still allowing application to landscapes of different sizes.

---

# Why not use a simple fixed action space over all polygons

A naive design would set:

- `action_space = Discrete(n_polygons)`

This has serious drawbacks for transfer learning:

- different landscapes have different numbers of polygons
- policy dimensions become landscape-specific
- checkpoints become tied to one action shape
- padding all polygons directly into a huge universal action head may become inefficient and noisy

For those reasons, the package does not use this as the primary design.

A broader padded observation over all units may still exist, but the actual action head is planned to be candidate-based.

---

# Why not use image-only observations as the main vector representation

The previous research code used a 3-band image-like observation and a convolutional network.

That can be useful in some settings, especially for raster-based actions. However, for vector-based restoration planning it has drawbacks:

- it obscures the natural planning-unit structure
- it makes variable numbers of polygons awkward
- it can encourage learning tied to rasterization artifacts rather than planning-unit semantics
- it does not align naturally with the candidate-action formulation

For that reason, image-only observations are not the primary vector representation in this package.

Raster observations remain a future possibility for raster-action environments.

---

# Graphab integration strategy

Graphab is an external Java application for habitat network analysis and connectivity metrics.

In this package, Graphab is treated as an **external evaluator** that computes exact ecological connectivity outputs.

## Principles for Graphab integration

### 1. Encapsulate Graphab behind a runner
A low-level runner should manage:

- command construction
- subprocess execution
- stdout/stderr handling
- JVM settings
- local or HPC path configuration
- scratch directory handling

### 2. Build a high-level evaluator on top
A higher-level evaluator should expose methods like:

- evaluate baseline landscape
- evaluate a landscape after a selected set of restoration actions
- return objective values and diagnostics

### 3. Avoid polluting environment logic with subprocess details
The Gymnasium environment should not know how Graphab command lines are assembled.

### 4. Prepare for caching
Connectivity evaluation may be expensive. The design should allow future caching wrappers.

### 5. Respect parallel safety
When using multiple environment workers, Graphab calls must not collide on project names, output files, or scratch locations.

---

# HPC and parallel training strategy

Training is expected to happen locally at first, and later on a single HPC node using multiple cores.

The design must support this safely.

## Key constraints

- Graphab is a command-line Java application
- environment steps may trigger external evaluation
- multiple workers cannot share the same scratch files or project names
- nested oversubscription can occur if Python processes and Graphab threads are both scaled naively

## Strategy

### 1. One worker, one scratch space
Each environment worker must have isolated temporary directories and isolated Graphab project/output namespaces.

### 2. Controlled threading
The package should allow explicit control over:

- number of environment workers
- Graphab thread count per call
- JVM memory allocation

### 3. Prefer process isolation
Parallel environments should run in separate processes, not threads, unless there is a very strong reason otherwise.

### 4. Benchmark carefully
There is no single universally correct setting. Small local benchmarks should determine:

- whether more workers or more Graphab threads give better throughput
- whether Graphab should usually run single-threaded inside each worker
- how much per-worker scratch overhead exists

---

# Package architecture

The project is organized into several layers.

## `problems/`
Defines static optimization problem objects.

## `state/`
Defines dynamic episode state and state transitions.

## `features/`
Builds padded observations and masks.

## `evaluators/`
Computes connectivity through Graphab or future alternatives.

## `envs/`
Wraps the simulator into Gymnasium.

## `models/`
Contains custom feature extractors and policy-side components.

## `training/`
Contains SB3-facing training and evaluation orchestration.

## `io/`
Handles reading vectors, rasters, and Graphab-related input material.

## `config/`
Loads and validates experiment configuration.

## `utils/`
Provides general utilities such as logging, profiling, seeding, and temporary directories.

This architecture is meant to prevent the package from collapsing into one giant environment file.

---

# Directory structure

A simplified view of the directory structure is:

```text
habconn/
    pyproject.toml
    README.md
    configs/
    data/
    docs/
    notebooks/
    scripts/
    src/habconn/
        config/
        envs/
        evaluators/
        experiments/
        features/
        io/
        models/
        problems/
        state/
        training/
        types/
        utils/
    tests/
    tools/