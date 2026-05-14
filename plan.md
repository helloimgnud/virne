# HRL-AC Integration Plan: 2-Stage Hierarchical RL for VNE in Virne

## Table of Contents
1. [Overview & Goals](#1-overview--goals)
2. [Architecture Mapping: hrl_ac → virne](#2-architecture-mapping-hrl_ac--virne)
3. [File Structure](#3-file-structure)
4. [Phase-by-Phase Implementation](#4-phase-by-phase-implementation)
   - Phase 1: Environment (`OnlineEnv`)
   - Phase 2: Feature Constructor
   - Phase 3: Reward Calculator
   - Phase 4: Network / Policy (`net.py`)
   - Phase 5: Solver (`HrlAcSolver`)
   - Phase 6: Registration & Config
5. [Component Specifications](#5-component-specifications)
6. [Data Flow & Contract](#6-data-flow--contract)
7. [Key Differences from hrl_ac Source](#7-key-differences-from-hrl_ac-source)
8. [Testing Checklist](#8-testing-checklist)
9. [Risk Register](#9-risk-register)

---

## 1. Overview & Goals

### What We Are Building

A **2-stage hierarchical RL solver** for Virtual Network Embedding (VNE) inside the `virne` framework:

- **Upper Agent (RL):** A GNN-based PPO actor-critic that decides **once per VNR** whether to accept (`action=1`) or early-reject (`action=0`).
- **Lower Agent (Heuristic or pretrained RL solver):** Runs only on accepted VNRs. Produces a full `Solution` object (node slots + link paths).
- **Environment:** `SolutionStepRLEnv` in `virne/solver/learning/rl_core/online_rl_environment.py` — already designed for solution-level actions. No changes to this class.

### Design Constraints

| Constraint | Source |
|---|---|
| `step()` takes a `Solution` object as action | `SolutionStepRLEnv` contract |
| `early_rejection` flag set only by upper agent | `2stage_hierarchical_rl_vne.md` |
| Lower heuristic sets `result=False` on failure, never `early_rejection=True` | `2stage_hierarchical_rl_vne.md` |
| Reward must penalise accepted-but-failed VNRs to avoid always-accept degenerate policy | `2stage_hierarchical_rl_vne.md` |
| All virne solvers must be registered via `@SolverRegistry.register` | `virne/solver/base_solver.py` |
| Config passed as `DictConfig` (OmegaConf) | `virne/solver/base_solver.py` |
| Logger, Counter, Recorder are injected — never instantiated inside solver | `virne/solver/base_solver.py` |

---

## 2. Architecture Mapping: hrl_ac → virne

### Concept Mapping

| hrl_ac concept | virne equivalent | Notes |
|---|---|---|
| `hrl_ac/env.py :: OnlineEnv` | New `HrlAcOnlineEnv(SolutionStepRLEnv)` | Re-implements `get_observation`, `compute_reward`, `step` |
| `hrl_ac/env.py :: sub_solver` | `NRMRankSolver` or `GRCRankSolver` or `HrlRaSolver` | Chosen via config |
| `hrl_ac/net.py :: Encoder` | New `HrlAcEncoder` (identical logic) | Moved to `virne/solver/learning/rl_policy/hrl_ac_policy.py` |
| `hrl_ac/net.py :: Actor / Critic` | New `HrlAcActor`, `HrlAcCritic` | Same as source |
| `hrl_ac/net.py :: ActorCritic` | New `HrlAcActorCritic` | Registered via `ActorCriticRegistry` |
| `hrl_ac/hrl_ac_solver.py :: HrlAcSolver` | New `HrlAcSolver(OnlineAgent, PPOSolver)` | Uses virne's `PPOSolver` base, not hrl_ac's |
| `hrl_ac/hrl_ac_solver.py :: obs_as_tensor` | New `HrlAcTensorConvertor` | Placed in `rl_core/tensor_convertor.py` or inline |
| `hrl_ac solver :: compute_reward` | New `HrlAcRewardCalculator` | Registered in `rl_core/reward_calculator.py` |
| `hrl_ac env :: get_observation` | `HrlAcOnlineEnv.get_observation()` | Builds GNN-compatible dict obs |
| `hrl_ac env :: _get_p_net_obs` | Reuses `ObservationHandler` from `virne/solver/learning/obs_handler.py` | Same calls, same field names |
| `hrl_ac env :: _get_v_net_obs` | Reuses `ObservationHandler` | Same |
| `hrl_ac env :: _get_v_net_attrs_obs` | Inline in `HrlAcOnlineEnv` | Normalised lifetime |

### Class Hierarchy (virne)

```
BaseEnvironment
└── OnlineRLEnvBase  (virne/solver/learning/rl_core/online_rl_environment.py)
    └── SolutionStepRLEnv  ← USE AS-IS (no changes)
        └── HrlAcOnlineEnv  ← NEW

Solver
└── PPOSolver  (virne/solver/learning/rl_core/rl_solver.py)
    ├── OnlineAgent  (virne/solver/learning/rl_core/online_agent.py)
    └── HrlAcSolver  ← NEW  (multiple inheritance)
```

---

## 3. File Structure

### Files to CREATE

```
virne/solver/learning/
├── rl_policy/
│   └── hrl_ac_policy.py          # Encoder, Actor, Critic, ActorCritic (GNN-based)
│
├── rl_core/
│   ├── feature_constructor.py    # ADD: HrlAcVNetLevelFeatureConstructor (optional MLP path)
│   └── reward_calculator.py      # ADD: HrlAcRewardCalculator
│
└── reinforcement_learning/
    └── hrl_ac/
        ├── __init__.py
        ├── hrl_ac_solver.py      # HrlAcSolver class
        └── hrl_ac_env.py         # HrlAcOnlineEnv class
```

### Files to MODIFY

```
virne/solver/learning/reinforcement_learning/__init__.py
    → add: from .hrl_ac import *

virne/solver/learning/rl_policy/__init__.py
    → add: from .hrl_ac_policy import HrlAcActorCritic

virne/conf/solver/
    → add: hrl_ac.yaml  (config file for the new solver)
```

### Files NOT to touch

```
virne/solver/learning/rl_core/online_rl_environment.py  ← SolutionStepRLEnv used as-is
virne/solver/learning/rl_core/online_agent.py           ← OnlineAgent used as-is
virne/solver/learning/rl_core/rl_solver.py              ← PPOSolver used as-is
virne/core/solution.py                                  ← Solution used as-is
virne/core/environment.py                               ← BaseEnvironment used as-is
```

---

## 4. Phase-by-Phase Implementation

---

### Phase 1: Environment — `hrl_ac_env.py`

**File:** `virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_env.py`

#### 1.1 Class Declaration

```python
import copy
import numpy as np
from gym import spaces

from virne.core import Solution
from virne.solver.heuristic.node_rank import GRCRankSolver, NRMRankSolver
from virne.solver.learning.rl_core.online_rl_environment import SolutionStepRLEnv
from virne.solver.learning.obs_handler import ObservationHandler


class HrlAcOnlineEnv(SolutionStepRLEnv):
    """
    Online environment for 2-stage hierarchical RL admission control.

    Upper agent sees GNN-encoded (p_net, v_net) and outputs action ∈ {0=reject, 1=accept}.
    Lower sub-solver runs only on accepted VNRs and returns a full Solution.
    """
```

#### 1.2 `__init__` — Critical Details

```python
def __init__(self, p_net, v_net_simulator, controller, recorder, counter,
             logger, config, **kwargs):
    # Force k_searching=1 for the sub-solver to avoid expensive beam search
    kwargs_for_solver = copy.deepcopy(kwargs)
    kwargs_for_solver['k_searching'] = 1

    super().__init__(p_net, v_net_simulator, controller, recorder,
                     counter, logger, config, **kwargs_for_solver)

    # Binary action space: 0=reject, 1=accept
    self.action_space = spaces.Discrete(2)

    # Cache observation-space dimensions for policy construction
    max_v_nodes = self.v_net_simulator.v_sim_setting['v_net_size']['high']
    self.max_num_v_net_nodes = max_v_nodes

    # Observation handler (same as base class, but referenced explicitly)
    self.obs_handler = ObservationHandler()

    # Running reward trackers (mirrors hrl_ac source exactly)
    self.global_timestep_count = 0
    self.global_moving_average_reward = 0
    self.global_cumulative_reward = 0
    self.actual_cumulative_reward = 0

    # Sub-solver selection
    sub_solver_name = config.get('sub_solver_name', 'nrm_rank')
    kwargs_for_sub = copy.deepcopy(kwargs)
    kwargs_for_sub['verbose'] = 0
    self._build_sub_solver(sub_solver_name, kwargs_for_sub, config)
```

#### 1.3 `_build_sub_solver` — Sub-Solver Factory

```python
def _build_sub_solver(self, sub_solver_name, kwargs_for_sub, config):
    """
    Factory method: instantiates the correct lower-level solver.

    Supported options (via config key 'sub_solver_name'):
      'nrm_rank'  → NRMRankSolver   (fast, deterministic)
      'grc_rank'  → GRCRankSolver   (slightly slower, often better)
      'hrl_ra'    → HrlRaSolver     (pretrained RL lower agent)

    When sub_solver_name == 'hrl_ra', the config key
    'pretrained_subsolver_model_path' may point to a checkpoint.
    If the path is empty or None, the lower agent starts randomly initialised.
    """
    if sub_solver_name == 'nrm_rank':
        self.sub_solver = NRMRankSolver(
            self.controller, self.recorder, self.counter, self.logger,
            config, **kwargs_for_sub)

    elif sub_solver_name == 'grc_rank':
        self.sub_solver = GRCRankSolver(
            self.controller, self.recorder, self.counter, self.logger,
            config, **kwargs_for_sub)

    elif sub_solver_name == 'hrl_ra':
        # Import lazily to avoid circular dependency
        from virne.solver.learning.reinforcement_learning.hrl_ra import HrlRaSolver
        self.sub_solver = HrlRaSolver(
            self.controller, self.recorder, self.counter, self.logger,
            config, **kwargs_for_sub)
        pretrained_path = config.get('pretrained_subsolver_model_path', None)
        if pretrained_path:
            self.sub_solver.load_model(pretrained_path)
        self.sub_solver.eval()
    else:
        raise NotImplementedError(
            f'Unknown sub_solver_name: {sub_solver_name}. '
            f'Choose from: nrm_rank, grc_rank, hrl_ra')
```

#### 1.4 `step` — The Core Decision Gate

```python
def step(self, action):
    """
    Upper agent makes ONE binary decision per VNR.

    action=1 (accept): call sub_solver, pass its Solution to parent step()
    action=0 (reject): create a failed Solution with early_rejection=True,
                       pass to parent step()

    CRITICAL:
      - Only this method sets solution['early_rejection'] = True.
      - Sub-solver failures set result=False but NOT early_rejection.
      - parent SolutionStepRLEnv.step() handles recorder + transit_obs.
    """
    if action:  # accept
        instance = {'v_net': self.v_net, 'p_net': self.p_net}
        solution = self.sub_solver.solve(instance)
        # Guarantee: sub_solver NEVER sets early_rejection=True
        assert not solution.get('early_rejection', False), (
            "Sub-solver must not set early_rejection=True. "
            "That flag is reserved for the upper agent.")
    else:  # reject
        solution = Solution(self.v_net)
        solution['early_rejection'] = True
        solution['result'] = False

    # Delegate to SolutionStepRLEnv.step() which handles:
    #   recorder.count(), compute_reward(), transit_obs(), get_observation()
    return super().step(solution)
```

**Note on `super().step(solution)`:** `SolutionStepRLEnv.step()` (in `online_rl_environment.py`) already handles:
- Calling `self.rollback_for_failure(reason=failure_reason)` when `solution['result']` is False
- Calling `self.recorder.count()`
- Calling `self.compute_reward(record)` (which we override)
- Calling `self.transit_obs()`
- Returning `(obs, reward, done, info)`

#### 1.5 `compute_reward` — Shaped Reward

```python
def compute_reward(self, record):
    """
    Shaped reward to prevent degenerate policies:

    Early rejection (upper agent said NO):
        reward = 0.0   — neutral; capacity preserved, no penalty

    Accepted + heuristic SUCCESS:
        reward = weight * (revenue/benchmark) * r2c_ratio
        where weight = time_weight_a + time_weight_b

    Accepted + heuristic FAILURE:
        reward = -0.01 * num_v_nodes   — small penalty per node, signals wasted attempt

    The reward is then adjusted relative to the global running average
    (average-reward baseline from hrl_ac source) to improve variance reduction.
    """
    w_a = 1.0
    # Normalise lifetime to range [0,1] using simulator's lifetime scale
    lifetime_scale = self.v_net_simulator.v_sim_setting['lifetime']['scale']
    w_b = record.get('v_net_lifetime', self.v_net.lifetime) / lifetime_scale
    revenue_benchmark = 100.0

    if record.get('result', False):
        # Success: reward proportional to quality of embedding
        basic_reward = record['v_net_revenue'] / revenue_benchmark
        weight = w_a + w_b
        reward = weight * basic_reward * record['v_net_r2c_ratio']

    elif not record.get('result', False) and not record.get('early_rejection', False):
        # Accepted but failed: small per-node penalty
        basic_reward = self.v_net.total_resource_demand / revenue_benchmark
        reward = -0.01 * self.v_net.num_nodes

    else:
        # Early rejection: neutral
        reward = 0.0

    # ── Average-reward baseline (reduces variance, from hrl_ac) ──────────
    self.actual_cumulative_reward += reward
    self.v_net_reward += reward
    self.global_timestep_count += 1
    self.global_cumulative_reward += reward

    running_average = self.global_cumulative_reward / self.global_timestep_count
    adjusted_reward = reward - running_average

    self.extra_record_info.update({
        'actual_cumulative_reward': self.actual_cumulative_reward,
        'global_cumulative_reward': self.global_cumulative_reward,
        'running_avg_reward': running_average,
        'actual_reward': reward,
    })
    self.cumulative_reward += adjusted_reward
    return adjusted_reward
```

#### 1.6 `get_observation` — GNN-Compatible Dict

```python
def get_observation(self):
    """
    Returns a dict of numpy arrays suitable for GNN processing:
      p_net_x          : [num_p_nodes, p_node_feature_dim]
      p_net_edge_index : [2, num_p_links*2]
      p_net_edge_attr  : [num_p_links*2, p_link_feature_dim]
      v_net_attrs      : [1]   (normalised lifetime scalar)
      v_net_x          : [num_v_nodes, v_node_feature_dim + 1]
      v_net_edge_index : [2, num_v_links*2]
      v_net_edge_attr  : [num_v_links*2, v_link_feature_dim]

    v_net_x has the normalised lifetime appended to each node row
    (broadcast from the VNR-level scalar) — matches hrl_ac's padding trick.
    """
    p_net_obs = self._get_p_net_obs()
    v_net_obs = self._get_v_net_obs()
    v_net_attrs = self._get_v_net_attrs_obs()

    # Broadcast VNR-level scalar onto each v-node row
    padding = np.expand_dims(v_net_attrs, axis=0).repeat(
        v_net_obs['x'].shape[0], axis=0)
    v_net_obs['x'] = np.concatenate(
        (v_net_obs['x'], padding), axis=-1).astype(np.float32)

    return {
        'p_net_x':          p_net_obs['x'],
        'p_net_edge_index': p_net_obs['edge_index'],
        'p_net_edge_attr':  p_net_obs['edge_attr'],
        'v_net_attrs':      v_net_attrs,
        'v_net_x':          v_net_obs['x'],
        'v_net_edge_index': v_net_obs['edge_index'],
        'v_net_edge_attr':  v_net_obs['edge_attr'],
    }
```

#### 1.7 Private Observation Helpers

```python
def _get_p_net_obs(self):
    """
    Physical network observation.

    node features = [resource_attrs | degree | max_link_resource | sum_link_resource]
    Feature dims:
      resource_attrs       : num_node_resource_attrs (e.g. 1 for CPU-only)
      degree               : 1
      max_link_resource    : num_link_resource_attrs (e.g. 1 for BW-only)
      sum_link_resource    : num_link_resource_attrs
    Total node feature dim = 1 + 1 + 1 + 1 = 4  (standard CPU+BW setup)
    """
    node_data = self.obs_handler.get_node_attrs_obs(
        self.p_net, node_attr_types=['resource'],
        node_attr_benchmarks=self.node_attr_benchmarks)
    p_degree = self.obs_handler.get_node_degree_obs(
        self.p_net, self.degree_benchmark)
    p_link_max = self.obs_handler.get_link_aggr_attrs_obs(
        self.p_net, link_attr_types=['resource'], aggr='max',
        link_attr_benchmarks=self.link_attr_benchmarks)
    p_link_sum = self.obs_handler.get_link_aggr_attrs_obs(
        self.p_net, link_attr_types=['resource'], aggr='sum',
        link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
    node_data = np.concatenate(
        (node_data, p_degree, p_link_max, p_link_sum), axis=-1)

    edge_index = self.obs_handler.get_link_index_obs(self.p_net)
    link_data = self.obs_handler.get_link_attrs_obs(
        self.p_net, link_attr_types=['resource'],
        link_attr_benchmarks=self.link_attr_benchmarks)

    return {'x': node_data, 'edge_index': edge_index, 'edge_attr': link_data}


def _get_v_net_obs(self):
    """
    Virtual network observation (before lifetime padding).

    node features = [resource_attrs | degree | max_link_resource | sum_link_resource]
    Same structure as p_net but for the VNR being evaluated.
    """
    node_data = self.obs_handler.get_node_attrs_obs(
        self.v_net, node_attr_types=['resource'],
        node_attr_benchmarks=self.node_attr_benchmarks)
    v_degree = self.obs_handler.get_node_degree_obs(
        self.v_net, self.degree_benchmark)
    v_link_max = self.obs_handler.get_link_aggr_attrs_obs(
        self.v_net, link_attr_types=['resource'], aggr='max',
        link_attr_benchmarks=self.link_attr_benchmarks)
    v_link_sum = self.obs_handler.get_link_aggr_attrs_obs(
        self.v_net, link_attr_types=['resource'], aggr='sum',
        link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
    node_data = np.concatenate(
        (node_data, v_degree, v_link_max, v_link_sum), axis=-1)

    edge_index = self.obs_handler.get_link_index_obs(self.v_net)
    link_data = self.obs_handler.get_link_attrs_obs(
        self.v_net, link_attr_types=['resource'],
        link_attr_benchmarks=self.link_attr_benchmarks)

    return {'x': node_data, 'edge_index': edge_index, 'edge_attr': link_data}


def _get_v_net_attrs_obs(self):
    """
    VNR-level scalar features: currently just normalised lifetime.
    Returns: np.array of shape [1], dtype float32.
    """
    norm_lifetime = (self.v_net.lifetime /
                     self.v_net_simulator.v_sim_setting['lifetime']['scale'])
    return np.array([norm_lifetime], dtype=np.float32)


def generate_action_mask(self):
    """
    Both actions always available (upper agent always chooses).
    Returns np.ones(2, dtype=bool) as stub.

    Override to implement curriculum: during early training, mask reject
    action (action=0) by returning np.array([False, True]).
    """
    return np.ones(2, dtype=bool)
```

#### 1.8 `ready` — Reset Per-VNR Trackers

```python
def ready(self, event_id=0):
    """Called by transit_obs() before each new VNR."""
    super().ready(event_id)
    # v_net_reward is reset per-VNR by the base class
    # No additional per-VNR state needed here unless curriculum is added
```

---

### Phase 2: Feature Constructor

**File:** `virne/solver/learning/rl_core/feature_constructor.py` (add new class)

The HRL-AC solver uses GNN observations returned directly from `get_observation()`, so a separate `FeatureConstructorRegistry` entry is not strictly required. However, for consistency with the virne pattern, register a passthrough constructor:

```python
@FeatureConstructorRegistry.register('hrl_ac')
class HrlAcFeatureConstructor(BaseFeatureConstructor):
    """
    Feature constructor for HRL-AC solver.

    The actual feature construction happens inside HrlAcOnlineEnv.get_observation().
    This class simply validates the observation dict and returns it unchanged.
    It is registered so that the solver_maker and config system can reference it.
    """

    def get_observation(self, p_net, v_net, solution=None):
        # Not used: HrlAcOnlineEnv.get_observation() is called directly
        raise NotImplementedError(
            "HrlAcFeatureConstructor does not produce observations directly. "
            "Use HrlAcOnlineEnv.get_observation() instead.")

    @staticmethod
    def get_p_net_feature_dim(config):
        """
        Returns the p_net node feature dimension for policy construction.
        dim = num_node_resource_attrs + 1 (degree) + num_link_resource_attrs * 2
        Default for CPU+BW: 1 + 1 + 1 + 1 = 4
        """
        num_node_res = config.get('num_node_resource_attrs', 1)
        num_link_res = config.get('num_link_resource_attrs', 1)
        return num_node_res + 1 + num_link_res * 2

    @staticmethod
    def get_v_net_feature_dim(config):
        """
        Same structure as p_net PLUS the lifetime scalar appended.
        Default: 4 + 1 = 5
        """
        return HrlAcFeatureConstructor.get_p_net_feature_dim(config) + 1
```

**Why this matters:** `HrlAcSolver.__init__` needs `p_net_feature_dim` and `v_net_feature_dim` to construct the `ActorCritic` network. These are derived from config, not hardcoded.

---

### Phase 3: Reward Calculator

**File:** `virne/solver/learning/rl_core/reward_calculator.py` (add new class)

The reward logic lives in `HrlAcOnlineEnv.compute_reward()` (Phase 1.5). A separate `RewardCalculator` registration is needed only if the virne config system routes `reward_calculator.name` to a class. If that system exists:

```python
@RewardCalculatorRegistry.register('hrl_ac')
class HrlAcRewardCalculator(BaseRewardCalculator):
    """
    Shaped reward for 2-stage hierarchical admission control.

    early_rejection → 0.0  (neutral: no wasted resources, no penalty)
    accepted + success   → weight * (revenue/benchmark) * r2c_ratio
    accepted + failure   → -0.01 * num_v_nodes  (penalty for wasted resources)

    Adjusts by running global average for variance reduction.
    """
    def __init__(self, config):
        self.revenue_benchmark = config.get('revenue_benchmark', 100.0)
        self.failure_penalty_per_node = config.get('failure_penalty_per_node', 0.01)

    def compute(self, solution_record, v_net, v_net_simulator,
                global_timestep_count, global_cumulative_reward):
        lifetime_scale = v_net_simulator.v_sim_setting['lifetime']['scale']
        w_a = 1.0
        w_b = solution_record.get('v_net_lifetime', v_net.lifetime) / lifetime_scale

        if solution_record.get('result', False):
            basic = solution_record['v_net_revenue'] / self.revenue_benchmark
            reward = (w_a + w_b) * basic * solution_record['v_net_r2c_ratio']
        elif not solution_record.get('early_rejection', False):
            reward = -self.failure_penalty_per_node * v_net.num_nodes
        else:
            reward = 0.0

        # Average-reward baseline
        running_avg = global_cumulative_reward / max(global_timestep_count, 1)
        return reward - running_avg
```

If `RewardCalculatorRegistry` does not exist in the current virne codebase, implement the reward directly in `HrlAcOnlineEnv.compute_reward()` only (already done in Phase 1.5) and skip this file.

---

### Phase 4: Network / Policy — `hrl_ac_policy.py`

**File:** `virne/solver/learning/rl_policy/hrl_ac_policy.py`

This is a direct port of `hrl_ac/net.py` with virne import paths and `ActorCriticRegistry` registration.

#### 4.1 Imports

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch

# virne internal
from virne.solver.learning.rl_policy.net import GraphAttentionPooling, GraphPooling, MLPNet
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry
```

**Note:** `GraphAttentionPooling`, `GraphPooling`, and `MLPNet` already exist in `virne/solver/learning/rl_policy/net.py` (same as `hrl_ac/net.py`). Do NOT duplicate them. Import and reuse.

#### 4.2 `DeepEdgeFeatureGAT`

This class exists in both `hrl_ac/net.py` and `virne/solver/learning/net.py` (or equivalent). Check if it already exists in virne's `rl_policy/net.py`. If not, port it:

```python
class DeepEdgeFeatureGAT(nn.Module):
    """
    Multi-layer GAT with deep edge features, initial residual connections,
    and identity mapping (from hrl_ac/net.py, Zhao et al.).

    Architecture:
      conv_s  : input_dim → embedding_dim   (entry)
      conv_0..conv_{L-3}: embedding_dim → embedding_dim  (middle, with residual)
      conv_e  : embedding_dim → output_dim   (exit)

    Residual formula (layer l):
      x = (1-alpha) * conv(x) + alpha * x_0  (initial residual)
      x = (1-beta) * x + beta * (x @ W_l)    (identity mapping)
      beta = log(theta/(l+1) + 1)
    """
    def __init__(self, input_dim, output_dim, edge_dim,
                 num_layers=5, alpha=0.2, theta=0.2,
                 embedding_dim=128, num_heads=1,
                 batch_norm=False, dropout_prob=1.0,
                 return_batch=False, pooling=None):
        super().__init__()
        assert num_layers >= 2
        self.alpha = alpha
        self.theta = theta
        self.edge_dim = edge_dim
        self.num_mid_layers = num_layers - 2
        self.return_batch = return_batch
        self.pooling = pooling

        self.conv_s = GATConv(input_dim, embedding_dim,
                              heads=num_heads, edge_dim=edge_dim)
        for i in range(self.num_mid_layers):
            conv = GATConv(embedding_dim, embedding_dim,
                           heads=num_heads, edge_dim=edge_dim)
            norm = nn.BatchNorm1d(embedding_dim) if batch_norm else nn.Identity()
            dout = nn.Dropout(dropout_prob) if dropout_prob < 1.0 else nn.Identity()
            weight = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
            self.add_module(f'conv_{i}', conv)
            self.add_module(f'norm_{i}', norm)
            self.add_module(f'dout_{i}', dout)
            self.register_parameter(f'weight_{i}', weight)
        self.conv_e = GATConv(embedding_dim, output_dim,
                              heads=num_heads, edge_dim=edge_dim)
        self._init_parameters()

    def _init_parameters(self):
        for i in list(range(self.num_mid_layers)) + ['s', 'e']:
            nn.init.orthogonal_(getattr(self, f'conv_{i}').lin_src.weight)
            nn.init.orthogonal_(getattr(self, f'conv_{i}').lin_dst.weight)
            if self.edge_dim is not None:
                nn.init.orthogonal_(getattr(self, f'conv_{i}').lin_edge.weight)
            if i not in ['s', 'e']:
                nn.init.orthogonal_(getattr(self, f'weight_{i}'))

    def forward(self, input):
        x, edge_index, edge_attr = (input['x'], input['edge_index'],
                                    input.get('edge_attr', None))
        x_0 = self.conv_s(x, edge_index, edge_attr)
        x = x_0
        for i in range(self.num_mid_layers):
            conv = getattr(self, f'conv_{i}')
            norm = getattr(self, f'norm_{i}')
            dout = getattr(self, f'dout_{i}')
            weight = getattr(self, f'weight_{i}')
            conv_x = conv(x, edge_index, edge_attr)
            beta = math.log(self.theta / (i + 1) + 1)
            conv_x.mul_(1 - self.alpha)
            x = conv_x.add_(self.alpha * x_0)
            x = torch.addmm(x, x, weight, beta=1.0 - beta, alpha=beta)
            x = F.leaky_relu(dout(norm(x)))
        x = self.conv_e(x, edge_index, edge_attr)
        return x
```

#### 4.3 `HrlAcEncoder`

```python
class HrlAcEncoder(nn.Module):
    """
    Dual GNN encoder for (p_net, v_net) pair.

    Each network is encoded by DeepEdgeFeatureGAT.
    Global embeddings are computed by 3 pooling methods
    (attention GAP + mean + sum) and summed for each network.
    Final embedding = concat([p_global, v_global]) ∈ R^{2*embedding_dim}.
    """
    def __init__(self, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.p_net_gnn = DeepEdgeFeatureGAT(
            p_net_feature_dim, embedding_dim, num_layers=5,
            embedding_dim=embedding_dim, edge_dim=p_net_edge_dim,
            dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_gnn = DeepEdgeFeatureGAT(
            v_net_feature_dim, embedding_dim, num_layers=3,
            embedding_dim=embedding_dim, edge_dim=v_net_edge_dim,
            dropout_prob=dropout_prob, batch_norm=batch_norm)

        self.v_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_mean_pool = GraphPooling(aggr='mean')
        self.v_net_mean_pool = GraphPooling(aggr='mean')
        self.p_net_sum_pool  = GraphPooling(aggr='sum')
        self.v_net_sum_pool  = GraphPooling(aggr='sum')

    def forward(self, p_net_batch, v_net_batch):
        # v_net
        v_emb = self.v_net_gnn(v_net_batch)
        v_gap = self.v_net_gap(v_emb, v_net_batch.batch)
        v_mean = self.v_net_mean_pool(v_emb, v_net_batch.batch)
        v_sum  = self.v_net_sum_pool(v_emb, v_net_batch.batch)
        v_global = v_gap + v_mean + v_sum

        # p_net
        p_emb = self.p_net_gnn(p_net_batch)
        p_gap = self.p_net_gap(p_emb, p_net_batch.batch)
        p_mean = self.p_net_mean_pool(p_emb, p_net_batch.batch)
        p_sum  = self.p_net_sum_pool(p_emb, p_net_batch.batch)
        p_global = p_gap + p_mean + p_sum

        return torch.cat([p_global, v_global], dim=-1)  # [B, 2*embedding_dim]
```

#### 4.4 `HrlAcActor` and `HrlAcCritic`

```python
class HrlAcActor(nn.Module):
    """
    Actor: outputs logits for {reject=0, accept=1}.
    Input: fusion embedding R^{2*embedding_dim}
    Output: logits R^{2}
    """
    def __init__(self, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.encoder = HrlAcEncoder(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)
        self.net = MLPNet(
            embedding_dim * 2, 2,
            num_layers=3,
            embedding_dims=[embedding_dim * 2, embedding_dim],
            batch_norm=False)

    def act(self, obs):
        """obs: dict with 'p_net' (Batch) and 'v_net' (Batch)"""
        fusion = self.encoder(obs['p_net'], obs['v_net'])
        return self.net(fusion)  # logits [B, 2]


class HrlAcCritic(nn.Module):
    """
    Critic: outputs scalar state-value estimate.
    Uses its own independent encoder (not shared with actor).
    """
    def __init__(self, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.encoder = HrlAcEncoder(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)
        self.net = MLPNet(
            embedding_dim * 2, 1,
            num_layers=3,
            embedding_dims=[embedding_dim * 2, embedding_dim],
            batch_norm=False)

    def evaluate(self, obs):
        """Returns value estimate [B, 1]"""
        fusion = self.encoder(obs['p_net'], obs['v_net'])
        return self.net(fusion)
```

#### 4.5 `HrlAcActorCritic`

```python
@ActorCriticRegistry.register('hrl_ac')
class HrlAcActorCritic(BaseActorCritic):
    """
    Combined actor-critic for HRL-AC solver.
    Actor and Critic have separate encoders (no weight sharing).
    This follows the hrl_ac source exactly.
    """
    def __init__(self, p_net_num_nodes,
                 p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.actor = HrlAcActor(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)
        self.critic = HrlAcCritic(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)

    def act(self, obs):
        return self.actor.act(obs)

    def evaluate(self, obs):
        return self.critic.evaluate(obs)
```

---

### Phase 5: Solver — `hrl_ac_solver.py`

**File:** `virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_solver.py`

#### 5.1 `obs_as_tensor` — Tensor Convertor

```python
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from virne.solver.learning.utils import get_pyg_data


def obs_as_tensor(obs, device):
    """
    Convert a single observation dict or a list of observation dicts
    into batched PyG Batch objects for the actor/critic forward pass.

    Single obs dict → wraps in a 1-element batch.
    List of obs dicts → stacks into a proper Batch.

    Returns:
        dict with keys: 'p_net' (Batch), 'v_net' (Batch), 'v_net_attrs' (FloatTensor)
    """
    if isinstance(obs, dict):
        obs_list = [obs]
    elif isinstance(obs, list):
        obs_list = obs
    else:
        raise TypeError(f"obs must be dict or list, got {type(obs)}")

    p_data_list, v_data_list, v_attrs_list = [], [], []
    for o in obs_list:
        p_data_list.append(get_pyg_data(
            o['p_net_x'], o['p_net_edge_index'], o['p_net_edge_attr']))
        v_data_list.append(get_pyg_data(
            o['v_net_x'], o['v_net_edge_index'], o['v_net_edge_attr']))
        v_attrs_list.append(o['v_net_attrs'])

    return {
        'p_net': Batch.from_data_list(p_data_list).to(device),
        'v_net': Batch.from_data_list(v_data_list).to(device),
        'v_net_attrs': torch.FloatTensor(
            np.array(v_attrs_list)).to(device),
    }
```

#### 5.2 `HrlAcSolver`

```python
from virne.solver import SolverRegistry
from virne.solver.learning.rl_core.rl_solver import PPOSolver
from virne.solver.learning.rl_core.online_agent import OnlineAgent
from virne.solver.learning.rl_policy.hrl_ac_policy import HrlAcActorCritic
from virne.solver.learning.reinforcement_learning.hrl_ac.hrl_ac_env import HrlAcOnlineEnv
from virne.solver.learning.rl_core.feature_constructor import HrlAcFeatureConstructor


@SolverRegistry.register(solver_name='hrl_ac', solver_type='r_learning')
class HrlAcSolver(OnlineAgent, PPOSolver):
    """
    2-Stage Hierarchical RL solver for VNE admission control.

    Upper agent: GNN-based PPO (this class).
    Lower agent: heuristic or pretrained RL sub-solver (inside HrlAcOnlineEnv).

    Training loop is inherited from OnlineAgent.learn_singly():
      for each epoch:
        reset env → iterate VNRs
        select_action() → env.step() → buffer.add()
        when buffer full: compute_returns_and_advantages() → update()

    Inference (solve()):
      obs → obs_as_tensor → select_action(sample=False) → action ∈ {0,1}
      Returns the integer action, NOT a Solution. The env.step() call in
      OnlineAgent.validate() passes the action to env.step() which internally
      calls the sub-solver and returns the full Solution.
    """

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        OnlineAgent.__init__(self)
        PPOSolver.__init__(self, controller, recorder, counter, logger,
                           config, **kwargs)

        # ── Derive feature dimensions from config ─────────────────────────
        num_p_nodes = config.p_net_setting.num_nodes
        p_feat_dim = HrlAcFeatureConstructor.get_p_net_feature_dim(config)
        v_feat_dim = HrlAcFeatureConstructor.get_v_net_feature_dim(config)
        p_edge_dim = 1   # bandwidth only (typical setup)
        v_edge_dim = 1

        # ── Build policy network ──────────────────────────────────────────
        self.policy = HrlAcActorCritic(
            p_net_num_nodes=num_p_nodes,
            p_net_feature_dim=p_feat_dim,
            p_net_edge_dim=p_edge_dim,
            v_net_feature_dim=v_feat_dim,
            v_net_edge_dim=v_edge_dim,
            embedding_dim=self.embedding_dim,
            dropout_prob=self.dropout_prob,
            batch_norm=self.batch_norm,
        ).to(self.device)

        # ── Optimizer: actor at lr/10, critic at lr/10 ────────────────────
        # Matches hrl_ac source (slower learning rate since GNNs converge
        # more stably at lower rates, especially with deep GAT encoders)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(),  'lr': self.lr_actor  / 10},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic / 10},
        ])

        # ── PPO hyperparameters ───────────────────────────────────────────
        self.gamma = 1.0          # no discounting: each VNR is one step
        self.gae_lambda = 0.98
        self.norm_reward = True
        self.compute_return_method = 'gae'

        # ── Observation preprocessor (used by OnlineAgent base) ───────────
        self.preprocess_obs = obs_as_tensor

    def learn(self, env, num_epochs=1, start_epoch=0, **kwargs):
        """
        Delegates to OnlineAgent.learn_singly() with env = HrlAcOnlineEnv.

        The env must be an instance of HrlAcOnlineEnv.
        """
        assert isinstance(env, HrlAcOnlineEnv), (
            f"HrlAcSolver.learn() requires HrlAcOnlineEnv, got {type(env)}")
        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            self.learn_singly(env, num_epochs=1, **kwargs)
```

**Key design choices:**
- `gamma=1.0` — each VNR arrival is a **single-step episode**, so no discounting is meaningful.
- Optimizer LR divided by 10 — matches hrl_ac source and is appropriate for deep GNNs.
- `preprocess_obs = obs_as_tensor` — plugged into `OnlineAgent.solve()` and `OnlineAgent.learn_singly()` without any changes to the base class.

#### 5.3 `__init__.py`

```python
# virne/solver/learning/reinforcement_learning/hrl_ac/__init__.py

from .hrl_ac_solver import HrlAcSolver
from .hrl_ac_env import HrlAcOnlineEnv

__all__ = ['HrlAcSolver', 'HrlAcOnlineEnv']
```

---

### Phase 6: Registration & Config

#### 6.1 Update `reinforcement_learning/__init__.py`

```python
# virne/solver/learning/reinforcement_learning/__init__.py
# ... existing imports ...
from .hrl_ac import HrlAcSolver, HrlAcOnlineEnv
```

#### 6.2 Update `rl_policy/__init__.py`

```python
# virne/solver/learning/rl_policy/__init__.py
# ... existing imports ...
from .hrl_ac_policy import HrlAcActorCritic
```

#### 6.3 Config File — `hrl_ac.yaml`

```yaml
# virne/conf/solver/hrl_ac.yaml

solver_name: hrl_ac
solver_type: r_learning

# ── Sub-solver (lower agent) ───────────────────────────────────────────────
sub_solver_name: nrm_rank        # options: nrm_rank | grc_rank | hrl_ra
pretrained_subsolver_model_path: ""   # path to HrlRaSolver checkpoint if sub_solver_name=hrl_ra

# ── Physical network reference ─────────────────────────────────────────────
# These must match the p_net_setting in simulation config
num_node_resource_attrs: 1       # e.g. CPU only
num_link_resource_attrs: 1       # e.g. BW only

# ── Policy network ─────────────────────────────────────────────────────────
embedding_dim: 128
dropout_prob: 0.0
batch_norm: false

# ── PPO hyperparameters ────────────────────────────────────────────────────
rl_gamma: 1.0                    # no temporal discounting (single-step MDP)
gae_lambda: 0.98
lr_actor: 0.001                  # effective lr = lr_actor / 10 = 0.0001
lr_critic: 0.001
eps_clip: 0.2
repeat_times: 10
batch_size: 128
target_steps: 128
norm_advantage: true
norm_reward: true
clip_grad: true
max_grad_norm: 1.0

# ── Reward shaping ─────────────────────────────────────────────────────────
revenue_benchmark: 100.0
failure_penalty_per_node: 0.01

# ── Training ───────────────────────────────────────────────────────────────
embedding_dim: 128
coef_critic_loss: 0.5
coef_entropy_loss: 0.01

# ── Virne solver base ──────────────────────────────────────────────────────
node_ranking_method: order
link_ranking_method: order
matching_mathod: greedy
shortest_method: k_shortest
k_shortest: 10
allow_rejection: true            # upper agent handles rejection
allow_revocable: false
reusable: false
```

---

## 5. Component Specifications

### Feature Dimension Reference (Standard CPU+BW Setup)

| Network | Feature Component | Count | Notes |
|---|---|---|---|
| p_net node | CPU (resource) | 1 | normalised by benchmark |
| p_net node | degree | 1 | normalised by max degree |
| p_net node | max BW per node | 1 | normalised |
| p_net node | sum BW per node | 1 | normalised |
| **p_net node total** | | **4** | `p_net_feature_dim = 4` |
| v_net node | CPU demand | 1 | |
| v_net node | degree | 1 | |
| v_net node | max BW demand | 1 | |
| v_net node | sum BW demand | 1 | |
| v_net node | lifetime (broadcast) | 1 | from `_get_v_net_attrs_obs` |
| **v_net node total** | | **5** | `v_net_feature_dim = 5` |
| p_net edge | BW | 1 | `p_net_edge_dim = 1` |
| v_net edge | BW demand | 1 | `v_net_edge_dim = 1` |

These match exactly `p_net_feature_dim=1+3` and `v_net_feature_dim=2+3` from the hrl_ac source (`HrlAcSolver` instantiation line: `ActorCritic(p_net_feature_dim=1+3, ..., v_net_feature_dim=2+3, ...)`).

### Policy Network Parameter Count (embedding_dim=128)

| Component | Layers | Approx Params |
|---|---|---|
| `HrlAcEncoder` p_net GNN (5-layer DeepEdgeFeatureGAT) | 5 GAT | ~750K |
| `HrlAcEncoder` v_net GNN (3-layer DeepEdgeFeatureGAT) | 3 GAT | ~450K |
| `HrlAcEncoder` 6 pooling heads | GAP × 2 + mean × 2 + sum × 2 | ~100K |
| Actor MLP | 3-layer, 256→128→2 | ~50K |
| Critic MLP | 3-layer, 256→128→1 | ~50K |
| **Actors total** | | **~1.4M** |
| **Actor + Critic (separate encoders)** | | **~2.8M** |

### Observation Dict Schema

```
{
  'p_net_x':          np.float32  [num_p_nodes, 4]
  'p_net_edge_index': np.int64    [2, num_p_links*2]
  'p_net_edge_attr':  np.float32  [num_p_links*2, 1]
  'v_net_attrs':      np.float32  [1]
  'v_net_x':          np.float32  [num_v_nodes, 5]
  'v_net_edge_index': np.int64    [2, num_v_links*2]
  'v_net_edge_attr':  np.float32  [num_v_links*2, 1]
}
```

Note: edges are stored bidirectionally (`obs_handler.get_link_pair_obs` concatenates `[links, links_reversed]`).

### Solution Object Contract

```python
# Accepted path (sub-solver success):
solution['result']           = True
solution['early_rejection']  = False   # default from Solution.reset()
solution['node_slots']       = {v_node_id: p_node_id, ...}
solution['link_paths']       = {(v_u, v_v): [(p_a, p_b), ...], ...}

# Accepted path (sub-solver failure):
solution['result']           = False
solution['early_rejection']  = False   ← MUST NOT be True
solution['place_result']     = False or True
solution['route_result']     = False or True

# Rejected path (upper agent):
solution['result']           = False
solution['early_rejection']  = True    ← SET ONLY HERE
```

---

## 6. Data Flow & Contract

### Per-VNR Lifecycle

```
env.transit_obs()
    └── env.ready(event_id)          # sets self.v_net, self.solution, self.p_net_backup
            └── env.get_observation()
                    └── HrlAcOnlineEnv.get_observation()
                            → obs dict

obs ──► solver.preprocess_obs(obs, device)
            → {'p_net': Batch, 'v_net': Batch, 'v_net_attrs': Tensor}

tensor_obs ──► solver.select_action(tensor_obs, sample=True)
                    → action ∈ {0, 1},  action_logprob

action ──► HrlAcOnlineEnv.step(action)
    │
    ├─ action=1 ──► sub_solver.solve({'v_net': self.v_net, 'p_net': self.p_net})
    │                   → solution (result=True|False, early_rejection=False)
    │
    └─ action=0 ──► Solution(self.v_net)
                        solution['early_rejection'] = True
                        solution['result'] = False

solution ──► SolutionStepRLEnv.step(solution)    [parent class]
    ├── if result: self.solution = solution; controller.deploy(...)
    └── else:      self.rollback_for_failure(reason)
    ├── record = recorder.count(v_net, p_net, solution)
    ├── reward = self.compute_reward(record)      [HrlAcOnlineEnv override]
    ├── record = self.add_record(record, extra_info)
    └── done = self.transit_obs()
    → (obs, reward, done, info)

solver.buffer.add(obs, action, reward, done, action_logprob, value)
```

### Buffer Flush & PPO Update

```
When done=True AND buffer.size() >= batch_size:
    last_value = solver.estimate_value(next_obs_tensor)   # = 0 if done
    buffer.compute_returns_and_advantages(
        last_value, gamma=1.0, gae_lambda=0.98, method='gae')
    solver.update()   # PPO clipped surrogate loss
    buffer.clear()
```

### Training / Validation Split

| Mode | Entry Point | sample= | env type |
|---|---|---|---|
| Training | `solver.learn(env)` | `True` | `HrlAcOnlineEnv` |
| Validation | `solver.validate(env)` | `False` (greedy) | same env, reset |
| Inference | `solver.solve(instance)` | `False` | standalone |

---

## 7. Key Differences from hrl_ac Source

### What Changed

| Aspect | hrl_ac source | virne port | Reason |
|---|---|---|---|
| Base class | custom `PPOSolver` in `hrl_ac/rl_solver.py` | virne's `PPOSolver` in `rl_core/rl_solver.py` | Reuse existing infra |
| Environment base | custom `SolutionStepRLEnv` in `hrl_ac/rl_environment.py` | virne's `SolutionStepRLEnv` in `rl_core/online_rl_environment.py` | Already in virne |
| Constructor args | `controller, recorder, counter, **kwargs` | `controller, recorder, counter, logger, config, **kwargs` | virne adds `logger` and `config` |
| Sub-solver constructor | `NRMRankSolver(controller, recorder, counter, **kwargs)` | `NRMRankSolver(controller, recorder, counter, logger, config, **kwargs)` | virne Solver signature |
| Config access | `kwargs.get('key', default)` | `config.key` (OmegaConf) + `kwargs.get` fallback | virne convention |
| Reward method signature | `compute_reward(self, solution)` | `compute_reward(self, record)` | virne passes `record` (dict), not Solution |
| Observation helper | `hrl_ac/obs_handler.py :: ObservationHandler` | `virne/solver/learning/obs_handler.py :: ObservationHandler` | Same class, different import |
| Net imports | local `from ..net import ...` | `from virne.solver.learning.rl_policy.net import ...` | virne path |
| Registry | none | `@SolverRegistry.register('hrl_ac')` | Required by virne |
| Logger | `print()` statements | `self.logger.info(...)` | virne convention |

### What is Identical

- All GNN architecture logic (`DeepEdgeFeatureGAT`, `HrlAcEncoder`, pooling)
- Reward formula and average-reward baseline
- Observation structure (p_net_x, v_net_x, v_net_attrs, edge_index, edge_attr)
- `early_rejection` flag semantics
- `gamma=1.0`, `gae_lambda=0.98`
- Optimizer with `lr/10` for both actor and critic
- Binary action space {reject, accept}

---

## 8. Testing Checklist

### Unit Tests

```
□ HrlAcOnlineEnv instantiates without error (nrm_rank sub-solver)
□ HrlAcOnlineEnv instantiates without error (grc_rank sub-solver)
□ env.reset() returns valid observation dict with correct shapes
□ env.step(0) → solution['early_rejection'] == True, solution['result'] == False
□ env.step(1) → solution['early_rejection'] == False (regardless of sub-solver result)
□ env.compute_reward(record) returns float for all 3 cases (success, failure, rejection)
□ obs_as_tensor(single_obs) returns correct Batch sizes
□ obs_as_tensor(list_of_obs) returns correct Batch sizes
□ HrlAcActorCritic.act(obs_tensor) returns shape [B, 2]
□ HrlAcActorCritic.evaluate(obs_tensor) returns shape [B, 1]
□ HrlAcSolver.__init__ completes (policy on correct device)
□ SolverRegistry.get('hrl_ac') returns HrlAcSolver class
```

### Integration Tests

```
□ solver.learn(env, num_epochs=1) runs without crashing for small num_v_nets
□ solver.validate(env) runs, returns summary_info dict with 'acceptance_rate' key
□ PPO update step runs after buffer fills (check loss is finite)
□ Model save/load: save_model() → load_model() → validate() gives same acceptance rate
□ Sub-solver 'hrl_ra' path: pretrained model loads, eval mode set correctly
□ done=True transition: transit_obs() correctly skips leave events
□ p_net resource balance: after n episodes, p_net resources are consistent
```

### Regression Tests (run against NRMRankSolver baseline)

```
□ After 100 epochs training: acceptance_rate > NRMRankSolver baseline
□ After 100 epochs: avg_r2c_ratio ≥ NRMRankSolver baseline (or within 5%)
□ No NaN in reward/loss after 1000 steps
□ Reward does not collapse to always 0.0 (degenerate reject policy)
□ Reward does not collapse to always-accept (check early_rejection_count > 0)
```

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Always-accept degenerate policy** | High (early training) | High | Shaped negative reward for accepted failures; curriculum option in `generate_action_mask` |
| **Always-reject degenerate policy** | Medium | High | Monitor `early_rejection_count`; add acceptance-rate regularisation term to actor loss if needed |
| **`early_rejection` flag collision** | Low (with tests) | High | Assert in `env.step(action=1)` that sub-solver does NOT set `early_rejection=True` |
| **Feature staleness** | Low | Medium | p_net features are computed fresh from `self.p_net` each `get_observation()` call |
| **Dimension mismatch in policy construction** | Medium | High | Use `HrlAcFeatureConstructor.get_p_net_feature_dim(config)` — derive from config, never hardcode |
| **`compute_reward(solution)` vs `compute_reward(record)` signature** | High | High | virne passes the recorder `record` dict; hrl_ac passes the `Solution` object. Port must use `record` keys e.g. `record['result']`, `record['v_net_revenue']` |
| **Sub-solver modifies p_net in-place** | Medium | High | After sub-solver failure, `SolutionStepRLEnv.step()` calls `rollback_for_failure()` which restores from `p_net_backup`. Verify backup is taken before sub-solver runs |
| **Logger not available in env** | Low | Low | `HrlAcOnlineEnv` inherits `self.logger` from `BaseEnvironment` |
| **OmegaConf vs dict config** | Medium | Medium | Use `config.get('key', default)` or `OmegaConf.select(config, 'key', default=...)` for all config access |
| **DeepEdgeFeatureGAT already in virne net.py** | Medium | Low | Check before porting; import from existing location if present |
| **HrlRaSolver import circular dependency** | Low | Medium | Use lazy import inside `_build_sub_solver` for `hrl_ra` path |
| **GPU memory: two separate encoders** | Medium | Medium | Use `embedding_dim=64` for GPU-limited environments; document in config comments |

---

## Appendix A: Minimal Smoke-Test Script

```python
# test_hrl_ac_smoke.py
"""
Minimal smoke test: import, instantiate, reset, step, learn 1 epoch.
Run from virne repo root: python test_hrl_ac_smoke.py
"""
import torch
from omegaconf import OmegaConf

from virne.network import PhysicalNetwork, VirtualNetworkRequestSimulator
from virne.core import Controller, Recorder, Counter, Logger, Solution
from virne.solver import SolverRegistry
from virne.solver.learning.reinforcement_learning.hrl_ac import HrlAcOnlineEnv

# ── Minimal config ─────────────────────────────────────────────────────────
config = OmegaConf.create({
    'solver': {
        'solver_name': 'hrl_ac',
        'sub_solver_name': 'nrm_rank',
        'node_ranking_method': 'order',
        'link_ranking_method': 'order',
        'matching_mathod': 'greedy',
        'shortest_method': 'k_shortest',
        'k_shortest': 10,
        'allow_rejection': True,
        'allow_revocable': False,
        'reusable': False,
        'embedding_dim': 64,
        'dropout_prob': 0.0,
        'batch_norm': False,
    },
    'experiment': {
        'run_id': 'smoke_test',
        'seed': 42,
        'save_root_dir': '/tmp/virne_smoke',
        'if_load_v_nets': False,
    },
    'simulation': {
        'p_net_dataset_dir': '...',
        'v_nets_dataset_dir': '...',
        'v_sim_setting_num_node_resource_attrs': 1,
        'v_sim_setting_num_link_resource_attrs': 1,
    },
    'recorder': {
        'if_save_records': False,
        'if_temp_save_records': False,
        'record_dir_name': 'records',
        'summary_file_name': 'summary.csv',
    },
    'logger': {
        'backends': ['console'],
        'level': 'WARNING',
        'log_file_name': 'run.log',
        'project_name': 'virne',
        'experiment_name': 'smoke',
        'log_dir_name': 'logs',
        'log_show_interval': 100,
    },
    'rl': {
        'gamma': 1.0,
        'gae_lambda': 0.98,
        'lr_actor': 0.001,
        'lr_critic': 0.001,
        'eps_clip': 0.2,
        'repeat_times': 2,
        'batch_size': 8,
        'target_steps': 8,
        'norm_advantage': True,
        'norm_reward': True,
        'clip_grad': True,
        'max_grad_norm': 1.0,
        'coef_critic_loss': 0.5,
        'coef_entropy_loss': 0.01,
        'reward_calculator': {'name': 'hrl_ac', 'intermediate_reward': False},
        'feature_constructor': {
            'if_use_node_status_flags': False,
            'if_use_aggregated_link_attrs': True,
            'if_use_degree_metric': True,
            'if_use_more_topological_metrics': False,
        },
        'mask_actions': False,
    },
    'training': {
        'num_train_epochs': 1,
        'eval_interval': 10,
        'save_interval': 10,
    },
    'p_net_setting': {'num_nodes': 100},
    'num_node_resource_attrs': 1,
    'num_link_resource_attrs': 1,
})

# Build infra (use your project's standard setup functions)
# p_net = PhysicalNetwork.from_setting(...)
# v_net_sim = VirtualNetworkRequestSimulator.from_setting(...)
# controller = Controller(...)
# counter = Counter(...)
# recorder = Recorder(counter, config)
# logger = Logger(config)

# env = HrlAcOnlineEnv(p_net, v_net_sim, controller, recorder, counter, logger, config)
# obs = env.reset()
# print("Obs shapes:", {k: v.shape for k, v in obs.items()})

# SolverCls = SolverRegistry.get('hrl_ac')
# solver = SolverCls(controller, recorder, counter, logger, config)
# solver.learn(env, num_epochs=1)
# print("Smoke test PASSED")
```

---

## Appendix B: Curriculum Schedule (Optional Enhancement)

If the always-accept problem persists after reward shaping, implement curriculum by overriding `generate_action_mask`:

```python
# Inside HrlAcOnlineEnv:

def generate_action_mask(self):
    """
    Curriculum schedule:
      Phase 0 (first N VNRs):  accept-only → force agent to learn quality
      Phase 1 (after N VNRs):  both actions available → learn rejection

    Set curriculum_phase via config: 'curriculum_accept_only_steps': 5000
    """
    accept_only_steps = getattr(self, 'curriculum_accept_only_steps', 0)
    if self.global_timestep_count < accept_only_steps:
        return np.array([False, True])   # only accept
    return np.ones(2, dtype=bool)        # both
```

Set `curriculum_accept_only_steps` in the yaml config.

---

*End of Plan*
