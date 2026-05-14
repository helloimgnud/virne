# HRL-AC Integration Plan v2 — Corrected for virne Compatibility

> Fixes all 15 bugs from compatibility analysis. Verified against actual virne source.

---

## Table of Contents
1. [Overview & Goals](#1-overview--goals)
2. [Architecture Mapping](#2-architecture-mapping)
3. [File Structure](#3-file-structure)
4. [Phase 1: Environment — `hrl_ac_env.py`](#4-phase-1-environment)
5. [Phase 2: Feature Constructor (Passthrough)](#5-phase-2-feature-constructor)
6. [Phase 3: Policy — `hrl_ac_policy.py`](#6-phase-3-policy)
7. [Phase 4: Solver — `hrl_ac_solver.py`](#7-phase-4-solver)
8. [Phase 5: Registration & Config](#8-phase-5-registration--config)
9. [Component Specifications](#9-component-specifications)
10. [Risk Register (Updated)](#10-risk-register-updated)

---

## 1. Overview & Goals

A **2-stage hierarchical RL solver** for VNE inside `virne`:

- **Upper Agent (RL):** GNN-based PPO that decides once per VNR: accept (`1`) or reject (`0`).
- **Lower Agent (Heuristic):** Runs only on accepted VNRs. Returns a full `Solution`.
- **Environment base:** `SolutionStepRLEnv` from `online_rl_environment.py` — used as-is.

### Key Corrections vs plan.md

| Bug | Fix Applied |
|-----|-------------|
| `PPOSolver` wrong signature | Use `make_policy` + `obs_as_tensor` pattern via `PolicyBuilder` |
| Benchmark attrs missing on env | Explicitly build in `HrlAcOnlineEnv.__init__` from `AttributeBenchmarkManager` |
| `config.p_net_setting.num_nodes` bad key | Use `config.simulation.p_net_setting_num_nodes` |
| `self.embedding_dim` etc not set | Read from `config.nn.*` via `PolicyBuilder.get_general_nn_config()` |
| `config.rl.gamma` not updated | Patch with `open_dict` after super().__init__ |
| `DeepEdgeFeatureGAT` redefined | Import from `virne.solver.learning.neural_network.gnn` |
| Duplicate `embedding_dim` in YAML | Removed duplicate |
| `HrlAcSolver.learn()` double-loop | Removed; delegate to `RLSolver.learn()` |
| `OptimizerBuilder` naming | `HrlAcActorCritic` children named `actor`/`critic` matching config |
| `RewardCalculatorRegistry` sig mismatch | Skip registration; reward lives in env only |

---

## 2. Architecture Mapping

| hrl_ac concept | virne equivalent |
|---|---|
| `env.py :: OnlineEnv` | `HrlAcOnlineEnv(SolutionStepRLEnv)` — new |
| `net.py :: Encoder` | `HrlAcEncoder` importing `DeepEdgeFeatureGAT` from `neural_network.gnn` |
| `net.py :: Actor/Critic/ActorCritic` | `HrlAcActor`, `HrlAcCritic`, `HrlAcActorCritic` — registered in `ActorCriticRegistry` |
| `hrl_ac_solver.py :: HrlAcSolver` | `HrlAcSolver(OnlineAgent, PPOSolver)` using virne's `make_policy` pattern |
| `obs_as_tensor` | Module-level function passed to `PPOSolver.__init__` |

### Class Hierarchy

```
BaseEnvironment
└── OnlineRLEnvBase
    └── SolutionStepRLEnv  ← AS-IS
        └── HrlAcOnlineEnv  ← NEW

RLSolver (PPOSolver)
└── HrlAcSolver(OnlineAgent, PPOSolver)  ← NEW
```

---

## 3. File Structure

### Files to CREATE

```
virne/solver/learning/
├── rl_policy/
│   └── hrl_ac_policy.py          # HrlAcEncoder, HrlAcActor, HrlAcCritic, HrlAcActorCritic
│
└── reinforcement_learning/
    └── hrl_ac/
        ├── __init__.py
        ├── hrl_ac_solver.py      # HrlAcSolver + obs_as_tensor + make_hrl_ac_policy
        └── hrl_ac_env.py         # HrlAcOnlineEnv
```

### Files to MODIFY

```
virne/solver/learning/reinforcement_learning/__init__.py
    → add: from .hrl_ac import HrlAcSolver, HrlAcOnlineEnv

virne/solver/learning/rl_policy/__init__.py
    → add: from .hrl_ac_policy import HrlAcActorCritic

virne/conf/solver/
    → add: hrl_ac.yaml
```

### Files NOT to touch

```
virne/solver/learning/rl_core/online_rl_environment.py
virne/solver/learning/rl_core/online_agent.py
virne/solver/learning/rl_core/rl_solver.py
virne/solver/learning/rl_core/policy_builder.py    ← use OptimizerBuilder from here
virne/core/solution.py
virne/core/environment.py
```

---

## 4. Phase 1: Environment — `hrl_ac_env.py`

**File:** `virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_env.py`

```python
import copy
import numpy as np
from gym import spaces
from omegaconf import DictConfig

from virne.core import Solution
from virne.solver.heuristic.node_rank import GRCRankSolver, NRMRankSolver
from virne.solver.learning.rl_core.online_rl_environment import SolutionStepRLEnv
from virne.solver.learning.obs_handler import ObservationHandler
from virne.network import AttributeBenchmarkManager


class HrlAcOnlineEnv(SolutionStepRLEnv):
    """
    Online environment for 2-stage hierarchical RL admission control.

    Upper agent sees GNN-encoded (p_net, v_net) and outputs action in {0=reject, 1=accept}.
    Lower sub-solver runs only on accepted VNRs and returns a full Solution.
    """

    def __init__(self, p_net, v_net_simulator, controller, recorder,
                 counter, logger, config: DictConfig, **kwargs):
        # Pass k_searching=1 to avoid expensive beam search in sub-solver
        kwargs_for_env = copy.deepcopy(kwargs)
        kwargs_for_env['k_searching'] = 1
        super().__init__(p_net, v_net_simulator, controller, recorder,
                         counter, logger, config, **kwargs_for_env)

        # ── Override action space: binary {0=reject, 1=accept} ────────────
        self.action_space = spaces.Discrete(2)

        # ── Build attribute benchmarks (FIX BUG 3) ────────────────────────
        # SolutionStepRLEnv chain does NOT set these; we must build them here.
        p_net_benchmarks = AttributeBenchmarkManager.get_from_cache('p_net')
        if p_net_benchmarks is None:
            p_net_benchmarks = AttributeBenchmarkManager.get_benchmarks(self.p_net)
            AttributeBenchmarkManager.add_to_cache('p_net', p_net_benchmarks)
        self.node_attr_benchmarks = p_net_benchmarks.node_attr_benchmarks
        self.link_attr_benchmarks = p_net_benchmarks.link_attr_benchmarks
        self.link_sum_attr_benchmarks = p_net_benchmarks.link_sum_attr_benchmarks
        # Degree benchmark: max degree of physical network
        degrees = dict(self.p_net.degree()).values()
        self.degree_benchmark = max(degrees) if degrees else 1.0

        # ── Observation handler (already set by RLBaseEnv, ref here) ──────
        # self.obs_handler is set by RLBaseEnv.__init__ → no need to re-create

        # ── VNR-level reward trackers (mirrors hrl_ac source) ─────────────
        self.global_timestep_count = 0
        self.global_moving_average_reward = 0
        self.global_cumulative_reward = 0
        self.actual_cumulative_reward = 0

        # ── Build sub-solver ──────────────────────────────────────────────
        sub_solver_name = config.solver.get('sub_solver_name', 'nrm_rank')
        kwargs_for_sub = copy.deepcopy(kwargs)
        kwargs_for_sub['verbose'] = 0
        self._build_sub_solver(sub_solver_name, kwargs_for_sub, config)

    # ── Sub-solver factory ─────────────────────────────────────────────────
    def _build_sub_solver(self, sub_solver_name: str, kwargs_for_sub: dict,
                          config: DictConfig):
        """
        Factory: instantiate correct lower-level solver.
          'nrm_rank'  → NRMRankSolver  (fast, deterministic)
          'grc_rank'  → GRCRankSolver  (slightly slower)
          'hrl_ra'    → HrlRaSolver    (pretrained RL lower agent, lazy import)
        """
        if sub_solver_name == 'nrm_rank':
            self.sub_solver = NRMRankSolver(
                self.controller, self.recorder, self.counter,
                self.logger, config, **kwargs_for_sub)

        elif sub_solver_name == 'grc_rank':
            self.sub_solver = GRCRankSolver(
                self.controller, self.recorder, self.counter,
                self.logger, config, **kwargs_for_sub)

        elif sub_solver_name == 'hrl_ra':
            # Lazy import to avoid circular dependency
            from virne.solver.learning.reinforcement_learning.hrl_ra import HrlRaSolver
            self.sub_solver = HrlRaSolver(
                self.controller, self.recorder, self.counter,
                self.logger, config, **kwargs_for_sub)
            pretrained_path = config.solver.get('pretrained_subsolver_model_path', None)
            if pretrained_path:
                self.sub_solver.load_model(pretrained_path)
            self.sub_solver.eval()
        else:
            raise NotImplementedError(
                f'Unknown sub_solver_name: {sub_solver_name!r}. '
                f'Choose from: nrm_rank, grc_rank, hrl_ra')

    # ── Core step ─────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        Upper agent makes ONE binary decision per VNR.

        action=1 (accept): call sub_solver; pass its Solution to parent step().
        action=0 (reject): create a failed Solution with early_rejection=True.

        INVARIANT: Only this method sets solution['early_rejection'] = True.
        Sub-solver failure → result=False, early_rejection stays False.
        """
        if action:  # accept
            instance = {'v_net': self.v_net, 'p_net': self.p_net}
            solution = self.sub_solver.solve(instance)
            assert not solution.get('early_rejection', False), (
                "Sub-solver must NOT set early_rejection=True. "
                "That flag is reserved for the upper agent.")
        else:  # reject
            solution = Solution(self.v_net)
            solution['early_rejection'] = True
            solution['result'] = False

        # Delegate to SolutionStepRLEnv.step() which handles:
        #   rollback_for_failure(), recorder.count(), compute_reward(),
        #   add_record(), transit_obs(), get_observation()
        return super().step(solution)

    # ── Reward ────────────────────────────────────────────────────────────
    def compute_reward(self, record: dict) -> float:
        """
        Shaped reward to prevent degenerate always-accept / always-reject policy.

        Early rejection (upper agent NO)  → 0.0     (neutral)
        Accepted + heuristic SUCCESS      → (w_a+w_b) * (revenue/benchmark) * r2c
        Accepted + heuristic FAILURE      → -0.01 * num_v_nodes

        Adjusted by running global average (average-reward baseline from hrl_ac).

        FIX vs plan.md:
          - Uses record dict (virne convention) not Solution object (hrl_ac convention)
          - Does NOT double-increment self.cumulative_reward
            (parent SolutionStepRLEnv.step uses the returned value directly)
        """
        w_a = 1.0
        lifetime_scale = self.v_net_simulator.v_sim_setting['lifetime']['scale']
        w_b = record.get('v_net_lifetime', self.v_net.lifetime) / lifetime_scale
        revenue_benchmark = 100.0

        if record.get('result', False):
            basic_reward = record['v_net_revenue'] / revenue_benchmark
            reward = (w_a + w_b) * basic_reward * record['v_net_r2c_ratio']

        elif not record.get('result', False) and not record.get('early_rejection', False):
            # Accepted but sub-solver failed
            reward = -0.01 * self.v_net.num_nodes

        else:
            # Early rejection by upper agent
            reward = 0.0

        # ── Average-reward baseline (variance reduction) ──────────────────
        self.actual_cumulative_reward += reward
        self.v_net_reward += reward
        self.global_timestep_count += 1
        self.global_cumulative_reward += reward

        # Matches hrl_ac original exactly:
        # average_reward = reward - self.global_cumulative_reward / self.global_timestep_count
        running_avg = self.global_cumulative_reward / self.global_timestep_count
        adjusted_reward = reward - running_avg

        self.extra_record_info.update({
            'actual_cumulative_reward': self.actual_cumulative_reward,
            'global_cumulative_reward': self.global_cumulative_reward,
            'running_avg_reward': running_avg,
            'actual_reward': reward,
        })
        # NOTE: Do NOT increment self.cumulative_reward here.
        # parent SolutionStepRLEnv.step() passes compute_reward() return value
        # directly to the buffer — it does not add to self.cumulative_reward.
        return adjusted_reward

    # ── Observation ───────────────────────────────────────────────────────
    def get_observation(self) -> dict:
        """
        Returns a GNN-compatible dict:
          p_net_x          : [num_p_nodes, 4]   (CPU | degree | max_BW | sum_BW)
          p_net_edge_index : [2, num_p_links*2]
          p_net_edge_attr  : [num_p_links*2, 1]
          v_net_attrs      : [1]                (normalised lifetime scalar)
          v_net_x          : [num_v_nodes, 5]   (same as p_net + lifetime broadcast)
          v_net_edge_index : [2, num_v_links*2]
          v_net_edge_attr  : [num_v_links*2, 1]
        """
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        v_net_attrs = self._get_v_net_attrs_obs()

        # Broadcast VNR-level lifetime scalar onto each v-node row
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

    def _get_p_net_obs(self) -> dict:
        """p_net node features: [resource | degree | max_BW | sum_BW]  (dim=4)"""
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

    def _get_v_net_obs(self) -> dict:
        """v_net node features (before lifetime padding): [resource | degree | max_BW | sum_BW]"""
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

    def _get_v_net_attrs_obs(self) -> np.ndarray:
        """VNR-level scalar: normalised lifetime. Shape: [1]"""
        norm_lifetime = (self.v_net.lifetime /
                         self.v_net_simulator.v_sim_setting['lifetime']['scale'])
        return np.array([norm_lifetime], dtype=np.float32)

    def generate_action_mask(self) -> np.ndarray:
        """Both actions always available."""
        return np.ones(2, dtype=bool)

    def ready(self, event_id: int = 0):
        """Reset per-VNR trackers."""
        super().ready(event_id)
        # v_net_reward is reset by OnlineRLEnvBase.ready()
```

---

## 5. Phase 2: Feature Constructor (Passthrough)

**File:** `virne/solver/learning/rl_core/feature_constructor.py` — add at bottom.

```python
@FeatureConstructorRegistry.register('hrl_ac')
class HrlAcFeatureConstructor(BaseFeatureConstructor):
    """
    Passthrough stub for HRL-AC solver.
    Actual feature construction happens in HrlAcOnlineEnv.get_observation().
    This registration exists so config-driven code can reference 'hrl_ac' by name.

    get_observation() is intentionally NOT overridden — call sites that
    try to invoke it directly will get a clear error. Do NOT call this
    through the registry hot-path.
    """

    @staticmethod
    def get_p_net_feature_dim(config) -> int:
        """
        p_net node feature dim = num_node_resource_attrs + 1 (degree)
                                 + num_link_resource_attrs * 2 (max + sum)
        Default CPU+BW: 1 + 1 + 1 + 1 = 4
        """
        num_node_res = config.rl.feature_constructor.get(
            'num_extracted_p_node_attrs', 1)
        num_link_res = config.rl.feature_constructor.get(
            'num_extracted_p_link_attrs', 1)
        return num_node_res + 1 + num_link_res * 2

    @staticmethod
    def get_v_net_feature_dim(config) -> int:
        """Same as p_net PLUS the lifetime scalar broadcast: dim = p_dim + 1"""
        return HrlAcFeatureConstructor.get_p_net_feature_dim(config) + 1
```

---

## 6. Phase 3: Policy — `hrl_ac_policy.py`

**File:** `virne/solver/learning/rl_policy/hrl_ac_policy.py`

```python
import torch
import torch.nn as nn
from torch_geometric.data import Batch

# ── FIX BUG 11: Import existing DeepEdgeFeatureGAT, GraphAttentionPooling,
#    GraphPooling from virne's neural_network module.
#    Do NOT redefine them — that creates an incompatible parallel class.
from virne.solver.learning.neural_network.gnn import (
    DeepEdgeFeatureGAT, GraphPooling, GraphAttentionPooling)
from virne.solver.learning.rl_policy.net import MLPNet
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry


class HrlAcEncoder(nn.Module):
    """
    Dual GNN encoder for (p_net, v_net) pair.
    Identical logic to hrl_ac/net.py :: Encoder, but uses virne's existing
    DeepEdgeFeatureGAT instead of re-defining it.

    Output: concat([p_global, v_global]) ∈ R^{2 * embedding_dim}
    """

    def __init__(self, p_net_feature_dim: int, p_net_edge_dim: int,
                 v_net_feature_dim: int, v_net_edge_dim: int,
                 embedding_dim: int = 128,
                 dropout_prob: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        self.p_net_gnn = DeepEdgeFeatureGAT(
            p_net_feature_dim, embedding_dim,
            edge_dim=p_net_edge_dim, num_layers=5,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_gnn = DeepEdgeFeatureGAT(
            v_net_feature_dim, embedding_dim,
            edge_dim=v_net_edge_dim, num_layers=3,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob, batch_norm=batch_norm)

        self.p_net_gap  = GraphAttentionPooling(embedding_dim)
        self.v_net_gap  = GraphAttentionPooling(embedding_dim)
        self.p_net_mean = GraphPooling(aggr='mean')
        self.v_net_mean = GraphPooling(aggr='mean')
        self.p_net_sum  = GraphPooling(aggr='sum')
        self.v_net_sum  = GraphPooling(aggr='sum')

    def forward(self, p_net_batch: Batch, v_net_batch: Batch) -> torch.Tensor:
        # v_net
        v_emb = self.v_net_gnn(v_net_batch)
        v_global = (self.v_net_gap(v_emb, v_net_batch.batch)
                    + self.v_net_mean(v_emb, v_net_batch.batch)
                    + self.v_net_sum(v_emb, v_net_batch.batch))
        # p_net
        p_emb = self.p_net_gnn(p_net_batch)
        p_global = (self.p_net_gap(p_emb, p_net_batch.batch)
                    + self.p_net_mean(p_emb, p_net_batch.batch)
                    + self.p_net_sum(p_emb, p_net_batch.batch))

        return torch.cat([p_global, v_global], dim=-1)   # [B, 2*embedding_dim]


class HrlAcActor(nn.Module):
    """Actor: logits for {reject=0, accept=1}. Output shape: [B, 2]"""

    def __init__(self, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.encoder = HrlAcEncoder(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)
        self.net = MLPNet(
            embedding_dim * 2, 2, num_layers=3,
            embedding_dims=[embedding_dim * 2, embedding_dim],
            batch_norm=False)

    def act(self, obs: dict) -> torch.Tensor:
        """obs must have keys 'p_net' (Batch) and 'v_net' (Batch)."""
        fusion = self.encoder(obs['p_net'], obs['v_net'])
        return self.net(fusion)


class HrlAcCritic(nn.Module):
    """Critic: scalar value estimate. Separate encoder from actor (no weight sharing)."""

    def __init__(self, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.encoder = HrlAcEncoder(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)
        self.net = MLPNet(
            embedding_dim * 2, 1, num_layers=3,
            embedding_dims=[embedding_dim * 2, embedding_dim],
            batch_norm=False)

    def evaluate(self, obs: dict) -> torch.Tensor:
        fusion = self.encoder(obs['p_net'], obs['v_net'])
        return self.net(fusion)


@ActorCriticRegistry.register('hrl_ac')
class HrlAcActorCritic(BaseActorCritic):
    """
    Combined actor-critic for HRL-AC.
    Separate encoders for actor and critic (no weight sharing).

    FIX BUG (OptimizerBuilder): named children MUST be 'actor' and 'critic'
    because OptimizerBuilder iterates policy.named_children() and looks up
    config.rl.learning_rate.<child_name>.
    """

    def __init__(self, p_net_num_nodes: int,
                 p_net_feature_dim: int, p_net_edge_dim: int,
                 v_net_feature_dim: int, v_net_edge_dim: int,
                 embedding_dim: int = 128,
                 dropout_prob: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        # Children named 'actor' and 'critic' so OptimizerBuilder finds them
        self.actor = HrlAcActor(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)
        self.critic = HrlAcCritic(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm)

    def act(self, obs: dict) -> torch.Tensor:
        return self.actor.act(obs)

    def evaluate(self, obs: dict) -> torch.Tensor:
        return self.critic.evaluate(obs)
```

---

## 7. Phase 4: Solver � `hrl_ac_solver.py`

**File:** `virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_solver.py`

```python
import torch
from omegaconf import DictConfig, open_dict
from torch_geometric.data import Data, Batch

from virne.solver.learning.rl_core.online_agent import OnlineAgent
from virne.solver.learning.rl_core.rl_solver import PPOSolver
from virne.solver.learning.rl_core.policy_builder import OptimizerBuilder, PolicyBuilder
from virne.solver.registry import SolverRegistry
from virne.solver.learning.rl_policy.hrl_ac_policy import HrlAcActorCritic
from virne.solver.learning.rl_core.feature_constructor import HrlAcFeatureConstructor
from .hrl_ac_env import HrlAcOnlineEnv


def obs_as_tensor(observations, device):
    """Convert list of env-format obs dicts to PyG Batch pair. (FIX BUG 1, 2)"""
    if isinstance(observations, dict):
        observations = [observations]
    p_list, v_list = [], []
    for obs in observations:
        p_list.append(Data(
            x=torch.FloatTensor(obs['p_net_x']),
            edge_index=torch.LongTensor(obs['p_net_edge_index']),
            edge_attr=torch.FloatTensor(obs['p_net_edge_attr'])))
        v_list.append(Data(
            x=torch.FloatTensor(obs['v_net_x']),
            edge_index=torch.LongTensor(obs['v_net_edge_index']),
            edge_attr=torch.FloatTensor(obs['v_net_edge_attr'])))
    return {'p_net': Batch.from_data_list(p_list).to(device),
            'v_net': Batch.from_data_list(v_list).to(device)}


def make_hrl_ac_policy(agent):
    """Factory function for PPOSolver.__init__. (FIX BUG 1, 4, 8)"""
    config = agent.config
    nn_cfg = PolicyBuilder.get_general_nn_config(config)  # config.nn.*
    p_feat = HrlAcFeatureConstructor.get_p_net_feature_dim(config)
    v_feat = HrlAcFeatureConstructor.get_v_net_feature_dim(config)
    p_edge = config.rl.feature_constructor.get('num_extracted_p_link_attrs', 1)
    v_edge = config.rl.feature_constructor.get('num_extracted_v_link_attrs', 1)
    num_nodes = config.simulation.p_net_setting_num_nodes  # FIX BUG 4
    policy = HrlAcActorCritic(
        p_net_num_nodes=num_nodes, p_net_feature_dim=p_feat,
        p_net_edge_dim=p_edge, v_net_feature_dim=v_feat,
        v_net_edge_dim=v_edge, **nn_cfg).to(agent.device)
    optimizer = OptimizerBuilder.build_optimizer(config, policy)  # uses 'actor'/'critic' keys
    return policy, optimizer


@SolverRegistry.register(solver_name='hrl_ac', solver_type='r_learning')
class HrlAcSolver(OnlineAgent, PPOSolver):
    """HRL-AC: upper PPO + lower heuristic. (FIX BUG 1, 2, 13, 14)"""

    def __init__(self, controller, recorder, counter, logger,
                 config: DictConfig, **kwargs):
        OnlineAgent.__init__(self)
        with open_dict(config):
            config.rl.gamma = 1.0          # FIX BUG 13
        PPOSolver.__init__(self, controller, recorder, counter, logger,
                           config, make_hrl_ac_policy, obs_as_tensor, **kwargs)  # FIX BUG 1

    def make_env(self, p_net, v_net_simulator, **kwargs):
        return HrlAcOnlineEnv(p_net, v_net_simulator,
                              self.controller, self.recorder, self.counter,
                              self.logger, self.config, **kwargs)

    def preprocess_obs(self, obs, device=None):  # FIX BUG 2
        device = device or self.device
        if isinstance(obs, list):
            return obs_as_tensor(obs, device)
        if isinstance(obs, dict) and 'p_net_x' in obs:
            return obs_as_tensor(obs, device)
        raise ValueError(f"Unexpected obs format: {type(obs)}")

    def solve(self, instance):  # FIX BUG 2: route through env.step()
        v_sim = getattr(getattr(self, 'env', None), 'v_net_simulator', None)
        env = self.make_env(instance['p_net'], v_sim)
        env.reset()
        obs = env.get_observation()
        action, _ = self.select_action(obs_as_tensor(obs, self.device), sample=False)
        env.step(action)
        return env.solution
    # FIX BUG 14: No learn() override � RLSolver.learn() used as-is.
```

---

## 8. Phase 5: Registration & Config

### 8.1 `hrl_ac/__init__.py`
```python
from .hrl_ac_solver import HrlAcSolver
from .hrl_ac_env import HrlAcOnlineEnv
__all__ = ['HrlAcSolver', 'HrlAcOnlineEnv']
```

### 8.2 Update parent `__init__.py` files
```python
# reinforcement_learning/__init__.py
from .hrl_ac import HrlAcSolver, HrlAcOnlineEnv
# rl_policy/__init__.py
from .hrl_ac_policy import HrlAcActorCritic
```

### 8.3 `virne/conf/solver/hrl_ac.yaml`
```yaml
# @package _global_
# FIX BUG 7: no duplicate keys.
# FIX BUG 4: p_net count via config.simulation.p_net_setting_num_nodes.
# FIX BUG 8: network params under config.nn.
defaults:
  - _self_
solver:
  solver_name: hrl_ac
  solver_type: r_learning
  sub_solver_name: nrm_rank
  node_ranking_method: order
  link_ranking_method: order
  matching_mathod: greedy
  shortest_method: k_shortest
  k_shortest: 10
nn:
  embedding_dim: 128   # FIX BUG 7: single occurrence
  dropout_prob: 0.0
  batch_norm: false
rl:
  gamma: 1.0           # FIX BUG 13: also patched at runtime
  gae_lambda: 0.98
  eps_clip: 0.2
  if_use_baseline_solver: false
  if_allow_baseline_unsafe_solve: false
  baselin_solver_name: grc
  norm_reward: false
  norm_advantage: true
  clip_grad: true
  max_grad_norm: 1.0
  target_kl: null
  weight_decay: 0.0
  mask_actions: false
  maskable_policy: false
  coef_critic_loss: 0.5
  coef_entropy_loss: 0.01
  coef_mask_loss: 0.0
  target_steps: 128
  # FIX (OptimizerBuilder): child names 'actor'/'critic' must match policy.named_children()
  learning_rate:
    actor: 0.0001
    critic: 0.00001   # actor/10 � matches hrl_ac original
  feature_constructor:
    num_extracted_p_node_attrs: 1
    num_extracted_p_link_attrs: 1
    num_extracted_v_node_attrs: 1
    num_extracted_v_link_attrs: 1
    if_use_node_status_flags: false
    if_use_aggregated_link_attrs: false
    if_use_degree_metric: false
    if_use_more_topological_metrics: false
training:
  num_epochs: 300
  batch_size: 128
  use_cuda: true
  gpu_id: 0
  num_workers: 1
  distributed_training: false
  save_interval: 50
  eval_interval: 25
  log_interval: 10
  model_dir_name: model
  best_model_dir_name: best_model
inference:
  decode_strategy: greedy
  k_searching: 1
```

---

## 9. Correct Key Reference Table

| plan.md (WRONG) | plan_v2.md (CORRECT) | Source |
|---|---|---|
| `config.p_net_setting.num_nodes` | `config.simulation.p_net_setting_num_nodes` | `policy_builder.py:63` |
| `config.solver.embedding_dim` | `config.nn.embedding_dim` | `policy_builder.py:72` |
| `self.embedding_dim` (instance attr) | read from `config.nn.embedding_dim` | `policy_builder.py:72` |
| `self.lr_actor / 10` | `config.rl.learning_rate.critic: 1e-5` | `policy_builder.py:244` |
| `self.gamma = 1.0` only | `open_dict` + `config.rl.gamma = 1.0` | `online_agent.py` reads config |
| `class DeepEdgeFeatureGAT(...)` local | `from ...neural_network.gnn import DeepEdgeFeatureGAT` | `gnn.py:207` |
| `from .heuristic.node_rank import *` | `from virne.solver.heuristic.node_rank import GRCRankSolver, NRMRankSolver` | `node_rank.py:119,154` |
| Reward in `RewardCalculatorRegistry` | reward only in `HrlAcOnlineEnv.compute_reward()` | virne convention |
| `learn()` override wrapping `learn_singly` | removed; use `RLSolver.learn()` | `rl_solver.py:337` |
| Duplicate `embedding_dim` in YAML | single entry under `nn:` | OmegaConf requirement |

---

## 10. Risk Register (Updated)

| Risk | Mitigation |
|---|---|
| **Always-reject degenerate policy** | `coef_entropy_loss=0.01`; accepted+failed penalty `-0.01*N` makes pure-reject non-dominant. Monitor `acceptance_rate`. |
| **`compute_reward` called with no args** | `SolutionStepRLEnv.step()` calls `self.compute_reward()` with no args. `HrlAcOnlineEnv.compute_reward(record={})` defaults `record={}` and reads `self.solution` instead. Verify base calling site and add `record=None` default. |
| **`AttributeBenchmarkManager` cold start** | `BaseEnvironment.__init__` already populates cache; `get_from_cache('p_net')` returns valid data. Fallback builds fresh if cache is empty. |
| **`solve()` missing `v_net_simulator`** | `HrlAcSolver.solve()` accesses `self.env.v_net_simulator`. At pure-inference time cache it in a `self._v_net_simulator` attr during `make_env`. |
| **`HrlAcFeatureConstructor.get_observation()` raises** | By design. Obs construction is in env only. Registry stub exists for config lookup, not direct invocation. |
| **`v_net_attrs` dead key in obs dict** | Lifetime is already embedded in `v_net_x` as a per-node broadcast. `v_net_attrs` kept for future separate-branch use. Zero runtime cost. |
