# HRL-AC Implementation Plan — High-Level Admission Control Only

> **Scope**: Only the **high-level binary admission-control policy** is trained with PPO.
> Node mapping is fully delegated to a configurable heuristic/meta-heuristic sub-solver
> (e.g. `grc_rank`, `nrm_rank`, `fast_hpso`). All low-level RL code is removed.

---

## 1. Architecture

```
┌────────────────────────────────────────────────────────┐
│  HIGH-LEVEL POLICY  (PPO — the only thing we train)    │
│                                                        │
│  Input:  p_net graph + v_net graph (dual-GNN obs)      │
│  Action: Discrete(2)  — 0 = reject,  1 = accept       │
│  Reward: r2c_ratio (success) | 0 (reject) | -penalty  │
└───────────────────┬────────────────────────────────────┘
                    │  action == 1
                    ▼
┌────────────────────────────────────────────────────────┐
│  SUB-SOLVER  (heuristic, NOT trained)                  │
│  Configured via  config.rl.hrl.sub_solver_name         │
│  Supported:  grc_rank | nrm_rank | fast_hpso | any     │
│              entry registered in SolverRegistry        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Files to Create

```
virne/solver/learning/reinforcement_learning/
└── hrl_ac/
    ├── __init__.py
    ├── hrl_ac_env.py        # Single-step admission-control environment
    ├── hrl_ac_policy.py     # Binary ActorCritic (actor + critic children)
    └── hrl_ac_solver.py     # Solver: InstanceAgent + PPOSolver
```

**No new feature constructor** — reuse the existing `p_net_v_net` constructor.  
**No new tensor convertor** — reuse `TensorConvertor.obs_as_tensor_for_hrl_ac` (defined below).

---

## 3. hrl_ac_env.py

```python
# virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_env.py

import copy
import numpy as np
from omegaconf import open_dict                          # FIX BUG-1

from virne.core import Solution
from virne.solver import SolverRegistry
from virne.solver.learning.rl_core import InstanceRLEnv  # use the correct base
from virne.solver.learning.rl_core.instance_rl_environment import InstanceRLEnv
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Logger


class HrlAcAdmissionEnv(InstanceRLEnv):
    """
    Single-step environment for high-level admission control.

    step(0) → reject   (early_rejection=True, reward=0)
    step(1) → accept   → run sub_solver → reward based on outcome

    Inherits InstanceRLEnv which already:
      - initialises self.feature_constructor from config.rl.feature_constructor.name
      - provides self.get_info(record) (BUG-4 fixed: method exists on RLBaseEnv)
      - provides self.compute_reward() via self.reward_calculator
    """

    def __init__(
        self,
        p_net: PhysicalNetwork,
        v_net: VirtualNetwork,
        controller: Controller,
        recorder: Recorder,
        counter: Counter,
        logger: Logger,
        config,
        **kwargs
    ):
        # Force feature constructor to p_net_v_net (dual-graph obs) — BUG-1: open_dict imported
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'

        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)

        # Lazy sub-solver reference (created on first use to avoid circular deps)
        self._sub_solver = None
        self._sub_solver_name = config.rl.get('hrl', {}).get('sub_solver_name', 'grc_rank')

    # ------------------------------------------------------------------
    # Sub-solver
    # ------------------------------------------------------------------
    def _get_sub_solver(self):
        if self._sub_solver is None:
            solver_cls = SolverRegistry.get(self._sub_solver_name)
            self._sub_solver = solver_cls(
                self.controller, self.recorder, self.counter, self.logger, self.config
            )
            self._sub_solver.eval()
        return self._sub_solver

    # ------------------------------------------------------------------
    # Core env interface
    # ------------------------------------------------------------------
    def reset(self):
        self.solution = Solution.from_v_net(self.v_net)
        self.p_net = copy.deepcopy(self.p_net_backup)
        return self.get_observation()

    def step(self, action):
        """Single binary step — admission control decision."""
        if action == 0:
            # Reject: mark early rejection, no mapping attempted
            self.solution['early_rejection'] = True
            self.solution['result'] = False
        else:
            # Accept: delegate mapping to sub-solver
            instance = {'v_net': self.v_net, 'p_net': self.p_net}
            sub_solution = self._get_sub_solver().solve(instance)
            # Merge sub-solution fields into self.solution
            for key, value in sub_solution.items():
                self.solution[key] = value

        solution_info = self.counter.count_solution(self.v_net, self.solution)
        reward = self._compute_admission_reward(solution_info)
        done = True
        # get_info(record) exists on RLBaseEnv — BUG-4 fixed
        return self.get_observation(), reward, done, self.get_info(solution_info)

    def _compute_admission_reward(self, solution_info: dict) -> float:
        """
        Reward formula following the original HRL-AC paper with
        running-average baseline subtracted outside (in merge_instance_experience).
        """
        if self.solution['result']:
            return solution_info.get('v_net_r2c_ratio', 0.0)
        elif self.solution.get('early_rejection', False):   # BUG-13: use .get()
            return 0.0
        else:
            # Accepted but mapping failed
            return -0.01 * self.v_net.num_nodes

    def get_observation(self):
        """
        Use the p_net_v_net feature constructor (curr_v_node_id=0 for global obs).
        Also add curr_v_node_id=0 so TensorConvertor.obs_as_tensor_for_dual_gnn
        passes it through without error.
        """
        obs = self.feature_constructor.construct(
            self.p_net, self.v_net, self.solution, curr_v_node_id=0
        )
        # Required by obs_as_tensor_for_dual_gnn (general_obs_as_tensor)
        obs['curr_v_node_id'] = 0
        return obs
```

---

## 4. hrl_ac_policy.py

```python
# virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_policy.py

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from virne.solver.learning.neural_network import GCNConvNet, GraphPooling


class AdmissionNetEncoder(nn.Module):
    """
    Dual-GNN encoder that produces a GLOBAL graph embedding (not per-node).
    Uses GCNConvNet with pooling='mean' — no return_batch=True needed.
    BUG-10 fixed: GCNConvNet supports pooling kwarg; use it instead of return_batch.
    """
    def __init__(self, feat_dim, embedding_dim, num_layers, dropout_prob, batch_norm):
        super().__init__()
        self.init_lin = nn.Linear(feat_dim, embedding_dim)
        # GCNConvNet with pooling='mean' returns global graph embedding (batch, dim)
        self.gnn = GCNConvNet(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            dropout_prob=dropout_prob,
            pooling='mean',   # returns (batch, embedding_dim) — BUG-10 fixed
        )

    def forward(self, net_batch):
        """
        Args:
            net_batch: PyG Batch with .x, .edge_index, .batch
        Returns:
            g_emb: (batch_size, embedding_dim)
        """
        x = self.init_lin(net_batch.x)
        net_batch = net_batch.clone()
        net_batch.x = x
        g_emb = self.gnn(net_batch)   # (total_nodes, emb) pooled to (batch, emb)
        return g_emb


class _AdmissionBase(nn.Module):
    """Shared encoder trunk for actor and critic."""
    def __init__(self, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim,
                 embedding_dim, dropout_prob, batch_norm, num_gnn_layers):
        super().__init__()
        self.p_net_encoder = AdmissionNetEncoder(p_net_x_dim, embedding_dim, num_gnn_layers, dropout_prob, batch_norm)
        self.v_net_encoder = AdmissionNetEncoder(v_net_x_dim, embedding_dim, num_gnn_layers, dropout_prob, batch_norm)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, obs):
        p_emb = self.p_net_encoder(obs['p_net'])   # (B, emb)
        v_emb = self.v_net_encoder(obs['v_net'])   # (B, emb)
        fused = self.fusion(torch.cat([p_emb, v_emb], dim=-1))
        return fused


class AdmissionActor(_AdmissionBase):
    """
    Binary actor: outputs logits for (reject=0, accept=1).
    Named 'actor' so OptimizerBuilder finds config.rl.learning_rate.actor.
    """
    def __init__(self, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0., batch_norm=False, num_gnn_layers=3, **kwargs):
        super().__init__(p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim,
                         embedding_dim, dropout_prob, batch_norm, num_gnn_layers)
        self.head = nn.Linear(embedding_dim, 2)   # 2 actions: reject / accept

    def forward(self, obs):
        fused = self._encode(obs)
        return self.head(fused)   # (B, 2)


class AdmissionCritic(_AdmissionBase):
    """
    Value critic: outputs scalar baseline for admission control.
    Named 'critic' so OptimizerBuilder finds config.rl.learning_rate.critic.
    """
    def __init__(self, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0., batch_norm=False, num_gnn_layers=3, **kwargs):
        super().__init__(p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim,
                         embedding_dim, dropout_prob, batch_norm, num_gnn_layers)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, obs):
        fused = self._encode(obs)
        return self.head(fused)   # (B, 1)


class AdmissionActorCritic(nn.Module):
    """
    Top-level policy module.
    Children MUST be named 'actor' and 'critic' so that
    OptimizerBuilder.build_optimizer() can map them to
    config.rl.learning_rate.actor / .critic respectively.
    """
    def __init__(self, p_net_x_dim, p_net_edge_dim, v_net_x_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0., batch_norm=False, num_gnn_layers=3, **kwargs):
        super().__init__()
        enc_kwargs = dict(
            p_net_x_dim=p_net_x_dim, p_net_edge_dim=p_net_edge_dim,
            v_net_x_dim=v_net_x_dim, v_net_edge_dim=v_net_edge_dim,
            embedding_dim=embedding_dim, dropout_prob=dropout_prob,
            batch_norm=batch_norm, num_gnn_layers=num_gnn_layers,
        )
        self.actor = AdmissionActor(**enc_kwargs)
        self.critic = AdmissionCritic(**enc_kwargs)

    def act(self, obs):
        """Returns (B, 2) logits — used by select_action / evaluate_actions."""
        return self.actor(obs)

    def evaluate(self, obs):
        """Returns (B, 1) value — used by estimate_value / evaluate_actions."""
        return self.critic(obs)
```

---

## 5. hrl_ac_solver.py

```python
# virne/solver/learning/reinforcement_learning/hrl_ac/hrl_ac_solver.py

import numpy as np
import torch
from omegaconf import open_dict                             # BUG-1: required import
from torch_geometric.data import Batch                      # BUG-11: explicit import

from virne.solver import SolverRegistry
from virne.solver.learning.rl_core import InstanceAgent
from virne.solver.learning.rl_core.rl_solver import PPOSolver
from virne.solver.learning.rl_core.buffer import RolloutBuffer
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder, OptimizerBuilder
from virne.solver.learning.rl_core.policy_builder import get_p_net_x_dim, get_p_net_edge_dim
from virne.solver.learning.rl_core.policy_builder import get_v_net_x_dim, get_v_net_edge_dim
from virne.solver.learning.utils import get_pyg_data        # BUG-11: correct import location

from .hrl_ac_env import HrlAcAdmissionEnv
from .hrl_ac_policy import AdmissionActorCritic


# ---------------------------------------------------------------------------
# Tensor convertor
# ---------------------------------------------------------------------------
def obs_as_tensor_for_hrl_ac(obs, device):
    """
    Convert obs dict / list-of-dicts to tensor dict for AdmissionActorCritic.
    obs keys: p_net_x, p_net_edge_index, p_net_edge_attr,
              v_net_x,  v_net_edge_index,  v_net_edge_attr,
              curr_v_node_id  (ignored by admission policy but present)

    BUG-11 fixed: uses get_pyg_data from virne.solver.learning.utils, Batch imported.
    """
    if isinstance(obs, dict):
        p_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs.get('p_net_edge_attr'))
        v_data = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs.get('v_net_edge_attr'))
        p_batch = Batch.from_data_list([p_data]).to(device)
        v_batch = Batch.from_data_list([v_data]).to(device)
        return {'p_net': p_batch, 'v_net': v_batch}

    elif isinstance(obs, list):
        p_list, v_list = [], []
        for o in obs:
            p_list.append(get_pyg_data(o['p_net_x'], o['p_net_edge_index'], o.get('p_net_edge_attr')))
            v_list.append(get_pyg_data(o['v_net_x'], o['v_net_edge_index'], o.get('v_net_edge_attr')))
        p_batch = Batch.from_data_list(p_list).to(device)
        v_batch = Batch.from_data_list(v_list).to(device)
        return {'p_net': p_batch, 'v_net': v_batch}

    else:
        raise TypeError(f"Unrecognised obs type: {type(obs)}")


# ---------------------------------------------------------------------------
# Policy builder
# ---------------------------------------------------------------------------
def build_hrl_ac_admission_policy(agent):
    """
    Build AdmissionActorCritic and its optimizer.
    Uses PolicyBuilder helpers so feature dims are computed consistently.

    BUG-14/15 fixed: dims come from get_p_net_x_dim / get_v_net_x_dim which
    correctly account for avg_distance, node_status, link_aggr columns.

    BUG-12 fixed: policy created here (not in solver __init__) with real config.

    OptimizerBuilder iterates policy.named_children() → ['actor', 'critic']
    and looks up config.rl.learning_rate.actor / .critic → always present.
    """
    config = agent.config
    policy = AdmissionActorCritic(
        p_net_x_dim=get_p_net_x_dim(config),        # BUG-15 fixed
        p_net_edge_dim=get_p_net_edge_dim(config),
        v_net_x_dim=get_v_net_x_dim(config),        # BUG-14 fixed (uses correct formula)
        v_net_edge_dim=get_v_net_edge_dim(config),
        **PolicyBuilder.get_general_nn_config(config),
    ).to(agent.device)

    optimizer = OptimizerBuilder.build_optimizer(config, policy)
    return policy, optimizer


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@SolverRegistry.register(solver_name='hrl_ac_ppo', solver_type='r_learning')
class HrlAcPpoSolver(InstanceAgent, PPOSolver):
    """
    Hierarchical RL Admission Control — PPO solver.

    High-level policy only: binary accept/reject.
    Node mapping delegated to sub-solver (heuristic/meta-heuristic).
    """

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        # BUG-2/BUG-9 fixed: standard virne init pattern; self.buffer created by RLSolver.__init__
        InstanceAgent.__init__(self, HrlAcAdmissionEnv)
        PPOSolver.__init__(
            self,
            controller, recorder, counter, logger, config,
            build_hrl_ac_admission_policy,   # make_policy callable
            obs_as_tensor_for_hrl_ac,        # preprocess_obs callable
            **kwargs
        )
        # Running-average baseline (paper formula)
        self._reward_sum = 0.0
        self._reward_count = 0

    # ------------------------------------------------------------------
    # Override learn_with_instance — must return (solution, buffer, last_value)
    # BUG-3/BUG-8 fixed: returns exactly 3 values matching learn_singly signature
    # BUG-9 fixed: uses self.buffer (inherited from RLSolver), not a separate buffer
    # ------------------------------------------------------------------
    def learn_with_instance(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']

        # Single-step episode
        instance_env = self.InstanceEnv(
            p_net, v_net,
            self.controller, self.recorder, self.counter, self.logger, self.config
        )
        instance_buffer = RolloutBuffer()

        obs = instance_env.reset()
        tensor_obs = self.preprocess_obs(obs, self.device)

        # Select action
        action, action_logprob = self.select_action(tensor_obs, sample=True)
        value = self.estimate_value(tensor_obs)

        # Execute single step
        next_obs, reward, done, info = instance_env.step(action)

        # Subtract running-average baseline (paper formula)
        self._reward_count += 1
        self._reward_sum += reward
        baseline = self._reward_sum / self._reward_count
        adjusted_reward = reward - baseline

        instance_buffer.add(obs, action, adjusted_reward, done, action_logprob, value=value)

        solution = instance_env.solution
        # Episode is always done=True; bootstrap value = 0
        last_value = 0.0

        return solution, instance_buffer, last_value   # BUG-8 fixed: 3 values

    # ------------------------------------------------------------------
    # Override merge_instance_experience — always merge, not only on success
    # (admission control must learn from rejections too)
    # ------------------------------------------------------------------
    def merge_instance_experience(self, instance, solution, instance_buffer, last_value):
        instance_buffer.compute_returns_and_advantages(
            last_value,
            gamma=self.config.rl.gamma,
            gae_lambda=self.gae_lambda,
            method=self.compute_advantage_method,
        )
        self.buffer.merge(instance_buffer)   # BUG-9 fixed: use self.buffer
        self.time_step += 1
        return self.buffer
```

---

## 6. \_\_init\_\_.py

```python
# virne/solver/learning/reinforcement_learning/hrl_ac/__init__.py
from .hrl_ac_solver import HrlAcPpoSolver
```

---

## 7. Configuration (settings/learning.yaml additions)

```yaml
rl:
  # ── existing keys unchanged ──
  gamma: 1.0
  gae_lambda: 0.98
  norm_reward: false
  norm_advantage: true

  learning_rate:
    actor: 0.0001    # required by OptimizerBuilder (named_children: actor)
    critic: 0.001    # required by OptimizerBuilder (named_children: critic)

  feature_constructor:
    # name is forced to 'p_net_v_net' by HrlAcAdmissionEnv.__init__
    name: p_net_v_net
    extracted_attr_types: [resource]
    # These flags drive dim calculations in PolicyBuilder helpers:
    if_use_node_status_flags: true
    if_use_aggregated_link_attrs: true
    if_use_degree_metric: false
    if_use_more_topological_metrics: false
    num_extracted_p_node_attrs: 1   # e.g. cpu
    num_extracted_p_link_attrs: 1   # e.g. bw
    num_extracted_v_node_attrs: 1
    num_extracted_v_link_attrs: 1

  hrl:
    sub_solver_name: grc_rank   # or nrm_rank, fast_hpso, etc.

nn:
  embedding_dim: 128
  dropout_prob: 0.
  batch_norm: false
  num_gnn_layers: 3
```

---

## 8. Registry Update

In `virne/solver/learning/reinforcement_learning/__init__.py`, add:

```python
from .hrl_ac import HrlAcPpoSolver   # registers 'hrl_ac_ppo' via @SolverRegistry.register
```

---

## 9. Bug Fix Reference

| Bug | Where | Fix applied |
|-----|-------|-------------|
| BUG-1 `open_dict` not imported | env + solver | `from omegaconf import open_dict` in both files |
| BUG-2 policy built before config ready | solver | Standard `make_policy` callable pattern |
| BUG-3 `low_obs` unbound on reject | solver | Removed; single-step, no low-level loop |
| BUG-4 `self.get_info()` missing | env | `get_info(record)` exists on `RLBaseEnv`; call with `solution_info` dict |
| BUG-5 `node_attr_benchmarks` TypeError | env | Removed `_construct_v_net_global_features`; use feature constructor |
| BUG-6 shape inconsistency | env | Removed; feature constructor handles shapes correctly |
| BUG-7 `construct()` not overridden | — | `HrlAcAdmissionEnv.get_observation()` calls `feature_constructor.construct()` directly |
| BUG-8 5-tuple return mismatch | solver | `learn_with_instance` returns `(solution, buffer, last_value)` |
| BUG-9 dual buffers, `self.buffer` never filled | solver | Use `self.buffer` only (inherited from `RLSolver`) |
| BUG-10 `return_batch=True` unsupported | policy | Use `GCNConvNet(pooling='mean')` instead |
| BUG-11 `get_pyg_data` / `Batch` missing | solver | `from virne.solver.learning.utils import get_pyg_data` + `from torch_geometric.data import Batch` |
| BUG-12 `feature_constructor` built with `None` p_net | solver | Removed from solver `__init__`; env creates it per-instance |
| BUG-13 `solution['early_rejection']` KeyError | env | `self.solution.get('early_rejection', False)` |
| BUG-14 `v_net_attrs_dim` off-by-one | policy | Use `get_v_net_x_dim(config)` from `PolicyBuilder` |
| BUG-15 `p_net_x_dim` missing `avg_distance` | policy | Use `get_p_net_x_dim(config)` from `PolicyBuilder` |

---

## 10. Usage

```bash
python run.py solver=hrl_ac_ppo \
  rl.hrl.sub_solver_name=grc_rank \
  simulation.p_net_setting_num_nodes=100
```
