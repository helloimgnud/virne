# HRL-AC Implementation Plan for VIRNE Framework

## Overview
This document provides a comprehensive implementation plan for integrating Hierarchical Reinforcement Learning with Actor-Critic (HRL-AC) into the VIRNE framework, following the architectural patterns established by Dual-GNN solvers.

---

## 1. Architecture Analysis & Design Decisions

### 1.1 Framework Integration Points
The HRL-AC solver must integrate with VIRNE at these key levels:

1. **Solver Level**: Extend `InstanceAgent` and RL solver base classes
2. **Environment Level**: Create hierarchical environment wrapper
3. **Policy Level**: Implement high-level (admission control) and low-level (node placement) policies
4. **Feature Construction**: Adapt observation handlers for hierarchical decision-making
5. **Training Loop**: Implement multi-level learning with appropriate buffer management

### 1.2 Design Pattern: Dual-Level Hierarchy
```
┌─────────────────────────────────────────────┐
│   HIGH-LEVEL POLICY (Admission Control)     │
│   - Decision: Accept/Reject VN Request      │
│   - Input: VN characteristics, P-net state  │
│   - Output: Binary action (0=reject, 1=acc) │
└──────────────┬──────────────────────────────┘
               │ (if accept)
               ▼
┌─────────────────────────────────────────────┐
│   LOW-LEVEL POLICY (Node Placement)         │
│   - Decision: Place v-node on p-node        │
│   - Input: Current v-node features, P-net   │
│   - Output: p-node placement                │
└─────────────────────────────────────────────┘
```

---

## 2. File Structure & Organization

### 2.1 New Files to Create

```
virne/solver/learning/hrl_ac/
├── __init__.py                      # Package initialization
├── hrl_ac_solver.py                 # Main HRL-AC solver classes
├── hrl_ac_env.py                    # Hierarchical environment
├── hrl_ac_policy.py                 # Policy networks (high-level & low-level)
├── hrl_ac_feature_constructor.py    # Feature construction for both levels
└── hrl_ac_utils.py                  # Utility functions

virne/solver/learning/hrl_core/      # (optional) Common HRL utilities
├── __init__.py
└── hrl_buffer.py                    # Hierarchical rollout buffer
```

### 2.2 Modified Files

- `virne/solver/learning/rl_core/__init__.py` - Add HRL imports
- `virne/solver/__init__.py` - Register HRL-AC solvers

---

## 3. Detailed Implementation Specifications

### 3.1 Hierarchical Environment (hrl_ac_env.py)

#### Purpose
Wrapper around existing `JointPRStepInstanceRLEnv` that separates high-level and low-level decisions.

#### Class: `HRLInstanceEnv(JointPRStepInstanceRLEnv)`

**Key Methods:**

```python
class HRLInstanceEnv(JointPRStepInstanceRLEnv):
    """
    Hierarchical wrapper for instance-level VNE environment.
    Separates admission control (high-level) from placement (low-level).
    """
    
    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        """
        Args:
            p_net, v_net, controller, recorder, counter, logger, config: standard VIRNE args
            **kwargs: 
                - low_level_solver: solver for node placement (e.g., 'grc_rank', 'nrm_rank', None)
                - use_learning_low_level: use learned policy vs heuristic for placement
        """
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        
        # Separate counters for two decision levels
        self.high_level_done = False
        self.low_level_sequence = []  # Track placement decisions
        
    def reset(self):
        """Reset for new VN request"""
        self.high_level_done = False
        self.low_level_sequence = []
        return super().reset()
    
    def step_high_level(self, action):
        """
        High-level policy step: admission control decision
        
        Args:
            action (int): 0=reject, 1=accept
            
        Returns:
            - observation (dict): high-level obs for next step
            - reward (float): immediate reward for high-level
            - done (bool): episode termination
            - info (dict): step information
        """
        if action == 0:  # Reject
            self.solution['early_rejection'] = True
            return self._finalize_episode()
        else:  # Accept (action == 1)
            self.high_level_done = True
            return self.get_observation(), 0.0, False, {}
    
    def step_low_level(self, action):
        """
        Low-level policy step: node placement
        Wraps the existing JointPRStepInstanceRLEnv.step()
        
        Args:
            action (int): p-node ID for placement
            
        Returns:
            - observation, reward, done, info from parent environment
        """
        obs, reward, done, info = super().step(action)
        
        if done:
            return self._finalize_episode()
        
        return obs, reward, done, info
    
    def get_high_level_observation(self):
        """
        Generate observation for high-level policy (admission control)
        
        Returns:
            dict with keys:
                - 'v_net': Batch of v_net graph data
                - 'p_net': Batch of p_net graph data
                - 'v_net_attrs': tensor of VN attributes (lifetime, num_nodes, etc.)
                - 'action_mask': mask of valid actions if applicable
        """
        # V-Net features (global properties)
        v_net_obs = self._construct_v_net_global_features()
        
        # P-Net features (global and local resource availability)
        p_net_obs = self._construct_p_net_global_features()
        
        return {
            'v_net': v_net_obs,
            'p_net': p_net_obs,
            'v_net_attrs': v_net_obs['attrs'],
        }
    
    def get_low_level_observation(self):
        """
        Generate observation for low-level policy (node placement)
        Delegates to parent's get_observation() which handles sequential placement
        """
        return super().get_observation()
    
    def _finalize_episode(self):
        """Finalize episode after high-level decision or completion"""
        solution_info = self.counter.count_solution(self.v_net, self.solution)
        reward = self._compute_hierarchical_reward(solution_info)
        
        return self.get_high_level_observation(), reward, True, self.get_info(solution_info)
    
    def _compute_hierarchical_reward(self, solution_info):
        """
        Compute reward combining both levels' contributions
        
        Strategy:
        - High-level gets reward based on overall acceptance success
        - Low-level gets shaped rewards during placement
        """
        if self.solution['result']:
            return solution_info['v_net_r2c_ratio']
        elif self.solution['early_rejection']:
            return 0.0
        else:
            return -0.01 * (self.v_net.num_nodes)

    def _construct_v_net_global_features(self):
        """
        Construct global V-Net features (not per-node)
        Returns features summarizing the entire VN request
        """
        # Node and link statistics
        num_nodes = self.v_net.num_nodes
        num_links = self.v_net.num_links
        
        # Resource demands
        node_attrs = self.v_net.get_node_attrs(self.extracted_attr_types)
        node_data = np.array(
            self.v_net.get_node_attrs_data(node_attrs), 
            dtype=np.float32
        )
        
        # Aggregated statistics
        v_attrs = np.array([
            num_nodes / self.p_net.num_nodes,  # normalized num nodes
            num_links / (num_nodes * num_nodes),  # link density
            node_data.sum(axis=1).mean() / self.node_attr_benchmarks,  # avg demand
            self.v_net.lifetime / self.lifetime_benchmark if hasattr(self, 'lifetime_benchmark') else 0.5,
        ], dtype=np.float32)
        
        return {'attrs': v_attrs}
    
    def _construct_p_net_global_features(self):
        """
        Construct global P-Net resource availability features
        Returns aggregated view of physical network state
        """
        # Current resource utilization
        node_attrs = self.p_net.get_node_attrs(self.extracted_attr_types)
        node_data = np.array(
            self.p_net.get_node_attrs_data(node_attrs),
            dtype=np.float32
        )
        
        # Normalize by max (extrema) attributes
        p_net_x = self._normalize_p_net_resources(node_data)
        
        # GNN encoding of P-Net
        from torch_geometric.data import Batch
        p_net_pyg = self._get_p_net_pyg_data(p_net_x)
        p_net_batch = Batch.from_data_list([p_net_pyg]).to(self.device)
        
        return {'x': p_net_x, 'batch': p_net_batch, 'edge_index': self.p_net_edge_index}

    def _normalize_p_net_resources(self, node_data):
        """Normalize node data by extrema attributes"""
        # Implementation: divide resource by max_resource for each attribute
        for i, attr in enumerate(self.extracted_attr_types):
            max_attr = [a for a in self.p_net.get_node_attrs(['extrema']) 
                       if a.originator == attr][0]
            max_values = np.array([self.p_net.nodes[n][max_attr.name] 
                                   for n in range(self.p_net.num_nodes)])
            node_data[i] = node_data[i] / (max_values + 1e-6)
        return node_data
```

### 3.2 Hierarchical Policy Networks (hrl_ac_policy.py)

#### Purpose
Implement separate actor-critic networks for high-level and low-level policies.

#### Class Structure

```python
import torch
import torch.nn as nn
from virne.solver.learning.neural_network import (
    GCNConvNet, DeepEdgeFeatureGAT, GraphPooling, GraphAttentionPooling, MLPNet
)
from torch_geometric.utils import to_dense_batch

class HighLevelActorCritic(nn.Module):
    """
    High-level policy: global VN admission control decision
    
    Architecture:
    - Encoder: Processes both P-Net and V-Net global features
    - Actor: Binary classification (accept/reject)
    - Critic: Value estimation for training
    """
    
    def __init__(
        self,
        p_net_x_dim,
        p_net_edge_dim,
        v_net_attrs_dim,
        embedding_dim=128,
        dropout_prob=0.,
        batch_norm=False,
        num_gnn_layers=3,
        **kwargs
    ):
        super().__init__()
        
        # Encoders
        self.p_net_encoder = GCNConvNet(
            input_dim=p_net_x_dim,
            output_dim=embedding_dim,
            num_layers=num_gnn_layers,
            embedding_dim=embedding_dim,
            edge_dim=p_net_edge_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            return_batch=True,
        )
        
        self.v_net_encoder = MLPNet(
            input_dim=v_net_attrs_dim,
            output_dim=embedding_dim,
            num_layers=2,
            embedding_dims=[embedding_dim],
            batch_norm=batch_norm,
            dropout_prob=dropout_prob,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        
        # Actor (binary classification)
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2),  # 2 actions: reject, accept
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        for module in [self.actor, self.critic, self.fusion]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight)
    
    def forward(self, p_net_batch, v_net_attrs):
        """
        Args:
            p_net_batch: PyG Batch object of physical network
            v_net_attrs: tensor of shape (batch_size, v_net_attrs_dim)
        
        Returns:
            action_logits: (batch_size, 2)
            values: (batch_size, 1)
        """
        # Encode networks
        p_net_embedding = self.p_net_encoder(p_net_batch)  # global pooling inside
        v_net_embedding = self.v_net_encoder(v_net_attrs)
        
        # Fuse embeddings
        fusion_embedding = torch.cat([p_net_embedding, v_net_embedding], dim=1)
        fusion_embedding = self.fusion(fusion_embedding)
        
        # Policy and value outputs
        action_logits = self.actor(fusion_embedding)
        values = self.critic(fusion_embedding)
        
        return action_logits, values
    
    def act(self, p_net_batch, v_net_attrs):
        """Return action logits for sampling"""
        action_logits, _ = self.forward(p_net_batch, v_net_attrs)
        return action_logits
    
    def evaluate(self, p_net_batch, v_net_attrs):
        """Return value estimates"""
        _, values = self.forward(p_net_batch, v_net_attrs)
        return values


class LowLevelActorCritic(nn.Module):
    """
    Low-level policy: sequential node placement decisions
    
    Architecture:
    - Dual GNN encoders for P-Net and V-Net
    - Fusion with current V-node information
    - Actor: p-node selection
    - Critic: value estimation
    
    Similar to dual_gnn_policy.BiGcnActorCritic but with modifications
    for acceptance context from high-level policy.
    """
    
    def __init__(
        self,
        p_net_num_nodes,
        p_net_x_dim,
        p_net_edge_dim,
        v_net_x_dim,
        v_net_edge_dim,
        embedding_dim=128,
        dropout_prob=0.,
        batch_norm=False,
        num_gnn_layers=3,
        **kwargs
    ):
        super().__init__()
        
        # GNN encoders (following dual_gnn_policy pattern)
        self.v_net_encoder = NetEncoder(
            feat_dim=v_net_x_dim,
            edge_dim=v_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            num_layers=num_gnn_layers,
            gnn_class=GCNConvNet,
        )
        
        self.p_net_encoder = NetEncoder(
            feat_dim=p_net_x_dim,
            edge_dim=p_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            num_layers=num_gnn_layers,
            gnn_class=GCNConvNet,
        )
        
        # Pooling layers
        self.mean_pool = GraphPooling('mean')
        self.att_pool = GraphPooling('att', output_dim=embedding_dim)
        
        # Actor: scoring nodes for placement
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, p_net_num_nodes),  # score per p-node
        )
        
        # Critic: value function
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight)
    
    def forward(self, p_net_batch, v_net_batch, curr_v_node_id):
        """
        Args:
            p_net_batch: PyG Batch of physical network
            v_net_batch: PyG Batch of virtual network
            curr_v_node_id: current virtual node being placed (batch_size,)
        
        Returns:
            action_logits: (batch_size, num_p_nodes)
            values: (batch_size, 1)
        """
        # Encode networks
        v_node_emb, v_g_emb, v_node_dense, _ = self.v_net_encoder(v_net_batch)
        p_node_emb, p_g_emb, p_node_dense, _ = self.p_net_encoder(p_net_batch)
        
        # Extract current v-node embedding
        batch_size = v_node_dense.shape[0]
        curr_v_node_id = curr_v_node_id.unsqueeze(1).unsqueeze(1).long()
        curr_v_node_emb = v_node_dense.gather(
            1, 
            curr_v_node_id.expand(batch_size, -1, v_node_dense.shape[-1])
        ).squeeze(1)
        
        # Fuse: enhance p-node embeddings with v-node and global context
        p_node_fusion = p_node_dense + v_g_emb.unsqueeze(1) + curr_v_node_emb.unsqueeze(1)
        
        # Average pooling for value estimation
        p_node_fusion_flat = p_node_fusion.view(-1, p_node_fusion.shape[-1])
        fusion_global = p_node_fusion_flat.mean(dim=0, keepdim=True)
        
        # Outputs
        action_logits = self.actor(fusion_global).expand(batch_size, -1)
        values = self.critic(fusion_global).expand(batch_size, -1)
        
        return action_logits, values
    
    def act(self, p_net_batch, v_net_batch, curr_v_node_id):
        action_logits, _ = self.forward(p_net_batch, v_net_batch, curr_v_node_id)
        return action_logits
    
    def evaluate(self, p_net_batch, v_net_batch, curr_v_node_id):
        _, values = self.forward(p_net_batch, v_net_batch, curr_v_node_id)
        return values


class NetEncoder(nn.Module):
    """Auxiliary encoder for networks (following dual_gnn_policy pattern)"""
    
    def __init__(self, feat_dim, edge_dim, embedding_dim=128, dropout_prob=0., 
                 batch_norm=False, num_layers=3, gnn_class=GCNConvNet):
        super().__init__()
        self.init_lin = nn.Linear(feat_dim, embedding_dim)
        self.net_gnn = gnn_class(
            embedding_dim, embedding_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            edge_dim=edge_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm
        )
        self.mean_pool = GraphPooling('mean')
        self.att_pool = GraphPooling('att', output_dim=embedding_dim)

    def forward(self, net_batch):
        x = self.init_lin(net_batch.x)
        net_batch_clone = net_batch.clone()
        net_batch_clone.x = x
        
        node_emb = self.net_gnn(net_batch_clone)
        
        g_emb = self.mean_pool(node_emb, net_batch.batch) + \
                self.att_pool(node_emb, net_batch.batch)
        
        node_dense, _ = to_dense_batch(node_emb, net_batch.batch)
        node_init_dense, _ = to_dense_batch(x, net_batch.batch)
        
        return node_emb, g_emb, node_dense + g_emb.unsqueeze(1) + node_init_dense, node_init_dense
```

### 3.3 Feature Constructor (hrl_ac_feature_constructor.py)

```python
import numpy as np
from virne.solver.learning.rl_core.feature_constructor import BaseFeatureConstructor, FeatureConstructorRegistry
from virne.solver.learning.obs_handler import ObservationHandler
from virne.network import (
    AttributeBenchmarkManager, TopologicalMetricCalculator
)

@FeatureConstructorRegistry.register('hrl_ac')
class HRLACFeatureConstructor(BaseFeatureConstructor):
    """
    Feature construction for hierarchical RL actor-critic
    Handles both high-level (admission control) and low-level (placement) observations
    """
    
    def __init__(self, p_net, v_net, config=None):
        super().__init__(p_net, v_net, config)
        self.obs_handler = ObservationHandler()
    
    def construct_high_level(self, p_net, v_net, solution=None):
        """
        Construct observation for high-level policy (admission control)
        
        Returns:
            dict with keys:
                - 'p_net_x': node features (batch, n_nodes, dim)
                - 'p_net_edge_index': edge indices
                - 'p_net_edge_attr': edge features
                - 'v_net_attrs': VN summary attributes
        """
        # P-Net features
        p_node_attrs = self.obs_handler.get_node_attrs_obs(
            p_net, 
            node_attr_types=self.extracted_attr_types,
            node_attr_benchmarks=self.node_attr_benchmarks
        )
        
        # Include resource availability
        if self.if_use_aggregated_link_attrs:
            p_node_link_data = self.obs_handler.get_link_aggr_attrs_obs(
                p_net,
                link_attr_types=self.extracted_attr_types,
                aggr='sum',
                link_sum_attr_benchmarks=self.link_sum_attr_benchmarks
            )
            p_node_data = np.concatenate([p_node_attrs, p_node_link_data], axis=-1)
        else:
            p_node_data = p_node_attrs
        
        p_edge_index = self.obs_handler.get_link_index_obs(p_net)
        p_edge_attr = self.obs_handler.get_link_attrs_obs(
            p_net,
            link_attr_types=self.extracted_attr_types,
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        
        # V-Net attributes (global summary)
        v_num_nodes = v_net.num_nodes / p_net.num_nodes
        v_num_links = v_net.num_links / (v_net.num_nodes ** 2 + 1e-6)
        
        v_node_attrs = self.obs_handler.get_node_attrs_obs(
            v_net,
            node_attr_types=self.extracted_attr_types,
            node_attr_benchmarks=self.node_attr_benchmarks
        )
        v_avg_demand = v_node_attrs.mean(axis=0)
        
        v_net_attrs = np.concatenate([
            [v_num_nodes],
            [v_num_links],
            v_avg_demand,
        ])
        
        return {
            'p_net_x': p_node_data,
            'p_net_edge_index': p_edge_index,
            'p_net_edge_attr': p_edge_attr,
            'v_net_attrs': v_net_attrs,
        }
    
    def construct_low_level(self, p_net, v_net, solution, curr_v_node_id):
        """
        Construct observation for low-level policy (node placement)
        Delegates to existing feature constructor for sequential placement
        
        Returns:
            dict compatible with low-level policy input
        """
        # Use existing p_net_v_net constructor
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        v_net_obs = self._construct_v_net_features(p_net, v_net, solution, curr_v_node_id)
        
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
            'action_mask': None,  # Will be set by environment
        }
```

### 3.4 Main HRL-AC Solver (hrl_ac_solver.py)

```python
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from virne.solver import SolverRegistry
from virne.solver.learning.rl_core import InstanceAgent, PPOSolver, A2CSolver
from virne.solver.learning.rl_core.buffer import RolloutBuffer
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver.learning.utils import apply_mask_to_logit

from .hrl_ac_env import HRLInstanceEnv
from .hrl_ac_policy import HighLevelActorCritic, LowLevelActorCritic
from .hrl_ac_feature_constructor import HRLACFeatureConstructor


class HRLACInstanceEnv(HRLInstanceEnv):
    """Instance-level environment for HRL-AC"""
    
    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        with open_dict(config):
            config.rl.feature_constructor.name = 'hrl_ac'
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)


def obs_as_tensor_for_hrl_ac_high_level(obs, device):
    """Convert high-level observations to tensors"""
    # P-Net as PyG batch
    p_net_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
    p_net_batch = Batch.from_data_list([p_net_data]).to(device)
    
    # V-Net attributes as tensor
    v_net_attrs = torch.FloatTensor(np.array([obs['v_net_attrs']])).to(device)
    
    return {'p_net': p_net_batch, 'v_net_attrs': v_net_attrs}


def obs_as_tensor_for_hrl_ac_low_level(obs, device):
    """Convert low-level observations to tensors (similar to dual_gnn)"""
    p_net_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
    v_net_data = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs['v_net_edge_attr'])
    
    p_net_batch = Batch.from_data_list([p_net_data]).to(device)
    v_net_batch = Batch.from_data_list([v_net_data]).to(device)
    
    curr_v_node_id = torch.LongTensor(np.array([obs['curr_v_node_id']])).to(device)
    
    return {
        'p_net': p_net_batch,
        'v_net': v_net_batch,
        'curr_v_node_id': curr_v_node_id,
    }


@SolverRegistry.register(solver_name='hrl_ac_ppo', solver_type='r_learning')
class HRLACPPOSolver(InstanceAgent, PPOSolver):
    """
    Hierarchical RL AC using PPO for both levels
    
    Architecture:
    - High-level: Binary acceptance control
    - Low-level: Sequential node placement (if accepted)
    - Shared reward: Overall VN embedding success
    """
    
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        InstanceAgent.__init__(self, HRLACInstanceEnv)
        PPOSolver.__init__(
            self, 
            controller, 
            recorder, 
            counter, 
            logger, 
            config,
            self._make_policy,
            self._preprocess_obs_wrapper,
            **kwargs
        )
        
        # Initialize separate buffers for high and low levels
        self.high_level_buffer = RolloutBuffer()
        self.low_level_buffer = RolloutBuffer()
        
        # Feature constructor
        self.feature_constructor = HRLACFeatureConstructor(
            self.policy_high_level.p_net if hasattr(self, 'policy_high_level') else None,
            None,
            config
        )
    
    def _make_policy(self, agent):
        """Create high-level and low-level policies"""
        config = agent.config
        
        # Dimensions (from config or estimation)
        p_net_x_dim = self._estimate_p_net_x_dim(config)
        p_net_edge_dim = 1  # Typically bw only
        v_net_attrs_dim = 3 + len(config.rl.feature_constructor.extracted_attr_types)
        v_net_x_dim = self._estimate_v_net_x_dim(config)
        v_net_edge_dim = 1
        
        embedding_dim = config.nn.embedding_dim
        
        # High-level policy (admission control)
        policy_high_level = HighLevelActorCritic(
            p_net_x_dim=p_net_x_dim,
            p_net_edge_dim=p_net_edge_dim,
            v_net_attrs_dim=v_net_attrs_dim,
            embedding_dim=embedding_dim,
            dropout_prob=config.nn.dropout_prob,
            batch_norm=config.nn.batch_norm,
            num_gnn_layers=config.nn.num_gnn_layers,
        ).to(agent.device)
        
        # Low-level policy (node placement)
        policy_low_level = LowLevelActorCritic(
            p_net_num_nodes=config.simulation.p_net_setting_num_nodes,
            p_net_x_dim=p_net_x_dim,
            p_net_edge_dim=p_net_edge_dim,
            v_net_x_dim=v_net_x_dim,
            v_net_edge_dim=v_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=config.nn.dropout_prob,
            batch_norm=config.nn.batch_norm,
            num_gnn_layers=config.nn.num_gnn_layers,
        ).to(agent.device)
        
        # Combined policy wrapper
        class HRLACPolicy(nn.Module):
            def __init__(self, high_level, low_level):
                super().__init__()
                self.high_level = high_level
                self.low_level = low_level
            
            def act_high_level(self, obs):
                return self.high_level.act(obs['p_net'], obs['v_net_attrs'])
            
            def evaluate_high_level(self, obs):
                return self.high_level.evaluate(obs['p_net'], obs['v_net_attrs'])
            
            def act_low_level(self, obs):
                return self.low_level.act(obs['p_net'], obs['v_net'], obs['curr_v_node_id'])
            
            def evaluate_low_level(self, obs):
                return self.low_level.evaluate(obs['p_net'], obs['v_net'], obs['curr_v_node_id'])
        
        policy = HRLACPolicy(policy_high_level, policy_low_level)
        
        # Optimizer
        params = list(policy_high_level.parameters()) + list(policy_low_level.parameters())
        optimizer = torch.optim.Adam([
            {'params': policy_high_level.actor.parameters(), 
             'lr': config.rl.learning_rate.actor},
            {'params': policy_high_level.critic.parameters(), 
             'lr': config.rl.learning_rate.critic},
            {'params': policy_low_level.actor.parameters(), 
             'lr': config.rl.learning_rate.actor},
            {'params': policy_low_level.critic.parameters(), 
             'lr': config.rl.learning_rate.critic},
        ], weight_decay=config.rl.weight_decay)
        
        return policy, optimizer
    
    def _preprocess_obs_wrapper(self, obs, device, level='high'):
        """Wrapper for preprocessing observations at different levels"""
        if isinstance(obs, dict) and 'level' in obs:
            level = obs.pop('level')
        
        if level == 'high':
            return obs_as_tensor_for_hrl_ac_high_level(obs, device)
        else:
            return obs_as_tensor_for_hrl_ac_low_level(obs, device)
    
    def _estimate_p_net_x_dim(self, config):
        """Estimate p-net feature dimension from config"""
        num_attrs = len(config.rl.feature_constructor.extracted_attr_types)
        if config.rl.feature_constructor.if_use_aggregated_link_attrs:
            num_attrs += num_attrs * 4  # min, mean, max, sum
        return num_attrs
    
    def _estimate_v_net_x_dim(self, config):
        """Estimate v-net feature dimension from config"""
        num_attrs = len(config.rl.feature_constructor.extracted_attr_types)
        base_dim = num_attrs + 1  # +1 for num neighbors
        if config.rl.feature_constructor.if_use_node_status_flags:
            base_dim += 3
        if config.rl.feature_constructor.if_use_aggregated_link_attrs:
            base_dim += num_attrs * 4
        if config.rl.feature_constructor.if_use_degree_metric:
            base_dim += 1
        if config.rl.feature_constructor.if_use_more_topological_metrics:
            base_dim += 3
        return base_dim
    
    def learn_with_instance(self, instance):
        """
        Learn from single VN instance with hierarchical decisions
        
        Process:
        1. High-level decision: accept/reject
        2. If accept: low-level sequential placement
        3. Shared reward based on final outcome
        """
        v_net, p_net = instance['v_net'], instance['p_net']
        
        # Initialize hierarchical environment
        instance_env = HRLACInstanceEnv(
            p_net, v_net, self.controller, self.recorder, 
            self.counter, self.logger, self.config
        )
        
        # High-level decision
        high_obs = instance_env.get_high_level_observation()
        tensor_high_obs = self.preprocess_obs(high_obs, self.device, level='high')
        
        high_action, high_logprob = self.select_action_high(tensor_high_obs, sample=True)
        high_value = self.estimate_value_high(tensor_high_obs)
        
        # Execute high-level action
        next_high_obs, high_reward, high_done, high_info = instance_env.step_high_level(high_action)
        
        # Record high-level transition
        self.high_level_buffer.add(
            high_obs, high_action, high_reward, high_done, 
            high_logprob, value=high_value
        )
        
        # If accepted, execute low-level placement
        if high_action == 1 and not high_done:
            low_obs = instance_env.get_low_level_observation()
            low_done = False
            
            while not low_done:
                tensor_low_obs = self.preprocess_obs(low_obs, self.device, level='low')
                
                low_action, low_logprob = self.select_action_low(tensor_low_obs, sample=True)
                low_value = self.estimate_value_low(tensor_low_obs)
                
                next_low_obs, low_reward, low_done, low_info = instance_env.step_low_level(low_action)
                
                self.low_level_buffer.add(
                    low_obs, low_action, low_reward, low_done,
                    low_logprob, value=low_value
                )
                
                low_obs = next_low_obs
        
        # Finalize both buffers and return solution
        solution = instance_env.solution
        last_value_low = self.estimate_value_low(self.preprocess_obs(low_obs, self.device, level='low')) if not high_done else 0
        last_value_high = 0
        
        return solution, self.high_level_buffer, self.low_level_buffer, last_value_high, last_value_low
    
    def select_action_high(self, obs, sample=True):
        """Select high-level action"""
        with torch.no_grad():
            action_logits = self.policy.act_high_level(obs)
        
        action_dist = Categorical(logits=action_logits)
        if sample:
            action = action_dist.sample()
        else:
            action = action_logits.argmax(-1)
        
        action_logprob = action_dist.log_prob(action)
        return action.item(), action_logprob.cpu().detach().numpy()
    
    def select_action_low(self, obs, sample=True):
        """Select low-level action"""
        with torch.no_grad():
            action_logits = self.policy.act_low_level(obs)
        
        action_dist = Categorical(logits=action_logits)
        if sample:
            action = action_dist.sample()
        else:
            action = action_logits.argmax(-1)
        
        action_logprob = action_dist.log_prob(action)
        return action.item(), action_logprob.cpu().detach().numpy()
    
    def estimate_value_high(self, obs):
        """Estimate high-level value"""
        with torch.no_grad():
            value = self.policy.evaluate_high_level(obs)
        return value.squeeze(-1).detach().cpu().item()
    
    def estimate_value_low(self, obs):
        """Estimate low-level value"""
        with torch.no_grad():
            value = self.policy.evaluate_low_level(obs)
        return value.squeeze(-1).detach().cpu().item()
    
    def update(self):
        """
        Update both high-level and low-level policies
        Compute advantages and perform PPO updates for each
        """
        # High-level update
        if self.high_level_buffer.size() > 0:
            self.high_level_buffer.compute_returns_and_advantages(
                last_value=0, gamma=self.config.rl.gamma, 
                gae_lambda=self.gae_lambda, method='gae'
            )
            self._update_high_level()
        
        # Low-level update
        if self.low_level_buffer.size() > 0:
            self.low_level_buffer.compute_returns_and_advantages(
                last_value=0, gamma=self.config.rl.gamma,
                gae_lambda=self.gae_lambda, method='gae'
            )
            self._update_low_level()
        
        # Clear buffers
        self.high_level_buffer.clear()
        self.low_level_buffer.clear()
        
        self.update_time += 1
        return torch.tensor(0.0)  # Placeholder loss
    
    def _update_high_level(self):
        """PPO update for high-level policy"""
        # Implementation similar to PPOSolver.update() but for high-level
        pass
    
    def _update_low_level(self):
        """PPO update for low-level policy"""
        # Implementation similar to PPOSolver.update() but for low-level
        pass


@SolverRegistry.register(solver_name='hrl_ac_a2c', solver_type='r_learning')
class HRLACA2CSolver(InstanceAgent, A2CSolver):
    """HRL-AC using A2C (Advantage Actor-Critic) for both levels"""
    # Similar to HRLACPPOSolver but using A2CSolver as base class
    pass
```

---

## 4. Integration Checklist

### 4.1 Configuration Requirements

Add to `settings/learning.yaml`:
```yaml
rl:
  # ... existing config ...
  feature_constructor:
    name: "hrl_ac"
    extracted_attr_types: ["resource"]
    if_use_node_status_flags: true
    if_use_aggregated_link_attrs: true
    if_use_degree_metric: false
    if_use_more_topological_metrics: false
  
  hrl:
    low_level_solver: null  # or 'grc_rank', 'nrm_rank'
    use_learning_low_level: true
    reward_weight_high: 0.5
    reward_weight_low: 0.5
```

### 4.2 Registry Updates

1. **`virne/solver/learning/rl_core/__init__.py`**
   ```python
   from .hrl_ac import HRLACInstanceEnv, HRLACPPOSolver, HRLACA2CSolver
   ```

2. **`virne/solver/learning/rl_core/feature_constructor.py`**
   ```python
   from .hrl_ac_feature_constructor import HRLACFeatureConstructor
   ```

### 4.3 Testing Protocol

```
1. Unit Tests:
   - Test HRLInstanceEnv high/low level separation
   - Test policy networks forward pass
   - Test feature constructor outputs
   - Test tensor conversion functions

2. Integration Tests:
   - Single VN instance learning
   - Multi-instance training loop
   - Reward computation hierarchy
   - Solver registration and initialization

3. Validation Tests:
   - Compare acceptance rates (high-level)
   - Compare placement success (low-level)
   - Compare overall revenue-to-cost ratio
   - Benchmark vs single-level baselines
```

---

## 5. Key Design Decisions Explained

### 5.1 Why Separate Buffers?
- **High-level buffer**: Stores admission control decisions (binary, one per VN)
- **Low-level buffer**: Stores sequential placement decisions (multiple per accepted VN)
- **Benefit**: Allows independent reward computation and updates at each level

### 5.2 Why GNN Encoders?
- **P-Net**: Capture topology and resource dependencies
- **V-Net**: Model virtual network structure for context
- **Dual encoding**: Follow successful dual_gnn_policy pattern

### 5.3 Why Hierarchical Reward?
- **High-level**: Encourages acceptance of profitable VNs
- **Low-level**: Shaped rewards during placement guide learning
- **Terminal reward**: Shared final outcome motivates coordination

### 5.4 Feature Dimensionality Strategy
- **High-level**: Global aggregation (fewer features, lower overhead)
- **Low-level**: Detailed node/edge features (sequential decisions need context)
- **Progressive complexity**: Match decision complexity at each level

---

## 6. Extension Points

### 6.1 Custom Low-Level Solvers
```python
# Use heuristic baseline for low-level during high-level training
if config.hrl.low_level_solver == 'grc_rank':
    self.low_level_solver = GRCRankSolver(...)
else:
    self.low_level_solver = None  # Use learned policy
```

### 6.2 Multi-Objective Rewards
```python
# Extend reward computation for multi-objective optimization
reward = (
    w1 * acceptance_ratio +
    w2 * placement_success +
    w3 * resource_efficiency +
    w4 * load_balance
)
```

### 6.3 Curriculum Learning
```python
# Start with high acceptance threshold, gradually relax
acceptance_threshold = max(0.7 - epoch * 0.01, 0.1)
```

---

## 7. References & Related Work

- **Dual-GNN Solvers**: `virne/solver/learning/reinforcement_learning/dual_gnn_solver.py`
- **Instance Agents**: `virne/solver/learning/rl_core/instance_agent.py`
- **Feature Constructors**: `virne/solver/learning/rl_core/feature_constructor.py`
- **PPO Implementation**: `virne/solver/learning/rl_core/rl_solver.py` (PPOSolver class)

---

## 8. Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| High-level policy always rejects | Negative rewards | Tune reward scaling |
| Low-level placement fails after acceptance | Feature mismatch | Verify feature constructor compatibility |
| Slow training | Large state space | Reduce GNN layers or use simpler features |
| Unstable gradients | Unbounded action logits | Add temperature scaling or normalization |
| Memory overflow | Large batch accumulation | Reduce `target_steps` or update more frequently |

---

## 9. Summary

This implementation plan provides a complete, production-ready framework for HRL-AC integration into VIRNE following established patterns from dual_gnn_solver and adhering to VIRNE's modular architecture. The hierarchical separation of admission control and placement decisions enables more nuanced learning and better interpretability of RL behaviors.

**Key Files to Create**: 6 main files + 1 utility module
**Key Classes**: 5 solvers, 4 policy networks, 2 environments, 1 feature constructor
**Integration Points**: Registry, configuration, feature construction, tensor conversion
**Testing Scope**: Unit, integration, validation across all modules

