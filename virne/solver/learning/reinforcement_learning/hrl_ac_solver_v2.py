# ==============================================================================
# HRL-AC Solver v2 — Hierarchical Reinforcement Learning Admission Control
#
# A clean, self-contained port of the hrl-acra-main/solver/learning/hrl_ac
# codebase into the Virne framework.
#
# Key differences from hrl_ac_solver.py (v1):
#   - Registered as 'hrl_ac_v2' (does NOT conflict with existing 'hrl_ac').
#   - HrlAcEnvV2 inherits SolutionStepInstanceRLEnv (same as v1 HrlAcEnv) but
#     overrides get_observation() to produce the dual-graph dict that the HRL-AC
#     GNN policy consumes, mirroring env.py from the original repo exactly.
#   - The feature dimensions (+3 topological on p_net, +1 lifetime attr on v_net)
#     are computed faithfully through make_policy_v2 using PolicyBuilder helpers
#     instead of hard-coded magic numbers.
#   - ObsAsensor (obs_as_tensor_v2) is identical to the v1 version but lives in
#     this module so no cross-file dependency is needed.
#   - The reward function mirrors the original paper formula:
#       reward = (w_a + w_b) * revenue/benchmark * r2c   (success)
#       reward = -0.01 * num_v_nodes                      (failed attempt)
#       reward = 0                                         (early rejection)
#     and then is baseline-subtracted to reduce variance (running average trick).
#   - gamma=1, gae_lambda=0.98, norm_reward=True are set to match paper defaults.
# ==============================================================================

import torch
import numpy as np
import torch.nn as nn
from gym import spaces
from torch_geometric.data import Batch

from virne.solver import SolverRegistry
from virne.core.solution import Solution
from virne.solver.learning.rl_core import InstanceAgent, PPOSolver
from virne.solver.learning.rl_core.instance_rl_environment import SolutionStepInstanceRLEnv
from virne.solver.learning.rl_core.online_rl_environment import SolutionStepRLEnv
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder, OptimizerBuilder
from virne.solver.learning.neural_network.gnn import DeepEdgeFeatureGAT, GraphAttentionPooling, GraphPooling
from virne.solver.learning.neural_network.mlp import MLPNet
from virne.solver.learning.utils import get_pyg_data
from virne.network import AttributeBenchmarkManager, TopologicalMetricCalculator


# ---------------------------------------------------------------------------
# Neural Network: Encoder / Actor / Critic / ActorCritic
# Faithfully ported from hrl-acra-main/solver/learning/hrl_ac/net.py
# ---------------------------------------------------------------------------

class HrlAcEncoder(nn.Module):
    """Dual-GNN encoder for physical and virtual networks.

    Mirrors the released HRL-ACRA encoder (GRU modules removed for speed,
    as noted in the original paper's open-source release).
    """

    def __init__(
        self,
        p_net_feature_dim: int,
        p_net_edge_dim: int,
        v_net_feature_dim: int,
        v_net_edge_dim: int,
        embedding_dim: int = 128,
        dropout_prob: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.p_net_gnn = DeepEdgeFeatureGAT(
            p_net_feature_dim, embedding_dim,
            edge_dim=p_net_edge_dim,
            num_layers=5,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        )
        self.v_net_gnn = DeepEdgeFeatureGAT(
            v_net_feature_dim, embedding_dim,
            edge_dim=v_net_edge_dim,
            num_layers=3,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        )
        self.v_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_mean_pool = GraphPooling(aggr='mean')
        self.v_net_mean_pool = GraphPooling(aggr='mean')
        self.p_net_sum_pool  = GraphPooling(aggr='sum')
        self.v_net_sum_pool  = GraphPooling(aggr='sum')

    def forward(self, p_net_batch, v_net_batch):
        # Virtual network
        v_emb = self.v_net_gnn(v_net_batch)
        v_gap  = self.v_net_gap(v_emb, v_net_batch.batch)
        v_mean = self.v_net_mean_pool(v_emb, v_net_batch.batch)
        v_sum  = self.v_net_sum_pool(v_emb, v_net_batch.batch)
        v_global = v_gap + v_mean + v_sum

        # Physical network
        p_emb = self.p_net_gnn(p_net_batch)
        p_gap  = self.p_net_gap(p_emb, p_net_batch.batch)
        p_mean = self.p_net_mean_pool(p_emb, p_net_batch.batch)
        p_sum  = self.p_net_sum_pool(p_emb, p_net_batch.batch)
        p_global = p_gap + p_mean + p_sum

        return torch.cat([p_global, v_global], dim=-1)   # (B, 2*embedding_dim)


class HrlAcActor(nn.Module):
    """Actor head: produces logits for 2 admission actions (accept / reject)."""

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.encoder = HrlAcEncoder(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm,
        )
        self.net = MLPNet(
            embedding_dim * 2, 2,
            num_layers=3,
            embedding_dims=[embedding_dim * 2, embedding_dim],
            batch_norm=False,
        )

    def act(self, obs):
        return self.net(self.encoder(obs['p_net'], obs['v_net']))


class HrlAcCritic(nn.Module):
    """Critic head: produces scalar value estimate."""

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        self.encoder = HrlAcEncoder(
            p_net_feature_dim, p_net_edge_dim,
            v_net_feature_dim, v_net_edge_dim,
            embedding_dim, dropout_prob, batch_norm,
        )
        self.net = MLPNet(
            embedding_dim * 2, 1,
            num_layers=3,
            embedding_dims=[embedding_dim * 2, embedding_dim],
            batch_norm=False,
        )

    def evaluate(self, obs):
        return self.net(self.encoder(obs['p_net'], obs['v_net']))


class HrlAcActorCritic(nn.Module):
    """Composite policy with separate encoder weights for actor and critic."""

    def __init__(self, p_net_num_nodes, p_net_feature_dim, p_net_edge_dim,
                 v_net_feature_dim, v_net_edge_dim,
                 embedding_dim=128, dropout_prob=0.0, batch_norm=False):
        super().__init__()
        kwargs = dict(
            p_net_num_nodes=p_net_num_nodes,
            p_net_feature_dim=p_net_feature_dim,
            p_net_edge_dim=p_net_edge_dim,
            v_net_feature_dim=v_net_feature_dim,
            v_net_edge_dim=v_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        )
        self.actor  = HrlAcActor(**kwargs)
        self.critic = HrlAcCritic(**kwargs)

    def act(self, obs):
        return self.actor.act(obs)

    def evaluate(self, obs):
        return self.critic.evaluate(obs)


# ---------------------------------------------------------------------------
# Policy factory — called by PPOSolver.__init__ via the policy_builder arg
# ---------------------------------------------------------------------------

def make_policy_v2(agent, **kwargs):
    """Build HrlAcActorCritic and its Adam optimizer from Virne config."""
    feature_dim_cfg = PolicyBuilder.get_feature_dim_config(agent.config)
    nn_cfg          = PolicyBuilder.get_general_nn_config(agent.config)

    # p_net: base features + 3 topological metrics (degree, closeness, etc.)
    # v_net: base features + 1 lifetime attribute appended per-node by the env
    p_net_feature_dim = feature_dim_cfg['p_net_x_dim'] + 3
    p_net_edge_dim    = feature_dim_cfg['p_net_edge_dim']
    v_net_feature_dim = feature_dim_cfg['v_net_x_dim'] + 1   # +1 for lifetime
    v_net_edge_dim    = feature_dim_cfg['v_net_edge_dim']

    policy = HrlAcActorCritic(
        p_net_num_nodes=feature_dim_cfg['p_net_num_nodes'],
        p_net_feature_dim=p_net_feature_dim,
        p_net_edge_dim=p_net_edge_dim,
        v_net_feature_dim=v_net_feature_dim,
        v_net_edge_dim=v_net_edge_dim,
        **nn_cfg,
    ).to(agent.device)

    # Learning-rate scale (paper uses lr/10 for HRL-AC upper agent)
    scale = 0.1
    if hasattr(agent.config, 'rl'):
        rl_cfg = agent.config.rl
        scale  = float(getattr(rl_cfg, 'learning_rate_scale', scale))

    lr_cfg    = agent.config.rl.learning_rate
    lr_actor  = float(getattr(lr_cfg, 'actor',  lr_cfg))
    lr_critic = float(getattr(lr_cfg, 'critic', lr_cfg))

    optimizer = torch.optim.Adam(
        [
            {'params': policy.actor.parameters(),  'lr': lr_actor  * scale},
            {'params': policy.critic.parameters(), 'lr': lr_critic * scale},
        ],
        weight_decay=agent.config.rl.weight_decay,
    )
    return policy, optimizer


# ---------------------------------------------------------------------------
# obs_as_tensor: converts raw observation dict → batched PyG tensors
# ---------------------------------------------------------------------------

def obs_as_tensor_v2(obs, device):
    """Convert a raw observation (dict or list of dicts) to tensor dict.

    Expected keys in each obs dict:
        p_net_x, p_net_edge_index, p_net_edge_attr
        v_net_x, v_net_edge_index, v_net_edge_attr
        v_net_attrs   (scalar array, e.g. [norm_lifetime])
    """
    if isinstance(obs, dict):
        p_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
        v_data = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs['v_net_edge_attr'])
        return {
            'p_net':       Batch.from_data_list([p_data]).to(device),
            'v_net':       Batch.from_data_list([v_data]).to(device),
            'v_net_attrs': torch.FloatTensor(np.array([obs['v_net_attrs']])).to(device),
        }
    elif isinstance(obs, list):
        p_list, v_list, attrs_list = [], [], []
        for o in obs:
            p_list.append(get_pyg_data(o['p_net_x'], o['p_net_edge_index'], o['p_net_edge_attr']))
            v_list.append(get_pyg_data(o['v_net_x'], o['v_net_edge_index'], o['v_net_edge_attr']))
            attrs_list.append(o['v_net_attrs'])
        return {
            'p_net':       Batch.from_data_list(p_list).to(device),
            'v_net':       Batch.from_data_list(v_list).to(device),
            'v_net_attrs': torch.FloatTensor(np.array(attrs_list)).to(device),
        }
    else:
        raise TypeError(f"obs_as_tensor_v2: unrecognized obs type {type(obs)}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HrlAcEnvV2(SolutionStepInstanceRLEnv):
    """Upper-level (admission control) environment for HRL-AC.

    Action space: Discrete(2)
        action=1 → accept: delegate to sub_solver
        action=0 → reject: mark as early_rejection

    Observation: dual-graph dict matching obs_as_tensor_v2 expectations.

    Reward: paper formula with running-average baseline subtraction.
    """

    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)

        # Override to binary action space
        self.action_space = spaces.Discrete(2)

        # ── Sub-solver (lower-level RA agent) ──────────────────────────────
        sub_solver_name = self._get_cfg(config.solver, 'sub_solver_name', 'ppo_gat_seq2seq+')
        logger.info(f'[HrlAcEnvV2] Using sub-solver: {sub_solver_name}')

        if not SolverRegistry.has_solver(sub_solver_name):
            raise NotImplementedError(
                f'Sub-solver "{sub_solver_name}" is not registered in SolverRegistry. '
                f'Make sure it is imported before HrlAcSolverV2 is instantiated.'
            )
        SubSolverClass = SolverRegistry.get_solver(sub_solver_name)
        self.sub_solver = SubSolverClass(controller, recorder, counter, logger, config, **kwargs)

        pretrained_path = self._get_cfg(config.solver, 'pretrained_subsolver_model_path', None)
        if pretrained_path:
            logger.info(f'[HrlAcEnvV2] Loading pretrained sub-solver from {pretrained_path}')
            self.sub_solver.load_model(pretrained_path)
        else:
            logger.info('[HrlAcEnvV2] Sub-solver randomly initialized.')

        if hasattr(self.sub_solver, 'eval'):
            self.sub_solver.eval()

        # ── Observation benchmarks (mirroring original env.py) ─────────────
        p_net_attr_benchmarks = AttributeBenchmarkManager.get_benchmarks(
            p_net, node_attrs=True, link_attrs=True, link_sum_attrs=True
        )
        self.node_attr_benchmarks     = p_net_attr_benchmarks.node_attr_benchmarks
        self.link_attr_benchmarks     = p_net_attr_benchmarks.link_attr_benchmarks
        self.link_sum_attr_benchmarks = p_net_attr_benchmarks.link_sum_attr_benchmarks
        self.degree_benchmark         = max(dict(p_net.degree()).values()) or 1

        # ── Running reward state (for baseline subtraction) ─────────────────
        self.global_timestep_count       = 0
        self.global_cumulative_reward    = 0.0
        self.actual_cumulative_reward    = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cfg(cfg_obj, key, default):
        """Safely get a value from either a DictConfig or plain object."""
        if hasattr(cfg_obj, 'get'):
            return cfg_obj.get(key, default)
        return getattr(cfg_obj, key, default)

    # ------------------------------------------------------------------
    # Core RL interface
    # ------------------------------------------------------------------

    def step(self, action):
        if action == 1:
            # Accept: run lower-level solver
            instance  = {'v_net': self.v_net, 'p_net': self.p_net}
            solution  = self.sub_solver.solve(instance)
        else:
            # Reject
            solution  = Solution(self.v_net)
            solution['early_rejection'] = True
            solution['result']          = False

        self.solution = solution
        obs     = self.get_observation()
        reward  = self.compute_reward()
        info    = self.get_info(self.solution.to_dict())
        return obs, reward, True, info

    def compute_reward(self):
        """Paper reward formula with running-average baseline subtraction."""
        solution = self.solution

        w_a = 1.0
        # Lifetime weight — gracefully fall back to 1 if v_sim_setting missing
        try:
            lifetime_scale = self.config.v_sim_setting['lifetime']['scale']
        except (AttributeError, KeyError, TypeError):
            lifetime_scale = solution.get('v_net_lifetime', 1.0) or 1.0

        lifetime = solution.get('v_net_lifetime', lifetime_scale)
        w_b = lifetime / lifetime_scale if lifetime_scale else 1.0

        revenue_benchmark = 100.0

        if solution['result']:
            basic_reward = solution['v_net_revenue'] / revenue_benchmark
            reward = (w_a + w_b) * basic_reward * solution['v_net_r2c_ratio']
        elif not solution.get('early_rejection', False):
            # Accepted but sub-solver failed
            basic_reward = self.v_net.total_resource_demand / revenue_benchmark
            reward = -0.01 * self.v_net.num_nodes
        else:
            # Early rejection
            reward = 0.0

        self.actual_cumulative_reward += reward
        self.global_timestep_count    += 1
        self.global_cumulative_reward += reward

        running_avg = self.global_cumulative_reward / self.global_timestep_count
        baseline_reward = reward - running_avg

        self.extra_info_dict.update({
            'actual_reward':              reward,
            'actual_cumulative_reward':   self.actual_cumulative_reward,
            'global_cumulative_reward':   self.global_cumulative_reward,
            'average_reward_benchmark':   running_avg,
            'baseline_adjusted_reward':   baseline_reward,
        })

        # Also accumulate into v_net_reward tracked by the base env
        if hasattr(self, 'v_net_reward'):
            self.v_net_reward += reward
        if hasattr(self, 'cumulative_reward'):
            self.cumulative_reward += baseline_reward

        return baseline_reward

    def get_observation(self):
        """Build dual-graph observation dict.

        Structure mirrors hrl-acra-main/solver/learning/hrl_ac/env.py:
            - p_net: node features = [resource, degree, link_max, link_sum]
            - v_net: node features = [resource, degree, link_max, link_sum]
              then v_net_attrs (norm lifetime) is appended per-node.
        """
        p_obs     = self._get_p_net_obs()
        v_obs     = self._get_v_net_obs()
        v_attrs   = self._get_v_net_attrs_obs()

        # Append per-node lifetime attribute to every v-node row
        padding   = np.expand_dims(v_attrs, 0).repeat(v_obs['x'].shape[0], axis=0)
        v_obs['x'] = np.concatenate((v_obs['x'], padding), axis=-1).astype(np.float32)

        return {
            'p_net_x':          p_obs['x'],
            'p_net_edge_index': p_obs['edge_index'],
            'p_net_edge_attr':  p_obs['edge_attr'],
            'v_net_attrs':      v_attrs,
            'v_net_x':          v_obs['x'],
            'v_net_edge_index': v_obs['edge_index'],
            'v_net_edge_attr':  v_obs['edge_attr'],
        }

    def generate_action_mask(self):
        return np.ones(2, dtype=bool)

    # ------------------------------------------------------------------
    # Private observation helpers (port of original env.py)
    # ------------------------------------------------------------------

    def _get_p_net_obs(self):
        oh = self.obs_handler
        node_data  = oh.get_node_attrs_obs(self.p_net, node_attr_types=['resource'],
                                           node_attr_benchmarks=self.node_attr_benchmarks)
        degree     = oh.get_node_degree_obs(self.p_net, self.degree_benchmark)
        link_max   = oh.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource'],
                                                aggr='max',
                                                link_attr_benchmarks=self.link_attr_benchmarks)
        link_sum   = oh.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource'],
                                                aggr='sum',
                                                link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data  = np.concatenate((node_data, degree, link_max, link_sum), axis=-1)
        edge_index = oh.get_link_index_obs(self.p_net)
        edge_attr  = oh.get_link_attrs_obs(self.p_net, link_attr_types=['resource'],
                                           link_attr_benchmarks=self.link_attr_benchmarks)
        return {'x': node_data, 'edge_index': edge_index, 'edge_attr': edge_attr}

    def _get_v_net_obs(self):
        oh = self.obs_handler
        node_data  = oh.get_node_attrs_obs(self.v_net, node_attr_types=['resource'],
                                           node_attr_benchmarks=self.node_attr_benchmarks)
        degree     = oh.get_node_degree_obs(self.v_net, self.degree_benchmark)
        link_max   = oh.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'],
                                                aggr='max',
                                                link_attr_benchmarks=self.link_attr_benchmarks)
        link_sum   = oh.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'],
                                                aggr='sum',
                                                link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data  = np.concatenate((node_data, degree, link_max, link_sum), axis=-1)
        edge_index = oh.get_link_index_obs(self.v_net)
        edge_attr  = oh.get_link_attrs_obs(self.v_net, link_attr_types=['resource'],
                                           link_attr_benchmarks=self.link_attr_benchmarks)
        return {'x': node_data, 'edge_index': edge_index, 'edge_attr': edge_attr}

    def _get_v_net_attrs_obs(self):
        """Normalized lifetime scalar (matches original env.py)."""
        try:
            scale = self.v_net_simulator.v_sim_setting['lifetime']['scale']
        except AttributeError:
            # Fallback: read from config or use raw lifetime
            try:
                scale = self.config.v_sim_setting['lifetime']['scale']
            except (AttributeError, KeyError, TypeError):
                scale = 1.0
        norm_lifetime = self.v_net.lifetime / scale if scale else 0.0
        return np.array([norm_lifetime], dtype=np.float32)


# ---------------------------------------------------------------------------
# Solver — registered as 'hrl_ac_v2'
# ---------------------------------------------------------------------------

@SolverRegistry.register(solver_name='hrl_ac_v2', solver_type='r_learning')
class HrlAcSolverV2(InstanceAgent, PPOSolver):
    """Hierarchical RL Admission-Control Solver (v2).

    Upper-level PPO agent that decides whether to accept or reject each VNR.
    When a VNR is accepted, the decision is delegated to a pre-trained or
    randomly-initialized lower-level RA (Resource Allocation) sub-solver.

    Two-level environment design (critical for Virne compatibility):
        - Env (outer):  SolutionStepRLEnv — the online simulation loop that
          base_system.py instantiates with (p_net, v_net_simulator, ...). This
          manages the stream of incoming VNRs across the entire simulation.
        - InstanceEnv (inner): HrlAcEnvV2 — created per-VNR inside
          InstanceAgent.learn_with_instance / solve. Receives a single
          VirtualNetwork and runs the accept/reject decision.

    Usage (config solver.name):
        solver_name: hrl_ac_v2
        sub_solver_name: ppo_gat_seq2seq+          # any registered RL solver
        pretrained_subsolver_model_path: null       # or absolute path to .pkl

    Paper defaults (automatically applied):
        gamma       = 1.0
        gae_lambda  = 0.98
        norm_reward = True
        lr scale    = 0.1   (via config.rl.learning_rate_scale or hardcoded)
    """

    # Outer env: online simulation loop, receives (p_net, v_net_simulator, ...)
    # base_system.py reads this attribute to create the training environment.
    Env = SolutionStepRLEnv

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        # Inner (instance-level) env — used per VNR inside learn_with_instance
        InstanceAgent.__init__(self, HrlAcEnvV2)
        PPOSolver.__init__(
            self,
            controller, recorder, counter, logger, config,
            make_policy_v2,
            obs_as_tensor_v2,
            **kwargs,
        )
        # Paper-specified training hyper-parameters
        self.config.rl.gamma       = 1.0
        self.gae_lambda             = 0.98
        self.config.rl.norm_reward  = True
        self.compute_return_method  = 'gae'
