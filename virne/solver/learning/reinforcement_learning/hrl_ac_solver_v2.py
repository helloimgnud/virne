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
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder, OptimizerBuilder
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.utils import get_pyg_data
from virne.network import AttributeBenchmarkManager, TopologicalMetricCalculator


obs_as_tensor_v2 = TensorConvertor.obs_as_tensor_for_hrl_ac_v2


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
        from omegaconf import open_dict
        with open_dict(config):
            config.rl.feature_constructor.name = 'p_net_v_net'
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)

        # Override to binary action space
        self.action_space = spaces.Discrete(2)

        # ── Sub-solver (lower-level RA agent) ──────────────────────────────
        sub_solver_name = self._get_cfg(config.solver, 'sub_solver_name', 'ppo_gat_seq2seq+')
        logger.info(f'[HrlAcEnvV2] Using sub-solver: {sub_solver_name}')

        try:
            SubSolverClass = SolverRegistry.get(sub_solver_name)
        except NotImplementedError:
            raise NotImplementedError(
                f'Sub-solver "{sub_solver_name}" is not registered in SolverRegistry. '
                f'Make sure it is imported before HrlAcSolverV2 is instantiated.'
            )
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
        """Build dual-graph observation dict using Virne's standard FeatureConstructor."""
        # 1. Get the rich 13-dimension observation from Virne's standard FeatureConstructor
        obs = self.feature_constructor.construct(self.p_net, self.v_net, self.solution, curr_v_node_id=0)
        
        # 2. Get the VNR's normalized lifetime
        v_attrs = self._get_v_net_attrs_obs()

        # 3. Append the lifetime to the virtual nodes
        padding   = np.expand_dims(v_attrs, 0).repeat(obs['v_net_x'].shape[0], axis=0)
        obs['v_net_x'] = np.concatenate((obs['v_net_x'], padding), axis=-1).astype(np.float32)
        obs['v_net_attrs'] = v_attrs

        return obs

    def generate_action_mask(self):
        return np.ones(2, dtype=bool)

    # ------------------------------------------------------------------
    # Private observation helpers
    # ------------------------------------------------------------------

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

    Compatible with the standard Virne InstanceAgent pattern:
        - No custom Env = ... override. The framework default (SolutionStepEnvironment)
          is used as the outer online env, just like ppo_dual_gat+ and all other
          InstanceAgent-based solvers.
        - HrlAcEnvV2 is passed only to InstanceAgent.__init__() as the inner
          per-VNR environment, which handles the accept/reject decision.

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

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        # Inner (instance-level) env — used per VNR inside learn_with_instance
        InstanceAgent.__init__(self, HrlAcEnvV2)
        PPOSolver.__init__(
            self,
            controller, recorder, counter, logger, config,
            PolicyBuilder.build_hrl_ac_policy,
            obs_as_tensor_v2,
            **kwargs,
        )
        # Paper-specified training hyper-parameters
        self.config.rl.gamma       = 1.0
        self.gae_lambda             = 0.98
        self.config.rl.norm_reward  = True
        self.compute_return_method  = 'gae'
