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

        # ââ Override action space: binary {0=reject, 1=accept} ââââââââââââ
        self.action_space = spaces.Discrete(2)

        # ââ Build attribute benchmarks (FIX BUG 3) ââââââââââââââââââââââââ
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

        # ââ Observation handler (already set by RLBaseEnv, ref here) ââââââ
        # self.obs_handler is set by RLBaseEnv.__init__ â no need to re-create

        # ââ VNR-level reward trackers (mirrors hrl_ac source) âââââââââââââ
        self.global_timestep_count = 0
        self.global_moving_average_reward = 0
        self.global_cumulative_reward = 0
        self.actual_cumulative_reward = 0

        # ââ Build sub-solver ââââââââââââââââââââââââââââââââââââââââââââââ
        sub_solver_name = config.solver.get('sub_solver_name', 'nrm_rank')
        kwargs_for_sub = copy.deepcopy(kwargs)
        kwargs_for_sub['verbose'] = 0
        self._build_sub_solver(sub_solver_name, kwargs_for_sub, config)

    # ââ Sub-solver factory âââââââââââââââââââââââââââââââââââââââââââââââââ
    def _build_sub_solver(self, sub_solver_name: str, kwargs_for_sub: dict,
                          config: DictConfig):
        """
        Factory: instantiate correct lower-level solver.
          'nrm_rank'  â NRMRankSolver  (fast, deterministic)
          'grc_rank'  â GRCRankSolver  (slightly slower)
          'hrl_ra'    â HrlRaSolver    (pretrained RL lower agent, lazy import)
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

    # ââ Core step âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def step(self, action: int):
        """
        Upper agent makes ONE binary decision per VNR.

        action=1 (accept): call sub_solver; pass its Solution to parent step().
        action=0 (reject): create a failed Solution with early_rejection=True.

        INVARIANT: Only this method sets solution['early_rejection'] = True.
        Sub-solver failure â result=False, early_rejection stays False.
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

    # ââ Reward ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    def compute_reward(self, record: dict) -> float:
        """
        Shaped reward to prevent degenerate always-accept / always-reject policy.

        Early rejection (upper agent NO)  â 0.0     (neutral)
        Accepted + heuristic SUCCESS      â (w_a+w_b) * (revenue/benchmark) * r2c
        Accepted + heuristic FAILURE      â -0.01 * num_v_nodes

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

        # ââ Average-reward baseline (variance reduction) ââââââââââââââââââ
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
        # directly to the buffer â it does not add to self.cumulative_reward.
        return adjusted_reward

    # ââ Observation âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
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
