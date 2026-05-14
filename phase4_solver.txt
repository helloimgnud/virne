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
    # FIX BUG 14: No learn() override  RLSolver.learn() used as-is.
