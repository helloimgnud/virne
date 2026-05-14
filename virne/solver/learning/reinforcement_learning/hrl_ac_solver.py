import os
import torch
import numpy as np
import torch.nn as nn
from gym import spaces
from torch_geometric.data import Batch

from virne.solver import SolverRegistry
from virne.solver.learning.rl_core import OnlineAgent, PPOSolver, RolloutBuffer
from virne.solver.learning.rl_core.policy_builder import OptimizerBuilder, PolicyBuilder
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.learning.rl_core.online_rl_environment import SolutionStepRLEnv
from virne.core.solution import Solution

# We will define a simple ActorCritic that maps (p_net, v_net) into a probability of acceptance (0 or 1)
# You can expand this to use the exact DeepEdgeFeatureGAT if needed.
class ActorCritic(nn.Module):
    def __init__(self, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128):
        super(ActorCritic, self).__init__()
        from torch_geometric.nn import GATConv, global_mean_pool
        
        # Simple GAT-based encoders
        self.p_net_gnn = GATConv(p_net_feature_dim, embedding_dim, edge_dim=p_net_edge_dim)
        self.v_net_gnn = GATConv(v_net_feature_dim, embedding_dim, edge_dim=v_net_edge_dim)
        
        # MLP for outputting [Reject_logit, Accept_logit]
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2)
        )
        
        self.critic_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def encode(self, obs):
        p_x = self.p_net_gnn(obs['p_net'].x, obs['p_net'].edge_index, obs['p_net'].edge_attr)
        v_x = self.v_net_gnn(obs['v_net'].x, obs['v_net'].edge_index, obs['v_net'].edge_attr)
        
        p_global = global_mean_pool(p_x, obs['p_net'].batch)
        v_global = global_mean_pool(v_x, obs['v_net'].batch)
        
        return torch.cat([p_global, v_global], dim=-1)

    def act(self, obs):
        features = self.encode(obs)
        return self.mlp(features)
        
    def evaluate(self, obs):
        features = self.encode(obs)
        return self.critic_mlp(features)

def make_policy(agent, **kwargs):
    feature_dim_config = PolicyBuilder.get_feature_dim_config(agent.config)
    # The actual dims might require +3 for extra attributes depending on config
    policy = ActorCritic(
        p_net_feature_dim=feature_dim_config['p_net_x_dim'] + 3,
        p_net_edge_dim=feature_dim_config['p_net_edge_dim'],
        v_net_feature_dim=feature_dim_config['v_net_x_dim'] + 4, # node attrs + lifetime
        v_net_edge_dim=feature_dim_config['v_net_edge_dim'],
        embedding_dim=128
    ).to(agent.device)
    optimizer = OptimizerBuilder.build_optimizer(agent.config, policy)
    return policy, optimizer

def obs_as_tensor(obs, device):
    # one
    get_pyg_data = TensorConvertor.get_pyg_data
    if isinstance(obs, dict):
        p_net_data = get_pyg_data(obs['p_net_x'], obs['p_net_edge_index'], obs['p_net_edge_attr'])
        v_net_data = get_pyg_data(obs['v_net_x'], obs['v_net_edge_index'], obs['v_net_edge_attr'])
        obs_p_net = Batch.from_data_list([p_net_data]).to(device)
        obs_v_net = Batch.from_data_list([v_net_data]).to(device)
        obs_v_net_attrs = torch.FloatTensor(np.array([obs['v_net_attrs']])).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'v_net_attrs': obs_v_net_attrs}
    # batch
    elif isinstance(obs, list):
        p_net_data_list, v_net_data_list, v_net_attrs_list = [], [], []
        for observation in obs:
            p_net_data = get_pyg_data(observation['p_net_x'], observation['p_net_edge_index'], observation['p_net_edge_attr'])
            p_net_data_list.append(p_net_data)
            v_net_data = get_pyg_data(observation['v_net_x'], observation['v_net_edge_index'], observation['v_net_edge_attr'])
            v_net_data_list.append(v_net_data)            
            v_net_attrs_list.append(observation['v_net_attrs'])
        obs_p_net = Batch.from_data_list(p_net_data_list).to(device)
        obs_v_net = Batch.from_data_list(v_net_data_list).to(device)
        obs_v_net_attrs = torch.FloatTensor(np.array(v_net_attrs_list)).to(device)
        return {'p_net': obs_p_net, 'v_net': obs_v_net, 'v_net_attrs': obs_v_net_attrs}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

class HrlAcEnv(SolutionStepRLEnv):
    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs):
        super(HrlAcEnv, self).__init__(p_net, v_net_simulator, controller, recorder, counter, logger, config, **kwargs)
        self.action_space = spaces.Discrete(2)
        
        # Instantiate lower-level RA solver
        sub_solver_name = config.rl.get('sub_solver_name', 'ppo_gat_seq2seq+')
        logger.info(f'Employing {sub_solver_name} as sub solver in HRL-AC')
        SolverClass = SolverRegistry.get_solver(sub_solver_name)
        self.sub_solver = SolverClass(controller, recorder, counter, logger, config, **kwargs)
        
        pretrained_subsolver_model_path = config.rl.get('pretrained_subsolver_model_path', None)
        if pretrained_subsolver_model_path:
            logger.info('Loading pretrained lower-level RA agent...')
            self.sub_solver.load_model(pretrained_subsolver_model_path)
            
        if hasattr(self.sub_solver, 'eval'):
            self.sub_solver.eval()
            
    def step(self, action):
        if action == 1:
            instance = {'v_net': self.v_net, 'p_net': self.p_net}
            solution = self.sub_solver.solve(instance)
        else:
            solution = Solution(self.v_net)
            solution['early_rejection'] = True
            solution['result'] = False
        return super().step(solution)
        
    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        v_net_attrs = self._get_v_net_attrs_obs()
        
        # Pad V_net attributes to match nodes
        padding_v_net_attrs = np.expand_dims(v_net_attrs, axis=0).repeat(v_net_obs['x'].shape[0], axis=0)
        v_net_obs['x'] = np.concatenate((v_net_obs['x'], padding_v_net_attrs), axis=-1).astype(np.float32)
        
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_attrs': v_net_attrs,
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
        }
        
    def _get_v_net_attrs_obs(self):
        norm_lifetime = self.v_net.lifetime / self.v_net_simulator.v_sim_setting['lifetime']['scale']
        return np.array([norm_lifetime], dtype=np.float32)

@SolverRegistry.register(solver_name='hrl_ac', solver_type='r_learning')
class HrlAcSolver(OnlineAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        OnlineAgent.__init__(self)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.compute_return_method = 'gae'
