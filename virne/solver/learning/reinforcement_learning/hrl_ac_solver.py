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

from virne.solver.learning.neural_network.gnn import DeepEdgeFeatureGAT, GraphAttentionPooling, GraphPooling
from virne.solver.learning.neural_network.mlp import MLPNet

class Encoder(nn.Module):
    def __init__(self, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128, dropout_prob=0., batch_norm=False):
        super(Encoder, self).__init__()
        self.p_net_gnn = DeepEdgeFeatureGAT(p_net_feature_dim, embedding_dim, edge_dim=p_net_edge_dim, num_layers=5, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_gnn = DeepEdgeFeatureGAT(v_net_feature_dim, embedding_dim, edge_dim=v_net_edge_dim, num_layers=3, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.v_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_gap = GraphAttentionPooling(embedding_dim)
        self.p_net_mean_pool = GraphPooling(aggr='mean')
        self.v_net_mean_pool = GraphPooling(aggr='mean')
        self.p_net_sum_pool = GraphPooling(aggr='sum')
        self.v_net_sum_pool = GraphPooling(aggr='sum')

    def forward(self, p_net_batch, v_net_batch):
        v_net_node_embeddings = self.v_net_gnn(v_net_batch)
        v_net_gap_global_embedding = self.v_net_gap(v_net_node_embeddings, v_net_batch.batch)
        p_net_node_embeddings = self.p_net_gnn(p_net_batch)
        p_net_gap_global_embedding = self.p_net_gap(p_net_node_embeddings, p_net_batch.batch)
        p_net_mean_global_embedding = self.p_net_mean_pool(p_net_node_embeddings, p_net_batch.batch)
        v_net_mean_global_embedding = self.v_net_mean_pool(v_net_node_embeddings, v_net_batch.batch)
        p_net_sum_global_embedding = self.p_net_sum_pool(p_net_node_embeddings, p_net_batch.batch)
        v_net_sum_global_embedding = self.v_net_sum_pool(v_net_node_embeddings, v_net_batch.batch)
        p_net_global_embedding = p_net_gap_global_embedding + p_net_mean_global_embedding + p_net_sum_global_embedding
        v_net_global_embedding = v_net_gap_global_embedding + v_net_mean_global_embedding + v_net_sum_global_embedding
        fusion_embedding = torch.concat([p_net_global_embedding, v_net_global_embedding], dim=-1) 
        return fusion_embedding

class ActorCritic(nn.Module):
    def __init__(self, p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim=128):
        super(ActorCritic, self).__init__()
        self.encoder = Encoder(p_net_feature_dim, p_net_edge_dim, v_net_feature_dim, v_net_edge_dim, embedding_dim)
        embedding_dims = [embedding_dim*2, embedding_dim]
        self.mlp = MLPNet(embedding_dim*2, 2, num_layers=3, embedding_dims=embedding_dims, batch_norm=False)
        self.critic_mlp = MLPNet(embedding_dim*2, 1, num_layers=3, embedding_dims=embedding_dims, batch_norm=False)

    def act(self, obs):
        features = self.encoder(obs['p_net'], obs['v_net'])
        return self.mlp(features)
        
    def evaluate(self, obs):
        features = self.encoder(obs['p_net'], obs['v_net'])
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

        self.global_timestep_count = 0
        self.global_cumulative_reward = 0
        self.actual_cumulative_reward = 0
            
    def compute_reward(self, solution):
        r"""Calculate deserved reward according to the result of taking action."""
        w_a = 1
        w_b = solution['v_net_lifetime'] / self.v_net_simulator.v_sim_setting['lifetime']['scale']
        revenue_benchmark = 100
        if solution['result']:
            basic_reward = solution['v_net_revenue'] / revenue_benchmark
            weight = w_a + w_b
            reward = weight * basic_reward * solution['v_net_r2c_ratio']
        elif (not solution['result']) and (not solution.get('early_rejection', False)):
            basic_reward = self.v_net.total_resource_demand / revenue_benchmark
            reward = - 0.01 * (self.v_net.num_nodes)
        else:
            reward = 0
        self.actual_cumulative_reward += reward
        self.v_net_reward += reward
        self.global_timestep_count += 1
        self.global_cumulative_reward += reward
        average_reward = reward - self.global_cumulative_reward / self.global_timestep_count
        self.extra_record_info.update({
            'actual_cumulative_reward': self.actual_cumulative_reward,
            'global_cumulative_reward': self.global_cumulative_reward,
            'average_reward_benchmark': self.global_cumulative_reward / self.global_timestep_count,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': average_reward,
            'actual_reward': reward,
        })
        self.cumulative_reward += average_reward
        return average_reward

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

    def _get_p_net_obs(self):
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_degree = self.obs_handler.get_node_degree_obs(self.p_net, self.degree_benchmark)
        p_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource'], aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data = np.concatenate((node_data, p_node_degree, p_node_link_max_resource, p_node_link_sum_resource), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        link_data = self.obs_handler.get_link_attrs_obs(self.p_net, link_attr_types=['resource'], link_attr_benchmarks=self.link_attr_benchmarks)
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data
        }
        return p_net_obs

    def _get_v_net_obs(self):
        node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=['resource'], node_attr_benchmarks=self.node_attr_benchmarks)
        v_node_degree = self.obs_handler.get_node_degree_obs(self.v_net, self.degree_benchmark)
        v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='max', link_attr_benchmarks=self.link_attr_benchmarks)
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data = np.concatenate((node_data, v_node_degree, v_node_link_max_resource, v_node_link_sum_resource), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        link_data = self.obs_handler.get_link_attrs_obs(self.v_net, link_attr_types=['resource'], link_attr_benchmarks=self.link_attr_benchmarks)
        v_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data,
        }
        return v_net_obs

    def _get_v_net_attrs_obs(self):
        norm_lifetime = self.v_net.lifetime / self.v_net_simulator.v_sim_setting['lifetime']['scale']
        return np.array([norm_lifetime], dtype=np.float32)

@SolverRegistry.register(solver_name='hrl_ac', solver_type='r_learning')
class HrlAcSolver(OnlineAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        OnlineAgent.__init__(self)
        PPOSolver.__init__(self, controller, recorder, counter, logger, config, make_policy, obs_as_tensor, **kwargs)
        self.compute_return_method = 'gae'
