import torch
import torch.nn as nn
from torch_geometric.data import Batch

# ââ FIX BUG 11: Import existing DeepEdgeFeatureGAT, GraphAttentionPooling,
#    GraphPooling from virne's neural_network module.
#    Do NOT redefine them â that creates an incompatible parallel class.
from virne.solver.learning.neural_network.gnn import (
    DeepEdgeFeatureGAT, GraphPooling, GraphAttentionPooling)
from virne.solver.learning.rl_policy.net import MLPNet
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry


class HrlAcEncoder(nn.Module):
    """
    Dual GNN encoder for (p_net, v_net) pair.
    Identical logic to hrl_ac/net.py :: Encoder, but uses virne's existing
    DeepEdgeFeatureGAT instead of re-defining it.

    Output: concat([p_global, v_global]) â R^{2 * embedding_dim}
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
