import torch
import torch.nn as nn
from virne.solver.learning.neural_network.gnn import DeepEdgeFeatureGAT, GraphAttentionPooling, GraphPooling
from virne.solver.learning.neural_network.mlp import MLPNet

class HrlAcEncoder(nn.Module):
    """Dual-GNN encoder for physical and virtual networks."""

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

    def forward(self, p_net_batch, v_net_batch):
        # Virtual network
        v_emb = self.v_net_gnn(v_net_batch)
        v_gap  = self.v_net_gap(v_emb, v_net_batch.batch)
        v_mean = self.v_net_mean_pool(v_emb, v_net_batch.batch)
        v_global = v_gap + v_mean

        # Physical network
        p_emb = self.p_net_gnn(p_net_batch)
        p_gap  = self.p_net_gap(p_emb, p_net_batch.batch)
        p_mean = self.p_net_mean_pool(p_emb, p_net_batch.batch)
        p_global = p_gap + p_mean

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
