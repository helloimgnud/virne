# Virne Learning Solver Documentation

This document provides a comprehensive overview of the learning-based solvers located in the `virne/solver/learning/` directory of the Virne project. These solvers employ varying techniques spanning Unsupervised Learning and Reinforcement Learning to address Virtual Network Embedding (VNE) problems.

## 1. Overview of Solvers

The learning models are structurally divided into basically two major paradigms: Reinforcement Learning (RL) and Unsupervised Learning.

### 1.1 Reinforcement Learning (RL) Solvers
RL in Virne revolves primarily around mapping virtual networks to physical substrates. The framework is highly modular, with core interfaces defined in `rl_core`:
* **Core Algorithms**: `PGSolver` (Policy Gradient), `A2CSolver` (Advantage Actor-Critic), `PPOSolver` (Proximal Policy Optimization), `DQNSolver` (Deep Q-Network), `DDPGSolver`.
* **Safe RL Solvers**: Focus on maintaining specific safety constraints while learning (e.g., `LagrangianPPOSolver`, `RewardCPOSolver`, `AdaptiveStateWiseSafePPOSolver`).
* **Seq2Seq Architecture Solvers**: Formulate the node mapping process as a sequence generation task, e.g., `A3CGcnSeq2SeqSolver`, `PpoGcnSeq2SeqSolver`, `PpoGatSeq2SeqSolver` (`ppo_gat_seq2seq+`).
* **MLP & CNN Solvers**: Feature extraction primarily utilizing Multilayer Perceptrons or Convolutional Neural Networks on state vectors `PgMlpSolver`, `PgCnnSolver`.
* **Graph Neural Network (GNN) Solvers**: Leverage the topological structure of networks for feature extraction, including:
  * **Dual GNN**: Extracting distinct graph representations using parallel GCNs (`A2CDualGcnSolver`).
  * **Heterogeneous GNN**: Employed in models like CONAL (`ConalSolver`) for distinct node/edge relations.
  * **GNN-MLP Combinations**: Features extracted via GNNs are projected into actions by MLPs (`A3cGcnSolver`).
* **Attention Mechanism Solvers**: `PpoAttSolver` uses attention networks over input feature representations.
* **MCTS (Monte Carlo Tree Search)**: Integrates search capability with RL guidance.

### 1.2 Unsupervised Learning Solvers
* **Hopfield Network Solver**: Utilizes Hopfield networks representing associative memory models to find optimal node configurations.
* **GAE (Graph Autoencoder) Clustering Solver**: Groups specific sub-networks or node attributes using latent feature representations.

## 2. Training and Inference Lifecycle

### 2.1 Training Workflow
The training pipeline across most RL agents follows a standardized approach defined within `OnlineAgent` and `InstanceAgent`.
1. **Interaction with Environment**: The agent loops over sub-instances (`InstanceEnv` or standard environments), generating episodes of state transitions (`obs, action, reward, next_obs, info`).
2. **Buffer Management**: Outputs during exploration (including states, actions, log probabilities, values) are stored in a `RolloutBuffer`. Time scaling utilizes metrics such as elapsed feature construction time.
3. **Reward and Advantage Calculation**: Often guided by an actor-critic model, terminal states are evaluated using Generic Advantage Estimation (GAE) to refine rewards vs. state values (`compute_returns_and_advantages`).
4. **Parameter Updates**: Once batch sizes or trajectories are sufficient, the buffer triggers an `update()` step propagating backward gradients (e.g., maximizing PPO clipped objectives). Both Actor and Critic networks modify their parameter weights.

### 2.2 Inference Workflow
When executing evaluating (`solve(instance)`), agents disable exploration noise / gradients (`torch.no_grad()` equivalent via sampling deterministic distributions depending on `sample=False` or passing deterministic flows):
1. **Pre-processing**: Observation maps corresponding to VNR and Substrate physical conditions are parsed.
2. **Action Selection**: Direct output parsing typically selects absolute argmax values rather than stochastic distributions. 
3. **Stepping**: The selected underlying actions are stepped through the actual VNE environment continuously until a sequence completes.

## 3. Important Parameters

Most algorithms define configurable structures exposed logically to Hydra (see `settings/`); key tunable elements internally consist of:
* **RL Parameters**:
  * `gamma`: Discount factor for future rewards (common: 0.99).
  * `gae_lambda`: Bias-variance tradeoff parameter for advantage estimation (common: ~0.95).
  * `batch_size` / `target_steps`: Determines precisely when network weights get updated.
  * `learning_rate`: Adjustments controlled occasionally by a shared Adam optimizer (`shared_adam.py`).
  * `num_epochs`: Iteration cycles per trajectory batch.
* **Network Parameters**:
  * Dimension of embedding sizes (`v_net_x_dim`, `p_net_x_dim`, `p_net_edge_dim`).
  * General neural network complexities (`hidden_dim`, layers counts).

## 4. Run Commands
Virne uses Hydra for configuration. To execute an experiment using a generic learning solver, standard execution modifies configurations logically via CLI:
```bash
python main.py solver.name=ppo_gat_seq2seq+ experiment.run_id=auto
```
This instructs `main.py` to instantiate the appropriate configuration profile (often stored in `settings/solver/`) and initiate the framework.

## 5. Agent Interaction

Agents coordinate to tackle different hierarchical stages of the decision-making process:

* **Admission Control (OnlineAgent)**: Operating at the macroeconomic scale, the `OnlineAgent` encounters incoming virtual network requests continuously. It functions equivalently to an "Admission Controller" making binary or queued selections of which graphs to process based on overall load capacities and long-term Revenue-to-Cost (R2C) predictions.
* **Resource Allocation / Mapping (InstanceAgent)**: After admission, individual VNRs are delegated to the `InstanceAgent`. This agent dives into the micro-perspective, iterating node-by-node (or block-by-block) applying placement logic against individual Substrate elements dynamically, handling single request lifecycles.

Such abstractions permit integrating them concurrently so hierarchical RL behaviors are intrinsically supported.
