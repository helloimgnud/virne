# Explaining `ppo_gat_seq2seq+` Solver in Virne

This document systematically explains the `ppo_gat_seq2seq+` learning solver, detailing its location, underlying architecture, function, and relationship with other components in the Virne framework.

## 1. Solver Identify and Location
* **Solver Name**: `ppo_gat_seq2seq+`
* **Class Configuration**: `PpoGatSeq2SeqSolver`
* **File Location**: `virne/solver/learning/reinforcement_learning/gnn_seq2seq_solver/gnn_seq2seq_solver.py`
* **Policy Architecture**: `GATSeq2SeqActorCritic` (located in `policy_with_encoder.py`)

## 2. Core Architecture Concept
The `ppo_gat_seq2seq+` leverages two powerful Neural Network formulations merged perfectly with a robust Reinforcement Learning algorithm:
* **Algorithm (PPO)**: Relies on Proximal Policy Optimization (PPO), deriving from `virne/solver/learning/rl_core/rl_solver.py`. It maximizes cumulative long-term rewards by constraining the policy's update step (via clipping) to prevent disastrous performance drops.
* **Feature Extraction (GAT)**: Employs Graph Attention Networks (GATs). Instead of just averaging features of neighbors (like GCNs), GATs natively assign attention coefficients (weights) indicating the relative importance of varying linked neighbors, enabling fine-grained insights into Physical or Virtual Network topologies.
* **Mapping Strategy (Seq2Seq)**: Models the embedding of a VNR onto a physical substrate as a Sequential Mapping Process, heavily informed by Autoregressive processes. It translates the raw initial environment into continuous steps where one output maps the specific next target node.

## 3. Training Process (`learn_with_instance`)
When instructed to train via `learn_singly` within epochs, `ppo_gat_seq2seq+` interacts closely with `A3CGcnSeq2SeqInstanceEnv`.
1. **Initial Encoding**: First, it acquires a unified view of the environment (`encoder_obs`). It passes features through the GAT-equipped encoder, which synthesizes topological outputs.
2. **Sequential Step Generation**: An auto-regressive process is kickstarted. The RL environment yields state observation maps (`instance_obs`), which incorporates:
   * Remaining unresolved VNR mappings.
   * `action_mask`: Highlighting exclusively safe/viable Substrate mappings for the current target.
   * `encoder_outputs` and `hidden_state` generated historically.
3. **Execution**: The Actor module extracts the values and queries the decoder matching current hidden states against GAT encoders, finalizing a probabilistic discrete output.
4. **Buffer Maintenance**: Every selected step registers action probabilities (`action_logprob`), values, and rewards back to a `RolloutBuffer`. After mapping terminates (`instance_done`), generic advantage estimation (GAE) handles final value derivations before triggering parameter refinements (`self.update()`) across networks.

## 4. Inference Process (`solve`)
During evaluation/testing (`solve(instance)`), learning metrics are bypassed (`no_grad` evaluations effectively apply):
1. **Initial Forward Pass**: Employs strictly the model's highest-confidence encoded graph state values (`sample=False`). 
2. **Looping execution**: Identical to training, it propagates observations cyclically by injecting hidden states backwards per-step into the network untill all targeted VNR components acquire assigned physical resources or unequivocally fail constraints. 
3. **Yielding Data**: Produces concrete `solution` artifacts directly matching the solver framework logic, ready for success metric counters (AC ratio, R2C predictions).

## 5. Main Parameters
Typically passed via Hydra settings config (e.g., `algorithm`, `gamma`, `lr_actor`) mapped directly into `PolicyBuilder`.
Unique parameters heavily influencing `ppo_gat_seq2seq+`:
* `p_net_num_nodes`, `p_net_x_dim`, `p_net_edge_dim`, `v_net_x_dim`.
* Shared dimensions `general_nn_config` dictates how deep or wide Attention layers exist inside `GATSeq2SeqActorCritic`.

## 6. How it interacts with Agents
The `PpoGatSeq2SeqSolver` structurally inherits from `PpoGcnSeq2SeqSolver` logically behaving as an `InstanceAgent`. 
This denotes precisely that the algorithm focuses solely on the micro-allocation (Node/Link mapping level per individual Virtual Network Request). 

In advanced topologies, this solver can explicitly pair with a separate `OnlineAgent` controlling macroeconomic admission heuristics. When the `OnlineAgent` approves a VNR for entry, it seamlessly relays payload structures down to `ppo_gat_seq2seq+` for exact embedding computations, functioning hierarchically.
