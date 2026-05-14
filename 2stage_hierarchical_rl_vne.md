# 2-Stage Hierarchical RL for VNE: Upper Agent + Lower Heuristic

## Overview

A hierarchical design where:
- **Upper Agent (RL):** decides once per VNR whether to accept or early-reject
- **Lower Agent (Heuristic/Metaheuristic):** runs only on accepted VNRs to produce node/link mapping

No existing solver in `virne` implements this. `flag_solver.py`, `advanced_policy.py`, and `tracks.py` are all empty placeholders suggesting it was planned but never built.

---

## Why `SolutionStepRLEnv` Is the Right Environment

`SolutionStepRLEnv.step()` accepts a full `Solution` object as the action, which matches the design contract exactly:

```python
def step(self, action):
    solution = action          # lower heuristic result passed as action
    if solution['result']:
        self.solution = solution                        # accept path
    else:
        self.rollback_for_failure(reason=failure_reason)  # reject path
    record = self.recorder.count(self.v_net, self.p_net, self.solution)
    reward = self.compute_reward(record)
    done = self.transit_obs()
    return self.get_observation(), reward, done, self.get_info(record)
```

- Acts **once per VNR** → matches upper agent's decision granularity
- Lower heuristic result comes in as `action` → env doesn't care how it was found
- `recorder`, `transit_obs()`, `rollback_for_failure()` all work as-is

### What Does NOT Need Changing
| Component | Status |
|---|---|
| `recorder.count()` | Works as-is |
| `transit_obs()` | Works as-is |
| `rollback_for_failure()` | Works as-is for both rejection types |
| `generate_action_mask()` | Already stubbed as `np.ones(2)` → binary {reject, accept} |

---

## Architecture

```
OnlineAgent (Upper RL Policy)
│
│  solve(instance):
│    1. Extract VNR-level features
│    2. RL policy → action: {0=reject, 1=accept}
│    3. if accept:
│         solution = lower_heuristic.solve(instance)
│         return solution
│       else:
│         solution = Solution.from_v_net(v_net)
│         solution['result'] = False
│         solution['early_rejection'] = True
│         return solution
│
└── SolutionStepRLEnv.step(solution)
      → recorder.count()
      → compute_reward()
      → transit_obs()
```

---

## What Needs to Be Built

### 1. VNR-Level Feature Constructor

Current `FeatureConstructorRegistry` is all node-level. A new constructor is needed:

```python
# Suggested features for upper agent input
{
    # VNR demand side
    'v_net_cpu_demand_sum': float,
    'v_net_bw_demand_sum': float,
    'v_net_num_nodes': int,
    'v_net_num_links': int,
    'v_net_max_cpu_demand': float,
    'v_net_max_bw_demand': float,

    # Physical network supply side
    'p_net_cpu_remaining_ratio': float,
    'p_net_bw_remaining_ratio': float,
    'p_net_avg_utilization': float,
    'p_net_inservice_count': int,
}
```

### 2. Upper Agent Policy

A simple MLP is sufficient — the decision is binary and the feature space is low-dimensional:

```python
class UpperAgentPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # logits for [reject, accept]
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

### 3. Lower Heuristic Contract

The lower heuristic must return a valid `Solution` object:

```python
solution = Solution.from_v_net(v_net)
solution['result'] = True | False
solution['node_slots'] = {v_node_id: p_node_id, ...}
solution['link_paths'] = {v_link: p_path, ...}
```

**Important:** if the heuristic fails internally, set `solution['result'] = False` but do NOT set `solution['early_rejection'] = True`. That flag is reserved for the upper agent's explicit reject action. The env uses this to distinguish failure types.

---

## Reward Design (Critical)

This is the biggest design risk. Naive reward causes the agent to always accept early in training because heuristics often succeed.

### Naive (baseline, slow to learn)
```python
reward = revenue - cost   # 0 if rejected
```

### Recommended: Shaped Reward
```python
if early_rejection:
    reward = 0.0              # neutral — preserves future capacity, no penalty
elif result:
    reward = r2c_ratio        # good: accepted and successfully embedded
else:
    reward = -penalty         # bad: accepted but heuristic failed, wasted resources
```

### Optional: Curriculum on Rejection
Mask the reject action during early training to force the agent to learn acceptance quality first, then gradually unlock rejection:

```python
def generate_action_mask(self):
    if self.curriculum_phase == 'accept_only':
        return np.array([False, True])   # only accept allowed
    return np.ones(2, dtype=bool)        # both allowed
```

---

## Key Risks

| Risk | Mitigation |
|---|---|
| Always-accept degenerate policy | Shape reward so failed acceptance has negative return |
| Always-reject degenerate policy | Use curriculum or add acceptance-rate regularization to reward |
| `early_rejection` flag collision | Only upper agent sets it; heuristic failures use `result=False` only |
| Feature staleness | Recompute p_net supply features fresh each VNR, not cached |
| Lower heuristic runtime in training | Heuristics are fast; metaheuristics may need iteration budget cap |

---

## File Locations to Create/Modify

```
virne/solver/learning/reinforcement_learning/
├── flag_solver.py              # implement upper agent solver here
├── advanced_policy.py          # implement UpperAgentPolicy MLP here
│
virne/solver/learning/rl_core/
├── feature_constructor.py      # add VNRLevelFeatureConstructor
├── reward_calculator.py        # add shaped reward for 2-stage design
│
virne/solver/learning/rl_core/
└── online_rl_environment.py    # SolutionStepRLEnv — use as-is
```

---

## References

- `SolutionStepRLEnv` — `virne/solver/learning/rl_core/online_rl_environment.py`
- `OnlineAgent` — `virne/solver/learning/rl_core/online_agent.py`
- `Solution` — `virne/core`
- `SolverRegistry` — `virne/solver`
