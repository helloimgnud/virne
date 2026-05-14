# `impl.md` Bug Analysis vs. Virne Codebase

> **Scope**: Only the **high-level admission-control path** is used.  
> Node mapping is handled by an external heuristic/meta-heuristic — the low-level RL policy (`LowLevelActorCritic`, `step_low_level`, `low_level_buffer`) is **dead code** and can be deleted, but several of its interactions still cause bugs in the surviving code.

---

## 🔴 Critical Bugs (will crash at runtime)

### BUG-1 — `open_dict` not imported in `hrl_ac_solver.py`

**Location**: `hrl_ac_solver.py`, `HRLACInstanceEnv.__init__` (impl.md line 661)

```python
class HRLACInstanceEnv(HRLInstanceEnv):
    def __init__(self, ...):
        with open_dict(config):   # ← NameError: open_dict is not imported
            config.rl.feature_constructor.name = 'hrl_ac'
```

`open_dict` comes from `omegaconf`. It is imported in `dual_gnn_solver.py` (`from omegaconf import DictConfig, open_dict`) but **not** in the solver file sketched by `impl.md`.

**Fix**: Add `from omegaconf import open_dict` to `hrl_ac_solver.py`.

---

### BUG-2 — `PPOSolver.__init__` does not accept `self._make_policy` / `self._preprocess_obs_wrapper` before they exist

**Location**: `hrl_ac_solver.py`, `HRLACPPOSolver.__init__` (impl.md lines 706–718)

```python
PPOSolver.__init__(
    self, ...,
    self._make_policy,           # bound method – OK
    self._preprocess_obs_wrapper, # bound method – OK
    **kwargs
)
```

Looking at `RLSolver.__init__` (line 96 in `rl_solver.py`):
```python
self.make_policy = make_policy
self.policy, self.optimizer = self.make_policy(self)   # called immediately
```

`_make_policy` itself calls `self._estimate_p_net_x_dim(config)` etc., which reference `config.simulation.p_net_setting_num_nodes` and `config.nn.*`. If those config paths don't exist (e.g. because Hydra hasn't fully composed them yet), this raises a `KeyError`/`MissingMandatoryValue` on init.  
This is an existing virne pattern issue, but the impl.md doesn't guard for it.

---

### BUG-3 — `last_value_low` references undefined `low_obs` when rejection occurs

**Location**: `hrl_ac_solver.py`, `learn_with_instance` (impl.md line 890)

```python
# If accepted, execute low-level placement
if high_action == 1 and not high_done:
    low_obs = ...
    while not low_done:
        ...

# Always executed:
last_value_low = self.estimate_value_low(
    self.preprocess_obs(low_obs, self.device, level='low')  # ← NameError if rejected!
) if not high_done else 0
```

When `high_action == 0` (reject), the `if` block is never entered, so `low_obs` is **never defined**. The ternary condition `if not high_done` is checking the wrong variable — after a rejection `high_done` is `True` (returned from `_finalize_episode`), so the guard happens to work *only* because of that coincidence. But this is fragile: if `step_high_level` is ever refactored (e.g., rejection no longer sets `done=True`), it becomes an immediate `UnboundLocalError`.

**Fix**: Since you're dropping the low-level policy entirely, just remove `last_value_low` and the low-level buffer references from `learn_with_instance`.

---

### BUG-4 — `_finalize_episode` calls `self.get_info()` which is not defined

**Location**: `hrl_ac_env.py`, `_finalize_episode` (impl.md line 175)

```python
return self.get_high_level_observation(), reward, True, self.get_info(solution_info)
```

`get_info` is not a method of `JointPRStepInstanceRLEnv`. The actual base class uses `info` as a plain dict built inline inside `step()`. Calling `self.get_info(solution_info)` will raise `AttributeError`.

**Fix**: Replace with `{'solution_info': solution_info}` or look up the actual info dict construction pattern in `instance_rl_environment.py`.

---

### BUG-5 — `_construct_v_net_global_features` shape mismatch in `node_data.sum(axis=1)`

**Location**: `hrl_ac_env.py`, `_construct_v_net_global_features` (impl.md lines 202–213)

```python
node_data = np.array(
    self.v_net.get_node_attrs_data(node_attrs), 
    dtype=np.float32
)
# ...
node_data.sum(axis=1).mean() / self.node_attr_benchmarks,  # ← likely wrong axis + scalar division
```

`get_node_attrs_data` returns shape `(num_attrs, num_nodes)` (attrs-first), but the code sums over `axis=1` assuming `(num_nodes, num_attrs)`. Also `self.node_attr_benchmarks` is an `AttributeBenchmarks` object (not a scalar), so dividing a `float` by it will raise a `TypeError`.

---

### BUG-6 — `_normalize_p_net_resources` iterates wrong axis

**Location**: `hrl_ac_env.py`, `_normalize_p_net_resources` (impl.md lines 243–248)

```python
for i, attr in enumerate(self.extracted_attr_types):
    ...
    node_data[i] = node_data[i] / (max_values + 1e-6)
```

`node_data` here comes from `_construct_p_net_global_features` which produces shape `(num_attrs, num_nodes)` from `get_node_attrs_data`. Indexing `node_data[i]` correctly gives the i-th attribute row — **but only if the shape really is `(num_attrs, num_nodes)`**. The `_construct_p_net_global_features` code passes this array directly to `_get_p_net_pyg_data` which expects `(num_nodes, num_features)`. The shapes are inconsistent between the two callers.

---

## 🟠 Architectural / Integration Bugs

### BUG-7 — `HRLACFeatureConstructor.construct()` is never implemented

**Location**: `hrl_ac_feature_constructor.py` (impl.md section 3.3)

`BaseFeatureConstructor` declares an abstract `construct(p_net, v_net, solution, curr_v_node_id)` that raises `NotImplementedError`. `HRLACFeatureConstructor` only defines `construct_high_level` and `construct_low_level` — **not** `construct()`.

The virne environment base (`instance_rl_environment.py`) calls `self.feature_constructor.construct(...)` inside `get_observation()`. Without overriding `construct()`, the high-level env will immediately raise `NotImplementedError`.

**Fix**: Override `construct()` to route to `construct_high_level`.

---

### BUG-8 — `merge_instance_experience` signature mismatch

**Location**: `hrl_ac_solver.py`, `learn_with_instance` (impl.md lines 888–893)

The impl returns:
```python
return solution, self.high_level_buffer, self.low_level_buffer, last_value_high, last_value_low
```

But `InstanceAgent.merge_instance_experience` (and `learn_singly`) expects:
```python
solution, instance_buffer, last_value = self.learn_with_instance(instance)
self.merge_instance_experience(instance, solution, instance_buffer, last_value)
```

The extra return values (`low_level_buffer`, `last_value_high`, `last_value_low`) break the tuple unpacking in the inherited `learn_singly` loop unless you also override that method. The impl only stubs `_update_high_level` / `_update_low_level` as `pass`, meaning **no gradient update ever fires**.

**Fix (high-level only)**: Return `(solution, self.high_level_buffer, last_value_high)` and override `merge_instance_experience` to merge into `self.buffer` (not `self.high_level_buffer`).

---

### BUG-9 — Dual buffer vs. inherited `self.buffer` confusion

**Location**: `hrl_ac_solver.py`, `__init__` and `update` (impl.md lines 720–958)

`PPOSolver` (via `RLSolver.__init__`) creates `self.buffer = RolloutBuffer()`. The impl additionally creates `self.high_level_buffer` and `self.low_level_buffer`. The inherited `learn_singly` loop checks `self.buffer.size() >= self.target_steps` to trigger `self.update()`.

The impl's `update()` reads from `self.high_level_buffer` and `self.low_level_buffer`, but the inherited `merge_instance_experience` appends to `self.buffer`. As a result:
- `self.buffer` grows forever (trigger condition for `update()` is met)  
- `self.high_level_buffer` stays empty (never triggers the actual PPO logic)  
- `self.update_time += 1` is incremented twice (once in the stub `update()`, once inside `_update_high_level`/`_update_low_level` if they were implemented)

**Fix**: Since you have only high-level policy, **drop the dual-buffer design entirely**. Use `self.buffer` directly (as all other virne solvers do) and let the standard `merge_instance_experience` → `update()` path handle it.

---

### BUG-10 — `GCNConvNet` called with `return_batch=True` — unsupported kwarg

**Location**: `hrl_ac_policy.py`, `HighLevelActorCritic.__init__` (impl.md line 291)

```python
self.p_net_encoder = GCNConvNet(
    ...
    return_batch=True,   # ← likely not a real parameter of GCNConvNet
)
```

`GCNConvNet` in virne does **not** accept `return_batch`. It returns node embeddings (shape `(N, dim)`). The forward pass then calls:
```python
p_net_embedding = self.p_net_encoder(p_net_batch)  # expects global pooled (batch, dim)
```
But without global pooling, this returns per-node embeddings — the `torch.cat` with `v_net_embedding` will fail on shape mismatch.

**Fix**: Add a `GraphPooling('mean')` layer after the encoder and call it explicitly, or use the `NetEncoder` class defined later in the same file.

---

### BUG-11 — `obs_as_tensor_for_hrl_ac_high_level` / `get_pyg_data` / `Batch` not imported

**Location**: `hrl_ac_solver.py` (impl.md lines 666–675)

```python
from torch_geometric.data import Batch   # missing import
p_net_data = get_pyg_data(...)           # function not defined/imported
```

`get_pyg_data` is not defined in the snippet and is not a virne public utility. The correct virne pattern is to use `TensorConvertor` methods (e.g. `TensorConvertor.obs_as_tensor_for_dual_gnn`).

---

### BUG-12 — `HRLACFeatureConstructor.__init__` passes `None` as `p_net` / `v_net`

**Location**: `hrl_ac_solver.py`, `__init__` (impl.md lines 724–729)

```python
self.feature_constructor = HRLACFeatureConstructor(
    self.policy_high_level.p_net if hasattr(self, 'policy_high_level') else None,
    None,        # v_net is None
    config
)
```

`BaseFeatureConstructor.__init__` immediately calls:
```python
p_net_attribute_benchmarks = AttributeBenchmarkManager.get_from_cache('p_net')
...
p_net_topological_metrics = TopologicalMetricCalculator.calculate(p_net, ...)
```
If `p_net` is `None`, this raises `AttributeError` (calling `.nodes` etc. on `None`).

Also `self.policy_high_level` does not exist on `self` at this point — it's only created inside `_make_policy` which is called from `PPOSolver.__init__` (which hasn't run yet when the feature constructor is constructed in the solver's own `__init__`).

**Fix**: Do not instantiate the feature constructor in the solver `__init__`. Let the environment create it per-instance (which is the virne pattern — the env calls `FeatureConstructorRegistry.get(name)(p_net, v_net, config)`).

---

## 🟡 Logic / Correctness Bugs

### BUG-13 — `_compute_hierarchical_reward` checks `solution['early_rejection']` before it's set

**Location**: `hrl_ac_env.py`, `_compute_hierarchical_reward` (impl.md lines 185–190)

```python
elif self.solution['early_rejection']:
    return 0.0
```

`Solution` objects in virne don't have `'early_rejection'` as a default key. Accessing `solution['early_rejection']` when it wasn't set will raise `KeyError`.

**Fix**: Use `self.solution.get('early_rejection', False)`.

---

### BUG-14 — `v_net_attrs_dim` estimation is wrong for admission-control-only case

**Location**: `hrl_ac_solver.py`, `_make_policy` (impl.md line 738)

```python
v_net_attrs_dim = 3 + len(config.rl.feature_constructor.extracted_attr_types)
```

But `construct_high_level` builds `v_net_attrs` as:
```python
v_net_attrs = np.concatenate([
    [v_num_nodes],          # 1
    [v_num_links],          # 1
    v_avg_demand,           # len(extracted_attr_types) per node attr, shape (num_attrs,)
])
```
So `v_net_attrs_dim = 2 + num_attrs`, not `3 + num_attrs`. The off-by-one causes a `size mismatch` error in the first linear layer of `HighLevelActorCritic`.

---

### BUG-15 — `_estimate_p_net_x_dim` misses the `avg_distance` column added by `_construct_p_net_features`

**Location**: `hrl_ac_solver.py` (impl.md line 816–819)

```python
def _estimate_p_net_x_dim(self, config):
    num_attrs = len(config.rl.feature_constructor.extracted_attr_types)
    if config.rl.feature_constructor.if_use_aggregated_link_attrs:
        num_attrs += num_attrs * 4  # min, mean, max, sum
    return num_attrs
```

Looking at `BaseFeatureConstructor._construct_p_net_features`:
- Node attrs: `num_attrs`
- Node status (if enabled): `+3`
- Link aggregated (if enabled): `+num_link_attrs * 4`
- **avg_distance**: always `+1`
- degree metric (if enabled): `+1`
- more topological (if enabled): `+3`

The impl's estimator is missing `+1` for `avg_distance` and the conditional `+3` for node status flags. This will cause a `size mismatch` error in the GNN's first layer.

---

## Summary Table

| ID | Severity | File | Will it crash? |
|----|----------|------|---------------|
| BUG-1 | 🔴 Critical | `hrl_ac_solver.py` | Yes — `NameError` on import |
| BUG-2 | 🔴 Critical | `hrl_ac_solver.py` | Yes — `KeyError` during init if config is incomplete |
| BUG-3 | 🔴 Critical | `hrl_ac_solver.py` | Yes — `UnboundLocalError` on any rejection (fragile) |
| BUG-4 | 🔴 Critical | `hrl_ac_env.py` | Yes — `AttributeError` in every episode end |
| BUG-5 | 🔴 Critical | `hrl_ac_env.py` | Yes — `TypeError` during feature construction |
| BUG-6 | 🔴 Critical | `hrl_ac_env.py` | Yes — shape inconsistency in normalization |
| BUG-7 | 🟠 Arch | `hrl_ac_feature_constructor.py` | Yes — `NotImplementedError` on first `get_observation()` |
| BUG-8 | 🟠 Arch | `hrl_ac_solver.py` | Yes — tuple unpack error in `learn_singly` |
| BUG-9 | 🟠 Arch | `hrl_ac_solver.py` | Silent — update never fires |
| BUG-10 | 🟠 Arch | `hrl_ac_policy.py` | Yes — shape mismatch in forward pass |
| BUG-11 | 🟠 Arch | `hrl_ac_solver.py` | Yes — `NameError` for `get_pyg_data` / `Batch` |
| BUG-12 | 🟠 Arch | `hrl_ac_solver.py` | Yes — `AttributeError` on `None` p_net |
| BUG-13 | 🟡 Logic | `hrl_ac_env.py` | Yes — `KeyError` on reward computation |
| BUG-14 | 🟡 Logic | `hrl_ac_solver.py` | Yes — size mismatch in first Linear layer |
| BUG-15 | 🟡 Logic | `hrl_ac_solver.py` | Yes — size mismatch in GNN encoder |

---

## Recommended Simplification for High-Level-Only Mode

Since you're not using a learned low-level policy:

1. **Delete** `LowLevelActorCritic`, `step_low_level`, `get_low_level_observation`, `construct_low_level`, `low_level_buffer`, `select_action_low`, `estimate_value_low`, `_update_low_level`.
2. **Route `construct()` → `construct_high_level()`** in `HRLACFeatureConstructor`.
3. **Use `self.buffer` directly** (inherited from `RLSolver`) instead of `self.high_level_buffer`.
4. **Return** `(solution, instance_buffer, last_value)` from `learn_with_instance` to match `learn_singly`'s expected signature.
5. **Fix imports**: `open_dict`, `Batch`, and use `TensorConvertor` instead of `get_pyg_data`.
6. **Delegate node mapping** to the heuristic solver inside `step_high_level(action=1)` (accept branch) using e.g. `self.sub_solver.solve(instance)`.
