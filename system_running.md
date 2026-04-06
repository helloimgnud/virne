# Running Solvers on Different Systems in Virne

The Virne framework supports multiple system environments that govern how Virtual Network Requests (VNRs) are surfaced to your solver. By default, Virne uses the **Online System** but contains alternative systems that simulate different dynamic scenarios (**Offline**, **Changeable**, and **Time Window**).

This guide explains how each system works, their API expectations, and the exact commands to run your solvers in these modes. We cover the standard sequential systems first, and conclude with the conceptually distinct Time Window System.

---

## 1. Online System (Default)

**Description**: The `OnlineSystem` processes VNRs strictly sequentially. Each arriving VNR is handed to the solver one at a time exactly as the arrival events occur. This matches the standard online Virtual Network Embedding parameterization.

**Solver API Requirements**: 
The solver must implement the standard `solve(instance) -> Solution` method.
```python
def solve(self, instance: dict) -> Solution:
    v_net = instance['v_net']
    p_net = instance['p_net']
    
    # Online sequential logic...
    solution = Solution(v_net)
    
    return solution
```

**Command to Run**:
Because the `OnlineSystem` is the default behavior, you can run a solver without specifying the system:
```bash
python main.py solver.solver_name=your_solver_name
```
Or you can enforce it explicitly:
```bash
python main.py solver.solver_name=your_solver_name system.if_offline_system=False system.if_changeable_v_nets=False system.if_time_window=False
```

---

## 2. Offline System

**Description**: The `OfflineSystem` sets up an environment where the physical network is treated as given and fixed natively, but explicitly rescales (diminishes and randomly scrambles) the physical network attributes prior to processing each VNR. It simulates degraded or strictly restrained topology settings per VNR iteration.

**Solver API Requirements**: 
It uses the exact same sequential API as the Online System.
```python
def solve(self, instance: dict) -> Solution:
    # Uses standard sequential API
    pass
```

**Command to Run**:
Enable the offline mode via Hydra overrides:
```bash
python main.py solver.solver_name=your_solver_name system.if_offline_system=True
```

---

## 3. Changeable System

**Description**: The `ChangeableSystem` represents a highly dynamic environment where the distribution, specifications, or profile of incoming VNRs changes fundamentally over time for each epoch. It dynamically swaps out the VNR dataset generator to evaluate how robustly a solver handles distribution shifts.

**Solver API Requirements**: 
It uses the exact same sequential API as the Online System.
```python
def solve(self, instance: dict) -> Solution:
    # Uses standard sequential API
    pass
```

**Command to Run**:
Enable the changeable mode via Hydra overrides:
```bash
python main.py solver.solver_name=your_solver_name system.if_changeable_v_nets=True
```

---

## 4. Time Window System (Batch Processing)

**Description**: 
Unlike the sequential systems detailed above, the `TimeWindowSystem` partitions the simulation timeline into fixed-size windows. Outstanding VNRs that arrive within a designated window are grouped into a **batch** and dispatched to the solver simultaneously. This enables solvers to enact global, cross-VNR optimizations—perfect for joint Integer Linear Programs (ILP), look-ahead heuristic processing, or batch RL inference arrays.

**Solver API Requirements**:
To fully capitalize on this structure, your solver **should** implement an extra `solve_batch(instances)` API.

```python
def solve_batch(self, instances: list) -> list:
    """
    Args:
        instances: List of instance dictionaries, each containing 'v_net', 'p_net', and 'event'.
    Returns:
        A list of `Solution` objects, strictly matching the incoming order of requested VNRs.
    """
    solutions = []
    
    # 1. Pull out all v_nets from the current batch
    v_nets = [inst['v_net'] for inst in instances]
    
    # 2. Extract the current snapshot of the physical network (shared across the batch)
    shared_p_net = instances[0]['p_net'] 
    
    # 3. Your bespoke joint-optimization logic goes here...
    
    # 4. Construct mapped solutions 
    for instance in instances:
        solution = Solution(instance['v_net'])
        # Apply the mapping extracted from your joint processing 
        # solution.node_slots = ...
        # solution.link_paths = ...
        solutions.append(solution)
        
    return solutions
```

> **Fallback Mode**: If your custom solver lacks a `solve_batch()` method, the `TimeWindowSystem` will gracefully revert. It manually loops over the internal batch and iteratively pipes occurrences through the standard `solve(instance)` per VNR (additionally printing a warning log). While preventing crashes, this strips away all batch optimization advantages.

**The Feasibility Guarantee**:
Even if your global algorithm computes a flawlessly synchronized joint outcome, Virne mandates tight physical capacity compliance. 
The system iterates through your returned `solutions` one by one when deploying. If deploying the first solution maxes out certain physical links, an initially accurate second solution may suddenly trigger overdrafts in physical bandwidth when applied sequentially. `TimeWindowSystem` conducts automatic pre-deployment sanity checks; it force-rejects compromised solutions dynamically to protect baseline consistency.

**Command to Run**:
Activate Time Window processing and parameterize the duration length:
```bash
python main.py solver.solver_name=fast_hpso system.if_time_window=True system.time_window_size=100
```
*(You may adjust `time_window_size` depending on how long you wish batches to naturally aggregate. By default, it falls back to 100)*

---

### Ensuring Reproducibility (Controlled Network Generation)
When running different solvers (or the same solver) and comparing their outcomes in the `TimeWindowSystem` (or any system), it's critical that the physical network (`p_net`) and the virtual network requests (`v_sim`) remain entirely consistent. To ensure strict consistency for experiments across varying algorithms at different times, you can enforce dataset generation seeds or instruct the framework to consistently generate datasets from scratch using the same seed.

**Example execution locking the seed for `fast_hpso`**:
```bash
python main.py solver.solver_name=fast_hpso system.if_time_window=True system.time_window_size=100 experiment.seed=42 experiment.if_load_p_net=False experiment.if_load_v_nets=False
```
* **`experiment.seed=42`**: Ensures the randomly generated numbers, typologies, and demands are identical.
* **`experiment.if_load_p_net=False`** & **`experiment.if_load_v_nets=False`**: Foregoes loading from previous disk caches and consistently recreates the pristine environment natively using the locked seed.
