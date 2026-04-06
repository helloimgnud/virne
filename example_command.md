# Virne Network Generators Guide

This guide explains how to use the built-in network settings generators in the Virne algorithms and outlines the functionality of the existing variable settings that determine how physical and virtual networks are generated.

## Generating the Dataset

Virne uses Hydra for configuration management and allows dataset generation seamlessly during training or evaluation. Datasets specify both the **Physical Network (P-Net)** and a sequence of dynamically arriving **Virtual Network Requests (V-Nets)**.

### Method 1: Using the Hydra CLI (Recommended)

When you run the framework via `main.py`, the network is generated automatically based on standard configurations. However, if you explicitly want to save the newly generated networks to your storage (so they can be loaded deterministically in future runs), you can overwrite the `experiment` directives through the Hydra CLI:

```bash
# Run simulation and save both the physical network and the virtual networks
python main.py \
    experiment.if_save_p_net=True \
    experiment.if_save_v_nets=True \
    experiment.seed=42
```

By default, the datasets will be saved under the designated output directories defined in the YAML settings (`dataset/p_net` and `dataset/v_nets`).

In future experiments, you could load the saved generic datasets rather than generating new ones:
```bash
python main.py \
    experiment.if_load_p_net=True \
    experiment.if_load_v_nets=True
```

### Method 2: Standalone Python Script

If you wish to use the `Generator` class to generate the datasets programmatically (without running the entire VNE solver/environment), you can use the following python code:

```python
from omegaconf import OmegaConf
from virne.network.dataset_generator import Generator

# 1. Load the target configurations (this mimics hydra's composed config)
config = OmegaConf.load("settings/main.yaml")
p_config = OmegaConf.load("settings/p_net_setting/default.yaml")
v_config = OmegaConf.load("settings/v_sim_setting/default.yaml")

# Combine configurations
config.p_net_setting = p_config
config.v_sim_setting = v_config

# 2. Invoke the Generator module to generate and save datasets
p_net, v_net_simulator = Generator.generate_dataset(
    config, 
    p_net=True,   # Generate Physical Network 
    v_nets=True,  # Generate Virtual Network Simulator instance
    save=True     # Execute saving procedures defined in settings 
)

print(f"Generated Physical Network with {p_net.num_nodes} nodes.")
print(f"Generated {v_net_simulator.num_v_nets} Virtual Network Requests.")
```

---

## Configuration Variables Explained

The generation mechanism relies heavily on parameters defined in the variable settings files. By default, these are found in `settings/p_net_setting/default.yaml` and `settings/v_sim_setting/default.yaml`.

### 1. Physical Network Settings (`p_net_setting/default.yaml`)

This file configures the large, stationary physical graph.

* **`topology`**: Defines structural properties.
  * `num_nodes` (int): Total number of nodes in the simulated physical network.
  * `type` (str): Graph generation algorithm. Examples: `waxman` or `random`.
  * `wm_alpha` / `wm_beta` (float): Specific parameters for the Waxman topological model determining the node spacing and edge density.
  * `file_path` (str, optional): You can supply a pre-built static topology (e.g. `datasets/topology/Geant.gml`).
* **`node_attrs_setting`**: Controls hardware attributes assigned to nodes. 
  * `name` (str): E.g., `cpu`.
  * `type` (str): e.g. `resource` (consumable) or `extrema` (trackable boundary limit). 
  * `distribution` (str): Distribution used to sample capacity (e.g., `uniform`).
  * `low` / `high` (int): Range of node processing capabilities sampled.
* **`link_attrs_setting`**: Controls connectivity attributes.
  * `name` (str): E.g., `bw` (bandwidth).
  * `distribution` (str): e.g., `uniform` for randomized allocation between `low` and `high` values.
* **`output`**: Directory mappings.
  * `save_dir`: Where the generated topology should be saved.
  * `file_name`: Typically `p_net.gml`. `gml` is the fundamental Graph Modeling Language format in Virne.

### 2. Virtual Network Simulation Settings (`v_sim_setting/default.yaml`)

This dictates the structure and temporal dynamics of the arriving VNRs.

* **`num_v_nets`** (int): The total number of incoming Virtual Network Requests to simulate over an epoch.
* **`topology`**: Defines the virtual networks spatial relationships. 
  * `type` (str): Type of graph (usually `random`).
  * `random_prob` (float): The probability of an edge existing between any two generated virtual nodes.
* **`v_net_size`**: The node dimension of each dynamically sized virtual request.
  * `distribution` (str): `uniform` distribution bounds the randomized sizes between `low` (e.g. 2 nodes min) and `high` (e.g. 10 nodes max).
* **`arrival_rate`**: Determines how rapidly requests arrive into the global queue.
  * `distribution` (str): Recommended as `poisson` to mimic realistic traffic flow arrivals.
  * `lam` (float): The standard `lambda` intensity metric for independent event arrivals.
  * `reciprocal` (bool): Usually set to true to interpret lambda effectively against timestamping intervals.
* **`lifetime`**: The longevity/duration of a VNR before it departs and releases occupied limits.
  * `distribution` (str): Expected as `exponential` for real-world survival analysis.
  * `scale` (int): Mean survival scale (e.g., 500 units of time).
* **`node_attrs_setting`** & **`link_attrs_setting`**: Work similarly to the physical networks, except they represent constrained **demands** rather than capacities. Therefore, their sampled values (like `low` ~ `high`: 0 to 20 for CPUs or 0 to 50 for Bandwidth) are drastically lower than the capabilities available on the Physical Network.
* **`output`**: Similar output paths where events (`events.yaml`), topologies (inside `v_nets/` folder), and individual setups are systematically stored.
