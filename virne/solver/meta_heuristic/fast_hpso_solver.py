# ==============================================================================
# Fast Hybrid PSO (HPSO) Solver for VNE — virne integration
#
# Core mechanics preserved from fast_hpso.py:
#   - Particle representation (list of p_node indices, one per v_node)
#   - operation_minus / operation_plus / operation_multiply (discrete PSO ops)
#   - Simulated-annealing neighbour step
#   - Fast proxy fitness (hop-distance × bandwidth) for inner-loop evaluation
#   - Full Dijkstra validation only on the final best particle
#
# Virne integration points:
#   - Registered via @SolverRegistry.register → plugs into BaseSystem.from_config
#   - Inherits BaseMetaHeuristicSolver → reuses solve(), ready(), construct_candidates
#   - Final deployment uses controller.deploy_with_node_slots (inplace=False)
#     so the base-class solve() can call controller.deploy() on the real p_net
#   - counter.count_solution fills all revenue/cost fields expected by Recorder
# ==============================================================================

import copy
import math
import random

import networkx as nx

from virne.core import Controller, Counter, Logger, Recorder, Solution
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.solver.base_solver import SolverRegistry
from virne.solver.meta_heuristic.base_meta_heuristic_solver import (
    BaseMetaHeuristicSolver,
    INFEASIBLE_FITNESS,
)

# ── internal sentinel ────────────────────────────────────────────────────────
_INFEASIBLE = 1e9


# ============================================================
# 1.  Fast proxy fitness  (no resource reservation)
# ============================================================

def _fast_fitness(particle: list, p_net: PhysicalNetwork, v_net: VirtualNetwork) -> float:
    """
    Estimate solution cost WITHOUT reserving any resources.

    Node constraint  → check current available CPU on p_net.
    Link cost        → hop-distance × requested bandwidth (proxy for Dijkstra).

    Returns _INFEASIBLE when any hard constraint is violated.
    """
    vnodes = list(v_net.nodes())

    # injective mapping required
    if len(set(particle)) < len(vnodes):
        return _INFEASIBLE

    # resolve resource attribute names once
    node_res_attrs = [a.name for a in v_net.get_node_attrs(types=['resource'])]
    link_res_attrs = [a.name for a in v_net.get_link_attrs(types=['resource'])]

    mapping: dict = {}
    for i, v in enumerate(vnodes):
        s = particle[i]
        for attr in node_res_attrs:
            if p_net.nodes[s].get(attr, 0) < v_net.nodes[v].get(attr, 0):
                return _INFEASIBLE
        mapping[v] = s

    est_link_cost = 0.0
    for (u, v) in v_net.edges():
        s_u, s_v = mapping[u], mapping[v]
        if s_u == s_v:
            continue
        try:
            hops = nx.shortest_path_length(p_net, s_u, s_v)
        except nx.NetworkXNoPath:
            return _INFEASIBLE
        # use the first link resource attribute as the demand signal (usually bw)
        demand = v_net.edges[u, v].get(link_res_attrs[0], 0) if link_res_attrs else 1
        est_link_cost += hops * demand

    return est_link_cost


# ============================================================
# 2.  Particle initialisation
# ============================================================

def _init_swarm(p_net: PhysicalNetwork, v_net: VirtualNetwork, n_particles: int) -> list:
    """
    Build an initial swarm.

    Virtual nodes are sorted by CPU demand (largest-first) and greedily
    assigned to physical nodes (with some top-K randomness).
    Fallback to pure random particles if the greedy pool runs dry.
    """
    node_res_attrs = [a.name for a in v_net.get_node_attrs(types=['resource'])]

    # sort vnodes by total resource demand descending
    vnodes_sorted = sorted(
        v_net.nodes(),
        key=lambda v: sum(v_net.nodes[v].get(a, 0) for a in node_res_attrs),
        reverse=True,
    )
    sub_sorted = sorted(
        p_net.nodes(),
        key=lambda s: sum(p_net.nodes[s].get(a, 0) for a in node_res_attrs),
        reverse=True,
    )
    vnode_idx = {v: i for i, v in enumerate(v_net.nodes())}

    swarm: list = []
    for _ in range(n_particles):
        particle = [None] * v_net.num_nodes
        used: set = set()
        ok = True
        for v in vnodes_sorted:
            cands = [
                s for s in sub_sorted
                if all(p_net.nodes[s].get(a, 0) >= v_net.nodes[v].get(a, 0)
                       for a in node_res_attrs)
                and s not in used
            ]
            if not cands:
                ok = False
                break
            k = random.randint(1, min(3, len(cands)))
            s = random.choice(cands[:k])
            particle[vnode_idx[v]] = s
            used.add(s)
        if ok:
            swarm.append(particle)

    # fallback random fill so we always have n_particles particles
    sub_list = list(p_net.nodes())
    while len(swarm) < n_particles:
        if len(sub_list) < v_net.num_nodes:
            break
        random.shuffle(sub_list)
        swarm.append(sub_list[: v_net.num_nodes])

    return swarm


# ============================================================
# 3.  Discrete PSO operators  (kept verbatim from fast_hpso.py)
# ============================================================

def _op_minus(Xi: list, Xj: list) -> list:
    """Velocity component: 1 where positions agree, 0 otherwise."""
    return [1 if Xi[k] == Xj[k] else 0 for k in range(len(Xi))]


def _op_plus(p: float, Vi: list, q: float, Vj: list) -> list:
    """Weighted probabilistic merge of two velocity vectors."""
    if p + q == 0:
        return Vi.copy()
    p_norm = p / (p + q)
    return [
        Vi[i] if (Vi[i] == Vj[i] or random.random() < p_norm) else Vj[i]
        for i in range(len(Vi))
    ]


def _op_multiply(Xi: list, V: list, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> list:
    """
    Apply velocity to position:
    dimensions where V[i]==0 are re-sampled from feasible physical nodes.
    """
    Xnew = Xi.copy()
    vnodes = list(v_net.nodes())
    node_res_attrs = [a.name for a in v_net.get_node_attrs(types=['resource'])]
    used = set(Xnew)

    for i in range(len(V)):
        if V[i] == 0:
            v = vnodes[i]
            used.discard(Xnew[i])
            cands = [
                s for s in p_net.nodes()
                if all(p_net.nodes[s].get(a, 0) >= v_net.nodes[v].get(a, 0)
                       for a in node_res_attrs)
                and s not in used
            ]
            if cands:
                Xnew[i] = random.choice(cands)
            used.add(Xnew[i])

    return Xnew


# ============================================================
# 4.  SA neighbour generator
# ============================================================

def _sa_neighbor(particle: list, p_net: PhysicalNetwork, v_net: VirtualNetwork) -> list:
    """Swap one virtual-node mapping to a new feasible physical node."""
    neighbor = particle.copy()
    used = set(neighbor)
    vnodes = list(v_net.nodes())
    node_res_attrs = [a.name for a in v_net.get_node_attrs(types=['resource'])]

    i = random.randrange(len(particle))
    v = vnodes[i]
    used.discard(particle[i])
    cands = [
        s for s in p_net.nodes()
        if all(p_net.nodes[s].get(a, 0) >= v_net.nodes[v].get(a, 0)
               for a in node_res_attrs)
        and s not in used
    ]
    if cands:
        neighbor[i] = random.choice(cands)
    return neighbor


# ============================================================
# 5.  Solver class
# ============================================================

@SolverRegistry.register(solver_name='fast_hpso', solver_type='meta_heuristic')
class FastHPSOSolver(BaseMetaHeuristicSolver):
    """
    Fast Hybrid PSO Solver for Virtual Network Embedding.

    Inner iterations use a cheap hop-distance proxy fitness to guide the
    swarm without running Dijkstra on every candidate.  The best particle
    found is deployed once using the framework's full shortest-path solver,
    which validates bandwidth constraints and reserves resources.

    Hyper-parameters
    ----------------
    num_particles   : swarm size                    (default 20)
    max_iteration   : PSO/SA iterations             (default 30)
    w_max / w_min   : inertia weight bounds         (default 0.9 / 0.5)
    beta            : cognitive acceleration        (default 0.3)
    gamma           : social acceleration           (default 0.3)
    T0              : initial SA temperature        (default 100)
    cooling_rate    : SA geometric cooling factor   (default 0.95)
    shortest_method : link-mapping method for final deploy (default 'k_shortest')
    k_shortest      : k for k-shortest paths        (default 10)
    """

    def __init__(
        self,
        controller: Controller,
        recorder: Recorder,
        counter: Counter,
        logger: Logger,
        config,
        **kwargs,
    ):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)

        # ── hyper-parameters (can be overridden via kwargs) ──────────────────
        self.num_particles: int = kwargs.get("num_particles", 20)
        self.max_iteration: int = kwargs.get("max_iteration", 30)
        self.w_max: float = kwargs.get("w_max", 0.9)
        self.w_min: float = kwargs.get("w_min", 0.5)
        self.beta: float = kwargs.get("beta", 0.3)
        self.gamma: float = kwargs.get("gamma", 0.3)
        self.T0: float = kwargs.get("T0", 100.0)
        self.cooling_rate: float = kwargs.get("cooling_rate", 0.95)
        # override base-class defaults to match HPSO preferred link mapper
        self.shortest_method: str = kwargs.get("shortest_method", "k_shortest")
        self.k_shortest: int = kwargs.get("k_shortest", 10)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _particle_to_node_slots(self, particle: list, v_net: VirtualNetwork) -> dict:
        """Convert list-based particle to the {v_node_id: p_node_id} dict expected by virne."""
        vnodes = list(v_net.nodes())
        return {vnodes[i]: particle[i] for i in range(len(vnodes))}

    def _try_deploy_particle(
        self,
        particle: list,
        v_net: VirtualNetwork,
        p_net: PhysicalNetwork,
    ) -> Solution:
        """
        Attempt to deploy the given particle via the framework's controller.

        Uses inplace=False so the actual p_net is NOT modified here;
        the base-class solve() calls controller.deploy() for the real update.
        """
        node_slots = self._particle_to_node_slots(particle, v_net)
        solution = Solution.from_v_net(v_net)
        self.controller.deploy_with_node_slots(
            v_net,
            p_net,
            node_slots,
            solution,
            inplace=False,                     # dry-run on a copy of p_net
            shortest_method=self.shortest_method,
            k_shortest=self.k_shortest,
        )
        self.counter.count_solution(v_net, solution)
        return solution

    # ------------------------------------------------------------------
    # Main algorithm  (called by BaseMetaHeuristicSolver.solve)
    # ------------------------------------------------------------------

    def meta_run(self, v_net: VirtualNetwork, p_net: PhysicalNetwork) -> Solution:
        """
        Run the HPSO+SA main loop and return the best feasible Solution found.

        The loop:
          1. Init swarm with smart CPU-aware placement.
          2. Each iteration: PSO position update → fast-fitness eval → SA perturbation.
          3. After all iterations: validate the global-best particle with full
             Dijkstra (via controller.deploy_with_node_slots).
        """
        num_v = v_net.num_nodes

        # ── 1. Initialise ────────────────────────────────────────────────────
        swarm = _init_swarm(p_net, v_net, self.num_particles)
        if not swarm:
            return Solution.from_v_net(v_net)

        velocities = [
            [random.randint(0, 1) for _ in range(num_v)]
            for _ in range(len(swarm))
        ]

        pbest = copy.deepcopy(swarm)
        pbest_cost = [_fast_fitness(p, p_net, v_net) for p in swarm]

        gbest: list | None = None
        gbest_cost: float = _INFEASIBLE

        for i, cost in enumerate(pbest_cost):
            if cost < gbest_cost:
                gbest_cost = cost
                gbest = swarm[i].copy()

        T = self.T0

        # ── 2. Main loop ─────────────────────────────────────────────────────
        for it in range(self.max_iteration):
            # linearly decaying inertia weight
            alpha = self.w_max - (self.w_max - self.w_min) * it / self.max_iteration
            total = alpha + self.beta + self.gamma
            a, b, c = (
                (alpha / total, self.beta / total, self.gamma / total)
                if total != 0
                else (0.33, 0.33, 0.34)
            )

            for i in range(len(swarm)):
                # ── PSO velocity update ───────────────────────────────────
                dp = _op_minus(pbest[i], swarm[i])
                dg = _op_minus(gbest, swarm[i]) if gbest else [0] * num_v

                v_inertia = _op_plus(a, velocities[i], b, dp)
                velocities[i] = _op_plus(1 - c, v_inertia, c, dg)

                # ── position update ───────────────────────────────────────
                new_pos = _op_multiply(swarm[i], velocities[i], v_net, p_net)
                new_cost = _fast_fitness(new_pos, p_net, v_net)

                # ── pBest / gBest update ──────────────────────────────────
                if new_cost < pbest_cost[i]:
                    pbest[i] = new_pos.copy()
                    pbest_cost[i] = new_cost
                    if new_cost < gbest_cost:
                        gbest = new_pos.copy()
                        gbest_cost = new_cost

                swarm[i] = new_pos

                # ── SA perturbation ───────────────────────────────────────
                if T > 0.1:
                    cand = _sa_neighbor(swarm[i], p_net, v_net)
                    cand_cost = _fast_fitness(cand, p_net, v_net)
                    delta = cand_cost - new_cost

                    if delta < 0:
                        accept = True
                    else:
                        try:
                            prob = math.exp(-delta / T)
                        except OverflowError:
                            prob = 0.0
                        accept = random.random() < prob

                    if accept:
                        swarm[i] = cand
                        if cand_cost < pbest_cost[i]:
                            pbest[i] = cand.copy()
                            pbest_cost[i] = cand_cost
                            if cand_cost < gbest_cost:
                                gbest = cand.copy()
                                gbest_cost = cand_cost

            T *= self.cooling_rate

        # ── 3. Final validation via full Dijkstra ─────────────────────────────
        if gbest is None or gbest_cost >= _INFEASIBLE:
            return Solution.from_v_net(v_net)

        solution = self._try_deploy_particle(gbest, v_net, p_net)

        # If the full link-mapping fails, try each pbest in ascending cost order
        if not solution["result"]:
            ranked = sorted(
                range(len(pbest)),
                key=lambda i: pbest_cost[i],
            )
            for idx in ranked:
                if pbest_cost[idx] >= _INFEASIBLE:
                    break
                solution = self._try_deploy_particle(pbest[idx], v_net, p_net)
                if solution["result"]:
                    break

        return solution