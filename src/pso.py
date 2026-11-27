from typing import Tuple, List, Optional
import numpy as np

from src.config import SimulationConfig
from src.network import IoTNetwork
from src.routing import RoutingEngine, GraphSnapshot

class Particle:
    """Represents a particle in PSO searching the 4-dimensional weight space."""

    def __init__(self, dim: int = 4, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()
        self.position = self.rng.uniform(0.1, 1.0, size=dim)
        self.velocity = self.rng.uniform(-0.1, 0.1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = float("inf")
        self.current_fitness = float("inf")

    @property
    def normalized_weights(self) -> Tuple[float, float, float, float]:
        pos = np.maximum(self.position, 1e-4)
        total = np.sum(pos)
        weights = pos / total
        return tuple(float(w) for w in weights)

class PSORoutingOptimizer:
    """Particle Swarm Optimizer (with Adaptive PSO support) for multi-objective routing weights."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        routing_engine: Optional[RoutingEngine] = None,
        use_ml: bool = True,
        seed: Optional[int] = None
    ):
        self.config = config or SimulationConfig()
        self.routing_engine = routing_engine or RoutingEngine(self.config)
        self.use_ml = use_ml
        self.seed = seed if seed is not None else self.config.random_seed
        self.rng = np.random.RandomState(self.seed + 10000)

        self.num_particles = self.config.pso_num_particles
        self.max_iter = self.config.pso_max_iter
        self.w_max = self.config.pso_w_max
        self.w_min = self.config.pso_w_min
        self.c1 = self.config.pso_c1
        self.c2 = self.config.pso_c2

        self.lambda_e = self.config.lambda_energy
        self.lambda_var = self.config.lambda_variance
        self.lambda_d = self.config.lambda_delay
        self.lambda_l = self.config.lambda_loss

        self.convergence_curve: List[float] = []

    def _adapt_fitness_weights(self, network: IoTNetwork):
        if not self.config.pso_adaptive:
            return

        alive_nodes = network.get_alive_sensor_nodes()
        if not alive_nodes:
            return

        avg_residual_ratio = float(np.mean([n.residual_ratio for n in alive_nodes]))

        if avg_residual_ratio > 0.7:
            self.lambda_e = 0.30
            self.lambda_var = 0.25
            self.lambda_d = 0.25
            self.lambda_l = 0.20
        elif avg_residual_ratio > 0.3:
            self.lambda_e = 0.40
            self.lambda_var = 0.35
            self.lambda_d = 0.10
            self.lambda_l = 0.15
        else:
            self.lambda_e = 0.45
            self.lambda_var = 0.40
            self.lambda_d = 0.05
            self.lambda_l = 0.10

    def evaluate_fitness_with_snapshot(
        self,
        weights: Tuple[float, float, float, float],
        snapshot: GraphSnapshot,
        network: IoTNetwork
    ) -> float:
        """Evaluate network-wide multi-objective fitness function using exact sequential
        fragmentation expectations:
        - Expected energy per generated packet: accounts for early stopping when any fragment
          fails (R_j = prod_{m<j} q_m) and aggregation energy only upon full hop success (P_app_succ).
        - Expected conditional delivery delay: weighted mean of sum_h E[D_h | hop_h succ]
          over successfully delivered packets (matching simulator.weighted_mean_delay).
        - Fast reverse Dijkstra via ``snapshot.find_all_paths_fast(weights)``.
        """
        paths = snapshot.find_all_paths_fast(weights)
        alive_nodes = network.get_alive_sensor_nodes()

        if not alive_nodes:
            return float("inf")

        total_network_energy = 0.0
        node_energy_use = {n.node_id: 0.0 for n in alive_nodes}
        weighted_deliv_delay_sum = 0.0
        deliv_prob_sum = 0.0
        path_pdrs = []
        disconnected_count = 0

        em = self.routing_engine.energy_model
        max_attempts = 1 + self.config.max_retransmissions
        frag_sizes = self.config.fragment_sizes_bits
        da_e = em.aggregation_energy()


        tx_exp_arr, rx_exp_arr, link_succ_arr, cond_delay_arr = snapshot.get_edge_expectations(
            em, max_attempts, frag_sizes
        )
        edge_to_idx = snapshot._edge_to_idx
        bs_id = network.bs_id

        for node in alive_nodes:
            u_id = node.node_id
            path = paths.get(u_id, [])

            if not path or len(path) < 2 or path[-1] != bs_id:
                disconnected_count += 1
                path_pdrs.append(0.0)
                continue

            p_reach = 1.0
            cond_path_delay = 0.0

            for i in range(len(path) - 1):
                from_id = path[i]
                to_id = path[i + 1]
                e_idx = edge_to_idx[(from_id, to_id)]

                tx_exp = float(tx_exp_arr[e_idx])
                rx_exp = float(rx_exp_arr[e_idx])
                link_succ_prob = float(link_succ_arr[e_idx])
                cond_hop_delay = float(cond_delay_arr[e_idx])

                hop_tx_energy = p_reach * tx_exp
                total_network_energy += hop_tx_energy
                node_energy_use[from_id] = node_energy_use.get(from_id, 0.0) + hop_tx_energy

                if to_id != bs_id and to_id in node_energy_use:
                    hop_rx_energy = p_reach * rx_exp
                    hop_da_energy = p_reach * link_succ_prob * da_e
                    node_energy_use[to_id] += (hop_rx_energy + hop_da_energy)
                    total_network_energy += (hop_rx_energy + hop_da_energy)

                cond_path_delay += cond_hop_delay
                p_reach *= link_succ_prob

            weighted_deliv_delay_sum += p_reach * cond_path_delay
            deliv_prob_sum += p_reach
            path_pdrs.append(p_reach)

        connected_count = len(alive_nodes) - disconnected_count
        if connected_count == 0:
            return 1e6 + disconnected_count * 100.0

        avg_delay = float(weighted_deliv_delay_sum / deliv_prob_sum) if deliv_prob_sum > 1e-12 else 150.0

        avg_pdr = float(sum(path_pdrs) / len(alive_nodes))
        energy_variance = float(np.var(list(node_energy_use.values())))

        norm_energy = total_network_energy / (connected_count * self.routing_engine.max_total_link_energy + 1e-9)
        norm_var = energy_variance / (self.routing_engine.max_total_link_energy ** 2 + 1e-9)
        norm_delay = min(1.0, avg_delay / 150.0)
        norm_loss = max(0.0, 1.0 - avg_pdr)


        fitness = (
            self.lambda_e * norm_energy +
            self.lambda_var * norm_var +
            self.lambda_d * norm_delay +
            self.lambda_l * norm_loss
        )

        return float(fitness)

    def optimize(
        self,
        network: IoTNetwork,
        snapshot: Optional[GraphSnapshot] = None,
        channel_rng: Optional[np.random.RandomState] = None
    ) -> Tuple[Tuple[float, float, float, float], float]:
        """Execute PSO search to find optimal routing cost weights w* = (w_E, w_D, w_L, w_R)."""
        self._adapt_fitness_weights(network)

        if snapshot is None:
            routing_mode = "hybrid" if self.use_ml else "pso"
            snapshot = self.routing_engine.create_graph_snapshot(
                network=network,
                use_ml=self.use_ml,
                routing_mode=routing_mode,
                rng=channel_rng
            )

        pso_rng = self.rng

        particles = [Particle(dim=4, rng=pso_rng) for _ in range(self.num_particles)]
        gbest_position = None
        gbest_fitness = float("inf")
        self.convergence_curve = []

        for p in particles:
            fit = self.evaluate_fitness_with_snapshot(p.normalized_weights, snapshot, network)
            p.current_fitness = fit
            p.best_fitness = fit
            p.best_position = np.copy(p.position)

            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = np.copy(p.position)

        if gbest_position is None:
            gbest_position = np.array([0.35, 0.20, 0.25, 0.20], dtype=np.float64)

        for t in range(self.max_iter):
            w = self.w_max - (t / max(1, self.max_iter)) * (self.w_max - self.w_min)

            for p in particles:
                r1 = pso_rng.uniform(0.0, 1.0, size=4)
                r2 = pso_rng.uniform(0.0, 1.0, size=4)

                cognitive = self.c1 * r1 * (p.best_position - p.position)
                social = self.c2 * r2 * (gbest_position - p.position)
                p.velocity = w * p.velocity + cognitive + social
                p.velocity = np.clip(p.velocity, -0.2, 0.2)

                p.position = np.clip(p.position + p.velocity, 0.05, 1.0)

                fit = self.evaluate_fitness_with_snapshot(p.normalized_weights, snapshot, network)
                p.current_fitness = fit

                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = np.copy(p.position)

                    if fit < gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = np.copy(p.position)

            self.convergence_curve.append(float(gbest_fitness))

        pos = np.maximum(gbest_position, 1e-4)
        best_weights = tuple(float(w) for w in pos / np.sum(pos))
        return best_weights, float(gbest_fitness)
