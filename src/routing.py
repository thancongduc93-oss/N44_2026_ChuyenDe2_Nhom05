import heapq
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np

from src.config import SimulationConfig
from src.network import IoTNetwork
from src.energy import EnergyModel
from src.channel import WirelessChannel, make_channel_rng
from src.ml_predictor import LinkQualityPredictor

class GraphSnapshot:
    """Immutable snapshot of network link features and states for one evaluation interval.

    Stores both routing-mode-specific ``pdrs``/``delays`` and ground-truth channel
    quantities:
    - ``physical_bers``: bit error rate on each link from deterministic keyed RSSI
    - ``physical_pdrs``: single-attempt frame PDR for a full-payload frame
    - ``physical_delays``: expected unconditional airtime per generated app packet (ms)
    - ``physical_cond_delays``: expected delay conditioned on successful app delivery (ms)

    Precomputes reverse adjacency lists and an edge-index map so that PSO can run
    exact single-source reverse Dijkstra via ``heapq`` without allocating hundreds
    of NetworkX DiGraph objects.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int]],
        distances: np.ndarray,
        tx_energies: np.ndarray,
        total_energies: np.ndarray,
        norm_energies: np.ndarray,
        norm_delays: np.ndarray,
        norm_losses: np.ndarray,
        rx_depletions: np.ndarray,
        pdrs: np.ndarray,
        delays: np.ndarray,
        physical_pdrs: np.ndarray,
        physical_delays: np.ndarray,
        alive_nodes: List[int],
        bs_id: int,
        physical_bers: Optional[np.ndarray] = None,
        physical_cond_delays: Optional[np.ndarray] = None,
        physical_rssis: Optional[np.ndarray] = None,
        ml_fallback_count: int = 0,
        ml_total_predictions: int = 0
    ):
        self.edges = edges
        self.distances = distances
        self.tx_energies = tx_energies
        self.total_energies = total_energies
        self.norm_energies = norm_energies
        self.norm_delays = norm_delays
        self.norm_losses = norm_losses
        self.rx_depletions = rx_depletions
        self.pdrs = pdrs
        self.delays = delays
        self.physical_pdrs = physical_pdrs
        self.physical_delays = physical_delays
        self.physical_bers = physical_bers if physical_bers is not None else np.zeros_like(physical_pdrs)
        self.physical_cond_delays = physical_cond_delays if physical_cond_delays is not None else physical_delays
        self.physical_rssis = physical_rssis if physical_rssis is not None else np.zeros_like(physical_pdrs)
        self.ml_fallback_count = int(ml_fallback_count)
        self.ml_total_predictions = int(ml_total_predictions)
        self.alive_nodes = alive_nodes
        self.bs_id = bs_id


        self._rev_adj: Dict[int, List[Tuple[int, int]]] = {nid: [] for nid in alive_nodes}
        self._edge_to_idx: Dict[Tuple[int, int], int] = {}
        for idx, (u, v) in enumerate(edges):
            self._edge_to_idx[(u, v)] = idx
            if v in self._rev_adj:
                self._rev_adj[v].append((u, idx))
        self._edge_expectations: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    def get_edge_expectations(
        self,
        energy_model: EnergyModel,
        max_attempts: int,
        frag_sizes: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute & cache (tx_exp, rx_exp, link_succ_prob, cond_hop_delay) arrays once per snapshot."""
        if self._edge_expectations is not None:
            return self._edge_expectations

        num_edges = len(self.edges)
        if num_edges == 0:
            empty = np.array([], dtype=np.float64)
            self._edge_expectations = (empty, empty, empty, empty)
            return self._edge_expectations

        bers = self.physical_bers
        frame_pdrs = self.physical_pdrs
        d = self.distances
        d0 = energy_model.d0
        e_elec = energy_model.e_elec
        tx_per_bit = np.where(d < d0, e_elec + energy_model.epsilon_fs * (d ** 2),
                              e_elec + energy_model.epsilon_mp * (d ** 4))

        tx_exp = np.zeros(num_edges, dtype=np.float64)
        rx_exp = np.zeros(num_edges, dtype=np.float64)
        r_j = np.ones(num_edges, dtype=np.float64)
        use_ber_mask = bers > 0.0

        for l_j in frag_sizes:
            p_j = np.where(
                use_ber_mask,
                np.clip(np.power(1.0 - bers, l_j), 0.0, 1.0),
                np.clip(frame_pdrs, 0.0, 1.0)
            )
            q_j = 1.0 - np.power(1.0 - p_j, max_attempts)
            e_n_j = np.where(p_j < 1e-6, float(max_attempts), q_j / np.maximum(p_j, 1e-12))
            tx_exp += r_j * e_n_j * (l_j * tx_per_bit)
            rx_exp += r_j * e_n_j * (l_j * e_elec)
            r_j *= q_j

        self._edge_expectations = (tx_exp, rx_exp, r_j, self.physical_cond_delays)
        return self._edge_expectations

    def get_edge_index(self, u: int, v: int) -> int:
        return self._edge_to_idx.get((u, v), -1)

    def find_all_paths_fast(
        self,
        weights: Tuple[float, float, float, float]
    ) -> Dict[int, List[int]]:
        """Fast single-source reverse Dijkstra using precomputed reverse adjacency list."""
        w_e, w_d, w_l, w_r = weights
        costs = (
            w_e * self.norm_energies +
            w_d * self.norm_delays +
            w_l * self.norm_losses +
            w_r * self.rx_depletions
        )
        return self.find_all_paths_with_costs(costs)

    def find_all_paths_with_costs(self, costs: np.ndarray) -> Dict[int, List[int]]:
        """Fast single-source reverse Dijkstra for any precomputed edge cost array."""
        costs_list = np.maximum(costs, 1e-5).tolist()

        paths: Dict[int, List[int]] = {nid: [] for nid in self.alive_nodes if nid != self.bs_id}
        rev_adj = self._rev_adj
        bs_id = self.bs_id
        if bs_id not in rev_adj:
            return paths

        heappop = heapq.heappop
        heappush = heapq.heappush
        inf = float("inf")
        dist: Dict[int, float] = {bs_id: 0.0}
        next_hop: Dict[int, int] = {}
        heap: List[Tuple[float, int]] = [(0.0, bs_id)]

        while heap:
            d_v, v = heappop(heap)
            if d_v > dist.get(v, inf):
                continue
            for u, e_idx in rev_adj[v]:
                nd = d_v + costs_list[e_idx]
                if nd < dist.get(u, inf):
                    dist[u] = nd
                    next_hop[u] = v
                    heappush(heap, (nd, u))

        for nid in paths:
            if nid not in next_hop:
                continue
            path = [nid]
            cur = nid
            while cur != bs_id:
                cur = next_hop[cur]
                path.append(cur)
            paths[nid] = path

        return paths

    def build_graph_with_weights(
        self,
        weights: Tuple[float, float, float, float]
    ) -> nx.DiGraph:
        """Construct weighted DiGraph for a specific candidate weight vector w."""
        w_e, w_d, w_l, w_r = weights
        costs = (
            w_e * self.norm_energies +
            w_d * self.norm_delays +
            w_l * self.norm_losses +
            w_r * self.rx_depletions
        )
        costs = np.maximum(costs, 1e-5)

        G = nx.DiGraph()
        G.add_nodes_from(self.alive_nodes)

        for idx, (u, v) in enumerate(self.edges):
            G.add_edge(
                u, v,
                cost=float(costs[idx]),
                distance=float(self.distances[idx]),
                tx_energy=float(self.tx_energies[idx]),
                total_energy=float(self.total_energies[idx]),
                delay=float(self.delays[idx]),
                pdr=float(self.pdrs[idx]),
                physical_pdr=float(self.physical_pdrs[idx]),
                physical_delay=float(self.physical_delays[idx]),
                physical_cond_delay=float(self.physical_cond_delays[idx]),
                physical_ber=float(self.physical_bers[idx])
            )

        return G

class RoutingEngine:
    """Builds multi-objective link cost metrics and computes shortest paths to Sink."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        ml_predictor: Optional[LinkQualityPredictor] = None
    ):
        self.config = config or SimulationConfig()
        self.energy_model = EnergyModel(self.config)
        self.channel = WirelessChannel(self.config)
        self.ml_predictor = ml_predictor

        self.max_tx_energy = self.energy_model.max_possible_tx_energy()
        self.max_total_link_energy = self.energy_model.expected_link_energy(
            self.config.comm_range, expected_attempts=1.0
        )
        self.max_link_delay = 150.0

    def create_graph_snapshot(
        self,
        network: IoTNetwork,
        use_ml: bool = True,
        routing_mode: str = "hybrid",
        rng: Optional[np.random.RandomState] = None,
        seed: Optional[int] = None,
        round_num: Optional[int] = None,
        link_history: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None
    ) -> GraphSnapshot:
        """Create a frozen snapshot of graph features for consistent PSO and routing evaluation.

        When ``link_history`` is supplied, ML predictions strictly use causal past-window
        observations (no future shadowing leakage) with analytical-physics fallback for
        unobserved or OOD links.
        """
        bs_id = network.bs_id
        alive_nodes = sorted(nid for nid, node in network.nodes.items() if node.is_alive)

        edges = []
        distances_list = []
        tx_costs_list = []
        total_costs_list = []
        rx_depletions_list = []
        rssi_list = []
        queue_loads_list = []

        frag_sizes = self.config.fragment_sizes_bits
        rx_sum_e = sum(self.energy_model.rx_energy(num_bits=l_j) for l_j in frag_sizes)
        da_e = self.energy_model.aggregation_energy()
        reciprocal_rssi_cache: Dict[Tuple[int, int], float] = {}

        for u_id in alive_nodes:
            if u_id == bs_id:
                continue
            node_u = network.nodes[u_id]

            for v_id in alive_nodes:
                if u_id == v_id:
                    continue

                dist = network.get_distance(u_id, v_id)
                if dist > self.config.comm_range:
                    continue

                node_v = network.nodes[v_id]
                tx_cost = sum(self.energy_model.tx_energy(dist, num_bits=l_j) for l_j in frag_sizes)

                if node_v.is_base_station:
                    total_energy = tx_cost
                    rx_dep = 0.0
                    rx_ratio = 1.0
                else:
                    total_energy = tx_cost + rx_sum_e + da_e
                    rx_ratio = node_v.residual_ratio
                    rx_dep = 1.0 - rx_ratio
                    if rx_ratio < 0.15:
                        rx_dep += 2.0

                if seed is not None and round_num is not None:
                    pair_key = (u_id, v_id) if u_id <= v_id else (v_id, u_id)
                    if pair_key in reciprocal_rssi_cache:
                        rssi = reciprocal_rssi_cache[pair_key]
                    else:
                        if not hasattr(self, "_shadow_state"):
                            self._shadow_state = {}
                        s_key = (int(seed), int(round_num), pair_key[0], pair_key[1])
                        if s_key in self._shadow_state:
                            shadow_t = self._shadow_state[s_key]
                        else:
                            edge_rng = make_channel_rng(seed, round_num, u_id, v_id)
                            eps_t = float(edge_rng.normal(0, self.channel.sigma))
                            prev_key = (int(seed), int(round_num) - 1, pair_key[0], pair_key[1])
                            if round_num <= 1 or prev_key not in self._shadow_state:
                                shadow_t = eps_t
                            else:
                                rho = 0.75
                                shadow_t = rho * self._shadow_state[prev_key] + float(np.sqrt(1.0 - rho * rho)) * eps_t
                            self._shadow_state[s_key] = shadow_t

                            old_prefix = int(round_num) - 2
                            if old_prefix >= 1:
                                self._shadow_state.pop((int(seed), old_prefix, pair_key[0], pair_key[1]), None)
                        rssi = float(self.channel.compute_rssi(dist, apply_shadowing=False) - shadow_t)
                        reciprocal_rssi_cache[pair_key] = rssi
                else:
                    rssi = self.channel.compute_rssi(dist, apply_shadowing=True, rng=rng)
                q_load = min(self.config.max_queue_load, node_v.queue_load)

                edges.append((u_id, v_id))
                distances_list.append(dist)
                tx_costs_list.append(tx_cost)
                total_costs_list.append(total_energy)
                rx_depletions_list.append(rx_dep)
                rssi_list.append(rssi)
                queue_loads_list.append(q_load)

        num_edges = len(edges)
        if num_edges == 0:
            empty = np.array([], dtype=np.float64)
            return GraphSnapshot([], empty, empty, empty, empty, empty, empty, empty,
                                 empty, empty, empty, empty, alive_nodes, bs_id, empty, empty, empty)

        distances = np.array(distances_list, dtype=np.float64)
        tx_energies = np.array(tx_costs_list, dtype=np.float64)
        total_energies = np.array(total_costs_list, dtype=np.float64)
        rx_depletions = np.array(rx_depletions_list, dtype=np.float64)
        rssi_array = np.array(rssi_list, dtype=np.float64)
        norm_energies = np.minimum(1.0, total_energies / (self.max_total_link_energy + 1e-12))

        q_arr = np.array(queue_loads_list, dtype=np.float64)
        snr_db = rssi_array - self.channel.noise_floor
        snr_lin = np.power(10.0, snr_db / 10.0)
        snr_chip = np.maximum(0.0, snr_lin * (self.channel.noise_bandwidth / self.channel.chip_rate))
        physical_bers = self.channel.ber_ieee_802_15_4_oqpsk_vec(snr_chip)
        physical_pdrs = np.clip(np.power(1.0 - physical_bers, self.config.frame_size_bits), 0.0, 1.0)

        max_att = self.channel.max_attempts
        t_prop = (np.maximum(0.0, distances) / self.config.propagation_speed) * 1000.0
        t_proc = self.config.proc_delay_base_ms
        physical_delays = np.zeros(num_edges, dtype=np.float64)
        physical_cond_delays = np.zeros(num_edges, dtype=np.float64)
        r_j = np.ones(num_edges, dtype=np.float64)

        for l_j in frag_sizes:
            p_j = np.clip(np.power(1.0 - physical_bers, l_j), 0.0, 1.0)
            q_j = 1.0 - np.power(1.0 - p_j, max_att)
            e_n_j = np.where(p_j < 1e-6, float(max_att), q_j / np.maximum(p_j, 1e-12))

            num_cond = 1.0 - np.power(1.0 - p_j, max_att) * (1.0 + max_att * p_j)
            e_n_cond = np.where(
                q_j <= 1e-12,
                0.5 * (max_att + 1.0),
                np.where(p_j >= 1.0 - 1e-12, 1.0, np.clip(num_cond / np.maximum(p_j * q_j, 1e-18), 1.0, float(max_att)))
            )
            t_tx = (l_j / self.channel.bit_rate) * 1000.0
            d_single = t_tx + t_prop + t_proc + q_arr * (t_tx * 0.4)
            physical_delays += r_j * e_n_j * d_single
            physical_cond_delays += e_n_cond * d_single
            r_j *= q_j

        phys_link_succ_probs = r_j
        ml_fallback_count = 0
        ml_total_predictions = 0

        if routing_mode == "sp":
            pdrs = np.ones(num_edges, dtype=np.float64)
            delays = distances.copy()
            norm_delays = np.zeros(num_edges, dtype=np.float64)
            norm_losses = np.zeros(num_edges, dtype=np.float64)

        elif routing_mode == "energy_dijkstra":
            pdrs = np.ones(num_edges, dtype=np.float64)
            delays = np.ones(num_edges, dtype=np.float64)
            norm_delays = np.zeros(num_edges, dtype=np.float64)
            norm_losses = np.zeros(num_edges, dtype=np.float64)

        elif routing_mode == "pso" or not use_ml or self.ml_predictor is None:

            d_clamped = np.maximum(distances, self.channel.d_ref)
            mean_rssi = self.channel.tx_power - (
                self.channel.pl_d_ref + 10.0 * self.channel.eta * np.log10(d_clamped / self.channel.d_ref)
            )
            fb_snr_db = mean_rssi - self.channel.noise_floor
            fb_snr_lin = np.power(10.0, fb_snr_db / 10.0)
            fb_snr_chip = np.maximum(0.0, fb_snr_lin * (self.channel.noise_bandwidth / self.channel.chip_rate))
            fb_bers = self.channel.ber_ieee_802_15_4_oqpsk_vec(fb_snr_chip)
            pdrs = np.clip(np.power(1.0 - fb_bers, self.config.frame_size_bits), 0.0, 1.0)

            delays = np.zeros(num_edges, dtype=np.float64)
            fb_r_j = np.ones(num_edges, dtype=np.float64)
            for l_j in frag_sizes:
                p_j = np.clip(np.power(1.0 - fb_bers, l_j), 0.0, 1.0)
                q_j = 1.0 - np.power(1.0 - p_j, max_att)
                e_n_j = np.where(p_j < 1e-6, float(max_att), q_j / np.maximum(p_j, 1e-12))
                t_tx = (l_j / self.channel.bit_rate) * 1000.0
                d_single = t_tx + t_prop + t_proc + q_arr * (t_tx * 0.4)
                delays += fb_r_j * e_n_j * d_single
                fb_r_j *= q_j

            norm_delays = np.minimum(1.0, delays / self.max_link_delay)
            q_frag = 1.0 - np.power(1.0 - pdrs, max_att)
            norm_losses = 1.0 - np.power(q_frag, self.config.num_fragments)

        else:

            d_clamped = np.maximum(distances, self.channel.d_ref)
            mean_rssi = self.channel.tx_power - (
                self.channel.pl_d_ref + 10.0 * self.channel.eta * np.log10(d_clamped / self.channel.d_ref)
            )
            fb_snr_db = mean_rssi - self.channel.noise_floor
            fb_snr_lin = np.power(10.0, fb_snr_db / 10.0)
            fb_snr_chip = np.maximum(0.0, fb_snr_lin * (self.channel.noise_bandwidth / self.channel.chip_rate))
            fb_bers = self.channel.ber_ieee_802_15_4_oqpsk_vec(fb_snr_chip)
            fb_pdrs = np.clip(np.power(1.0 - fb_bers, self.config.frame_size_bits), 0.0, 1.0)

            fb_delays = np.zeros(num_edges, dtype=np.float64)
            fb_r_j = np.ones(num_edges, dtype=np.float64)
            for l_j in frag_sizes:
                p_j = np.clip(np.power(1.0 - fb_bers, l_j), 0.0, 1.0)
                q_j = 1.0 - np.power(1.0 - p_j, max_att)
                e_n_j = np.where(p_j < 1e-6, float(max_att), q_j / np.maximum(p_j, 1e-12))
                t_tx = (l_j / self.channel.bit_rate) * 1000.0
                d_single = t_tx + t_prop + t_proc + q_arr * (t_tx * 0.4)
                fb_delays += fb_r_j * e_n_j * d_single
                fb_r_j *= q_j

            feat_rows = []
            unobserved_mask = np.zeros(num_edges, dtype=bool)
            for idx, (u_id, v_id) in enumerate(edges):
                if link_history is not None:
                    hist = link_history.get((u_id, v_id))
                    if hist and hist.get("obs_count", 0) > 0:
                        feat_rows.append([
                            float(distances[idx]),
                            float(hist["prev_rssi_dbm"]),
                            float(hist["prev_snr_db"]),
                            float(hist["ewma_pdr"]),
                            float(hist["ewma_retx"]),
                            float(hist["ewma_delay_ms"]),
                            float(q_arr[idx]),
                        ])
                    else:
                        unobserved_mask[idx] = True
                        feat_rows.append([
                            float(distances[idx]),
                            float(mean_rssi[idx]),
                            float(fb_snr_db[idx]),
                            float(fb_pdrs[idx]),
                            float(max(0.0, self.channel.expected_transmissions(float(fb_pdrs[idx])) - 1.0)),
                            float(fb_delays[idx]),
                            float(q_arr[idx]),
                        ])
                else:

                    obs_rssi = float(rssi_array[idx])
                    obs_pdr = float(physical_pdrs[idx])
                    feat_rows.append([
                        float(distances[idx]),
                        obs_rssi,
                        obs_rssi - self.channel.noise_floor,
                        obs_pdr,
                        float(max(0.0, self.channel.expected_transmissions(obs_pdr) - 1.0)),
                        float(physical_delays[idx]),
                        float(q_arr[idx]),
                    ])

            feat_matrix = np.array(feat_rows, dtype=np.float64)
            ml_total_predictions = num_edges
            if hasattr(self.ml_predictor, "predict_with_fallback") and link_history is not None:
                pdrs, delays, fb_mask = self.ml_predictor.predict_with_fallback(
                    feat_matrix, fb_pdrs, fb_delays, force_fallback_mask=unobserved_mask
                )
                ml_fallback_count = int(np.sum(fb_mask))
            else:
                pdrs, delays = self.ml_predictor.predict_batch(feat_matrix)

            norm_delays = np.minimum(1.0, delays / self.max_link_delay)
            p_clipped = np.clip(pdrs, 0.0, 1.0)
            q_frag = 1.0 - np.power(1.0 - p_clipped, max_att)
            norm_losses = 1.0 - np.power(q_frag, self.config.num_fragments)

        return GraphSnapshot(
            edges=edges,
            distances=distances,
            tx_energies=tx_energies,
            total_energies=total_energies,
            norm_energies=norm_energies,
            norm_delays=norm_delays,
            norm_losses=norm_losses,
            rx_depletions=rx_depletions,
            pdrs=pdrs,
            delays=delays,
            physical_pdrs=physical_pdrs,
            physical_delays=physical_delays,
            alive_nodes=alive_nodes,
            bs_id=bs_id,
            physical_bers=physical_bers,
            physical_cond_delays=physical_cond_delays,
            physical_rssis=rssi_array,
            ml_fallback_count=ml_fallback_count,
            ml_total_predictions=ml_total_predictions,
        )

    def build_weighted_graph(
        self,
        network: IoTNetwork,
        weights: Tuple[float, float, float, float] = (0.35, 0.20, 0.25, 0.20),
        use_ml: bool = True,
        routing_mode: str = "hybrid",
        snapshot: Optional[GraphSnapshot] = None,
        rng: Optional[np.random.RandomState] = None,
        seed: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> nx.DiGraph:
        if snapshot is None:
            snapshot = self.create_graph_snapshot(
                network, use_ml=use_ml, routing_mode=routing_mode,
                rng=rng, seed=seed, round_num=round_num
            )

        if routing_mode == "sp":
            G = nx.DiGraph()
            G.add_nodes_from(snapshot.alive_nodes)
            for idx, (u, v) in enumerate(snapshot.edges):
                G.add_edge(
                    u, v,
                    cost=float(snapshot.distances[idx]),
                    distance=float(snapshot.distances[idx]),
                    tx_energy=float(snapshot.tx_energies[idx]),
                    total_energy=float(snapshot.total_energies[idx]),
                    delay=float(snapshot.delays[idx]),
                    pdr=float(snapshot.pdrs[idx]),
                    physical_pdr=float(snapshot.physical_pdrs[idx]),
                    physical_delay=float(snapshot.physical_delays[idx]),
                    physical_cond_delay=float(snapshot.physical_cond_delays[idx]),
                    physical_ber=float(snapshot.physical_bers[idx])
                )
            return G

        elif routing_mode == "energy_dijkstra":
            G = nx.DiGraph()
            G.add_nodes_from(snapshot.alive_nodes)
            for idx, (u, v) in enumerate(snapshot.edges):
                G.add_edge(
                    u, v,
                    cost=float(snapshot.total_energies[idx] * 1e6),
                    distance=float(snapshot.distances[idx]),
                    tx_energy=float(snapshot.tx_energies[idx]),
                    total_energy=float(snapshot.total_energies[idx]),
                    delay=float(snapshot.delays[idx]),
                    pdr=float(snapshot.pdrs[idx]),
                    physical_pdr=float(snapshot.physical_pdrs[idx]),
                    physical_delay=float(snapshot.physical_delays[idx]),
                    physical_cond_delay=float(snapshot.physical_cond_delays[idx]),
                    physical_ber=float(snapshot.physical_bers[idx])
                )
            return G

        return snapshot.build_graph_with_weights(weights)

    def find_all_paths_to_sink(
        self,
        G: nx.DiGraph,
        sink_id: int = 0
    ) -> Dict[int, List[int]]:
        paths: Dict[int, List[int]] = {
            node_id: [] for node_id in G.nodes() if node_id != sink_id
        }

        if sink_id not in G:
            return paths

        G_rev = G.reverse(copy=False)
        try:
            rev_paths = nx.single_source_dijkstra_path(G_rev, source=sink_id, weight="cost")
            for node_id, rev_path in rev_paths.items():
                if node_id != sink_id:
                    paths[node_id] = list(reversed(rev_path))
        except Exception:
            for node_id in G.nodes():
                if node_id == sink_id:
                    continue
                try:
                    paths[node_id] = nx.shortest_path(G, source=node_id, target=sink_id, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    paths[node_id] = []

        return paths
