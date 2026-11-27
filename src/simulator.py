import math
import hashlib
import struct
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import networkx as nx
from tqdm import tqdm

from src.config import SimulationConfig
from src.network import IoTNetwork
from src.energy import EnergyModel
from src.channel import WirelessChannel, make_channel_rng
from src.routing import RoutingEngine
from src.pso import PSORoutingOptimizer
from src.ml_predictor import LinkQualityPredictor

def _packet_outcome_u01(
    base_seed: int,
    round_num: int,
    src_id: int,
    tx_id: int,
    rx_id: int,
    packet_idx: int,
    frag_idx: int,
    attempt: int
) -> float:
    """Fast deterministic CRN uniform draw in [0.0, 1.0) keyed on physical link & packet attempt.

    Uses 53-bit IEEE-754 uniform mapping from MD5 digest of
    (base_seed, round_num, src_id, tx_id, rx_id, packet_idx, frag_idx, attempt).
    Avoids allocating millions of Mersenne Twister RandomState arrays in the inner loop
    while preserving strict key-based Common Random Numbers across protocols.
    """
    digest = hashlib.md5(struct.pack(
        '>IIIIIIII',
        int(base_seed), int(round_num), int(src_id),
        int(tx_id), int(rx_id), int(packet_idx),
        int(frag_idx), int(attempt)
    )).digest()
    return float((int.from_bytes(digest[:8], 'big') >> 11) / 9007199254740992.0)


def _make_packet_rng(
    base_seed: int,
    round_num: int,
    src_id: int,
    tx_id: int,
    rx_id: int,
    packet_idx: int,
    frag_idx: int,
    attempt: int
) -> np.random.RandomState:
    """Create a deterministic RandomState keyed on the physical link and packet attempt (CRN)."""
    h = hashlib.md5()
    h.update(struct.pack(
        '>IIIIIIII',
        int(base_seed), int(round_num), int(src_id),
        int(tx_id), int(rx_id), int(packet_idx),
        int(frag_idx), int(attempt)
    ))
    derived_seed = int.from_bytes(h.digest()[:4], 'big') % (2**31)
    return np.random.RandomState(derived_seed)


class SimulationResult:
    """Stores full time-series and aggregate benchmark metrics for one protocol simulation."""

    def __init__(self, protocol_name: str, config: Optional[SimulationConfig] = None):
        self.protocol_name = protocol_name
        self.config = config or SimulationConfig()

        self.rounds: List[int] = []
        self.alive_nodes: List[int] = []
        self.dead_nodes: List[int] = []
        self.total_residual_energy: List[float] = []
        self.cumulative_energy_consumed: List[float] = []
        self.round_energy_consumed: List[float] = []
        self.round_pdr: List[float] = []
        self.cumulative_pdr: List[float] = []
        self.avg_delays: List[float] = []
        self.energy_variance: List[float] = []

        self.fnd_round: int = -1
        self.hnd_round: int = -1
        self.lnd_round: int = -1

        self.total_packets_generated: int = 0
        self.total_packets_delivered: int = 0
        self.total_packets_dropped: int = 0
        self.packets_remaining_in_queue: int = 0
        self.total_delivered_delay_sum: float = 0.0
        self.total_airtime_ms: float = 0.0
        self.queue_conservation_history: List[Dict[int, Dict[str, int]]] = []

        self.pso_convergence_curve: List[float] = []
        self.initial_weights: Optional[Tuple[float, float, float, float]] = None
        self.final_weights: Optional[Tuple[float, float, float, float]] = None
        self.ml_fallback_count: int = 0
        self.ml_total_predictions: int = 0

    @property
    def ml_fallback_rate(self) -> float:
        if self.ml_total_predictions == 0:
            return 0.0
        return float(self.ml_fallback_count / self.ml_total_predictions)

    @property
    def overall_pdr(self) -> float:
        if self.total_packets_generated == 0:
            return 0.0
        return float(self.total_packets_delivered / self.total_packets_generated)

    @property
    def weighted_mean_delay(self) -> float:
        """Mean end-to-end delay conditioned on successful application packet delivery (ms)."""
        if self.total_packets_delivered == 0:
            return 0.0
        return float(self.total_delivered_delay_sum / self.total_packets_delivered)

    @property
    def mean_airtime_per_generated_packet_ms(self) -> float:
        """Expected airtime per generated application packet (including dropped packets)."""
        if self.total_packets_generated == 0:
            return 0.0
        return float(self.total_airtime_ms / self.total_packets_generated)

    @property
    def final_residual_energy(self) -> float:
        return self.total_residual_energy[-1] if self.total_residual_energy else 0.0

    @property
    def energy_per_delivered_packet_mj(self) -> float:
        if self.total_packets_delivered == 0:
            return 0.0
        tot_e = self.cumulative_energy_consumed[-1] if self.cumulative_energy_consumed else 0.0
        return float((tot_e * 1000.0) / self.total_packets_delivered)

    def compute_milestones(self, total_sensor_nodes: int):
        half_threshold = int(math.ceil(total_sensor_nodes / 2.0))
        for r, dead_count in zip(self.rounds, self.dead_nodes):
            if dead_count >= 1 and self.fnd_round == -1:
                self.fnd_round = r
            if dead_count >= half_threshold and self.hnd_round == -1:
                self.hnd_round = r
            if dead_count >= total_sensor_nodes and self.lnd_round == -1:
                self.lnd_round = r

class NetworkSimulator:
    """Discrete-round simulator executing network routing protocols with sequential
    IEEE 802.15.4 fragmentation, synchronized queue_load, and CRN packet RNG.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        ml_predictor: Optional[LinkQualityPredictor] = None
    ):
        self.config = config or SimulationConfig()
        self.energy_model = EnergyModel(self.config)
        self.channel = WirelessChannel(self.config)
        self.ml_predictor = ml_predictor
        self.routing_engine = RoutingEngine(self.config, ml_predictor=self.ml_predictor)

    def run_protocol(
        self,
        protocol: str,
        initial_network: IoTNetwork,
        initial_weights: Optional[Tuple[float, float, float, float]] = None,
        pso_update_interval: Optional[int] = None,
        show_progress: bool = True,
        seed: Optional[int] = None
    ) -> SimulationResult:
        network = initial_network.clone()
        result = SimulationResult(protocol_name=protocol, config=self.config)

        actual_seed = seed if seed is not None else self.config.random_seed

        update_interval = (
            pso_update_interval if pso_update_interval is not None
            else self.config.pso_update_interval
        )

        pso_optimizer: Optional[PSORoutingOptimizer] = None
        if protocol == "pso":
            pso_optimizer = PSORoutingOptimizer(
                self.config, routing_engine=self.routing_engine, use_ml=False, seed=actual_seed
            )
        elif protocol == "ml_pso":
            pso_optimizer = PSORoutingOptimizer(
                self.config, routing_engine=self.routing_engine, use_ml=True, seed=actual_seed
            )

        current_weights = initial_weights or (0.35, 0.20, 0.25, 0.20)
        last_dead_count = 0
        max_attempts = 1 + self.config.max_retransmissions
        frag_sizes = self.config.fragment_sizes_bits
        da_cost_app = self.energy_model.aggregation_energy()


        node_queues: Dict[int, deque] = {}
        for nid, n in network.nodes.items():
            n.queue_load = 0
            if not n.is_base_station:
                node_queues[nid] = deque()

        link_history: Dict[Tuple[int, int], Dict[str, float]] = {}
        ewma_alpha = 0.35

        iterator = range(1, self.config.total_rounds + 1)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Simulating {protocol:<16}")

        for current_round in iterator:
            alive_sensor_nodes = network.get_alive_sensor_nodes()
            num_alive = len(alive_sensor_nodes)
            num_dead = self.config.num_nodes - num_alive

            if num_alive == 0:
                result.rounds.append(current_round)
                result.alive_nodes.append(0)
                result.dead_nodes.append(self.config.num_nodes)
                result.total_residual_energy.append(0.0)
                prev_e = result.cumulative_energy_consumed[-1] if result.cumulative_energy_consumed else 0.0
                result.cumulative_energy_consumed.append(prev_e)
                result.round_energy_consumed.append(0.0)
                result.round_pdr.append(0.0)
                result.cumulative_pdr.append(result.overall_pdr)
                result.avg_delays.append(0.0)
                result.energy_variance.append(0.0)
                continue


            for nid, n in network.nodes.items():
                if not n.is_base_station and not n.is_alive and node_queues[nid]:
                    result.total_packets_dropped += len(node_queues[nid])
                    node_queues[nid].clear()
                    n.queue_load = 0

            round_q_stats: Dict[int, Dict[str, int]] = {
                nid: {
                    "q_start": network.nodes[nid].queue_load,
                    "arrivals": 0,
                    "served": 0,
                    "dropped": 0,
                    "q_end": network.nodes[nid].queue_load,
                }
                for nid in node_queues
            }

            use_ml_flag = (protocol == "ml_pso")
            routing_mode = protocol if protocol in ("sp", "energy_dijkstra") else ("hybrid" if use_ml_flag else "pso")
            round_snapshot = self.routing_engine.create_graph_snapshot(
                network=network,
                use_ml=use_ml_flag,
                routing_mode=routing_mode,
                seed=actual_seed,
                round_num=current_round,
                link_history=link_history if use_ml_flag else None,
            )
            result.ml_fallback_count += round_snapshot.ml_fallback_count
            result.ml_total_predictions += round_snapshot.ml_total_predictions

            edge_ber_lookup: Dict[Tuple[int, int], float] = {
                (eu, ev): float(round_snapshot.physical_bers[edge_idx])
                for edge_idx, (eu, ev) in enumerate(round_snapshot.edges)
            }
            edge_rssi_lookup: Dict[Tuple[int, int], float] = {
                (eu, ev): float(round_snapshot.physical_rssis[edge_idx])
                for edge_idx, (eu, ev) in enumerate(round_snapshot.edges)
            }
            edge_phys_pdr_lookup: Dict[Tuple[int, int], float] = {
                (eu, ev): float(round_snapshot.physical_pdrs[edge_idx])
                for edge_idx, (eu, ev) in enumerate(round_snapshot.edges)
            }

            if pso_optimizer is not None:
                should_update_pso = (
                    (current_round == 1 and initial_weights is None) or
                    (current_round > 1 and (
                        current_round % update_interval == 0 or
                        num_dead >= last_dead_count + 3
                    ))
                )
                if should_update_pso:
                    current_weights, _ = pso_optimizer.optimize(network, snapshot=round_snapshot)
                    last_dead_count = num_dead
                    if current_round == 1:
                        result.initial_weights = current_weights
                        result.pso_convergence_curve = list(pso_optimizer.convergence_curve)

            if current_round == 1 and result.initial_weights is None:
                result.initial_weights = current_weights


            if protocol in ("pso", "ml_pso"):
                paths = round_snapshot.find_all_paths_fast(current_weights)
            elif protocol == "sp":
                paths = round_snapshot.find_all_paths_with_costs(round_snapshot.distances)
            elif protocol == "energy_dijkstra":
                paths = round_snapshot.find_all_paths_with_costs(round_snapshot.total_energies * 1e6)
            else:
                G = self.routing_engine.build_weighted_graph(
                    network=network,
                    weights=current_weights,
                    use_ml=use_ml_flag,
                    routing_mode=protocol,
                    snapshot=round_snapshot
                )
                paths = self.routing_engine.find_all_paths_to_sink(G, sink_id=network.bs_id)

            round_energy_start = network.get_total_energy_consumed()
            round_packets_gen = num_alive * self.config.packets_per_round
            round_packets_deliv = 0
            round_delays = []

            result.total_packets_generated += round_packets_gen

            service_remaining: Dict[int, int] = {
                n.node_id: self.config.queue_service_capacity
                for n in alive_sensor_nodes
            }
            max_q = self.config.max_queue_load

            def _flush_dead_backlog(nid: int) -> None:
                q = node_queues.get(nid)
                if q:
                    dropped_cnt = len(q)
                    q.clear()
                    network.nodes[nid].queue_load = 0
                    result.total_packets_dropped += dropped_cnt
                    round_q_stats[nid]["dropped"] += dropped_cnt

            def _forward_from_node(start_nid: int, orig_src_id: int, pkt_idx: int, initial_delay: float) -> None:
                nonlocal round_packets_deliv
                path = paths.get(start_nid, [])
                if not path or len(path) < 2 or path[-1] != network.bs_id:
                    result.total_packets_dropped += 1
                    round_q_stats[start_nid]["dropped"] += 1
                    return


                service_remaining[start_nid] -= 1
                round_q_stats[start_nid]["served"] += 1
                if start_nid != orig_src_id:
                    network.nodes[start_nid].packets_forwarded += 1

                path_delay = initial_delay

                for hop_idx in range(len(path) - 1):
                    tx_id = path[hop_idx]
                    rx_id = path[hop_idx + 1]

                    tx_node = network.nodes[tx_id]
                    rx_node = network.nodes[rx_id]

                    if not tx_node.is_alive or not rx_node.is_alive:
                        result.total_packets_dropped += 1
                        result.total_airtime_ms += path_delay
                        return

                    d = network.get_distance(tx_id, rx_id)
                    ber = edge_ber_lookup.get((tx_id, rx_id))
                    if ber is None:
                        fallback_rng = make_channel_rng(actual_seed, current_round, tx_id, rx_id)
                        ber = self.channel.compute_ber(d, apply_shadowing=True, rng=fallback_rng)

                    hop_success = True
                    hop_delay = 0.0
                    hop_total_attempts = 0
                    hop_frags_attempted = 0

                    rx_q = min(max_q, rx_node.queue_load)

                    for frag_idx, frag_bits in enumerate(frag_sizes):
                        tx_cost_frame = self.energy_model.tx_energy(d, num_bits=frag_bits)
                        rx_cost_frame = self.energy_model.rx_energy(num_bits=frag_bits)
                        single_frame_delay = self.channel.compute_single_hop_delay(
                            d, queue_load=rx_q, frame_bits=frag_bits
                        )
                        frame_pdr_physical = float(np.clip((1.0 - ber) ** frag_bits, 0.0, 1.0))

                        frag_success = False
                        attempts_used = 0

                        for attempt in range(1, max_attempts + 1):
                            if not tx_node.is_alive or not rx_node.is_alive:
                                break
                            if not tx_node.can_consume(tx_cost_frame):
                                break

                            attempts_used = attempt
                            tx_node.consume_energy(tx_cost_frame, current_round)
                            if not tx_node.is_alive:
                                _flush_dead_backlog(tx_id)

                            if not rx_node.is_base_station:
                                if not rx_node.can_consume(rx_cost_frame):
                                    break
                                rx_node.consume_energy(rx_cost_frame, current_round)
                                if not rx_node.is_alive:
                                    _flush_dead_backlog(rx_id)

                            if _packet_outcome_u01(
                                actual_seed, current_round, orig_src_id,
                                tx_id, rx_id, pkt_idx, frag_idx, attempt
                            ) < frame_pdr_physical:
                                frag_success = True
                                break

                        hop_total_attempts += attempts_used
                        if attempts_used > 0:
                            hop_frags_attempted += 1
                        hop_delay += attempts_used * single_frame_delay

                        if not frag_success:
                            hop_success = False
                            break

                    path_delay += hop_delay


                    if use_ml_flag and hop_frags_attempted > 0:
                        obs_rssi = edge_rssi_lookup.get((tx_id, rx_id), self.channel.compute_rssi(d, apply_shadowing=False))
                        obs_snr = float(obs_rssi - self.channel.noise_floor)
                        obs_pdr = edge_phys_pdr_lookup.get((tx_id, rx_id), 1.0 if hop_success else 0.0)
                        obs_retx = max(0.0, (hop_total_attempts / hop_frags_attempted) - 1.0)
                        obs_delay = max(0.1, hop_delay)
                        h_key = (tx_id, rx_id)
                        if h_key not in link_history:
                            link_history[h_key] = {
                                "prev_rssi_dbm": obs_rssi,
                                "prev_snr_db": obs_snr,
                                "ewma_pdr": obs_pdr,
                                "ewma_retx": obs_retx,
                                "ewma_delay_ms": obs_delay,
                                "obs_count": 1.0,
                            }
                        else:
                            st = link_history[h_key]
                            st["prev_rssi_dbm"] = obs_rssi
                            st["prev_snr_db"] = obs_snr
                            st["ewma_pdr"] = (1.0 - ewma_alpha) * st["ewma_pdr"] + ewma_alpha * obs_pdr
                            st["ewma_retx"] = (1.0 - ewma_alpha) * st["ewma_retx"] + ewma_alpha * obs_retx
                            st["ewma_delay_ms"] = (1.0 - ewma_alpha) * st["ewma_delay_ms"] + ewma_alpha * obs_delay
                            st["obs_count"] += 1.0

                    if not hop_success:
                        result.total_packets_dropped += 1
                        result.total_airtime_ms += path_delay
                        return

                    if rx_node.is_base_station:
                        result.total_airtime_ms += path_delay
                        round_packets_deliv += 1
                        result.total_packets_delivered += 1
                        round_delays.append(path_delay)
                        result.total_delivered_delay_sum += path_delay
                        network.nodes[orig_src_id].packets_delivered += 1
                        return


                    if not rx_node.is_alive or not rx_node.can_consume(da_cost_app):
                        result.total_packets_dropped += 1
                        result.total_airtime_ms += path_delay
                        return

                    rx_node.consume_energy(da_cost_app, current_round)
                    if not rx_node.is_alive:
                        _flush_dead_backlog(rx_id)
                        result.total_packets_dropped += 1
                        result.total_airtime_ms += path_delay
                        return


                    round_q_stats[rx_id]["arrivals"] += 1


                    if service_remaining.get(rx_id, 0) > 0 and len(node_queues[rx_id]) == 0:
                        service_remaining[rx_id] -= 1
                        round_q_stats[rx_id]["served"] += 1
                        rx_node.packets_forwarded += 1
                        continue
                    else:

                        if len(node_queues[rx_id]) < max_q:
                            node_queues[rx_id].append((orig_src_id, pkt_idx, path_delay))
                            rx_node.queue_load = len(node_queues[rx_id])
                        else:
                            result.total_packets_dropped += 1
                            round_q_stats[rx_id]["dropped"] += 1
                        result.total_airtime_ms += path_delay
                        return


            for n in alive_sensor_nodes:
                nid = n.node_id
                q = node_queues[nid]
                while q and n.is_alive and service_remaining.get(nid, 0) > 0:
                    path = paths.get(nid, [])
                    if not path or len(path) < 2 or path[-1] != network.bs_id:
                        break
                    q_src_id, q_pkt_idx, q_delay = q.popleft()
                    n.queue_load = len(q)
                    _forward_from_node(nid, q_src_id, q_pkt_idx, q_delay)


            for src_node in alive_sensor_nodes:
                src_id = src_node.node_id
                for packet_idx in range(self.config.packets_per_round):
                    src_node.packets_generated += 1
                    round_q_stats[src_id]["arrivals"] += 1

                    if not src_node.is_alive:
                        result.total_packets_dropped += 1
                        round_q_stats[src_id]["dropped"] += 1
                        continue

                    path = paths.get(src_id, [])
                    if not path or len(path) < 2 or path[-1] != network.bs_id:
                        result.total_packets_dropped += 1
                        round_q_stats[src_id]["dropped"] += 1
                        continue

                    if service_remaining.get(src_id, 0) > 0 and len(node_queues[src_id]) == 0:
                        _forward_from_node(src_id, src_id, packet_idx, 0.0)
                    elif len(node_queues[src_id]) < max_q:
                        node_queues[src_id].append((src_id, packet_idx, 0.0))
                        src_node.queue_load = len(node_queues[src_id])
                    else:
                        result.total_packets_dropped += 1
                        round_q_stats[src_id]["dropped"] += 1

            for nid, q in node_queues.items():
                round_q_stats[nid]["q_end"] = len(q)
            result.queue_conservation_history.append(round_q_stats)

            round_energy_used = network.get_total_energy_consumed() - round_energy_start
            pdr_val = round_packets_deliv / max(1, round_packets_gen)
            avg_delay = float(np.mean(round_delays)) if round_delays else 0.0

            result.rounds.append(current_round)
            result.alive_nodes.append(network.count_alive_nodes())
            result.dead_nodes.append(network.count_dead_nodes())
            result.total_residual_energy.append(network.get_total_residual_energy())
            result.cumulative_energy_consumed.append(network.get_total_energy_consumed())
            result.round_energy_consumed.append(round_energy_used)
            result.round_pdr.append(pdr_val)
            result.cumulative_pdr.append(result.overall_pdr)
            result.avg_delays.append(avg_delay)
            result.energy_variance.append(network.get_energy_variance())


        remaining_queued = sum(len(q) for q in node_queues.values())
        result.packets_remaining_in_queue = remaining_queued
        result.total_packets_dropped += remaining_queued

        result.final_weights = current_weights
        result.compute_milestones(self.config.num_nodes)
        return result
