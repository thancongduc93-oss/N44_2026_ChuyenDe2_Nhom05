import math
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from src.config import SimulationConfig

class Node:
    """Represents an individual IoT sensor node or the Base Station (Sink)."""

    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        initial_energy: float,
        is_base_station: bool = False
    ):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.initial_energy = float("inf") if is_base_station else float(initial_energy)
        self.residual_energy = self.initial_energy
        self.is_base_station = is_base_station
        self.is_alive = True

        self.packets_generated = 0
        self.packets_forwarded = 0
        self.packets_delivered = 0
        self.energy_consumed = 0.0
        self.queue_load = 0
        self.death_round = -1

    @property
    def residual_ratio(self) -> float:
        if self.is_base_station:
            return 1.0
        if self.initial_energy <= 0:
            return 0.0
        return float(max(0.0, min(1.0, self.residual_energy / self.initial_energy)))

    def can_consume(self, amount: float) -> bool:
        """Return True if node has enough residual energy to complete an operation."""
        if self.is_base_station:
            return True
        if not self.is_alive:
            return False
        amt = max(0.0, float(amount))
        return (self.residual_energy + 1e-12) >= amt

    def consume_energy(self, amount: float, current_round: int = 0) -> bool:
        if self.is_base_station or not self.is_alive:
            return self.is_alive

        amt = max(0.0, float(amount))
        actual_consumed = min(self.residual_energy, amt)
        self.energy_consumed += actual_consumed
        self.residual_energy = max(0.0, self.residual_energy - actual_consumed)

        if self.residual_energy < 5.1e-5:
            self.residual_energy = 0.0
            self.is_alive = False
            self.death_round = current_round
            return False
        return True

    def reset(self):
        self.residual_energy = self.initial_energy
        self.is_alive = True
        self.packets_generated = 0
        self.packets_forwarded = 0
        self.packets_delivered = 0
        self.energy_consumed = 0.0
        self.queue_load = 0
        self.death_round = -1

class IoTNetwork:
    """Manages sensor topology, Base Station, distance calculations, and NetworkX graphs."""

    def __init__(self, config: Optional[SimulationConfig] = None, seed: Optional[int] = None):
        self.config = config or SimulationConfig()
        self.seed = seed if seed is not None else self.config.random_seed
        self.nodes: Dict[int, Node] = {}
        self.bs_id = 0
        self._distance_cache: Dict[Tuple[int, int], float] = {}
        self._generate_topology(self.seed)

    def _generate_topology(self, seed: int):
        rng = np.random.RandomState(seed)

        bs_x, bs_y = self.config.base_station_pos
        self.nodes[self.bs_id] = Node(
            node_id=self.bs_id,
            x=bs_x,
            y=bs_y,
            initial_energy=self.config.initial_energy,
            is_base_station=True
        )

        xs = rng.uniform(5.0, self.config.area_width - 5.0, size=self.config.num_nodes)
        ys = rng.uniform(5.0, self.config.area_height - 5.0, size=self.config.num_nodes)

        for i in range(self.config.num_nodes):
            node_id = i + 1
            self.nodes[node_id] = Node(
                node_id=node_id,
                x=float(xs[i]),
                y=float(ys[i]),
                initial_energy=self.config.initial_energy,
                is_base_station=False
            )

        self._precompute_distances()

    def _precompute_distances(self):
        self._distance_cache.clear()
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            u_id = node_ids[i]
            u = self.nodes[u_id]
            for j in range(i, len(node_ids)):
                v_id = node_ids[j]
                v = self.nodes[v_id]
                d = float(math.hypot(u.x - v.x, u.y - v.y))
                self._distance_cache[(u_id, v_id)] = d
                self._distance_cache[(v_id, u_id)] = d

    def get_distance(self, u_id: int, v_id: int) -> float:
        if (u_id, v_id) in self._distance_cache:
            return self._distance_cache[(u_id, v_id)]
        u = self.nodes[u_id]
        v = self.nodes[v_id]
        d = float(math.hypot(u.x - v.x, u.y - v.y))
        self._distance_cache[(u_id, v_id)] = d
        self._distance_cache[(v_id, u_id)] = d
        return d

    def get_alive_sensor_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if not n.is_base_station and n.is_alive]

    def count_alive_nodes(self) -> int:
        return len(self.get_alive_sensor_nodes())

    def count_dead_nodes(self) -> int:
        return self.config.num_nodes - self.count_alive_nodes()

    def get_total_residual_energy(self) -> float:
        return float(sum(n.residual_energy for n in self.nodes.values() if not n.is_base_station))

    def get_total_energy_consumed(self) -> float:
        return float(sum(n.energy_consumed for n in self.nodes.values() if not n.is_base_station))

    def get_energy_variance(self) -> float:
        alives = [n.residual_energy for n in self.get_alive_sensor_nodes()]
        if len(alives) <= 1:
            return 0.0
        return float(np.var(alives))

    def build_graph(self) -> nx.DiGraph:
        """Construct directed graph of all reachable links between alive nodes."""
        G = nx.DiGraph()

        for node_id, node in self.nodes.items():
            if node.is_alive:
                G.add_node(node_id, pos=(node.x, node.y), is_bs=node.is_base_station)

        alive_ids = list(G.nodes())
        comm_range = self.config.comm_range
        for i in range(len(alive_ids)):
            u_id = alive_ids[i]
            if u_id == self.bs_id:

                continue
            for j in range(len(alive_ids)):
                if i == j:
                    continue
                v_id = alive_ids[j]
                d = self.get_distance(u_id, v_id)
                if d <= comm_range:
                    G.add_edge(u_id, v_id, distance=d)

        return G

    def clone(self) -> "IoTNetwork":
        return copy.deepcopy(self)

    def reset(self):
        for node in self.nodes.values():
            node.reset()
