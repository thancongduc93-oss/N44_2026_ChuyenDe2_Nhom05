import math
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class SimulationConfig:
    """Central configuration for IoT WSN routing simulation and optimization."""

    area_width: float = 100.0
    area_height: float = 100.0
    num_nodes: int = 50
    base_station_pos: Tuple[float, float] = (50.0, 50.0)
    comm_range: float = 35.0
    random_seed: int = 42


    initial_energy: float = 0.25
    e_elec: float = 50e-9
    epsilon_fs: float = 10e-12
    epsilon_mp: float = 0.0013e-12
    e_da: float = 5e-9


    packet_size_bits: int = 4000
    max_frame_payload_bits: int = 800
    mac_header_fcs_bits: int = 216
    control_packet_bits: int = 200


    frequency_hz: float = 2.4e9
    path_loss_exponent: float = 3.3
    shadowing_std: float = 6.0
    tx_power_dbm: float = 0.0
    noise_floor_dbm: float = -85.0
    d_ref: float = 1.0
    pl_d_ref: float = 40.0


    bandwidth_bps: float = 250000.0
    chip_rate_cps: float = 2000000.0
    receiver_noise_bandwidth_hz: float = 2000000.0
    proc_delay_base_ms: float = 1.0
    propagation_speed: float = 3e8
    max_retransmissions: int = 3


    total_rounds: int = 300
    packets_per_round: int = 1
    queue_service_capacity: int = 8
    max_queue_load: int = 8
    benchmark_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])


    pso_num_particles: int = 25
    pso_max_iter: int = 35
    pso_w_max: float = 0.9
    pso_w_min: float = 0.4
    pso_c1: float = 1.5
    pso_c2: float = 1.5
    pso_adaptive: bool = True
    pso_update_interval: int = 20


    lambda_energy: float = 0.35
    lambda_variance: float = 0.30
    lambda_delay: float = 0.15
    lambda_loss: float = 0.20


    ml_n_estimators: int = 100
    ml_max_depth: int = 12
    ml_dataset_samples: int = 6000
    ml_model_path: str = "models/link_quality_predictor.joblib"
    dataset_path: str = "data/network_data.csv"
    dataset_metadata_path: str = "data/network_data_metadata.json"
    ood_dataset_path: str = "data/ood_test_data.csv"
    ood_metadata_path: str = "data/ood_test_metadata.json"

    @property
    def num_fragments(self) -> int:
        """Number of IEEE 802.15.4 frames required to carry packet_size_bits."""
        return max(1, int(math.ceil(self.packet_size_bits / self.max_frame_payload_bits)))

    def fragment_payload_bits(self, frag_idx: int) -> int:
        """Payload bits in fragment ``frag_idx`` (0-indexed); last fragment may be shorter."""
        if frag_idx < 0 or frag_idx >= self.num_fragments:
            raise IndexError(f"Fragment index {frag_idx} out of range [0, {self.num_fragments})")
        if frag_idx < self.num_fragments - 1:
            return int(self.max_frame_payload_bits)
        rem = self.packet_size_bits - (self.num_fragments - 1) * self.max_frame_payload_bits
        return int(rem if rem > 0 else self.max_frame_payload_bits)

    def fragment_frame_bits(self, frag_idx: int) -> int:
        """Total PSDU bits for fragment ``frag_idx`` (<= 1016 bits / 127 octets)."""
        return int(self.fragment_payload_bits(frag_idx) + self.mac_header_fcs_bits)

    @property
    def fragment_sizes_bits(self) -> List[int]:
        """List of total frame sizes (bits) for all fragments 0..F-1."""
        return [self.fragment_frame_bits(j) for j in range(self.num_fragments)]

    @property
    def frame_size_bits(self) -> int:
        """Maximum PHY/MAC frame size (bits) for full-payload fragments (<= 1016 bits)."""
        return self.fragment_frame_bits(0)
