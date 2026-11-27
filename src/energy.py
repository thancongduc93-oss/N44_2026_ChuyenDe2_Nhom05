import math
from typing import Optional
from src.config import SimulationConfig

class EnergyModel:
    """Heinzelman First-Order Radio Model (LEACH) for energy dissipation in WSN."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.e_elec = self.config.e_elec
        self.epsilon_fs = self.config.epsilon_fs
        self.epsilon_mp = self.config.epsilon_mp
        self.e_da = self.config.e_da


        self.d0 = math.sqrt(self.epsilon_fs / self.epsilon_mp)

    def tx_energy(self, distance: float, num_bits: Optional[int] = None) -> float:
        if distance < 0:
            raise ValueError(f"Distance cannot be negative, got {distance}")
        if num_bits is None:
            num_bits = self.config.packet_size_bits

        d = float(distance)
        if d < self.d0:
            return float(num_bits * self.e_elec + num_bits * self.epsilon_fs * (d ** 2))
        else:
            return float(num_bits * self.e_elec + num_bits * self.epsilon_mp * (d ** 4))

    def rx_energy(self, num_bits: Optional[int] = None) -> float:
        if num_bits is None:
            num_bits = self.config.packet_size_bits
        return float(num_bits * self.e_elec)

    def frame_tx_energy(self, distance: float, frag_idx: int = 0) -> float:
        """Transmission energy for fragment ``frag_idx`` (uses actual fragment size)."""
        return self.tx_energy(distance, num_bits=self.config.fragment_frame_bits(frag_idx))

    def frame_rx_energy(self, frag_idx: int = 0) -> float:
        """Reception energy for fragment ``frag_idx`` (uses actual fragment size)."""
        return self.rx_energy(num_bits=self.config.fragment_frame_bits(frag_idx))

    def aggregation_energy(self, num_bits: Optional[int] = None) -> float:
        if num_bits is None:
            num_bits = self.config.packet_size_bits
        return float(num_bits * self.e_da)

    def control_tx_energy(self, distance: float) -> float:
        return self.tx_energy(distance, num_bits=self.config.control_packet_bits)

    def control_rx_energy(self) -> float:
        return self.rx_energy(num_bits=self.config.control_packet_bits)

    def max_possible_tx_energy(self) -> float:
        """Single-attempt TX energy across all F fragments at comm_range."""
        return float(sum(
            self.tx_energy(self.config.comm_range, num_bits=l_j)
            for l_j in self.config.fragment_sizes_bits
        ))

    def expected_link_energy(
        self,
        distance: float,
        expected_attempts: float = 1.0,
        is_rx_base_station: bool = False,
        single_frame_pdr: Optional[float] = None,
        ber: Optional[float] = None
    ) -> float:
        """Expected total energy dissipated per generated application packet on link (u, v).

        When ``single_frame_pdr`` or ``ber`` is provided, uses the exact sequential
        early-stop fragmentation formula:
            TX = sum_{j=1}^F R_j * E[N_j] * E_tx(d, L_j)
            RX = sum_{j=1}^F R_j * E[N_j] * E_rx(L_j)   (0 if BS)
            DA = P_app_succ * E_da(B_app)               (0 if BS; only when all F succeed)
        When neither is provided, uses ``expected_attempts`` identically per fragment
        (e.g. for upper-bound normalization with ``expected_attempts=1.0``).
        """
        max_attempts = 1 + self.config.max_retransmissions
        frag_sizes = self.config.fragment_sizes_bits

        if single_frame_pdr is None and ber is None:
            tx_e = sum(expected_attempts * self.tx_energy(distance, num_bits=l_j) for l_j in frag_sizes)
            if is_rx_base_station:
                return float(tx_e)
            rx_e = sum(expected_attempts * self.rx_energy(num_bits=l_j) for l_j in frag_sizes)
            return float(tx_e + rx_e + self.aggregation_energy())

        tx_e = 0.0
        rx_e = 0.0
        r_j = 1.0
        for l_j in frag_sizes:
            p_j = float(max(0.0, min(1.0, (1.0 - ber) ** l_j))) if ber is not None else float(max(0.0, min(1.0, single_frame_pdr)))
            q_j = 1.0 - (1.0 - p_j) ** max_attempts
            e_n_j = float(max_attempts) if p_j < 1e-6 else (q_j / p_j)
            tx_e += r_j * e_n_j * self.tx_energy(distance, num_bits=l_j)
            if not is_rx_base_station:
                rx_e += r_j * e_n_j * self.rx_energy(num_bits=l_j)
            r_j *= q_j

        p_app_succ = r_j
        da_e = 0.0 if is_rx_base_station else (p_app_succ * self.aggregation_energy())
        return float(tx_e + rx_e + da_e)
