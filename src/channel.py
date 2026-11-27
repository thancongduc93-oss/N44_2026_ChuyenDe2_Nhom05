import math
import hashlib
import struct
from typing import Optional, List
import numpy as np
from src.config import SimulationConfig


def make_channel_rng(base_seed: int, round_num: int, u_id: int, v_id: int) -> np.random.RandomState:
    """Create a deterministic keyed RNG for link (u_id, v_id) at (base_seed, round_num).

    Because all sensor nodes have homogeneous radio parameters (identical TX power,
    antenna gain, and receiver noise floor) and the propagation medium is reciprocal,
    the link pair is canonicalized as (min(u_id, v_id), max(u_id, v_id)) so that
    shadowing on u->v and v->u is symmetric in the same round.
    """
    a, b = (int(u_id), int(v_id)) if u_id <= v_id else (int(v_id), int(u_id))
    h = hashlib.md5()
    h.update(struct.pack('>IIII', int(base_seed), int(round_num), a, b))
    derived_seed = int.from_bytes(h.digest()[:4], 'big') % (2**31)
    return np.random.RandomState(derived_seed)


class WirelessChannel:
    """Wireless channel model implementing log-distance path loss with log-normal
    shadowing and IEEE 802.15.4 2.4 GHz O-QPSK physical layer error rate model
    with sequential frame fragmentation and early-stop ARQ.

    Theoretical References:
    - IEEE Std 802.15.4-2015 (2.4 GHz O-QPSK PHY: 250 kbps bit rate, 62.5 ksymbols/s,
      2.0 Mchip/s chip rate, 32 chips per 4-bit symbol, max 127-octet PSDU).
    - Non-coherent 16-ary quasi-orthogonal BER approximation (factor 20 in exponent)
      from IEEE Std 802.15.4-2015, Annex E (Eq. E.2) / Wang et al., ACM SenSys 2006.
      - In-band SNR: SNR_linear = 10^((RSSI - P_N) / 10)
      - Chip SNR: SNR_chip = SNR_linear * (B_n / R_c)
      - Bit SNR: Eb/N0 = SNR_chip * (R_c / R_b) = 8 * SNR_chip
      - Symbol SNR: Es/N0 = 4 * Eb/N0 = 32 * SNR_chip
      - Quasi-orthogonal approximation:
          P_s = (1/16) * sum_{k=2}^{16} (-1)^k * C(16,k) * exp(20 * SNR_chip * (1/k - 1))
          P_b = (8/15) * P_s

    Sequential Fragmentation Model (with early stopping on fragment failure):
    - An application payload of B_app bits is segmented into F fragments with sizes
      L_1, ..., L_F (where L_F may be smaller if B_app is not a multiple of max payload).
    - For each fragment j with single-attempt frame PDR p_j = (1 - P_b)^(L_j) and
      up to A = 1 + max_retx attempts:
        * Fragment success probability: q_j = 1 - (1 - p_j)^A
        * Probability fragment j is reached: R_j = prod_{m=1}^{j-1} q_m  (R_1 = 1)
        * Unconditional expected attempts on fragment j (given reached):
            E[N_j] = (1 - (1 - p_j)^A) / p_j   (or A if p_j <= 1e-6)
        * Conditional expected attempts on fragment j GIVEN fragment j succeeds:
            E[N_j | succ] = [1 - (1 - p_j)^A * (1 + A * p_j)] / (p_j * q_j)
        * Unconditional expected airtime (all generated packets, stops on fail):
            E[D_uncond] = sum_{j=1}^F R_j * E[N_j] * D_single(d, L_j, Q)
        * Conditional mean delay GIVEN full application packet succeeds:
            E[D | app_succ] = sum_{j=1}^F E[N_j | succ] * D_single(d, L_j, Q)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.eta = self.config.path_loss_exponent
        self.sigma = self.config.shadowing_std
        self.tx_power = self.config.tx_power_dbm
        self.noise_floor = self.config.noise_floor_dbm
        self.d_ref = self.config.d_ref
        self.pl_d_ref = self.config.pl_d_ref
        self.bit_rate = self.config.bandwidth_bps
        self.chip_rate = self.config.chip_rate_cps
        self.noise_bandwidth = self.config.receiver_noise_bandwidth_hz
        self.packet_bits = self.config.packet_size_bits
        self.frame_bits = self.config.frame_size_bits
        self.num_fragments = self.config.num_fragments
        self.fragment_sizes = self.config.fragment_sizes_bits
        self.max_retx = self.config.max_retransmissions
        self.max_attempts = 1 + self.max_retx

    def compute_path_loss(
        self,
        distance: float,
        apply_shadowing: bool = True,
        rng: Optional[np.random.RandomState] = None
    ) -> float:
        d = max(float(distance), self.d_ref)
        pl = self.pl_d_ref + 10.0 * self.eta * math.log10(d / self.d_ref)
        if apply_shadowing:
            shadow = float(rng.normal(0, self.sigma)) if rng is not None else 0.0
            pl += shadow
        return float(pl)

    def compute_rssi(
        self,
        distance: float,
        apply_shadowing: bool = True,
        rng: Optional[np.random.RandomState] = None
    ) -> float:
        pl = self.compute_path_loss(distance, apply_shadowing=apply_shadowing, rng=rng)
        return float(self.tx_power - pl)

    def compute_snr(
        self,
        distance: float,
        apply_shadowing: bool = True,
        rng: Optional[np.random.RandomState] = None
    ) -> float:
        rssi = self.compute_rssi(distance, apply_shadowing=apply_shadowing, rng=rng)
        return float(rssi - self.noise_floor)

    def snr_to_chip_snr(self, snr_linear: float) -> float:
        scale = self.noise_bandwidth / self.chip_rate
        return float(max(0.0, snr_linear * scale))

    def snr_to_eb_n0(self, snr_linear: float) -> float:
        snr_chip = self.snr_to_chip_snr(snr_linear)
        return float(snr_chip * (self.chip_rate / self.bit_rate))

    def snr_to_es_n0(self, snr_linear: float) -> float:
        return float(4.0 * self.snr_to_eb_n0(snr_linear))

    @staticmethod
    def ber_ieee_802_15_4_oqpsk(snr_chip: float) -> float:
        if snr_chip <= 0.0:
            return 0.5

        ps_sum = 0.0
        for k in range(2, 17):
            binom_16_k = math.comb(16, k)
            sign = (-1.0) ** k
            exponent = 20.0 * snr_chip * ((1.0 / k) - 1.0)
            if exponent < -700.0:
                continue
            ps_sum += sign * binom_16_k * math.exp(exponent)

        ps = (1.0 / 16.0) * ps_sum
        ps = max(0.0, min(15.0 / 16.0, ps))
        pb = (8.0 / 15.0) * ps
        return float(np.clip(pb, 0.0, 0.5))

    @staticmethod
    def ber_ieee_802_15_4_oqpsk_vec(snr_chip_arr: np.ndarray) -> np.ndarray:
        """Vectorized IEEE 802.15.4 O-QPSK BER computation for fast batch evaluation."""
        snr = np.asarray(snr_chip_arr, dtype=np.float64)
        pb = np.full(snr.shape, 0.5, dtype=np.float64)
        pos_mask = snr > 0.0
        if not np.any(pos_mask):
            return pb

        s = snr[pos_mask]
        ps_sum = np.zeros_like(s)
        for k in range(2, 17):
            binom_16_k = float(math.comb(16, k))
            sign = (-1.0) ** k
            exponent = np.maximum(20.0 * s * ((1.0 / k) - 1.0), -700.0)
            ps_sum += sign * binom_16_k * np.exp(exponent)

        ps = np.clip((1.0 / 16.0) * ps_sum, 0.0, 15.0 / 16.0)
        pb[pos_mask] = np.clip((8.0 / 15.0) * ps, 0.0, 0.5)
        return pb

    def compute_ber(
        self,
        distance: float,
        apply_shadowing: bool = True,
        rng: Optional[np.random.RandomState] = None,
        rssi: Optional[float] = None
    ) -> float:
        """Compute Bit Error Rate (BER) for given distance or observed RSSI."""
        if distance > self.config.comm_range or distance < 0:
            return 0.5
        snr_db = float(rssi - self.noise_floor) if rssi is not None else self.compute_snr(distance, apply_shadowing=apply_shadowing, rng=rng)
        snr_linear = 10.0 ** (snr_db / 10.0)
        snr_chip = self.snr_to_chip_snr(snr_linear)
        return self.ber_ieee_802_15_4_oqpsk(snr_chip)

    def compute_frame_pdr(
        self,
        distance: float,
        apply_shadowing: bool = True,
        rng: Optional[np.random.RandomState] = None,
        rssi: Optional[float] = None,
        frame_bits: Optional[int] = None
    ) -> float:
        """Compute single-frame Packet Delivery Ratio for a frame of ``frame_bits``."""
        if distance > self.config.comm_range or distance < 0:
            return 0.0
        bits = self.frame_bits if frame_bits is None else int(frame_bits)
        ber = self.compute_ber(distance, apply_shadowing=apply_shadowing, rng=rng, rssi=rssi)
        pdr = (1.0 - ber) ** bits
        return float(np.clip(pdr, 0.0, 1.0))

    def compute_pdr(
        self,
        distance: float,
        apply_shadowing: bool = True,
        rng: Optional[np.random.RandomState] = None,
        rssi: Optional[float] = None
    ) -> float:
        return self.compute_frame_pdr(distance, apply_shadowing=apply_shadowing, rng=rng, rssi=rssi)

    def compute_link_success_prob(self, single_frame_pdr: float, ber: Optional[float] = None) -> float:
        """Application-level hop success probability across all F fragments (accounting for
        shorter last fragment when ``ber`` is provided).
        """
        if ber is not None:
            prob = 1.0
            for l_j in self.fragment_sizes:
                p_j = float(np.clip((1.0 - ber) ** l_j, 0.0, 1.0))
                q_j = 1.0 - (1.0 - p_j) ** self.max_attempts
                prob *= q_j
            return float(prob)
        p = float(np.clip(single_frame_pdr, 0.0, 1.0))
        q = 1.0 - (1.0 - p) ** self.max_attempts
        return float(q ** self.num_fragments)

    def expected_transmissions(self, single_pdr: float) -> float:
        """Expected number of attempts for ONE fragment truncated at A attempts:
        E[N_frame] = (1 - (1 - p)^A) / p  if p > 1e-6 else A
        """
        p = float(np.clip(single_pdr, 0.0, 1.0))
        if p < 1e-6:
            return float(self.max_attempts)
        return float((1.0 - (1.0 - p) ** self.max_attempts) / p)

    def expected_transmissions_conditional_success(self, single_pdr: float) -> float:
        """Expected number of attempts for ONE fragment CONDITIONED on that fragment succeeding:
        E[N_frame | succ] = [1 - (1 - p)^A * (1 + A * p)] / (p * q)
        """
        p = float(np.clip(single_pdr, 0.0, 1.0))
        if p < 1e-6:
            return float((self.max_attempts + 1) / 2.0)
        q = 1.0 - (1.0 - p) ** self.max_attempts
        if q < 1e-12:
            return float(self.max_attempts)
        num = 1.0 - ((1.0 - p) ** self.max_attempts) * (1.0 + self.max_attempts * p)
        return float(num / (p * q))

    def expected_app_fragment_attempts_per_packet(
        self,
        single_frame_pdr: float,
        ber: Optional[float] = None
    ) -> float:
        """Expected total fragment transmission attempts per generated application packet
        under sequential transmission with early stop on fragment failure:
            E[N_app] = sum_{j=1}^F R_j * E[N_j], where R_j = prod_{m=1}^{j-1} q_m.
        """
        total = 0.0
        r_j = 1.0
        for l_j in self.fragment_sizes:
            if ber is not None:
                p_j = float(np.clip((1.0 - ber) ** l_j, 0.0, 1.0))
            else:
                p_j = float(np.clip(single_frame_pdr, 0.0, 1.0))
            q_j = 1.0 - (1.0 - p_j) ** self.max_attempts
            total += r_j * self.expected_transmissions(p_j)
            r_j *= q_j
        return float(total)

    def compute_single_hop_delay(
        self,
        distance: float,
        queue_load: int = 0,
        frame_bits: Optional[int] = None
    ) -> float:
        """Single-frame transmission latency in ms: T_tx_frame + T_prop + T_proc + T_queue."""
        bits = self.frame_bits if frame_bits is None else int(frame_bits)
        t_tx_ms = (bits / self.bit_rate) * 1000.0
        t_prop_ms = (max(0.0, float(distance)) / self.config.propagation_speed) * 1000.0
        t_proc_ms = self.config.proc_delay_base_ms
        t_queue_ms = queue_load * (t_tx_ms * 0.4)
        return float(t_tx_ms + t_prop_ms + t_proc_ms + t_queue_ms)

    def compute_unconditional_link_airtime(
        self,
        distance: float,
        pdr: Optional[float] = None,
        queue_load: int = 0,
        ber: Optional[float] = None
    ) -> float:
        """Expected airtime (ms) per generated application packet (stops upon fragment failure):
            E[D_uncond] = sum_{j=1}^F R_j * E[N_j] * D_single(d, L_j, Q)
        """
        if pdr is None and ber is None:
            pdr = self.compute_frame_pdr(distance, apply_shadowing=False)
        total_ms = 0.0
        r_j = 1.0
        for l_j in self.fragment_sizes:
            p_j = float(np.clip((1.0 - ber) ** l_j, 0.0, 1.0)) if ber is not None else float(np.clip(pdr, 0.0, 1.0))
            q_j = 1.0 - (1.0 - p_j) ** self.max_attempts
            d_single = self.compute_single_hop_delay(distance, queue_load=queue_load, frame_bits=l_j)
            total_ms += r_j * self.expected_transmissions(p_j) * d_single
            r_j *= q_j
        return float(total_ms)

    def compute_conditional_link_delay(
        self,
        distance: float,
        pdr: Optional[float] = None,
        queue_load: int = 0,
        ber: Optional[float] = None
    ) -> float:
        """Mean link delay (ms) CONDITIONED on successful application packet delivery (all F fragments succeed):
            E[D | app_succ] = sum_{j=1}^F E[N_j | succ] * D_single(d, L_j, Q)
        """
        if pdr is None and ber is None:
            pdr = self.compute_frame_pdr(distance, apply_shadowing=False)
        total_ms = 0.0
        for l_j in self.fragment_sizes:
            p_j = float(np.clip((1.0 - ber) ** l_j, 0.0, 1.0)) if ber is not None else float(np.clip(pdr, 0.0, 1.0))
            d_single = self.compute_single_hop_delay(distance, queue_load=queue_load, frame_bits=l_j)
            total_ms += self.expected_transmissions_conditional_success(p_j) * d_single
        return float(total_ms)

    def compute_link_delay(
        self,
        distance: float,
        pdr: Optional[float] = None,
        queue_load: int = 0,
        ber: Optional[float] = None
    ) -> float:
        """Expected unconditional airtime per generated application packet (alias for
        ``compute_unconditional_link_airtime`` for backward compatibility).
        """
        return self.compute_unconditional_link_airtime(distance, pdr=pdr, queue_load=queue_load, ber=ber)

    def compute_frame_pdr_batch(self, distances: np.ndarray, rssis: np.ndarray) -> np.ndarray:
        """Vectorized batch frame PDR computation for fair comparison with ML."""
        snr_db = rssis - self.noise_floor
        snr_lin = np.power(10.0, snr_db / 10.0)
        snr_chip = np.maximum(0.0, snr_lin * (self.noise_bandwidth / self.chip_rate))
        bers = self.ber_ieee_802_15_4_oqpsk_vec(snr_chip)
        pdrs = np.power(1.0 - bers, self.frame_bits)
        pdrs = np.where((distances > self.config.comm_range) | (distances < 0), 0.0, pdrs)
        return np.clip(pdrs, 0.0, 1.0)
