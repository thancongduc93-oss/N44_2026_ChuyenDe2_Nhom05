import os
import json
import math
import datetime
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.config import SimulationConfig
from src.energy import EnergyModel
from src.channel import WirelessChannel, make_channel_rng
from src.network import IoTNetwork


FEATURE_COLS = [
    "distance",
    "prev_rssi_dbm",
    "prev_snr_db",
    "ewma_pdr",
    "ewma_retx",
    "ewma_delay_ms",
    "rx_queue_load",
]

TARGET_COLS = ["pdr", "link_delay_ms"]

GROUP_COL = "group_seed"

DATASET_VERSION = "4.0.0-temporal-sim"


def _simulate_temporal_link_traces(
    config: SimulationConfig,
    num_samples: int,
    seed: int,
    ewma_alpha: float = 0.35,
) -> pd.DataFrame:
    """Generate causal temporal link-quality traces from independent network topologies.

    For each independent topology seed (ensuring group separation between train and test),
    sensor pairs within ``comm_range`` are tracked across consecutive rounds.
    Features at round t ONLY use observations from rounds <= t - 1 (plus known distance
    and current rx_queue_load before transmission). Targets (``pdr``, ``link_delay_ms``)
    are measured from the actual channel & packet transmission outcome at round t.
    """
    channel = WirelessChannel(config)
    rng = np.random.RandomState(seed)


    samples_per_topo = max(200, min(600, num_samples // 6))
    num_topos = max(4, int(math.ceil(num_samples / samples_per_topo)))

    records: List[Dict[str, Any]] = []
    frag_sizes = config.fragment_sizes_bits
    max_attempts = 1 + config.max_retransmissions

    for topo_idx in range(num_topos):
        if len(records) >= num_samples:
            break
        topo_seed = int(seed * 100 + topo_idx + 1)
        net = IoTNetwork(config, seed=topo_seed)


        valid_edges: List[Tuple[int, int, float]] = []
        node_ids = sorted(net.nodes.keys())
        for u_id in node_ids:
            if u_id == net.bs_id:
                continue
            for v_id in node_ids:
                if u_id == v_id:
                    continue
                d = net.get_distance(u_id, v_id)
                if 1.0 <= d <= config.comm_range:
                    valid_edges.append((u_id, v_id, d))

        if not valid_edges:
            continue


        topo_rng = np.random.RandomState(topo_seed + 9999)
        if len(valid_edges) > 120:
            chosen_idx = topo_rng.choice(len(valid_edges), size=120, replace=False)
            valid_edges = [valid_edges[i] for i in chosen_idx]


        rho = 0.75
        ewma_state: Dict[Tuple[int, int], Dict[str, float]] = {}
        for u_id, v_id, d in valid_edges:
            r0_rng = make_channel_rng(topo_seed, 0, u_id, v_id)
            shadow0 = float(r0_rng.normal(0, channel.sigma))
            rssi0 = float(channel.compute_rssi(d, apply_shadowing=False) - shadow0)
            snr0 = float(rssi0 - channel.noise_floor)
            ber0 = channel.compute_ber(d, apply_shadowing=False, rssi=rssi0)
            pdr0 = float(np.clip((1.0 - ber0) ** config.frame_size_bits, 0.0, 1.0))
            retx0 = max(0.0, channel.expected_transmissions(pdr0) - 1.0)
            q0 = int(topo_rng.randint(0, config.max_queue_load + 1))
            delay0 = channel.compute_unconditional_link_airtime(d, queue_load=q0, ber=ber0)
            ewma_state[(u_id, v_id)] = {
                "prev_shadow": shadow0,
                "prev_rssi_dbm": float(rssi0),
                "prev_snr_db": float(snr0),
                "ewma_pdr": float(pdr0),
                "ewma_retx": float(retx0),
                "ewma_delay_ms": float(delay0),
            }

        rounds_needed = max(3, int(math.ceil(samples_per_topo / max(1, len(valid_edges)))) + 1)
        for round_num in range(1, rounds_needed + 1):
            if len(records) >= num_samples:
                break
            for u_id, v_id, d in valid_edges:
                if len(records) >= num_samples:
                    break
                st = ewma_state[(u_id, v_id)]

                q_load = int(topo_rng.randint(0, config.max_queue_load + 1))


                feat_row = {
                    "distance": round(float(d), 6),
                    "prev_rssi_dbm": round(st["prev_rssi_dbm"], 6),
                    "prev_snr_db": round(st["prev_snr_db"], 6),
                    "ewma_pdr": round(st["ewma_pdr"], 6),
                    "ewma_retx": round(st["ewma_retx"], 6),
                    "ewma_delay_ms": round(st["ewma_delay_ms"], 6),
                    "rx_queue_load": q_load,
                }


                r_rng = make_channel_rng(topo_seed, round_num, u_id, v_id)
                eps_t = float(r_rng.normal(0, channel.sigma))
                shadow_t = rho * st["prev_shadow"] + math.sqrt(1.0 - rho * rho) * eps_t
                st["prev_shadow"] = shadow_t
                rssi_t = float(channel.compute_rssi(d, apply_shadowing=False) - shadow_t)
                snr_t = float(rssi_t - channel.noise_floor)
                ber_t = channel.compute_ber(d, apply_shadowing=False, rssi=rssi_t)
                pdr_t = float(np.clip((1.0 - ber_t) ** config.frame_size_bits, 0.0, 1.0))
                delay_t = channel.compute_unconditional_link_airtime(d, queue_load=q_load, ber=ber_t)


                retx_t = max(0.0, channel.expected_transmissions(pdr_t) - 1.0)

                feat_row["pdr"] = float(pdr_t)
                feat_row["link_delay_ms"] = float(delay_t)
                feat_row[GROUP_COL] = topo_seed
                records.append(feat_row)


                st["prev_rssi_dbm"] = float(rssi_t)
                st["prev_snr_db"] = float(snr_t)
                st["ewma_pdr"] = float((1.0 - ewma_alpha) * st["ewma_pdr"] + ewma_alpha * pdr_t)
                st["ewma_retx"] = float((1.0 - ewma_alpha) * st["ewma_retx"] + ewma_alpha * retx_t)
                st["ewma_delay_ms"] = float((1.0 - ewma_alpha) * st["ewma_delay_ms"] + ewma_alpha * delay_t)

    df = pd.DataFrame(records[:num_samples])
    return df


def generate_link_dataset(
    config: Optional[SimulationConfig] = None,
    num_samples: int = 6000,
    save_path: str = "data/network_data.csv",
    metadata_path: str = "data/network_data_metadata.json",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate causal temporal link-quality dataset from independent WSN simulations.

    Features strictly use past link observations (``prev_rssi_dbm``, ``prev_snr_db``,
    ``ewma_pdr``, ``ewma_retx``, ``ewma_delay_ms``), static link geometry (``distance``),
    and pre-transmission receiver queue state (``rx_queue_load``). Targets (``pdr``,
    ``link_delay_ms``) are measured from the subsequent round's physical channel outcome.
    """
    config = config or SimulationConfig()
    actual_seed = seed if seed is not None else config.random_seed

    df = _simulate_temporal_link_traces(config, num_samples=num_samples, seed=actual_seed)

    assert not df.isnull().values.any(), "Generated dataset contains NaN values!"
    assert not np.isinf(df.values).any(), "Generated dataset contains infinite values!"
    assert (df["pdr"] >= 0.0).all() and (df["pdr"] <= 1.0).all(), "PDR out of [0, 1] range!"
    assert (df["link_delay_ms"] > 0.0).all(), "Link delay must be strictly positive!"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)

    metadata: Dict[str, Any] = {
        "dataset_type": "causal_temporal_simulation_traces",
        "description": (
            "Causal multi-topology IEEE 802.15.4 link quality traces: past-window features "
            "predicting next-round link PDR and expected airtime, grouped by independent topology seed."
        ),
        "version": DATASET_VERSION,
        "generation_timestamp": datetime.datetime.now().isoformat(),
        "random_seed": actual_seed,
        "num_samples": len(df),
        "num_unique_topologies": int(df[GROUP_COL].nunique()),
        "feature_columns": FEATURE_COLS,
        "target_columns": TARGET_COLS,
        "group_column": GROUP_COL,
        "config_parameters": {
            "comm_range": config.comm_range,
            "path_loss_exponent": config.path_loss_exponent,
            "shadowing_std": config.shadowing_std,
            "tx_power_dbm": config.tx_power_dbm,
            "noise_floor_dbm": config.noise_floor_dbm,
            "bandwidth_bps": config.bandwidth_bps,
            "packet_size_bits": config.packet_size_bits,
            "frame_size_bits": config.frame_size_bits,
            "num_fragments": config.num_fragments,
            "max_retransmissions": config.max_retransmissions,
            "max_queue_load": config.max_queue_load,
        },
    }

    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(
        f"[Dataset] Generated {len(df)} temporal trace samples across "
        f"{df[GROUP_COL].nunique()} topologies (seed={actual_seed}) saved to {save_path}"
    )
    print(f"[Dataset] Metadata saved to {metadata_path}")
    return df


def is_dataset_valid(
    csv_path: str = "data/network_data.csv",
    metadata_path: str = "data/network_data_metadata.json",
    expected_seed: Optional[int] = None,
    expected_samples: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
) -> bool:
    if not os.path.exists(csv_path) or not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != DATASET_VERSION:
            return False
        if meta.get("feature_columns") != FEATURE_COLS or meta.get("target_columns") != TARGET_COLS:
            return False
        if expected_seed is not None and meta.get("random_seed") != expected_seed:
            return False
        if expected_samples is not None and meta.get("num_samples") != expected_samples:
            return False
        if config is not None:
            cfg_meta = meta.get("config_parameters", {})
            for key, val in [
                ("comm_range", config.comm_range),
                ("path_loss_exponent", config.path_loss_exponent),
                ("shadowing_std", config.shadowing_std),
                ("tx_power_dbm", config.tx_power_dbm),
                ("noise_floor_dbm", config.noise_floor_dbm),
            ]:
                if not math.isclose(float(cfg_meta.get(key, 0.0)), float(val), rel_tol=1e-4):
                    return False
            if int(cfg_meta.get("packet_size_bits", 0)) != int(config.packet_size_bits):
                return False
            if int(cfg_meta.get("frame_size_bits", 0)) != int(config.frame_size_bits):
                return False
            if int(cfg_meta.get("max_retransmissions", 0)) != int(config.max_retransmissions):
                return False
            if "max_queue_load" in cfg_meta and int(cfg_meta["max_queue_load"]) != int(config.max_queue_load):
                return False

        df = pd.read_csv(csv_path)
        expected_cols = FEATURE_COLS + TARGET_COLS + [GROUP_COL]
        if list(df.columns) != expected_cols:
            return False
        if len(df) != int(meta.get("num_samples", -1)):
            return False
        if expected_samples is not None and len(df) != expected_samples:
            return False
        if df.isnull().values.any() or np.isinf(df[FEATURE_COLS + TARGET_COLS].values).any():
            return False
        if (df["distance"] <= 0).any() or (df["ewma_pdr"] < 0.0).any() or (df["ewma_pdr"] > 1.0).any():
            return False
        if (df["ewma_retx"] < 0.0).any() or (df["ewma_delay_ms"] <= 0.0).any():
            return False
        if (df["rx_queue_load"] < 0).any() or (config is not None and (df["rx_queue_load"] > config.max_queue_load).any()):
            return False
        if (df["pdr"] < 0.0).any() or (df["pdr"] > 1.0).any() or (df["link_delay_ms"] <= 0.0).any():
            return False
        return True
    except Exception:
        return False


if __name__ == "__main__":
    generate_link_dataset()
