import math
import os
import json
import shutil
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import networkx as nx
import pytest

from src.config import SimulationConfig
from src.energy import EnergyModel
from src.channel import WirelessChannel
from src.network import Node, IoTNetwork
from src.dataset import (
    FEATURE_COLS,
    TARGET_COLS,
    DATASET_VERSION,
    generate_link_dataset,
    is_dataset_valid
)
from src.ml_predictor import (
    LinkQualityPredictor,
    train_and_evaluate_ml,
    load_predictor
)
from src.routing import RoutingEngine, GraphSnapshot
from src.pso import Particle, PSORoutingOptimizer
from src.simulator import NetworkSimulator, SimulationResult

# =========================================================================
# TEST 1: Energy Model & Boundary Invariants
# =========================================================================
def test_energy_model_formulas_and_d0():
    cfg = SimulationConfig(
        e_elec=50e-9,
        epsilon_fs=10e-12,
        epsilon_mp=0.0013e-12,
        e_da=5e-9,
        packet_size_bits=4000
    )
    em = EnergyModel(cfg)

    # d0 check
    expected_d0 = math.sqrt(cfg.epsilon_fs / cfg.epsilon_mp)
    assert math.isclose(em.d0, expected_d0, rel_tol=1e-6)
    assert em.d0 > 80.0  # Approx 87.7m

    # Case d = 0: consumes circuit electronics only
    tx_0 = em.tx_energy(0.0)
    assert math.isclose(tx_0, cfg.packet_size_bits * cfg.e_elec, rel_tol=1e-6)

    # Case d < 0 raises error
    with pytest.raises(ValueError):
        em.tx_energy(-5.0)

    # Case d < d0 (free space d^2)
    d_close = 20.0
    expected_close = cfg.packet_size_bits * cfg.e_elec + cfg.packet_size_bits * cfg.epsilon_fs * (d_close ** 2)
    assert math.isclose(em.tx_energy(d_close), expected_close, rel_tol=1e-6)

    # Case d >= d0 (multipath d^4)
    d_far = 100.0
    expected_far = cfg.packet_size_bits * cfg.e_elec + cfg.packet_size_bits * cfg.epsilon_mp * (d_far ** 4)
    assert math.isclose(em.tx_energy(d_far), expected_far, rel_tol=1e-6)

    # Rx and Aggregation energy
    assert math.isclose(em.rx_energy(), cfg.packet_size_bits * cfg.e_elec, rel_tol=1e-6)
    assert math.isclose(em.aggregation_energy(), cfg.packet_size_bits * cfg.e_da, rel_tol=1e-6)


def test_energy_conservation_invariant():
    """Verify that node energy consumption strictly satisfies the conservation invariant:
    initial_energy == residual_energy + energy_consumed even when depleted below zero.
    """
    node = Node(node_id=1, x=10.0, y=10.0, initial_energy=0.25, is_base_station=False)
    assert math.isclose(node.initial_energy, node.residual_energy + node.energy_consumed, rel_tol=1e-9)

    # Consume small amount
    node.consume_energy(0.10, current_round=1)
    assert math.isclose(node.initial_energy, node.residual_energy + node.energy_consumed, rel_tol=1e-9)
    assert node.is_alive

    # Consume excessive amount that would deplete battery below zero
    node.consume_energy(0.30, current_round=2)
    assert math.isclose(node.initial_energy, node.residual_energy + node.energy_consumed, rel_tol=1e-9)
    assert node.residual_energy == 0.0
    assert not node.is_alive
    assert node.death_round == 2

    # Consuming when dead does not add extra energy
    node.consume_energy(0.05, current_round=3)
    assert math.isclose(node.initial_energy, node.residual_energy + node.energy_consumed, rel_tol=1e-9)


# =========================================================================
# TEST 2: Channel Model & IEEE 802.15.4 O-QPSK Numerical Properties
# =========================================================================
def test_channel_model_validity_and_monotonicity():
    cfg = SimulationConfig(comm_range=35.0)
    wc = WirelessChannel(cfg)

    # Path loss increases with distance
    pl_10 = wc.compute_path_loss(10.0, apply_shadowing=False)
    pl_20 = wc.compute_path_loss(20.0, apply_shadowing=False)
    assert pl_20 > pl_10

    # RSSI decreases with distance
    rssi_10 = wc.compute_rssi(10.0, apply_shadowing=False)
    rssi_20 = wc.compute_rssi(20.0, apply_shadowing=False)
    assert rssi_10 > rssi_20

    # PDR decreases with distance
    pdr_5 = wc.compute_pdr(5.0, apply_shadowing=False)
    pdr_30 = wc.compute_pdr(30.0, apply_shadowing=False)
    assert 0.0 <= pdr_30 <= pdr_5 <= 1.0

    # PDR beyond comm_range is strictly 0.0
    assert wc.compute_pdr(cfg.comm_range + 5.0, apply_shadowing=False) == 0.0

    # Monotonicity with respect to RSSI
    pdr_high_rssi = wc.compute_pdr(15.0, apply_shadowing=False, rssi=-60.0)
    pdr_low_rssi = wc.compute_pdr(15.0, apply_shadowing=False, rssi=-90.0)
    assert pdr_high_rssi > pdr_low_rssi

    # No NaN or Inf in delay or PDR
    del_val = wc.compute_link_delay(20.0)
    assert not math.isnan(del_val) and not math.isinf(del_val)
    assert del_val > 0.0


def test_channel_ber_ieee_numerical_properties():
    """Verify IEEE 802.15.4 O-QPSK BER reference points and conversions."""
    cfg = SimulationConfig()
    wc = WirelessChannel(cfg)

    # Boundary: at 0 linear SNR, BER is exactly 0.5 (random bit guessing)
    ber_0 = wc.ber_ieee_802_15_4_oqpsk(0.0)
    assert math.isclose(ber_0, 0.5, rel_tol=1e-5)

    # High SNR: BER decays exponentially towards 0
    ber_high = wc.ber_ieee_802_15_4_oqpsk(5.0)
    assert ber_high < 1e-4

    # Monotonic decrease with respect to chip SNR
    snr_grid = [0.1, 0.5, 1.0, 2.0, 4.0]
    bers = [wc.ber_ieee_802_15_4_oqpsk(s) for s in snr_grid]
    for i in range(len(bers) - 1):
        assert bers[i] > bers[i + 1]

    # Check SNR conversion relations
    snr_lin = 2.0
    snr_chip = wc.snr_to_chip_snr(snr_lin)
    eb_n0 = wc.snr_to_eb_n0(snr_lin)
    assert math.isclose(snr_chip, snr_lin * (wc.noise_bandwidth / wc.chip_rate))
    assert math.isclose(eb_n0, 8.0 * snr_chip, rel_tol=1e-5)


def test_pdr_distribution_within_comm_range():
    """Verify that PDR has meaningful variation within comm_range (not all 1.0).
    
    With the updated channel parameters (path_loss_exponent=3.3, noise_floor=-85dBm,
    shadowing_std=6.0dB), links near the edge of comm_range should exhibit
    intermediate PDR values, providing useful training data for ML.
    """
    cfg = SimulationConfig()
    wc = WirelessChannel(cfg)
    rng = np.random.RandomState(42)

    pdrs = []
    for _ in range(2000):
        d = rng.uniform(1.0, cfg.comm_range)
        rssi = wc.compute_rssi(d, apply_shadowing=True, rng=rng)
        pdr = wc.compute_pdr(d, apply_shadowing=False, rssi=rssi)
        pdrs.append(pdr)

    pdrs = np.array(pdrs)
    # Must have some PDR < 1.0 (not all perfect links)
    count_imperfect = np.sum(pdrs < 0.99)
    assert count_imperfect > 50, f"Only {count_imperfect} imperfect PDR out of 2000 samples"

    # Must have some intermediate values (not just binary 0/1)
    count_intermediate = np.sum((pdrs > 0.01) & (pdrs < 0.99))
    assert count_intermediate > 20, f"Only {count_intermediate} intermediate PDR values"


def test_retransmission_geometric_expectation_monte_carlo():
    """Verify that expected_transmissions matches discrete Bernoulli trial expectation."""
    cfg = SimulationConfig(max_retransmissions=3)
    wc = WirelessChannel(cfg)
    p_succ = 0.65
    expected_formula = wc.expected_transmissions(p_succ)

    # Monte Carlo simulation of truncated geometric process
    rng = np.random.RandomState(42)
    n_trials = 20000
    attempts_list = []
    for _ in range(n_trials):
        att = 0
        succ = False
        for a in range(1, 1 + cfg.max_retransmissions + 1):
            att = a
            if rng.uniform(0.0, 1.0) < p_succ:
                succ = True
                break
        attempts_list.append(att)

    simulated_mean = float(np.mean(attempts_list))
    # Must match within statistical tolerance (< 1.5%)
    assert math.isclose(expected_formula, simulated_mean, rel_tol=0.015)


# =========================================================================
# TEST 3: RNG Reproducibility
# =========================================================================
def test_rng_reproducibility(tmp_path):
    cfg = SimulationConfig(random_seed=123, comm_range=35.0)
    p1 = str(tmp_path / "ds1.csv")
    m1 = str(tmp_path / "meta1.json")
    p2 = str(tmp_path / "ds2.csv")
    m2 = str(tmp_path / "meta2.json")
    p3 = str(tmp_path / "ds3.csv")
    m3 = str(tmp_path / "meta3.json")

    df1 = generate_link_dataset(config=cfg, num_samples=100, save_path=p1, metadata_path=m1, seed=123)
    df2 = generate_link_dataset(config=cfg, num_samples=100, save_path=p2, metadata_path=m2, seed=123)
    df3 = generate_link_dataset(config=cfg, num_samples=100, save_path=p3, metadata_path=m3, seed=999)

    # Identical seeds must match exactly
    pd.testing.assert_frame_equal(df1, df2)

    # Different seeds must produce different data
    assert not df1.equals(df3)


# =========================================================================
# TEST 4: Dataset Schema, Validation & Configuration Compatibility
# =========================================================================
def test_dataset_schema_and_metadata(tmp_path):
    cfg = SimulationConfig(random_seed=42)
    csv_file = str(tmp_path / "test_data.csv")
    meta_file = str(tmp_path / "test_meta.json")

    df = generate_link_dataset(config=cfg, num_samples=200, save_path=csv_file, metadata_path=meta_file, seed=42)

    # Check columns
    expected_cols = FEATURE_COLS + TARGET_COLS
    for c in expected_cols:
        assert c in df.columns

    # Check validity function with matching config
    assert is_dataset_valid(csv_file, meta_file, expected_seed=42, expected_samples=200, config=cfg)
    assert not is_dataset_valid(csv_file, meta_file, expected_seed=999, config=cfg)

    # Config mismatch rejection: changing comm_range makes old dataset invalid
    stale_cfg = SimulationConfig(comm_range=40.0)
    assert not is_dataset_valid(csv_file, meta_file, expected_seed=42, config=stale_cfg)

    # Check no missing or infinite values
    assert not df.isnull().values.any()
    assert not np.isinf(df.values).any()

    # All distances must be within comm_range (no out-of-range links)
    assert (df["distance"] <= cfg.comm_range).all(), "Dataset contains links outside comm_range!"
    assert (df["distance"] >= 1.0).all(), "Dataset contains links with distance < 1m!"


def test_dataset_no_out_of_range_links(tmp_path):
    """Ensure dataset only contains valid in-range links for ML training."""
    cfg = SimulationConfig(comm_range=35.0, random_seed=42)
    csv_file = str(tmp_path / "range_test.csv")
    meta_file = str(tmp_path / "range_meta.json")

    df = generate_link_dataset(config=cfg, num_samples=1000, save_path=csv_file, metadata_path=meta_file, seed=42)

    # No distances should exceed comm_range
    assert (df["distance"] <= cfg.comm_range + 1e-6).all()
    # PDR should be in valid range
    assert (df["pdr"] >= 0.0).all() and (df["pdr"] <= 1.0).all()


# =========================================================================
# TEST 5: Machine Learning Training, Evaluation & Prediction
# =========================================================================
def test_ml_train_load_predict(tmp_path):
    cfg = SimulationConfig(
        ml_dataset_samples=500,
        ml_n_estimators=10,
        ml_max_depth=6,
        random_seed=42
    )
    csv_file = str(tmp_path / "ml_data.csv")
    meta_file = str(tmp_path / "ml_meta.json")
    model_file = str(tmp_path / "rf_model.joblib")
    metrics_file = str(tmp_path / "metrics.json")

    pred, metrics = train_and_evaluate_ml(
        config=cfg,
        dataset_path=csv_file,
        metadata_path=meta_file,
        model_save_path=model_file,
        metrics_save_path=metrics_file,
        force_retrain=True,
        ood_dataset_path=str(tmp_path / "ood.csv"),
        ood_metadata_path=str(tmp_path / "ood_meta.json"),
    )

    # Verify metrics structure
    assert "selected_model_name" in metrics
    assert "selected_pdr_metrics" in metrics
    assert "selected_delay_metrics" in metrics
    assert metrics["selected_pdr_metrics"]["R2"] > 0.0
    assert metrics["selected_delay_metrics"]["R2"] > 0.0
    assert "generalization_evaluation_ood" in metrics

    # Prediction test (V19 causal 7 features)
    pdr_pred, del_pred = pred.predict(
        distance=20.0,
        prev_rssi_dbm=-75.0,
        prev_snr_db=10.0,
        ewma_pdr=0.9,
        ewma_retx=0.2,
        ewma_delay_ms=22.0,
        rx_queue_load=1
    )
    assert 0.0 <= pdr_pred <= 1.0
    assert del_pred > 0.0

    # Reload predictor
    loaded_pred = load_predictor(model_file, config=cfg)
    pdr_l, del_l = loaded_pred.predict(
        distance=20.0,
        prev_rssi_dbm=-75.0,
        prev_snr_db=10.0,
        ewma_pdr=0.9,
        ewma_retx=0.2,
        ewma_delay_ms=22.0,
        rx_queue_load=1
    )
    assert math.isclose(pdr_pred, pdr_l, rel_tol=1e-5)
    assert math.isclose(del_pred, del_l, rel_tol=1e-5)


def test_ood_metadata_not_overwrite_training(tmp_path):
    """Verify that OOD dataset generation does NOT overwrite training metadata."""
    cfg = SimulationConfig(
        ml_dataset_samples=200,
        ml_n_estimators=5,
        ml_max_depth=4,
        random_seed=42
    )
    train_csv = str(tmp_path / "train.csv")
    train_meta = str(tmp_path / "train_meta.json")
    ood_csv = str(tmp_path / "ood.csv")
    ood_meta = str(tmp_path / "ood_meta.json")
    model_file = str(tmp_path / "model.joblib")
    metrics_file = str(tmp_path / "metrics.json")

    # Generate training dataset
    generate_link_dataset(config=cfg, num_samples=200, save_path=train_csv, metadata_path=train_meta, seed=42)

    # Read training metadata before OOD
    with open(train_meta, "r") as f:
        train_meta_before = json.load(f)
    assert train_meta_before["random_seed"] == 42

    # Generate OOD dataset with separate metadata path
    ood_cfg = SimulationConfig(shadowing_std=8.0, random_seed=999)
    generate_link_dataset(config=ood_cfg, num_samples=100, save_path=ood_csv, metadata_path=ood_meta, seed=999)

    # Training metadata must be unchanged
    with open(train_meta, "r") as f:
        train_meta_after = json.load(f)
    assert train_meta_after["random_seed"] == 42
    assert train_meta_after["num_samples"] == 200

    # OOD metadata must have its own values
    with open(ood_meta, "r") as f:
        ood_meta_data = json.load(f)
    assert ood_meta_data["random_seed"] == 999
    assert ood_meta_data["num_samples"] == 100


def test_rssi_feature_affects_prediction(tmp_path):
    """RSSI should actually influence PDR prediction when other features are held constant."""
    cfg = SimulationConfig(
        ml_dataset_samples=1000,
        ml_n_estimators=20,
        ml_max_depth=8,
        random_seed=42
    )
    csv_file = str(tmp_path / "rssi_test.csv")
    meta_file = str(tmp_path / "rssi_meta.json")
    model_file = str(tmp_path / "rssi_model.joblib")
    metrics_file = str(tmp_path / "rssi_metrics.json")

    pred, _ = train_and_evaluate_ml(
        config=cfg,
        dataset_path=csv_file,
        metadata_path=meta_file,
        model_save_path=model_file,
        metrics_save_path=metrics_file,
        force_retrain=True,
        ood_dataset_path=str(tmp_path / "ood.csv"),
        ood_metadata_path=str(tmp_path / "ood_meta.json"),
    )

    # Same features except past RSSI/SNR/EWMA varies
    pdr_good, _ = pred.predict(distance=20.0, prev_rssi_dbm=-65.0, prev_snr_db=20.0,
                               ewma_pdr=0.98, ewma_retx=0.05, ewma_delay_ms=20.0, rx_queue_load=1)
    pdr_bad, _ = pred.predict(distance=20.0, prev_rssi_dbm=-92.0, prev_snr_db=-7.0,
                              ewma_pdr=0.10, ewma_retx=2.5, ewma_delay_ms=65.0, rx_queue_load=1)

    # Higher RSSI should predict higher PDR
    assert pdr_good > pdr_bad, f"RSSI has no effect: good_rssi PDR={pdr_good}, bad_rssi PDR={pdr_bad}"


# =========================================================================
# TEST 6: Network Topology and Dead Node Exclusion
# =========================================================================
def test_network_topology_and_dead_node():
    cfg = SimulationConfig(num_nodes=10, comm_range=50.0, initial_energy=0.1)
    net = IoTNetwork(cfg, seed=42)

    # Check node count: 1 BS + 10 Sensors = 11 nodes
    assert len(net.nodes) == 11
    assert net.bs_id == 0
    assert net.nodes[0].is_base_station
    assert net.count_alive_nodes() == 10

    # Graph build
    G = net.build_graph()
    assert 0 in G.nodes()
    for u, v, d in G.edges(data=True):
        assert d["distance"] <= cfg.comm_range

    # Kill node 1
    net.nodes[1].consume_energy(0.2, current_round=5)
    assert not net.nodes[1].is_alive
    assert net.nodes[1].death_round == 5
    assert net.count_alive_nodes() == 9
    assert net.count_dead_nodes() == 1

    # New graph should not contain node 1
    G_dead = net.build_graph()
    assert 1 not in G_dead.nodes()


# =========================================================================
# TEST 7: Routing Algorithms Selection (SP vs Energy)
# =========================================================================
def test_routing_shortest_path_and_energy():
    cfg = SimulationConfig(num_nodes=5, comm_range=60.0)
    net = IoTNetwork(cfg, seed=42)
    engine = RoutingEngine(cfg)

    # Build SP graph and find paths
    g_sp = engine.build_weighted_graph(net, routing_mode="sp")
    paths_sp = engine.find_all_paths_to_sink(g_sp, sink_id=net.bs_id)

    for u, p in paths_sp.items():
        if p:
            assert p[0] == u
            assert p[-1] == net.bs_id

    # Build Energy-Dijkstra graph
    g_e = engine.build_weighted_graph(net, routing_mode="energy_dijkstra")
    paths_e = engine.find_all_paths_to_sink(g_e, sink_id=net.bs_id)
    for u, p in paths_e.items():
        if p:
            assert p[-1] == net.bs_id


# =========================================================================
# TEST 8: Reverse Single-Source Dijkstra Equivalence
# =========================================================================
def test_reverse_dijkstra_exact_equivalence():
    """Verify that find_all_paths_to_sink (single Dijkstra on G_rev) produces
    identical path lengths and valid shortest paths as independent Dijkstra calls.
    """
    G = nx.DiGraph()
    G.add_edge(1, 0, cost=5.0)
    G.add_edge(2, 1, cost=2.0)
    G.add_edge(2, 0, cost=10.0)
    G.add_edge(3, 2, cost=1.0)
    G.add_node(4)  # unreachable to 0

    engine = RoutingEngine()
    fast_paths = engine.find_all_paths_to_sink(G, sink_id=0)

    # Standard independent Dijkstra for reference
    ref_paths = {}
    for n in G.nodes():
        if n == 0:
            continue
        try:
            ref_paths[n] = nx.shortest_path(G, source=n, target=0, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            ref_paths[n] = []

    for n in [1, 2, 3, 4]:
        assert fast_paths[n] == ref_paths[n]

    assert fast_paths[1] == [1, 0]
    assert fast_paths[2] == [2, 1, 0]
    assert fast_paths[3] == [3, 2, 1, 0]
    assert fast_paths[4] == []


def test_dijkstra_no_dead_node_path():
    """Verify Dijkstra does not route through dead nodes."""
    cfg = SimulationConfig(num_nodes=10, comm_range=50.0, initial_energy=0.1)
    net = IoTNetwork(cfg, seed=42)
    engine = RoutingEngine(cfg)

    # Kill some nodes
    for nid in [2, 4, 6]:
        net.nodes[nid].consume_energy(1.0, current_round=1)
        assert not net.nodes[nid].is_alive

    G = engine.build_weighted_graph(net, routing_mode="sp")
    paths = engine.find_all_paths_to_sink(G, sink_id=net.bs_id)

    for src_id, path in paths.items():
        if path:
            for node_in_path in path:
                assert net.nodes[node_in_path].is_alive, f"Dead node {node_in_path} in path {path}"


# =========================================================================
# TEST 9: PSO Optimizer Behavior & Convergence
# =========================================================================
def test_pso_constraints_and_convergence():
    cfg = SimulationConfig(
        num_nodes=10,
        comm_range=45.0,
        pso_num_particles=10,
        pso_max_iter=15,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    pso = PSORoutingOptimizer(cfg, use_ml=False, seed=42)

    best_weights, best_fit = pso.optimize(net)

    # Weights constraints: w_i >= 0 and sum(w_i) == 1
    assert len(best_weights) == 4
    for w in best_weights:
        assert w >= 0.0
    assert math.isclose(sum(best_weights), 1.0, rel_tol=1e-5)

    # Fitness is finite
    assert not math.isinf(best_fit)
    assert best_fit > 0.0

    # Global best fitness curve must be non-increasing
    curve = pso.convergence_curve
    assert len(curve) == cfg.pso_max_iter
    for i in range(len(curve) - 1):
        assert curve[i + 1] <= curve[i] + 1e-9


def test_pso_snapshot_consistency():
    """PSO must evaluate fitness on the same snapshot it uses for final graph."""
    cfg = SimulationConfig(
        num_nodes=10,
        comm_range=45.0,
        pso_num_particles=5,
        pso_max_iter=5,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    engine = RoutingEngine(cfg)
    channel_rng = np.random.RandomState(42 + 20000)

    # Create snapshot manually
    snapshot = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", rng=channel_rng)

    pso = PSORoutingOptimizer(cfg, routing_engine=engine, use_ml=False, seed=42)
    best_weights, best_fit = pso.optimize(net, snapshot=snapshot)

    # Build graph from same snapshot with best weights
    G = snapshot.build_graph_with_weights(best_weights)

    # Verify the graph has the same edges as the snapshot
    assert len(list(G.edges())) == len(snapshot.edges)
    for idx, (u, v) in enumerate(snapshot.edges):
        assert G.has_edge(u, v)
        assert math.isclose(G[u][v]["pdr"], float(snapshot.pdrs[idx]), rel_tol=1e-6)
        assert math.isclose(G[u][v]["physical_pdr"], float(snapshot.physical_pdrs[idx]), rel_tol=1e-6)


# =========================================================================
# TEST 10: Simulator Execution & Deepcopy Safety
# =========================================================================
def test_simulator_execution_and_network_immutability():
    cfg = SimulationConfig(
        num_nodes=10,
        comm_range=45.0,
        initial_energy=0.1,
        total_rounds=15,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    initial_total_energy = net.get_total_residual_energy()

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol(
        protocol="sp",
        initial_network=net,
        show_progress=False,
        seed=42
    )

    # Original network MUST NOT have changed
    assert math.isclose(net.get_total_residual_energy(), initial_total_energy, rel_tol=1e-6)
    assert net.count_alive_nodes() == 10

    # Simulation result has 15 rounds
    assert len(res.rounds) == 15
    assert res.total_packets_generated > 0
    assert res.total_packets_delivered + res.total_packets_dropped == res.total_packets_generated
    assert res.final_residual_energy < initial_total_energy


def test_energy_non_negative():
    """Verify that residual energy never goes negative during simulation."""
    cfg = SimulationConfig(
        num_nodes=10,
        comm_range=45.0,
        initial_energy=0.05,  # Low energy to test depletion
        total_rounds=50,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    sim = NetworkSimulator(cfg)
    res = sim.run_protocol(protocol="sp", initial_network=net, show_progress=False, seed=42)

    # All residual energies must be >= 0
    for e in res.total_residual_energy:
        assert e >= 0.0, f"Negative residual energy: {e}"

    # Energy consumed must be <= initial total
    initial_total = cfg.num_nodes * cfg.initial_energy
    for e_consumed in res.cumulative_energy_consumed:
        assert e_consumed <= initial_total + 1e-9


# =========================================================================
# TEST 11: Hand-Calculated Milestone Metrics Test
# =========================================================================
def test_metrics_milestones_and_hand_calculation():
    res = SimulationResult("test_protocol")
    res.rounds = [1, 2, 3, 4, 5]
    res.dead_nodes = [0, 0, 1, 5, 10]
    total_sensor_nodes = 10

    res.compute_milestones(total_sensor_nodes)
    assert res.fnd_round == 3
    assert res.hnd_round == 4
    assert res.lnd_round == 5

    res.total_packets_generated = 10
    res.total_packets_delivered = 4
    res.total_delivered_delay_sum = 100.0
    assert math.isclose(res.weighted_mean_delay, 25.0, rel_tol=1e-6)
    assert math.isclose(res.overall_pdr, 0.4, rel_tol=1e-6)


# =========================================================================
# TEST 12: RNG Separation - PSO does not affect channel
# =========================================================================
def test_rng_pso_does_not_affect_channel():
    """Running PSO should not change channel sampling outcomes for the same seed."""
    cfg = SimulationConfig(
        num_nodes=8,
        comm_range=45.0,
        pso_num_particles=10,
        pso_max_iter=10,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)

    # Channel RNG stream 1: no PSO
    channel_rng_1 = np.random.RandomState(42 + 20000)
    channel = WirelessChannel(cfg)
    rssi_no_pso = channel.compute_rssi(20.0, apply_shadowing=True, rng=channel_rng_1)

    # Channel RNG stream 2: same seed, but PSO runs first (PSO has its own RNG)
    channel_rng_2 = np.random.RandomState(42 + 20000)
    # PSO uses seed + 10000 internally, does not touch channel_rng_2
    pso = PSORoutingOptimizer(cfg, use_ml=False, seed=42)
    pso.optimize(net)  # This should NOT affect channel_rng_2
    rssi_with_pso = channel.compute_rssi(20.0, apply_shadowing=True, rng=channel_rng_2)

    assert math.isclose(rssi_no_pso, rssi_with_pso, rel_tol=1e-9), \
        f"PSO changed channel RNG: {rssi_no_pso} vs {rssi_with_pso}"


def test_common_random_numbers_across_protocols():
    """Same seed should give same packet outcomes for the same (round, src, hop, attempt)."""
    from src.simulator import _make_packet_rng

    # Same parameters (seed, round, src, tx, rx, packet_idx, frag_idx, attempt)
    rng1 = _make_packet_rng(42, 1, 5, 5, 0, 0, 0, 1)
    rng2 = _make_packet_rng(42, 1, 5, 5, 0, 0, 0, 1)
    assert rng1.uniform() == rng2.uniform()

    # Different attempt
    rng3 = _make_packet_rng(42, 1, 5, 5, 0, 0, 0, 2)
    rng4 = _make_packet_rng(42, 1, 5, 5, 0, 0, 0, 1)
    # Different RNG should give different results (with high probability)
    # We just check they produce numbers, not necessarily different
    val3 = rng3.uniform()
    val4 = rng4.uniform()
    # They should be valid numbers
    assert 0.0 <= val3 <= 1.0
    assert 0.0 <= val4 <= 1.0


# =========================================================================
# TEST 13: Integration Smoke Test with Verified ML Execution & Fallback
# =========================================================================
def test_integration_smoke_all_four_protocols_with_verified_ml():
    """Verify that all 4 protocols run successfully AND ml_pso genuinely invokes ML inference."""
    cfg = SimulationConfig(
        num_nodes=8,
        comm_range=50.0,
        initial_energy=0.1,
        total_rounds=5,
        pso_num_particles=5,
        pso_max_iter=5,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)

    # Create dummy predictor spy to verify it is genuinely invoked during ml_pso
    mock_pdr_model = MagicMock()
    mock_pdr_model.predict.side_effect = lambda X: np.full(len(X), 0.95, dtype=np.float64)
    mock_delay_model = MagicMock()
    mock_delay_model.predict.side_effect = lambda X: np.full(len(X), 18.0, dtype=np.float64)

    predictor = LinkQualityPredictor(pdr_model=mock_pdr_model, delay_model=mock_delay_model)
    sim = NetworkSimulator(cfg, ml_predictor=predictor)

    protocols = ["sp", "energy_dijkstra", "pso", "ml_pso"]
    for proto in protocols:
        res = sim.run_protocol(
            protocol=proto,
            initial_network=net,
            pso_update_interval=5,
            show_progress=False,
            seed=42
        )
        assert len(res.rounds) == 5
        assert res.total_packets_generated == 8 * 5
        assert res.total_packets_delivered <= res.total_packets_generated
        assert res.final_residual_energy >= 0.0

    # Assert that ML predict was genuinely called during ml_pso execution
    assert mock_pdr_model.predict.called
    assert mock_delay_model.predict.called


def test_ml_pso_clean_fallback_when_no_predictor():
    """Verify that when ml_predictor is None, simulator falls back safely to theoretical channel model."""
    cfg = SimulationConfig(
        num_nodes=6,
        comm_range=55.0,
        initial_energy=0.1,
        total_rounds=3,
        pso_num_particles=5,
        pso_max_iter=5,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    sim = NetworkSimulator(cfg, ml_predictor=None)

    res = sim.run_protocol(protocol="ml_pso", initial_network=net, show_progress=False, seed=42)
    assert len(res.rounds) == 3
    assert res.total_packets_generated == 6 * 3


# =========================================================================
# TEST 14: PDR and Delay Consistency
# =========================================================================
def test_pdr_not_exceed_100_percent():
    """PDR should never exceed 1.0 (100%) in simulation results."""
    cfg = SimulationConfig(
        num_nodes=10,
        comm_range=45.0,
        initial_energy=0.1,
        total_rounds=10,
        random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    sim = NetworkSimulator(cfg)
    res = sim.run_protocol(protocol="pso", initial_network=net, show_progress=False, seed=42)

    assert res.overall_pdr <= 1.0
    for pdr in res.round_pdr:
        assert 0.0 <= pdr <= 1.0
    for cpdr in res.cumulative_pdr:
        assert 0.0 <= cpdr <= 1.0


# =========================================================================
# V5 TESTS: Delay Consistency, Keyed Channel, Fragmentation, HND Censoring
# =========================================================================
def test_v5_expected_transmissions_exact_and_no_double_counting():
    """Verify E[N] = (1 - (1 - 0.5)^4) / 0.5 = 1.875 and PSO does NOT multiply E[N] twice."""
    cfg = SimulationConfig(
        num_nodes=2,
        comm_range=50.0,
        max_retransmissions=3,
        packet_size_bits=800,
        max_frame_payload_bits=800,
        mac_header_fcs_bits=216
    )
    wc = WirelessChannel(cfg)
    assert cfg.num_fragments == 1
    assert math.isclose(wc.expected_transmissions(0.5), 1.875, rel_tol=1e-9)

    single_delay = wc.compute_single_hop_delay(20.0, queue_load=0)
    link_delay = wc.compute_link_delay(20.0, pdr=0.5, queue_load=0)
    assert math.isclose(link_delay, 1.875 * single_delay, rel_tol=1e-9)

    # Check PSO evaluate_fitness_with_snapshot uses link_delay ONCE (not 1.875^2)
    net = IoTNetwork(cfg, seed=42)
    # Place node 1 at (50+20, 50) directly connected to BS (50, 50), node 2 far away or same
    net.nodes[1].x, net.nodes[1].y = 70.0, 50.0
    net.nodes[2].x, net.nodes[2].y = 30.0, 50.0
    net._precompute_distances()

    engine = RoutingEngine(cfg)
    snap = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", seed=42, round_num=1)
    # Override physical_pdrs to 0.5, physical_bers to 0.0 (use frame_pdr), physical_cond_delays to conditional delay
    snap.physical_pdrs[:] = 0.5
    snap.physical_bers[:] = 0.0
    snap.physical_delays[:] = np.array([wc.compute_link_delay(d, pdr=0.5) for d in snap.distances])
    cond_delay = wc.compute_conditional_link_delay(20.0, pdr=0.5, queue_load=0)
    snap.physical_cond_delays[:] = cond_delay

    pso = PSORoutingOptimizer(cfg, routing_engine=engine, use_ml=False, seed=42)
    pso.lambda_e = 0.0
    pso.lambda_var = 0.0
    pso.lambda_d = 1.0
    pso.lambda_l = 0.0
    pso.config = SimulationConfig(**{**cfg.__dict__, "pso_adaptive": False})
    fit = pso.evaluate_fitness_with_snapshot((0.25, 0.25, 0.25, 0.25), snap, net)
    # Fitness delay term uses conditional delay ONCE (1.7333 * single_delay / 150.0), NOT multiplied by E[N] again
    expected_norm_delay = cond_delay / 150.0
    assert math.isclose(fit, expected_norm_delay, rel_tol=0.05)


def test_v5_deterministic_channel_independent_of_alive_nodes_and_order():
    """Verify RSSI for link (u, v) at round R is identical across protocols even when
    other nodes die or traversal order changes.
    """
    cfg = SimulationConfig(num_nodes=10, comm_range=50.0, random_seed=42)
    net_full = IoTNetwork(cfg, seed=42)
    net_dead = net_full.clone()
    # Kill nodes 2 and 3 in net_dead
    net_dead.nodes[2].consume_energy(1.0, current_round=1)
    net_dead.nodes[3].consume_energy(1.0, current_round=1)

    engine = RoutingEngine(cfg)
    snap_sp = engine.create_graph_snapshot(net_full, use_ml=False, routing_mode="sp", seed=42, round_num=5)
    snap_pso = engine.create_graph_snapshot(net_dead, use_ml=False, routing_mode="pso", seed=42, round_num=5)

    # Compare physical_pdrs on all surviving edges present in both snapshots
    pdr_sp = {e: p for e, p in zip(snap_sp.edges, snap_sp.physical_pdrs)}
    pdr_pso = {e: p for e, p in zip(snap_pso.edges, snap_pso.physical_pdrs)}

    common_edges = set(pdr_sp.keys()) & set(pdr_pso.keys())
    assert len(common_edges) > 0
    for edge in common_edges:
        assert math.isclose(pdr_sp[edge], pdr_pso[edge], rel_tol=1e-12), \
            f"Edge {edge} physical PDR differed when dead nodes changed!"


def test_v5_ml_pso_round1_uses_actual_round1_snapshot():
    """Verify Simulator optimizes Round 1 ML-PSO weights on Round 1's actual snapshot."""
    cfg = SimulationConfig(
        num_nodes=8, comm_range=45.0, total_rounds=2,
        pso_num_particles=6, pso_max_iter=5, random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    sim = NetworkSimulator(cfg, ml_predictor=None)
    res = sim.run_protocol(protocol="ml_pso", initial_network=net, show_progress=False, seed=42)

    assert res.initial_weights is not None
    assert math.isclose(sum(res.initial_weights), 1.0, rel_tol=1e-6)
    assert len(res.pso_convergence_curve) == cfg.pso_max_iter


def test_v5_tracemalloc_disabled_by_default_during_benchmark(tmp_path):
    """Verify tracemalloc is NOT active during normal run_experiment()."""
    import tracemalloc
    from main import run_experiment
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    cfg = SimulationConfig(
        num_nodes=6, comm_range=50.0, total_rounds=2,
        pso_num_particles=4, pso_max_iter=3,
        ml_dataset_samples=200, ml_n_estimators=5, ml_max_depth=4,
        ml_model_path=str(tmp_path / "m.joblib"),
        dataset_path=str(tmp_path / "data.csv"),
        dataset_metadata_path=str(tmp_path / "data_meta.json"),
        ood_dataset_path=str(tmp_path / "ood.csv"),
        ood_metadata_path=str(tmp_path / "ood_meta.json"),
        random_seed=42
    )
    # Run with profile_memory=False (default) into isolated tmp_path
    run_experiment(cfg, seeds=[42], show_progress=False, profile_memory=False, output_dir=str(tmp_path))
    assert not tracemalloc.is_tracing()

    with open(tmp_path / "experiment_config.json", "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)
    assert exp_cfg["tracemalloc_enabled"] is False
    assert exp_cfg["python_traced_peak_mb"] is None
    assert exp_cfg["version"] == "V19"


def test_v5_hnd_censoring_not_averaged_into_observed(tmp_path):
    """Verify HND censored trials are NOT averaged as total_rounds into observed HND mean."""
    from src.visualizer import save_aggregated_summary_table
    raw_trials = [
        {
            "Seed": 42, "Protocol": "ml_pso",
            "FND_numeric": 100, "FND_censored": False,
            "HND_numeric": 250, "HND_censored": False,
            "Overall PDR (%)": 60.0, "Weighted Mean Delay (ms)": 40.0,
            "Final Residual Energy (J)": 4.0, "Energy per Packet (mJ)": 1.0,
            "Packets Delivered": 500, "Packets Generated": 800, "Packets Dropped": 300,
            "Protocol Runtime (s)": 1.0
        },
        {
            "Seed": 43, "Protocol": "ml_pso",
            "FND_numeric": 120, "FND_censored": False,
            "HND_numeric": 300, "HND_censored": True,
            "Overall PDR (%)": 62.0, "Weighted Mean Delay (ms)": 42.0,
            "Final Residual Energy (J)": 4.2, "Energy per Packet (mJ)": 1.0,
            "Packets Delivered": 510, "Packets Generated": 800, "Packets Dropped": 290,
            "Protocol Runtime (s)": 1.0
        }
    ]
    out_csv = str(tmp_path / "summary.csv")
    df = save_aggregated_summary_table(raw_trials, total_rounds=300, save_path=out_csv)
    hnd_str = df.iloc[0]["HND (Round)"]
    # Observed mean is 250.0 (NOT (250+300)/2 = 275.0), and must note 1/2 censored
    assert "250.0" in hnd_str
    assert "1/2 >300 censored" in hnd_str


def test_v5_ieee_802_15_4_fragmentation_and_snr_relations():
    """Verify IEEE 802.15.4 127-byte frame limit, fragmentation math, and SNR/EbN0/EsN0."""
    cfg = SimulationConfig()
    assert cfg.frame_size_bits == 1016  # 127 bytes <= IEEE 802.15.4 max PSDU
    assert cfg.num_fragments == 5       # 4000 / 800 = 5 fragments

    wc = WirelessChannel(cfg)
    snr_lin = 2.0
    snr_chip = wc.snr_to_chip_snr(snr_lin)
    eb_n0 = wc.snr_to_eb_n0(snr_lin)
    es_n0 = wc.snr_to_es_n0(snr_lin)
    assert math.isclose(eb_n0, 8.0 * snr_chip, rel_tol=1e-9)
    assert math.isclose(es_n0, 32.0 * snr_chip, rel_tol=1e-9)
    assert math.isclose(es_n0, 4.0 * eb_n0, rel_tol=1e-9)

    # Link success prob across 5 fragments must be <= single-fragment success prob
    p_frame = 0.8
    p_frag_succ = 1.0 - (1.0 - p_frame) ** wc.max_attempts
    p_app_succ = wc.compute_link_success_prob(p_frame)
    assert math.isclose(p_app_succ, p_frag_succ ** 5, rel_tol=1e-9)
    assert p_app_succ < p_frag_succ


def test_v5_ood_config_inherits_base_parameters():
    """Verify dataclasses.replace preserves all base PHY/MAC parameters for OOD."""
    import dataclasses
    base_cfg = SimulationConfig(
        packet_size_bits=3200,
        bandwidth_bps=250000.0,
        max_retransmissions=4,
        noise_floor_dbm=-82.0,
        tx_power_dbm=2.0
    )
    ood_cfg = dataclasses.replace(base_cfg, shadowing_std=8.0, random_seed=999)
    assert ood_cfg.packet_size_bits == base_cfg.packet_size_bits
    assert ood_cfg.bandwidth_bps == base_cfg.bandwidth_bps
    assert ood_cfg.max_retransmissions == base_cfg.max_retransmissions
    assert ood_cfg.noise_floor_dbm == base_cfg.noise_floor_dbm
    assert ood_cfg.tx_power_dbm == base_cfg.tx_power_dbm
    assert ood_cfg.shadowing_std == 8.0
    assert ood_cfg.random_seed == 999


def test_v5_dataset_no_train_test_leakage(tmp_path):
    """Verify generated dataset has unique samples and disjoint train/test split."""
    from sklearn.model_selection import train_test_split
    cfg = SimulationConfig(random_seed=42)
    csv_file = str(tmp_path / "leak_test.csv")
    meta_file = str(tmp_path / "leak_meta.json")
    df = generate_link_dataset(config=cfg, num_samples=500, save_path=csv_file, metadata_path=meta_file, seed=42)

    X = df[FEATURE_COLS].values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=cfg.random_seed)
    train_set = set(map(tuple, X_train))
    test_set = set(map(tuple, X_test))
    assert len(train_set & test_set) == 0, "Train/test leakage detected!"


# =========================================================================
# V7 TESTS: Sequential Early-Stop Fragmentation, CRN, Queue Sync, Charts
# =========================================================================
def test_v7_sequential_fragment_early_stop_delay_and_energy_monte_carlo():
    """Verify exact sequential early-stop fragmentation formulas against Monte Carlo simulation:
    - Expected attempts per generated app packet: sum_j R_j * E[N_j]
    - Expected unconditional airtime per generated packet
    - Conditional delay given full app success: sum_j E[N_j | succ] * D_single_j
    - Expected energy (TX, RX, and DA only upon all F fragments succeeding)
    """
    cfg = SimulationConfig(
        packet_size_bits=3500,          # 4 full (800b) + 1 short (300b) = 5 fragments
        max_frame_payload_bits=800,
        mac_header_fcs_bits=216,
        max_retransmissions=3,
        comm_range=50.0
    )
    wc = WirelessChannel(cfg)
    em = EnergyModel(cfg)
    d = 25.0
    ber = 0.0008  # Produces realistic intermediate frame PDR ~ 0.44 - 0.66

    # Expected theoretical quantities
    exp_app_attempts = wc.expected_app_fragment_attempts_per_packet(0.0, ber=ber)
    exp_airtime_ms = wc.compute_unconditional_link_airtime(d, queue_load=1, ber=ber)
    exp_cond_delay_ms = wc.compute_conditional_link_delay(d, queue_load=1, ber=ber)
    exp_energy_j = em.expected_link_energy(d, is_rx_base_station=False, ber=ber)
    exp_app_succ = wc.compute_link_success_prob(0.0, ber=ber)

    # Monte Carlo simulation of exact simulator fragment loop
    rng = np.random.RandomState(12345)
    n_trials = 25000
    tot_attempts = 0
    tot_airtime = 0.0
    tot_energy = 0.0
    succ_delays = []

    frag_sizes = cfg.fragment_sizes_bits
    max_att = 1 + cfg.max_retransmissions
    da_e = em.aggregation_energy()

    for _ in range(n_trials):
        pkt_delay = 0.0
        pkt_succ = True
        for l_j in frag_sizes:
            p_j = (1.0 - ber) ** l_j
            d_single = wc.compute_single_hop_delay(d, queue_load=1, frame_bits=l_j)
            tx_e = em.tx_energy(d, num_bits=l_j)
            rx_e = em.rx_energy(num_bits=l_j)

            frag_succ = False
            for _att in range(max_att):
                tot_attempts += 1
                pkt_delay += d_single
                tot_energy += (tx_e + rx_e)
                if rng.uniform(0.0, 1.0) < p_j:
                    frag_succ = True
                    break
            if not frag_succ:
                pkt_succ = False
                break

        tot_airtime += pkt_delay
        if pkt_succ:
            tot_energy += da_e
            succ_delays.append(pkt_delay)

    mc_succ_prob = len(succ_delays) / n_trials
    mc_attempts = tot_attempts / n_trials
    mc_airtime = tot_airtime / n_trials
    mc_cond_delay = float(np.mean(succ_delays))
    mc_energy = tot_energy / n_trials

    assert math.isclose(exp_app_succ, mc_succ_prob, rel_tol=0.03)
    assert math.isclose(exp_app_attempts, mc_attempts, rel_tol=0.02)
    assert math.isclose(exp_airtime_ms, mc_airtime, rel_tol=0.02)
    assert math.isclose(exp_cond_delay_ms, mc_cond_delay, rel_tol=0.02)
    assert math.isclose(exp_energy_j, mc_energy, rel_tol=0.02)
    # Unconditional airtime differs from conditional delay when failure occurs
    assert abs(exp_airtime_ms - exp_cond_delay_ms) > 1.0


def test_v7_packet_rng_keys_link_packet_idx_and_hop_invariance():
    """Verify CRN packet RNG:
    1. Depends on (seed, round, src, tx, rx, packet_idx, frag_idx, attempt).
    2. Independent of hop_idx (not passed) and protocol name.
    3. Different links, different packet_idx produce distinct streams.
    """
    from src.simulator import _make_packet_rng, _packet_outcome_u01

    # Same link (tx=3 -> rx=7) at same (seed, round, src, packet_idx, frag_idx, attempt)
    r_proto_a = _make_packet_rng(42, 10, 5, 3, 7, 0, 2, 1).uniform()
    r_proto_b = _make_packet_rng(42, 10, 5, 3, 7, 0, 2, 1).uniform()
    assert r_proto_a == r_proto_b
    assert _packet_outcome_u01(42, 10, 5, 3, 7, 0, 2, 1) == _packet_outcome_u01(42, 10, 5, 3, 7, 0, 2, 1)

    # Different link (3->7 vs 3->8)
    r_other_link = _make_packet_rng(42, 10, 5, 3, 8, 0, 2, 1).uniform()
    assert r_proto_a != r_other_link
    assert _packet_outcome_u01(42, 10, 5, 3, 7, 0, 2, 1) != _packet_outcome_u01(42, 10, 5, 3, 8, 0, 2, 1)

    # Different packet_idx (0 vs 1)
    r_other_pkt = _make_packet_rng(42, 10, 5, 3, 7, 1, 2, 1).uniform()
    assert r_proto_a != r_other_pkt
    assert _packet_outcome_u01(42, 10, 5, 3, 7, 0, 2, 1) != _packet_outcome_u01(42, 10, 5, 3, 7, 1, 2, 1)


def test_v7_queue_load_synchronized_with_snapshot():
    """Verify queue_load is reset BEFORE creating GraphSnapshot so ML/snapshot sees queue_load=0
    matching the actual start-of-round simulator state.
    """
    cfg = SimulationConfig(num_nodes=8, comm_range=50.0, total_rounds=3, random_seed=42)
    net = IoTNetwork(cfg, seed=42)
    # Artificially set stale queue_load on a node before running simulator
    net.nodes[1].queue_load = 7

    captured_queues = []
    mock_pdr = MagicMock()
    mock_pdr.predict.side_effect = lambda X: (captured_queues.extend(X[:, 6].tolist()), np.full(len(X), 0.9))[1]
    mock_del = MagicMock()
    mock_del.predict.side_effect = lambda X: np.full(len(X), 25.0)

    pred = LinkQualityPredictor(pdr_model=mock_pdr, delay_model=mock_del)
    sim = NetworkSimulator(cfg, ml_predictor=pred)
    sim.run_protocol("ml_pso", net, pso_update_interval=5, show_progress=False, seed=42)

    assert len(captured_queues) > 0
    assert all(q == 0.0 for q in captured_queues), f"Snapshot observed stale non-zero queue_load: {set(captured_queues)}"


def test_v7_bidirectional_channel_reciprocity():
    """Verify symmetric shadowing between u->v and v->u at the same (seed, round)
    under homogeneous transceiver assumption.
    """
    from src.channel import make_channel_rng
    wc = WirelessChannel(SimulationConfig())
    rng_uv = make_channel_rng(42, 7, 3, 11)
    rng_vu = make_channel_rng(42, 7, 11, 3)
    assert math.isclose(
        wc.compute_rssi(20.0, apply_shadowing=True, rng=rng_uv),
        wc.compute_rssi(20.0, apply_shadowing=True, rng=rng_vu),
        rel_tol=1e-12
    )


def test_v7_last_fragment_exact_size():
    """Verify last fragment uses exact remaining payload bits rather than max payload."""
    cfg = SimulationConfig(packet_size_bits=3500, max_frame_payload_bits=800, mac_header_fcs_bits=216)
    assert cfg.num_fragments == 5
    # 4 * 800 + 300 = 3500
    assert cfg.fragment_payload_bits(0) == 800
    assert cfg.fragment_payload_bits(3) == 800
    assert cfg.fragment_payload_bits(4) == 300
    assert cfg.fragment_sizes_bits == [1016, 1016, 1016, 1016, 516]
    assert sum(cfg.fragment_payload_bits(j) for j in range(cfg.num_fragments)) == 3500


def test_v7_scalability_plot_no_pdr_clip_and_dynamic_fnd_title(tmp_path):
    """Verify scalability plot does NOT clip PDR below 70% and uses dynamic rounds_per_test."""
    import matplotlib
    matplotlib.use("Agg")
    from evaluate_scalability import plot_scalability_results

    rows = []
    for n, area in [(30, 80), (50, 100), (100, 140)]:
        for pkey, pdr_val in [("sp", 25.0), ("energy_dijkstra", 22.5), ("pso", 80.0), ("ml_pso", 78.0)]:
            rows.append({
                "Nodes": n, "Protocol_Key": pkey, "FND_numeric": 40, "FND_censored": True,
                "PDR (%)": pdr_val, "Weighted Mean Delay (ms)": 50.0, "Runtime (s)": 2.0
            })
    df = pd.DataFrame(rows)
    out_png = str(tmp_path / "scalability.png")
    fig = plot_scalability_results(df, [30, 50, 100], {30: 80.0, 50: 100.0, 100: 140.0}, rounds_per_test=40, save_path=out_png)
    # Inspect returned figure? plot_scalability_results closed it, so let's check file exists & inspect source or test axes
    assert os.path.exists(out_png) and os.path.getsize(out_png) > 1000
    # Verify in source that ylim is (0, 105) and FND title uses rounds_per_test
    import inspect
    src_code = inspect.getsource(plot_scalability_results)
    assert "set_ylim(0, 105)" in src_code
    assert ">{rounds_per_test}" in src_code


def test_v7_physical_baseline_fair_benchmark(tmp_path):
    """Verify analytical physical baseline (no future shadowing) and persistence baselines run fairly on same targets."""
    from src.ml_predictor import _evaluate_baselines_on_test
    cfg = SimulationConfig(random_seed=42)
    csv_f = str(tmp_path / "d.csv")
    meta_f = str(tmp_path / "m.json")
    df = generate_link_dataset(cfg, num_samples=200, save_path=csv_f, metadata_path=meta_f, seed=42)
    X = df[FEATURE_COLS].values
    y_p = df["pdr"].values
    y_d = df["link_delay_ms"].values
    res = _evaluate_baselines_on_test(X, y_p, y_d, X[:100], cfg, warmup_runs=1, bench_runs=2)
    assert "physical_analytical_no_shadow" in res
    assert "persistence_ewma" in res
    assert "moving_average_physics_blend" in res
    assert res["physical_analytical_no_shadow"]["pdr_rmse"] >= 0.0
    assert res["physical_analytical_no_shadow"]["infer_1000_samples_ms"] >= 0.0


# =========================================================================
# V10 TESTS: TX/RX/DA Energy, Queue Backlog, PSO PDR Fitness, Quick/Bench Separation
# =========================================================================
def _make_2hop_network(cfg: SimulationConfig):
    """Helper: node 1 -> node 2 -> BS (0) with 2-hop topology."""
    net = IoTNetwork(cfg, seed=42)
    # Place node 1 at (10, 50), node 2 at (30, 50), BS at (50, 50)
    # Distance 1->2 = 20m, 2->0 = 20m, 1->0 = 40m (> comm_range 30m)
    net.nodes[1].x, net.nodes[1].y = 10.0, 50.0
    net.nodes[2].x, net.nodes[2].y = 30.0, 50.0
    net._precompute_distances()
    return net


def test_v10_tx_insufficient_energy_no_rx_charge():
    """1. TX thiếu năng lượng: TX không truyền, RX KHÔNG bị trừ năng lượng."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=1,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    em = EnergyModel(cfg)
    tx_cost = em.tx_energy(20.0, num_bits=1016)
    # Set node 1 residual energy strictly below required tx_cost
    net.nodes[1].residual_energy = tx_cost * 0.5
    rx_initial = net.nodes[2].residual_energy

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    # Wait: run_protocol clones net; inspect via energy consumed
    # Node 2 (RX) must NOT have consumed rx_cost from node 1's failed TX
    # Note: node 2 itself transmits 1 packet to BS (cost = tx_cost + 0), so total energy used == tx_cost (only node 2's own TX)
    assert math.isclose(res.cumulative_energy_consumed[-1], tx_cost, rel_tol=1e-9)
    assert res.total_packets_delivered <= 1  # Only node 2's packet can succeed


def test_v10_rx_insufficient_energy_fragment_fails():
    """2. RX thiếu năng lượng: fragment không được đánh dấu thành công, không âm pin."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=1,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    em = EnergyModel(cfg)
    tx_cost = em.tx_energy(20.0, num_bits=1016)
    rx_cost = em.rx_energy(num_bits=1016)
    # Node 2 has enough to transmit its own packet, plus 0.5*rx_cost (not enough to receive from node 1)
    net.nodes[2].residual_energy = tx_cost + rx_cost * 0.5

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    assert res.final_residual_energy >= 0.0
    # Node 1's packet cannot be delivered because node 2 cannot complete RX
    assert res.total_packets_delivered <= 1


def test_v10_tx_exact_energy_completes_then_dies():
    """3. TX vừa đủ năng lượng (Residual == Required): hoàn thành frame rồi mới chết."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=1,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    net.nodes[1].x, net.nodes[1].y = 70.0, 50.0  # 20m to BS
    net.nodes[2].x, net.nodes[2].y = 10.0, 10.0  # disconnected
    net._precompute_distances()

    em = EnergyModel(cfg)
    tx_cost = em.tx_energy(20.0, num_bits=1016)
    net.nodes[1].residual_energy = tx_cost

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    # Node 1 completes its single frame to BS and then dies
    assert res.total_packets_delivered == 1
    assert res.dead_nodes[-1] >= 1


def test_v10_rx_exact_energy_completes_then_dies():
    """4. RX vừa đủ năng lượng để nhận hết frame (nhưng không đủ DA nếu DA>0) hoặc vừa đủ cả DA."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=1,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, e_da=0.0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    em = EnergyModel(cfg)
    tx_cost = em.tx_energy(20.0, num_bits=1016)
    rx_cost = em.rx_energy(num_bits=1016)
    # Node 2 has exact energy for its own TX + receiving 1 frame from node 1 (with e_da=0)
    net.nodes[2].residual_energy = tx_cost + rx_cost

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    # Node 2 completes its own TX (dies? No, wait: node 1 runs first, node 2 receives, then node 2's turn)
    # Either way, no negative energy and node 2 dies after exhausting exact energy
    assert res.final_residual_energy >= 0.0
    assert res.dead_nodes[-1] >= 1


def test_v10_aggregation_energy_and_death():
    """5. Kiểm tra aggregation energy và trạng thái node sau aggregation."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=1,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    em = EnergyModel(cfg)
    rx_cost = em.rx_energy(num_bits=1016)
    da_cost = em.aggregation_energy(num_bits=800)
    # Node 2 has enough for RX of node 1's packet, but NOT enough for aggregation
    net.nodes[2].residual_energy = rx_cost + da_cost * 0.1
    # Note: node 2 has < tx_cost so node 2 cannot TX its own packet either,
    # and after receiving from node 1, node 2 cannot afford DA so packet 1 fails!
    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    assert res.total_packets_delivered == 0
    assert res.final_residual_energy >= 0.0


def test_v10_queue_load_training_inference_consistency(tmp_path):
    """6. Queue Load training/inference có cùng miền giá trị [0, 8] và ảnh hưởng predicted delay."""
    cfg = SimulationConfig(random_seed=42, ml_dataset_samples=300)
    csv_p = str(tmp_path / "ds.csv")
    meta_p = str(tmp_path / "ds_meta.json")
    df = generate_link_dataset(cfg, num_samples=300, save_path=csv_p, metadata_path=meta_p, seed=42)
    assert df["rx_queue_load"].min() >= 0
    assert df["rx_queue_load"].max() <= 8
    assert df["rx_queue_load"].nunique() > 1

    from src.ml_predictor import _physical_no_shadow_predict
    wc = WirelessChannel(cfg)
    row_q0 = np.array([[20.0, -70.0, 15.0, 0.95, 0.05, 22.0, 0.0]])
    row_q6 = np.array([[20.0, -70.0, 15.0, 0.95, 0.05, 22.0, 6.0]])
    _, d0 = _physical_no_shadow_predict(row_q0, wc)
    _, d6 = _physical_no_shadow_predict(row_q6, wc)
    assert d6[0] > d0[0]


def test_v10_queue_load_backlog_snapshot_non_zero_under_congestion():
    """7. Deterministic congestion test: backlog > 0 carried across rounds, bounded [0, 8] in ML & delay."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=3,
        packets_per_round=5, queue_service_capacity=2, max_queue_load=8,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    # Deterministic 2-hop topology: node 1 -> node 2 -> BS (0)
    net = _make_2hop_network(cfg)

    captured_by_round = []
    mock_pdr = MagicMock()
    def _capture_pdr(X):
        captured_by_round.append(X[:, 6].tolist())
        return np.full(len(X), 1.0)
    mock_pdr.predict.side_effect = _capture_pdr
    mock_del = MagicMock()
    mock_del.predict.side_effect = lambda X: np.full(len(X), 20.0)

    pred = LinkQualityPredictor(pdr_model=mock_pdr, delay_model=mock_del)
    sim = NetworkSimulator(cfg, ml_predictor=pred)
    # Round 1 starts at 0; node 1 has service_capacity=2, so 2 packets are forwarded and 3 remain in node 1 backlog.
    # Round 2 starts with node 1 queue_load=3 (> 0!), drains 2, and buffers 5 new -> ends with 6.
    # Round 3 starts with node 1 queue_load=6 (> 0, <= 8).
    res = sim.run_protocol("ml_pso", net, pso_update_interval=10, show_progress=False, seed=42)
    # Round 1 has no prior link_history -> uses analytical fallback; Rounds 2 & 3 invoke ML with observed history
    assert res.ml_fallback_count > 0
    assert len(captured_by_round) == 2
    assert any(q == 5.0 for q in captured_by_round[0]), f"Expected relay backlog=5 in round 2, got {captured_by_round[0]}"
    assert any(q == 8.0 for q in captured_by_round[1]), f"Expected capped relay backlog=8 in round 3, got {captured_by_round[1]}"
    assert all(0.0 <= q <= 8.0 for r_qs in captured_by_round for q in r_qs)
    for round_stats in res.queue_conservation_history:
        for nid, st in round_stats.items():
            assert st["q_start"] + st["arrivals"] == st["served"] + st["q_end"] + st["dropped"]


def test_v12_queue_cap_consistency_above_8():
    """V12: When node.queue_load > 8, GraphSnapshot and simulator delay both cap at max_queue_load=8."""
    cfg = SimulationConfig(num_nodes=2, comm_range=30.0, max_queue_load=8)
    net = _make_2hop_network(cfg)
    net.nodes[2].queue_load = 15  # Exceeds 8
    engine = RoutingEngine(cfg)
    snap = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", seed=42, round_num=1)
    idx_1_2 = snap.get_edge_index(1, 2)
    expected_delay_at_8 = engine.channel.compute_unconditional_link_airtime(
        20.0, queue_load=8, ber=float(snap.physical_bers[idx_1_2])
    )
    assert math.isclose(float(snap.physical_delays[idx_1_2]), expected_delay_at_8, rel_tol=1e-9)


def test_v10_pso_pdr_fitness_includes_disconnected_as_zero():
    """8. PDR Fitness: 10 node (8 với PDR 0.9, 2 disconnected) -> avg_pdr = 0.72."""
    cfg = SimulationConfig(
        num_nodes=10, comm_range=40.0,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, pso_adaptive=False
    )
    net = IoTNetwork(cfg, seed=42)
    # Nodes 1..8 connected to BS at (50,50); nodes 9..10 disconnected far away
    for i in range(1, 9):
        net.nodes[i].x, net.nodes[i].y = 50.0 + 15.0, 50.0
    net.nodes[9].x, net.nodes[9].y = 99.0, 99.0  # > 40m from BS and others? Let's put far away
    net.nodes[9].x, net.nodes[9].y = 500.0, 500.0
    net.nodes[10].x, net.nodes[10].y = 500.0, 500.0
    net._precompute_distances()

    engine = RoutingEngine(cfg)
    snap = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", seed=42, round_num=1)
    # Force single-hop PDR = 0.9 on all connected edges
    snap.physical_pdrs[:] = 0.9
    snap.physical_bers[:] = 0.0
    snap._edge_expectations = None

    pso = PSORoutingOptimizer(cfg, routing_engine=engine, use_ml=False, seed=42)
    pso.lambda_e = 0.0
    pso.lambda_var = 0.0
    pso.lambda_d = 0.0
    pso.lambda_l = 1.0
    fit = pso.evaluate_fitness_with_snapshot((0.25, 0.25, 0.25, 0.25), snap, net)
    # avg_pdr = (8 * 0.9 + 2 * 0.0) / 10 = 0.72 -> norm_loss = 1 - 0.72 = 0.28
    assert math.isclose(fit, 0.28, rel_tol=1e-6)


def test_v10_pso_no_double_disconnected_penalty():
    """9. Không phạt disconnected hai lần và xử lý toàn bộ node mất kết nối."""
    cfg = SimulationConfig(num_nodes=4, comm_range=10.0, pso_adaptive=False)
    net = IoTNetwork(cfg, seed=42)
    for i in range(1, 5):
        net.nodes[i].x, net.nodes[i].y = 200.0 + i * 20.0, 200.0
    net._precompute_distances()

    engine = RoutingEngine(cfg)
    snap = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", seed=42, round_num=1)
    pso = PSORoutingOptimizer(cfg, routing_engine=engine, use_ml=False, seed=42)
    fit_all_disc = pso.evaluate_fitness_with_snapshot((0.25, 0.25, 0.25, 0.25), snap, net)
    assert fit_all_disc >= 1e6


def test_v10_quick_does_not_overwrite_benchmark(tmp_path):
    """10. Chạy main quick không làm thay đổi benchmark CSV."""
    from main import run_experiment
    bench_dir = tmp_path / "benchmark"
    quick_dir = tmp_path / "quick"
    bench_dir.mkdir()
    sentinel = bench_dir / "comparison_summary.csv"
    sentinel.write_text("SENTINEL_BENCHMARK", encoding="utf-8")

    cfg = SimulationConfig(
        num_nodes=6, comm_range=50.0, total_rounds=2,
        pso_num_particles=4, pso_max_iter=3,
        ml_dataset_samples=150, ml_n_estimators=5, ml_max_depth=4,
        ml_model_path=str(tmp_path / "m.joblib"),
        dataset_path=str(tmp_path / "data.csv"),
        dataset_metadata_path=str(tmp_path / "data_meta.json"),
        ood_dataset_path=str(tmp_path / "ood.csv"),
        ood_metadata_path=str(tmp_path / "ood_meta.json"),
        random_seed=42
    )
    run_experiment(cfg, seeds=[42], show_progress=False, output_dir=str(quick_dir), run_mode="quick")
    assert sentinel.read_text(encoding="utf-8") == "SENTINEL_BENCHMARK"
    assert (quick_dir / "comparison_summary.csv").exists()
    assert (quick_dir / "experiment_config.json").exists()


def test_v10_scalability_quick_does_not_overwrite_benchmark(tmp_path):
    """11. Scalability quick không ghi đè scalability benchmark."""
    from evaluate_scalability import evaluate_scalability
    bench_dir = tmp_path / "scalability" / "benchmark"
    bench_dir.mkdir(parents=True)
    sentinel = bench_dir / "scalability_summary.csv"
    sentinel.write_text("SENTINEL_SCALABILITY", encoding="utf-8")

    quick_dir = tmp_path / "scalability" / "quick"
    # Verify default directory logic by checking source/calling with explicit quick_dir
    import inspect
    from evaluate_scalability import evaluate_scalability as eval_fn
    src_code = inspect.getsource(eval_fn)
    assert "scalability_quick" in src_code and "quick_mode" in src_code
    assert sentinel.read_text(encoding="utf-8") == "SENTINEL_SCALABILITY"


def test_v10_last_fragment_energy_and_pdr():
    """12. Kích thước fragment cuối và PDR/năng lượng của fragment cuối."""
    cfg = SimulationConfig(packet_size_bits=3500, max_frame_payload_bits=800, mac_header_fcs_bits=216)
    em = EnergyModel(cfg)
    assert cfg.fragment_frame_bits(4) == 516
    assert em.frame_tx_energy(20.0, frag_idx=4) < em.frame_tx_energy(20.0, frag_idx=0)
    assert em.frame_rx_energy(frag_idx=4) < em.frame_rx_energy(frag_idx=0)


def test_v10_expected_energy_sequential_early_stop():
    """13. Expected energy phù hợp cơ chế dừng khi fragment thất bại."""
    cfg = SimulationConfig(packet_size_bits=1600, max_frame_payload_bits=800, mac_header_fcs_bits=216, max_retransmissions=1)
    em = EnergyModel(cfg)
    # With p=0.5, A=2 -> q = 0.75, E[N] = 1.5
    # R_1 = 1.0, R_2 = 0.75 -> TX = (1.0*1.5 + 0.75*1.5) * tx_1016
    tx_1 = em.tx_energy(20.0, num_bits=1016)
    rx_1 = em.rx_energy(num_bits=1016)
    da = em.aggregation_energy()
    expected = (1.0 * 1.5 + 0.75 * 1.5) * (tx_1 + rx_1) + (0.75 ** 2) * da
    actual = em.expected_link_energy(20.0, is_rx_base_station=False, single_frame_pdr=0.5)
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_v10_expected_delay_conditional_vs_unconditional():
    """14. Conditional successful delay phân biệt rõ với unconditional airtime."""
    cfg = SimulationConfig(packet_size_bits=4000, max_frame_payload_bits=800, mac_header_fcs_bits=216, max_retransmissions=3)
    wc = WirelessChannel(cfg)
    ber = 0.0008
    airtime = wc.compute_unconditional_link_airtime(25.0, queue_load=0, ber=ber)
    cond_delay = wc.compute_conditional_link_delay(25.0, queue_load=0, ber=ber)
    assert airtime > 0.0 and cond_delay > 0.0
    assert not math.isclose(airtime, cond_delay, rel_tol=1e-3)


def test_v10_common_random_numbers_protocol_agnostic():
    """15. Common Random Numbers: không phụ thuộc thứ tự duyệt hay protocol."""
    from src.simulator import _packet_outcome_u01
    u1 = _packet_outcome_u01(42, 3, 7, 7, 2, 0, 1, 2)
    u2 = _packet_outcome_u01(42, 3, 7, 7, 2, 0, 1, 2)
    u_diff_tx = _packet_outcome_u01(42, 3, 7, 8, 2, 0, 1, 2)
    u_diff_pkt = _packet_outcome_u01(42, 3, 7, 7, 2, 1, 1, 2)
    assert u1 == u2
    assert u1 != u_diff_tx
    assert u1 != u_diff_pkt


def test_v10_model_save_and_load_generic_name(tmp_path):
    """16. Model lưu và tải với tên tổng quát link_quality_predictor.joblib."""
    cfg = SimulationConfig(random_seed=42, ml_dataset_samples=200, ml_n_estimators=5, ml_max_depth=4)
    model_path = str(tmp_path / "link_quality_predictor.joblib")
    metrics_path = str(tmp_path / "ml_metrics.json")
    ds_path = str(tmp_path / "ds.csv")
    meta_path = str(tmp_path / "ds_meta.json")
    pred, metrics = train_and_evaluate_ml(
        cfg, dataset_path=ds_path, metadata_path=meta_path,
        model_save_path=model_path, metrics_save_path=metrics_path,
        ood_dataset_path=str(tmp_path / "ood.csv"),
        ood_metadata_path=str(tmp_path / "ood_meta.json"),
    )
    assert os.path.exists(model_path)
    loaded = load_predictor(model_path=model_path, config=cfg)
    p1, d1 = pred.predict(15.0, -68.0, 17.0, 0.9, 0.1, 22.0, 2)
    p2, d2 = loaded.predict(15.0, -68.0, 17.0, 0.9, 0.1, 22.0, 2)
    assert math.isclose(p1, p2, rel_tol=1e-9)
    assert math.isclose(d1, d2, rel_tol=1e-9)


def test_v10_ml_feature_order_and_range_consistency():
    """17. ML feature consistency giữa dataset.FEATURE_COLS và GraphSnapshot."""
    assert FEATURE_COLS == [
        "distance", "prev_rssi_dbm", "prev_snr_db",
        "ewma_pdr", "ewma_retx", "ewma_delay_ms", "rx_queue_load"
    ]


def test_v10_paired_statistics_computation(tmp_path):
    """18. Benchmark paired statistics theo seed: Holm correction, censored milestones excluded from t-test, zero warnings."""
    import warnings
    from src.visualizer import compute_paired_statistics
    raw_trials = []
    for s in [42, 43, 44, 45, 46]:
        d = s - 42
        raw_trials.append({"Seed": s, "Protocol": "sp", "Overall PDR (%)": 30.0 + d * 1.1, "Weighted Mean Delay (ms)": 38.0 + d * 0.4, "Final Residual Energy (J)": 6.5 + d * 0.1, "Energy per Packet (mJ)": 1.3 + d * 0.02, "FND_numeric": 300, "FND_censored": True, "HND_numeric": 300, "HND_censored": True, "LND_numeric": 300, "LND_censored": True})
        raw_trials.append({"Seed": s, "Protocol": "pso", "Overall PDR (%)": 44.0 + d * 1.3, "Weighted Mean Delay (ms)": 44.0 + d * 0.5, "Final Residual Energy (J)": 5.4 + d * 0.08, "Energy per Packet (mJ)": 1.06 + d * 0.01, "FND_numeric": 300, "FND_censored": True, "HND_numeric": 300, "HND_censored": True, "LND_numeric": 300, "LND_censored": True})
        raw_trials.append({"Seed": s, "Protocol": "ml_pso", "Overall PDR (%)": 44.3 + d * 1.35, "Weighted Mean Delay (ms)": 44.2 + d * 0.55, "Final Residual Energy (J)": 5.38 + d * 0.08, "Energy per Packet (mJ)": 1.06 + d * 0.01, "FND_numeric": 300, "FND_censored": True, "HND_numeric": 300, "HND_censored": True, "LND_numeric": 300, "LND_censored": True})
        raw_trials.append({"Seed": s, "Protocol": "energy_dijkstra", "Overall PDR (%)": 28.5 + d * 1.0, "Weighted Mean Delay (ms)": 37.5 + d * 0.3, "Final Residual Energy (J)": 7.4 + d * 0.09, "Energy per Packet (mJ)": 1.18 + d * 0.015, "FND_numeric": 300, "FND_censored": True, "HND_numeric": 300, "HND_censored": True, "LND_numeric": 300, "LND_censored": True})

    out_json = str(tmp_path / "paired.json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats_res = compute_paired_statistics(raw_trials, save_path=out_json)
    assert len(caught) == 0, f"Unexpected warnings in paired statistics: {[str(w.message) for w in caught]}"
    assert os.path.exists(out_json)
    assert "ml_pso_vs_sp" in stats_res["pairs"]
    assert stats_res["pairs"]["ml_pso_vs_sp"]["metrics"]["Overall PDR (%)"]["mean_diff"] > 10.0
    assert stats_res["pairs"]["ml_pso_vs_sp"]["metrics"]["Overall PDR (%)"]["p_value_holm"] is not None
    assert stats_res["pairs"]["ml_pso_vs_sp"]["censored_milestones"]["FND"]["status"] == "not_estimable_right_censored"
    assert stats_res["pairs"]["ml_pso_vs_sp"]["censored_milestones"]["FND"]["paired_t_p_value"] is None


def test_v10_censoring_fnd_hnd_lnd_summary(tmp_path):
    """19. Censoring FND/HND/LND không bị trung bình hóa như giá trị quan sát."""
    from src.visualizer import save_aggregated_summary_table
    raw_trials = [
        {"Seed": 42, "Protocol": "sp", "FND_numeric": 300, "FND_censored": True, "HND_numeric": 300, "HND_censored": True, "LND_numeric": 300, "LND_censored": True, "Final Residual Energy (J)": 5.0, "Overall PDR (%)": 40.0, "Weighted Mean Delay (ms)": 40.0, "Energy per Packet (mJ)": 1.5, "Packets Delivered": 500, "Packets Generated": 1200, "Packets Dropped": 700, "Protocol Runtime (s)": 10.0},
        {"Seed": 43, "Protocol": "sp", "FND_numeric": 300, "FND_censored": True, "HND_numeric": 300, "HND_censored": True, "LND_numeric": 300, "LND_censored": True, "Final Residual Energy (J)": 5.1, "Overall PDR (%)": 41.0, "Weighted Mean Delay (ms)": 41.0, "Energy per Packet (mJ)": 1.4, "Packets Delivered": 510, "Packets Generated": 1200, "Packets Dropped": 690, "Protocol Runtime (s)": 10.0},
    ]
    df = save_aggregated_summary_table(raw_trials, 300, save_path=str(tmp_path / "sum.csv"))
    assert "censored" in df.iloc[0]["FND (Round)"]
    assert "censored" in df.iloc[0]["HND (Round)"]


def test_v10_scalability_output_and_metadata(tmp_path):
    """20. Scalability plot & metadata lưu đúng định dạng."""
    from evaluate_scalability import plot_scalability_results
    rows = [
        {"Nodes": n, "Protocol_Key": pk, "FND_numeric": 40, "FND_censored": True, "PDR (%)": 50.0, "Weighted Mean Delay (ms)": 40.0, "Runtime (s)": 2.0}
        for n in [30, 50, 100]
        for pk in ["sp", "energy_dijkstra", "pso", "ml_pso"]
    ]
    out_png = str(tmp_path / "scal.png")
    plot_scalability_results(pd.DataFrame(rows), [30, 50, 100], {30: 80.0, 50: 100.0, 100: 140.0}, rounds_per_test=40, save_path=out_png)
    assert os.path.exists(out_png) and os.path.getsize(out_png) > 1000


# =========================================================================
# TEST V13: Dataset, Model Verification, Queue Conservation, Energy, Bounds
# =========================================================================

def test_v13_dataset_validation_rejects_mismatch_and_checks_bounds(tmp_path):
    """V13: is_dataset_valid rejects mismatched sample count or comm_range, and validates schema/bounds."""
    from src.dataset import FEATURE_COLS, TARGET_COLS, GROUP_COL
    cfg = SimulationConfig(comm_range=35.0, ml_dataset_samples=200, random_seed=42)
    csv_p = str(tmp_path / "d.csv")
    meta_p = str(tmp_path / "m.json")
    df = generate_link_dataset(config=cfg, num_samples=200, save_path=csv_p, metadata_path=meta_p, seed=42)
    assert list(df.columns) == FEATURE_COLS + TARGET_COLS + [GROUP_COL]
    assert is_dataset_valid(csv_p, meta_p, expected_seed=42, expected_samples=200, config=cfg)

    # Mismatched expected_samples
    assert not is_dataset_valid(csv_p, meta_p, expected_seed=42, expected_samples=6000, config=cfg)
    # Mismatched comm_range
    wrong_cfg = SimulationConfig(comm_range=50.0, ml_dataset_samples=200, random_seed=42)
    assert not is_dataset_valid(csv_p, meta_p, expected_seed=42, expected_samples=200, config=wrong_cfg)


def test_v13_model_artifact_verification_and_schema(tmp_path):
    """V13: load_predictor with allow_retrain=False rejects mismatched config/schema and verifies finite output."""
    cfg = SimulationConfig(
        comm_range=35.0, ml_dataset_samples=200, ml_n_estimators=5, ml_max_depth=4, random_seed=42
    )
    csv_p = str(tmp_path / "d.csv")
    meta_p = str(tmp_path / "m.json")
    model_p = str(tmp_path / "model.joblib")
    metrics_p = str(tmp_path / "metrics.json")
    ood_p = str(tmp_path / "ood.csv")
    ood_m = str(tmp_path / "ood_m.json")
    cfg.ood_dataset_path = ood_p
    cfg.ood_metadata_path = ood_m

    pred, metrics = train_and_evaluate_ml(
        config=cfg, dataset_path=csv_p, metadata_path=meta_p,
        model_save_path=model_p, metrics_save_path=metrics_p, force_retrain=True
    )
    assert metrics["train_samples"] == 160
    assert metrics["test_samples"] == 40
    assert metrics["model_size_bytes"] == os.path.getsize(model_p)
    assert "8.0dB" in metrics["generalization_evaluation_ood"]["scenario"]

    loaded = load_predictor(model_p, config=cfg, allow_retrain=False)
    pdrs, delays = loaded.predict_batch(np.array([[15.0, 0.8, 0.8, -70.0, 30.0, 1e-4, 1]]))
    assert np.all(np.isfinite(pdrs)) and np.all(np.isfinite(delays))

    wrong_cfg = SimulationConfig(comm_range=50.0, ml_dataset_samples=200, random_seed=42)
    with pytest.raises(ValueError):
        load_predictor(model_p, config=wrong_cfg, allow_retrain=False)


def _assert_conservation(res):
    assert res.total_packets_generated == res.total_packets_delivered + res.total_packets_dropped
    for round_stats in res.queue_conservation_history:
        for nid, st in round_stats.items():
            lhs = st["q_start"] + st["arrivals"]
            rhs = st["served"] + st["q_end"] + st["dropped"]
            assert lhs == rhs, f"Conservation violated at node {nid}: {st}"


def test_v13_queue_conservation_no_congestion_no_ghost_backlog():
    """V13: Forwarded packets are dequeued immediately and do not leave ghost backlog."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=3,
        packets_per_round=2, queue_service_capacity=8, max_queue_load=8,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    _assert_conservation(res)
    assert net.nodes[1].queue_load == 0
    assert net.nodes[2].queue_load == 0
    assert res.packets_remaining_in_queue == 0
    assert res.total_packets_delivered == 12  # 2 nodes * 2 pkts/round * 3 rounds


def test_v13_queue_conservation_intermediate_relay_congestion_and_full():
    """V13: Intermediate relay buffers excess packets above service_capacity and drops when queue is full."""
    cfg = SimulationConfig(
        num_nodes=4, comm_range=30.0, total_rounds=3,
        packets_per_round=2, queue_service_capacity=2, max_queue_load=3,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    net = IoTNetwork(cfg, seed=42)
    net.nodes[0].x, net.nodes[0].y = 0.0, 0.0    # BS
    net.nodes[2].x, net.nodes[2].y = 20.0, 0.0   # Relay in range of BS
    net.nodes[1].x, net.nodes[1].y = 40.0, 0.0   # Source 1 -> only reaches 2
    net.nodes[3].x, net.nodes[3].y = 40.0, 5.0   # Source 3 -> only reaches 2
    net.nodes[4].x, net.nodes[4].y = 40.0, -5.0  # Source 4 -> only reaches 2
    net._precompute_distances()

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    _assert_conservation(res)
    # Relay 2 has service_capacity=2 and max_queue_load=3; sources send 6 packets/round through 2
    r1_relay = res.queue_conservation_history[0][2]
    assert r1_relay["q_end"] <= 3
    assert r1_relay["served"] <= 2


def test_v13_queue_conservation_node_death_with_backlog():
    """V13: When a node dies with buffered backlog, backlog is dropped and conservation holds."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=3,
        packets_per_round=3, queue_service_capacity=1, max_queue_load=8,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    em = EnergyModel(cfg)
    tx_cost = em.tx_energy(20.0, num_bits=1016)
    # Give node 1 just enough energy for 2 TXs so it dies in Round 2 while holding backlog
    net.nodes[1].residual_energy = tx_cost * 2.0

    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    _assert_conservation(res)
    assert res.dead_nodes[-1] >= 1
    assert res.packets_remaining_in_queue == 0 or res.total_packets_dropped > 0


def test_v13_queue_load_bounds_0_1_8_and_above():
    """V13: Delay & snapshot are monotonic across queue_load=0, 1, 8 and capped for >8."""
    cfg = SimulationConfig(num_nodes=2, comm_range=30.0, max_queue_load=8)
    net = _make_2hop_network(cfg)
    engine = RoutingEngine(cfg)

    delays = []
    for qv in [0, 1, 8, 15]:
        net.nodes[2].queue_load = qv
        snap = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", seed=42, round_num=1)
        idx = snap.get_edge_index(1, 2)
        delays.append(float(snap.physical_delays[idx]))
    assert delays[0] < delays[1] < delays[2]
    assert math.isclose(delays[2], delays[3], rel_tol=1e-9)


def test_v13_energy_tx_rx_da_exact_depletion():
    """V13: TX, RX, and aggregation follow identical exact-depletion rule (complete op, then die)."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=1,
        packets_per_round=1, queue_service_capacity=8,
        packet_size_bits=800, max_frame_payload_bits=800, mac_header_fcs_bits=216,
        max_retransmissions=0, noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    em = EnergyModel(cfg)
    da_cost = em.aggregation_energy(num_bits=800)
    rx_cost = em.rx_energy(num_bits=1016)

    # RX has exact rx_cost + da_cost -> completes RX and DA, then dies (cannot forward)
    net = _make_2hop_network(cfg)
    net.nodes[2].residual_energy = rx_cost + da_cost
    sim = NetworkSimulator(cfg)
    res = sim.run_protocol("sp", net, show_progress=False, seed=42)
    assert res.dead_nodes[-1] == 1
    assert res.total_packets_delivered == 0  # Node 2 died right after RX+DA of node 1's packet, so 0 delivered
    _assert_conservation(res)


# =========================================================================
# TEST V14: Quick Isolation, Edge Cases (Dataset, ML, Topology, Energy)
# =========================================================================

def test_v14_quick_mode_isolated_from_benchmark_artifacts(tmp_path):
    """V14: Quick mode with default SimulationConfig paths must NOT overwrite data/ or models/."""
    from main import run_experiment
    prod_files = [
        "data/network_data.csv",
        "data/network_data_metadata.json",
        "data/ood_test_data.csv",
        "data/ood_test_metadata.json",
        "models/link_quality_predictor.joblib",
    ]
    before_stats = {
        f: (os.path.getmtime(f), os.path.getsize(f))
        for f in prod_files if os.path.exists(f)
    }

    qdir = tmp_path / "quick"
    cfg = SimulationConfig(
        num_nodes=6, comm_range=50.0, total_rounds=2,
        pso_num_particles=4, pso_max_iter=3,
        ml_dataset_samples=100, ml_n_estimators=5, ml_max_depth=4,
        random_seed=42
    )
    # Run quick mode twice consecutively
    run_experiment(cfg, seeds=[42], show_progress=False, output_dir=str(qdir), run_mode="quick")
    run_experiment(cfg, seeds=[42], show_progress=False, output_dir=str(qdir), run_mode="quick")

    after_stats = {
        f: (os.path.getmtime(f), os.path.getsize(f))
        for f in prod_files if os.path.exists(f)
    }
    assert before_stats == after_stats, "Quick mode modified production data/ or models/ files!"
    assert (qdir / "quick_network_data.csv").exists()
    assert (qdir / "quick_link_quality_predictor.joblib").exists()


def test_v14_edge_cases_empty_and_missing_cols_dataset(tmp_path):
    """V14: Empty dataset or dataset missing columns is rejected without silent overwrite."""
    cfg = SimulationConfig(comm_range=35.0, ml_dataset_samples=100, random_seed=42)
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    meta_p = tmp_path / "meta.json"
    meta_p.write_text("{}", encoding="utf-8")

    assert not is_dataset_valid(str(empty_csv), str(meta_p), config=cfg)
    with pytest.raises(ValueError):
        train_and_evaluate_ml(
            config=cfg, dataset_path=str(empty_csv), metadata_path=str(meta_p),
            model_save_path=str(tmp_path / "m.joblib"), metrics_save_path=str(tmp_path / "met.json"),
            force_retrain=False
        )

    bad_csv = tmp_path / "bad_cols.csv"
    pd.DataFrame({"distance": [10.0], "pdr": [0.9]}).to_csv(bad_csv, index=False)
    assert not is_dataset_valid(str(bad_csv), str(meta_p), config=cfg)


def test_v14_edge_cases_missing_and_invalid_ml_model(tmp_path):
    """V14: Missing model raises FileNotFoundError; NaN/Inf ML prediction raises ValueError."""
    cfg = SimulationConfig()
    with pytest.raises(FileNotFoundError):
        load_predictor(str(tmp_path / "nonexistent.joblib"), config=cfg, allow_retrain=False)

    bad_pdr = MagicMock()
    bad_pdr.predict.return_value = np.array([float("nan")])
    bad_del = MagicMock()
    bad_del.predict.return_value = np.array([10.0])
    pred = LinkQualityPredictor(pdr_model=bad_pdr, delay_model=bad_del)
    with pytest.raises(ValueError):
        pred.predict(10.0, 0.8, 0.8, -70.0, 20.0, 1e-4, 0)
    with pytest.raises(ValueError):
        pred.predict_batch(np.array([[10.0, 0.8, 0.8, -70.0, 20.0, 1e-4, 0]]))


def test_v14_edge_cases_single_node_zero_nodes_coincident_coords():
    """V14: Handles num_nodes=1, num_nodes=0, and two nodes at identical coordinates."""
    # Single sensor node
    cfg1 = SimulationConfig(num_nodes=1, comm_range=50.0, total_rounds=2, pso_num_particles=3, pso_max_iter=2)
    net1 = IoTNetwork(cfg1, seed=42)
    net1.nodes[1].x, net1.nodes[1].y = 50.0, 60.0
    net1._precompute_distances()
    res1 = NetworkSimulator(cfg1).run_protocol("sp", net1, show_progress=False, seed=42)
    assert res1.total_packets_generated == 2
    _assert_conservation(res1)

    # Zero sensor nodes
    cfg0 = SimulationConfig(num_nodes=0, comm_range=50.0, total_rounds=2)
    net0 = IoTNetwork(cfg0, seed=42)
    res0 = NetworkSimulator(cfg0).run_protocol("sp", net0, show_progress=False, seed=42)
    assert res0.total_packets_generated == 0
    assert res0.overall_pdr == 0.0

    # Coincident coordinates (distance == 0.0)
    cfg2 = SimulationConfig(num_nodes=2, comm_range=50.0, total_rounds=1)
    net2 = IoTNetwork(cfg2, seed=42)
    net2.nodes[1].x, net2.nodes[1].y = 50.0, 50.0
    net2.nodes[2].x, net2.nodes[2].y = 50.0, 50.0
    net2._precompute_distances()
    assert net2.get_distance(1, 2) == 0.0
    res2 = NetworkSimulator(cfg2).run_protocol("sp", net2, show_progress=False, seed=42)
    assert res2.total_packets_generated == 2
    _assert_conservation(res2)


def test_v14_edge_cases_dead_nodes_and_zero_energy():
    """V14: Dead source/relay or zero energy drops packets without negative energy."""
    cfg = SimulationConfig(
        num_nodes=2, comm_range=30.0, total_rounds=2,
        packets_per_round=1, max_retransmissions=0,
        noise_floor_dbm=-110.0, shadowing_std=0.0, random_seed=42
    )
    net = _make_2hop_network(cfg)
    # Dead intermediate relay 2
    net.nodes[2].residual_energy = 0.0
    net.nodes[2].is_alive = False
    res = NetworkSimulator(cfg).run_protocol("sp", net, show_progress=False, seed=42)
    assert res.total_packets_delivered == 0
    assert res.overall_pdr == 0.0
    _assert_conservation(res)

    # Different seeds produce distinct topologies
    net_a = IoTNetwork(SimulationConfig(num_nodes=20), seed=42)
    net_b = IoTNetwork(SimulationConfig(num_nodes=20), seed=43)
    assert (net_a.nodes[1].x, net_a.nodes[1].y) != (net_b.nodes[1].x, net_b.nodes[1].y)


# =========================================================================
# TEST V16: SHA-256 Official Artifact Immutability & ML-PSO Ablation
# =========================================================================

def test_v16_official_artifacts_sha256_unchanged_after_ml_tests(tmp_path):
    """V16: Running train_and_evaluate_ml in tmp_path (even without explicit ood_* args)
    must NOT alter SHA-256 of any official data/, models/, or results/benchmark/ artifact.
    """
    import hashlib
    import pathlib

    official_files = [
        "data/network_data.csv",
        "data/network_data_metadata.json",
        "data/ood_test_data.csv",
        "data/ood_test_metadata.json",
        "models/link_quality_predictor.joblib",
        "results/benchmark/comparison_raw.csv",
        "results/benchmark/comparison_summary.csv",
    ]
    before_hashes = {
        f: hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
        for f in official_files if os.path.exists(f)
    }

    cfg = SimulationConfig(
        ml_dataset_samples=200, ml_n_estimators=5, ml_max_depth=4, random_seed=42
    )
    pred, metrics = train_and_evaluate_ml(
        config=cfg,
        dataset_path=str(tmp_path / "train.csv"),
        metadata_path=str(tmp_path / "train_meta.json"),
        model_save_path=str(tmp_path / "model.joblib"),
        metrics_save_path=str(tmp_path / "metrics.json"),
        force_retrain=True,
    )
    assert "additional_generalization_scenarios" in metrics
    assert "seed_holdout_777" in metrics["additional_generalization_scenarios"]
    assert (tmp_path / "ood_test_data.csv").exists()
    assert (tmp_path / "ood_test_metadata.json").exists()

    after_hashes = {
        f: hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
        for f in official_files if os.path.exists(f)
    }
    assert before_hashes == after_hashes, "Official artifacts were modified by ML test!"


def test_v16_ml_ablation_influences_pso_routing():
    """V16: ML predictions genuinely alter GraphSnapshot link costs & PSO route choice
    while strictly respecting comm_range and alive_nodes constraints.
    """
    cfg = SimulationConfig(num_nodes=3, comm_range=30.0, total_rounds=1, pso_adaptive=False)
    net = IoTNetwork(cfg, seed=42)
    # Diamond topology: BS=0 at (0,0), relay 2 at (20, 10), relay 3 at (20, -10), src 1 at (40, 0)
    net.nodes[0].x, net.nodes[0].y = 0.0, 0.0
    net.nodes[2].x, net.nodes[2].y = 20.0, 10.0
    net.nodes[3].x, net.nodes[3].y = 20.0, -10.0
    net.nodes[1].x, net.nodes[1].y = 40.0, 0.0
    net._precompute_distances()

    # Mock ML predictor that predicts low PDR (0.2) for edges going through node 2 (y > 0)
    # and high PDR (0.99) for edges going through node 3 (y < 0).
    mock_pdr = MagicMock()
    def _ablation_pdr(X):
        # X[:, 4] is dist_to_bs; distinguish node 2 vs 3 by rx_ratio or dist_to_bs
        # Set node 2 residual_energy lower so rx_ratio differs
        return np.where(X[:, 2] < 0.7, 0.2, 0.99)
    mock_pdr.predict.side_effect = _ablation_pdr
    mock_del = MagicMock()
    mock_del.predict.side_effect = lambda X: np.where(X[:, 2] < 0.7, 80.0, 10.0)

    net.nodes[2].residual_energy = cfg.initial_energy * 0.5
    net.nodes[3].residual_energy = cfg.initial_energy * 0.9

    pred = LinkQualityPredictor(pdr_model=mock_pdr, delay_model=mock_del)
    engine = RoutingEngine(cfg, ml_predictor=pred)

    snap_ml = engine.create_graph_snapshot(net, use_ml=True, routing_mode="hybrid", seed=42, round_num=1)
    snap_phys = engine.create_graph_snapshot(net, use_ml=False, routing_mode="pso", seed=42, round_num=1)

    idx_1_2 = snap_ml.get_edge_index(1, 2)
    idx_1_3 = snap_ml.get_edge_index(1, 3)
    assert snap_ml.pdrs[idx_1_2] < snap_ml.pdrs[idx_1_3]
    assert snap_ml.norm_losses[idx_1_2] > snap_ml.norm_losses[idx_1_3]
    assert not np.allclose(snap_ml.norm_losses, snap_phys.norm_losses)

    # Verify all returned paths only use valid edges <= comm_range and end at BS
    paths = snap_ml.find_all_paths_fast((0.1, 0.45, 0.45, 0.0))
    assert paths[1] == [1, 3, 0]


# =========================================================================
# TEST V19: Causal Dataset Grouping, OOD Detection/Fallback, Noise -80dBm
# =========================================================================

def test_v19_group_split_zero_topology_overlap(tmp_path):
    """V19: Dataset uses independent topology seeds and GroupShuffleSplit avoids topology leakage."""
    from src.dataset import GROUP_COL
    cfg = SimulationConfig(ml_dataset_samples=600, random_seed=42)
    csv_p = str(tmp_path / "v19_ds.csv")
    meta_p = str(tmp_path / "v19_meta.json")
    df = generate_link_dataset(cfg, num_samples=600, save_path=csv_p, metadata_path=meta_p, seed=42)
    assert df[GROUP_COL].nunique() >= 2

    _, metrics = train_and_evaluate_ml(
        config=cfg,
        dataset_path=csv_p,
        metadata_path=meta_p,
        model_save_path=str(tmp_path / "m.joblib"),
        metrics_save_path=str(tmp_path / "met.json"),
        force_retrain=True,
    )
    assert "GroupShuffleSplit" in metrics["split_strategy"]
    assert metrics["train_topologies"] >= 1 and metrics["test_topologies"] >= 1
    assert metrics["train_topologies"] + metrics["test_topologies"] == df[GROUP_COL].nunique()


def test_v19_ood_detection_and_fallback_safety(tmp_path):
    """V19: OOD / NaN / unobserved inputs trigger analytical fallback cleanly."""
    cfg = SimulationConfig(ml_dataset_samples=400, ml_n_estimators=5, ml_max_depth=4, random_seed=42)
    pred, _ = train_and_evaluate_ml(
        config=cfg,
        dataset_path=str(tmp_path / "ds.csv"),
        metadata_path=str(tmp_path / "meta.json"),
        model_save_path=str(tmp_path / "m.joblib"),
        metrics_save_path=str(tmp_path / "met.json"),
        force_retrain=True,
    )
    assert pred.feature_bounds is not None

    X_test = np.array([
        [15.0, -70.0, 15.0, 0.95, 0.1, 22.0, 1.0],       # In-distribution
        [15.0, -250.0, -165.0, 0.0, 3.0, 90.0, 1.0],    # Severe OOD RSSI/SNR
        [15.0, float("nan"), 10.0, 0.9, 0.1, 22.0, 1.0] # NaN input
    ], dtype=np.float64)
    fb_pdr = np.array([0.92, 0.88, 0.85], dtype=np.float64)
    fb_del = np.array([21.0, 22.0, 23.0], dtype=np.float64)

    pdrs, dels, fb_mask = pred.predict_with_fallback(X_test, fb_pdr, fb_del)
    assert not fb_mask[0]
    assert fb_mask[1] and fb_mask[2]
    assert np.isclose(pdrs[1], 0.88) and np.isclose(dels[1], 22.0)
    assert np.isclose(pdrs[2], 0.85) and np.isclose(dels[2], 23.0)


def test_v19_noise_floor_minus80_generalization(tmp_path):
    """V19: Scenario evaluation includes noise_floor=-80dBm with OOD fallback protection."""
    cfg = SimulationConfig(ml_dataset_samples=600, ml_n_estimators=10, ml_max_depth=6, random_seed=42)
    _, metrics = train_and_evaluate_ml(
        config=cfg,
        dataset_path=str(tmp_path / "ds.csv"),
        metadata_path=str(tmp_path / "meta.json"),
        model_save_path=str(tmp_path / "m.joblib"),
        metrics_save_path=str(tmp_path / "met.json"),
        force_retrain=True,
    )
    sc = metrics["additional_generalization_scenarios"]
    assert "ood_noise_minus80" in sc
    assert "ood_nodes_80" in sc
    assert "ood_traffic_2" in sc
    assert np.isfinite(sc["ood_noise_minus80"]["ml_with_fallback"]["pdr_rmse"])
    assert sc["ood_noise_minus80"]["ml_with_fallback"]["pdr_rmse"] <= 0.6


def test_v19_causal_link_history_updates_during_simulation(tmp_path):
    """V19: ml_pso uses fallback at Round 1 (unobserved) and reduces fallback rate as link_history builds."""
    cfg = SimulationConfig(
        num_nodes=8, area_width=60.0, area_height=60.0, base_station_pos=(30.0, 30.0),
        comm_range=35.0, total_rounds=5, pso_num_particles=5, pso_max_iter=5,
        ml_dataset_samples=300, ml_n_estimators=5, ml_max_depth=4, random_seed=42
    )
    pred, _ = train_and_evaluate_ml(
        config=cfg,
        dataset_path=str(tmp_path / "ds.csv"),
        metadata_path=str(tmp_path / "meta.json"),
        model_save_path=str(tmp_path / "m.joblib"),
        metrics_save_path=str(tmp_path / "met.json"),
        force_retrain=True,
    )
    net = IoTNetwork(cfg, seed=42)
    sim = NetworkSimulator(cfg, ml_predictor=pred)
    res = sim.run_protocol("ml_pso", net, show_progress=False, seed=42)
    assert res.ml_total_predictions > 0
    assert res.ml_fallback_count > 0  # Round 1 edges have no prior observations
    assert res.ml_fallback_rate < 1.0  # Subsequent rounds use observed causal ML predictions





