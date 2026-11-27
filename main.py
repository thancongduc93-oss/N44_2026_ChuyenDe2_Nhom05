import os
import sys
import time
import json
import hashlib
import pathlib
import datetime
import dataclasses
import argparse
from typing import Dict, List, Any
import pandas as pd
import psutil

from src.config import SimulationConfig
from src.network import IoTNetwork
from src.ml_predictor import train_and_evaluate_ml
from src.routing import RoutingEngine
from src.simulator import NetworkSimulator, SimulationResult
from src.visualizer import (
    plot_network_topology,
    plot_comparative_metrics,
    plot_pso_convergence,
    plot_ml_feature_importances,
    save_aggregated_summary_table,
    save_raw_results,
    compute_paired_statistics
)

def compute_source_code_sha256() -> str:
    """Compute a deterministic SHA-256 digest over project Python source files."""
    h = hashlib.sha256()
    files = sorted(pathlib.Path("src").glob("*.py")) + [pathlib.Path("main.py"), pathlib.Path("evaluate_scalability.py")]
    for fp in files:
        if fp.exists():
            h.update(fp.as_posix().encode("utf-8"))
            h.update(fp.read_bytes())
    return h.hexdigest()


def get_process_rss_mb() -> float:
    """Return process resident set size (RSS) memory in megabytes."""
    try:
        process = psutil.Process()
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0

def run_experiment(
    config: SimulationConfig,
    seeds: List[int] = [42],
    show_progress: bool = True,
    profile_memory: bool = False,
    output_dir: str = "results",
    run_mode: str = "benchmark"
) -> Dict[str, Any]:
    """Execute end-to-end multi-seed benchmark across all routing protocols.

    Note: ``tracemalloc`` is NOT enabled during normal benchmark runs so that
    Python heap tracing overhead does not distort runtime measurements.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)


    if run_mode == "quick":
        repl = {}
        if config.dataset_path in ("data/v19_network_data.csv", "data/network_data.csv"):
            repl["dataset_path"] = os.path.join(output_dir, "quick_network_data.csv")
        if config.dataset_metadata_path in ("data/v19_network_data_metadata.json", "data/network_data_metadata.json"):
            repl["dataset_metadata_path"] = os.path.join(output_dir, "quick_network_data_metadata.json")
        if config.ood_dataset_path in ("data/v19_ood_test_data.csv", "data/ood_test_data.csv"):
            repl["ood_dataset_path"] = os.path.join(output_dir, "quick_ood_test_data.csv")
        if config.ood_metadata_path in ("data/v19_ood_test_metadata.json", "data/ood_test_metadata.json"):
            repl["ood_metadata_path"] = os.path.join(output_dir, "quick_ood_test_metadata.json")
        if config.ml_model_path in ("models/v19_link_quality_predictor.joblib", "models/link_quality_predictor.joblib"):
            repl["ml_model_path"] = os.path.join(output_dir, "quick_link_quality_predictor.joblib")
        if repl:
            config = dataclasses.replace(config, **repl)
    tracemalloc_mod = None
    if profile_memory:
        import tracemalloc as tracemalloc_mod
        tracemalloc_mod.start()

    total_start = time.time()


    print("\n--- BƯỚC 1: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH MACHINE LEARNING ---")
    ml_start = time.time()
    ml_metrics_path = os.path.join(output_dir, "ml_evaluation_metrics.json")
    predictor, ml_metrics = train_and_evaluate_ml(
        config,
        dataset_path=config.dataset_path,
        metadata_path=config.dataset_metadata_path,
        model_save_path=config.ml_model_path,
        metrics_save_path=ml_metrics_path,
        ood_dataset_path=config.ood_dataset_path,
        ood_metadata_path=config.ood_metadata_path
    )
    ml_time = time.time() - ml_start
    plot_ml_feature_importances(
        ml_metrics["feature_importances"],
        save_path=os.path.join(charts_dir, "ml_feature_importance.png"),
        model_name=ml_metrics.get("selected_model_name", "ML Surrogate")
    )


    print("\n--- BƯỚC 2: KHỞI TẠO MẠNG CẢM BIẾN IoT VÀ VẼ SƠ ĐỒ TOPOLOGY ---")
    ref_network = IoTNetwork(config, seed=seeds[0])
    routing_engine = RoutingEngine(config, ml_predictor=predictor)
    test_graph = routing_engine.build_weighted_graph(
        ref_network, use_ml=True, routing_mode="hybrid", seed=seeds[0], round_num=1
    )
    sample_paths = routing_engine.find_all_paths_to_sink(test_graph, sink_id=ref_network.bs_id)

    sample_path = []
    max_hops = 0
    for nid, p in sample_paths.items():
        if len(p) > max_hops:
            max_hops = len(p)
            sample_path = p

    plot_network_topology(ref_network, sample_path=sample_path, save_path=os.path.join(charts_dir, "topology.png"))
    if sample_path:
        print(f"[Topology] Sample multi-hop path found: {' -> '.join(map(str, sample_path))} (Hops: {len(sample_path) - 1})")


    print(f"\n--- BƯỚC 3: THỰC THI MÔ PHỎNG SO SÁNH 4 PHƯƠNG PHÁP TRÊN {len(seeds)} SEEDS ---")
    simulator = NetworkSimulator(config, ml_predictor=predictor)
    protocols = ["sp", "energy_dijkstra", "pso", "ml_pso"]

    all_trial_results: List[Dict[str, Any]] = []
    first_seed_results: Dict[str, SimulationResult] = {}
    runtime_profile: List[Dict[str, Any]] = [
        {"Stage": "ML Training & Evaluation", "Protocol": "N/A", "Runtime (s)": round(ml_time, 3)}
    ]

    for trial_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n>>> Running Trial {trial_idx + 1}/{len(seeds)} (Seed={seed}) <<<")
        trial_network = IoTNetwork(config, seed=seed)

        for proto in protocols:
            p_start = time.time()
            res = simulator.run_protocol(
                protocol=proto,
                initial_network=trial_network,
                initial_weights=None,
                pso_update_interval=config.pso_update_interval,
                show_progress=show_progress,
                seed=seed
            )
            p_duration = time.time() - p_start

            if trial_idx == 0:
                first_seed_results[proto] = res
                runtime_profile.append({
                    "Stage": "Protocol Simulation",
                    "Protocol": proto,
                    "Runtime (s)": round(p_duration, 3)
                })

            fnd_censored = (res.fnd_round == -1)
            hnd_censored = (res.hnd_round == -1)
            lnd_censored = (res.lnd_round == -1)

            all_trial_results.append({
                "Seed": seed,
                "Protocol": proto,
                "FND_numeric": res.fnd_round if not fnd_censored else config.total_rounds,
                "FND_censored": fnd_censored,
                "HND_numeric": res.hnd_round if not hnd_censored else config.total_rounds,
                "HND_censored": hnd_censored,
                "LND_numeric": res.lnd_round if not lnd_censored else config.total_rounds,
                "LND_censored": lnd_censored,
                "Final Residual Energy (J)": round(res.final_residual_energy, 4),
                "Overall PDR (%)": round(res.overall_pdr * 100.0, 2),
                "Weighted Mean Delay (ms)": round(res.weighted_mean_delay, 2),
                "Energy per Packet (mJ)": round(res.energy_per_delivered_packet_mj, 3),
                "Packets Delivered": res.total_packets_delivered,
                "Packets Generated": res.total_packets_generated,
                "Packets Dropped": res.total_packets_dropped,
                "Protocol Runtime (s)": round(p_duration, 2)
            })


    ml_pso_ref = first_seed_results.get("ml_pso")
    if ml_pso_ref is not None and ml_pso_ref.pso_convergence_curve:
        w_e, w_d, w_l, w_r = ml_pso_ref.initial_weights or (0.0, 0.0, 0.0, 0.0)
        print(f"\n[PSO] Round 1 ML-PSO weights (on actual Round 1 GraphSnapshot): "
              f"w_E={w_e:.4f}, w_D={w_d:.4f}, w_L={w_l:.4f}, w_R={w_r:.4f} | "
              f"Best Fitness={ml_pso_ref.pso_convergence_curve[-1]:.5f}")
        plot_pso_convergence(ml_pso_ref.pso_convergence_curve, save_path=os.path.join(charts_dir, "pso_convergence.png"))


    print("\n--- BƯỚC 4: XUẤT BIỂU ĐỒ SO SÁNH VÀ BẢNG TỔNG HỢP KẾT QUẢ ---")
    plot_comparative_metrics(first_seed_results, save_dir=charts_dir)
    raw_df = save_raw_results(all_trial_results, save_path=os.path.join(output_dir, "comparison_raw.csv"))
    summary_df = save_aggregated_summary_table(
        all_trial_results, config.total_rounds, save_path=os.path.join(output_dir, "comparison_summary.csv")
    )
    paired_stats = compute_paired_statistics(
        all_trial_results, save_path=os.path.join(output_dir, "comparison_paired_stats.json")
    )

    peak_mem_mb = None
    if tracemalloc_mod is not None:
        _, peak_mem = tracemalloc_mod.get_traced_memory()
        tracemalloc_mod.stop()
        peak_mem_mb = round(peak_mem / (1024 * 1024), 2)

    rss_peak = get_process_rss_mb()
    total_elapsed = time.time() - total_start

    runtime_df = pd.DataFrame(runtime_profile)
    runtime_profile_path = os.path.join(output_dir, "runtime_profile.csv")
    runtime_df.to_csv(runtime_profile_path, index=False)

    import numpy as _np
    import sklearn as _sk
    exp_config_dict = {
        "version": "V19",
        "source_code_sha256": compute_source_code_sha256(),
        "run_mode": run_mode,
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "dependency_versions": {
            "numpy": _np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": _sk.__version__,
        },
        "output_dir": output_dir,
        "charts_dir": charts_dir,
        "num_nodes": config.num_nodes,
        "area": f"{config.area_width}x{config.area_height}m",
        "comm_range": config.comm_range,
        "initial_energy": config.initial_energy,
        "total_rounds": config.total_rounds,
        "queue_service_capacity": config.queue_service_capacity,
        "max_queue_load": config.max_queue_load,
        "protocols": protocols,
        "benchmark_seeds": seeds,
        "num_trials": len(seeds),
        "model_name": ml_metrics.get("selected_model_name", "Gradient Boosting"),
        "model_path": config.ml_model_path,
        "benchmark_completed": True,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "python_traced_peak_mb": peak_mem_mb,
        "process_rss_mb": round(rss_peak, 2),
        "tracemalloc_enabled": profile_memory,
        "phy_channel": {
            "model": "Log-distance with Log-normal shadowing & IEEE 802.15.4 O-QPSK non-coherent AWGN (Annex E)",
            "frequency_hz": config.frequency_hz,
            "path_loss_exponent": config.path_loss_exponent,
            "shadowing_std_db": config.shadowing_std,
            "noise_floor_dbm": config.noise_floor_dbm,
            "receiver_noise_bandwidth_hz": config.receiver_noise_bandwidth_hz,
            "chip_rate_cps": config.chip_rate_cps,
            "bit_rate_bps": config.bandwidth_bps,
            "app_payload_bits": config.packet_size_bits,
            "frame_size_bits": config.frame_size_bits,
            "num_fragments": config.num_fragments
        }
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config_dict, f, indent=4)

    print("\n" + "=" * 105)
    print(f"            BẢNG TỔNG HỢP HIỆU SUẤT ĐỊNH TUYẾN ({len(seeds)} TRIALS, {config.total_rounds} ROUNDS)")
    print("=" * 105)
    print(summary_df.to_string(index=False))
    print("=" * 105)

    mem_msg = f"Process RSS = {rss_peak:.2f} MB"
    if peak_mem_mb is not None:
        mem_msg += f" | Python traced heap peak = {peak_mem_mb:.2f} MB"
    print(f"[*] Memory Footprint: {mem_msg}")
    print(f"[*] Complete runtime profile saved to {runtime_profile_path}")
    print(f"[Done] Toàn bộ quy trình hoàn tất trong {total_elapsed:.2f} giây!\n")

    return {
        "summary": summary_df,
        "raw": raw_df,
        "paired_stats": paired_stats,
        "first_seed_results": first_seed_results,
        "total_elapsed": total_elapsed,
        "output_dir": output_dir
    }

def main():
    parser = argparse.ArgumentParser(description="IoT WSN PSO+ML Routing Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run rapid smoke test (30 rounds, 1 seed)")
    parser.add_argument("--rounds", type=int, default=None, help="Override total simulation rounds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of random seeds")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--profile-memory", action="store_true", help="Enable tracemalloc heap profiling (slows runtime)")
    args = parser.parse_args()

    total_rounds = 30 if args.quick else (args.rounds or 300)
    pso_particles = 15 if args.quick else 25
    pso_iters = 15 if args.quick else 35
    run_mode = "quick" if args.quick else "benchmark"
    output_dir = args.output_dir or (os.path.join("results", "quick") if args.quick else os.path.join("results", "benchmark"))

    if args.quick:
        seeds = [42]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = [42, 43, 44, 45, 46]

    config = SimulationConfig(
        num_nodes=50,
        area_width=100.0,
        area_height=100.0,
        comm_range=35.0,
        initial_energy=0.25,
        total_rounds=total_rounds,
        pso_num_particles=pso_particles,
        pso_max_iter=pso_iters,
        random_seed=seeds[0]
    )

    print("=" * 80)
    print("  TỐI ƯU ĐỊNH TUYẾN VÀ TIẾT KIỆM NĂNG LƯỢNG TRONG MẠNG IoT BẰNG THUẬT TOÁN")
    print("       PARTICLE SWARM OPTIMIZATION (PSO) KẾT HỢP MACHINE LEARNING       ")
    print("=" * 80)
    print(f"[Config] Mode: {'QUICK SMOKE' if args.quick else 'BENCHMARK'}, Nodes: {config.num_nodes}, Rounds: {config.total_rounds}, Seeds: {seeds}, Output: {output_dir}")

    run_experiment(
        config=config,
        seeds=seeds,
        show_progress=not args.quick,
        profile_memory=args.profile_memory,
        output_dir=output_dir,
        run_mode=run_mode
    )

if __name__ == "__main__":
    main()
