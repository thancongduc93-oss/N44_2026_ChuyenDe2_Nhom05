import time
import os
import sys
import json
import datetime
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.config import SimulationConfig
from src.network import IoTNetwork
from src.ml_predictor import load_predictor
from src.simulator import NetworkSimulator
from src.visualizer import COLOR_MAP, LABEL_MAP


def plot_scalability_results(
    df: pd.DataFrame,
    node_counts: list,
    area_map: dict,
    rounds_per_test: int,
    save_path: str = "results/scalability_analysis.png"
):
    """Generate 2x2 scalability comparison figure without clipping low PDR values
    and using dynamic ``rounds_per_test`` in the FND title.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.rcParams.update({"font.size": 10})

    df_sp = df[df["Protocol_Key"] == "sp"]
    df_ed = df[df["Protocol_Key"] == "energy_dijkstra"]
    df_pso = df[df["Protocol_Key"] == "pso"]
    df_prop = df[df["Protocol_Key"] == "ml_pso"]

    x = np.arange(len(node_counts))
    width = 0.2
    protocols_to_test = ["sp", "energy_dijkstra", "pso", "ml_pso"]


    ax = axes[0, 0]
    bar_sp = ax.bar(x - 1.5 * width, df_sp["FND_numeric"], width, label="Shortest Path", color=COLOR_MAP["sp"])
    bar_ed = ax.bar(x - 0.5 * width, df_ed["FND_numeric"], width, label="Dijkstra (Energy)", color=COLOR_MAP["energy_dijkstra"])
    bar_pso = ax.bar(x + 0.5 * width, df_pso["FND_numeric"], width, label="PSO + Dijkstra", color=COLOR_MAP["pso"])
    bar_prop = ax.bar(x + 1.5 * width, df_prop["FND_numeric"], width, label="ML-PSO (Proposed)", color=COLOR_MAP["ml_pso"])

    for bars, sub_df in zip([bar_sp, bar_ed, bar_pso, bar_prop], [df_sp, df_ed, df_pso, df_prop]):
        for bar, censored in zip(bars, sub_df["FND_censored"]):
            if censored:
                bar.set_hatch("//")
                bar.set_edgecolor("black")

    ax.set_title(f"First Node Dead Round (Hatched = Censored >{rounds_per_test})", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} Nodes\n({int(area_map[n])}x{int(area_map[n])}m)" for n in node_counts])
    ax.set_ylabel("Round Number")
    ax.set_ylim(0, rounds_per_test * 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)


    ax = axes[0, 1]
    ax.bar(x - 1.5 * width, df_sp["PDR (%)"], width, label="Shortest Path", color=COLOR_MAP["sp"])
    ax.bar(x - 0.5 * width, df_ed["PDR (%)"], width, label="Dijkstra (Energy)", color=COLOR_MAP["energy_dijkstra"])
    ax.bar(x + 0.5 * width, df_pso["PDR (%)"], width, label="PSO + Dijkstra", color=COLOR_MAP["pso"])
    ax.bar(x + 1.5 * width, df_prop["PDR (%)"], width, label="ML-PSO (Proposed)", color=COLOR_MAP["ml_pso"])
    ax.set_title("Packet Delivery Ratio (PDR %)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} Nodes" for n in node_counts])
    ax.set_ylabel("PDR (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)


    ax = axes[1, 0]
    for proto in protocols_to_test:
        sub = df[df["Protocol_Key"] == proto]
        ax.plot(node_counts, sub["Weighted Mean Delay (ms)"], marker="o", label=LABEL_MAP[proto], color=COLOR_MAP[proto], linewidth=2.0)
    ax.set_title("Average End-to-End Latency vs. Network Size", fontweight="bold")
    ax.set_xlabel("Number of Sensor Nodes (Constant Density ~0.005 nodes/m²)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(node_counts)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)


    ax = axes[1, 1]
    for proto in protocols_to_test:
        sub = df[df["Protocol_Key"] == proto]
        ax.plot(node_counts, sub["Runtime (s)"], marker="^", label=LABEL_MAP[proto], color=COLOR_MAP[proto], linewidth=2.0)
    ax.set_title("Simulation Execution Time vs. Network Size", fontweight="bold")
    ax.set_xlabel("Number of Sensor Nodes")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_xticks(node_counts)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig


def evaluate_scalability(quick_mode: bool = False, output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join("results", "scalability_quick" if quick_mode else os.path.join("scalability", "benchmark"))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("      SCALABILITY ANALYSIS: SCALING SENSOR NETWORK SIZE (N = 30, 50, 100)      ")
    print(f"      Mode: {'QUICK' if quick_mode else 'BENCHMARK'} | Output: {output_dir}")
    print("=" * 80)

    node_counts = [30, 50, 100]
    rounds_per_test = 40 if quick_mode else 150
    comm_range = 35.0
    base_ml_config = SimulationConfig(comm_range=comm_range, random_seed=42)
    predictor = load_predictor(
        model_path=base_ml_config.ml_model_path,
        config=base_ml_config,
        allow_retrain=False
    )

    results_data = []
    area_map = {30: 80.0, 50: 100.0, 100: 140.0}
    protocols_to_test = ["sp", "energy_dijkstra", "pso", "ml_pso"]
    size_runtimes = {}
    total_start = time.time()

    for num_nodes in node_counts:
        size_start = time.time()
        area_size = area_map[num_nodes]
        bs_pos = (area_size / 2.0, area_size / 2.0)
        density = num_nodes / (area_size ** 2)
        print(f"\n---> Quy mô N = {num_nodes} nút (Vùng: {area_size}x{area_size}m, BS: {bs_pos}, Mật độ: {density:.5f} nodes/m², R_comm: {comm_range}m, Rounds: {rounds_per_test})...")

        config = SimulationConfig(
            num_nodes=num_nodes,
            area_width=area_size,
            area_height=area_size,
            base_station_pos=bs_pos,
            comm_range=comm_range,
            initial_energy=0.25,
            total_rounds=rounds_per_test,
            pso_num_particles=15 if quick_mode else 20,
            pso_max_iter=15 if quick_mode else 30,
            random_seed=42
        )

        network = IoTNetwork(config)
        simulator = NetworkSimulator(config, ml_predictor=predictor)

        for proto in protocols_to_test:
            proto_name = LABEL_MAP.get(proto, proto)
            print(f"     [N={num_nodes}] Running {proto_name} ({rounds_per_test} rounds)...", flush=True)
            start_t = time.time()
            res = simulator.run_protocol(
                protocol=proto,
                initial_network=network,
                pso_update_interval=20,
                show_progress=False
            )
            runtime = time.time() - start_t

            is_censored = (res.fnd_round == -1)
            fnd_val = res.fnd_round if not is_censored else rounds_per_test
            fnd_str = str(res.fnd_round) if not is_censored else f">{rounds_per_test}"

            results_data.append({
                "Nodes": num_nodes,
                "Area (m^2)": f"{int(area_size)}x{int(area_size)}",
                "BS_Pos": f"({bs_pos[0]:.1f}, {bs_pos[1]:.1f})",
                "Comm_Range (m)": comm_range,
                "Density (nodes/m^2)": round(density, 5),
                "Protocol": proto_name,
                "Protocol_Key": proto,
                "FND": fnd_str,
                "FND_numeric": fnd_val,
                "FND_censored": is_censored,
                "PDR (%)": round(res.overall_pdr * 100.0, 2),
                "Weighted Mean Delay (ms)": round(res.weighted_mean_delay, 2),
                "Runtime (s)": round(runtime, 2),
                "Packets Delivered": res.total_packets_delivered,
                "Packets Generated": res.total_packets_generated
            })
            print(f"     * [{proto_name:<30}] FND: {fnd_str:<6} | PDR: {res.overall_pdr * 100.0:.2f}% | Delay: {res.weighted_mean_delay:.2f}ms | Runtime: {runtime:.2f}s")

        size_elapsed = time.time() - size_start
        size_runtimes[str(num_nodes)] = round(size_elapsed, 2)
        print(f"     [N={num_nodes}] Total node-size elapsed: {size_elapsed:.2f}s")

    total_elapsed = time.time() - total_start
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "scalability_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Scalability] Saved summary table to {csv_path}")

    png_path = os.path.join(charts_dir, "scalability_analysis.png")
    plot_scalability_results(df, node_counts, area_map, rounds_per_test, save_path=png_path)
    print(f"[Scalability] Saved publication plot to {png_path}")

    meta_path = os.path.join(output_dir, "experiment_config.json")
    from main import compute_source_code_sha256
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "V19",
            "source_code_sha256": compute_source_code_sha256(),
            "run_mode": "scalability_quick" if quick_mode else "scalability_benchmark",
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "output_dir": output_dir,
            "charts_dir": charts_dir,
            "node_counts": node_counts,
            "rounds_per_test": rounds_per_test,
            "comm_range": comm_range,
            "seed": 42,
            "single_seed_note": "Single-seed (seed=42) scalability experiment; not a multi-seed average.",
            "protocols": protocols_to_test,
            "pso_num_particles": 15 if quick_mode else 20,
            "pso_max_iter": 15 if quick_mode else 30,
            "size_runtimes_s": size_runtimes,
            "total_elapsed_s": round(total_elapsed, 2),
            "benchmark_completed": True
        }, f, indent=4)
    print(f"[Scalability] Saved metadata to {meta_path} (Total: {total_elapsed:.2f}s)\n")


def main():
    parser = argparse.ArgumentParser(description="IoT WSN Scalability Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run rapid scalability test (40 rounds)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()
    evaluate_scalability(quick_mode=args.quick, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
