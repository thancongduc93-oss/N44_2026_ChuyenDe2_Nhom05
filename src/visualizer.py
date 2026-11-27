import os
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from src.network import IoTNetwork
from src.simulator import SimulationResult

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

COLOR_MAP = {
    "sp": "#E74C3C",
    "energy_dijkstra": "#E67E22",
    "pso": "#2980B9",
    "ml_pso": "#27AE60"
}

LABEL_MAP = {
    "sp": "Shortest Path (SP)",
    "energy_dijkstra": "Dijkstra (Energy)",
    "pso": "PSO + Dijkstra",
    "ml_pso": "ML + PSO + Dijkstra (Proposed)"
}

def plot_network_topology(
    network: IoTNetwork,
    sample_path: Optional[List[int]] = None,
    save_path: str = "results/topology.png"
):
    """Plot 2D network topology showing sensor nodes, Sink Base Station, and sample route."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))

    G = network.build_graph()

    for u, v in G.edges():
        u_node = network.nodes[u]
        v_node = network.nodes[v]
        ax.plot([u_node.x, v_node.x], [u_node.y, v_node.y], color="#BDC3C7", alpha=0.35, linewidth=0.8, zorder=1)

    sensor_xs = [n.x for n in network.nodes.values() if not n.is_base_station]
    sensor_ys = [n.y for n in network.nodes.values() if not n.is_base_station]
    sensor_ratios = [n.residual_ratio * 100.0 for n in network.nodes.values() if not n.is_base_station]

    scatter = ax.scatter(
        sensor_xs, sensor_ys,
        c=sensor_ratios, cmap="RdYlGn",
        s=120, edgecolors="black", linewidths=1.2, zorder=3,
        vmin=0, vmax=100
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Residual Energy (%)")

    bs = network.nodes[network.bs_id]
    ax.scatter(
        bs.x, bs.y,
        marker="*", s=450, color="#C0392B", edgecolors="gold", linewidths=2.0,
        zorder=5, label="Base Station (Sink)"
    )

    if sample_path and len(sample_path) > 1:
        path_xs = [network.nodes[nid].x for nid in sample_path]
        path_ys = [network.nodes[nid].y for nid in sample_path]
        ax.plot(path_xs, path_ys, color="#8E44AD", linewidth=3.0, linestyle="-", marker="o", markersize=8, zorder=4, label="Sample Multi-Hop Route")
        for i in range(len(sample_path) - 1):
            n1 = network.nodes[sample_path[i]]
            n2 = network.nodes[sample_path[i + 1]]
            ax.annotate("", xy=(n2.x, n2.y), xytext=(n1.x, n1.y),
                        arrowprops=dict(arrowstyle="-|>", color="#8E44AD", lw=2.5, mutation_scale=15))

    for nid, node in network.nodes.items():
        if node.is_base_station:
            ax.text(node.x, node.y + 2.5, "BS", fontsize=11, fontweight="bold", ha="center", color="#900C3F", zorder=6)
        else:
            ax.text(node.x, node.y - 2.8, str(nid), fontsize=8, ha="center", color="#2C3E50", zorder=4)

    ax.set_title(f"IoT Network Topology ({network.config.num_nodes} Sensors + Base Station)", fontweight="bold")
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_xlim(-2, network.config.area_width + 5)
    ax.set_ylim(-2, network.config.area_height + 5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualizer] Saved topology plot to: {save_path}")

def plot_comparative_metrics(
    results: Dict[str, SimulationResult],
    save_dir: str = "results"
):
    """Plot multi-protocol benchmark curves: residual energy, lifetime, PDR, and delay."""
    os.makedirs(save_dir, exist_ok=True)


    fig, ax = plt.subplots(figsize=(8, 5))
    for proto, res in results.items():
        color = COLOR_MAP.get(proto, "#333333")
        label = LABEL_MAP.get(proto, proto)
        ax.plot(res.rounds, res.total_residual_energy, label=label, color=color, linewidth=2.2)
    ax.set_title("Total Network Residual Energy vs. Rounds", fontweight="bold")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel("Total Remaining Energy (Joules)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "energy_consumption.png"))
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5))
    for proto, res in results.items():
        color = COLOR_MAP.get(proto, "#333333")
        label = LABEL_MAP.get(proto, proto)
        ax.plot(res.rounds, res.dead_nodes, label=label, color=color, linewidth=2.2)

        if res.fnd_round > 0:
            ax.plot(res.fnd_round, 1, marker="x", markersize=10, markeredgewidth=2.5, color=color)

    ax.set_title("Dead Nodes vs. Rounds (Network Lifetime)", fontweight="bold")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel("Dead Sensor Nodes Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "dead_nodes.png"))
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5))
    for proto, res in results.items():
        color = COLOR_MAP.get(proto, "#333333")
        label = LABEL_MAP.get(proto, proto)
        pdr_pct = [val * 100.0 for val in res.cumulative_pdr]
        ax.plot(res.rounds, pdr_pct, label=label, color=color, linewidth=2.2)
    ax.set_title("Cumulative Packet Delivery Ratio (PDR) vs. Rounds", fontweight="bold")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel("Cumulative PDR (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "packet_delivery_ratio.png"))
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8, 5))
    for proto, res in results.items():
        color = COLOR_MAP.get(proto, "#333333")
        label = LABEL_MAP.get(proto, proto)
        window = 10
        smoothed_delay = pd.Series(res.avg_delays).rolling(window, min_periods=1).mean()
        ax.plot(res.rounds, smoothed_delay, label=label, color=color, linewidth=2.0)
    ax.set_title("Average End-to-End Latency vs. Rounds", fontweight="bold")
    ax.set_xlabel("Simulation Round")
    ax.set_ylabel("Latency (milliseconds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "end_to_end_delay.png"))
    plt.close(fig)

    print(f"[Visualizer] Generated comparative evaluation charts in {save_dir}/")

def plot_pso_convergence(
    convergence_curve: List[float],
    save_path: str = "results/pso_convergence.png"
):
    """Plot PSO swarm global-best fitness convergence trajectory."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    iters = list(range(1, len(convergence_curve) + 1))
    ax.plot(iters, convergence_curve, marker="o", markersize=4, color="#2980B9", linewidth=2.0, label="Global Best Fitness")
    ax.set_title("Particle Swarm Optimization (PSO) Convergence Curve", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Multi-Objective Cost Fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualizer] Saved PSO convergence plot to: {save_path}")

def plot_ml_feature_importances(
    feature_importances: Dict[str, float],
    save_path: str = "results/ml_feature_importance.png",
    model_name: str = "ML Surrogate"
):
    """Plot ML surrogate feature importances for wireless link prediction."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]

    bars = ax.barh(names, values, color="#3498DB", edgecolor="#2980B9")
    ax.set_title(f"{model_name} Feature Importances (Link Quality)", fontweight="bold")
    ax.set_xlabel("Relative Importance")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualizer] Saved ML feature importance plot to: {save_path}")

def save_aggregated_summary_table(
    raw_trials: List[Dict[str, Any]],
    total_rounds: int,
    save_path: str = "results/comparison_summary.csv"
) -> pd.DataFrame:
    """Export summary metrics table aggregated across all benchmark trial seeds."""
    df_raw = pd.DataFrame(raw_trials)
    records = []

    protocols = ["sp", "energy_dijkstra", "pso", "ml_pso"]

    for proto in protocols:
        sub = df_raw[df_raw["Protocol"] == proto]
        if sub.empty:
            continue

        n_trials = len(sub)
        p_name = LABEL_MAP.get(proto, proto)


        fnds = sub["FND_numeric"].values
        censored_fnd_count = int(np.sum(sub["FND_censored"].values))

        if censored_fnd_count == n_trials:
            fnd_display = f">{total_rounds} ({n_trials}/{n_trials} censored)"
        elif censored_fnd_count == 0:
            fnd_display = f"{np.mean(fnds):.1f} ± {np.std(fnds):.1f}"
        else:
            uncensored = fnds[~sub["FND_censored"].values]
            fnd_display = f"{np.mean(uncensored):.1f} ± {np.std(uncensored):.1f} ({censored_fnd_count}/{n_trials} >{total_rounds})"


        hnds = sub["HND_numeric"].values
        censored_hnd_count = int(np.sum(sub["HND_censored"].values))
        if censored_hnd_count == n_trials:
            hnd_display = f">{total_rounds} ({n_trials}/{n_trials} censored)"
        elif censored_hnd_count == 0:
            hnd_display = f"{np.mean(hnds):.1f} ± {np.std(hnds):.1f}"
        else:
            uncensored_hnd = hnds[~sub["HND_censored"].values]
            hnd_display = f"{np.mean(uncensored_hnd):.1f} ± {np.std(uncensored_hnd):.1f} ({censored_hnd_count}/{n_trials} >{total_rounds} censored)"


        pdr_mean = float(np.mean(sub["Overall PDR (%)"]))
        pdr_std = float(np.std(sub["Overall PDR (%)"]))

        delay_mean = float(np.mean(sub["Weighted Mean Delay (ms)"]))
        delay_std = float(np.std(sub["Weighted Mean Delay (ms)"]))

        energy_mean = float(np.mean(sub["Final Residual Energy (J)"]))
        energy_std = float(np.std(sub["Final Residual Energy (J)"]))

        epkt_mean = float(np.mean(sub["Energy per Packet (mJ)"]))
        epkt_std = float(np.std(sub["Energy per Packet (mJ)"]))

        deliv_mean = int(round(np.mean(sub["Packets Delivered"])))
        gen_mean = int(round(np.mean(sub["Packets Generated"])))
        drop_mean = int(round(np.mean(sub["Packets Dropped"])))

        proto_runtime_mean = float(np.mean(sub["Protocol Runtime (s)"]))

        records.append({
            "Protocol": p_name,
            "FND (Round)": fnd_display,
            "HND (Round)": hnd_display,
            "Overall PDR (%)": f"{pdr_mean:.2f} ± {pdr_std:.2f}",
            "Weighted Mean Delay (ms)": f"{delay_mean:.2f} ± {delay_std:.2f}",
            "Final Residual Energy (J)": f"{energy_mean:.4f} ± {energy_std:.4f}",
            "Energy per Packet (mJ)": f"{epkt_mean:.3f} ± {epkt_std:.3f}",
            "Packets Delivered": f"{deliv_mean} / {gen_mean}",
            "Packets Dropped": drop_mean,
            "Protocol Runtime (s)": round(proto_runtime_mean, 2),
            "Trials": n_trials
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Visualizer] Saved aggregated summary table to: {save_path}")
    return df

def save_raw_results(
    raw_trials: List[Dict[str, Any]],
    save_path: str = "results/comparison_raw.csv"
) -> pd.DataFrame:
    """Export raw per-trial benchmark results for statistical analysis."""
    df = pd.DataFrame(raw_trials)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Visualizer] Saved raw trial data to: {save_path}")
    return df


def compute_paired_statistics(
    raw_trials: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Compute paired seed-by-seed statistical comparisons with Holm correction.

    - Right-censored survival milestones (FND/HND/LND) are NEVER fed into paired t-tests
      with imputed censoring thresholds; censoring counts and status are reported explicitly.
    - Continuous metrics use paired t-test when differences have non-degenerate variance;
      zero or near-constant differences are marked without triggering SciPy precision warnings.
    - Holm-Bonferroni correction is applied across all testable continuous comparisons.
    """
    import json
    from scipy import stats

    df = pd.DataFrame(raw_trials)
    continuous_metrics = [
        "Overall PDR (%)",
        "Weighted Mean Delay (ms)",
        "Final Residual Energy (J)",
        "Energy per Packet (mJ)",
    ]
    milestone_metrics = [
        ("FND", "FND_numeric", "FND_censored"),
        ("HND", "HND_numeric", "HND_censored"),
        ("LND", "LND_numeric", "LND_censored"),
    ]
    pairs = [("ml_pso", "pso"), ("ml_pso", "sp"), ("pso", "sp"), ("ml_pso", "energy_dijkstra")]
    report: Dict[str, Any] = {
        "statistical_method": "Paired t-test across matched seeds with Holm-Bonferroni multiple-testing correction",
        "pairs": {},
    }

    testable_refs = []

    for p_a, p_b in pairs:
        sub_a = df[df["Protocol"] == p_a].sort_values("Seed")
        sub_b = df[df["Protocol"] == p_b].sort_values("Seed")
        merged = pd.merge(sub_a, sub_b, on="Seed", suffixes=("_a", "_b"))
        if len(merged) < 2:
            continue

        n_seeds = int(len(merged))
        pair_key = f"{p_a}_vs_{p_b}"
        report["pairs"][pair_key] = {"n_seeds": n_seeds, "metrics": {}, "censored_milestones": {}}

        for m in continuous_metrics:
            a_vals = merged[f"{m}_a"].values.astype(np.float64)
            b_vals = merged[f"{m}_b"].values.astype(np.float64)
            if np.isnan(a_vals).any() or np.isnan(b_vals).any():
                report["pairs"][pair_key]["metrics"][m] = {
                    "status": "not_estimable_missing_values",
                    "paired_t_p_value": None,
                    "p_value_holm": None,
                    "significant_holm_alpha_0_05": False,
                }
                continue

            diffs = a_vals - b_vals
            mean_d = float(np.mean(diffs))
            std_d = float(np.std(diffs, ddof=1))

            if std_d < 1e-12:
                status = "constant_zero_diff" if abs(mean_d) < 1e-12 else "constant_non_zero_diff"
                report["pairs"][pair_key]["metrics"][m] = {
                    "status": status,
                    "mean_diff": round(mean_d, 4),
                    "std_diff": 0.0,
                    "ci95_low": round(mean_d, 4),
                    "ci95_high": round(mean_d, 4),
                    "paired_t_p_value": None,
                    "p_value_holm": None,
                    "significant_holm_alpha_0_05": False,
                }
                continue

            se_d = std_d / np.sqrt(n_seeds)
            t_crit = float(stats.t.ppf(0.975, df=n_seeds - 1))
            ci_low = mean_d - t_crit * se_d
            ci_high = mean_d + t_crit * se_d
            _, p_val = stats.ttest_rel(a_vals, b_vals)
            p_val = float(p_val) if not np.isnan(p_val) else None

            entry = {
                "status": "tested" if p_val is not None else "not_estimable",
                "mean_diff": round(mean_d, 4),
                "std_diff": round(std_d, 4),
                "ci95_low": round(ci_low, 4),
                "ci95_high": round(ci_high, 4),
                "paired_t_p_value": round(p_val, 6) if p_val is not None else None,
                "p_value_holm": None,
                "significant_alpha_0_05": bool(p_val < 0.05) if p_val is not None else False,
                "significant_holm_alpha_0_05": False,
            }
            report["pairs"][pair_key]["metrics"][m] = entry
            if p_val is not None:
                testable_refs.append((pair_key, m, p_val))


        for label, num_col, cens_col in milestone_metrics:
            if f"{cens_col}_a" in merged.columns and f"{cens_col}_b" in merged.columns:
                c_a = int(np.sum(merged[f"{cens_col}_a"].values))
                c_b = int(np.sum(merged[f"{cens_col}_b"].values))
                if c_a > 0 or c_b > 0:
                    report["pairs"][pair_key]["censored_milestones"][label] = {
                        "status": "not_estimable_right_censored",
                        f"{p_a}_censored": f"{c_a}/{n_seeds} right-censored",
                        f"{p_b}_censored": f"{c_b}/{n_seeds} right-censored",
                        "paired_t_p_value": None,
                        "note": "Right-censored milestone excluded from paired t-test."
                    }
                else:
                    a_vals = merged[f"{num_col}_a"].values.astype(np.float64)
                    b_vals = merged[f"{num_col}_b"].values.astype(np.float64)
                    diffs = a_vals - b_vals
                    mean_d = float(np.mean(diffs))
                    std_d = float(np.std(diffs, ddof=1))
                    if std_d < 1e-12:
                        report["pairs"][pair_key]["censored_milestones"][label] = {
                            "status": "constant_diff_not_testable",
                            "mean_diff": round(mean_d, 4),
                            "paired_t_p_value": None,
                        }
                    else:
                        _, p_val = stats.ttest_rel(a_vals, b_vals)
                        report["pairs"][pair_key]["censored_milestones"][label] = {
                            "status": "tested_uncensored",
                            "mean_diff": round(mean_d, 4),
                            "std_diff": round(std_d, 4),
                            "paired_t_p_value": round(float(p_val), 6),
                        }


    if testable_refs:
        m_tests = len(testable_refs)
        sorted_refs = sorted(testable_refs, key=lambda x: x[2])
        running_max = 0.0
        for rank, (pk, mn, pv) in enumerate(sorted_refs):
            adj = min(1.0, pv * (m_tests - rank))
            running_max = max(running_max, adj)
            entry = report["pairs"][pk]["metrics"][mn]
            entry["p_value_holm"] = round(running_max, 6)
            entry["significant_holm_alpha_0_05"] = bool(running_max < 0.05)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"[Visualizer] Saved paired statistical comparisons to: {save_path}")

    return report
