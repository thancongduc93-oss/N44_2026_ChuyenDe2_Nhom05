import os
import time
import json
import dataclasses
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import SimulationConfig
from src.channel import WirelessChannel
from src.dataset import (
    FEATURE_COLS,
    TARGET_COLS,
    GROUP_COL,
    DATASET_VERSION,
    generate_link_dataset,
    is_dataset_valid,
    _simulate_temporal_link_traces,
)


def _build_config_signature(config: SimulationConfig) -> Dict[str, Any]:
    return {
        "comm_range": float(config.comm_range),
        "path_loss_exponent": float(config.path_loss_exponent),
        "shadowing_std": float(config.shadowing_std),
        "tx_power_dbm": float(config.tx_power_dbm),
        "noise_floor_dbm": float(config.noise_floor_dbm),
        "max_queue_load": int(config.max_queue_load),
        "packet_size_bits": int(config.packet_size_bits),
        "frame_size_bits": int(config.frame_size_bits),
        "ml_dataset_samples": int(config.ml_dataset_samples),
        "random_seed": int(config.random_seed),
    }


def _compute_feature_bounds(X_train: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute per-feature training bounds and z-score statistics for OOD detection."""
    bounds: Dict[str, Dict[str, float]] = {}
    for idx, col in enumerate(FEATURE_COLS):
        vals = X_train[:, idx]
        bounds[col] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "q01": float(np.quantile(vals, 0.01)),
            "q99": float(np.quantile(vals, 0.99)),
            "mean": float(np.mean(vals)),
            "std": float(max(1e-6, np.std(vals))),
        }
    return bounds


class LinkQualityPredictor:
    """Causal ML link-quality predictor with OOD detection and safe analytical fallback."""

    def __init__(
        self,
        pdr_model=None,
        delay_model=None,
        model_name: str = "Unknown",
        feature_cols: Optional[list] = None,
        config_signature: Optional[Dict[str, Any]] = None,
        dataset_num_samples: Optional[int] = None,
        random_seed: Optional[int] = None,
        dataset_version: Optional[str] = None,
        feature_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.pdr_model = pdr_model
        self.delay_model = delay_model
        self.model_name = model_name
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
        self.config_signature = config_signature
        self.dataset_num_samples = dataset_num_samples
        self.random_seed = random_seed
        self.dataset_version = dataset_version
        self.feature_bounds = feature_bounds

    def check_ood_batch(self, features_matrix: np.ndarray) -> np.ndarray:
        """Return boolean mask (True = sample is Out-of-Distribution or invalid)."""
        X = np.asarray(features_matrix, dtype=np.float64)
        ood = ~np.all(np.isfinite(X), axis=1)
        if self.feature_bounds is None:
            return ood

        for idx, col in enumerate(self.feature_cols):
            b = self.feature_bounds.get(col)
            if not b:
                continue
            vals = X[:, idx]
            span = max(1e-6, b["max"] - b["min"])
            margin = 0.15 * span
            low_bound = b["min"] - margin
            high_bound = b["max"] + margin
            z_dist = np.abs(vals - b["mean"]) / max(1e-6, b["std"])

            col_ood = (vals < low_bound) | (vals > high_bound) | (z_dist > 4.5)
            ood = ood | col_ood
        return ood

    def predict(
        self,
        distance: float,
        prev_rssi_dbm: float,
        prev_snr_db: float,
        ewma_pdr: float,
        ewma_retx: float,
        ewma_delay_ms: float,
        rx_queue_load: int,
    ) -> Tuple[float, float]:
        feats = np.array([[
            distance,
            prev_rssi_dbm,
            prev_snr_db,
            ewma_pdr,
            ewma_retx,
            ewma_delay_ms,
            rx_queue_load,
        ]], dtype=np.float64)
        if not np.all(np.isfinite(feats)):
            raise ValueError("Input features to LinkQualityPredictor.predict contain NaN or Inf.")

        raw_pdr = float(self.pdr_model.predict(feats)[0])
        raw_delay = float(self.delay_model.predict(feats)[0])
        if not (np.isfinite(raw_pdr) and np.isfinite(raw_delay)):
            raise ValueError(f"ML model returned non-finite prediction: pdr={raw_pdr}, delay={raw_delay}")

        pred_pdr = float(np.clip(raw_pdr, 0.0, 1.0))
        pred_delay = max(raw_delay, 0.1)
        return pred_pdr, pred_delay

    def predict_batch(self, features_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not np.all(np.isfinite(features_matrix)):
            raise ValueError("Input features_matrix to predict_batch contains NaN or Inf.")
        raw_pdrs = np.asarray(self.pdr_model.predict(features_matrix), dtype=np.float64)
        raw_delays = np.asarray(self.delay_model.predict(features_matrix), dtype=np.float64)
        if not (np.all(np.isfinite(raw_pdrs)) and np.all(np.isfinite(raw_delays))):
            raise ValueError("ML model returned NaN or Inf in predict_batch.")
        pred_pdrs = np.clip(raw_pdrs, 0.0, 1.0).astype(np.float64)
        pred_delays = np.maximum(raw_delays, 0.1).astype(np.float64)
        return pred_pdrs, pred_delays

    def predict_with_fallback(
        self,
        features_matrix: np.ndarray,
        fallback_pdrs: np.ndarray,
        fallback_delays: np.ndarray,
        force_fallback_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict link PDR & delay with automatic OOD/unobserved fallback to physical baseline.

        Returns (pred_pdrs, pred_delays, fallback_mask).
        """
        X = np.asarray(features_matrix, dtype=np.float64)
        fb_pdr = np.clip(np.asarray(fallback_pdrs, dtype=np.float64), 0.0, 1.0)
        fb_del = np.maximum(np.asarray(fallback_delays, dtype=np.float64), 0.1)

        ood_mask = self.check_ood_batch(X)
        if force_fallback_mask is not None:
            ood_mask = ood_mask | np.asarray(force_fallback_mask, dtype=bool)

        pred_pdrs = fb_pdr.copy()
        pred_delays = fb_del.copy()

        in_dist_idx = np.where(~ood_mask)[0]
        if len(in_dist_idx) > 0:
            try:
                ml_p, ml_d = self.predict_batch(X[in_dist_idx])
                invalid_out = (~np.isfinite(ml_p)) | (~np.isfinite(ml_d)) | (ml_p < 0.0) | (ml_p > 1.0) | (ml_d <= 0.0)
                valid_sub = ~invalid_out
                pred_pdrs[in_dist_idx[valid_sub]] = ml_p[valid_sub]
                pred_delays[in_dist_idx[valid_sub]] = ml_d[valid_sub]
                if np.any(invalid_out):
                    ood_mask[in_dist_idx[invalid_out]] = True
            except Exception:
                ood_mask[in_dist_idx] = True

        return pred_pdrs, pred_delays, ood_mask


def _physical_no_shadow_predict(X_batch: np.ndarray, channel: WirelessChannel) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical physical baseline using ONLY distance, noise_floor, and rx_queue_load
    (mean path-loss expectation without knowing future random shadowing).
    """
    distances = X_batch[:, 0]
    q_loads = X_batch[:, 6]

    d_clamped = np.maximum(distances, channel.d_ref)
    pl_mean = channel.pl_d_ref + 10.0 * channel.eta * np.log10(d_clamped / channel.d_ref)
    mean_rssi = channel.tx_power - pl_mean

    snr_db = mean_rssi - channel.noise_floor
    snr_lin = np.power(10.0, snr_db / 10.0)
    snr_chip = np.maximum(0.0, snr_lin * (channel.noise_bandwidth / channel.chip_rate))
    bers = channel.ber_ieee_802_15_4_oqpsk_vec(snr_chip)

    phys_pdrs = np.clip(np.power(1.0 - bers, channel.frame_bits), 0.0, 1.0)
    phys_pdrs = np.where((distances > channel.config.comm_range) | (distances < 0), 0.0, phys_pdrs)

    max_attempts = channel.max_attempts
    total_ms = np.zeros(len(distances), dtype=np.float64)
    r_j = np.ones(len(distances), dtype=np.float64)
    t_prop = (np.maximum(0.0, distances) / channel.config.propagation_speed) * 1000.0
    t_proc = channel.config.proc_delay_base_ms

    for l_j in channel.fragment_sizes:
        p_j = np.clip(np.power(1.0 - bers, l_j), 0.0, 1.0)
        q_j = 1.0 - np.power(1.0 - p_j, max_attempts)
        e_n_j = np.where(p_j < 1e-6, float(max_attempts), q_j / np.maximum(p_j, 1e-12))
        t_tx = (l_j / channel.bit_rate) * 1000.0
        d_single = t_tx + t_prop + t_proc + q_loads * (t_tx * 0.4)
        total_ms += r_j * e_n_j * d_single
        r_j *= q_j

    return phys_pdrs, total_ms


def _eval_regression_metrics(
    y_pdr_true: np.ndarray,
    y_del_true: np.ndarray,
    pred_pdr: np.ndarray,
    pred_del: np.ndarray,
) -> Dict[str, float]:
    return {
        "pdr_rmse": float(np.sqrt(mean_squared_error(y_pdr_true, pred_pdr))),
        "pdr_mae": float(mean_absolute_error(y_pdr_true, pred_pdr)),
        "pdr_r2": float(r2_score(y_pdr_true, pred_pdr)),
        "delay_rmse": float(np.sqrt(mean_squared_error(y_del_true, pred_del))),
        "delay_mae": float(mean_absolute_error(y_del_true, pred_del)),
        "delay_r2": float(r2_score(y_del_true, pred_del)),
    }


def _evaluate_baselines_on_test(
    X_test: np.ndarray,
    y_pdr_test: np.ndarray,
    y_del_test: np.ndarray,
    X_bench_batch: np.ndarray,
    config: SimulationConfig,
    warmup_runs: int = 3,
    bench_runs: int = 10,
) -> Dict[str, Any]:
    """Evaluate analytical physical baseline (no future shadowing), persistence baseline,
    and moving-average baseline on the exact same future targets.
    """
    channel = WirelessChannel(config)
    for _ in range(warmup_runs):
        _physical_no_shadow_predict(X_bench_batch, channel)

    t0 = time.time()
    for _ in range(bench_runs):
        _physical_no_shadow_predict(X_bench_batch, channel)
    phys_infer_ms = ((time.time() - t0) / bench_runs) * 1000.0

    phys_pdr, phys_del = _physical_no_shadow_predict(X_test, channel)
    phys_metrics = _eval_regression_metrics(y_pdr_test, y_del_test, phys_pdr, phys_del)
    phys_metrics["infer_1000_samples_ms"] = round(phys_infer_ms, 3)


    ewma_pdr = np.clip(X_test[:, 3], 0.0, 1.0)
    ewma_del = np.maximum(X_test[:, 5], 0.1)
    persistence_metrics = _eval_regression_metrics(y_pdr_test, y_del_test, ewma_pdr, ewma_del)


    ma_pdr = np.clip(0.5 * ewma_pdr + 0.5 * phys_pdr, 0.0, 1.0)
    ma_del = np.maximum(0.5 * ewma_del + 0.5 * phys_del, 0.1)
    ma_metrics = _eval_regression_metrics(y_pdr_test, y_del_test, ma_pdr, ma_del)

    return {
        "physical_analytical_no_shadow": phys_metrics,
        "persistence_ewma": persistence_metrics,
        "moving_average_physics_blend": ma_metrics,
    }


def train_and_evaluate_ml(
    config: Optional[SimulationConfig] = None,
    dataset_path: str = "data/network_data.csv",
    metadata_path: str = "data/network_data_metadata.json",
    model_save_path: str = "models/link_quality_predictor.joblib",
    metrics_save_path: str = "results/ml_evaluation_metrics.json",
    force_retrain: bool = False,
    ood_dataset_path: Optional[str] = None,
    ood_metadata_path: Optional[str] = None,
) -> Tuple[LinkQualityPredictor, Dict[str, Any]]:
    """Train causal ML link-quality models with topology-grouped split and OOD fallback."""
    config = config or SimulationConfig()

    if force_retrain or not os.path.exists(dataset_path):
        print(f"[ML] Generating fresh temporal link-trace dataset at {dataset_path}...")
        generate_link_dataset(
            config=config,
            num_samples=config.ml_dataset_samples,
            save_path=dataset_path,
            metadata_path=metadata_path,
            seed=config.random_seed,
        )
    elif not is_dataset_valid(
        dataset_path,
        metadata_path,
        expected_seed=config.random_seed,
        expected_samples=config.ml_dataset_samples,
        config=config,
    ):
        raise ValueError(
            f"Existing dataset at '{dataset_path}' does not match config/metadata "
            f"(expected samples={config.ml_dataset_samples}, seed={config.random_seed}, comm_range={config.comm_range}). "
            f"Pass force_retrain=True or use a separate dataset_path."
        )

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"Dataset '{dataset_path}' is empty.")
    missing_cols = [c for c in FEATURE_COLS + TARGET_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset '{dataset_path}' missing required columns: {missing_cols}")
    if df[FEATURE_COLS + TARGET_COLS].isnull().values.any() or np.isinf(df[FEATURE_COLS + TARGET_COLS].values).any():
        raise ValueError(f"Dataset '{dataset_path}' contains NaN or Inf values.")

    X = df[FEATURE_COLS].values
    y_pdr = df["pdr"].values
    y_delay = df["link_delay_ms"].values


    if GROUP_COL in df.columns and df[GROUP_COL].nunique() >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.random_seed)
        train_idx, test_idx = next(gss.split(X, y_pdr, groups=df[GROUP_COL].values))
        X_train, X_test = X[train_idx], X[test_idx]
        y_pdr_train, y_pdr_test = y_pdr[train_idx], y_pdr[test_idx]
        y_del_train, y_del_test = y_delay[train_idx], y_delay[test_idx]
        split_strategy = f"GroupShuffleSplit by {GROUP_COL} (zero topology overlap)"
        train_groups = int(df.iloc[train_idx][GROUP_COL].nunique())
        test_groups = int(df.iloc[test_idx][GROUP_COL].nunique())
    else:
        X_train, X_test, y_pdr_train, y_pdr_test, y_del_train, y_del_test = train_test_split(
            X, y_pdr, y_delay, test_size=0.2, random_state=config.random_seed
        )
        split_strategy = "Random 80/20 split"
        train_groups = 1
        test_groups = 1

    feature_bounds = _compute_feature_bounds(X_train)

    models_pdr = {
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=config.random_seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=config.ml_n_estimators,
            max_depth=config.ml_max_depth,
            random_state=config.random_seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=config.random_seed
        ),
    }

    models_delay = {
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=config.random_seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=config.ml_n_estimators,
            max_depth=config.ml_max_depth,
            random_state=config.random_seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.08, random_state=config.random_seed
        ),
    }

    pdr_comparison = {}
    delay_comparison = {}
    dummy_batch = X_test[:1000] if len(X_test) >= 1000 else X_test
    warmup_runs = 3
    bench_runs = 10

    print("\n" + "=" * 70)
    print(" MACHINE LEARNING BENCHMARK: NEXT-ROUND PDR PREDICTION (CAUSAL) ")
    print("=" * 70)
    for name, model in models_pdr.items():
        t0 = time.time()
        model.fit(X_train, y_pdr_train)
        fit_time = time.time() - t0

        for _ in range(warmup_runs):
            model.predict(dummy_batch)
        t_inf0 = time.time()
        for _ in range(bench_runs):
            model.predict(dummy_batch)
        inf_time_ms = ((time.time() - t_inf0) / bench_runs) * 1000.0

        preds_test = np.clip(model.predict(X_test), 0.0, 1.0)
        rmse = float(np.sqrt(mean_squared_error(y_pdr_test, preds_test)))
        mae = float(mean_absolute_error(y_pdr_test, preds_test))
        r2 = float(r2_score(y_pdr_test, preds_test))
        pdr_comparison[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "fit_time_s": round(fit_time, 3),
            "infer_1000_samples_ms": round(inf_time_ms, 3),
        }
        print(f"| {name:<20} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f} | Infer: {inf_time_ms:.2f}ms |")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" MACHINE LEARNING BENCHMARK: NEXT-ROUND DELAY PREDICTION (CAUSAL) ")
    print("=" * 70)
    for name, model in models_delay.items():
        t0 = time.time()
        model.fit(X_train, y_del_train)
        fit_time = time.time() - t0

        for _ in range(warmup_runs):
            model.predict(dummy_batch)
        t_inf0 = time.time()
        for _ in range(bench_runs):
            model.predict(dummy_batch)
        inf_time_ms = ((time.time() - t_inf0) / bench_runs) * 1000.0

        preds_test = np.maximum(model.predict(X_test), 0.1)
        rmse = float(np.sqrt(mean_squared_error(y_del_test, preds_test)))
        mae = float(mean_absolute_error(y_del_test, preds_test))
        r2 = float(r2_score(y_del_test, preds_test))
        delay_comparison[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "fit_time_s": round(fit_time, 3),
            "infer_1000_samples_ms": round(inf_time_ms, 3),
        }
        print(f"| {name:<20} | RMSE: {rmse:.4f} ms | MAE: {mae:.4f} ms | R²: {r2:.4f} | Infer: {inf_time_ms:.2f}ms |")
    print("=" * 70)

    baselines_test = _evaluate_baselines_on_test(
        X_test, y_pdr_test, y_del_test, dummy_batch, config,
        warmup_runs=warmup_runs, bench_runs=bench_runs,
    )
    phys_b = baselines_test["physical_analytical_no_shadow"]
    pers_b = baselines_test["persistence_ewma"]
    print(
        f"[ML] Physical Analytical (No-Future-Shadow): PDR RMSE={phys_b['pdr_rmse']:.4f} (R²={phys_b['pdr_r2']:.4f}), "
        f"Delay RMSE={phys_b['delay_rmse']:.2f}ms (Infer={phys_b['infer_1000_samples_ms']:.3f}ms)"
    )
    print(
        f"[ML] Persistence (EWMA) Baseline:           PDR RMSE={pers_b['pdr_rmse']:.4f} (R²={pers_b['pdr_r2']:.4f}), "
        f"Delay RMSE={pers_b['delay_rmse']:.2f}ms"
    )

    rf_score = pdr_comparison["Random Forest"]["RMSE"] + delay_comparison["Random Forest"]["RMSE"] / 100.0
    gb_score = pdr_comparison["Gradient Boosting"]["RMSE"] + delay_comparison["Gradient Boosting"]["RMSE"] / 100.0

    if gb_score <= rf_score * 1.05:
        selected_pdr = models_pdr["Gradient Boosting"]
        selected_delay = models_delay["Gradient Boosting"]
        selected_name = "Gradient Boosting"
    else:
        selected_pdr = models_pdr["Random Forest"]
        selected_delay = models_delay["Random Forest"]
        selected_name = "Random Forest"

    selection_reason = (
        f"{selected_name} selected on topology-grouped validation (PDR RMSE={pdr_comparison[selected_name]['RMSE']:.4f}, "
        f"Delay RMSE={delay_comparison[selected_name]['RMSE']:.4f}ms, Infer={pdr_comparison[selected_name]['infer_1000_samples_ms']:.2f}ms). "
        f"Analytical no-shadow physics achieves PDR RMSE={phys_b['pdr_rmse']:.4f} and Delay RMSE={phys_b['delay_rmse']:.2f}ms "
        f"because independent round-to-round shadowing cannot be predicted from past RSSI alone; ML acts as a causal "
        f"history-aware forecaster with OOD fallback to analytical physics."
    )
    print(f"\n[ML] Selected model: {selected_name}")
    print(f"[ML] {selection_reason}")

    predictor = LinkQualityPredictor(
        pdr_model=selected_pdr,
        delay_model=selected_delay,
        model_name=selected_name,
        feature_cols=FEATURE_COLS,
        config_signature=_build_config_signature(config),
        dataset_num_samples=len(df),
        random_seed=config.random_seed,
        dataset_version=DATASET_VERSION,
        feature_bounds=feature_bounds,
    )


    if ood_dataset_path is not None:
        ood_save_path = ood_dataset_path
    elif getattr(config, "ood_dataset_path", "data/ood_test_data.csv") != "data/ood_test_data.csv":
        ood_save_path = config.ood_dataset_path
    elif os.path.normpath(dataset_path) != os.path.normpath("data/network_data.csv"):
        ood_save_path = os.path.join(os.path.dirname(dataset_path) or ".", "ood_test_data.csv")
    else:
        ood_save_path = "data/ood_test_data.csv"

    if ood_metadata_path is not None:
        ood_meta_path = ood_metadata_path
    elif getattr(config, "ood_metadata_path", "data/ood_test_metadata.json") != "data/ood_test_metadata.json":
        ood_meta_path = config.ood_metadata_path
    elif os.path.normpath(dataset_path) != os.path.normpath("data/network_data.csv"):
        ood_meta_path = os.path.join(os.path.dirname(dataset_path) or ".", "ood_test_metadata.json")
    else:
        ood_meta_path = "data/ood_test_metadata.json"

    ood_cfg = dataclasses.replace(config, shadowing_std=8.0, random_seed=999)
    if not force_retrain and os.path.exists(ood_save_path) and is_dataset_valid(
        ood_save_path, ood_meta_path, expected_seed=999, expected_samples=1000, config=ood_cfg
    ):
        ood_df = pd.read_csv(ood_save_path)
    else:
        ood_df = generate_link_dataset(
            config=ood_cfg,
            num_samples=1000,
            save_path=ood_save_path,
            metadata_path=ood_meta_path,
            seed=999,
        )

    X_ood = ood_df[FEATURE_COLS].values
    y_ood_pdr = ood_df["pdr"].values
    y_ood_del = ood_df["link_delay_ms"].values
    ood_ch = WirelessChannel(ood_cfg)
    fb_ood_pdr, fb_ood_del = _physical_no_shadow_predict(X_ood, ood_ch)
    raw_ood_pdr, raw_ood_del = predictor.predict_batch(X_ood)
    safe_ood_pdr, safe_ood_del, ood_mask = predictor.predict_with_fallback(X_ood, fb_ood_pdr, fb_ood_del)

    ood_raw_m = _eval_regression_metrics(y_ood_pdr, y_ood_del, raw_ood_pdr, raw_ood_del)
    ood_safe_m = _eval_regression_metrics(y_ood_pdr, y_ood_del, safe_ood_pdr, safe_ood_del)


    extra_eval_scenarios: Dict[str, Any] = {}
    for sc_name, sc_cfg in [
        ("seed_holdout_777", dataclasses.replace(config, random_seed=777)),
        ("ood_sigma10_eta3_6", dataclasses.replace(config, shadowing_std=10.0, path_loss_exponent=3.6, random_seed=1001)),
        ("ood_noise_minus80", dataclasses.replace(config, noise_floor_dbm=-80.0, random_seed=1002)),
        ("ood_nodes_80", dataclasses.replace(config, num_nodes=80, area_width=125.0, area_height=125.0, base_station_pos=(62.5, 62.5), random_seed=1003)),
        ("ood_traffic_2", dataclasses.replace(config, packets_per_round=2, max_queue_load=12, random_seed=1004)),
    ]:
        n_sc = min(500, len(df))
        sc_df = _simulate_temporal_link_traces(sc_cfg, num_samples=n_sc, seed=sc_cfg.random_seed)
        sc_X = sc_df[FEATURE_COLS].values
        sc_yp = sc_df["pdr"].values
        sc_yd = sc_df["link_delay_ms"].values
        sc_ch = WirelessChannel(sc_cfg)
        fb_p, fb_d = _physical_no_shadow_predict(sc_X, sc_ch)
        raw_p, raw_d = predictor.predict_batch(sc_X)
        safe_p, safe_d, sc_ood_mask = predictor.predict_with_fallback(sc_X, fb_p, fb_d)

        raw_m = _eval_regression_metrics(sc_yp, sc_yd, raw_p, raw_d)
        safe_m = _eval_regression_metrics(sc_yp, sc_yd, safe_p, safe_d)
        phys_m = _eval_regression_metrics(sc_yp, sc_yd, fb_p, fb_d)
        extra_eval_scenarios[sc_name] = {
            "samples": len(sc_df),
            "ood_fallback_rate": round(float(np.mean(sc_ood_mask)), 4),
            "raw_ml": raw_m,
            "ml_with_fallback": safe_m,
            "physical_analytical_baseline": phys_m,
            "pdr_rmse": safe_m["pdr_rmse"],
            "pdr_r2": safe_m["pdr_r2"],
            "delay_rmse": safe_m["delay_rmse"],
            "delay_r2": safe_m["delay_r2"],
        }

    pdr_importances = {col: float(imp) for col, imp in zip(FEATURE_COLS, selected_pdr.feature_importances_)}
    delay_importances = {col: float(imp) for col, imp in zip(FEATURE_COLS, selected_delay.feature_importances_)}
    feature_target_correlations = {
        col: {
            "corr_pdr": float(np.corrcoef(df[col].values, y_pdr)[0, 1]),
            "corr_delay": float(np.corrcoef(df[col].values, y_delay)[0, 1]),
        }
        for col in FEATURE_COLS
    }

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    joblib.dump(predictor, model_save_path)
    model_size_bytes = os.path.getsize(model_save_path)
    model_size_kb = model_size_bytes / 1024.0
    print(f"[ML] Saved LinkQualityPredictor model to {model_save_path} ({model_size_bytes} bytes / {model_size_kb:.2f} KB)")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            train_meta = json.load(f)
        assert train_meta.get("random_seed") == config.random_seed, (
            f"Training metadata corrupted! Expected seed {config.random_seed}, got {train_meta.get('random_seed')}"
        )

    metrics = {
        "model_nature_note": (
            "Causal temporal ML forecaster trained on multi-topology simulation traces. "
            "Features strictly use past-window link history + static distance + pre-TX queue load; "
            "targets are next-round PDR and airtime. Includes OOD detection & analytical physics fallback."
        ),
        "split_strategy": split_strategy,
        "train_topologies": train_groups,
        "test_topologies": test_groups,
        "pdr_model_comparison": pdr_comparison,
        "delay_model_comparison": delay_comparison,
        "baselines_on_same_target": baselines_test,
        "vectorized_physical_formula_baseline": baselines_test["physical_analytical_no_shadow"],
        "selected_model_name": selected_name,
        "selected_pdr_metrics": pdr_comparison[selected_name],
        "selected_delay_metrics": delay_comparison[selected_name],
        "generalization_evaluation_ood": {
            "scenario": f"synthetic_ood shadowing_std=8.0dB, path_loss_exp={config.path_loss_exponent}, seed=999",
            "samples": len(X_ood),
            "ood_fallback_rate": round(float(np.mean(ood_mask)), 4),
            "raw_ml": ood_raw_m,
            "ml_with_fallback": ood_safe_m,
            "pdr_rmse": ood_safe_m["pdr_rmse"],
            "pdr_mae": ood_safe_m["pdr_mae"],
            "pdr_r2": ood_safe_m["pdr_r2"],
            "delay_rmse": ood_safe_m["delay_rmse"],
            "delay_mae": ood_safe_m["delay_mae"],
            "delay_r2": ood_safe_m["delay_r2"],
        },
        "additional_generalization_scenarios": extra_eval_scenarios,
        "model_selection_justification": selection_reason,
        "model_size_bytes": int(model_size_bytes),
        "model_size_kb": round(model_size_kb, 2),
        "feature_importances": pdr_importances,
        "delay_feature_importances": delay_importances,
        "feature_target_correlations": feature_target_correlations,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_seed": config.random_seed,
    }

    os.makedirs(os.path.dirname(metrics_save_path) or ".", exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[ML] Saved evaluation metrics to: {metrics_save_path}")

    return predictor, metrics


def load_predictor(
    model_path: str = "models/link_quality_predictor.joblib",
    config: Optional[SimulationConfig] = None,
    allow_retrain: bool = True,
) -> LinkQualityPredictor:
    cfg = config or SimulationConfig()
    expected_sig = _build_config_signature(cfg)

    if os.path.exists(model_path):
        try:
            pred = joblib.load(model_path)
            if not isinstance(pred, LinkQualityPredictor):
                raise ValueError("Loaded artifact is not a LinkQualityPredictor instance.")
            if getattr(pred, "feature_cols", None) != FEATURE_COLS:
                raise ValueError(f"Feature schema mismatch: {getattr(pred, 'feature_cols', None)} != {FEATURE_COLS}")
            sig = getattr(pred, "config_signature", None)
            if sig is None:
                raise ValueError("Model artifact missing config_signature metadata.")
            for k, v in expected_sig.items():
                if sig.get(k) != v:
                    raise ValueError(f"Model config mismatch on '{k}': artifact={sig.get(k)} != expected={v}")
            dummy = np.array([[15.0, -75.0, 10.0, 0.9, 0.2, 22.0, 1]], dtype=np.float64)
            pdrs, delays = pred.predict_batch(dummy)
            if not (np.all(np.isfinite(pdrs)) and np.all(np.isfinite(delays))):
                raise ValueError("Model produced NaN/Inf on validation probe.")
            return pred
        except Exception as e:
            if not allow_retrain:
                raise ValueError(f"Model artifact verification failed for {model_path}: {e}") from e
            print(f"[ML] Stored model check failed ({e}). Re-training...")
    elif not allow_retrain:
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    predictor, _ = train_and_evaluate_ml(
        config=cfg,
        dataset_path=cfg.dataset_path,
        metadata_path=cfg.dataset_metadata_path,
        model_save_path=model_path,
        force_retrain=True,
    )
    return predictor


if __name__ == "__main__":
    train_and_evaluate_ml()
