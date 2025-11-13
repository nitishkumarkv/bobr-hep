# 3d_script.py
from bobr_hep.utils.data_generation import generate_toy_data_3class
from bobr_hep.utils.plotting import plot_stacked_histograms
from bobr_hep.bobr import bobr_gmm, equidistant, bobr_1d

import os
import argparse
import json
import numpy as np
import pandas as pd
import hist
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

def create_hist(df, var=None, class_num=None, bin_edges=None):
    """Build a hist.Hist for either vector scores, scalar column, or bin_index.

    Use ONLY for continuous scalar variables (e.g. NN_output component, max_score_sel).
    For 'bin_index' use create_binindex_hist instead.
    """
    if var is None:
        var = "NN_output"

    if bin_edges is None:
        h = hist.Hist(hist.axis.Regular(50, 0, 1, name=var), storage=hist.storage.Weight())
    else:
        h = hist.Hist(hist.axis.Variable(bin_edges, name=var), storage=hist.storage.Weight())

    if class_num is None and var == "NN_output":
        raise ValueError("For var='NN_output' you must provide class_num (0, 1, or 2)")
    if class_num is not None and var == "NN_output":
        h.fill(np.stack(df[var])[:, class_num], weight=df["weight"])
    else:
        h.fill(df[var], weight=df["weight"])
    return h


def create_binindex_hist(df: pd.DataFrame, n_bins: int, var: str = "bin_index"):
    """Histogram discrete bin indices [0, n_bins) using an Integer axis.

    Coerces the column to integers (dropping NaNs) to avoid edge/gap issues.
    """
    idx = pd.to_numeric(df[var], errors="coerce").to_numpy()
    idx = idx[~np.isnan(idx)].astype(int)
    h = hist.Hist(hist.axis.Integer(0, n_bins, name=var), storage=hist.storage.Weight())
    h.fill(idx, weight=df["weight"].to_numpy())
    return h


def make_json_serializable(obj):
    """Recursively convert numpy types and arrays to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        return float(obj)
    return obj


def combine_two_z(z1: float, z2: float, method: str = "geometric") -> float:
    """Combine two per-signal Z values using the same rule as bobr_gmm."""
    z1 = float(z1) if z1 is not None else 0.0
    z2 = float(z2) if z2 is not None else 0.0
    if method == "quadrature":
        return float(np.sqrt(z1 * z1 + z2 * z2))
    if method == "harmonic":
        if z1 > 0 and z2 > 0:
            return float(2.0 / (1.0 / z1 + 1.0 / z2))
        return 0.0
    if method == "max":
        return float(max(z1, z2))
    # default geometric
    if z1 > 0 and z2 > 0:
        return float(np.sqrt(z1 * z2))
    return 0.0


def save_bin_metrics(out_dir: str, optimizer, prefix: str):
    """
    Save per-bin background yields and relative uncertainties to CSV and JSON.

    Expects optimizer.best_metrics with keys:
      - 'bkg_counts': np.ndarray
      - 'rel_unc_bkg': np.ndarray
      - optional 'per_signal' dict with 'combined_Z'
      - 'combined_Z': float
    """
    os.makedirs(out_dir, exist_ok=True)
    bm = getattr(optimizer, "best_metrics", {}) or {}
    bkg = np.asarray(bm.get("bkg_counts", []), dtype=float)
    rel = np.asarray(bm.get("rel_unc_bkg", []), dtype=float)

    # CSV
    df = pd.DataFrame({
        "bin": np.arange(len(bkg), dtype=int),
        "bkg_counts": bkg,
        "rel_unc_bkg": rel,
    })
    csv_path = os.path.join(out_dir, f"{prefix}_bin_metrics.csv")
    df.to_csv(csv_path, index=False)

    # JSON (include combined_Z and per-signal)
    js = {
        "combined_Z": make_json_serializable(bm.get("combined_Z")),
        "per_signal": make_json_serializable(bm.get("per_signal")),
        "bkg_counts": make_json_serializable(bkg),
        "rel_unc_bkg": make_json_serializable(rel),
        "penalties": make_json_serializable(bm.get("penalties", {})),
    }
    json_path = os.path.join(out_dir, f"{prefix}_bin_metrics.json")
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2)

    # console summary
    print(f"[{prefix}] per-bin background yields:", bkg)
    print(f"[{prefix}] per-bin relative uncertainties:", rel)
    print(f"[{prefix}] combined_Z:", bm.get("combined_Z"))


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(description="Run the 3D toy example (GMM vs argmax-1D)")
    parser.add_argument("--output-dir", type=str, default="toy_results_3d", help="Output directory")
    parser.add_argument("--plot-toy-data", action="store_true", help="Plot toy data")
    parser.add_argument("--run-bobr", action="store_true", help="Run BOBR (GMM) optimizer")
    parser.add_argument("--run-equidistant", action="store_true", help="Run equidistant and 1D BOBR on argmax subsets")
    parser.add_argument("--nbins", type=int, nargs="+", default=[3, 5, 10, 20], help="List of numbers of bins/components to test")
    parser.add_argument("--n-trials", type=int, default=300, help="Optuna trials per nbins")
    parser.add_argument("--restart-check-trials", type=int, default=50, help="Trials before a beta-halving restart check")
    parser.add_argument("--n-bkg", type=int, default=500000, help="Number of background events for toy data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--beta", type=float, default=0.25, help="Initial beta for beta-halving restarts")

    # penalties/thresholds consistent with 1D script
    parser.add_argument("--min-bkg-per-bin", type=int, default=1, help="Minimum background per bin (penalty threshold)")
    parser.add_argument("--penalty-low-lambda", type=float, default=10.0, help="Weight for low-background penalty")
    parser.add_argument("--penalty-unc-lambda", type=float, default=0, help="Weight for relative-uncertainty penalty")
    parser.add_argument("--rel-unc-threshold", type=float, default=0.07, help="Relative uncertainty threshold")

    # which NN_output coordinates to optimize on (defaults to 0,1 = signals)
    parser.add_argument("--dims", type=str, default="0,1,2",
                        help="Comma-separated dims of NN_output to optimise on (default '0,1')")
    # how to combine S1/S2 for the argmax methods (match GMM default)
    parser.add_argument("--combine", type=str, default="geometric",
                        choices=["geometric", "quadrature", "harmonic", "max"],
                        help="How to combine S1 and S2 Z for argmax methods")

    parser.add_argument("--assign-mode", type=str, default="hard",
                        choices=["hard", "soft"],
                        help="Assignment used in optimization and metrics (default: hard)")

    args = parser.parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, "toy_data")
    os.makedirs(data_path, exist_ok=True)

    # Labels
    bkg_list = ["bkg1", "bkg2", "bkg3", "bkg4", "bkg5"]
    signal_list = ["signal1", "signal2"]

    # Parse dims_to_use
    dims_to_use = [int(s.strip()) for s in args.dims.split(",") if s.strip()]
    if not dims_to_use:
        dims_to_use = [0, 1]  # fallback
    for d in dims_to_use:
        if d not in (0, 1, 2):
            raise ValueError("--dims must pick among 0,1,2")

    # --- Generate toy data (dict of DataFrames) ---
    toy_data = generate_toy_data_3class(seed=args.seed, n_bkg=args.n_bkg)

    # Quick plots per coordinate (before any selection)
    if args.plot_toy_data:
        for i in (0, 1, 2):
            bkg_hists = [create_hist(toy_data[bkg], var="NN_output", class_num=i) for bkg in bkg_list]
            signal_hists = [create_hist(toy_data[signal], var="NN_output", class_num=i) for signal in signal_list]
            plot_stacked_histograms(
                bkg_hists, bkg_list,
                output_filename=f"{data_path}/plot_log_score_{i}.pdf",
                axis_labels=(f"NN output score {i}", "Events"),
                signal_hists=signal_hists, signal_labels=signal_list,
                signal_scale=100, normalize=False, log=True, log_min=None,
            )
            plot_stacked_histograms(
                bkg_hists, bkg_list,
                output_filename=f"{data_path}/plot_score_{i}.pdf",
                axis_labels=(f"NN output score {i}", "Events"),
                signal_hists=signal_hists, signal_labels=signal_list,
                signal_scale=100, normalize=False, log=False, log_min=None,
            )

    # Containers for summary plot across nbins
    nbins_checked = []
    gmm_Zs = []            # combined Z from GMM
    equi_combined_Zs = []  # combined Z for Argmax+Equidistant
    bobr1_combined_Zs = [] # combined Z for Argmax+BOBR-1D
    summary = {}

    for n_bins in args.nbins:
        print(f"\n=== Running n_bins={n_bins} ===")
        nb_dir = os.path.join(args.output_dir, f"n_{n_bins}")
        os.makedirs(nb_dir, exist_ok=True)

        # ---------------------------
        # 1) Multi-D BOBR (bobr_gmm)
        # ---------------------------
        gmm_dir = os.path.join(nb_dir, "gmm")
        os.makedirs(gmm_dir, exist_ok=True)

        optimizer = bobr_gmm(
            toy_data,
            bkg_label_lst=bkg_list,
            signal_label_lst=signal_list,
            var_label="NN_output",
            weight_label="weight",
            n_components=n_bins,       # mapping: n_bins ↔ components
            dims_to_use=dims_to_use,
            n_trials=int(args.n_trials),
            output_dir=gmm_dir,
            combination="geometric",   # match combine default
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            restart_check_trials=args.restart_check_trials,
            beta=args.beta,
            #min_edge=-0.2,
            #max_edge=1.2,
        )

        best_components, best_hist, best_score = (None, None, None)
        gmm_Z = None
        if args.run_bobr:
            best_components, best_hist, best_score = optimizer.run()
            # prefer stored combined_Z (unpenalized)
            try:
                gmm_Z = float(optimizer.best_metrics.get("combined_Z", best_score))
            except Exception:
                gmm_Z = float(best_score) if best_score is not None else float("nan")
            gmm_Zs.append(gmm_Z)

            optimizer.visualize_optimization()
            #optimizer.visualize_labelled_ellipses()
            # Always draw the 3 simplex-pair plots (01,02,12) irrespective of dims_to_use
            optimizer.visualize_bin_boundaries()

            # Plot stacked histograms OVER ASSIGNED bin_index (discrete Integer axis)
            best_signal_hists = [create_binindex_hist(toy_data[s], n_bins, var="bin_index") for s in signal_list]
            best_bkg_hists = [create_binindex_hist(toy_data[b], n_bins, var="bin_index") for b in bkg_list]
            plot_stacked_histograms(
                best_bkg_hists, bkg_list,
                output_filename=os.path.join(gmm_dir, "plot_log.pdf"),
                axis_labels=("Bin index", "Events"),
                signal_hists=best_signal_hists, signal_labels=signal_list,
                signal_scale=100, normalize=False, log=True, log_min=None,
            )
            plot_stacked_histograms(
                best_bkg_hists, bkg_list,
                output_filename=os.path.join(gmm_dir, "plot.pdf"),
                axis_labels=("Bin index", "Events"),
                signal_hists=best_signal_hists, signal_labels=signal_list,
                signal_scale=100, normalize=False, log=False, log_min=None,
            )

            # Save per-bin background yields & rel uncertainty
            save_bin_metrics(gmm_dir, optimizer, prefix="gmm")

            # summary
            summary.setdefault(str(n_bins), {})
            summary[str(n_bins)]["gmm_best_Z"] = gmm_Z
            summary[str(n_bins)]["gmm_components"] = make_json_serializable(best_components)

        # ----------------------------------------------------------------
        # 2) Argmax(selected dims) → 1D methods on S1/S2 subsets
        # ----------------------------------------------------------------
        def _max_on_selected(vec):
            arr = np.asarray(vec, dtype=float)
            return float(np.max(arr[dims_to_use]))

        def _argmax_on_selected(vec):
            arr = np.asarray(vec, dtype=float)
            return int(np.argmax(arr[dims_to_use]))  # 0 or 1 typically

        toy_data_with_sel = {}
        for proc, df in toy_data.items():
            df2 = df.copy()
            df2["max_score_sel"] = df2["NN_output"].apply(_max_on_selected)
            df2["argmax_sel"] = df2["NN_output"].apply(_argmax_on_selected)
            toy_data_with_sel[proc] = df2

        toy_data_s1 = {k: v[v["argmax_sel"] == 0].copy() for k, v in toy_data_with_sel.items()}
        toy_data_s2 = {k: v[v["argmax_sel"] == 1].copy() for k, v in toy_data_with_sel.items()}

        # ------- Equidistant on S1 -------
        equi_s1_dir = os.path.join(nb_dir, "equidistant_s1")
        os.makedirs(equi_s1_dir, exist_ok=True)
        equi_s1 = equidistant(
            toy_data_s1,
            bkg_label_lst=bkg_list,
            signal_label_lst=["signal1"],
            var_label="max_score_sel",
            weight_label="weight",
            n_bins=n_bins,
            output_dir=equi_s1_dir,
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            min_edge=0.3
        )
        best_bins_eq_s1 = best_hist_eq_s1 = best_Z_eq_s1 = None
        if args.run_equidistant:
            best_bins_eq_s1, best_hist_eq_s1, best_Z_eq_s1 = equi_s1.run()
            try:
                best_Z_eq_s1 = float(equi_s1.best_metrics.get("combined_Z", best_Z_eq_s1))
            except Exception:
                pass
            if best_bins_eq_s1 is not None:
                s1_sig_h = [create_hist(toy_data_s1["signal1"], bin_edges=best_bins_eq_s1, var="max_score_sel")]
                s1_bkg_h = [create_hist(toy_data_s1[b], bin_edges=best_bins_eq_s1, var="max_score_sel") for b in bkg_list]
                plot_stacked_histograms(
                    s1_bkg_h, bkg_list,
                    output_filename=os.path.join(equi_s1_dir, "plot_log.pdf"),
                    axis_labels=("Max-score (S1 region)", "Events"),
                    signal_hists=s1_sig_h, signal_labels=["signal1"],
                    signal_scale=100, normalize=False, log=True, log_min=None,
                )
                plot_stacked_histograms(
                    s1_bkg_h, bkg_list,
                    output_filename=os.path.join(equi_s1_dir, "plot.pdf"),
                    axis_labels=("Max-score (S1 region)", "Events"),
                    signal_hists=s1_sig_h, signal_labels=["signal1"],
                    signal_scale=100, normalize=False, log=False, log_min=None,
                )
            save_bin_metrics(equi_s1_dir, equi_s1, prefix="equi_s1")

        # ------- Equidistant on S2 -------
        equi_s2_dir = os.path.join(nb_dir, "equidistant_s2")
        os.makedirs(equi_s2_dir, exist_ok=True)
        equi_s2 = equidistant(
            toy_data_s2,
            bkg_label_lst=bkg_list,
            signal_label_lst=["signal2"],
            var_label="max_score_sel",
            weight_label="weight",
            n_bins=n_bins,
            output_dir=equi_s2_dir,
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            min_edge=0.3
        )
        best_bins_eq_s2 = best_hist_eq_s2 = best_Z_eq_s2 = None
        if args.run_equidistant:
            best_bins_eq_s2, best_hist_eq_s2, best_Z_eq_s2 = equi_s2.run()
            try:
                best_Z_eq_s2 = float(equi_s2.best_metrics.get("combined_Z", best_Z_eq_s2))
            except Exception:
                pass
            if best_bins_eq_s2 is not None:
                s2_sig_h = [create_hist(toy_data_s2["signal2"], bin_edges=best_bins_eq_s2, var="max_score_sel")]
                s2_bkg_h = [create_hist(toy_data_s2[b], bin_edges=best_bins_eq_s2, var="max_score_sel") for b in bkg_list]
                plot_stacked_histograms(
                    s2_bkg_h, bkg_list,
                    output_filename=os.path.join(equi_s2_dir, "plot_log.pdf"),
                    axis_labels=("Max-score (S2 region)", "Events"),
                    signal_hists=s2_sig_h, signal_labels=["signal2"],
                    signal_scale=100, normalize=False, log=True, log_min=None,
                )
                plot_stacked_histograms(
                    s2_bkg_h, bkg_list,
                    output_filename=os.path.join(equi_s2_dir, "plot.pdf"),
                    axis_labels=("Max-score (S2 region)", "Events"),
                    signal_hists=s2_sig_h, signal_labels=["signal2"],
                    signal_scale=100, normalize=False, log=False, log_min=None,
                )
            save_bin_metrics(equi_s2_dir, equi_s2, prefix="equi_s2")

        # ------- BOBR-1D on S1 -------
        bobr1_s1_dir = os.path.join(nb_dir, "bobr1_s1")
        os.makedirs(bobr1_s1_dir, exist_ok=True)
        bobr1_s1 = bobr_1d(
            toy_data_s1,
            bkg_label_lst=bkg_list,
            signal_label_lst=["signal1"],
            var_label="max_score_sel",
            weight_label="weight",
            n_bins=n_bins,
            output_dir=bobr1_s1_dir,
            n_trials=int(args.n_trials),
            gamma_strategy="linear",
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            restart_check_trials=args.restart_check_trials,
            min_edge=0.3,
        )
        best_bins_b1_s1 = best_hist_b1_s1 = best_Z_b1_s1 = None
        if args.run_equidistant:
            best_bins_b1_s1, best_hist_b1_s1, best_Z_b1_s1 = bobr1_s1.run()
            try:
                best_Z_b1_s1 = float(bobr1_s1.best_metrics.get("combined_Z", best_Z_b1_s1))
            except Exception:
                pass
            if best_bins_b1_s1 is not None:
                s1_sig_h = [create_hist(toy_data_s1["signal1"], bin_edges=best_bins_b1_s1, var="max_score_sel")]
                s1_bkg_h = [create_hist(toy_data_s1[b], bin_edges=best_bins_b1_s1, var="max_score_sel") for b in bkg_list]
                plot_stacked_histograms(
                    s1_bkg_h, bkg_list,
                    output_filename=os.path.join(bobr1_s1_dir, "plot_log.pdf"),
                    axis_labels=("Max-score (S1 region)", "Events"),
                    signal_hists=s1_sig_h, signal_labels=["signal1"],
                    signal_scale=100, normalize=False, log=True, log_min=None,
                )
                plot_stacked_histograms(
                    s1_bkg_h, bkg_list,
                    output_filename=os.path.join(bobr1_s1_dir, "plot.pdf"),
                    axis_labels=("Max-score (S1 region)", "Events"),
                    signal_hists=s1_sig_h, signal_labels=["signal1"],
                    signal_scale=100, normalize=False, log=False, log_min=None,
                )
            save_bin_metrics(bobr1_s1_dir, bobr1_s1, prefix="bobr1_s1")

        # ------- BOBR-1D on S2 -------
        bobr1_s2_dir = os.path.join(nb_dir, "bobr1_s2")
        os.makedirs(bobr1_s2_dir, exist_ok=True)
        bobr1_s2 = bobr_1d(
            toy_data_s2,
            bkg_label_lst=bkg_list,
            signal_label_lst=["signal2"],
            var_label="max_score_sel",
            weight_label="weight",
            n_bins=n_bins,
            output_dir=bobr1_s2_dir,
            n_trials=int(args.n_trials),
            gamma_strategy="linear",
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            restart_check_trials=args.restart_check_trials,
            min_edge=0.3,
        )
        best_bins_b1_s2 = best_hist_b1_s2 = best_Z_b1_s2 = None
        if args.run_equidistant:
            best_bins_b1_s2, best_hist_b1_s2, best_Z_b1_s2 = bobr1_s2.run()
            try:
                best_Z_b1_s2 = float(bobr1_s2.best_metrics.get("combined_Z", best_Z_b1_s2))
            except Exception:
                pass
            if best_bins_b1_s2 is not None:
                s2_sig_h = [create_hist(toy_data_s2["signal2"], bin_edges=best_bins_b1_s2, var="max_score_sel")]
                s2_bkg_h = [create_hist(toy_data_s2[b], bin_edges=best_bins_b1_s2, var="max_score_sel") for b in bkg_list]
                plot_stacked_histograms(
                    s2_bkg_h, bkg_list,
                    output_filename=os.path.join(bobr1_s2_dir, "plot_log.pdf"),
                    axis_labels=("Max-score (S2 region)", "Events"),
                    signal_hists=s2_sig_h, signal_labels=["signal2"],
                    signal_scale=100, normalize=False, log=True, log_min=None,
                )
                plot_stacked_histograms(
                    s2_bkg_h, bkg_list,
                    output_filename=os.path.join(bobr1_s2_dir, "plot.pdf"),
                    axis_labels=("Max-score (S2 region)", "Events"),
                    signal_hists=s2_sig_h, signal_labels=["signal2"],
                    signal_scale=100, normalize=False, log=False, log_min=None,
                )
            save_bin_metrics(bobr1_s2_dir, bobr1_s2, prefix="bobr1_s2")

        # Record summary for this n_bins
        nbins_checked.append(n_bins)
        summary.setdefault(str(n_bins), {})
        if gmm_Z is not None:
            summary[str(n_bins)]["gmm_best_Z"] = float(gmm_Z)

        # Combined Z for argmax methods using the same rule as GMM (default geometric)
        equi_s1_Z = None if not args.run_equidistant else (None if equi_s1.best_metrics is None else equi_s1.best_metrics.get("combined_Z"))
        equi_s2_Z = None if not args.run_equidistant else (None if equi_s2.best_metrics is None else equi_s2.best_metrics.get("combined_Z"))
        if (equi_s1_Z is not None) and (equi_s2_Z is not None):
            eq_comb = combine_two_z(equi_s1_Z, equi_s2_Z, method=args.combine)
            summary[str(n_bins)]["equi_combined_best_Z"] = float(eq_comb)
            equi_combined_Zs.append(eq_comb)

        b1_s1_Z = None if not args.run_equidistant else (None if bobr1_s1.best_metrics is None else bobr1_s1.best_metrics.get("combined_Z"))
        b1_s2_Z = None if not args.run_equidistant else (None if bobr1_s2.best_metrics is None else bobr1_s2.best_metrics.get("combined_Z"))
        if (b1_s1_Z is not None) and (b1_s2_Z is not None):
            b1_comb = combine_two_z(b1_s1_Z, b1_s2_Z, method=args.combine)
            summary[str(n_bins)]["bobr1_combined_best_Z"] = float(b1_comb)
            bobr1_combined_Zs.append(b1_comb)

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "summary_results_3d.json")
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=2)
    print("Wrote summary results to", summary_path)

    # Comparison plot of best Z vs n_bins
    try:
        plt.figure(figsize=(9, 6))
        if gmm_Zs:
            plt.plot(nbins_checked[: len(gmm_Zs)], gmm_Zs, marker="o", label="BOBR-GMM (multi-D)")
        if equi_combined_Zs:
            plt.plot(nbins_checked[: len(equi_combined_Zs)], equi_combined_Zs,
                     marker="D", linestyle="--", label="Argmax+Equidistant (combined)")
        if bobr1_combined_Zs:
            plt.plot(nbins_checked[: len(bobr1_combined_Zs)], bobr1_combined_Zs,
                     marker="^", linestyle=":", label="Argmax+BOBR-1D (combined)")
        plt.xlabel("Number of bins / components")
        plt.ylabel("Best significance (combined Z)")
        plt.title("3D: BOBR-GMM vs Argmax(Selected Dims) 1D methods")
        plt.grid(True)
        plt.legend()
        out_plot = os.path.join(args.output_dir, "summary_bestZ_comparison_3d.pdf")
        plt.savefig(out_plot, dpi=150)
        plt.clf()
        print("Wrote comparison plot to", out_plot)
    except Exception as e:
        print("Failed to write comparison plot:", e)


if __name__ == "__main__":
    print("INFO: Running the 3D toy example")
    main()
