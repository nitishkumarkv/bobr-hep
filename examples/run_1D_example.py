from bobr_hep.utils.data_generation import generate_toy_data_1D
from bobr_hep.utils.plotting import plot_stacked_histograms
from bobr_hep import bobr_1d, equidistant
import os
import argparse
import pandas as pd
import hist
import matplotlib.pyplot as plt
import json
import numpy as np


def create_hist(df, bin_edges=None):
    # create and fill a hist.Hist using either regular or variable binning
    if bin_edges is None:
        h = hist.Hist(hist.axis.Regular(50, 0, 1, name="NN_output"), storage=hist.storage.Weight())
    else:
        h = hist.Hist(hist.axis.Variable(bin_edges, name="NN_output"), storage=hist.storage.Weight())
    h.fill(df["NN_output"], weight=df["weight"])
    return h


def assign_bins_with_predict(binner, data_dict, save_path=None, column_name="bin_index"):
    """Use the binner.predict(X) to assign bins to each DataFrame in data_dict.

    If save_path is provided, writes per-label parquet files with the added column.
    Returns a dict label -> np.ndarray of assigned indices.
    """
    assigned = {}
    os.makedirs(save_path, exist_ok=True) if save_path is not None else None
    for label, df in data_dict.items():
        inds = binner.predict(df["NN_output"].values)
        assigned[label] = inds
        if save_path is not None:
            out_df = df.copy()
            out_df[column_name] = inds
            out_df.to_parquet(os.path.join(save_path, f"{label}_with_bins.parquet"))
    return assigned


def main():
    def make_json_serializable(obj):
        """Recursively convert numpy types and arrays to native Python types for JSON dumping."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return make_json_serializable(obj.tolist())
        # numpy scalar
        if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
            try:
                return make_json_serializable(obj.item())
            except Exception:
                pass
        # handle floats with nan
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            return float(obj)
        return obj

    parser = argparse.ArgumentParser(description="Run the toy 1D example using BOBR binners")
    parser.add_argument("--output-dir", type=str, default="toy_results", help="Output directory")
    parser.add_argument("--plot-toy-data", action="store_true", help="Plot toy data")
    parser.add_argument("--run-bobr", action="store_true", help="Run Bayesian BOBR optimizer")
    parser.add_argument("--run-equidistant", action="store_true", help="Run equidistant binning")
    parser.add_argument("--nbins", type=int, nargs='+', default=[3, 5, 10, 15, 20], help="List of n_bins values to test")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials override for all nbins")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for toy data generation")
    parser.add_argument("--save-assigned", action="store_true", help="Save per-event assigned bin indices (parquet)")
    parser.add_argument("--min-bkg-per-bin", type=int, default=1, help="Minimum background count per bin (overrides binner default)")
    parser.add_argument("--penalty-low-lambda", type=float, default=1.0, help="Weight for low-background penalty")
    parser.add_argument("--penalty-unc-lambda", type=float, default=1.0, help="Weight for relative-uncertainty penalty")
    parser.add_argument("--rel-unc-threshold", type=float, default=0.1, help="Relative uncertainty threshold")
    parser.add_argument("--restart-check-trials", type=int, default=200, help="Trials per restart attempt before halving beta and restarting optimization")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    bkg_list = ["bkg1", "bkg2", "bkg3"]
    signal_list = ["signal"]

    # Generate toy data (dict of DataFrames)
    toy_data = generate_toy_data_1D(
        n_signal=100000,
        n_bkg=100000,
        xs_signal=0.5,
        xs_bkg1=100,
        xs_bkg2=80,
        xs_bkg3=50,
        lumi=100,
        seed=args.seed,
    )

    # Quick plots of the toy data
    if args.plot_toy_data:
        data_path = os.path.join(args.output_dir, "toy_data")
        os.makedirs(data_path, exist_ok=True)
        bkg_hists = [create_hist(toy_data[bkg]) for bkg in bkg_list]
        signal_hists = [create_hist(toy_data[signal]) for signal in signal_list]
        plot_stacked_histograms(
            bkg_hists,
            bkg_list,
            output_filename=f"{data_path}/plot_log.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=True,
            log_min=1e-5,
        )

    nbins_list = args.nbins
    n_trials_default = 100
    if args.n_trials is not None:
        n_trials_default = int(args.n_trials)

    summary = {}
    # containers for summary comparison plot
    nbins_checked = []
    bobr_Zs = []
    equi_Zs = []

    for n_bins in nbins_list:
        print(f"\n=== Running n_bins={n_bins} ===")

        # --- Bayesian BOBR ---
        bobr_output_dir = os.path.join(args.output_dir, f"optimizer_results_nbins_{n_bins}")
        os.makedirs(bobr_output_dir, exist_ok=True)
        best_bins_bobr = None
        best_Z_bobr = None

        optimizer = bobr_1d(
            toy_data,
            bkg_label_lst=bkg_list,
            signal_label_lst=signal_list,
            var_label="NN_output",
            weight_label="weight",
            n_bins=n_bins,
            output_dir=bobr_output_dir,
            n_trials=n_trials_default,
            gamma_strategy="linear",
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            restart_check_trials=args.restart_check_trials,
        )

        if args.run_bobr:
            best_bins_bobr, best_hist_dict_bobr, best_Z_bobr = optimizer.run()
            print("BOBR: Best bins:", best_bins_bobr)
            # prefer the stored combined significance (computed without penalties)
            try:
                best_Z_from_metrics = optimizer.best_metrics.get("combined_Z")
                if best_Z_from_metrics is not None:
                    best_Z_bobr = float(best_Z_from_metrics)
            except Exception:
                pass
            print("BOBR: Best Z:", best_Z_bobr)
            optimizer.visualize_optimization()
            # save metrics and bins
            to_save = {"best_bins": best_bins_bobr, "best_Z": best_Z_bobr}
            try:
                to_save["best_metrics"] = optimizer.best_metrics
            except Exception:
                pass
            with open(os.path.join(bobr_output_dir, "optimized_bins.json"), "w") as f:
                json.dump(make_json_serializable(to_save), f, indent=2)
            summary[str(n_bins)] = summary.get(str(n_bins), {})
            summary[str(n_bins)]["bobr_best_bins"] = best_bins_bobr
            summary[str(n_bins)]["bobr_best_Z"] = best_Z_bobr

        # if results exist on disk, load them (useful for running plotting-only)
        opt_path = os.path.join(bobr_output_dir, "optimized_bins.json")
        if os.path.exists(opt_path) and best_bins_bobr is None:
            with open(opt_path, "r") as f:
                data = json.load(f)
                best_bins_bobr = data.get("best_bins")
                best_Z_bobr = data.get("best_Z")
                summary.setdefault(str(n_bins), {})["bobr_best_bins"] = best_bins_bobr
                summary.setdefault(str(n_bins), {})["bobr_best_Z"] = best_Z_bobr

        # Plot and assign if bins are available
        if best_bins_bobr is not None:
            best_signal_hists = [create_hist(toy_data[signal], bin_edges=best_bins_bobr) for signal in signal_list]
            best_bkg_hists = [create_hist(toy_data[bkg], bin_edges=best_bins_bobr) for bkg in bkg_list]
            plot_stacked_histograms(
                best_bkg_hists,
                bkg_list,
                output_filename=os.path.join(bobr_output_dir, "plot_log.pdf"),
                axis_labels=("NN output", "Events"),
                signal_hists=best_signal_hists,
                signal_labels=signal_list,
                signal_scale=100,
                normalize=False,
                log=True,
                log_min=1e-5,
            )

            # Use predict to assign bin indices and optionally save per-event assignments
            # Create a binner instance that has best_bins set so we can call predict
            pred_binner = bobr_1d(
                toy_data,
                bkg_label_lst=bkg_list,
                signal_label_lst=signal_list,
                var_label="NN_output",
                weight_label="weight",
                n_bins=n_bins,
                min_bkg_per_bin=args.min_bkg_per_bin,
            )
            pred_binner.best_bins = best_bins_bobr
            pred_binner.n_bins = len(best_bins_bobr) - 1
            assigned = assign_bins_with_predict(pred_binner, toy_data, save_path=(bobr_output_dir if args.save_assigned else None))
            summary[str(n_bins)]["bobr_assigned_counts"] = {k: int(np.bincount(v, minlength=pred_binner.n_bins).sum()) for k, v in assigned.items()}
            # record Z for summary plot
            try:
                bobr_Zs.append(float(best_Z_bobr))
            except Exception:
                bobr_Zs.append(np.nan)

        # --- Equidistant ---
        equi_output_dir = os.path.join(args.output_dir, f"equidistance_results_nbins_{n_bins}")
        os.makedirs(equi_output_dir, exist_ok=True)
        best_bins_equi = None
        best_Z_equi = None

        equi = equidistant(
            toy_data,
            bkg_label_lst=bkg_list,
            signal_label_lst=signal_list,
            var_label="NN_output",
            weight_label="weight",
            n_bins=n_bins,
            min_bkg_per_bin=args.min_bkg_per_bin,
            penalty_low_lambda=args.penalty_low_lambda,
            penalty_unc_lambda=args.penalty_unc_lambda,
            rel_unc_threshold=args.rel_unc_threshold,
            restart_check_trials=args.restart_check_trials,
            output_dir=equi_output_dir,
        )
        if args.run_equidistant:
            best_bins_equi, best_hist_dict_equi, best_Z_equi = equi.run()
            equi.visualize_optimization()
            # prefer stored metrics for significance
            try:
                best_Z_from_metrics = equi.best_metrics.get("combined_Z")
                if best_Z_from_metrics is not None:
                    best_Z_equi = float(best_Z_from_metrics)
            except Exception:
                pass
            # save metrics and bins
            to_save_eq = {"best_bins": best_bins_equi, "best_Z": best_Z_equi}
            try:
                to_save_eq["best_metrics"] = equi.best_metrics
            except Exception:
                pass
            with open(os.path.join(equi_output_dir, "optimized_bins.json"), "w") as f:
                json.dump(make_json_serializable(to_save_eq), f, indent=2)
            summary.setdefault(str(n_bins), {})["equi_best_bins"] = best_bins_equi
            summary.setdefault(str(n_bins), {})["equi_best_Z"] = best_Z_equi

        # load if present
        eqi_path = os.path.join(equi_output_dir, "optimized_bins.json")
        if os.path.exists(eqi_path) and best_bins_equi is None:
            with open(eqi_path, "r") as f:
                data = json.load(f)
                best_bins_equi = data.get("best_bins")
                best_Z_equi = data.get("best_Z")
                summary.setdefault(str(n_bins), {})["equi_best_bins"] = best_bins_equi
                summary.setdefault(str(n_bins), {})["equi_best_Z"] = best_Z_equi

        if best_bins_equi is not None:
            best_signal_hists = [create_hist(toy_data[signal], bin_edges=best_bins_equi) for signal in signal_list]
            best_bkg_hists = [create_hist(toy_data[bkg], bin_edges=best_bins_equi) for bkg in bkg_list]
            plot_stacked_histograms(
                best_bkg_hists,
                bkg_list,
                output_filename=os.path.join(equi_output_dir, "plot_log.pdf"),
                axis_labels=("NN output", "Events"),
                signal_hists=best_signal_hists,
                signal_labels=signal_list,
                signal_scale=100,
                normalize=False,
                log=True,
                log_min=1e-5,
            )

            # Use equi.predict (or numpy) to assign bins
            pred_equi = equidistant(
                toy_data,
                bkg_label_lst=bkg_list,
                signal_label_lst=signal_list,
                var_label="NN_output",
                weight_label="weight",
                n_bins=n_bins,
                min_bkg_per_bin=args.min_bkg_per_bin,
            )
            pred_equi.best_bins = best_bins_equi
            pred_equi.n_bins = len(best_bins_equi) - 1
            assigned_equi = assign_bins_with_predict(pred_equi, toy_data, save_path=(equi_output_dir if args.save_assigned else None))
            summary.setdefault(str(n_bins), {})["equi_assigned_counts"] = {k: int(np.bincount(v, minlength=pred_equi.n_bins).sum()) for k, v in assigned_equi.items()}
            try:
                equi_Zs.append(float(best_Z_equi))
            except Exception:
                equi_Zs.append(np.nan)

        # if either method produced a result, mark this nbins for plotting
        if (best_Z_bobr is not None) or (best_Z_equi is not None):
            nbins_checked.append(n_bins)

    # write summary
    summary_path = os.path.join(args.output_dir, "summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=2)
    print("Wrote summary results to", summary_path)

    # summary comparison plot
    if nbins_checked:
        try:
            plt.figure(figsize=(8, 5))
            # Plot each series only if it has values and slice nbins_checked to match lengths
            if bobr_Zs:
                plt.plot(nbins_checked[: len(bobr_Zs)], bobr_Zs, marker='o', label='BOBR (bayesian)')
            if equi_Zs:
                plt.plot(nbins_checked[: len(equi_Zs)], equi_Zs, marker='s', label='Equidistant')
            plt.xlabel('Number of bins')
            plt.ylabel('Best significance (Z)')
            plt.title('BOBR vs Equidistant: Best Z by n_bins')
            plt.legend()
            plt.grid(True)
            out_plot = os.path.join(args.output_dir, 'summary_bestZ_comparison.png')
            plt.savefig(out_plot)
            plt.clf()
            print('Wrote summary comparison plot to', out_plot)
        except Exception as e:
            print('Failed to write summary comparison plot:', e)


if __name__ == "__main__":
    print("INFO: Running the toy example")
    main()

