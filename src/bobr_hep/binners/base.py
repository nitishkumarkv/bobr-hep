from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import optuna
import matplotlib.pyplot as plt
import mplhep as hep
import re
plt.style.use(hep.style.ROOT)

legend_fontsize = 18
xy_label_fontsize = 22
tick_label_fontsize = 17
fig_size = (7, 6)

class bobr_base:
    def __init__(
        self,
        df_dict: Dict[str, Any],
        bkg_label_lst: Sequence[str],
        signal_label_lst: Sequence[str],
        var_label: str,
        weight_label: str,
        n_bins: int = 10,
        n_trials: int = 50,
        output_dir: str = "./optimizer_results",
        gamma_strategy: str = "sqrt",
        beta: float = 0.25,
        min_edge: float = 0.0,
        max_edge: float = 1.0,
        min_bkg_per_bin: int = 10,
        penalty_low_lambda: float = 1.0,
        penalty_unc_lambda: float = 1.0,
        rel_unc_threshold: float = 0.1,
        restart_check_trials: int = 200,
        seed_optimizer: int = 42,
    ) -> None:
        self.df_dict = df_dict
        self.bkg_label_lst = list(bkg_label_lst)
        self.signal_label_lst = list(signal_label_lst)
        self.var_label = var_label
        self.weight_label = weight_label
        self.n_bins = int(n_bins)
        self.n_trials = int(n_trials)
        self.output_dir = output_dir
        self.gamma_strategy = gamma_strategy
        self.beta = beta
        if max_edge <= min_edge:
            raise ValueError(f"max_edge ({max_edge}) must be greater than min_edge ({min_edge}).")
        self.min_edge = float(min_edge)
        self.max_edge = float(max_edge)

        # outputs filled by subclasses
        self.best_bins: Optional[list] = None
        self.best_components: Optional[list] = None
        self.best_hist_dict: Optional[Dict[str, np.ndarray]] = None
        self.best_score: Optional[float] = None
        self.study: Optional[optuna.study.Study] = None
        self.min_bkg_per_bin = int(min_bkg_per_bin)
        self.penalty_low_lambda = penalty_low_lambda
        self.penalty_unc_lambda = penalty_unc_lambda
        self.rel_unc_threshold = rel_unc_threshold
        self.restart_check_trials = int(restart_check_trials)
        self.seed_optimizer = int(seed_optimizer)

    def gamma_fn(self):
        def gamma_linear(n):
            return min(int(np.ceil(self.beta * n)), 25)

        def gamma_sqrt(n):
            return min(int(np.ceil(self.beta * np.sqrt(n))), 25)

        if self.gamma_strategy == "linear":
            return gamma_linear
        elif self.gamma_strategy == "sqrt":
            return gamma_sqrt
        else:
            raise ValueError("Unsupported gamma_strategy. Use 'linear' or 'sqrt'.")
        
    def plot_optimization_history(self, study, output_dir="."):
        # Collect finished (non-pruned) trials with objective values
        trials = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]

        if not trials:
            print("No completed trials to plot.")
            return

        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]

        # Compute "best so far" at each trial
        best_values = []
        current_best = None
        for v in values:
            if current_best is None:
                current_best = v
            else:
                current_best = max(current_best, v)
            best_values.append(current_best)

        fig, ax = plt.subplots(figsize=fig_size)

        # Scatter of all trial outcomes
        ax.plot(
                trial_numbers,
                values,
                marker="o",
                label="Objective value",
                linestyle="None",
            )

        # Best-so-far line
        ax.plot(trial_numbers, best_values, label="Best value", color="red")

        ax.set_xlabel("Trial", fontsize=xy_label_fontsize)
        ax.set_ylabel("Objective value", fontsize=xy_label_fontsize)
        yrange = max(best_values) - min(best_values)
        min_ =  min(best_values) - (yrange*0.1)
        max_ = max(best_values) + (yrange*0.3)
        ax.set_ylim(min_, max_)
        ax.legend(loc="upper right", fontsize=legend_fontsize)
        ax.tick_params(labelsize=tick_label_fontsize)    


        fig.tight_layout(pad=0.1, w_pad=0., h_pad=0., rect=(0, 0., 1, 1))
        fig.savefig(os.path.join(output_dir, "optimization_history_custom.pdf"))
        plt.close(fig)


    def visualize_optimization(self) -> None:
        if self.study is None:
            print("No study found. Run run() first.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        with plt.style.context("default"):
            ax = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            fig = ax.get_figure()
            fig.suptitle("Parallel Coordinates", fontsize=14)
            fig.savefig(os.path.join(self.output_dir, "parallel_coordinate_plot.pdf"))
            plt.clf()

        with plt.style.context("default"):
            ax = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            fig = ax.get_figure()
            fig.suptitle("Optimization History", fontsize=14)
            fig.savefig(os.path.join(self.output_dir, "optimization_history_plot.pdf"))
            plt.clf()
        
        # Custom optimization history plot
        self.plot_optimization_history(self.study, output_dir=self.output_dir)
        plt.clf()

        trials = [trial for trial in self.study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            return

        trial_numbers = [t.number for t in trials]
        first_params = trials[0].params

        # 1D bin boundary evolution
        bin_keys = sorted([k for k in first_params.keys() if k.startswith("bin_")], key=lambda x: int(x.split("_")[1]))
        if bin_keys:
            try:
                bin_evolution = {k: [t.params.get(k, float("nan")) for t in trials] for k in bin_keys}
                plt.figure(figsize=(10, 6))
                for k in bin_keys:
                    plt.plot(trial_numbers, bin_evolution[k], label=k)
                plt.xlabel("Trial Number")
                plt.ylabel("Bin Boundaries")
                plt.title(f"Evolution of bin boundaries (n_bins={self.n_bins})")
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f"bin_evolution.pdf"))
                plt.clf()
            except Exception as e:
                print("bin evolution plot failed:", e)
            return

        # GMM mu evolution (fallback)
        mu_keys = [k for k in first_params.keys() if k.startswith("mu_")]
        if mu_keys:
            try:
                pattern = re.compile(r"mu_(\d+)_(\d+)")
                mu_map = {}
                for k in mu_keys:
                    m = pattern.match(k)
                    if not m:
                        continue
                    binidx = int(m.group(1))
                    dim = int(m.group(2))
                    mu_map.setdefault((binidx, dim), []).append(k)

                plt.figure(figsize=(10, 6))
                count = 0
                for (binidx, dim) in sorted(mu_map.keys()):
                    if count >= 8:
                        break
                    key = f"mu_{binidx}_{dim}"
                    vals = [t.params.get(key, float('nan')) for t in trials]
                    plt.plot(trial_numbers, vals, label=key)
                    count += 1
                plt.xlabel("Trial Number")
                plt.ylabel("mu value")
                plt.title("Evolution of Gaussian means (sample)")
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f"mu_evolution.pdf"))
                plt.clf()
            except Exception as e:
                print("mu evolution plot failed:", e)

    @staticmethod
    def asymptotic_significance(s: np.ndarray, b: np.ndarray, eps: float = 1e-10, ratio_threshold: float = 0.1) -> float:
        safe_b = np.maximum(b, eps)
        ratio = s / safe_b
        Z_asimov = np.sqrt(2.0 * ((s + safe_b) * np.log1p(ratio) - s))
        Z_approx = s / np.sqrt(safe_b)
        Z_per_bin = np.where(ratio < ratio_threshold, Z_approx, Z_asimov)
        return float(np.sqrt(np.sum(Z_per_bin**2)))

    def _compute_1d_counts(self, bin_edges: Sequence[float]):
        edges = np.asarray(bin_edges)
        nbins = len(edges) - 1
        counts: Dict[str, np.ndarray] = {}
        for lbl, df in self.df_dict.items():
            vals = np.asarray(df[self.var_label])
            wts = np.asarray(df[self.weight_label])
            inds = np.searchsorted(edges, vals, side="right") - 1
            inds = np.clip(inds, 0, nbins - 1)
            arr = np.zeros(nbins, dtype=float)
            for i in range(nbins):
                arr[i] = wts[inds == i].sum()
            counts[lbl] = arr
        return counts

    def _compute_1d_counts_and_sumsq(self, bin_edges: Sequence[float]):
        """Compute per-label sums (weights) and sum-of-squares (weights**2) for 1D binning.

        Returns (counts_dict, sumsq_dict) where each is a dict label -> np.ndarray of length nbins.
        """
        edges = np.asarray(bin_edges)
        nbins = len(edges) - 1
        counts: Dict[str, np.ndarray] = {}
        sumsq: Dict[str, np.ndarray] = {}
        for lbl, df in self.df_dict.items():
            vals = np.asarray(df[self.var_label])
            wts = np.asarray(df[self.weight_label])
            inds = np.searchsorted(edges, vals, side="right") - 1
            inds = np.clip(inds, 0, nbins - 1)
            arr = np.zeros(nbins, dtype=float)
            arr2 = np.zeros(nbins, dtype=float)
            for i in range(nbins):
                sel = inds == i
                if sel.any():
                    w = wts[sel]
                    arr[i] = float(w.sum())
                    arr2[i] = float((w ** 2).sum())
            counts[lbl] = arr
            sumsq[lbl] = arr2
        return counts, sumsq

    def _compute_3d_counts_and_sumsq(self, components_or_edges):
        """Compute per-label sums and sum-of-squares for GMM-based 3D binner.

        The argument may be either:
          - a list of Gaussian component dicts (with 'mean' and 'cov'), or
          - None (in which case `self.best_components` will be used).

        Returns (counts_dict, sumsq_dict).
        """
        # lazily import scipy multivariate_normal only when needed
        try:
            from scipy.stats import multivariate_normal
        except Exception:
            raise RuntimeError("scipy is required for 3D GMM counts computation")

        comps = components_or_edges
        if comps is None:
            comps = getattr(self, "best_components", None)

        if not isinstance(comps, list) or (len(comps) > 0 and not isinstance(comps[0], dict)):
            raise ValueError("_compute_3d_counts_and_sumsq expects a list of component dicts or None (use best_components)")

        # build stacked scores, weights and labels across all processes
        scores_list = []
        wts_list = []
        labels = []
        for label, df in self.df_dict.items():
            arr = np.vstack(df[self.var_label].to_numpy())
            # If the binner supplied an explicit `dims_to_use`, apply it.
            # Automatic N vs N-1 reduction (`use_n_minus_one`) was removed; the
            # user should explicitly set which score-dimensions to use.
            if hasattr(self, "dims_to_use") and self.dims_to_use is not None:
                arr = arr[:, self.dims_to_use]
            scores_list.append(arr)
            wts_list.append(df[self.weight_label].to_numpy())
            labels.extend([label] * len(df))

        scores = np.vstack(scores_list)
        wts = np.concatenate(wts_list)
        labels = np.array(labels)

        N = scores.shape[0]
        K = len(comps)
        logpdf = np.zeros((N, K))
        for k, g in enumerate(comps):
            rv = multivariate_normal(mean=g["mean"], cov=g["cov"], allow_singular=True)
            logpdf[:, k] = rv.logpdf(scores)
        assgn = np.argmax(logpdf, axis=1)

        counts: Dict[str, np.ndarray] = {}
        sumsq: Dict[str, np.ndarray] = {}
        for label in self.df_dict:
            arr = np.zeros(K, dtype=float)
            arr2 = np.zeros(K, dtype=float)
            mask = labels == label
            for k in range(K):
                sel = (assgn == k) & mask
                if sel.any():
                    ww = wts[sel]
                    arr[k] = float(ww.sum())
                    arr2[k] = float((ww ** 2).sum())
            counts[label] = arr
            sumsq[label] = arr2
        return counts, sumsq

    def _run_opt_with_beta_restarts(
        self,
        optimize_callable,
        max_restarts: int = 3,
        trials_total: Optional[int] = None,
        check_interval: Optional[int] = None,
    ) -> None:
        """Run an optimization callable and, if the best objective value is not positive
        after the first `check_interval` trials, halve `self.beta` and restart the
        whole optimization up to `max_restarts` times.

        Behavior:
        - Run up to `check_interval` trials first (or all trials if smaller).
        - If the best objective after that checkpoint is positive, continue and run the
          remaining trials (so total trials equals `trials_total`).
        - If the best objective is non-positive, halve `self.beta` and restart from
          scratch (create a fresh study in the provided `optimize_callable`) and try
          again (up to `max_restarts`).

        `optimize_callable(n_trials)` is expected to create `self.study` (fresh) and
        run `n_trials` trials on that study. This helper will call it multiple times
        as needed according to the logic above.
        """
        if trials_total is None:
            trials_total = int(getattr(self, "n_trials", 50))
        if check_interval is None:
            check_interval = int(getattr(self, "restart_check_trials", 200))

        attempts = 0
        while True:
            attempts += 1

            # ensure fresh study is created by the optimize_callable for each attempt
            self.study = None

            # run the initial checkpoint
            first_run = min(check_interval, trials_total)
            optimize_callable(first_run)

            # if study not set, give up
            if getattr(self, "study", None) is None:
                return

            best_val = getattr(self.study, "best_value", None)

            # if best is positive, continue and run remaining trials (if any)
            if best_val is not None and float(best_val) > 0.0:
                remaining = trials_total - first_run
                if remaining > 0:
                    optimize_callable(remaining)
                return

            # otherwise, if we've exhausted restart attempts, stop
            if attempts > max_restarts:
                return

            # halve beta to increase exploration and retry the whole optimization
            try:
                old_beta = float(self.beta)
            except Exception:
                old_beta = None
            self.beta = float(self.beta) / 2.0
            print(
                f"Optimization did not find positive objective (best={best_val}); halving beta {old_beta} -> {self.beta} and retrying (attempt {attempts}/{max_restarts})"
            )

    def compute_and_store_metrics(self, hist: Dict[str, np.ndarray], sumsq: Dict[str, np.ndarray]) -> None:
        """Compute and store per-bin metrics: background counts, sumsq, relative uncertainty,
        per-bin significance for each signal, and combined significance.

        Stores results in `self.best_metrics`.
        """
        eps = 1e-10
        # background counts and sumsq
        bkg_counts = np.sum([hist[l] for l in self.bkg_label_lst], axis=0)
        bkg_sumsq = np.sum([sumsq[l] for l in self.bkg_label_lst], axis=0)

        rel_unc = np.sqrt(bkg_sumsq) / np.maximum(bkg_counts, eps)

        per_signal = {}
        combined_vals = []
        for sig in self.signal_label_lst:
            s = hist[sig]
            b = np.sum([arr for lbl, arr in hist.items() if lbl != sig], axis=0)
            safe_b = np.maximum(b, eps)
            ratio = s / safe_b
            Z_asimov = np.sqrt(2.0 * ((s + safe_b) * np.log1p(ratio) - s))
            Z_approx = s / np.sqrt(safe_b)
            Z_per_bin = np.where(ratio < 0.1, Z_approx, Z_asimov)
            combined = float(np.sqrt(np.sum(Z_per_bin ** 2)))
            per_signal[sig] = {
                "per_bin_Z": Z_per_bin,
                "combined_Z": combined,
            }
            combined_vals.append(combined)

        # combine multi-signal Z if needed across arbitrary number of signals
        if len(combined_vals) == 0:
            overall_Z = 0.0
        else:
            comb = getattr(self, "combination", "geometric")
            if comb == "quadrature":
                overall_Z = float(math.sqrt(sum(z * z for z in combined_vals)))
            elif comb == "geometric":
                # geometric mean only defined for positive values
                if all(z > 0 for z in combined_vals):
                    prod = 1.0
                    for z in combined_vals:
                        prod *= z
                    overall_Z = float(prod ** (1.0 / len(combined_vals)))
                else:
                    overall_Z = 0.0
            elif comb == "harmonic":
                if all(z > 0 for z in combined_vals):
                    overall_Z = float(len(combined_vals) / sum(1.0 / (z + 1e-12) for z in combined_vals))
                else:
                    overall_Z = 0.0
            else:
                overall_Z = float(max(combined_vals))

        self.best_metrics = {
            "bkg_counts": bkg_counts,
            "bkg_sumsq": bkg_sumsq,
            "rel_unc_bkg": rel_unc,
            "per_signal": per_signal,
            "combined_Z": overall_Z,
        }

    def _compute_low_penalty(self, bkg: np.ndarray, min_bkg: float) -> float:
        """Quadratic penalty: sum_k [max(0, min_bkg - B_k)]^2"""
        deficit = np.maximum(0.0, min_bkg - bkg)
        return float(np.sum(deficit**2))

    def _compute_unc_penalty(
        self,
        bkg: np.ndarray,
        bkg_sumsq: np.ndarray,
        rel_unc_threshold: float,
    ) -> float:
        """Quadratic penalty: sum_k [max(0, σ_rel,k - r)]^2, with σ_rel,k = sqrt(sum w^2)/B_k."""
        eps = 1e-10  # avoid divide-by-zero
        rel_unc = np.sqrt(bkg_sumsq) / np.maximum(bkg, eps)
        excess = np.maximum(0.0, rel_unc - rel_unc_threshold)
        return float(np.sum(excess**2))
