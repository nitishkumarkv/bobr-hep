"""Generic GMM-based binner implementation (lowercase class/file name).

This module provides `bobr_gmm`, a generic Gaussian Mixture Model based
binner. It intentionally does not embed any automatic N vs N-1 behavior.
The user explicitly passes which NN_output dims to use via `dims_to_use`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

import logging
logger = logging.getLogger(__name__)
from .base import bobr_base
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import ListedColormap, BoundaryNorm


class bobr_gmm(bobr_base):
    """Generic GMM binner with consistent (hard/soft) assignment.

    - `assign_mode`: "hard" (default) or "soft"
        * "hard": argmax component → mutually exclusive bins
        * "soft": responsibilities γ_k → fractional contributions
      The SAME mode is used during optimization and for stored metrics.
    """

    def __init__(
        self,
        df_dict: dict = None,
        bkg_label_lst: Optional[Sequence[str]] = None,
        signal_label_lst: Optional[Sequence[str]] = None,
        var_label: Optional[str] = None,      # typically "NN_output"
        weight_label: Optional[str] = None,   # typically "weight"
        n_components: int = 6,
        dims_to_use: Optional[Sequence[int]] = None,
        combination: str = "quadrature",
        penalty: float = 0.0,
        assign_mode: str = "hard",            # <--- NEW, default hard
        temperature: float = 1.0,             # used only if assign_mode="soft"
        **kwargs,
    ) -> None:
        super().__init__(
            df_dict=df_dict,
            bkg_label_lst=bkg_label_lst,
            signal_label_lst=signal_label_lst,
            var_label=var_label,
            weight_label=weight_label,
            **kwargs,
        )
        self.n_components = int(n_components)
        self.dims_to_use = None if dims_to_use is None else list(dims_to_use)
        self.combination = combination
        self.penalty = float(penalty)
        self.assign_mode = str(assign_mode).lower()
        if self.assign_mode not in ("hard", "soft"):
            raise ValueError("assign_mode must be 'hard' or 'soft'")
        self.temperature = float(temperature)

    # -- Helpers -------------------------------------------------
    def _infer_n_dims(self, data) -> int:
        first_key = next(iter(sorted(data.keys())))
        sample = data[first_key]
        if isinstance(sample, pd.DataFrame):
            X0 = np.vstack(sample[self.var_label].to_numpy())
        else:  # tuple (X, w)
            X0 = sample[0]
        if self.dims_to_use is None:
            return X0.shape[1]
        return len(self.dims_to_use)

    def _slice_dims(self, X: np.ndarray) -> np.ndarray:
        if self.dims_to_use is None:
            return X
        return X[:, self.dims_to_use]

    def _build_param_vector(self, trial, n_dims: int):
        """Sample means, raw Cholesky entries, and mixture logits from Optuna."""
        means = np.zeros((self.n_components, n_dims))
        tril_raw = np.zeros((self.n_components, n_dims, n_dims))
        mix_logit = np.zeros(self.n_components)
#
#        for k in range(self.n_components):
#            mix_logit[k] = trial.suggest_float(f"mix_logit_{k}", -6.0, 6.0)
#            for d in range(n_dims):
#                means[k, d] = trial.suggest_float(f"mu_{k}_{d}", self.min_edge, self.max_edge)
#                for d2 in range(d + 1):
#                    tril_raw[k, d, d2] = trial.suggest_float(f"L_{k}_{d}_{d2}", -3.0, 3.0)

        
        R = max(1e-6, float(self.max_edge) - float(self.min_edge))  # characteristic scale

        for k in range(self.n_components):
            # mixture weights: wide but finite is fine
            mix_logit[k] = trial.suggest_float(f"mix_logit_{k}", -6.0, 6.0)

            for d in range(n_dims):
                # means stay within the variable domain
                means[k, d] = trial.suggest_float(f"mu_{k}_{d}", self.min_edge, self.max_edge)

                for d2 in range(d + 1):
                    if d2 == d:
                        # diagonal: raw is log(σ); keep σ relative to data scale
                        sigma_min = 0.02 * R   # narrower end (2% of range)
                        sigma_max = 0.80 * R   # broad but not absurd
                        lo = math.log(max(sigma_min, 1e-6))
                        hi = math.log(max(sigma_max, 1e-6))
                        tril_raw[k, d, d2] = trial.suggest_float(f"L_{k}_{d}_{d2}", lo, hi)
                    else:
                        # off-diagonals live in "data units" (same as σ); cap to a fraction of R
                        off = 0.50 * R  # 50% of range is generous; tighten to 0.3*R if needed
                        tril_raw[k, d, d2] = trial.suggest_float(f"L_{k}_{d}_{d2}", -off, off)

        return means, tril_raw, mix_logit

    def _tril_to_cov(self, tril_raw: np.ndarray) -> np.ndarray:
        covs = []
        for k in range(tril_raw.shape[0]):
            L = np.tril(tril_raw[k])
            for i in range(L.shape[0]):
                L[i, i] = math.exp(L[i, i])
            cov = L @ L.T
            covs.append(cov)
        return np.array(covs)

    def _assign_components(self, X: np.ndarray, means: np.ndarray, covs: np.ndarray, mix_logit: np.ndarray):
        """Return argmax assignments and per-component log-scores with mixture weights."""
        n_events = X.shape[0]
        n_components = means.shape[0]
        logps = np.empty((n_events, n_components))
        logpi = mix_logit - np.logaddexp.reduce(mix_logit)  # log-softmax

        for k in range(n_components):
            try:
                rv = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=False)
                logps[:, k] = rv.logpdf(X) + logpi[k]
            except Exception:
                logps[:, k] = -1e12
        return np.argmax(logps, axis=1), logps

    def _soft_responsibilities(self, logps: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = logps / max(temperature, 1e-8)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def _canonicalize_components(self, means: np.ndarray, covs: np.ndarray, mix_logit: np.ndarray, key: str = "mean0"):
        """Deterministically order components to break label switching symmetry."""
        if key == "mean0":
            order = np.argsort(means[:, 0])
        elif key == "weight":
            w = np.exp(mix_logit - np.logaddexp.reduce(mix_logit))
            order = np.argsort(-w)
        else:
            trace = np.array([np.trace(c) for c in covs])
            order = np.argsort(trace)
        return means[order], covs[order], mix_logit[order], order

    def _bin_significance(self, s: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        b = np.maximum(b, eps)
        r = s / b
        Z_asimov = np.sqrt(2.0 * ((s + b) * np.log1p(r) - s))
        Z_gauss  = s / np.sqrt(b)
        return np.where(r < 0.1, Z_gauss, Z_asimov)

    def _reindex_by_descending_significance(self, means, covs, mix_logit, hist_dict):
        """Reorder components by descending per-bin Z and reorder `hist_dict` accordingly."""
        s = np.add.reduce([hist_dict[l] for l in self.signal_label_lst])
        b = np.add.reduce([hist_dict[l] for l in self.bkg_label_lst])
        Zb = self._bin_significance(s, b)      # (K,)
        order = np.argsort(-Zb)                # descending

        means     = means[order]
        covs      = covs[order]
        mix_logit = mix_logit[order]
        hist_dict = {k: v[order] for k, v in hist_dict.items()}
        return means, covs, mix_logit, hist_dict, order

    # -- Objective for optuna -----------------------------------
    def objective(self, trial, data):
        """Optimize *in the same assignment mode used for reporting* (default: hard)."""
        n_dims = self._infer_n_dims(data)

        means, tril_raw, mix_logit = self._build_param_vector(trial, n_dims)
        covs = self._tril_to_cov(tril_raw)
        means, covs, mix_logit, _ = self._canonicalize_components(means, covs, mix_logit, key="mean0")

        hist_dict, sumsq_dict = {}, {}

        for lab, entry in data.items():
            if isinstance(entry, pd.DataFrame):
                X_full = np.vstack(entry[self.var_label].to_numpy())
                w = entry[self.weight_label].to_numpy()
            else:
                X_full, w = entry
            X = self._slice_dims(np.asarray(X_full))
            w = np.ones(X.shape[0], dtype=float) if w is None else np.asarray(w, dtype=float)

            hard_idx, logps = self._assign_components(X, means, covs, mix_logit)

            if self.assign_mode == "soft":
                gamma = self._soft_responsibilities(logps, temperature=self.temperature)
                hist  = (gamma * w[:, None]).sum(axis=0)
                ssq   = ((gamma * w[:, None]) ** 2).sum(axis=0)
            else:
                # hard
                hist = np.zeros(self.n_components, dtype=float)
                ssq  = np.zeros(self.n_components, dtype=float)
                for k in range(self.n_components):
                    sel = (hard_idx == k)
                    if sel.any():
                        ww = w[sel]
                        hist[k] = float(ww.sum())
                        ssq[k]  = float((ww ** 2).sum())

            hist_dict[lab], sumsq_dict[lab] = hist, ssq

        # metrics & penalties (on this assignment mode)
        self.compute_and_store_metrics(hist_dict, sumsq_dict)
        combined_Z = float(self.best_metrics.get("combined_Z", 0.0))

        bkg_counts = np.add.reduce([hist_dict[l]  for l in self.bkg_label_lst])
        bkg_sumsq  = np.add.reduce([sumsq_dict[l] for l in self.bkg_label_lst])
        P_low = self._compute_low_penalty(bkg_counts, int(self.min_bkg_per_bin))
        P_unc = self._compute_unc_penalty(bkg_counts, bkg_sumsq, float(self.rel_unc_threshold))
        self._last_penalties = {"P_low": float(P_low), "P_unc": float(P_unc)}

        score = combined_Z - float(self.penalty_low_lambda) * P_low - float(self.penalty_unc_lambda) * P_unc
        return float(score)

    # -- Run wrapper --------------------------------------------
    def run(self, data=None, n_trials: int = None, **optuna_kwargs):
        import optuna

        if data is None:
            data = getattr(self, "df_dict", None)
        if n_trials is None:
            n_trials = int(getattr(self, "n_trials", 50))

        def _objective(trial):
            return self.objective(trial, data)

        sampler = optuna.samplers.TPESampler(
            gamma=self.gamma_fn(),
            multivariate=True,
            seed=42,
            n_startup_trials=50,
            group=True,
        )

        def _run_opt(n_local: int):
            if getattr(self, "study", None) is None:
                self.study = optuna.create_study(direction="maximize", sampler=sampler)
            self.study.optimize(_objective, n_trials=n_local, **optuna_kwargs)

        self._run_opt_with_beta_restarts(
            _run_opt,
            max_restarts=3,
            trials_total=n_trials,
            check_interval=self.restart_check_trials,
        )

        if getattr(self, "study", None) is None:
            return None

        # Reconstruct the *best* parameter set
        trial = self.study.best_trial
        n_dims = self._infer_n_dims(data)
        means, tril_raw, mix_logit = self._build_param_vector_from_trial(trial, n_dims)
        covs = self._tril_to_cov(tril_raw)
        means, covs, mix_logit, _ = self._canonicalize_components(means, covs, mix_logit, key="mean0")

        # Build counts/sumsq again in the chosen assignment mode
        hist_dict, sumsq_dict = {}, {}
        logpi = mix_logit - np.logaddexp.reduce(mix_logit)

        for lab, entry in data.items():
            if isinstance(entry, pd.DataFrame):
                X_full = np.vstack(entry[self.var_label].to_numpy())
                w = entry[self.weight_label].to_numpy()
            else:
                X_full, w = entry
            X = self._slice_dims(np.asarray(X_full))
            w = np.ones(X.shape[0], dtype=float) if w is None else np.asarray(w, dtype=float)

            rv_logps = np.stack([
                multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True).logpdf(X)
                for k in range(self.n_components)
            ], axis=1)
            logps = rv_logps + logpi

            if self.assign_mode == "soft":
                gamma = self._soft_responsibilities(logps, temperature=self.temperature)
                hist = (gamma * w[:, None]).sum(axis=0)
                ssq  = ((gamma * w[:, None]) ** 2).sum(axis=0)
            else:
                hard_idx = np.argmax(logps, axis=1)
                hist = np.zeros(self.n_components, dtype=float)
                ssq  = np.zeros(self.n_components, dtype=float)
                for k in range(self.n_components):
                    sel = (hard_idx == k)
                    if sel.any():
                        ww = w[sel]
                        hist[k] = float(ww.sum())
                        ssq[k]  = float((ww ** 2).sum())

            hist_dict[lab], sumsq_dict[lab] = hist, ssq

        # Reorder components by descending bin significance ON THE SAME COUNTS
        means, covs, mix_logit, hist_dict, order = self._reindex_by_descending_significance(
            means, covs, mix_logit, hist_dict
        )

        # Finalize outputs
        self.best_components = [{"mean": means[k], "cov": covs[k]} for k in range(self.n_components)]
        self.best_hist_dict = hist_dict
        self.compute_and_store_metrics(hist_dict, sumsq_dict)

        # penalties & best score
        bkg_counts = np.add.reduce([hist_dict[l]  for l in self.bkg_label_lst])
        bkg_sumsq  = np.add.reduce([sumsq_dict[l] for l in self.bkg_label_lst])
        P_low = self._compute_low_penalty(bkg_counts, int(self.min_bkg_per_bin))
        P_unc = self._compute_unc_penalty(bkg_counts, bkg_sumsq, float(self.rel_unc_threshold))
        self.best_metrics.setdefault("penalties", {})
        self.best_metrics["penalties"]["P_low"] = float(P_low)
        self.best_metrics["penalties"]["P_unc"] = float(P_unc)

        combined_Z = float(self.best_metrics.get("combined_Z", 0.0))
        self.best_score = float(combined_Z - float(self.penalty_low_lambda) * P_low - float(self.penalty_unc_lambda) * P_unc)

        self._last_study = self.study

        # Assign bins to the original data using the *reordered* components.
        # IMPORTANT: no second permutation (no rank_map). Argmax directly matches final bin order.
        try:
            self.assign_bins_to_data()
        except Exception:
            logger.exception("assign_bins_to_data failed")

        return self.best_components, self.best_hist_dict, self.best_score

    def assign_bins_to_data(self):
        """Assign each event a 'bin_index' using the FINAL (reordered) components.

        No extra permutation applied (no rank_map). This fixes the double-permutation bug.
        """
        if not hasattr(self, "best_components") or self.best_components is None:
            raise RuntimeError("No best components available. Run optimization first.")

        for lbl, df in self.df_dict.items():
            arr_full = np.vstack(df[self.var_label].to_numpy())
            arr = self._slice_dims(arr_full)
            logpdf = np.stack([
                multivariate_normal(mean=g["mean"], cov=g["cov"], allow_singular=True).logpdf(arr)
                for g in self.best_components
            ], axis=1)
            df["bin_index"] = np.argmax(logpdf, axis=1).astype(int)

    # -- Predict / visualize helpers ---------------------------
    def predict(self, X: np.ndarray):
        if not hasattr(self, "_last_study"):
            return None
        trial = self._last_study.best_trial
        keys = [k for k in trial.params.keys() if k.startswith("mu_")]
        if not keys:
            return None
        _, k_str, d_str = keys[0].split("_")
        n_dims = int(d_str) + 1
        means, tril_raw, mix_logit = self._build_param_vector_from_trial(trial, n_dims)
        covs = self._tril_to_cov(tril_raw)
        return self._assign_components(X[:, :n_dims], means, covs)

    def _build_param_vector_from_trial(self, trial, n_dims: int):
        means = np.zeros((self.n_components, n_dims))
        tril_raw = np.zeros((self.n_components, n_dims, n_dims))
        mix_logit = np.zeros(self.n_components)
        for k in range(self.n_components):
            mix_logit[k] = trial.params.get(f"mix_logit_{k}", 0.0)
            for d in range(n_dims):
                means[k, d] = trial.params.get(f"mu_{k}_{d}", 0.0)
                for d2 in range(d + 1):
                    tril_raw[k, d, d2] = trial.params.get(f"L_{k}_{d}_{d2}", 0.0)
        return means, tril_raw, mix_logit

    # ---------------- Visualization helpers -----------------
    def _plot_ellipse(self, mean: np.ndarray, cov: np.ndarray, dims: Tuple[int, int], ax, color: str, label: str = None):
        subcov = cov[np.ix_(dims, dims)]
        vals, vecs = np.linalg.eigh(subcov)
        angle = math.degrees(math.atan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        ell = Ellipse(mean[list(dims)], width, height, angle=angle, edgecolor=color, facecolor='none', linewidth=2)
        ax.add_patch(ell)
        if label is not None:
            ax.text(mean[dims[0]], mean[dims[1]], label, color=color, ha='center', va='center')

    def visualize_labelled_ellipses(self, sample_frac: float = 1.0):
        """Plot per-class clouds and final-component ellipses in each 2D pair."""
        dims_list = [(0, 1), (0, 2), (1, 2)]
        for dims in dims_list:
            fig, ax = plt.subplots(figsize=(8, 6))
            for lbl, df in self.df_dict.items():
                alpha = 0.1 if lbl in self.bkg_label_lst else 0.2
                arr = np.vstack(df[self.var_label].to_numpy())
                if 0.0 < sample_frac < 1.0:
                    idx = np.random.choice(arr.shape[0], int(arr.shape[0] * sample_frac), replace=False)
                    sarr = arr[idx]
                else:
                    sarr = arr
                ax.scatter(sarr[:, dims[0]], sarr[:, dims[1]], s=5, alpha=alpha, label=lbl)

            proxies = []
            for k, g in enumerate(self.best_components):
                color = plt.cm.get_cmap('tab20', self.n_components)(k)
                self._plot_ellipse(g['mean'], g['cov'], dims, ax, color)
                proxies.append(Patch(edgecolor=color, facecolor='none', label=f'Bin {k}'))

            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            handles, labels = ax.get_legend_handles_labels()
            handles += proxies
            labels += [p.get_label() for p in proxies]
            ax.legend(handles, labels, ncol=2, fontsize=12)
            y0, y1 = ax.get_ylim()
            dy = (y1 - y0) * 0.1
            ax.set_ylim(y0 - dy, y1 + dy)
            plt.tight_layout()
            fig.savefig(self.output_dir + f"/labelled_ellipse_{dims[0]}{dims[1]}.pdf")
            plt.clf()

    def visualize_bins_2d(self):
        """Color points by their assigned final bin_index in each 2D score pair."""
        dims_list = [(0, 1), (0, 2), (1, 2)]
        for dims in dims_list:
            all_scores = []
            all_bins = []
            for df in self.df_dict.values():
                arr = np.vstack(df[self.var_label].to_numpy())
                bins = df['bin_index'].to_numpy()
                all_scores.append(arr)
                all_bins.append(bins)
            scores = np.vstack(all_scores)
            bins = np.concatenate(all_bins)

            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(scores[:, dims[0]], scores[:, dims[1]], c=bins,
                            cmap='tab20', vmin=0, vmax=self.n_components - 1, s=10, alpha=0.2)
            proxies = [Patch(color=plt.cm.get_cmap('tab20', self.n_components)(k), label=f'Bin {k}')
                       for k in range(self.n_components)]
            ax.legend(handles=proxies, ncol=2)
            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            fig.savefig(self.output_dir + f"/bins_2d_{dims[0]}{dims[1]}.pdf")
            plt.clf()

    def visualize_bin_boundaries(self, resolution: int = 1000):
        """
        Always draw (0,1), (0,2), (1,2) simplex-pair plots, independent of dims_to_use.
        Points p=(p0,p1,p2) are projected to model space q=p[dims_to_use] then
        evaluated under the final components. Bins are numbered in final order.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from scipy.stats import multivariate_normal

        if not hasattr(self, "best_components") or self.best_components is None:
            print("No best components to visualize; run optimization first.")
            return

        try:
            k = int(np.asarray(self.best_components[0]["mean"]).shape[0])
        except Exception:
            print("Cannot infer component dimensionality; skipping visualization.")
            return

        model_dims = list(self.dims_to_use) if getattr(self, "dims_to_use", None) else list(range(k))

        os.makedirs(self.output_dir, exist_ok=True)
        colors = [plt.cm.get_cmap('tab20')(i) for i in range(self.n_components)]
        cmap_bins = ListedColormap(colors)
        bounds = np.arange(self.n_components + 1) - 0.5
        norm = BoundaryNorm(bounds, cmap_bins.N)

        def eval_logpdf_k(Pk: np.ndarray) -> np.ndarray:
            N = Pk.shape[0]
            out = np.zeros((N, self.n_components))
            for idx, g in enumerate(self.best_components):
                rv = multivariate_normal(mean=g["mean"], cov=g["cov"], allow_singular=True)
                out[:, idx] = rv.logpdf(Pk)
            return out

        pairs = [(0, 1), (0, 2), (1, 2)]
        xs = np.linspace(0.0, 1.0, resolution)
        ys = np.linspace(0.0, 1.0, resolution)

        for (i, j) in pairs:
            X, Y = np.meshgrid(xs, ys)
            mask = (X + Y <= 1.0)
            if not np.any(mask):
                continue

            P3 = np.zeros((int(mask.sum()), 3), dtype=float)
            P3[:, i] = X[mask]
            P3[:, j] = Y[mask]
            rem = 3 - i - j
            P3[:, rem] = 1.0 - P3[:, i] - P3[:, j]

            q = P3[:, model_dims]  # project to model dims

            logps = eval_logpdf_k(q)
            assign = np.full(X.shape, np.nan)
            assign[mask] = np.argmax(logps, axis=1)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.contourf(X, Y, assign, levels=bounds, cmap=cmap_bins, norm=norm, alpha=0.6)
            ax.contour(X, Y, assign, levels=bounds, colors='k', linewidths=0.8)

            for c in range(self.n_components):
                xi = X[assign == c]
                yi = Y[assign == c]
                if xi.size:
                    ax.text(float(xi.mean()), float(yi.mean()), str(c),
                            color=colors[c], fontsize=10, fontweight='bold',
                            ha='center', va='center')

            proxies = [Patch(color=colors[c], label=f'Bin {c}') for c in range(self.n_components)]
            ax.legend(handles=proxies, ncol=2, fontsize=9, loc='upper right')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f"score_{i}")
            ax.set_ylabel(f"score_{j}")
            #ax.set_title(f"Bin regions (dims={i},{j})")
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"bin_boundaries_{i}{j}.pdf"))
            plt.clf()
