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
import json
from pathlib import Path
import datetime


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
        var_label: Optional[str] = None,
        weight_label: Optional[str] = None,
        n_components: int = 6,
        dims_to_use: Optional[Sequence[int]] = None,
        combination: str = "quadrature",
        penalty: float = 0.0,
        assign_mode: str = "hard",
        temperature: float = 1.0,
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

        # Mean parameterization
        self.mean_norm = str(kwargs.pop("mean_norm", "softmax")).lower()  # "none" | "softmax" | "sigmoid"
        if self.mean_norm not in ("none", "softmax", "sigmoid"):
            raise ValueError("mean_norm must be one of: 'none', 'softmax', 'sigmoid'")

        # Optional range for sigmoid means (old-style box constraint)
        self.mean_range = kwargs.pop("mean_range", (float(self.min_edge), float(self.max_edge)))


    def _infer_n_dims(self, data) -> int:
        first_key = next(iter(sorted(data.keys())))
        sample = data[first_key]
        if isinstance(sample, pd.DataFrame):
            X0 = np.vstack(sample[self.var_label].to_numpy())
        else:
            X0 = sample[0]

        # cache full NN-output dimensionality (before slicing)
        self._full_dim = int(X0.shape[1])

        # model dimension after slicing
        if self.dims_to_use is None:
            return self._full_dim
        return len(self.dims_to_use)

    def _slice_dims(self, X: np.ndarray) -> np.ndarray:
        if self.dims_to_use is None:
            return X
        return X[:, self.dims_to_use]

    def _build_param_vector(self, trial, n_dims: int):
        means = np.zeros((self.n_components, n_dims))
        tril_raw = np.zeros((self.n_components, n_dims, n_dims))
        mix_logit = np.zeros(self.n_components)

        for k in range(self.n_components):
            mix_logit[k] = trial.suggest_float(f"mix_logit_{k}", -6.0, 6.0)

            for d in range(n_dims):
                # raw means (will be softmax-normalised later if needed)
                means[k, d] = trial.suggest_float(
                    f"mu_{k}_{d}", -4, 4
                )

                for d2 in range(d + 1):
                    if d2 == d:
                        # dimensionless raw log-scale
                        tril_raw[k, d, d2] = trial.suggest_float(
                            f"L_{k}_{d}_{d2}", -2.0, 2.0
                        )
                    else:
                        # dimensionless correlation control
                        tril_raw[k, d, d2] = trial.suggest_float(
                            f"L_{k}_{d}_{d2}", -1.0, 1.0
                        )

        return means, tril_raw, mix_logit


    def _effective_means(self, means_raw: np.ndarray) -> np.ndarray:
        """
        Map raw mean parameters into effective mean space.
    
        If mean_norm="softmax":
          - If we use ALL coords (model_dim == full_dim): softmax over model_dim -> sum==1
          - If we use a SUBSET (model_dim < full_dim): softmax over (model_dim+1) and drop last
            -> sums <= 1, matching (p0,p1) triangle from a 3-class softmax.
    
        If mean_norm="none": identity.
        If mean_norm="sigmoid": sigmoid + affine map into mean_range (box domain).
        """
        means_raw = np.asarray(means_raw, dtype=float)
    
        if self.mean_norm == "none":
            return means_raw
    
        if self.mean_norm == "sigmoid":
            lo, hi = self.mean_range
            span = float(hi) - float(lo)
            return float(lo) + (1.0 / (1.0 + np.exp(-means_raw))) * span
    
        # --- softmax mode ---
        model_dim = int(means_raw.shape[1])
        full_dim = int(getattr(self, "_full_dim", model_dim))
    
        if model_dim == full_dim:
            # using all outputs: means on full simplex (sum == 1)
            z = means_raw - means_raw.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
    
        if model_dim < full_dim:
            # using subset: represent as first model_dim coords of a (model_dim+1)-simplex (sum <= 1)
            zeros = np.zeros((means_raw.shape[0], 1), dtype=float)
            full = np.concatenate([means_raw, zeros], axis=1)  # (K, model_dim+1)
            full = full - full.max(axis=1, keepdims=True)
            e = np.exp(full)
            probs = e / e.sum(axis=1, keepdims=True)
            return probs[:, :model_dim]
    
        raise RuntimeError(f"Invalid dims: model_dim={model_dim} > full_dim={full_dim}")

    def _compute_sigma_base(self, model_dim: int) -> float:
        """
        GATO-style sigma_base but using the *intrinsic* simplex dimension when mean_norm='softmax'.

        For C-class simplex:
          intrinsic dimension = C - 1

        - If model_dim == full_dim (using all C coords), C = model_dim
          -> m = model_dim - 1
        - If model_dim < full_dim (using subset), the simplex is still C = full_dim
          -> m = full_dim - 1

        If not softmax, use m = model_dim (box-like space).
        """
        full_dim = int(getattr(self, "_full_dim", model_dim))

        if self.mean_norm == "softmax":
            C = full_dim  # underlying simplex size
            m = max(C - 1, 1)
            # use V_simp for the C-simplex (dimension m)
            V_simp = math.sqrt(C) / math.factorial(C - 1)
        else:
            m = max(model_dim, 1)
            # treat as box-like; V_simp not meaningful, but keep a sane scale:
            # use V_simp=1 so sigma_base ~ (1/(K*V_ball))^(1/m)
            V_simp = 1.0

        V_ball = math.pi ** (m / 2.0) / math.gamma(m / 2.0 + 1.0)
        return (V_simp / (self.n_components * V_ball)) ** (1.0 / m)


    def _tril_to_cov(self, tril_raw: np.ndarray, kappa: float = 0.1) -> np.ndarray:
        K, D, _ = tril_raw.shape

        if not hasattr(self, "_sigma_base"):
            self._sigma_base = self._compute_sigma_base(D)

        covs = []
        for k in range(K):
            L_raw = np.tril(tril_raw[k])

            # strictly lower-triangular, damped
            off = L_raw.copy()
            np.fill_diagonal(off, 0.0)
            off *= kappa

            # diagonal: sigma_base * exp(raw_diag)
            raw_diag = np.diag(L_raw)
            sigma = self._sigma_base * np.exp(raw_diag)
            Dmat = np.diag(sigma)

            L = off + Dmat
            covs.append(L @ L.T)

        return np.array(covs)

    def _assign_components(self, X: np.ndarray, means: np.ndarray, covs: np.ndarray, mix_logit: np.ndarray):
        """Return argmax assignments and per-component log-scores with mixture weights."""
        n_events = X.shape[0]
        n_components = means.shape[0]
        logps = np.empty((n_events, n_components))
        logpi = mix_logit - np.logaddexp.reduce(mix_logit)  # log-softmax

        for k in range(n_components):
            try:
                rv = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
                logps[:, k] = rv.logpdf(X) + logpi[k]
            except Exception:
                logps[:, k] = -1e12
        return np.argmax(logps, axis=1), logps

    def _soft_responsibilities(self, logps: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        z = logps / max(temperature, 1e-8)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def _canonicalize_components(self, means: np.ndarray, covs: np.ndarray, mix_logit: np.ndarray, tril_raw=None, key: str = "mean0"):
        """Deterministically order components to break label switching symmetry.

        Returns (means_c, covs_c, mix_logit_c, order, tril_raw_canonical_or_None).
        If `tril_raw` is provided it will be reordered to match the canonicalization
        and returned as the fifth value.
        """
        if key == "mean0":
            order = np.argsort(means[:, 0])
        elif key == "weight":
            w = np.exp(mix_logit - np.logaddexp.reduce(mix_logit))
            order = np.argsort(-w)
        else:
            trace = np.array([np.trace(c) for c in covs])
            order = np.argsort(trace)

        means_c = means[order]
        covs_c = covs[order]
        mix_logit_c = mix_logit[order]

        tril_c = None
        if tril_raw is not None:
            tril_arr = np.asarray(tril_raw)
            tril_c = tril_arr[order]

        return means_c, covs_c, mix_logit_c, tril_c, order

    def _bin_significance(self, s: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        b = np.maximum(b, eps)
        r = s / b
        Z_asimov = np.sqrt(2.0 * ((s + b) * np.log1p(r) - s))
        Z_gauss  = s / np.sqrt(b)
        return np.where(r < 0.1, Z_gauss, Z_asimov)

    def _reindex_by_descending_significance(self, means, covs, mix_logit, trial_raw, hist_dict):
        """Reorder components by descending per-bin Z and reorder `hist_dict` accordingly."""
        s = np.add.reduce([hist_dict[l] for l in self.signal_label_lst])
        b = np.add.reduce([hist_dict[l] for l in self.bkg_label_lst])
        Zb = self._bin_significance(s, b)      # (K,)
        order = np.argsort(-Zb)                # descending

        means     = means[order]
        covs      = covs[order]
        mix_logit = mix_logit[order]
        trial_raw = trial_raw[order]
        hist_dict = {k: v[order] for k, v in hist_dict.items()}
        return means, covs, mix_logit, trial_raw, hist_dict, order

    # -- Objective for optuna -----------------------------------
    def objective(self, trial, data):
        """Optimize *in the same assignment mode used for reporting* (default: hard)."""
        n_dims = self._infer_n_dims(data)

        means_raw, tril_raw, mix_logit = self._build_param_vector(trial, n_dims)
        means = self._effective_means(means_raw)          # <-- ADD THIS
        covs = self._tril_to_cov(tril_raw)
        means, covs, mix_logit, tril_raw, _ = self._canonicalize_components(means, covs, mix_logit, tril_raw, key="mean0")

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
            seed=self.seed_optimizer,
            n_startup_trials=100,
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
        means_raw, tril_raw, mix_logit = self._build_param_vector_from_trial(trial, n_dims)
        means = self._effective_means(means_raw)          # <-- ADD THIS
        covs = self._tril_to_cov(tril_raw)
        means, covs, mix_logit, tril_raw, _ = self._canonicalize_components(means, covs, mix_logit, tril_raw, key="mean0")

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
        means, covs, mix_logit, tril_raw, hist_dict, order = self._reindex_by_descending_significance(
            means, covs, mix_logit, tril_raw, hist_dict
        )

        # Finalize outputs
        self.best_components = [
            {"mean": means[k], "cov": covs[k], "logit": mix_logit[k], "tril_raw": tril_raw[k]}
            for k in range(self.n_components)
        ]
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
        try:
            self.assign_bins_to_data()
        except Exception:
            logger.exception("assign_bins_to_data failed")

        return self.best_components, self.best_hist_dict, self.best_score

    def assign_bins_to_data(self):
        """Assign bins using FINAL (mean, cov, logit)."""
        if not hasattr(self, "best_components") or self.best_components is None:
            raise RuntimeError("No best components available. Run optimization first.")

        # Extract reordered mixture logits from best_components
        mix_logit = np.array([c["logit"] for c in self.best_components])
        logpi = mix_logit - np.logaddexp.reduce(mix_logit)   # log π_k

        for lbl, df in self.df_dict.items():
            arr_full = np.vstack(df[self.var_label].to_numpy())
            X = self._slice_dims(arr_full)

            # log N_k(x)
            rv_logps = np.stack([
                multivariate_normal(
                    mean=c["mean"],
                    cov=c["cov"],
                    allow_singular=True,
                ).logpdf(X)
                for c in self.best_components
            ], axis=1)

            # score = logN + logπ
            logps = rv_logps + logpi[None, :]

            if self.assign_mode == "soft":
                gamma = self._soft_responsibilities(logps, temperature=self.temperature)
                df["bin_index"] = np.argmax(gamma, axis=1).astype(int)
            else:
                # hard assignment
                df["bin_index"] = np.argmax(logps, axis=1).astype(int)

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

        means_raw, tril_raw, mix_logit = self._build_param_vector_from_trial(trial, n_dims)
        means = self._effective_means(means_raw)          # <-- ADD THIS
        covs = self._tril_to_cov(tril_raw)
        return self._assign_components(X[:, :n_dims], means, covs, mix_logit)

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

    # ---------------- Checkpointing (JSON-only) ----------------
    def save_checkpoint(self, path_prefix: str) -> None:
        """Save checkpoint (single JSON) containing components, per-label hist/sumsq (if present), and metadata."""
        if not hasattr(self, "best_components") or self.best_components is None:
            raise RuntimeError("No best_components available. Run optimizer first.")

        p = Path(path_prefix)
        json_path = p.with_suffix(".json")

        params = {
            "n_components": int(self.n_components),
            "dims_to_use": None if self.dims_to_use is None else list(self.dims_to_use),
            "combination": self.combination,
            "penalty": float(self.penalty),
            "assign_mode": self.assign_mode,
            "temperature": float(self.temperature),
            "var_label": self.var_label,
            "weight_label": self.weight_label,
            "bkg_label_lst": list(self.bkg_label_lst),
            "signal_label_lst": list(self.signal_label_lst),
        }

        data = {
            "params": params,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "best_score": None if self.best_score is None else float(self.best_score),
            "best_metrics": getattr(self, "best_metrics", None),
        }

        # components: convert numpy arrays to lists
        comps = []
        for c in self.best_components:
            comp = {
                "mean": np.asarray(c["mean"]).tolist(),
                "cov": np.asarray(c["cov"]).tolist(),
                "logit": float(c.get("logit", 0.0)),
                "tril_raw": np.asarray(c.get("tril_raw", [])).tolist(),
            }
            comps.append(comp)
        data["components"] = comps

        # optionally include stored hist and sumsq
        if getattr(self, "best_hist_dict", None) is not None:
            data["hist"] = {k: np.asarray(v).tolist() for k, v in self.best_hist_dict.items()}
        if getattr(self, "_best_sumsq_dict", None) is not None:
            data["sumsq"] = {k: np.asarray(v).tolist() for k, v in getattr(self, "_best_sumsq_dict").items()}

        def _json_default(o):
            try:
                import numpy as _np
            except Exception:
                _np = None
            if _np is not None:
                if isinstance(o, _np.ndarray):
                    return o.tolist()
                if isinstance(o, _np.generic):
                    return o.item()
            if hasattr(o, "tolist"):
                try:
                    return o.tolist()
                except Exception:
                    pass
            if hasattr(o, "item"):
                try:
                    return o.item()
                except Exception:
                    pass
            return str(o)

        with open(json_path, "w") as jf:
            json.dump(data, jf, indent=2, default=_json_default)

    @classmethod
    def load_checkpoint(cls, path_prefix: str, df_dict: Optional[dict] = None) -> "bobr_gmm":
        """Load checkpoint JSON and return a populated `bobr_gmm` instance.

        If `df_dict` is provided it will be attached to the instance, but loading
        does not require it; you may call `apply_to_df()` explicitly later.
        """
        p = Path(path_prefix)
        json_path = p.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {json_path}")

        with open(json_path, "r") as jf:
            data = json.load(jf)

        params = data.get("params", {})

        inst = cls(
            df_dict or {},
            bkg_label_lst=params.get("bkg_label_lst", []),
            signal_label_lst=params.get("signal_label_lst", []),
            var_label=params.get("var_label", "NN_output"),
            weight_label=params.get("weight_label", "weight"),
            n_components=int(params.get("n_components", 0)),
            dims_to_use=params.get("dims_to_use", None),
            combination=params.get("combination", "geometric"),
            penalty=float(params.get("penalty", 0.0)),
            assign_mode=params.get("assign_mode", "hard"),
            temperature=float(params.get("temperature", 1.0)),
        )

        # restore components
        comps = data.get("components", [])
        best_components = []
        for c in comps:
            best_components.append({
                "mean": np.asarray(c.get("mean", []), dtype=float),
                "cov": np.asarray(c.get("cov", []), dtype=float),
                "logit": float(c.get("logit", 0.0)),
                "tril_raw": np.asarray(c.get("tril_raw", []), dtype=float),
            })
        inst.best_components = best_components
        inst.n_components = len(best_components)

        # restore hist / sumsq dicts if present
        hist_dict = {}
        sumsq_dict = {}
        if "hist" in data and isinstance(data["hist"], dict):
            for k, v in data["hist"].items():
                hist_dict[k] = np.asarray(v, dtype=float)
        if "sumsq" in data and isinstance(data["sumsq"], dict):
            for k, v in data["sumsq"].items():
                sumsq_dict[k] = np.asarray(v, dtype=float)

        inst.best_hist_dict = hist_dict or None
        inst._best_sumsq_dict = sumsq_dict or None

        inst.best_score = data.get("best_score")
        inst.best_metrics = data.get("best_metrics")

        if df_dict is not None:
            inst.df_dict = df_dict

        return inst

    def apply_to_df(self, df_dict: dict):
        """Apply stored components to new data and compute hist, sumsq and metrics.

        Returns (hist_dict, sumsq_dict, metrics)
        """
        if not hasattr(self, "best_components") or self.best_components is None:
            raise RuntimeError("No best_components available. Load a checkpoint or run optimizer first.")

        old_df = getattr(self, "df_dict", None)
        try:
            self.df_dict = df_dict

            means = np.array([c["mean"] for c in self.best_components])
            covs = np.array([c["cov"] for c in self.best_components])
            mix_logit = np.array([c.get("logit", 0.0) for c in self.best_components])

            hist_dict = {}
            sumsq_dict = {}
            for lab, entry in df_dict.items():
                if isinstance(entry, pd.DataFrame):
                    X_full = np.vstack(entry[self.var_label].to_numpy())
                    w = entry[self.weight_label].to_numpy()
                else:
                    X_full, w = entry
                X = self._slice_dims(np.asarray(X_full))
                w = np.ones(X.shape[0], dtype=float) if w is None else np.asarray(w, dtype=float)

                # logpdf per component
                rv_logps = np.stack([
                    multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True).logpdf(X)
                    for k in range(self.n_components)
                ], axis=1)
                logpi = mix_logit - np.logaddexp.reduce(mix_logit)
                logps = rv_logps + logpi

                if self.assign_mode == "soft":
                    gamma = self._soft_responsibilities(logps, temperature=self.temperature)
                    hist = (gamma * w[:, None]).sum(axis=0)
                    ssq = ((gamma * w[:, None]) ** 2).sum(axis=0)
                else:
                    hard_idx = np.argmax(logps, axis=1)
                    hist = np.zeros(self.n_components, dtype=float)
                    ssq = np.zeros(self.n_components, dtype=float)
                    for k in range(self.n_components):
                        sel = (hard_idx == k)
                        if sel.any():
                            ww = w[sel]
                            hist[k] = float(ww.sum())
                            ssq[k] = float((ww ** 2).sum())

                hist_dict[lab] = hist
                sumsq_dict[lab] = ssq

            # compute and store metrics
            self.compute_and_store_metrics(hist_dict, sumsq_dict)
            metrics = getattr(self, "best_metrics", {})
        finally:
            self.df_dict = old_df

        return hist_dict, sumsq_dict, metrics

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
        base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        tab = ["tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:brown", "black"]
        needed = self.n_components - (len(base) + len(tab))

        # Optionally define a helper for extra distinct colors if you want exact match:
        def get_distinct_colors(n):
            import colorsys
            hues = np.linspace(0, 1, n, endpoint=False)
            return [colorsys.hsv_to_rgb(h, 0.7, 0.9) for h in hues]
        
        extra = [] if needed < 1 else get_distinct_colors(needed)
        colors = (base + tab + extra)[: self.n_components]
        
        cmap_bins = ListedColormap(colors)
        bounds = np.arange(self.n_components + 1) - 0.5
        norm = BoundaryNorm(bounds, len(colors))

        def eval_logpdf_k(Pk: np.ndarray) -> np.ndarray:
            """
            Return scores s_k(x) = log N_k(x) + log π_k for all points in Pk.
            This makes the plotted bin boundaries match the real assignment logic.
            """
            # compute log π_k from stored logits
            mix_logit = np.array([c.get("logit", 0.0) for c in self.best_components], dtype=float)
            logpi = mix_logit - np.logaddexp.reduce(mix_logit)
        
            N = Pk.shape[0]
            out = np.zeros((N, self.n_components))
            for idx, g in enumerate(self.best_components):
                rv = multivariate_normal(mean=g["mean"], cov=g["cov"], allow_singular=True)
                out[:, idx] = rv.logpdf(Pk) + logpi[idx]   # <-- FIX: include mixture weight
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

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.contourf(X, Y, assign, levels=bounds, cmap=cmap_bins, norm=norm, alpha=0.6)
            ax.contour(X, Y, assign, levels=bounds, colors='k', linewidths=0.8)

            for c in range(self.n_components):
                xi = X[assign == c]
                yi = Y[assign == c]
                if xi.size:
                    ax.text(float(xi.mean()), float(yi.mean()), str(c),
                            color=colors[c], fontsize=15, fontweight='bold',
                            ha='center', va='center')

            proxies = [Patch(color=colors[c], label=f'Bin {c}') for c in range(self.n_components)]
            ax.legend(handles=proxies, ncol=2, fontsize=12, loc='upper right', labelspacing=0.4, columnspacing=1.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f"Discriminant dim. {i}")
            ax.set_ylabel(f"Discriminant dim. {j}")
            #ax.set_title(f"Bin regions (dims={i},{j})")
            plt.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"bin_boundaries_{i}{j}.pdf"))
            plt.clf()
