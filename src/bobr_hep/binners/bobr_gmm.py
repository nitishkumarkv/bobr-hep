"""Generic GMM-based binner implementation (lowercase class/file name).

This module provides `bobr_gmm`, a generic Gaussian Mixture Model based
binner. It intentionally does not embed any automatic N vs N-1 behavior.
The user explicitly passes `fit_label_lst` to choose which labels/dimensions
are used to fit the GMM.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import multivariate_normal

import logging
logger = logging.getLogger(__name__)
from .base import BOBRBase


class bobr_gmm(BOBRBase):
    """Generic GMM binner.

    Accepts `fit_label_lst` to choose which labels are used to fit the GMM
    and supports arbitrary number of labels/signals for metrics combination.
    """

    def __init__(
        self,
        n_components: int = 6,
        fit_label_lst: Optional[Sequence[int]] = None,
        combination: str = "quadrature",
        penalty: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_components = int(n_components)
        # If None, fit on all labels found in the dataset (0..n_labels-1)
        self.fit_label_lst = None if fit_label_lst is None else list(fit_label_lst)
        self.combination = combination
        self.penalty = float(penalty)

    # -- Helpers -------------------------------------------------
    def _build_param_vector(self, trial, n_dims: int) -> Tuple[np.ndarray, np.ndarray]:
        means = np.zeros((self.n_components, n_dims))
        tril_raw = np.zeros((self.n_components, n_dims, n_dims))
        for k in range(self.n_components):
            for d in range(n_dims):
                means[k, d] = trial.suggest_float(f"mu_{k}_{d}", -5.0, 5.0)
                for d2 in range(d + 1):
                    tril_raw[k, d, d2] = trial.suggest_float(f"L_{k}_{d}_{d2}", -3.0, 3.0)
        return means, tril_raw

    def _tril_to_cov(self, tril_raw: np.ndarray) -> np.ndarray:
        covs = []
        for k in range(tril_raw.shape[0]):
            L = np.tril(tril_raw[k])
            for i in range(L.shape[0]):
                L[i, i] = math.exp(L[i, i])
            cov = L @ L.T
            covs.append(cov)
        return np.array(covs)

    def _assign_components(self, X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
        n_events = X.shape[0]
        n_components = means.shape[0]
        logps = np.empty((n_events, n_components))
        for k in range(n_components):
            try:
                rv = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=False)
                logps[:, k] = rv.logpdf(X)
            except Exception:
                logps[:, k] = -1e12
        return np.argmax(logps, axis=1)

    # -- Objective for optuna -----------------------------------
    def objective(self, trial, data):
        labels = sorted(list(data.keys()))
        if self.fit_label_lst is None:
            labels_to_fit = labels
        else:
            labels_to_fit = [l for l in self.fit_label_lst if l in labels]

        first_X = data[labels_to_fit[0]][0]
        n_dims = first_X.shape[1]

        means, tril_raw = self._build_param_vector(trial, n_dims)
        covs = self._tril_to_cov(tril_raw)

        counts_per_label = {}
        sumsq_per_label = {}
        for lab in labels:
            X_lab, w_lab = data[lab]
            comp_assign = self._assign_components(X_lab[:, :n_dims], means, covs)
            counts = np.bincount(comp_assign, minlength=self.n_components)
            sumsq = np.zeros_like(counts, dtype=float)
            if w_lab is None:
                w = np.ones(X_lab.shape[0], dtype=float)
            else:
                w = w_lab
            for k in range(self.n_components):
                sumsq[k] = w[comp_assign == k].sum()
            counts_per_label[lab] = counts
            sumsq_per_label[lab] = sumsq

        n_labels = len(labels)
        hist = np.zeros((n_labels, self.n_components), dtype=float)
        sumsq = np.zeros_like(hist, dtype=float)
        for i, lab in enumerate(labels):
            hist[i, :] = counts_per_label[lab]
            sumsq[i, :] = sumsq_per_label[lab]

        combined_z = self.compute_and_store_metrics(hist, sumsq, labels=labels)
        best = float(np.nanmax(combined_z))
        obj = -(best - self.penalty)
        return obj

    # -- Run wrapper --------------------------------------------
    def run(self, data, n_trials: int = 200, **optuna_kwargs):
        import optuna

        def _objective(trial):
            return self.objective(trial, data)

        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=n_trials, **optuna_kwargs)
        self._last_study = study
        return study

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
        means, tril_raw = self._build_param_vector_from_trial(trial, n_dims)
        covs = self._tril_to_cov(tril_raw)
        return self._assign_components(X[:, :n_dims], means, covs)

    def _build_param_vector_from_trial(self, trial, n_dims: int):
        means = np.zeros((self.n_components, n_dims))
        tril_raw = np.zeros((self.n_components, n_dims, n_dims))
        for k in range(self.n_components):
            for d in range(n_dims):
                means[k, d] = trial.params.get(f"mu_{k}_{d}", 0.0)
                for d2 in range(d + 1):
                    tril_raw[k, d, d2] = trial.params.get(f"L_{k}_{d}_{d2}", 0.0)
        return means, tril_raw