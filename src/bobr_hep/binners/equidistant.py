from typing import Dict, List, Tuple
import numpy as np
from .base import bobr_base


class equidistant(bobr_base):
    def run(self) -> Tuple[List[float], Dict[str, np.ndarray], float]:
        min_edge, max_edge = self.min_edge, self.max_edge
        edges = np.linspace(min_edge, max_edge, self.n_bins + 1)
        self.best_bins = list(edges)

        # compute counts and sumsq
        hist, sumsq = self._compute_1d_counts_and_sumsq(self.best_bins)

        # background counts per bin
        bkg = np.sum([hist[l] for l in self.bkg_label_lst], axis=0)

        # compute penalties
        P_low = self._compute_low_penalty(bkg, self.min_bkg_per_bin)
        bkg_sumsq = np.sum([sumsq[l] for l in self.bkg_label_lst], axis=0)
        P_unc = self._compute_unc_penalty(bkg, bkg_sumsq, self.rel_unc_threshold)

        # significance
        sig_label = self.signal_label_lst[0]
        s = hist[sig_label]
        b = np.sum([arr for lbl, arr in hist.items() if lbl != sig_label], axis=0)
        Z = self.asymptotic_significance(s, b)

        # final score: significance minus weighted penalties
        self.best_score = float(Z) - float(self.penalty_low_lambda) * float(P_low) - float(self.penalty_unc_lambda) * float(P_unc)
        self.best_hist_dict = hist
        # store metrics for later inspection
        self.compute_and_store_metrics(hist, sumsq)
        return self.best_bins, self.best_hist_dict, self.best_score

    def predict(self, X) -> np.ndarray:
        """Assign 1D scores in X to bin indices according to `self.best_bins`.

        X may be an array-like of shape (n,) or (n,1). Returns integer array
        of length n with indices in [0, n_bins-1].
        """
        if self.best_bins is None:
            raise RuntimeError("No bins found. Run `run()` first.")
        arr = np.asarray(X).squeeze()
        # handle (n,1) or scalar
        if arr.ndim == 0:
            arr = arr.reshape(1)
        inds = np.searchsorted(self.best_bins, arr, side="right") - 1
        inds = np.clip(inds, 0, self.n_bins - 1)
        return inds
