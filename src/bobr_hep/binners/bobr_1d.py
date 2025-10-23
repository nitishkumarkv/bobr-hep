from typing import Dict, List, Tuple
import numpy as np
import optuna
from .base import bobr_base


class bobr_1d(bobr_base):
    def run(self) -> Tuple[List[float], Dict[str, 'np.ndarray'], float]:
        min_edge, max_edge = self.min_edge, self.max_edge

        def objective(trial: optuna.trial.Trial) -> float:
            bin_edges = [min_edge]
            for i in range(self.n_bins - 1):
                low = bin_edges[-1]
                val = trial.suggest_float(f"bin_{i}", low, max_edge)
                bin_edges.append(val)
            bin_edges.append(max_edge)

            if not all(bin_edges[i] < bin_edges[i + 1] for i in range(len(bin_edges) - 1)):
                raise optuna.TrialPruned()
            # get counts and sumw2 per bin
            hist, sumsq = self._compute_1d_counts_and_sumsq(bin_edges)

            # background counts per bin
            bkg = np.sum([hist[l] for l in self.bkg_label_lst], axis=0)

            # compute penalties
            P_low = self._compute_low_penalty(bkg, self.min_bkg_per_bin)
            bkg_sumsq = np.sum([sumsq[l] for l in self.bkg_label_lst], axis=0)
            P_unc = self._compute_unc_penalty(bkg, bkg_sumsq, self.rel_unc_threshold)

            # significance
            s = hist[self.signal_label_lst[0]]
            Z = self.asymptotic_significance(s, bkg)

            # objective: significance minus weighted penalties
            obj = float(Z) - float(self.penalty_low_lambda) * float(P_low) - float(self.penalty_unc_lambda) * float(P_unc)
            return obj

        def _run_opt(n_trials_local: int):
            sampler = optuna.samplers.TPESampler(gamma=self.gamma_fn())
            # create study only if it doesn't exist so multiple calls accumulate trials
            if getattr(self, "study", None) is None:
                self.study = optuna.create_study(direction="maximize", sampler=sampler)
            self.study.optimize(objective, n_trials=n_trials_local)

        # run optimization with automatic beta-halving restarts if objective stays non-positive
        # pass the total n_trials so the helper can checkpoint using restart_check_trials
        self._run_opt_with_beta_restarts(_run_opt, max_restarts=3, trials_total=self.n_trials, check_interval=self.restart_check_trials)

        trial = self.study.best_trial
        self.best_bins = [min_edge] + [trial.params[f"bin_{i}"] for i in range(self.n_bins - 1)] + [max_edge]
        self.best_hist_dict, sumsq = self._compute_1d_counts_and_sumsq(self.best_bins)

        bkg = np.sum([self.best_hist_dict[l] for l in self.bkg_label_lst], axis=0)
        P_low = self._compute_low_penalty(bkg, self.min_bkg_per_bin)
        bkg_sumsq = np.sum([sumsq[l] for l in self.bkg_label_lst], axis=0)
        P_unc = self._compute_unc_penalty(bkg, bkg_sumsq, self.rel_unc_threshold)

        s = self.best_hist_dict[self.signal_label_lst[0]]
        b = np.sum([arr for lbl, arr in self.best_hist_dict.items() if lbl != self.signal_label_lst[0]], axis=0)
        self.best_score = float(self.asymptotic_significance(s, b) - (self.penalty_low_lambda * P_low + self.penalty_unc_lambda * P_unc))
        # store metrics
        self.compute_and_store_metrics(self.best_hist_dict, sumsq)
        return self.best_bins, self.best_hist_dict, self.best_score

    def predict(self, X) -> 'np.ndarray':
        """Assign 1D scores in X to bin indices using the learned `best_bins`.

        Returns indices in [0, n_bins-1].
        """
        if self.best_bins is None:
            raise RuntimeError("No bins found. Run `run()` first.")
        arr = np.asarray(X).squeeze()
        if arr.ndim == 0:
            arr = arr.reshape(1)
        inds = np.searchsorted(self.best_bins, arr, side="right") - 1
        inds = np.clip(inds, 0, self.n_bins - 1)
        return inds
