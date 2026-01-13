from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import optuna
import json
from pathlib import Path
import datetime
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
            sampler = optuna.samplers.TPESampler(
            gamma=self.gamma_fn(),
            seed=self.seed_optimizer,
        )
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

    def save_checkpoint(self, path_prefix: str) -> None:
        """Save a checkpoint consisting of JSON metadata and an NPZ of arrays.

        Writes two files: ``{path_prefix}.json`` (metadata) and
        ``{path_prefix}.npz`` (numpy arrays). JSON contains parameters
        required to reconstruct the binner and scalar values; NPZ stores
        arrays like `best_bins`, per-label histograms and sumsq arrays.

        This avoids pickle for portability and security. Only call when
        `self.best_bins` is available (after a successful `run()`).
        """
        if self.best_bins is None:
            raise RuntimeError("Cannot save checkpoint: best_bins is None. Run the optimizer first.")

        p = Path(path_prefix)
        json_path = p.with_suffix(".json")
        npz_path = p.with_suffix(".npz")

        # gather metadata
        meta: Dict[str, Any] = {}
        params = {
            "n_bins": int(self.n_bins),
            "var_label": self.var_label,
            "weight_label": self.weight_label,
            "gamma_strategy": self.gamma_strategy,
            "beta": float(self.beta),
            "min_edge": float(self.min_edge),
            "max_edge": float(self.max_edge),
            "min_bkg_per_bin": int(self.min_bkg_per_bin),
            "penalty_low_lambda": float(self.penalty_low_lambda),
            "penalty_unc_lambda": float(self.penalty_unc_lambda),
            "rel_unc_threshold": float(self.rel_unc_threshold),
            "restart_check_trials": int(self.restart_check_trials),
            "bkg_label_lst": list(self.bkg_label_lst),
            "signal_label_lst": list(self.signal_label_lst),
        }
        meta["params"] = params
        meta["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        meta["best_score"] = None if self.best_score is None else float(self.best_score)
        try:
            meta["best_metrics"] = self.best_metrics
        except Exception:
            meta["best_metrics"] = None

        # prepare data to save directly into JSON (lists instead of NPZ)
        data: Dict[str, Any] = {}
        data.update(meta)
        data["best_bins"] = [float(x) for x in self.best_bins]

        # if we have df_dict available, compute up-to-date hist and sumsq
        try:
            if self.df_dict is not None:
                hist_dict, sumsq_dict = self._compute_1d_counts_and_sumsq(self.best_bins)
            else:
                hist_dict = getattr(self, "best_hist_dict", None)
                # try to get stored sumsq if present on object
                sumsq_dict = getattr(self, "_best_sumsq_dict", None)
        except Exception:
            hist_dict = getattr(self, "best_hist_dict", None)
            sumsq_dict = getattr(self, "_best_sumsq_dict", None)

        if hist_dict is not None:
            data["hist"] = {lbl: [float(x) for x in arr] for lbl, arr in hist_dict.items()}
        if sumsq_dict is not None:
            data["sumsq"] = {lbl: [float(x) for x in arr] for lbl, arr in sumsq_dict.items()}

        # write single JSON file containing metadata and arrays
        def _json_default(o):
            # Convert numpy arrays and scalars to Python types for JSON
            try:
                import numpy as _np
            except Exception:
                _np = None
            if _np is not None:
                if isinstance(o, _np.ndarray):
                    return o.tolist()
                if isinstance(o, _np.generic):
                    return o.item()
            # try common conversions
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
            # fallback to string representation
            return str(o)

        with open(json_path, "w") as jf:
            json.dump(data, jf, indent=2, default=_json_default)

    @classmethod
    def load_checkpoint(cls, path_prefix: str, df_dict: Optional[Dict[str, Any]] = None) -> "bobr_1d":
        """Load a checkpoint (JSON + NPZ) and return a populated `bobr_1d` instance.

        If `df_dict` is provided it will be attached to the returned instance so
        `apply_to_df()`/`predict()` can be used immediately.
        """
        p = Path(path_prefix)
        json_path = p.with_suffix(".json")

        if not json_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {json_path}")

        with open(json_path, "r") as jf:
            data = json.load(jf)

        params = data.get("params", {})

        # prepare df_dict for constructor (can't reconstruct original df_dict from checkpoint)
        ctor_df = df_dict if df_dict is not None else {}

        # instantiate binner
        inst = cls(
            ctor_df,
            bkg_label_lst=params.get("bkg_label_lst", []),
            signal_label_lst=params.get("signal_label_lst", []),
            var_label=params.get("var_label", ""),
            weight_label=params.get("weight_label", "weight"),
            n_bins=int(params.get("n_bins", 0)),
            output_dir=".",
            gamma_strategy=params.get("gamma_strategy", "sqrt"),
            beta=float(params.get("beta", 0.25)),
            min_edge=float(params.get("min_edge", 0.0)),
            max_edge=float(params.get("max_edge", 1.0)),
            min_bkg_per_bin=int(params.get("min_bkg_per_bin", 0)),
            penalty_low_lambda=float(params.get("penalty_low_lambda", 1.0)),
            penalty_unc_lambda=float(params.get("penalty_unc_lambda", 1.0)),
            rel_unc_threshold=float(params.get("rel_unc_threshold", 0.1)),
            restart_check_trials=int(params.get("restart_check_trials", 200)),
        )

        # restore arrays from JSON
        if "best_bins" in data:
            inst.best_bins = [float(x) for x in data.get("best_bins", [])]
            inst.n_bins = len(inst.best_bins) - 1

        hist_dict = {}
        sumsq_dict = {}
        if "hist" in data and isinstance(data["hist"], dict):
            for lbl, arr in data["hist"].items():
                hist_dict[lbl] = np.asarray(arr, dtype=float)
        if "sumsq" in data and isinstance(data["sumsq"], dict):
            for lbl, arr in data["sumsq"].items():
                sumsq_dict[lbl] = np.asarray(arr, dtype=float)

        inst.best_hist_dict = {k: v for k, v in hist_dict.items()} if hist_dict else None
        inst._best_sumsq_dict = {k: v for k, v in sumsq_dict.items()} if sumsq_dict else None

        inst.best_score = data.get("best_score")
        inst.best_metrics = data.get("best_metrics")

        # attach df_dict if provided
        if df_dict is not None:
            inst.df_dict = df_dict

        return inst

    def apply_to_df(self, df_dict: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """Apply saved binning to a new dataset and compute counts, sumsq and metrics.

        Returns (hist_dict, sumsq_dict, metrics)
        """
        if self.best_bins is None:
            raise RuntimeError("No saved bins available. Load a checkpoint or run the optimizer first.")

        # temporarily set df_dict for computation, restore afterwards
        old_df = getattr(self, "df_dict", None)
        try:
            self.df_dict = df_dict
            hist, sumsq = self._compute_1d_counts_and_sumsq(self.best_bins)
            # store metrics in the object as well
            self.compute_and_store_metrics(hist, sumsq)
            metrics = getattr(self, "best_metrics", {})
        finally:
            # restore original df_dict
            self.df_dict = old_df

        return hist, sumsq, metrics
