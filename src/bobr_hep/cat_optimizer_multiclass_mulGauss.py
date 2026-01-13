#!/usr/bin/env python
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse, Patch


class MixtureGMMBinOptimizer:
    """
    Bayesian Optimization of mixture-of-Gaussians bins in 3D NN-score space.
    Each of the `n_bins` bins is modeled as a mixture of `M` Gaussian components.
    """
    def __init__(
        self,
        df_dict: Dict[str,pd.DataFrame],
        bkg_label_lst:    List[str],
        signal_label_lst: List[str],
        var_label:        str,
        weight_label:     str,
        n_bins:           int     = 10,
        M:                int     = 2,
        n_trials:         int     = 20,
        output_dir:       str     = "./mix_optimizer_results",
        combination:      Literal["quadrature","geometric","harmonic"] = "quadrature"
    ):
        self.df_dict          = df_dict
        self.bkg_label_lst    = bkg_label_lst
        self.signal_label_lst = signal_label_lst
        self.var_label        = var_label
        self.weight_label     = weight_label
        self.n_bins           = n_bins
        self.M                = M
        self.n_trials         = n_trials
        self.output_dir       = output_dir
        self.combination      = combination
        self.cmap             = plt.cm.get_cmap('tab20', n_bins)
        self.study            = None
        self.best_mixtures    = None
        self.best_hist_dict   = None
        self.best_Z           = None

    def asymptotic_significance_(
        self, s: np.ndarray, b: np.ndarray,
        eps: float = 1e-10, ratio_threshold: float = 0.1
    ) -> float:
        """
        Combined per-bin significance with thresholded Asimov approximation.
        """
        safe_b = np.maximum(b, eps)
        ratio = s / safe_b
        Z_asimov = np.sqrt(2.0 * ((s + safe_b) * np.log1p(ratio) - s))
        Z_approx = s / np.sqrt(safe_b)
        Z_per_bin = np.where(ratio < ratio_threshold, Z_approx, Z_asimov)
        return float(np.sqrt(np.sum(Z_per_bin**2)))

    def mixture_pdf(
        self, x: np.ndarray, mixture: List[Dict]
    ) -> np.ndarray:
        """
        Evaluate mixture PDF bulk: sum w_m * N(x|mu_m, cov_m).
        """
        pdf_vals = np.zeros(x.shape[0])
        for comp in mixture:
            rv = multivariate_normal(
                mean=comp['mean'], cov=comp['cov'], allow_singular=True
            )
            pdf_vals += comp['weight'] * rv.pdf(x)
        return pdf_vals

    def assign_events(
        self,
        mixtures: List[List[Dict]],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Assign each event to the bin whose mixture PDF is largest.
        """
        N = x.shape[0]
        pdf_matrix = np.zeros((N, self.n_bins))
        for k, mix in enumerate(mixtures):
            pdf_matrix[:,k] = self.mixture_pdf(x, mix)
        return np.argmax(pdf_matrix, axis=1)

    def compute_bin_counts(
        self, mixtures: List[List[Dict]]
    ) -> Dict[str,np.ndarray]:
        """
        Weight-sum of each label in each bin assignment.
        """
        scores, wts, labels = [], [], []
        for lbl, df in self.df_dict.items():
            arr = np.vstack(df[self.var_label].to_numpy())
            scores.append(arr)
            wts.append(df[self.weight_label].to_numpy())
            labels.extend([lbl]*len(df))
        X = np.vstack(scores)
        W = np.concatenate(wts)
        labels = np.array(labels)
        assgn = self.assign_events(mixtures, X)
        counts = {}
        for lbl in self.df_dict:
            arr = np.zeros(self.n_bins)
            mask = (labels==lbl)
            for k in range(self.n_bins):
                arr[k] = W[(assgn==k)&mask].sum()
            counts[lbl] = arr
        return counts

    def objective(self, trial: optuna.Trial) -> float:
        """Bayesian objective: build mixtures, compute Z."""
        mixtures = []
        D = len(self.df_dict[self.signal_label_lst[0]][self.var_label].iloc[0])
        for k in range(self.n_bins):
            raw = np.array([
                trial.suggest_float(f'w_{k}_{m}', 1e-2, 1.0)
                for m in range(self.M)
            ])
            weights = raw / raw.sum()
            comps = []
            for m in range(self.M):
                mu = np.array([
                    trial.suggest_float(f'mu_{k}_{m}_{d}', 0.0, 1.0)
                    for d in range(D)
                ])
                L = np.zeros((D,D))
                for i in range(D):
                    for j in range(i+1):
                        nm = f'L_{k}_{m}_{i}{j}'
                        if i==j:
                            L[i,j] = trial.suggest_float(nm, 1e-4, 1.0, log=True)
                        else:
                            L[i,j] = trial.suggest_float(nm, -0.5, 0.5)
                cov = L @ L.T
                comps.append({'weight':weights[m],'mean':mu,'cov':cov})
            mixtures.append(comps)
        self.best_mixtures = mixtures
        counts = self.compute_bin_counts(mixtures)
        bkg_total = np.sum([counts[b] for b in self.bkg_label_lst], axis=0)
        if np.any(bkg_total < 10):
            return -float((bkg_total < 10).sum())
        def Zsig(lbl):
            s = counts[lbl]
            b = np.sum([counts[x] for x in counts if x!=lbl],axis=0)
            return self.asymptotic_significance_(s,b)
        Z1 = Zsig(self.signal_label_lst[0])
        Z2 = Zsig(self.signal_label_lst[1])
        if self.combination=='quadrature': Z = math.hypot(Z1,Z2)
        elif self.combination=='geometric': Z = math.sqrt(max(Z1,0)*max(Z2,0))
        else: Z = 2.0/(1.0/(Z1+1e-12)+1.0/(Z2+1e-12))
        return Z

    def optimize_bins(self
    ) -> Tuple[List[List[Dict]],Dict[str,np.ndarray],float]:
        os.makedirs(self.output_dir, exist_ok=True)
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=self.n_trials)
        self.best_Z = self.study.best_value
        # best_mixtures already stored
        self.best_hist_dict = self.compute_bin_counts(self.best_mixtures)
        return self.best_mixtures, self.best_hist_dict, self.best_Z

    def assign_bins_to_data(self):
        """
        Append 'bin_index' column to each DF via mixture assignment.
        """
        for lbl, df in self.df_dict.items():
            arr = np.vstack(df[self.var_label].to_numpy())
            df['bin_index'] = self.assign_events(self.best_mixtures, arr)

    def assign_bins_to_data(self):
        """
        Append 'bin_index' column to each DF via mixture assignment,
        ordering bins by combined per-bin Z.
        Also stores Z1_sum_quad, Z2_sum_quad, and rank_map.
        """
        # compute per-bin signal1 vs background Z
        counts = self.best_hist_dict
        # per-bin significance function
        def per_bin_Z_arr(sig_label):
            s = counts[sig_label]
            b = np.sum([counts[l] for l in counts if l != sig_label], axis=0)
            # thresholded Asimov formula per bin
            eps = 1e-10
            ratio = s / np.maximum(b, eps)
            Z_asimov = np.sqrt(2.0 * ((s + b) * np.log1p(ratio) - s))
            Z_approx = s / np.sqrt(np.maximum(b, eps))
            Z_bin = np.where(ratio < 0.1, Z_approx, Z_asimov)
            return Z_bin
        Z1_bins = per_bin_Z_arr(self.signal_label_lst[0])
        Z2_bins = per_bin_Z_arr(self.signal_label_lst[1])
        # store quadrature sums over all bins
        self.Z1_sum_quad = np.sqrt(np.sum(Z1_bins**2))
        self.Z2_sum_quad = np.sqrt(np.sum(Z2_bins**2))
        # combined per-bin Z for ranking
        if self.combination == 'quadrature':
            Zb = np.hypot(Z1_bins, Z2_bins)
        elif self.combination == 'geometric':
            Zb = np.sqrt(np.clip(Z1_bins,0,None) * np.clip(Z2_bins,0,None))
        else:
            Zb = 2.0 / (1.0/(Z1_bins+1e-12) + 1.0/(Z2_bins+1e-12))
        # rank bins: highest Zb -> rank 0, etc.
        order = np.argsort(-Zb)
        self.rank_map = {orig: rank for rank, orig in enumerate(order)}
        # assign each event based on original mixture
        for lbl, df in self.df_dict.items():
            arr = np.vstack(df[self.var_label].to_numpy())
            orig_bins = self.assign_events(self.best_mixtures, arr)
            df['bin_index'] = [self.rank_map[b] for b in orig_bins]

            arr = np.vstack(df[self.var_label].to_numpy())
            df['bin_index'] = self.assign_events(self.best_mixtures, arr)

    def _plot_ellipse(self,mean:np.ndarray,cov:np.ndarray,dims:Tuple[int,int],ax,color:str):
        sub = cov[np.ix_(dims,dims)]
        vals, vecs = np.linalg.eigh(sub)
        ang = math.degrees(math.atan2(*vecs[:,0][::-1]))
        w,h = 2*np.sqrt(vals)
        ell = Ellipse(mean[list(dims)], w, h, angle=ang,
                      edgecolor=color,facecolor='none',linewidth=2)
        ax.add_patch(ell)

    def visualize_labelled_ellipses(self):
        """
        Scatter true labels + ellipses of each mixture comp.
        """
        if 'bin_index' not in next(iter(self.df_dict.values())).columns:
            self.assign_bins_to_data()
        dims_list = [(0,1),(0,2),(1,2)]
        for dims in dims_list:
            fig, ax = plt.subplots(figsize=(8,6))
            for lbl, df in self.df_dict.items():
                arr = np.vstack(df[self.var_label].to_numpy())
                ax.scatter(arr[:,dims[0]], arr[:,dims[1]], s=10, alpha=0.5, label=lbl)
            proxies = []
            for k, mix in enumerate(self.best_mixtures):
                col = self.cmap(k)
                for comp in mix:
                    self._plot_ellipse(comp['mean'], comp['cov'], dims, ax, col)
                proxies.append(Patch(edgecolor=col,facecolor='none',label=f'Bin {k}'))
            handles, labels = ax.get_legend_handles_labels()
            handles += proxies; labels += [p.get_label() for p in proxies]
            ax.legend(handles, labels, ncol=2)
            ax.set_xlabel(f"score_{dims[0]}"); ax.set_ylabel(f"score_{dims[1]}")
            plt.tight_layout(); fig.savefig(Path(self.output_dir)/f'mix_labelled_ellipse_{dims[0]}{dims[1]}.png')
            plt.clf()

    def visualize_bins_2d(self):
        """
        2D scatter colored by assigned bin_index.
        """
        if 'bin_index' not in next(iter(self.df_dict.values())).columns:
            self.assign_bins_to_data()
        dims_list = [(0,1),(0,2),(1,2)]
        for dims in dims_list:
            fig, ax = plt.subplots(figsize=(8,6))
            all_s, all_b = [], []
            for df in self.df_dict.values():
                arr = np.vstack(df[self.var_label].to_numpy())
                all_s.append(arr); all_b.append(df['bin_index'].to_numpy())
            s = np.vstack(all_s); b = np.concatenate(all_b)
            ax.scatter(s[:,dims[0]], s[:,dims[1]], c=b, cmap=self.cmap,
                       vmin=0,vmax=self.n_bins-1, s=10, alpha=0.6)
            proxies = [Patch(color=self.cmap(k),label=f'Bin {k}') for k in range(self.n_bins)]
            ax.legend(handles=proxies,ncol=2)
            ax.set_xlabel(f"score_{dims[0]}"); ax.set_ylabel(f"score_{dims[1]}")
            plt.tight_layout(); fig.savefig(Path(self.output_dir)/f'mix_bins2d_{dims[0]}{dims[1]}.png')
            plt.clf()


if __name__ == "__main__":
    data = generate_toy_data_3D(seed=42)
    optimizer = MixtureGMMBinOptimizer(
        df_dict=data,
        bkg_label_lst=["bkg1","bkg2","bkg3","bkg4","bkg5"],
        signal_label_lst=["signal1","signal2"],
        var_label="NN_output",
        weight_label="weight",
        n_bins=5,
        M=2,
        n_trials=30,
        output_dir="./mix_results",
        combination="quadrature"
    )
    mixtures, hist_dict, best_Z = optimizer.optimize_bins()
    print("Best combined Z =", best_Z)
    optimizer.visualize_labelled_ellipses()
    optimizer.visualize_bins_2d()
