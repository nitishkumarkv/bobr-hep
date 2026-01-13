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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.tri as mtri

class GMMBinOptimizer:
    """
    Bayesian Optimization of Gaussian‐Mixture “Bins” in 3D NN‐score space.
    Exactly like BOBRBinOptimizer but using K Gaussians instead of 1D boundaries.
    """
    def __init__(
        self,
        df_dict: Dict[str, pd.DataFrame],
        bkg_label_lst:    List[str],
        signal_label_lst: List[str],
        var_label:        str,     # = "NN_output"
        weight_label:     str,     # = "weight"
        n_bins:           int     = 10,
        n_trials:         int     = 10,
        output_dir:       str     = "./optimizer_results",
        gamma_strategy:   str     = "sqrt",  # placeholder for API match
        beta:             float   = 0.25,    # placeholder for API match
        combination:      Literal["quadrature","geometric","harmonic"] = "geometric"
    ):
        self.df_dict          = df_dict
        self.bkg_label_lst    = bkg_label_lst
        self.signal_label_lst = signal_label_lst
        self.var_label        = var_label
        self.weight_label     = weight_label
        self.n_bins           = n_bins
        self.n_trials         = n_trials
        self.output_dir       = output_dir
        self.gamma_strategy   = gamma_strategy
        self.beta             = beta
        self.combination      = combination
        self.cmap             = plt.cm.get_cmap('tab20', n_bins)

        self.study            = None
        self.best_gaussians   = None
        self.best_hist_dict   = None
        self.best_Z           = None
        self.Z1_sum_quad      = None
        self.Z2_sum_quad      = None
        self.rank_map         = None

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

    def asymptotic_significance(self, s, b, eps=1e-10, ratio_threshold=0.1):
        """
        Compute the combined Asimov significance Z =
          sqrt( Σ_i Z_i^2 )
        where for each bin i:
          Z_i = Asimov formula if (s/b)_i >= ratio_threshold,
                Gaussian approximation if (s/b)_i < ratio_threshold.
        """

        # avoid division by zero or tiny b
        safe_b = np.maximum(b, eps)

        # per‑bin signal/background ratio
        ratio = s / safe_b

        # full Asimov significance per bin
        Z_asimov = np.sqrt(
            2.0 * ((s + safe_b) * np.log(1 + ratio) - s)
        )

        # Gaussian-limit approximation for small s/b
        Z_approx = s / np.sqrt(safe_b)

        # pick approximation when ratio is below threshold
        Z_per_bin = np.where(ratio < ratio_threshold, Z_approx, Z_asimov)

        # combine per‑bin Z’s in quadrature
        Z_total = np.sqrt(np.sum(Z_per_bin**2))

        return Z_total
    
    def asymptotic_significance_(self, s: np.ndarray, b: np.ndarray, eps: float = 1e-10, ratio_threshold: float = 0.1) -> float:
        """
        Combined per-bin significance with thresholded approximation.
        """
        safe_b = np.maximum(b, eps)
        ratio = s / safe_b
        Z_asimov = np.sqrt(2.0 * ((s + safe_b) * np.log1p(ratio) - s))
        Z_approx = s / np.sqrt(safe_b)
        Z_per_bin = np.where(ratio < ratio_threshold, Z_approx, Z_asimov)
        return float(np.sqrt(np.sum(Z_per_bin**2)))

    def compute_bin_counts(self, gaussians: List[Dict[str,np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Assign every event to the highest‑PDF Gaussian and
        return counts[label] = array of length n_bins.
        """
        # stack
        scores_list, wts_list, labels = [], [], []
        for label, df in self.df_dict.items():
            scores_list.append(np.vstack(df[self.var_label].to_numpy()))
            wts_list.append(df[self.weight_label].to_numpy())
            labels.extend([label]*len(df))
        scores = np.vstack(scores_list)
        wts    = np.concatenate(wts_list)
        labels = np.array(labels)

        # pdfs
        N, K = scores.shape[0], self.n_bins
        logpdf = np.zeros((N,K))
        for k,g in enumerate(gaussians):
            rv = multivariate_normal(mean=g["mean"], cov=g["cov"], allow_singular=True)
            logpdf[:,k] = rv.logpdf(scores)
        assgn = np.argmax(logpdf, axis=1)

        # counts per label
        counts: Dict[str, np.ndarray] = {}
        for label in self.df_dict:
            arr = np.zeros(K)
            mask = (labels==label)
            for k in range(K):
                arr[k] = wts[(assgn==k)&mask].sum()
            counts[label] = arr
        return counts

    def objective(self, trial: optuna.Trial) -> float:
        D = len(self.df_dict[self.signal_label_lst[0]][self.var_label].iloc[0])
        gaussians: List[Dict[str, np.ndarray]] = []
        for k in range(self.n_bins):
            mu = np.array([trial.suggest_float(f"mu_{k}_{d}", 0.0, 1.0) for d in range(D)])
            L = np.zeros((D, D))
            for i in range(D):
                for j in range(i + 1):
                    name = f"L_{k}_{i}{j}"
                    if i == j:
                        L[i, j] = trial.suggest_float(name, 1e-4, 1.0, log=True)
                    else:
                        L[i, j] = trial.suggest_float(name, -0.5, 0.5)
            gaussians.append({"mean": mu, "cov": L @ L.T})

        counts = self.compute_bin_counts(gaussians)
        # penalty on background bins
        bkg_counts = np.zeros(self.n_bins)
        for bkg in self.bkg_label_lst:
            bkg_counts += counts[bkg]
        short_bins = np.sum(bkg_counts < 10)
        if short_bins > 0:
            return -float(short_bins)

        # compute separate significances
        def compute_Z(sig_label):
            s = counts[sig_label]
            b = np.sum([counts[l] for l in counts if l != sig_label], axis=0)
            return self.asymptotic_significance(s, b)

        Z1 = compute_Z(self.signal_label_lst[0])
        Z2 = compute_Z(self.signal_label_lst[1])
        if self.combination == "quadrature":
            Z = math.hypot(Z1, Z2)
        elif self.combination == "geometric":
            Z = math.sqrt(max(Z1, 0) * max(Z2, 0))
        elif self.combination == "harmonic":
            Z = 2.0 / (1.0 / (Z1 + 1e-12) + 1.0 / (Z2 + 1e-12))
        else:
            raise ValueError(f"Unknown combination: {self.combination}")
        return Z

    def optimize_bins(self
    ) -> Tuple[List[Dict[str,np.ndarray]], Dict[str,np.ndarray], float]:
        os.makedirs(self.output_dir, exist_ok=True)
        sampler = optuna.samplers.TPESampler(gamma=self.gamma_fn())
        self.study = optuna.create_study(direction="maximize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)

        self.best_Z = self.study.best_value
        trial = self.study.best_trial

        # rebuild Gaussians
        D = len(trial.params[f"mu_0_0"].shape) if False else len(
            self.df_dict[self.signal_label_lst[0]][self.var_label].iloc[0]
        )
        print(D)
        self.best_gaussians = []
        for k in range(self.n_bins):
            mu = np.array([trial.params[f"mu_{k}_{d}"] for d in range(D)])
            L  = np.zeros((D,D))
            for i in range(D):
                for j in range(i+1):
                    L[i,j] = trial.params[f"L_{k}_{i}{j}"]
            self.best_gaussians.append({"mean":mu, "cov":L@L.T})

        self.best_hist_dict = self.compute_bin_counts(self.best_gaussians)
        return self.best_gaussians, self.best_hist_dict, self.best_Z

    def visualize_optimization(self):
        if self.study is None:
            raise RuntimeError("No optimization run yet.")
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        ax.get_figure().savefig(Path(self.output_dir)/"parallel_coord.png"); plt.clf()
        ax = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        ax.get_figure().savefig(Path(self.output_dir)/"history.png"); plt.clf()

    def _plot_ellipse(self,mean:np.ndarray,cov:np.ndarray,dims:Tuple[int,int],ax,color:str,label:str=None):
        subcov=cov[np.ix_(dims,dims)]; vals,vecs=np.linalg.eigh(subcov)
        angle=math.degrees(math.atan2(*vecs[:,0][::-1])); width,height=2*np.sqrt(vals)
        ell=Ellipse(mean[list(dims)],width,height,angle=angle,edgecolor=color,facecolor='none',linewidth=2)
        ax.add_patch(ell)
        if label is not None:
            ax.text(mean[dims[0]],mean[dims[1]],label,color=color,ha='center',va='center')

    def assign_bins_to_data(self):
        """
        Add column 'bin_index' to each DataFrame in df_dict, ranking bins by combined Z desc.
        """
        counts=self.best_hist_dict
        # per-bin Z
        def per_bin_Z(sig):
            s=counts[sig]
            b=np.sum([counts[l] for l in counts if l!=sig],axis=0)
            return np.sqrt(2*((s+b)*np.log1p(s/(b+1e-9))-s))
        #def per_bin_Z(sig):
        #    s = counts[sig]
        #    b = np.sum([counts[l] for l in counts if l != sig], axis=0)
        #    return self.asymptotic_significance_(s, b)
        Z1=per_bin_Z(self.signal_label_lst[0])
        Z2=per_bin_Z(self.signal_label_lst[1])

        self.Z1_sum_quad=np.sqrt(np.sum(np.square(Z1)))
        self.Z2_sum_quad=np.sqrt(np.sum(np.square(Z2)))

        if self.combination=='quadrature': Zb=np.hypot(Z1,Z2)
        elif self.combination=='geometric': Zb=np.sqrt(np.clip(Z1,0,None)*np.clip(Z2,0,None))
        else: Zb=2.0/(1.0/(Z1+1e-12)+1.0/(Z2+1e-12))
        order=np.argsort(-Zb)
        rank_map={orig:rank for rank,orig in enumerate(order)}
        self.rank_map = rank_map
        for lbl,df in self.df_dict.items():
            arr=np.vstack(df[self.var_label].to_numpy()); logpdf=np.stack([multivariate_normal(mean=g['mean'],cov=g['cov'],allow_singular=True).logpdf(arr) for g in self.best_gaussians],axis=1)
            orig_bins=np.argmax(logpdf,axis=1)
            df['bin_index']= [rank_map[b] for b in orig_bins]

    def visualize_labelled_ellipses(self):
        if self.rank_map is None:
            self.assign_bins_to_data()

        dims_list = [(0,1), (0,2), (1,2)]
        for dims in dims_list:
            fig, ax = plt.subplots(figsize=(8,6))
            # scatter true labels
            for lbl, df in self.df_dict.items():
                alpha = 0.1 if lbl in self.bkg_label_lst else 0.2
                arr = np.vstack(df[self.var_label].to_numpy())
                ax.scatter(
                    arr[:, dims[0]], arr[:, dims[1]],
                    s=5, alpha=alpha, label=lbl
                )
            # plot ellipses with proxies
            proxies = []
            for k, g in enumerate(self.best_gaussians):
                bin_idx = self.rank_map[k]
                color = self.cmap(bin_idx)
                self._plot_ellipse(g['mean'], g['cov'], dims, ax, color)
                proxies.append(Patch(edgecolor=color, facecolor='none', label=f'Bin {bin_idx}'))
            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            #ax.set_title(f"True labels + Ellipses (dims={dims})")
            handles, labels = ax.get_legend_handles_labels()
            handles += proxies
            labels += [p.get_label() for p in proxies]
            ax.legend(handles, labels, ncol=2, fontsize=12)
            y0, y1 = ax.get_ylim()
            dy = (y1 - y0) * 0.1
            ax.set_ylim(y0-dy, y1+dy)

            plt.tight_layout()
            fig.savefig(Path(self.output_dir)/f'labelled_ellipse_{dims[0]}{dims[1]}.png')
            plt.clf()

    def visualize_bins_2d(self):
        """
        2D scatter of all points colored by their assigned 'bin_index'.
        Uses the 'bin_index' column and fixed colormap.
        """
        dims_list = [(0,1), (0,2), (1,2)]
        for dims in dims_list:
            # gather all scores and bin indices
            all_scores = []
            all_bins = []
            for df in self.df_dict.values():
                arr = np.vstack(df[self.var_label].to_numpy())
                bins = df['bin_index'].to_numpy()
                all_scores.append(arr)
                all_bins.append(bins)
            scores = np.vstack(all_scores)
            bins   = np.concatenate(all_bins)

            fig, ax = plt.subplots(figsize=(8,6))
            sc = ax.scatter(
                scores[:, dims[0]], scores[:, dims[1]],
                c=bins, cmap=self.cmap, vmin=0, vmax=self.n_bins-1,
                s=10, alpha=0.2
            )

            # legend proxies
            proxies = [Patch(color=self.cmap(k), label=f'Bin {k}') for k in range(self.n_bins)]
            ax.legend(handles=proxies, ncol=2)

            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            #ax.set_title(f"Bins by assigned index (dims={dims})")

            plt.tight_layout()
            fig.savefig(Path(self.output_dir)/f'bins_2d_{dims[0]}{dims[1]}.png')
            plt.clf()



    def visualize_bin_boundaries_2d(self, resolution: int = 300):
        """
        Plot decision regions between Gaussian bins in 2D projections.
        Colors each grid‐cell by its assigned bin, then outlines the bin edges.
        """
        dims_list = [(0, 1), (0, 2), (1, 2)]
        # build a discrete colormap for contourf
        cmap_bins = ListedColormap([self.cmap(k) for k in range(self.n_bins)])

        for dims in dims_list:
            # grid over [0,1]×[0,1]
            xs = np.linspace(0, 1, resolution)
            ys = np.linspace(0, 1, resolution)
            X, Y = np.meshgrid(xs, ys)
            grid = np.column_stack([X.ravel(), Y.ravel()])

            # compute assignment
            logpdf = np.zeros((grid.shape[0], self.n_bins))
            for k, g in enumerate(self.best_gaussians):
                subm = g['mean'][list(dims)]
                subc = g['cov'][np.ix_(dims, dims)]
                rv = multivariate_normal(mean=subm, cov=subc, allow_singular=True)
                logpdf[:, k] = rv.logpdf(grid)
            assignment = np.argmax(logpdf, axis=1).reshape(X.shape)

            fig, ax = plt.subplots(figsize=(8, 6))
            # fill regions
            cf = ax.contourf(
                X, Y, assignment,
                levels=np.arange(self.n_bins + 1) - 0.5,
                cmap=cmap_bins,
                alpha=0.4
            )
            # draw crisp boundaries
            ax.contour(
                X, Y, assignment,
                levels=np.arange(self.n_bins + 1) - 0.5,
                colors='k',
                linewidths=0.5
            )

            # include a dashed line going from top left to bottom right
            ax.plot([0, 1], [1, 0], 'k--', lw=0.5)

            # legend
            proxies = [Patch(color=self.cmap(k), label=f'Bin {k}') for k in range(self.n_bins)]
            ax.legend(handles=proxies, ncol=2, fontsize=9, loc='upper right')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            ax.set_title(f"Bin boundaries")

            plt.tight_layout()
            fig.savefig(Path(self.output_dir) / f'bin_boundaries_{dims[0]}{dims[1]}.png')
            plt.clf()
            
    def visualize_bin_boundaries_2d(self, resolution: int = 300):
        """
        Plot decision regions between Gaussian bins in 2D projections.
        Colors each grid‐cell by its assigned bin (with a BoundaryNorm to
        guarantee a one‐to‐one mapping), then outlines the bin edges and
        overplots the real data for context.
        """
        dims_list = [(0, 1), (0, 2), (1, 2)]
        # 1. build a discrete colormap and matching norm
        cmap_bins = ListedColormap([self.cmap(k) for k in range(self.n_bins)])
        bounds = np.arange(self.n_bins + 1) - 0.5   # boundaries at -0.5, 0.5, 1.5, ...
        norm   = BoundaryNorm(bounds, cmap_bins.N)

        for dims in dims_list:
            # 2. grid over [0,1]×[0,1]
            xs = np.linspace(0, 1, resolution)
            ys = np.linspace(0, 1, resolution)
            X, Y    = np.meshgrid(xs, ys)
            grid    = np.column_stack([X.ravel(), Y.ravel()])

            # 3. assign each grid‐point to the max‐pdf bin
            logpdf = np.zeros((grid.shape[0], self.n_bins))
            for k, g in enumerate(self.best_gaussians):
                subm = g['mean'][list(dims)]
                subc = g['cov'][np.ix_(dims, dims)]
                rv   = multivariate_normal(mean=subm, cov=subc, allow_singular=True)
                logpdf[:, k] = rv.logpdf(grid)
            assignment = np.argmax(logpdf, axis=1).reshape(X.shape)

            fig, ax = plt.subplots(figsize=(8, 6))
            # 4. fill with discrete colors
            cf = ax.contourf(
                X, Y, assignment,
                levels=bounds,
                cmap=cmap_bins,
                norm=norm,
                alpha=0.3
            )
            # 5. draw crisp black boundaries
            ax.contour(
                X, Y, assignment,
                levels=bounds,
                colors='k',
                linewidths=0.5
            )

            # 6. optionally overplot the real points (very low alpha)
            for lbl, df in self.df_dict.items():
                arr = np.vstack(df[self.var_label].to_numpy())[:, dims]
                ax.scatter(
                    arr[:, 0], arr[:, 1],
                    s=3, alpha=0.05,
                    color='k' if lbl in self.bkg_label_lst else 'r',
                    label=None
                )

            # 7. legend
            proxies = [Patch(color=self.cmap(k), label=f'Bin {k}') for k in range(self.n_bins)]
            ax.legend(handles=proxies, ncol=2, fontsize=9, loc='upper right')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            ax.set_title(f"Bin regions & boundaries (dims={dims[0]},{dims[1]})")

            plt.tight_layout()
            fig.savefig(Path(self.output_dir) / f'bin_boundaries_{dims[0]}{dims[1]}.png')
            plt.clf()

    def visualize_bin_boundaries_simplex_pairs(self, resolution: int = 300):
        """
        For each pair of dims (i,j), consider the plane x_k = 1 - x_i - x_j
        (so x_i+x_j<=1, x_k>=0).  At each grid‐point (x_i,x_j), evaluate the
        full 3D PDF p_k(x_i,x_j,1−x_i−x_j) for each bin and color by the argmax.
        """
        dims_list = [(0,1), (0,2), (1,2)]
        # discrete colormap + norm
        colors = [self.cmap(k) for k in range(self.n_bins)]
        cmap_bins = ListedColormap(colors)
        bounds = np.arange(self.n_bins + 1) - 0.5
        norm   = BoundaryNorm(bounds, cmap_bins.N)

        all_dims = {0,1,2}
        for dims in dims_list:
            rem = (all_dims - set(dims)).pop()

            # grid on dims 0..1
            xs = np.linspace(0, 1, resolution)
            ys = np.linspace(0, 1, resolution)
            X, Y = np.meshgrid(xs, ys)
            mask = (X + Y <= 1.0)

            # prepare array of assignments, fill outside simplex with NaN
            assignment = np.full(X.shape, np.nan, dtype=float)

            # build full‐PDF grid‐points
            pts = np.zeros((mask.sum(), 3))
            pts[:, dims[0]] = X[mask]
            pts[:, dims[1]] = Y[mask]
            pts[:, rem]     = 1.0 - X[mask] - Y[mask]

            # evaluate logpdfs
            logps = np.zeros((pts.shape[0], self.n_bins))
            for k, g in enumerate(self.best_gaussians):
                rv = multivariate_normal(mean=g['mean'], cov=g['cov'], allow_singular=True)
                logps[:, k] = rv.logpdf(pts)
            assignment[mask] = np.argmax(logps, axis=1)

            # plot
            fig, ax = plt.subplots(figsize=(8,6))
            cf = ax.contourf(
                X, Y, assignment,
                levels=bounds,
                cmap=cmap_bins,
                norm=norm,
                alpha=0.6
            )
            ax.contour(
                X, Y, assignment,
                levels=bounds,
                colors='k',
                linewidths=0.8
            )

            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            ax.set_title(f"Bin regions (simplex slice dims={dims[0]},{dims[1]})")

            plt.tight_layout()
            fig.savefig(Path(self.output_dir)/f'bin_boundaries_{dims[0]}{dims[1]}.png')
            plt.clf()

        
    def visualize_bin_boundaries_simplex_pairs(self, resolution: int = 300):
        """
        For each pair of dims (i,j), slice the simplex x_k = 1-x_i-x_j,
        compute the full 3D Gaussian argmax at each grid pt, and plot
        filled regions + boundaries, with bin‐color legend and numeric labels.
        """
        dims_list = [(0,1), (0,2), (1,2)]
        # prepare discrete colormap & norm
        colors = [self.cmap(k) for k in range(self.n_bins)]
        cmap_bins = ListedColormap(colors)
        bounds = np.arange(self.n_bins+1) - 0.5
        norm   = BoundaryNorm(bounds, cmap_bins.N)

        all_dims = {0,1,2}
        for dims in dims_list:
            rem = (all_dims - set(dims)).pop()

            # 1) build grid on [0,1]^2 & mask simplex x_i+x_j <=1
            xs = np.linspace(0,1,resolution)
            ys = np.linspace(0,1,resolution)
            X, Y = np.meshgrid(xs, ys)
            mask = (X + Y <= 1.0)

            # 2) assemble full (x,y,z) points
            pts = np.zeros((mask.sum(), 3))
            pts[:, dims[0]] = X[mask]
            pts[:, dims[1]] = Y[mask]
            pts[:, rem]     = 1.0 - X[mask] - Y[mask]

            # 3) evaluate full-PDF logps and assign bins
            logps = np.zeros((pts.shape[0], self.n_bins))
            for k, g in enumerate(self.best_gaussians):
                rv = multivariate_normal(mean=g['mean'], cov=g['cov'], allow_singular=True)
                logps[:, k] = rv.logpdf(pts)
            assign = np.full(X.shape, np.nan)
            assign[mask] = np.argmax(logps, axis=1)

            # 4) plot regions & boundaries
            fig, ax = plt.subplots(figsize=(8,6))
            cf = ax.contourf(
                X, Y, assign,
                levels=bounds,
                cmap=cmap_bins,
                norm=norm,
                alpha=0.6
            )
            ax.contour(
                X, Y, assign,
                levels=bounds,
                colors='k',
                linewidths=0.8
            )

            # 5) numeric labels at region centroids
            for k in range(self.n_bins):
                xi = X[assign==k]
                yi = Y[assign==k]
                if xi.size:
                    xc, yc = xi.mean(), yi.mean()
                    ax.text(
                        xc, yc, str(k),
                        color=self.cmap(k),
                        fontsize=10, fontweight='bold',
                        ha='center', va='center'
                    )

            # 6) legend proxies
            proxies = [Patch(color=colors[k], label=f'Bin {k}') for k in range(self.n_bins)]
            ax.legend(handles=proxies, ncol=2, fontsize=9, loc='upper right')

            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_xlabel(f"score_{dims[0]}")
            ax.set_ylabel(f"score_{dims[1]}")
            ax.set_title(f"Bin regions (dims={dims[0]},{dims[1]})")

            plt.tight_layout()
            fig.savefig(Path(self.output_dir)/f'bin_boundaries_{dims[0]}{dims[1]}.png')
            plt.clf()
"""    def visualize_labelled_ellipses(self):
        dims_list=[(0,1),(0,2),(1,2)]; handles_all=[]
        for dims in dims_list:
            fig,ax=plt.subplots(figsize=(8,6))
            for lbl,df in self.df_dict.items():
                arr=np.vstack(df[self.var_label].to_numpy())
                ax.scatter(arr[:,dims[0]],arr[:,dims[1]],s=10,alpha=0.5,label=lbl)
            # ellipses
            for k,g in enumerate(self.best_gaussians):
                color=self.cmap(k)
                self._plot_ellipse(g['mean'],g['cov'],dims,ax,color,None)
                handles_all.append(Patch(edgecolor=color,facecolor='none',label=f'Bin {k}'))
            ax.set_xlabel(f"score_{dims[0]}"); ax.set_ylabel(f"score_{dims[1]}")
            ax.set_title(f"Labels + Ellipses (dims={dims})")
            # legend: 2 cols, combined handles
            handles, labels = ax.get_legend_handles_labels()
            handles+=handles_all; labels+=[h.get_label() for h in handles_all]
            ax.legend(handles,labels,ncol=2)
            y0,y1=ax.get_ylim()
            dy=(y1-y0)*0.1; ax.set_ylim(y0-dy,y1+dy)
            plt.tight_layout()
            fig.savefig(Path(self.output_dir)/f'labelled_ellipse_{dims[0]}{dims[1]}.png')
            plt.clf()

    def visualize_bins_2d(self):
        dims_list=[(0,1),(0,2),(1,2)]
        for dims in dims_list:
            # gather points
            scores_list=[]
            for df in self.df_dict.values(): scores_list.append(np.vstack(df[self.var_label].to_numpy()))
            scores=np.vstack(scores_list)
            # assign
            logpdf=np.stack([multivariate_normal(mean=g['mean'],cov=g['cov'],allow_singular=True).logpdf(scores) for g in self.best_gaussians],axis=1)
            assgn=np.argmax(logpdf,axis=1)
            fig,ax=plt.subplots(figsize=(8,6))
            sc=ax.scatter(scores[:,dims[0]],scores[:,dims[1]],c=assgn,cmap='tab20',s=10,alpha=0.6)
            # proxies
            proxies=[Patch(color=self.cmap(k),label=f'Bin {k}') for k in range(self.n_bins)]
            ax.legend(handles=proxies,ncol=2)
            ax.set_xlabel(f"score_{dims[0]}"); ax.set_ylabel(f"score_{dims[1]}")
            #ax.set_title(f"Bins by index (dims={dims})")
            plt.tight_layout(); fig.savefig(Path(self.output_dir)/f'bins_2d_{dims[0]}{dims[1]}.png')
            plt.clf()"""


if __name__ == "__main__":
    data = generate_toy_data_3D(seed=42)
    optimizer = GMMBinOptimizer(
        df_dict=data,
        bkg_label_lst=["bkg1","bkg2","bkg3","bkg4","bkg5"],
        signal_label_lst=["signal1","signal2"],
        var_label="NN_output",
        weight_label="weight",
        n_bins=5,
        n_trials=50,
        output_dir="./gmm_results",
        combination="quadrature"
    )
    best_gauss, best_hist, best_Z = optimizer.optimize_bins()
    print("Best combined Z =", best_Z)
    optimizer.visualize_labelled_ellipses(sample_frac=0.2)
    optimizer.visualize_bins_3d(sample_frac=0.2)



