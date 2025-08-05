import numpy as np
import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
from hist import Hist
import os


class BOBRBinOptimizer:
    """
    Bayesian Optimization of Bin boundaRies (BOBR)
    Optimizes bin boundaries to maximize signal significance.
    """
    def __init__(self, df_dict, bkg_label_lst, signal_label_lst, var_label, weight_label, n_bins=10, n_trials=10,
                 output_dir="./optimizer_results", gamma_strategy="sqrt", beta=0.25):
        """
        Initialize the optimizer.

        :param df_dict: Dictionary containing signal and background dataframes.
        :param bkg_label_lst: List of background labels.
        :param signal_label_lst: List of signal labels.
        :param var_label: variable used for binning.
        :param weight_label: Column name for event weights.
        :param n_bins: Number of bins (n). The optimizer finds (n-1) bin boundaries.
        :param n_trials: Number of optimization trials.
        :param gamma_strategy: Strategy for computing gamma ('linear' or 'sqrt').
        :param beta: Coefficient for gamma function.
        """
        self.df_dict = df_dict
        self.bkg_label_lst = bkg_label_lst
        self.signal_label_lst = signal_label_lst
        self.var_label = var_label
        self.weight_label = weight_label
        self.n_bins = n_bins
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.gamma_strategy = gamma_strategy
        self.beta = beta
        self.study = None
        self.best_bins = None
        self.best_hist_dict = None
        self.best_Z = None

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

    def asymptotic_significance_(self, s, b):
        
        """Compute the asymptotic significance"""

        Z = np.sqrt(2 * ((s + b) * np.log(1 + (s / (b + 1e-10))) - s))

        # compute sum of Z in qudrature
        Z_sum_quad = np.sqrt(np.sum(Z**2))
        return Z_sum_quad
    
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

    def compute_bin_counts(self, bin_edges):
        """Assign classifier scores to bins and compute signal/background counts."""
        hist_signal = Hist.new.Variable(bin_edges).Weight()
        hist_background = Hist.new.Variable(bin_edges).Weight()

        for sig_key in self.signal_label_lst:
            hist_signal.fill(self.df_dict[sig_key][self.var_label].values, weight=self.df_dict[sig_key][self.weight_label].values)
        for bkg_key in self.bkg_label_lst:
            hist_background.fill(self.df_dict[bkg_key][self.var_label].values, weight=self.df_dict[bkg_key][self.weight_label].values)

        signal_counts = np.array(hist_signal.values())
        background_counts = np.array(hist_background.values())
        
        return signal_counts, background_counts
    
#    def optimize_bins(self):
#        """Optimize bin boundaries using Bayesian Optimization."""
#        min_edge, max_edge = 0, 1
#
#        def objective(trial):
#            bin_edges = [min_edge]
#            for i in range(self.n_bins - 1):
#                bin_edges.append(trial.suggest_float(f'bin_{i}', bin_edges[-1], max_edge))
#            bin_edges.append(max_edge)
#
#            if not all(bin_edges[i] < bin_edges[i + 1] for i in range(len(bin_edges) - 1)):
#                raise optuna.TrialPruned()
#
#            signal_hist, background_hist = self.compute_bin_counts(bin_edges)
#            penalty = np.sum(background_hist < 10) * -1
#            if penalty < 0:
#                return penalty
#            else:
#                return self.asymptotic_significance(signal_hist, background_hist)
#
#        #def objective_with_penalty(trial, penalty_factor=10):
#        #    bin_edges = [min_edge]
#        #    penalty = 0  # Initialize penalty score
##
#        #    for i in range(self.n_bins - 1):
#        #        suggested_bin = trial.suggest_float(f'bin_{i}', min_edge, max_edge)
##
#        #        # Check if the suggested bin is out of order
#        #        if suggested_bin <= bin_edges[-1]:
#        #            penalty -= penalty_factor * abs(bin_edges[-1] - suggested_bin)  # Penalize inversely to distance
##
#        #        bin_edges.append(suggested_bin)
##
#        #        bin_edges.append(max_edge)
##
#        #    # Compute histogram counts
#        #    signal_hist, background_hist = self.compute_bin_counts(bin_edges)
##
#        #    # Apply soft penalty for bins with low background counts
#        #    low_background_penalty = np.sum(background_hist < 10) * -1  
##
#        #    # Final objective value: Asymptotic significance + penalties
#        #    return self.asymptotic_significance(signal_hist, background_hist) + penalty + low_background_penalty
#
#        
#        self.study = optuna.create_study(direction='maximize')
#        self.study.optimize(objective, n_trials=self.n_trials)
#        self.best_bins = [min_edge] + [self.study.best_trial.params[f'bin_{i}'] for i in range(self.n_bins - 1)] + [max_edge]
#        best_signal_counts, best_background_counts = self.compute_bin_counts(self.best_bins)
#        self.best_hist_dict = {'signal': best_signal_counts, 'background': best_background_counts}
#        self.best_Z = self.asymptotic_significance(best_signal_counts, best_background_counts)
#
#        return self.best_bins, self.best_hist_dict, self.best_Z
    
    def optimize_bins(self):
        """Optimize bin boundaries using Bayesian Optimization."""
        min_edge, max_edge = 0, 1

        def objective(trial):
            bin_edges = [min_edge]
            for i in range(self.n_bins - 1):
                bin_edges.append(trial.suggest_float(f'bin_{i}', bin_edges[-1], max_edge))
            bin_edges.append(max_edge)

            if not all(bin_edges[i] < bin_edges[i + 1] for i in range(len(bin_edges) - 1)):
                raise optuna.TrialPruned()

            signal_hist, background_hist = self.compute_bin_counts(bin_edges)
            penalty = np.sum(background_hist < 10) * -1
            if penalty < 0:
                return penalty
            else:
                return self.asymptotic_significance(signal_hist, background_hist)

        sampler = optuna.samplers.TPESampler(gamma=self.gamma_fn())
        self.study = optuna.create_study(direction='maximize', sampler=sampler)
        self.study.optimize(objective, n_trials=self.n_trials)

        self.best_bins = [min_edge] + [self.study.best_trial.params[f'bin_{i}'] for i in range(self.n_bins - 1)] + [max_edge]
        best_signal_counts, best_background_counts = self.compute_bin_counts(self.best_bins)
        self.best_hist_dict = {'signal': best_signal_counts, 'background': best_background_counts}
        self.best_Z = self.asymptotic_significance(best_signal_counts, best_background_counts)

        return self.best_bins, self.best_hist_dict, self.best_Z
    
    def equidistant_bins(self):
        """Generate equidistant bin edges."""
        min_edge, max_edge = 0, 1
        bin_edges = np.linspace(min_edge, max_edge, self.n_bins + 1)
        self.best_bins = list(bin_edges)
        # compute significance
        best_signal_counts, best_background_counts = self.compute_bin_counts(self.best_bins)
        self.best_hist_dict = {'signal': best_signal_counts, 'background': best_background_counts}
        self.best_Z = self.asymptotic_significance(best_signal_counts, best_background_counts)
        

        return self.best_bins, self.best_hist_dict, self.best_Z

    def visualize_optimization(self):
        """Generate visualization plots for the optimization process."""
        if self.study is None:
            print("No study found. Run optimize_bins() first.")
            return
        
        # make output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parallel Coordinate Plot
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        fig = ax.get_figure()  # Retrieve the parent Figure.
        fig.suptitle(f"Parallel Coordinates ", fontsize=14)
        fig.savefig(os.path.join(self.output_dir, f"parallel_coordinate_plot.png"))
        plt.clf()

        # Optimization History Plot
        ax = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        fig = ax.get_figure()  # Retrieve the parent Figure.
        fig.suptitle(f"Optimization History", fontsize=14)
        fig.savefig(os.path.join(self.output_dir, f"optimization_history_plot.png"))
        plt.clf()
        
        # Progression of bin boundaries
        trials = [trial for trial in self.study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        trial_numbers = [trial.number for trial in trials]
        bin_evolution = {f'bin_{i}': [trial.params[f'bin_{i}'] for trial in trials] for i in range(self.n_bins - 1)}
        
        plt.figure(figsize=(10, 6))
        for i in range(self.n_bins - 1):
            plt.plot(trial_numbers, bin_evolution[f'bin_{i}'], label=f'Bin boundary {i+1}')
        plt.xlabel("Trial Number")
        plt.ylabel("Bin Boundaries")
        plt.title(F"Evolution of bin boundaries (n_bins={self.n_bins})")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"bin_evolution.png"))
        plt.clf()

if __name__ == "__main__":
    np.random.seed(42)
    df_signal = {'score': np.random.rand(1000), 'weight': np.ones(1000)}
    df_bkg1 = {'score': np.random.rand(800), 'weight': np.ones(800)}
    df_bkg2 = {'score': np.random.rand(600), 'weight': np.ones(600)}
    df_bkg3 = {'score': np.random.rand(400), 'weight': np.ones(400)}
    
    data = {"signal": df_signal, "bkg1": df_bkg1, "bkg2": df_bkg2, "bkg3": df_bkg3}
    
    optimizer = BOBRBinOptimizer(data, bkg_label_lst=["bkg1", "bkg2", "bkg3"], signal_label_lst=["signal"],
                                 var_label='score', weight_label='weight', n_bins=10, n_trials=100,
                                 gamma_strategy="sqrt", beta=0.25)
    best_bins = optimizer.optimize_bins()
    print("Optimized bin edges:", best_bins)
    optimizer.visualize_optimization()
