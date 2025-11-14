import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep  # assuming you use mplhep for histplot
plt.style.use(hep.style.ROOT)

########################################################################################
# functions copied from: https://github.com/FloMau/gato/tree/master/
########################################################################################
    
def plot_stacked_histograms(
    stacked_hists,           # list of hist.hist objects for backgrounds
    process_labels,          # list of labels for backgrounds
    output_filename="./plot.pdf",
    axis_labels=("x-axis", "Events"),
    signal_hists=None,       # optional list of hist.hist objects for signals
    signal_labels=None,      # optional labels for signal histograms
    signal_scale=1.0,        # optional scaling factor for signal histograms
    normalize=False,
    log=False,
    log_min=None,
    include_flow=False,
    colors=None,
    return_figure=False,
    ax=None,
    equidistant_bins=True,  # <<< NEW: plot all bins with equal graphical width
):
    """
    Plots stacked histograms for backgrounds and overlays signal histograms.
    """

    # Normalization if requested.
    if normalize:
        stack_integral = sum([_hist.sum().value for _hist in stacked_hists])
        stacked_hists = [_hist / stack_integral for _hist in stacked_hists]
        if signal_hists:
            for i, sig in enumerate(signal_hists):
                integral_ = sig.sum().value
                if integral_ > 0:
                    signal_hists[i] = sig / integral_

    # Original binning from the first histogram (physics binning)
    orig_bin_edges = stacked_hists[0].to_numpy()[1]

    # Binning used *for plotting*
    if equidistant_bins:
        # N bins -> edges 0..N
        n_bins = len(orig_bin_edges) - 1
        plot_bin_edges = np.arange(n_bins + 1, dtype=float)
    else:
        plot_bin_edges = orig_bin_edges

    # Gather values and uncertainties for each background histogram.
    mc_values_list = [_hist.values() for _hist in stacked_hists]
    mc_errors_list = [np.sqrt(_hist.variances()) for _hist in stacked_hists]

    n_bkg = len(mc_values_list)

    if colors is None:
        colors = [
            "#3f90da",
            "#ffa90e",
            "#bd1f01",
            "#94a4a2",
            "#832db6",
            "#a96b59",
            "#e76300",
            "#b9ac70",
            "#717581",
            "#92dadd",
        ]

    if len(colors) < n_bkg:
        repeats = n_bkg // len(colors) + 1
        colors = (colors * repeats)[:n_bkg]
    else:
        colors = colors[:n_bkg]

    # Setup figure and axis.
    if ax is None:
        fig, ax_main = plt.subplots(figsize=(10, 9))
    else:
        fig = None
        ax_main = ax

    # Plot stacked backgrounds.
    hep.histplot(
        mc_values_list,
        label=process_labels,
        bins=plot_bin_edges,
        stack=True,
        histtype="fill",
        edgecolor="black",
        linewidth=1,
        yerr=mc_errors_list,
        ax=ax_main,
        color=colors,
        alpha=0.8,
    )

    # Total MC band
    mc_total = np.sum(mc_values_list, axis=0)
    mc_total_var = np.sum([err**2 for err in mc_errors_list], axis=0)
    mc_total_err = np.sqrt(mc_total_var)
    hep.histplot(
        mc_total,
        bins=plot_bin_edges,
        histtype="band",
        yerr=mc_total_err,
        ax=ax_main,
        alpha=0.5,
        label=None,
    )

    # Overlay signals
    if signal_hists:
        for sig_hist, label in zip(signal_hists, signal_labels):
            if signal_scale != 1.0:
                sig_hist_ = sig_hist * signal_scale
                label += f" x {signal_scale}"
                sig_values = sig_hist_.values()
                sig_errors = np.sqrt(sig_hist_.variances())
            else:
                sig_values = sig_hist.values()
                sig_errors = np.sqrt(sig_hist.variances())

            hep.histplot(
                [sig_values],
                label=[label],
                bins=plot_bin_edges,
                linewidth=3,
                linestyle="--",
                yerr=sig_errors,
                ax=ax_main,
            )

    # Axis labels & scaling
    ax_main.set_xlabel(axis_labels[0], fontsize=36)
    ax_main.set_ylabel(axis_labels[1], fontsize=36)
    ax_main.margins(y=0.15)

    if log:
        ax_main.set_yscale("log")
        ax_main.set_ylim(ax_main.get_ylim()[0], 30 * ax_main.get_ylim()[1])
        if log_min is not None:
            ax_main.set_ylim(log_min, ax_main.get_ylim()[1])
    else:
        ax_main.set_ylim(0, 1.25 * ax_main.get_ylim()[1])

    ax_main.tick_params(labelsize=24)

    # --- HERE: make the x-axis look like your second plot ---
    if equidistant_bins:
        # place ticks at the (equal) bin edges, label with original edges
        ax_main.set_xticks(np.arange(len(orig_bin_edges)))
        ax_main.set_xticklabels(
            [f"{edge:.3f}" for edge in orig_bin_edges],
            rotation=45,
            ha="right",
        )
        ax_main.set_xlim(0, len(orig_bin_edges) - 1)

    # Legend
    handles, labels = ax_main.get_legend_handles_labels()
    ncols = 2 if len(labels) < 6 else 3
    ax_main.legend(
        loc="upper right",
        fontsize=23,
        ncols=ncols,
        labelspacing=0.4,
        columnspacing=1.5,
    )

    # Save or return
    if not return_figure:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.tight_layout()
        fig.savefig(output_filename)
        plt.close(fig)
    else:
        return fig, ax_main

def create_hist(df, bin_edges=None):

    # create list of hist and fill them
    if bin_edges is None:
        h = hist.Hist(hist.axis.Regular(50, 0, 1, name="NN_output"), storage=hist.storage.Weight())
    else:
        h = hist.Hist(hist.axis.Variable(bin_edges, name="NN_output"), storage=hist.storage.Weight())
    h.fill(df["NN_output"], weight=df["weight"])

    return h
    
if __name__ == "__main__":
    import pandas as pd
    import hist
    # load the background and signal dataframe
    bkg_list = ["bkg1", "bkg2", "bkg3"]
    signal_list = ["signal"]
    bkg_dict = {}
    signal_dict = {}
    for bkg in bkg_list:
        bkg_dict[bkg] = pd.read_parquet(f"toy_data/{bkg}.parquet")
    for signal in signal_list:
        signal_dict[signal] = pd.read_parquet(f"toy_data/{signal}.parquet")

    # create list of hist
    bkg_hists = [hist.Hist(hist.axis.Regular(50, 0, 1, name="NN_output"), storage=hist.storage.Weight()) for _ in bkg_list]
    signal_hists = [hist.Hist(hist.axis.Regular(50, 0, 1, name="NN_output"), storage=hist.storage.Weight()) for _ in signal_list]

    for i, bkg in enumerate(bkg_list):
        bkg_hists[i].fill(bkg_dict[bkg]["NN_output"], weight=bkg_dict[bkg]["weight"])

    for i, signal in enumerate(signal_list):
        signal_hists[i].fill(signal_dict[signal]["NN_output"], weight=signal_dict[signal]["weight"])

    plot_stacked_histograms(
        bkg_hists,
        bkg_list,
        output_filename="toy_data/plot.pdf",
        axis_labels=("NN output", "Events"),
        signal_hists=signal_hists,
        signal_labels=signal_list,
        normalize=False,
        log=False,
        log_min=10,
    )