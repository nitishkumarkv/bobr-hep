from examples.toy_example.generate_toy_data import generate_toy_data, generate_toy_data_gauss, generate_toy_data_3D, generate_toy_data_multiclass
from utils.plotting import plot_stacked_histograms
from bobr.cat_optimizer import BOBRBinOptimizer
from bobr.cat_optimizer_multiclass import GMMBinOptimizer
import os
import argparse
import pandas as pd
import hist
import matplotlib.pyplot as plt
import json
import numpy as np

def create_hist(df, var=None, class_num=None, bin_edges=None):

    if var is None:
        var = "NN_output"

    # create list of hist and fill them
    if bin_edges is None:
        h = hist.Hist(hist.axis.Regular(50, 0, 1, name=var), storage=hist.storage.Weight())
    else:
        h = hist.Hist(hist.axis.Variable(bin_edges, name=var), storage=hist.storage.Weight())
    if class_num is None:
        h.fill(df[var], weight=df["weight"])
    else:
        h.fill(np.stack(df[var])[:, class_num], weight=df["weight"])

    return h

def main():

    # initialize the parser
    parser = argparse.ArgumentParser(description="Run the toy example")
    parser.add_argument("--output-dir", type=str, default="toy_results", help="Output directory")
    #parser.add_argument("--generate-toy-data", action="store_true", help="Generate toy data")
    parser.add_argument("--plot-toy-data", action="store_true", help="Plot toy data")
    parser.add_argument("--run-BOBR", action="store_true", help="Run BOBR optimizer")
    parser.add_argument("--run-equidistance", action="store_true", help="Run equidistance optimizer")
    args = parser.parse_args()

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, "toy_data")

    bkg_list = ["bkg1", "bkg2", "bkg3", "bkg4", "bkg5"]
    signal_list = ["signal1", "signal2"]

    toy_data = generate_toy_data_3D(seed=42)

    n_classes = 3

    for i in range(n_classes):
        bkg_hists = [create_hist(toy_data[bkg], class_num=i) for bkg in bkg_list]
        signal_hists = [create_hist(toy_data[signal], class_num=i) for signal in signal_list]
        plot_stacked_histograms(
            bkg_hists,
            bkg_list,
            output_filename=f"{data_path}/plot_log_score_{i}.pdf",
            axis_labels=(f"NN output score {i}", "Events"),
            signal_hists=signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=True,
            log_min=1e-5,
        )
        plot_stacked_histograms(
            bkg_hists,
            bkg_list,
            output_filename=f"{data_path}/plot_score_{i}.pdf",
            axis_labels=(f"NN output score {i}", "Events"),
            signal_hists=signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )

    num_bins_lst = [3, 5, 10, 15, 20]
    #num_bins_lst = [3, 5]
    z1_sum_quad_list = []
    z2_sum_quad_list = []

    for num_bins in num_bins_lst:

        bin_directory = args.output_dir + f"/n_{num_bins}"

        optimizer = GMMBinOptimizer(
            df_dict          = toy_data,
            bkg_label_lst    = ["bkg1","bkg2","bkg3","bkg4","bkg5"],
            signal_label_lst = ["signal1","signal2"],
            var_label        = "NN_output",
            weight_label     = "weight",
            n_bins           = num_bins,
            n_trials         = 100,
            output_dir       = bin_directory,
            combination      = "geometric",
        )

        # 3) optimize & visualize
        best_gaussians, best_hist, best_Z = optimizer.optimize_bins()

        print("Best combined Z =", best_Z)

        #optimizer.visualize_optimization()
        #optimizer.assign_bins_to_data()
        optimizer.visualize_labelled_ellipses()
        optimizer.visualize_bins_2d()
        #optimizer.visualize_bin_boundaries_2d()
        optimizer.visualize_bin_boundaries_simplex_pairs()

        bin_edges = [i for i in range(num_bins+1)]

        best_signal_hists = [create_hist(toy_data[signal], bin_edges=bin_edges, var="bin_index") for signal in signal_list]
        best_bkg_hists = [create_hist(toy_data[bkg], bin_edges=bin_edges, var="bin_index") for bkg in bkg_list]

        plot_stacked_histograms(
            best_bkg_hists,
            bkg_list,
            output_filename=f"{bin_directory}/plot_log.pdf",
            axis_labels=("Bin Index", "Events"),
            signal_hists=best_signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=True,
            log_min=1e-5,
        )
        plot_stacked_histograms(
            best_bkg_hists,
            bkg_list,
            output_filename=f"{bin_directory}/plot.pdf",
            axis_labels=("Bin Index", "Events"),
            signal_hists=best_signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )

        # save optimized gaussian parameters
        print("INFO: Saving the optimized gaussian parameters")
        print("Best gaussians:", best_gaussians)
        print("Z1 sum quad:", optimizer.Z1_sum_quad)
        print("Z2 sum quad:", optimizer.Z2_sum_quad)
        #with open(f"{bin_directory}/optimized_bins.json", "w") as f:
        #    to_save = {
        #        "best_gaussians": best_gaussians,
        #        "best_Z": best_Z
        #    }
        #    json.dump(to_save, f, indent=4)

        z1_sum_quad_list.append(optimizer.Z1_sum_quad)
        z2_sum_quad_list.append(optimizer.Z2_sum_quad)

    
    # max score based binning for comparison
        
    
    if args.run_equidistance:
        for proc, df in toy_data.items():
            # add max score to the dataframe
            df["max_score"] = df["NN_output"].apply(lambda x: np.max(x))
            # add argmax score to the dataframe
            df["argmax_score"] = df["NN_output"].apply(lambda x: np.argmax(x))

        n_bins_lst_equi = [2, 3, 4, 5, 8, 11]
        s1_bkg_list = ["bkg1", "bkg2", "bkg3", "bkg4", "bkg5", "signal2"]
        s1_signal_list = ["signal1"]

        toy_data_s1 = toy_data.copy()
        for proc, df in toy_data_s1.items():
            df = df[df["argmax_score"] == 0]
            toy_data_s1[proc] = df

        s2_bkg_list = ["bkg1", "bkg2", "bkg3", "bkg4", "bkg5", "signal1"]
        s2_signal_list = ["signal2"]

        toy_data_s2 = toy_data.copy()
        for proc, df in toy_data_s2.items():
            df = df[df["argmax_score"] == 1]
            toy_data_s2[proc] = df

        z1_equ_lst = []
        z2_equ_lst = []

        for n_bins in n_bins_lst_equi:
            optimize_equidistance_s1 = BOBRBinOptimizer(toy_data, bkg_label_lst=s1_bkg_list, signal_label_lst=s1_signal_list,
                                        var_label='max_score', weight_label='weight', n_bins=n_bins, output_dir=args.output_dir+f"/equidistant_s1_{n_bins}")
        
            optimize_equidistance_s2 = BOBRBinOptimizer(toy_data, bkg_label_lst=s2_bkg_list, signal_label_lst=s2_signal_list,
                                        var_label='max_score', weight_label='weight', n_bins=n_bins, output_dir=args.output_dir+f"/equidistant_s2_{n_bins}")
            
            best_bins_eqi_s1, best_hist_dict_eqi_s1, best_Z_eqi_s1 = optimize_equidistance_s1.equidistant_bins()
            best_bins_eqi_s2, best_hist_dict_eqi_s2, best_Z_eqi_s2 = optimize_equidistance_s2.equidistant_bins()
            z1_equ_lst.append(best_Z_eqi_s1)
            z2_equ_lst.append(best_Z_eqi_s2)

            # for s1
            best_signal_hists = [create_hist(toy_data_s1[signal], bin_edges=best_bins_eqi_s1, var="max_score") for signal in signal_list]
            best_bkg_hists = [create_hist(toy_data_s1[bkg], bin_edges=best_bins_eqi_s1, var="max_score") for bkg in bkg_list]
            plot_stacked_histograms(
                best_bkg_hists,
                bkg_list,
                output_filename=f"{args.output_dir}//equidistant_s1_{n_bins}/plot_log.pdf",
                axis_labels=("Max-score", "Events"),
                signal_hists=best_signal_hists,
                signal_labels=signal_list,
                signal_scale=100,
                normalize=False,
                log=True,
                log_min=1e-5,
            )
            plot_stacked_histograms(
                best_bkg_hists,
                bkg_list,
                output_filename=f"{args.output_dir}//equidistant_s1_{n_bins}/plot.pdf",
                axis_labels=("Max-score", "Events"),
                signal_hists=best_signal_hists,
                signal_labels=signal_list,
                signal_scale=100,
                normalize=False,
                log=False,
                log_min=1e-5,
            )

            # for s2
            best_signal_hists = [create_hist(toy_data_s2[signal], bin_edges=best_bins_eqi_s2, var="max_score") for signal in signal_list]
            best_bkg_hists = [create_hist(toy_data_s2[bkg], bin_edges=best_bins_eqi_s2, var="max_score") for bkg in bkg_list]
            plot_stacked_histograms(
                best_bkg_hists,
                bkg_list,
                output_filename=f"{args.output_dir}//equidistant_s2_{n_bins}/plot_log.pdf",
                axis_labels=("Max-score", "Events"),
                signal_hists=best_signal_hists,
                signal_labels=signal_list,
                signal_scale=100,
                normalize=False,
                log=True,
                log_min=1e-5,
            )
            plot_stacked_histograms(
                best_bkg_hists,
                bkg_list,
                output_filename=f"{args.output_dir}//equidistant_s2_{n_bins}/plot.pdf",
                axis_labels=("Max-score", "Events"),
                signal_hists=best_signal_hists,
                signal_labels=signal_list,
                signal_scale=100,
                normalize=False,
                log=False,
                log_min=1e-5,
            )

        
        true_n_bins_lst_equi = [i*2 for i in n_bins_lst_equi]


        plt.figure()
        plt.plot(num_bins_lst, z1_sum_quad_list, marker="o", label="BOBR-Signal1", color="red")
        plt.plot(num_bins_lst, z2_sum_quad_list, marker="o", label="BOBR-Signal2", color="blue")
        plt.plot(true_n_bins_lst_equi, z1_equ_lst, marker="o", label="Equidistant-Signal1", color="red", linestyle="--")
        plt.plot(true_n_bins_lst_equi, z2_equ_lst, marker="o", label="Equidistant-Signal2", color="blue", linestyle="--")
        plt.xlabel("n_bins")
        plt.ylabel("Significance")
        plt.legend()
        plt.savefig(f"{args.output_dir}/significance_vs_nbins.pdf")


if __name__ == "__main__":
    print("INFO: Running the toy example")
    main()

    


