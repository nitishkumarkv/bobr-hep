from examples.toy_example.generate_toy_data import generate_toy_data, generate_toy_data_gauss
from utils.plotting import plot_stacked_histograms
from bobr.cat_optimizer import BOBRBinOptimizer
import os
import argparse
import pandas as pd
import hist
import matplotlib.pyplot as plt
import json

def create_hist(df, bin_edges=None):

    # create list of hist and fill them
    if bin_edges is None:
        h = hist.Hist(hist.axis.Regular(50, 0, 1, name="NN_output"), storage=hist.storage.Weight())
    else:
        h = hist.Hist(hist.axis.Variable(bin_edges, name="NN_output"), storage=hist.storage.Weight())
    h.fill(df["NN_output"], weight=df["weight"])

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

    bkg_list = ["bkg1", "bkg2", "bkg3"]
    signal_list = ["signal"]
    
    # Generate toy data
    #if args.generate_toy_data:
    #    print("INFO: Generating toy data")
    #    os.makedirs(data_path, exist_ok=True)
#
    #    #toy_data = generate_toy_data(
    #    #    n_signal=100000,
    #    #    n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
    #    #    lam_signal=6, lam_bkg1=8, lam_bkg2=7, lam_bkg3=3,
    #    #    xs_signal=0.5,  # 500 fb = 0.5 pb
    #    #    xs_bkg1=50, xs_bkg2=15, xs_bkg3=1,
    #    #    lumi=100,  # in /fb
    #    #    seed=42,
    #    #    output_dir=data_path
    #    #)
    #    toy_data = generate_toy_data_gauss(seed=42)
    
    #toy_data = {}
    ## load toy_data from all files in data_path
    #print("INFO: Loading toy data")
    #for files in os.listdir(data_path):
    #    if files.endswith(".parquet"):
    #        df = pd.read_parquet(os.path.join(data_path, files))
    #        toy_data[files.split(".")[0]] = df

    #toy_data = generate_toy_data_gauss(seed=42)
    toy_data = generate_toy_data_gauss(
        n_signal=100000,
        n_bkg1=200000, n_bkg2=100000, n_bkg3=100000,
        xs_signal=0.5,    # 500 fb = 0.5 pb
        xs_bkg1=50, xs_bkg2=15, xs_bkg3=10,
        lumi=100,         # in /fb
        seed=42
    )
    

    print("INFO: Plotting toy data")
    # create list of hist
    bkg_hists = [create_hist(toy_data[bkg]) for bkg in bkg_list]
    signal_hists = [create_hist(toy_data[signal]) for signal in signal_list]

    if args.plot_toy_data:
        plot_stacked_histograms(
            bkg_hists,
            bkg_list,
            output_filename=f"{data_path}/plot_log.pdf",
            axis_labels=("NN output", "Events"),
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
            output_filename=f"{data_path}/plot.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )
    nbins_list = [3, 5, 10, 15, 20]
    beta_list = [0.05, 0.05, 0.05, 0.05, 0.05]
    n_trials_list = [300, 300, 300, 300, 400]
    #nbins_list = [20]
    #beta_list = [0.1]
    #n_trials_list = [400]

    
    # Run the optimizer
    print("INFO: Running the optimizer")
    
    #nbins_list = [30]
    #beta_list = [0.05]
    #n_trials_list = [4]
    best_Z_list_bobr = []
    best_Z_list_equidistance = []
    for i, n_bins in enumerate(nbins_list):
        #if n_bins > 10:
        #    n_trials = 400
        #else:
        #    n_trials = 200
        #optimizer = BOBRBinOptimizer(toy_data, bkg_label_lst=bkg_list, signal_label_lst=signal_list,
        #                             var_label='NN_output', weight_label='weight', n_bins=n_bins, output_dir=args.output_dir+f"/optimizer_results_nbins_{n_bins}", n_trials=n_trials)
        optimizer = BOBRBinOptimizer(toy_data, bkg_label_lst=bkg_list, signal_label_lst=signal_list,
                                     var_label='NN_output', weight_label='weight', n_bins=n_bins, output_dir=args.output_dir+f"/optimizer_results_nbins_{n_bins}", n_trials=n_trials_list[i],
                                     gamma_strategy="linear", beta=beta_list[i])
        if args.run_BOBR:

            best_bins_bobr, best_hist_dict_bobr, best_Z_bobr = optimizer.optimize_bins()
            best_Z_list_bobr.append(best_Z_bobr)
            print("Optimized bin edges:", best_bins_bobr)
            print("Best significance:", best_Z_bobr)
            optimizer.visualize_optimization()
            # Save the optimized bins and significance as json
            print("INFO: Saving the optimized bins")
            os.makedirs(f"{args.output_dir}/optimizer_results_nbins_{n_bins}", exist_ok=True)
            to_save = {
                "best_bins": best_bins_bobr,
                "best_Z": best_Z_bobr
            }
            with open(f"{args.output_dir}/optimizer_results_nbins_{n_bins}/optimized_bins.json", "w") as f:
                json.dump(to_save, f, indent=4)

        # load the optimized bins
        print("INFO: Loading the optimized bins")
        with open(f"{args.output_dir}/optimizer_results_nbins_{n_bins}/optimized_bins.json", "r") as f:
            optimized_bins = json.load(f)
            best_bins_bobr = optimized_bins["best_bins"]
            best_Z_bobr = optimized_bins["best_Z"]

        # Plot the optimized bins
        print("INFO: Plotting the optimized bins")
        best_signal_hists = [create_hist(toy_data[signal], bin_edges=best_bins_bobr) for signal in signal_list]
        best_bkg_hists = [create_hist(toy_data[bkg], bin_edges=best_bins_bobr) for bkg in bkg_list]
        plot_stacked_histograms(
            best_bkg_hists,
            bkg_list,
            output_filename=f"{args.output_dir}/optimizer_results_nbins_{n_bins}/plot_log.pdf",
            axis_labels=("NN output", "Events"),
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
            output_filename=f"{args.output_dir}/optimizer_results_nbins_{n_bins}/plot.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=best_signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )
        # run the equidistance optimizer
        if args.run_equidistance:
            best_bins_eqi, best_hist_dict_eqi, best_Z_eqi = optimizer.equidistant_bins()
            best_Z_list_equidistance.append(best_Z_eqi)
            print("Optimized bin edges:", best_bins_eqi)
            print("Best significance:", best_Z_eqi)
            optimizer.visualize_optimization()
            # Save the optimized bins and significance as json
            print("INFO: Saving the optimized bins")
            os.makedirs(f"{args.output_dir}/equidistance_results_nbins_{n_bins}", exist_ok=True)
            to_save = {
                "best_bins": best_bins_eqi,
                "best_Z": best_Z_eqi
            }
            with open(f"{args.output_dir}/equidistance_results_nbins_{n_bins}/optimized_bins.json", "w") as f:
                json.dump(to_save, f, indent=4)

        # load the optimized bins
        print("INFO: Loading the optimized bins")
        with open(f"{args.output_dir}/equidistance_results_nbins_{n_bins}/optimized_bins.json", "r") as f:
            optimized_bins = json.load(f)
            best_bins_equidistance = optimized_bins["best_bins"]
            best_Z_equidistance = optimized_bins["best_Z"]

        best_signal_hists = [create_hist(toy_data[signal], bin_edges=best_bins_equidistance) for signal in signal_list]
        best_bkg_hists = [create_hist(toy_data[bkg], bin_edges=best_bins_equidistance) for bkg in bkg_list]
        plot_stacked_histograms(
            best_bkg_hists,
            bkg_list,
            output_filename=f"{args.output_dir}/equidistance_results_nbins_{n_bins}/plot_log.pdf",
            axis_labels=("NN output", "Events"),
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
            output_filename=f"{args.output_dir}/equidistance_results_nbins_{n_bins}/plot.pdf",
            axis_labels=("NN output", "Events"),
            signal_hists=best_signal_hists,
            signal_labels=signal_list,
            signal_scale=100,
            normalize=False,
            log=False,
            log_min=1e-5,
        )
        

    # plot the significance vs n_bins for BOBR and equidistance
        
    nbins_list = [3, 5, 10, 15, 20]
    bobr_z = []
    equi_z = []
    for i, n_bins in enumerate(nbins_list):
        

        # load the optimized bins
        print("INFO: Loading the optimized bins")
        with open(f"{args.output_dir}/optimizer_results_nbins_{n_bins}/optimized_bins.json", "r") as f:
            optimized_bins = json.load(f)
            bobr_z.append(optimized_bins["best_Z"])
        
        with open(f"{args.output_dir}/equidistance_results_nbins_{n_bins}/optimized_bins.json", "r") as f:
            optimized_bins = json.load(f)
            equi_z.append(optimized_bins["best_Z"])

    print(bobr_z)
    print(equi_z)
    plt.figure()
    plt.plot(nbins_list, bobr_z, marker="o", label="BOBR")
    plt.plot(nbins_list, equi_z, marker="o", label="Equidistant")
    plt.xlabel("n_bins")
    plt.ylabel("Significance")
    plt.legend()
    plt.savefig(f"{args.output_dir}/significance_vs_nbins.pdf")



if __name__ == "__main__":
    print("INFO: Running the toy example")
    main()

    


