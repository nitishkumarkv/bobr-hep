from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

###############################################################
# scripts from: https://github.com/FloMau/gato-hep/tree/master
###############################################################

def get_gaussian_scenario(case: int = 0):
    """
    Return (MEANS, COV) for different toy scenarios.

    case = 0: baseline (original means + weak correlations)
    case = 1: signals moved closer together (more S1–S2 overlap)
    case = 2: signals further apart with backgrounds in between
    case = 3: signals share x, differ mainly in y (orthogonal separation)
    case = 4: backgrounds clustered near each signal
    case = 5: baseline means but stronger global correlations in COV
    """

    # --- baseline (your current setup) ---------------------------- #
    base_MEANS = {
        "signal1": np.array([1.5, -1.0, -1.0]),
        "signal2": np.array([-1.0, 1.5, -1.0]),
        "bkg1":    np.array([-0.5, -0.5, 1.0]),
        "bkg2":    np.array([0.5, -0.5, 0.8]),
        "bkg3":    np.array([0.5,  0.5, -0.6]),
        "bkg4":    np.array([-0.5, 1.0, -0.4]),
        "bkg5":    np.array([-0.5, 0.5, -0.2]),
    }
    base_COV = np.eye(3) * 1.0 + 0.2 * (np.ones((3, 3)) - np.eye(3))

    # start from baseline
    MEANS = {k: v.copy() for k, v in base_MEANS.items()}
    COV = base_COV.copy()

    # --- case 3: signals share x, differ mainly in y ------------- #
    if case == 0:
        MEANS["signal1"] = np.array([0.4, -0.4, 1.0])
        MEANS["signal2"] = np.array([-0.4,  0.4, 1.0])
        # backgrounds roughly around origin
        MEANS["bkg1"]    = np.array([ 0.0, 0.0,  -0.5])
        MEANS["bkg2"]    = np.array([ -0.2, 0.0,  0.0])
        MEANS["bkg3"]    = np.array([ 0.1,  0.4, -0.3])
        MEANS["bkg4"]    = np.array([-0.1,  0.6, -0.2])
        MEANS["bkg5"]    = np.array([-0.2,  0.1, -0.1])
        return MEANS, COV

    if case == 1:
        MEANS["signal1"] = np.array([0.9, -0.9, 1.0])
        MEANS["signal2"] = np.array([-0.9,  0.9, 1.0])
        # backgrounds roughly around origin
        MEANS["bkg1"]    = np.array([ 0.2, 0.2,  0.5])
        MEANS["bkg2"]    = np.array([ -0.2, 0.0,  0.0])
        MEANS["bkg3"]    = np.array([ 0.1,  0.4, -0.3])
        MEANS["bkg4"]    = np.array([-0.1,  0.6, -0.2])
        MEANS["bkg5"]    = np.array([-0.2,  0.1, -0.1])
        return MEANS, COV

    raise ValueError(f"Unknown case={case}, expected 0..5")


# 1.  Unchanged 3-D generator (now just uses global MEANS / COV)
def generate_toy_data_3class(
    case: int = 0,
    n_signal1: int = 100_000,
    n_signal2: int = 100_000,
    n_bkg: int = 500_000,
    xs_signal1: float = 0.5,
    xs_signal2: float = 0.1,
    xs_bkg1: float = 100,
    xs_bkg2: float = 80,
    xs_bkg3: float = 50,
    xs_bkg4: float = 20,
    xs_bkg5: float = 10,
    lumi: float = 100.0,
    noise_scale: float = 0.1,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Same as generate_toy_data_3class_3D, but with configurable Gaussian
    scenarios via the `case` argument.
    """

    MEANS, COV = get_gaussian_scenario(case)

    def _sample(name: str, n: int, seed_local: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed_local)
        return rng.multivariate_normal(MEANS[name], COV, size=n)

    if seed is not None:
        np.random.seed(seed)

    total_xs_bkg = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5
    n_bkg1 = int(n_bkg * xs_bkg1 / total_xs_bkg)
    n_bkg2 = int(n_bkg * xs_bkg2 / total_xs_bkg)
    n_bkg3 = int(n_bkg * xs_bkg3 / total_xs_bkg)
    n_bkg4 = int(n_bkg * xs_bkg4 / total_xs_bkg)
    n_bkg5 = n_bkg - (n_bkg1 + n_bkg2 + n_bkg3 + n_bkg4)

    processes = [
        "signal1", "signal2",
        "bkg1", "bkg2", "bkg3", "bkg4", "bkg5"
    ]
    counts = dict(
        signal1=n_signal1, signal2=n_signal2,
        bkg1=n_bkg1, bkg2=n_bkg2, bkg3=n_bkg3, bkg4=n_bkg4, bkg5=n_bkg5,
    )
    xs = dict(
        signal1=xs_signal1, signal2=xs_signal2,
        bkg1=xs_bkg1, bkg2=xs_bkg2, bkg3=xs_bkg3, bkg4=xs_bkg4, bkg5=xs_bkg5,
    )

    raw = {
        p: _sample(p, counts[p], seed + i if seed is not None else None)
        for i, p in enumerate(processes)
    }
    # apply noise to the sampled data to mimic detector effects
    for p in processes:
        raw[p] *= np.random.normal(1.0, noise_scale, size=raw[p].shape)

    pdfs = {p: multivariate_normal(MEANS[p], COV) for p in processes}

    bkg_procs = [p for p in processes if p.startswith("bkg")]
    total_bkg_xs = sum(xs[p] for p in bkg_procs)

    def pb(X):
        return sum((xs[p] / total_bkg_xs) * pdfs[p].pdf(X) for p in bkg_procs)

    data = {}
    for proc in processes:
        X = raw[proc]
        w = xs[proc] * lumi / counts[proc]

        p1 = pdfs["signal1"].pdf(X)
        p2 = pdfs["signal2"].pdf(X)
        pB = pb(X)
        tot = p1 + p2 + pB + 1e-12

        nn_output = np.column_stack((p1 / tot, p2 / tot, pB / tot))
        data[proc] = pd.DataFrame({
            "NN_output": list(nn_output),
            "weight": w,
        })
    return data


def generate_toy_data_1D(
    case: int = 0,
    n_signal: int = 100_000,
    n_bkg: int = 100_000,
    xs_signal: float = 0.5,
    xs_bkg1: float = 100,
    xs_bkg2: float = 80,
    xs_bkg3: float = 50,
    xs_bkg4: float = 20,
    xs_bkg5: float = 10,
    lumi: float = 100.0,
    noise_scale: float = 0.1,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate 1D toy data for signal and background events.

    Parameters
    ----------
    n_signal : int, optional
        Number of signal events to generate. Default is 100000.
    n_bkg : int, optional
        Number of background events to generate. Default is 300000.
    xs_signal : float, optional
        Cross-section for signal events. Default is 0.5.
    xs_bkg1 : float, optional
        Cross-section for the first background component. Default is 50.
    xs_bkg2 : float, optional
        Cross-section for the second background component. Default is 15.
    xs_bkg3 : float, optional
        Cross-section for the third background component. Default is 10.
    xs_bkg4 : float, optional
        Cross-section for the fourth background component. Default is 20.
    xs_bkg5 : float, optional
        Cross-section for the fifth background component. Default is 10.
    lumi : float, optional
        Luminosity for scaling event weights. Default is 100.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    dict of pandas.DataFrame
        A dictionary of DataFrames, each containing the generated toy data
        with columns "NN_output" and "weight".
    """

    MEANS, COV = get_gaussian_scenario(case)

    def _sample(name: str, n: int, seed_local: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed_local)
        return rng.multivariate_normal(MEANS[name], COV, size=n)

    if seed is not None:
        np.random.seed(seed)

    tot_xs_bkg = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5
    n_bkg1 = int(n_bkg * xs_bkg1 / tot_xs_bkg)
    n_bkg2 = int(n_bkg * xs_bkg2 / tot_xs_bkg)
    n_bkg3 = int(n_bkg * xs_bkg3 / tot_xs_bkg)
    n_bkg4 = int(n_bkg * xs_bkg4 / tot_xs_bkg)
    n_bkg5 = n_bkg - (n_bkg1 + n_bkg2 + n_bkg3 + n_bkg4)

    counts = dict(
        signal=n_signal,
        bkg1=n_bkg1,
        bkg2=n_bkg2,
        bkg3=n_bkg3,
        bkg4=n_bkg4,
        bkg5=n_bkg5,
    )
    xs = dict(
        signal=xs_signal,
        bkg1=xs_bkg1,
        bkg2=xs_bkg2,
        bkg3=xs_bkg3,
        bkg4=xs_bkg4,
        bkg5=xs_bkg5,
    )

    X = {
        "signal": _sample("signal1", n_signal, seed),
        "bkg1":   _sample("bkg1",    n_bkg1,  seed + 1 if seed else None),
        "bkg2":   _sample("bkg2",    n_bkg2,  seed + 2 if seed else None),
        "bkg3":   _sample("bkg3",    n_bkg3,  seed + 3 if seed else None),
        "bkg4":   _sample("bkg4",    n_bkg4,  seed + 4 if seed else None),
        "bkg5":   _sample("bkg5",    n_bkg5,  seed + 5 if seed else None),
    }

    # apply noise to the sampled data to mimic detector effects
    for proc in X:
        X[proc] *= np.random.normal(1.0, noise_scale, size=X[proc].shape)

    pdf_sig = multivariate_normal(MEANS["signal1"], COV)
    pdf_bkg = {
        p: multivariate_normal(MEANS[p], COV)
        for p in ("bkg1", "bkg2", "bkg3", "bkg4", "bkg5")
    }
    total_bkg_xs = xs_bkg1 + xs_bkg2 + xs_bkg3 + xs_bkg4 + xs_bkg5

    def _pb(x):
        return sum((xs[p] / total_bkg_xs) * pdf_bkg[p].pdf(x) for p in pdf_bkg)

    data = {}
    for proc in ("signal", "bkg1", "bkg2", "bkg3", "bkg4", "bkg5"):
        Xp = X[proc]
        ps = pdf_sig.pdf(Xp)
        pb = _pb(Xp)

        lr = ps / (pb + 1e-12)
        disc = lr / (1.0 + lr)        # map to (0,1)

        data[proc] = pd.DataFrame({
            "NN_output": disc,
            "weight": xs[proc] * lumi / counts[proc],
        })

    return data