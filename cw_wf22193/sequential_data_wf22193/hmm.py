# hmms.py
import argparse

import numpy as np
import pandas as pd
import pymc as pm
from hmmlearn.hmm import CategoricalHMM

RANDOM_SEED = 42


def loadData() -> pd.DataFrame:
    """
    Load the deaths and temperature data used in the PyMC example.
    """
    df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def discretiseDeaths(df: pd.DataFrame, nLevels: int = 3):
    """
    Discretise deaths into nLevels ordered categories (0..nLevels-1)
    using quantiles, so that each category has roughly the same number
    of observations.
    """
    # Use qcut with duplicates='drop' to handle duplicate bin edges
    df = df.copy()
    deathLevel, bins = pd.qcut(
        df["deaths"],
        q=nLevels,
        labels=list(range(nLevels)),
        retbins=True,
        duplicates="drop",
    )
    df["deathLevel"] = deathLevel.astype(int)

    return df, bins


def discretiseTemp(df: pd.DataFrame, nStates: int = 3):
    """
    Discretise temperature into nStates ordered categories (0..nStates-1)
    using quantiles.
    """
    df = df.copy()
    tempState, tempBins = pd.qcut(
        df["temp"],
        q=nStates,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    df["tempState"] = tempState.astype(int)
    return df, tempBins


def estimateSupervisedParams(
    tempStates: np.ndarray,
    deathLevels: np.ndarray,
    nStates: int,
    nObs: int,
):
    """
    Maximum-likelihood parameter estimates for an HMM when the state
    sequence (tempStates) is known (supervised learning).

    Returns startProb, transMat, emissionProb.
    """
    T = len(tempStates)
    assert T == len(deathLevels)

    startCounts = np.zeros(nStates)
    transCounts = np.zeros((nStates, nStates))
    emitCounts = np.zeros((nStates, nObs))

    # Initial state distribution
    startCounts[tempStates[0]] += 1

    # Transitions
    for t in range(T - 1):
        i = tempStates[t]
        j = tempStates[t + 1]
        transCounts[i, j] += 1

    # Emissions
    for t in range(T):
        i = tempStates[t]
        k = deathLevels[t]
        emitCounts[i, k] += 1

    # Add-one (Laplace) smoothing to avoid zeros
    startProb = (startCounts + 1.0) / (startCounts.sum() + nStates)
    transMat = (transCounts + 1.0) / (
        transCounts.sum(axis=1, keepdims=True) + nStates
    )
    emissionProb = (emitCounts + 1.0) / (
        emitCounts.sum(axis=1, keepdims=True) + nObs
    )

    return startProb, transMat, emissionProb


def buildHMM1(
    tempStates: np.ndarray,
    deathLevels: np.ndarray,
    nStates: int,
    nObs: int,
) -> CategoricalHMM:
    """
    HMM1: supervised learning.
    State sequence = discretised temperature.
    Observation sequence = discretised deaths (0, 1, 2).
    """
    startProb, transMat, emissionProb = estimateSupervisedParams(
        tempStates, deathLevels, nStates, nObs
    )

    model = CategoricalHMM(
        n_components=nStates,
        init_params="",       # do not re-initialise params
        n_iter=1,
        random_state=RANDOM_SEED,
    )
    model.startprob_ = startProb
    model.transmat_ = transMat
    model.emissionprob_ = emissionProb
    model.n_features = nObs  # number of possible observation symbols

    return model


def buildHMM2(
    deathLevels: np.ndarray,
    nStates: int,
    nObs: int,
) -> CategoricalHMM:
    """
    HMM2: unsupervised learning with hmmlearn.
    Only the deathLevels sequence is used as data.
    """
    X = deathLevels.reshape(-1, 1)

    model = CategoricalHMM(
        n_components=nStates,
        random_state=RANDOM_SEED,
        n_iter=100,
    )
    # Fit on the full sequence (single sequence, so length is len(X))
    model.fit(X, lengths=[len(X)])

    return model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequential data task: HMMs for deaths and temperatures "
            "in England & Wales."
        )
    )
    parser.add_argument(
        "--n_temp_states",
        type=int,
        default=3,
        help="Number of discrete temperature states.",
    )
    parser.add_argument(
        "--n_death_levels",
        type=int,
        default=3,
        help="Number of discrete death levels (low/med/high).",
    )
    parser.add_argument(
        "--sample_length",
        type=int,
        default=120,
        help="Length of sequences to sample from each HMM.",
    )
    args = parser.parse_args()

    # Load and discretise data
    df = loadData()
    df, tempBins = discretiseTemp(df, nStates=args.n_temp_states)
    df, deathBins = discretiseDeaths(df, nLevels=args.n_death_levels)

    tempStates = df["tempState"].to_numpy()
    deathLevels = df["deathLevel"].to_numpy()

    nStates = args.n_temp_states
    nObs = args.n_death_levels

    # HMM1: supervised
    hmm1 = buildHMM1(tempStates, deathLevels, nStates, nObs)

    # HMM2: unsupervised
    hmm2 = buildHMM2(deathLevels, nStates, nObs)

    print("=== HMM1 (supervised) ===")
    print("Start probabilities:\n", hmm1.startprob_)
    print("Transition matrix:\n", hmm1.transmat_)
    print("Emission matrix (P(deathLevel | tempState)):\n", hmm1.emissionprob_)

    print("\n=== HMM2 (unsupervised) ===")
    print("Start probabilities:\n", hmm2.startprob_)
    print("Transition matrix:\n", hmm2.transmat_)
    print("Emission matrix:\n", hmm2.emissionprob_)

    # Sample sequences for Report Task 8
    sampleLength = args.sample_length

    deathsSample1, _ = hmm1.sample(sampleLength, random_state=RANDOM_SEED)
    deathsSample2, _ = hmm2.sample(sampleLength, random_state=RANDOM_SEED)

    print(f"\nFirst 20 sampled death levels from HMM1: {deathsSample1[:20].ravel()}")
    print(f"First 20 sampled death levels from HMM2: {deathsSample2[:20].ravel()}")


if __name__ == "__main__":
    main()
