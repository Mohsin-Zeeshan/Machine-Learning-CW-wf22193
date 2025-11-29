# hmms.py
#
# Hidden Markov Models for deaths_and_temps_england_wales.csv
#
# HMM1: parameters learned "by hand" (supervised) using the
#       discretised temperature sequence as the hidden state
#       and discretised deaths as the observations.
#
# HMM2: parameters learned using hmmlearn (unsupervised) from
#       the discretised deaths sequence only.
#
# Both models can be sampled using hmmlearn's sample() method.

import numpy as np
import pandas as pd

import pymc as pm
from hmmlearn import hmm

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def load_data() -> pd.DataFrame:
    """
    Load deaths and temperature data as in the PyMC notebook.
    """
    df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def discretise_temperature(df: pd.DataFrame, n_temp_states: int = 4):
    """
    Discretise the 'temp' column into n_temp_states bins.
    Returns:
        temp_states: np.ndarray of shape (T,) with integer codes [0, ..., n_temp_states-1]
        bin_edges: np.ndarray of bin edges used by qcut
    """
    temp = df["temp"].to_numpy()

    # Use quantile-based bins so each state has roughly equal frequency
    temp_codes, bin_edges = pd.qcut(
        temp,
        q=n_temp_states,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    # temp_codes can be a Series or a NumPy array depending on pandas version
    temp_codes = np.asarray(temp_codes).astype(int)  # <-- changed
    return temp_codes, bin_edges


def discretise_deaths(df: pd.DataFrame):
    """
    Discretise 'deaths' into three categories: low, medium, high.
    Returns:
        deaths_codes: np.ndarray of shape (T,) with values 0,1,2
        labels: list of category names
        bin_edges: np.ndarray of thresholds for the categories
    """
    deaths = df["deaths"].to_numpy()

    # Tertiles: roughly equal numbers of months in each category
    deaths_codes, bin_edges = pd.qcut(
        deaths,
        q=3,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    # deaths_codes can be a Series or a NumPy array depending on pandas version
    deaths_codes = np.asarray(deaths_codes).astype(int)  # <-- changed

    labels = ["low", "medium", "high"]
    return deaths_codes, labels, bin_edges


def learn_hmm1_supervised(
    temp_states: np.ndarray,
    deaths_codes: np.ndarray,
    n_temp_states: int,
    n_death_levels: int = 3,
) -> hmm.CategoricalHMM:
    """
    Learn HMM1 parameters using supervised frequency estimates.

    Hidden state z_t = discretised temperature (known).
    Observation o_t = discretised deaths (0,1,2).

    Returns:
        model: hmm.CategoricalHMM with parameters set by hand.
    """
    assert temp_states.shape == deaths_codes.shape, "Lengths must match"
    T = len(temp_states)

    # --- initial state distribution pi ---
    # Use the first observed temperature as the start state, with a bit of smoothing.
    start_counts = np.zeros(n_temp_states, dtype=float)
    start_counts[temp_states[0]] += 1.0
    # Laplace smoothing so we don't get exact zeros
    start_counts += 1.0
    startprob = start_counts / start_counts.sum()

    # --- transition matrix A ---
    trans_counts = np.ones((n_temp_states, n_temp_states), dtype=float)  # Laplace(1)
    for t in range(T - 1):
        i = temp_states[t]
        j = temp_states[t + 1]
        trans_counts[i, j] += 1.0

    transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    # --- emission matrix B ---
    emission_counts = np.ones((n_temp_states, n_death_levels), dtype=float)  # Laplace(1)
    for z, o in zip(temp_states, deaths_codes):
        emission_counts[z, o] += 1.0

    emissionprob = emission_counts / emission_counts.sum(axis=1, keepdims=True)

    # Build the hmmlearn CategoricalHMM
    model = hmm.CategoricalHMM(
        n_components=n_temp_states,
        random_state=RANDOM_SEED,
    )

    # Set learned parameters
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob

    return model


def learn_hmm2_unsupervised(
    deaths_codes: np.ndarray,
    n_states: int,
    n_iter: int = 100,
) -> hmm.CategoricalHMM:
    """
    Learn HMM2 using standard Baum-Welch (unsupervised) on deaths only.

    Observations: deaths_codes (0,1,2). Hidden states are not constrained
    to correspond to temperatures – they are latent.
    """
    X = deaths_codes.reshape(-1, 1)  # shape (T, 1)

    model = hmm.CategoricalHMM(
        n_components=n_states,
        random_state=RANDOM_SEED,
        n_iter=n_iter,
        verbose=False,
    )

    model.fit(X, lengths=[len(deaths_codes)])

    return model


def main():
    df = load_data()

    # Discretise variables
    temp_states, temp_bins = discretise_temperature(df, n_temp_states=4)
    deaths_codes, death_labels, death_bins = discretise_deaths(df)

    n_temp_states = len(np.unique(temp_states))
    n_death_levels = len(death_labels)

    print("Temperature bins (edges):", temp_bins)
    print("Death bins (edges):", death_bins)
    print("Number of temperature states:", n_temp_states)
    print("Death categories:", death_labels)

    # --- HMM1: supervised learning ---
    hmm1 = learn_hmm1_supervised(
        temp_states=temp_states,
        deaths_codes=deaths_codes,
        n_temp_states=n_temp_states,
        n_death_levels=n_death_levels,
    )

    # --- HMM2: unsupervised learning ---
    hmm2 = learn_hmm2_unsupervised(
        deaths_codes=deaths_codes,
        n_states=n_temp_states,
        n_iter=100,
    )

    # Small sanity-check: sample short sequences from each model
    n_samples = 24  # e.g. 2 years of monthly data
    deaths1, states1 = hmm1.sample(n_samples=n_samples, random_state=RANDOM_SEED)
    deaths2, states2 = hmm2.sample(n_samples=n_samples, random_state=RANDOM_SEED)

    print("\nSampled deaths sequence from HMM1:", deaths1.ravel())
    print("Sampled hidden states (temperature) from HMM1:", states1)

    print("\nSampled deaths sequence from HMM2:", deaths2.ravel())
    print("Sampled hidden states from HMM2:", states2)


if __name__ == "__main__":
    main()
