#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from hmmlearn import hmm

RANDOM_SEED = 42

def load_data():
    df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

def discretise_temperature(df, n_temp_states=4):
    temp = df["temp"].to_numpy()
    codes, bins = pd.qcut(temp, q=n_temp_states, labels=False, retbins=True, duplicates="drop")
    return np.asarray(codes).astype(int), bins

def discretise_deaths(df):
    deaths = df["deaths"].to_numpy()
    codes, bins = pd.qcut(deaths, q=3, labels=False, retbins=True, duplicates="drop")
    return np.asarray(codes).astype(int), ["low", "medium", "high"], bins

def learn_hmm1_supervised(temp_states, deaths_codes, n_temp_states, n_death_levels=3):
    startprob = np.bincount([temp_states[0]], minlength=n_temp_states).astype(float) + 1
    startprob /= startprob.sum()

    trans = np.ones((n_temp_states, n_temp_states))
    for i, j in zip(temp_states[:-1], temp_states[1:]): trans[i, j] += 1
    trans /= trans.sum(axis=1, keepdims=True)

    emiss = np.ones((n_temp_states, n_death_levels))
    for z, o in zip(temp_states, deaths_codes): emiss[z, o] += 1
    emiss /= emiss.sum(axis=1, keepdims=True)

    model = hmm.CategoricalHMM(n_components=n_temp_states, random_state=RANDOM_SEED)
    model.startprob_ = startprob
    model.transmat_ = trans
    model.emissionprob_ = emiss
    return model

def learn_hmm2_unsupervised(deaths_codes, n_states, n_iter=100):
    X = deaths_codes.reshape(-1, 1)
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=n_iter, random_state=RANDOM_SEED)
    model.fit(X, lengths=[len(deaths_codes)])
    return model

def compute_frequencies(seq, n_levels=3):
    c = np.bincount(seq, minlength=n_levels).astype(float)
    return c / c.sum()

def compute_transition_matrix(seq, n_levels=3):
    t = np.zeros((n_levels, n_levels))
    for a, b in zip(seq[:-1], seq[1:]): t[a, b] += 1
    row_sums = t.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return t / row_sums

def transition_mse(A, B):
    return np.mean((A - B) ** 2)

def plot_frequencies(f_true, f1, f2, labels):
    x = np.arange(len(labels))
    plt.figure(figsize=(10,4))
    w = 0.25
    plt.bar(x-w, f_true, w, label="Actual")
    plt.bar(x,   f1,     w, label="HMM1")
    plt.bar(x+w, f2,     w, label="HMM2")
    plt.xticks(x, labels)
    plt.ylabel("Relative frequency")
    plt.title("Death-category frequencies")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_transition_matrices(Tt, T1, T2, labels):
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    for a, M, title in zip(ax, [Tt,T1,T2], ["Actual","HMM1","HMM2"]):
        im = a.imshow(M, origin="upper")
        a.set_xticks(range(3)), a.set_yticks(range(3))
        a.set_xticklabels(labels), a.set_yticklabels(labels)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    temp_states, _ = discretise_temperature(df)
    deaths_codes, labels, _ = discretise_deaths(df)

    n_states = len(np.unique(temp_states))

    hmm1 = learn_hmm1_supervised(temp_states, deaths_codes, n_states)
    hmm2 = learn_hmm2_unsupervised(deaths_codes, n_states)

    T = len(deaths_codes)
    d1 = hmm1.sample(T, random_state=RANDOM_SEED)[0].ravel()
    d2 = hmm2.sample(T, random_state=RANDOM_SEED)[0].ravel()

    f_true = compute_frequencies(deaths_codes)
    f1 = compute_frequencies(d1)
    f2 = compute_frequencies(d2)

    print("Frequencies:\n Actual:", f_true, "\n HMM1:", f1, "\n HMM2:", f2)

    T_true = compute_transition_matrix(deaths_codes)
    T1 = compute_transition_matrix(d1)
    T2 = compute_transition_matrix(d2)

    print("\nTransition MSE:\n HMM1:", transition_mse(T_true, T1), "\n HMM2:", transition_mse(T_true, T2))

    plot_frequencies(f_true, f1, f2, labels)
    plot_transition_matrices(T_true, T1, T2, labels)

if __name__ == "__main__":
    main()
