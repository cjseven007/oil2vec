"""STEP 10 - Sequential design: "which simulations should we ask for NEXT?"  (upgrade 4)

Plain-language idea
-------------------
Each simulation is expensive. Instead of running 100 simulations at random spots in advance, run a few,
look at where the proxy is MOST UNSURE, run the next ones there, and repeat. Like studying for an exam:
you do not re-read the chapters you already know, you drill the ones you keep getting wrong.

Two parts:
  A. RETROSPECTIVE TEST (we only have the 100 sims we already have, so we pretend we are choosing
     them one batch at a time) -> does "ask where unsure" beat "choose at random"?
  B. RECOMMENDATION -> a list of new parameter sets worth simulating, taken from the region the history
     match says is plausible (the posterior from step 8). This needs the organisers' simulator
     (or permission to ask for more runs): it is a question for the Q&A session.

Run alone:  python -m src.active_design
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import embed as E
from . import load_data as L
from . import mcmc as M
from . import proxy_baseline as PB

warnings.filterwarnings("ignore")   # sklearn GP convergence chatter
BAND = (1.28, 1.42)             # porosity band that the history match says is plausible (approx. posterior +/- margin)
N_PCA, N_PC = 8, 6


def fit(theta, X, x_obs):
    pca = E.fit_pca(X, x_obs, N_PCA)[0]
    return PB.PCAGPProxy(N_PC, n_restarts=1).fit(theta, X, pca)


def curve_error(proxy, theta, X):
    """mean over cases of RMSE(curves)/RMS(truth)  (all 4 blocks together, scaled units) in %"""
    P = proxy.predict_curves(theta)
    return 100 * np.mean(np.sqrt(((P - X) ** 2).mean(axis=1)) / np.sqrt((X ** 2).mean(axis=1)))


# ---------------------------------------------------------------------------
# Part A - retrospective test
# ---------------------------------------------------------------------------
def choose(strategy, proxy, theta, pool, chosen, rng, k, nrm):
    """Pick k more cases from `pool` (indices not yet chosen)."""
    pool = np.array(pool)
    if strategy == "random":
        return list(rng.choice(pool, size=k, replace=False))
    picked = []
    cur = list(chosen)
    for _ in range(k):
        rest = np.array([i for i in pool if i not in picked])
        if strategy == "maximin":                             # farthest from everything chosen so far
            d = np.linalg.norm((theta[rest][:, None] - theta[cur][None]) / nrm, axis=2).min(axis=1)
            j = rest[np.argmax(d)]
        else:                                                 # 'unsure' strategies
            sd = proxy.predict_std(theta[rest])
            if strategy == "unsure_focus":                    # care more about the plausible band
                sd = sd * np.exp(-0.5 * ((theta[rest][:, 1] - 1.34) / 0.10) ** 2)
            j = rest[np.argmax(sd)]
        picked.append(j); cur.append(j)
    return picked


def retrospective(X, x_obs, theta, n_seeds=6, n0=12, batch=4, n_max=56):
    va = np.array(C.VAL_CASES) - 1
    in_band = np.where((theta[:, 1] > BAND[0]) & (theta[:, 1] < BAND[1]))[0]
    in_band = in_band[in_band < 85]
    band_test = in_band[::2]                                   # hold half of the in-band cases out as a special exam
    pool0 = [i for i in range(70) if i not in set(band_test)]
    nrm = theta.std(axis=0)
    nrm[0] = 1e6                                               # fault transmissibility does nothing -> ignore it
    strategies = ["random", "maximin", "unsure", "unsure_focus"]
    rows = []
    for seed in range(n_seeds):
        for st in strategies:
            rng = np.random.default_rng(seed)
            chosen = list(rng.choice(pool0, size=n0, replace=False))   # same starting set for every strategy
            while True:
                proxy = fit(theta[chosen], X[chosen], x_obs)
                rows.append({"seed": seed, "strategy": st, "n_sims": len(chosen),
                             "err_all_%": curve_error(proxy, theta[va], X[va]),
                             "err_inband_%": curve_error(proxy, theta[band_test], X[band_test])})
                if len(chosen) >= n_max:
                    break
                pool = [i for i in pool0 if i not in chosen]
                chosen += choose(st, proxy, theta, pool, chosen, rng, batch, nrm)
        print(f"  seed {seed + 1}/{n_seeds} done", flush=True)
    return pd.DataFrame(rows), band_test


def plot_retro(R, band_test):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, t in zip(axs, ["err_all_%", "err_inband_%"],
                          ["Exam 1: 15 validation cases (all parameter values)",
                           f"Exam 2: {len(band_test)} cases in the plausible porosity band"]):
        for st, g in R.groupby("strategy"):
            m = g.groupby("n_sims")[col].agg(["mean", "std"])
            ax.plot(m.index, m["mean"], marker="o", ms=3, label=st)
            ax.fill_between(m.index, m["mean"] - m["std"] / 2, m["mean"] + m["std"] / 2, alpha=0.12)
        ax.set_yscale("log"); ax.set_xlabel("number of simulations used for training")
        ax.set_ylabel("proxy error (% of curve size)"); ax.set_title(t, fontsize=10)
    axs[0].legend(title="how the next sim is chosen", fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "60_active_design_retrospective.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Part B - recommendation
# ---------------------------------------------------------------------------
def recommend(X, x_obs, theta, k=10, seed=0):
    """Greedy: among posterior samples, repeatedly take the one that is (a) most uncertain for the proxy and
    (b) far from every simulation we already have or already picked."""
    idx = np.array(C.TRAIN_CASES + C.VAL_CASES) - 1
    proxy = fit(theta[idx], X[idx], x_obs)
    post = np.load(C.CACHE_DIR / "posterior_mcmc.npy")[:, :4]
    rng = np.random.default_rng(seed)
    cand = post[rng.choice(len(post), size=min(2000, len(post)), replace=False)]
    sd = proxy.predict_std(cand)
    sd = sd / sd.max()
    scale = post.std(axis=0)
    scale[0] = 1e6                                             # ignore fault_tran for 'distance'
    have = list(theta[idx])
    picked = []
    for _ in range(k):
        d = np.linalg.norm((cand[:, None] - np.array(have)[None]) / scale, axis=2).min(axis=1)
        score = sd * (1 - np.exp(-0.5 * d ** 2))
        j = int(np.argmax(score))
        picked.append(cand[j]); have.append(cand[j])
    rec = pd.DataFrame(picked, columns=C.PARAMS)
    rec.index = np.arange(1, k + 1); rec.index.name = "new_sim"
    n_in = int(((theta[idx, 1] > BAND[0]) & (theta[idx, 1] < BAND[1])).sum())
    print(f"existing sims (cases 1-85) with porosity in the plausible band {BAND}: {n_in}")
    return rec, theta[idx], post, proxy, cand, sd


def plot_recommend(rec, theta_have, post):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (a, b) in zip(axs, [(1, 2), (1, 3), (2, 3)]):
        ax.scatter(post[::4, a], post[::4, b], s=4, c="#bbb", label="plausible region (posterior)")
        ax.scatter(theta_have[:, a], theta_have[:, b], s=16, c="#4c78a8", label="sims we have")
        ax.scatter(rec.iloc[:, a], rec.iloc[:, b], s=70, marker="*", c="#e45756", label="recommended new sims")
        ax.set_xlabel(C.PARAMS[a]); ax.set_ylabel(C.PARAMS[b])
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "61_recommended_simulations.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def run_all(n_seeds=6):
    C.ensure_dirs()
    U = L.load_uncertainty()
    X, x_obs, dates, lens = M.build_data()
    theta = U[C.PARAMS].values
    print("A. retrospective test (fits many small GPs, ~2 min) ...")
    R, band_test = retrospective(X, x_obs, theta, n_seeds=n_seeds)
    R.to_csv(C.TABLE_DIR / "active_design_retrospective.csv", index=False)
    plot_retro(R, band_test)
    for n in (24, 40, 56):
        t = R[R.n_sims == n].groupby("strategy")[["err_all_%", "err_inband_%"]].mean().round(3)
        print(f"\nproxy error (%), mean over {n_seeds} repeats, with {n} sims:\n", t.to_string())
    print("\nB. recommendation ...")
    rec, have, post, *_ = recommend(X, x_obs, theta)
    rec.to_csv(C.TABLE_DIR / "recommended_new_simulations.csv")
    plot_recommend(rec, have, post)
    print(rec.round(4).to_string())
    return R, rec


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    run_all()
