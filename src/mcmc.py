"""STEP 8 - Better history matching  (upgrades 1, 2 and 3)

 (1) Match on cumulative AND rate curves (gas dropped: gas = oil x constant, so it only double-counts oil)
 (2) Admit that the model is imperfect: add "discrepancy" noise levels sigma that are LEARNED, so the
     posterior is not over-confident when no case can reproduce the observed history exactly
 (3) Replace the ABC rejection trick by real MCMC - the affine-invariant ensemble sampler
     (Goodman & Weare 2010) used in paper 1 (SPE-203976) - implemented from scratch below.

Simple picture of MCMC: 64 "walkers" wander around the 4-parameter space. Each step a walker jumps
towards/away from another walker; it keeps the jump if the new spot explains the data better (or,
sometimes, even if slightly worse). After many steps, the cloud of walkers is spread out in
proportion to how plausible each parameter combination is = the posterior.

Run alone:  python -m src.mcmc
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import config as C
from . import clean as K
from . import embed as E
from . import load_data as L
from . import proxy_baseline as PB

BLOCK_NAMES = ["cum_oil", "cum_wat", "rate_oil", "rate_wat"]
GROUP_OF_BLOCK = [0, 0, 1, 1]               # group 0 = cumulative, group 1 = rate
GROUP_NAMES = ["sigma_cum", "sigma_rate"]
LOG_SIG_BOUNDS = (np.log(1e-4), np.log(0.5))
PARAM_NAMES = C.PARAMS + GROUP_NAMES


# ---------------------------------------------------------------------------
# Data in the shape the proxy needs
# ---------------------------------------------------------------------------
def build_data():
    """Features = [cum_oil | cum_wat | rate_oil | rate_wat], every block divided by the observed final
    value of that variable.  Rates use the 'average over the next quarter' definition (see
    clean.interval_rates) so simulated and observed rates mean the same thing.
    Returns X (100, total), x_obs (1, total), dates, lens (length of each block)."""
    cube, obs, dates = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")
    sim_r, obs_r, _ = K.interval_rates(cube, robs, dates)
    blocks = [cube["oil"] / obs["oil"][-1], cube["wat"] / obs["wat"][-1],
              sim_r["oil"] / robs["oil"][-1], sim_r["wat"] / robs["wat"][-1]]
    ob = [obs["oil"] / obs["oil"][-1], obs["wat"] / obs["wat"][-1],
          obs_r["oil"] / robs["oil"][-1], obs_r["wat"] / robs["wat"][-1]]
    lens = [b.shape[1] for b in blocks]
    return np.hstack(blocks), np.hstack(ob)[None, :], dates, lens


def fit_final_proxy(theta, X, x_obs, n_pca=8, n_pc=6):
    """PCA + one Gaussian process per component, trained on cases 1-85 (train+val)."""
    idx = np.array(C.TRAIN_CASES + C.VAL_CASES) - 1
    pca = E.fit_pca(X[idx], x_obs, n_pca)[0]
    return PB.PCAGPProxy(n_pc).fit(theta[idx], X[idx], pca)


# ---------------------------------------------------------------------------
# Likelihood with learned discrepancy (and a correction for correlated residuals)
# ---------------------------------------------------------------------------
class Likelihood:
    """log p(data | theta, sigma).  Residuals of neighbouring quarters are strongly correlated (a
    systematic offset, not independent noise), so 41 points are NOT 41 independent pieces of
    evidence.  We estimate the lag-1 autocorrelation rho of the residuals at a reference fit and count
    N_eff = N (1-rho)/(1+rho) 'independent' points instead (a standard AR(1) correction)."""

    def __init__(self, proxy, x_obs, lens, bounds, n_eff_factor=(1.0, 1.0), keep=None):
        """keep = how many points of each block to USE (default all). Used by the forecast back-test
        to pretend that the last years are not yet observed."""
        self.proxy, self.x_obs, self.bounds = proxy, x_obs, bounds
        ends = np.cumsum(lens)
        keep = list(lens) if keep is None else list(keep)
        self.slices = [slice(e - l, e - l + k) for e, l, k in zip(ends, lens, keep)]
        self.n_pts = np.array([sum(k for k, g in zip(keep, GROUP_OF_BLOCK) if g == j) for j in (0, 1)], float)
        self.f = np.array(n_eff_factor)            # N_eff / N per group

    def sse(self, F):
        """Sum of squared (scaled) residuals per group for predicted features F (m, 4*n_t)."""
        out = np.zeros((len(F), 2))
        for sl, g in zip(self.slices, GROUP_OF_BLOCK):
            out[:, g] += ((F[:, sl] - self.x_obs[:, sl]) ** 2).sum(axis=1)
        return out

    def __call__(self, P):
        P = np.atleast_2d(P)
        lo, hi = self.bounds
        th, ls = P[:, :4], P[:, 4:]
        ok = np.all((th >= lo) & (th <= hi), axis=1) & np.all((ls >= LOG_SIG_BOUNDS[0]) & (ls <= LOG_SIG_BOUNDS[1]), axis=1)
        ll = np.full(len(P), -np.inf)
        if ok.any():
            S = self.sse(self.proxy.predict_curves(th[ok]))
            n_eff = self.n_pts * self.f
            sig2 = np.exp(2 * ls[ok])
            ll[ok] = np.sum(-n_eff * ls[ok] - 0.5 * self.f * S / sig2, axis=1)
        return ll


def residual_lag1(proxy, x_obs, slices, theta):
    """lag-1 autocorrelation of the residual curves at a given theta, per group."""
    F = proxy.predict_curves(theta[None, :])[0]
    rho = [[], []]
    for sl, g in zip(slices, GROUP_OF_BLOCK):
        r = F[sl] - x_obs[0, sl]
        r = r - r.mean()
        rho[g].append((r[:-1] * r[1:]).sum() / max((r * r).sum(), 1e-30))
    return np.clip([np.mean(r) for r in rho], 0.0, 0.95)


# ---------------------------------------------------------------------------
# Affine-invariant ensemble sampler (Goodman & Weare) - the "AIES" of paper 1
# ---------------------------------------------------------------------------
def aies(logp, p0, n_steps, a=2.0, seed=0):
    rng = np.random.default_rng(seed)
    nw, d = p0.shape
    assert nw % 2 == 0
    h = nw // 2
    p, lp = p0.copy(), logp(p0)
    chain = np.empty((n_steps, nw, d))
    lps = np.empty((n_steps, nw))
    acc = 0
    for t in range(n_steps):
        for s in (0, 1):                               # update one half using the other half
            me = slice(0, h) if s == 0 else slice(h, nw)
            other = p[h:] if s == 0 else p[:h]
            z = ((a - 1) * rng.random(h) + 1) ** 2 / a
            partner = other[rng.integers(h, size=h)]
            prop = partner + z[:, None] * (p[me] - partner)
            lp_new = logp(prop)
            log_r = (d - 1) * np.log(z) + lp_new - lp[me]
            ok = np.log(rng.random(h)) < log_r
            p[me] = np.where(ok[:, None], prop, p[me])
            lp[me] = np.where(ok, lp_new, lp[me])
            acc += ok.sum()
        chain[t], lps[t] = p, lp
    return chain, lps, acc / (n_steps * nw)


def autocorr_time(chain, c=5.0):
    """Integrated autocorrelation time per dimension (how many steps until samples are ~independent).
    Estimated from the walker-averaged chain (Sokal's automatic windowing)."""
    x = chain.mean(axis=1)                              # (steps, dim)
    taus = []
    for j in range(x.shape[1]):
        v = x[:, j] - x[:, j].mean()
        f = np.fft.rfft(v, n=2 * len(v))
        acf = np.fft.irfft(f * np.conj(f))[: len(v)]
        acf /= acf[0]
        tau_run = 2 * np.cumsum(acf) - 1
        m = np.arange(len(tau_run)) < c * tau_run
        w = np.argmin(m) if (~m).any() else len(m) - 1
        taus.append(tau_run[w])
    return np.array(taus)


# ---------------------------------------------------------------------------
def initial_walkers(like, n_walkers, seed=0):
    """Start the walkers in a sensible place: the best prior draws (by mismatch), not random junk."""
    lo, hi = like.bounds
    th = lo + qmc.Sobol(4, scramble=True, seed=seed).random(2 ** 15) * (hi - lo)
    S = like.sse(like.proxy.predict_curves(th))
    best = np.argsort((S / like.n_pts).sum(axis=1))[:n_walkers]
    ls0 = np.log([0.02, 0.1])
    rng = np.random.default_rng(seed)
    return np.column_stack([th[best], ls0 + 0.1 * rng.standard_normal((n_walkers, 2))]), th[best[0]]


def plot_corner(S, bounds, names, fname):
    d = S.shape[1]
    fig, axs = plt.subplots(d, d, figsize=(2.0 * d, 2.0 * d))
    for i in range(d):
        for j in range(d):
            ax = axs[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                ax.hist(S[:, i], bins=25, color="#4c78a8")
            else:
                ax.scatter(S[:, j], S[:, i], s=2, alpha=0.25)
            if i == d - 1: ax.set_xlabel(names[j], fontsize=7)
            if j == 0 and i > 0: ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / fname, dpi=120); plt.close(fig)


def plot_traces(chain, burn, fname):
    d = chain.shape[2]
    fig, axs = plt.subplots(d, 1, figsize=(9, 1.6 * d), sharex=True)
    for j in range(d):
        axs[j].plot(chain[:, ::4, j], lw=0.3, alpha=0.5)
        axs[j].axvline(burn, color="r", ls="--"); axs[j].set_ylabel(PARAM_NAMES[j], fontsize=8)
    axs[-1].set_xlabel("step  (red line = end of burn-in)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / fname, dpi=120); plt.close(fig)


def plot_predictive(proxy, S, like, dates, fname):
    """Posterior predictive: do the history-matched models cover the observed data?"""
    idx = np.random.default_rng(0).choice(len(S), 600, replace=False)
    F = proxy.predict_curves(S[idx, :4])
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.4))
    for b, ax in enumerate(axs):
        sl = like.slices[b]
        d = dates[: sl.stop - sl.start]                 # rate blocks have one point fewer
        lo, med, hi = np.percentile(F[:, sl], [5, 50, 95], axis=0)
        ax.fill_between(d, lo, hi, color="#4c78a8", alpha=0.35, label="posterior 5-95%")
        ax.plot(d, med, color="#4c78a8")
        ax.plot(d, like.x_obs[0, sl], "k.", ms=4, label="observed")
        ax.set_title(BLOCK_NAMES[b] + " (scaled)")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / fname, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
def run_all(n_walkers=64, n_steps=20000, burn=5000, seed=C.SEED):
    C.ensure_dirs()
    U = L.load_uncertainty()
    X, x_obs, dates, lens = build_data()
    theta = U[C.PARAMS].values
    bounds = (theta.min(0), theta.max(0))

    t0 = time.time()
    proxy = fit_final_proxy(theta, X, x_obs)
    print(f"proxy fitted in {time.time() - t0:.1f}s")

    like0 = Likelihood(proxy, x_obs, lens, bounds)
    P0, best_theta = initial_walkers(like0, n_walkers, seed)
    rho = residual_lag1(proxy, x_obs, like0.slices, best_theta)
    f = (1 - rho) / (1 + rho)
    print(f"lag-1 residual autocorrelation (cum, rate) = {rho.round(2)}  ->  "
          f"{like0.n_pts.astype(int).tolist()} data points per group count as ~{(like0.n_pts * f).round(1).tolist()} independent")
    like = Likelihood(proxy, x_obs, lens, bounds, n_eff_factor=f)

    t0 = time.time()
    chain, lps, acc = aies(like, P0, n_steps, seed=seed)
    print(f"MCMC: {n_walkers} walkers x {n_steps} steps in {time.time() - t0:.0f}s, acceptance = {acc:.2f} "
          f"(healthy range ~0.2-0.5)")

    tau = autocorr_time(chain[burn:])
    ess = (n_steps - burn) * n_walkers / tau
    print("autocorrelation time (steps):", dict(zip(PARAM_NAMES, tau.round(1))))
    print("effective independent samples:", dict(zip(PARAM_NAMES, ess.round(0))))
    if (n_steps - burn) < 50 * tau.max():
        print("WARNING: chain shorter than ~50 autocorrelation times - increase n_steps for final results")

    step = max(1, int(tau.max()))
    S = chain[burn::step].reshape(-1, chain.shape[2])
    print(f"kept {len(S)} thinned posterior samples")

    lo, hi = bounds
    prior_std = np.r_[(hi - lo) / np.sqrt(12), [np.nan, np.nan]]
    q = np.percentile(S, [5, 50, 95], axis=0)
    summ = pd.DataFrame({"p5": q[0], "median": q[1], "p95": q[2], "post_std/prior_std": S.std(axis=0) / prior_std},
                        index=PARAM_NAMES)
    summ.loc[GROUP_NAMES, ["p5", "median", "p95"]] = np.exp(summ.loc[GROUP_NAMES, ["p5", "median", "p95"]].values)
    print("\nPosterior (sigma rows are the learned model-error levels, as a fraction of the observed final value):")
    print(summ.round(4).to_string())
    summ.to_csv(C.TABLE_DIR / "mcmc_posterior_summary.csv")
    print("\ncorrelations among accepted samples:\n", pd.DataFrame(S[:, :4], columns=C.PARAMS).corr().round(2))

    np.save(C.CACHE_DIR / "posterior_mcmc.npy", S)
    plot_corner(np.column_stack([S[:, :4], np.exp(S[:, 4:])]), bounds, PARAM_NAMES, "30_mcmc_corner.png")
    plot_traces(chain, burn, "31_mcmc_traces.png")
    plot_predictive(proxy, S, like, dates, "32_posterior_predictive.png")
    return S


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    run_all()
