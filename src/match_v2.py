"""STEP 7 - Simple ABC comparison (kept as a sanity check next to the MCMC of step 8).

Questions:  does the 'posterior' depend on (a) the arbitrary tolerance and (b) using rates as well as
cumulatives?   NOTE: rates here use the CORRECTED definition (average over the next quarter), see
clean.interval_rates.  An earlier version compared instantaneous simulator rates with the observed
rates and over-stated the rate misfit (a one-quarter shift made the 1998 ramp-up look like a huge error).

Run alone:  python -m src.match_v2
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import config as C
from . import load_data as L
from . import mcmc as M


def block_error(F, x_obs, slices, use):
    """Weighted mean of per-block normalised RMSE (water blocks count double)."""
    tot, w = 0, 0
    for b in use:
        wt = 2.0 if M.BLOCK_NAMES[b].endswith("wat") else 1.0
        tot = tot + wt * np.sqrt(((F[:, slices[b]] - x_obs[:, slices[b]]) ** 2).mean(axis=1))
        w += wt
    return tot / w


def run_all(n=2 ** 17, keep_fracs=(0.001, 0.003, 0.01, 0.03)):
    C.ensure_dirs()
    U = L.load_uncertainty()
    X, x_obs, dates, lens = M.build_data()
    theta = U[C.PARAMS].values
    bounds = (theta.min(0), theta.max(0))
    proxy = M.fit_final_proxy(theta, X, x_obs)
    like = M.Likelihood(proxy, x_obs, lens, bounds)
    th = bounds[0] + qmc.Sobol(4, scramble=True, seed=C.SEED).random(n) * (bounds[1] - bounds[0])
    F = proxy.predict_curves(th)
    prior_std = (bounds[1] - bounds[0]) / np.sqrt(12)
    rows = []
    for label, use in {"A: cumulative only": [0, 1], "B: cumulative + rates": [0, 1, 2, 3]}.items():
        err = block_error(F, x_obs, like.slices, use)
        order = np.argsort(err)
        for kf in keep_fracs:
            post = th[order[: int(n * kf)]]
            c = np.corrcoef(post.T)
            rows.append({"variant": label, "keep_frac": kf, "n_kept": len(post), "best_err": err[order[0]],
                         **{f"ratio_{p}": post[:, i].std() / prior_std[i] for i, p in enumerate(C.PARAMS)},
                         "corr(poro,perm)": c[1, 2], "corr(poro,aq)": c[1, 3]})
    T = pd.DataFrame(rows)
    T.to_csv(C.TABLE_DIR / "match_variants.csv", index=False)
    print("Posterior std / prior std (1.0 = history says nothing about that parameter):")
    print(T.round(3).to_string(index=False))
    return T


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    run_all()
