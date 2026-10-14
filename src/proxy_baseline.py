"""STEP 5 - Baseline proxy + history match + diverse top-10 (the thing to BEAT).

  forward proxy : 4 parameters --GP--> PCA scores --decode--> gas/oil/water curves (1998-2008)
  validation    : fit on train (1-70), score on validation (71-85). Test (86-100) is NOT touched.
  history match : ABC rejection - draw many parameter sets from the prior box, push them through
                  the proxy, keep the ones closest to the observed history  (a cheap posterior)
  top-10        : k-means on the accepted samples -> 10 DISTINCT parameter sets

Run alone to test:   python -m src.proxy_baseline
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

from . import config as C
from . import clean as K
from . import embed as E
from . import load_data as L


# ---------------------------------------------------------------------------
class PCAGPProxy:
    """params (4) -> curves.  One GP per principal component."""

    def __init__(self, n_pc=4, n_restarts=3):
        self.n_pc, self.n_restarts = n_pc, n_restarts

    def fit(self, theta, X, pca):
        self.pca = pca
        self.sx = MinMaxScaler().fit(theta)
        Z = pca.transform(X)[:, : self.n_pc]
        t = self.sx.transform(theta)
        self.gps = []
        for j in range(self.n_pc):
            k = ConstantKernel(1.0) * Matern(np.ones(theta.shape[1]), nu=2.5) + WhiteKernel(1e-3)
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=self.n_restarts, random_state=C.SEED)
            self.gps.append(gp.fit(t, Z[:, j]))
        return self

    def predict_scores(self, theta):
        t = self.sx.transform(theta)
        return np.column_stack([gp.predict(t) for gp in self.gps])

    def predict_std(self, theta):
        """Total predictive std (sum of the GP variances over the kept components, then sqrt):
        'how unsure is the proxy about this whole curve set?'  Used by active_design."""
        t = self.sx.transform(theta)
        var = sum(gp.predict(t, return_std=True)[1] ** 2 for gp in self.gps)
        return np.sqrt(var)

    def predict_curves(self, theta):
        Z = np.zeros((len(theta), self.pca.n_components_))
        Z[:, : self.n_pc] = self.predict_scores(theta)
        return self.pca.inverse_transform(Z)          # scaled features (n, n_vars*n_times)


# ---------------------------------------------------------------------------
def split_features(F, n_times, vars_=None):
    vars_ = vars_ or C.VARS
    return {v: F[:, i * n_times:(i + 1) * n_times] for i, v in enumerate(vars_)}


def mismatch(F, x_obs, n_times, weights=None):
    """Weighted normalised RMSE of predicted features F vs observed x_obs (scaled space)."""
    weights = weights or C.MISMATCH_WEIGHTS
    Fv, Ov = split_features(F, n_times), split_features(x_obs, n_times)
    tot = sum(weights[v] * np.sqrt(((Fv[v] - Ov[v]) ** 2).mean(axis=1)) for v in C.VARS)
    return tot / sum(weights.values())


def evaluate_proxy(proxy, theta, X, n_times):
    """Error on held-out cases, in physically meaningful terms."""
    P = split_features(proxy.predict_curves(theta), n_times)
    T = split_features(X, n_times)
    rows = {}
    for v in C.VARS:
        final_err = np.abs(P[v][:, -1] - T[v][:, -1]) / np.abs(T[v][:, -1])
        nrmse = np.sqrt(((P[v] - T[v]) ** 2).mean(axis=1)) / np.abs(T[v][:, -1])
        rows[v] = {"final_value_MAPE_%": 100 * final_err.mean(), "final_value_max_err_%": 100 * final_err.max(),
                   "curve_nRMSE_%": 100 * nrmse.mean()}
    return pd.DataFrame(rows).T


def abc_history_match(proxy, x_obs, n_times, bounds, n_samples=2 ** 17, keep_frac=0.003):
    """Quasi-random prior draws -> proxy -> keep the best `keep_frac`."""
    lo, hi = bounds
    s = qmc.Sobol(d=len(lo), scramble=True, seed=C.SEED).random(n_samples)
    theta = lo + s * (hi - lo)
    err = mismatch(proxy.predict_curves(theta), x_obs, n_times)
    keep = np.argsort(err)[: int(n_samples * keep_frac)]
    return theta[keep], err[keep], theta, err


def posterior_summary(theta_post, bounds):
    """How much did history narrow each parameter?  ratio ~1 => history says nothing."""
    lo, hi = bounds
    prior_std = (hi - lo) / np.sqrt(12)           # std of a uniform distribution
    q = np.percentile(theta_post, [5, 50, 95], axis=0)
    return pd.DataFrame({"prior_range_lo": lo, "prior_range_hi": hi, "post_p5": q[0], "post_median": q[1],
                         "post_p95": q[2], "post_std/prior_std": theta_post.std(axis=0) / prior_std},
                        index=C.PARAMS)


def pick_top10(theta_post, err_post, k=10):
    """k-means in scaled parameter space; take the best-matching member of each cluster."""
    sc = MinMaxScaler().fit(theta_post)
    lab = KMeans(k, n_init=10, random_state=C.SEED).fit_predict(sc.transform(theta_post))
    idx = [int(np.where(lab == c)[0][np.argmin(err_post[lab == c])]) for c in range(k)]
    out = pd.DataFrame(theta_post[idx], columns=C.PARAMS)
    out["mismatch"] = err_post[idx]
    return out.sort_values("mismatch").reset_index(drop=True)


def plot_posterior(theta_post, bounds):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    lo, hi = bounds
    for i, (ax, p) in enumerate(zip(axs, C.PARAMS)):
        ax.hist(theta_post[:, i], bins=25, density=True, color="#4c78a8")
        ax.axhline(1 / (hi[i] - lo[i]), color="r", ls="--", label="prior (uniform)")
        ax.set_xlim(lo[i], hi[i]); ax.set_title(p)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "20_posterior_vs_prior.png", dpi=130); plt.close(fig)


def plot_match(proxy, top10, x_obs, dates, n_times):
    P = split_features(proxy.predict_curves(top10[C.PARAMS].values), n_times)
    O = split_features(x_obs, n_times)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, v in zip(axs, C.VARS):
        for r in P[v]:
            ax.plot(dates, r, color="#4c78a8", alpha=0.6)
        ax.plot(dates, O[v][0], "k.", label="observed"); ax.set_title(f"{v} (scaled by obs final)")
    axs[0].legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "21_top10_vs_observed.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def run_all():
    C.ensure_dirs()
    U = L.load_uncertainty()
    cube, obs, dates = K.load_cache("cum")
    X, x_obs = E.build_features(cube, obs)
    n_t = len(dates)
    theta = U[C.PARAMS].values
    tr, va = np.array(C.TRAIN_CASES) - 1, np.array(C.VAL_CASES) - 1

    # 1) validate on 71-85 (PCA + GP fitted on train only)
    pca_tr = E.fit_pca(X[tr], x_obs, 6)[0]
    proxy_tr = PCAGPProxy(n_pc=4).fit(theta[tr], X[tr], pca_tr)
    print("Validation error (cases 71-85), proxy fitted on cases 1-70:")
    ev = evaluate_proxy(proxy_tr, theta[va], X[va], n_t)
    print(ev.round(3), "\n")
    ev.to_csv(C.TABLE_DIR / "proxy_validation.csv")

    # per-PC: which parameters matter? (GP length-scales: small = important)
    print("GP length-scales per PC (scaled inputs; SMALL = parameter matters):")
    ls = pd.DataFrame([gp.kernel_.k1.k2.length_scale for gp in proxy_tr.gps], columns=C.PARAMS,
                      index=[f"PC{i + 1}" for i in range(4)])
    print(ls.round(2), "\n")
    ls.to_csv(C.TABLE_DIR / "gp_lengthscales.csv")

    # 2) refit on train+val, history-match
    trva = np.concatenate([tr, va])
    pca = E.fit_pca(X[trva], x_obs, 6)[0]
    proxy = PCAGPProxy(n_pc=4).fit(theta[trva], X[trva], pca)
    bounds = (theta.min(axis=0), theta.max(axis=0))
    post, err_post, _, err_all = abc_history_match(proxy, x_obs, n_t, bounds)
    print(f"ABC: kept {len(post)} of {len(err_all)} prior draws; "
          f"best mismatch={err_post.min():.5f}, worst kept={err_post.max():.5f}\n"
          "(if best ~= worst the kept samples sit on the noise floor: ranking inside them is meaningless)\n")
    summ = posterior_summary(post, bounds)
    print("Posterior vs prior (std ratio ~1 => history cannot constrain it):")
    print(summ.round(3), "\n")
    summ.to_csv(C.TABLE_DIR / "posterior_summary.csv")

    top10 = pick_top10(post, err_post, 10)
    print("10 diverse best-matching parameter sets (k-means on posterior):")
    print(top10.round(5))
    top10.to_csv(C.TABLE_DIR / "top10_param_sets.csv", index=False)
    plot_posterior(post, bounds); plot_match(proxy, top10, x_obs, dates, n_t)
    np.save(C.CACHE_DIR / "posterior_samples.npy", post)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    run_all()
