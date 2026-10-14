"""STEP 6 - Deeper EDA: "what does each parameter actually DO?"

Key idea: a parameter's usefulness is not its RANGE but its EFFECT PER UNIT OF RANGE, and a
dominant parameter (porosity) hides the smaller ones until you remove its trend.

Figures:  07 range vs effect | 08 residual-after-porosity | 09 feature heatmaps |
          10b in-band rate curves | 11b Sobol sensitivity
Run alone:   python -m src.eda_advanced
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import clean as K
from . import embed as E
from . import load_data as L
from . import proxy_baseline as PB

TRIO = ["fault_tran", "perm_mult", "aq_pv"]          # the three "non-porosity" parameters


def _save(fig, name):
    C.ensure_dirs()
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Ranges vs effects
# ---------------------------------------------------------------------------
def ranges_table(U):
    P = U[C.PARAMS]
    return pd.DataFrame({"min": P.min(), "max": P.max(), "max/min": P.max() / P.min(),
                         "CV(std/mean)": P.std() / P.mean()})


def swing_table(cube, U):
    """Linear-model 'swing': how many std-devs does the FINAL output move when one parameter goes
    from its min to its max (others fixed)?  Inputs are min-max normalised so it is per full range."""
    Xn = ((U[C.PARAMS] - U[C.PARAMS].min()) / (U[C.PARAMS].max() - U[C.PARAMS].min())).values
    A = np.column_stack([np.ones(len(U)), Xn])
    out = {}
    for v in ("oil", "wat"):
        y = cube[v][:, -1]
        y = (y - y.mean()) / y.std()
        out[f"final_{v}"] = dict(zip(C.PARAMS, np.abs(np.linalg.lstsq(A, y, rcond=None)[0][1:])))
    return pd.DataFrame(out)


def plot_range_vs_effect(cube, U):
    R, S = ranges_table(U), swing_table(cube, U)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    axs[0].barh(R.index, R["max/min"], color="#999"); axs[0].set_xscale("log")
    axs[0].set_title("INPUT range: max/min (log axis)")
    for i, v in enumerate(R["max/min"]):
        axs[0].text(v, i, f" {v:.1f}x", va="center")
    S.plot.barh(ax=axs[1], color=["#4c78a8", "#54a24b"])
    axs[1].set_title("OUTPUT effect: swing of final cumulative\nacross the full range (std units)")
    _save(fig, "07_range_vs_effect")
    return R, S


# ---------------------------------------------------------------------------
# 2. Remove the porosity trend, look at what is left
# ---------------------------------------------------------------------------
def porosity_residual(y, poro, deg=4):
    """y minus a smooth polynomial function of porosity."""
    return y - np.polyval(np.polyfit(poro, y, deg), poro)


def engineer_features(cube, rcube, dates):
    """Physically meaningful summary numbers per case (rates carry TIMING information that the
    integrated cumulative curves smooth away)."""
    yrs = 4  # decline measured over the last 8 quarters = 2 years
    f = pd.DataFrame(index=range(1, C.N_CASES + 1))
    f["oil_cum"] = cube["oil"][:, -1]
    f["wat_cum"] = cube["wat"][:, -1]
    f["oil_rate_2008"] = rcube["oil"][:, -1]
    f["wat_rate_2008"] = rcube["wat"][:, -1]
    f["water_cut_2008"] = f.wat_rate_2008 / (f.wat_rate_2008 + f.oil_rate_2008)
    f["oil_decline_per_yr"] = np.log(rcube["oil"][:, -1] / rcube["oil"][:, -1 - 2 * yrs]) / 2
    f["wat_rate_growth_per_yr"] = (rcube["wat"][:, -1] - rcube["wat"][:, -1 - 2 * yrs]) / 2
    yrfrac = (dates - dates[0]).days.values / 365.25
    bt = []
    for w in rcube["wat"]:
        idx = np.argmax(w > 100)                     # first quarter with >100 STB/d water
        bt.append(yrfrac[idx] if w.max() > 100 else np.nan)
    f["water_breakthrough_yr"] = bt
    return f


def residual_scatter(F, U):
    """Final oil/water RATE after removing porosity, against the three other parameters."""
    poro = U.poro_mult.values
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    rows = []
    for r, (col, lab) in enumerate([("oil_rate_2008", "oil rate 2008"), ("wat_rate_2008", "water rate 2008")]):
        res = porosity_residual(F[col].values, poro)
        for c, p in enumerate(TRIO):
            rho = pd.Series(res).corr(pd.Series(U[p].values), method="spearman")
            axs[r, c].scatter(U[p], res, s=12, c=poro, cmap="viridis")
            axs[r, c].set_title(f"{lab}: residual vs {p}  (rho={rho:.2f})", fontsize=9)
            rows.append({"feature": col, "param": p, "spearman_resid": rho})
    _save(fig, "08_residual_after_porosity")
    return pd.DataFrame(rows).pivot(index="feature", columns="param", values="spearman_resid")


def feature_heatmaps(F, U):
    """Spearman(param, feature): raw  vs  after removing the porosity trend."""
    poro = U.poro_mult.values
    raw = pd.DataFrame({c: {p: F[c].corr(U[p], method="spearman") for p in C.PARAMS} for c in F.columns}).T
    resid = pd.DataFrame({c: {p: pd.Series(porosity_residual(F[c].values, poro)).corr(
        pd.Series(U[p].values), method="spearman") for p in C.PARAMS} for c in F.columns}).T
    resid["poro_mult"] = np.nan                        # by construction
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, M, t in zip(axs, (raw, resid), ("RAW", "porosity trend removed")):
        im = ax.imshow(M.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), C.PARAMS, rotation=20); ax.set_yticks(range(len(M)), M.index)
        for i in range(M.shape[0]):
            for j in range(4):
                if not np.isnan(M.values[i, j]):
                    ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Spearman: {t}")
    fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    _save(fig, "09_feature_heatmaps")
    return raw, resid


def inband_rate_curves(cube, robs, dates, U, lo=1.30, hi=1.42):
    """Among cases whose porosity is close to the history-matched value, what separates them?
    Oil rate -> permeability, water rate -> aquifer size.
    Rates use the corrected definition (average over the NEXT quarter, = change in cumulative / days),
    the same meaning the observed rates have - see clean.interval_rates."""
    sim_r, obs_r, _ = K.interval_rates(cube, robs, dates)
    rcube = {v: sim_r[v] for v in ("oil", "wat")}
    robs = {v: obs_r[v] for v in ("oil", "wat")}
    dates = dates[:-1]
    m = ((U.poro_mult > lo) & (U.poro_mult < hi)).values
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, var, p, lab in [(axs[0], "oil", "perm_mult", "oil rate [STB/d]"),
                            (axs[1], "wat", "aq_pv", "water rate [STB/d]")]:
        norm = plt.Normalize(U.loc[m, p].min(), U.loc[m, p].max())
        for i in np.where(m)[0]:
            ax.plot(dates, rcube[var][i], color=plt.cm.viridis(norm(U[p].iloc[i])), lw=1.2)
        ax.plot(dates, robs[var], "k.", ms=4, label="observed")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label=p)
        ax.set_title(f"{lab}: {m.sum()} cases with porosity in [{lo},{hi}]")
    axs[0].legend()
    _save(fig, "10b_inband_rate_curves")


# ---------------------------------------------------------------------------
# 3. Variance-based (Sobol) sensitivity through the proxy
# ---------------------------------------------------------------------------
def sobol_indices(fn, bounds, n=8192, seed=0):
    """fn: theta (n,4) -> outputs (n,k). Jansen estimators: S1 (alone), ST (incl. interactions)."""
    lo, hi = bounds
    rng = np.random.default_rng(seed)
    A, B = lo + rng.random((n, 4)) * (hi - lo), lo + rng.random((n, 4)) * (hi - lo)
    fa, fb = fn(A), fn(B)
    var = np.var(np.vstack([fa, fb]), axis=0)
    S1, ST = np.zeros((4, fa.shape[1])), np.zeros((4, fa.shape[1]))
    for i in range(4):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fab = fn(ABi)
        S1[i] = 1 - 0.5 * np.mean((fb - fab) ** 2, axis=0) / var
        ST[i] = 0.5 * np.mean((fa - fab) ** 2, axis=0) / var
    return S1, ST


def plot_sobol(proxy, n_t, bounds):
    names = ["final oil cum", "final water cum", "PC1", "PC2", "PC3", "PC4"]

    def fn(th):
        F = PB.split_features(proxy.predict_curves(th), n_t)
        return np.column_stack([F["oil"][:, -1], F["wat"][:, -1], proxy.predict_scores(th)])

    S1, ST = sobol_indices(fn, bounds)
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for ax, M, t in zip(axs, (S1, ST), ("S1: effect of the parameter ALONE", "ST: incl. interactions")):
        bottom = np.zeros(len(names))
        for i, p in enumerate(C.PARAMS):
            ax.bar(names, np.clip(M[i], 0, None), bottom=bottom, label=p)
            bottom += np.clip(M[i], 0, None)
        ax.set_title("Share of output variance - " + t); ax.tick_params(axis="x", rotation=20)
    axs[0].legend(fontsize=8)
    _save(fig, "11b_sobol")
    return pd.DataFrame(ST, index=C.PARAMS, columns=names)


# ---------------------------------------------------------------------------
def run_all():
    C.ensure_dirs()
    U = L.load_uncertainty()
    cube, obs, dates = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")

    R, S = plot_range_vs_effect(cube, U)
    print("Parameter ranges:\n", R.round(3), "\n")
    print("Output swing across full range (std units):\n", S.round(3), "\n")

    F = engineer_features(cube, rcube, dates)
    F.to_csv(C.TABLE_DIR / "engineered_features.csv")
    print("Spearman of residual (porosity removed) vs parameter:\n", residual_scatter(F, U).round(2), "\n")
    raw, resid = feature_heatmaps(F, U)
    resid.to_csv(C.TABLE_DIR / "feature_corr_after_porosity.csv")
    inband_rate_curves(cube, robs, dates, U)

    X, x_obs = E.build_features(cube, obs)
    pca = E.fit_pca(X, x_obs, 6)[0]
    theta = U[C.PARAMS].values
    proxy = PB.PCAGPProxy(4).fit(theta, X, pca)
    ST = plot_sobol(proxy, len(dates), (theta.min(0), theta.max(0)))
    print("Sobol total-effect indices (rows=param):\n", ST.round(3))
    ST.to_csv(C.TABLE_DIR / "sobol_total.csv")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    run_all()
