"""STEP 11 - Forecast 2008 -> 2028  (upgrade 6)

How the forecast is built (plain version)
-----------------------------------------
1. Take a history of RATES (barrels per day, one number per quarter).
2. Extend the recent trend: oil rate with a decline curve (exponential or Arps hyperbolic), water rate with
   a saturating curve (rises, then levels off), gas = oil x a constant gas/oil ratio.
3. Turn rates into running totals: cumulative(next) = cumulative(now) + rate x days_in_quarter.
   (This is exactly how the observed cumulative was built - verified to 8 digits.)

Where the "histories" come from
  F1  the single OBSERVED history (no simulator information)
  F2  every plausible parameter set from the MCMC posterior -> proxy predicts its history -> extend the
      trend -> shift/scale so the first forecast quarter equals the observed rate.  The spread between
      the F2 curves is the forecast uncertainty that comes from not knowing the parameters.

Because there are no forecast labels, we TEST the method by hiding the last years we do have:
  * backtest_sims:   hide the last 2-8 years of all 100 simulated cases
  * e2e_backtest:    hide 2004-2008 of the OBSERVED field, history-match on the earlier years only, forecast
                     the hidden years, compare with what really happened (full pipeline, out of sample)

Run alone:  python -m src.forecast
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from . import clean as K
from . import config as C
from . import load_data as L
from . import mcmc as M

WINDOW = 8                                    # quarters of recent history used to fit the trend
OIL_KINDS = ("exp", "arps", "blend")
WAT_KINDS = ("sat_exp", "liq_exp", "blend")


# ---------------------------------------------------------------------------
# Trend models
# ---------------------------------------------------------------------------
def _fit_exp(q, tt):
    b = np.polyfit(tt, np.log(np.maximum(q, 1e-9)), 1)
    return lambda x: np.exp(b[1] + b[0] * x)


def _fit_arps(q, tt):
    """Arps hyperbolic decline q(t) = qi / (1 + b*D*t)^(1/b),  0 < b <= 1 (b->0 is exponential)."""
    t0 = tt[0]
    f = lambda x, qi, D, b: qi / (1 + b * D * (x - t0)) ** (1 / b)
    try:
        p, _ = curve_fit(f, tt, q, p0=[q[0], 0.02, 0.5], bounds=([0, 1e-5, 0.01], [q.max() * 2, 1.0, 1.0]), maxfev=5000)
        return lambda x: f(x, *p)
    except Exception:
        return _fit_exp(q, tt)


def oil_trend(qo, tt, fut, kind):
    if kind == "exp":
        return _fit_exp(qo, tt)(fut)
    if kind == "arps":
        return _fit_arps(qo, tt)(fut)
    return 0.5 * (_fit_exp(qo, tt)(fut) + _fit_arps(qo, tt)(fut))


def _water_sat_exp(qw, tt, fut):
    """q(t) = q_inf - (q_inf - q_last) * exp(-(t - t_last)/tau): water rises then levels off."""
    tl = tt[-1]
    f = lambda x, qinf, tau: qinf - (qinf - qw[-1]) * np.exp(-(x - tl) / tau)
    try:
        p, _ = curve_fit(f, tt, qw, p0=[qw[-1] * 1.2, 10], bounds=([qw[-1] * 0.9, 1.0], [qw[-1] * 5, 400]), maxfev=5000)
        return f(fut, *p)
    except Exception:
        return np.full(len(fut), qw[-1])


def _water_liq_exp(qo, qw, tt, fut, oil_f):
    """Total liquid (oil+water) decays exponentially; water = liquid - oil."""
    return np.maximum(_fit_exp(qo + qw, tt)(fut) - oil_f, 0)


def extrapolate(qo_hist, qw_hist, fut, window=WINDOW, oil_kind="blend", wat_kind="blend"):
    """qo_hist/qw_hist: rates of intervals 0..h-1.  fut: interval indices to predict. -> (oil, water)"""
    h = len(qo_hist)
    tt = np.arange(h - window, h)
    qo, qw = qo_hist[-window:], qw_hist[-window:]
    oil_f = oil_trend(qo, tt, fut, oil_kind)
    if wat_kind == "sat_exp":
        wat_f = _water_sat_exp(qw, tt, fut)
    elif wat_kind == "liq_exp":
        wat_f = _water_liq_exp(qo, qw, tt, fut, oil_f)
    else:
        wat_f = 0.5 * (_water_sat_exp(qw, tt, fut) + _water_liq_exp(qo, qw, tt, fut, oil_trend(qo, tt, fut, "exp")))
    return oil_f, wat_f


# ---------------------------------------------------------------------------
# Back-test 1: hide the last years of every simulated case
# ---------------------------------------------------------------------------
def backtest_sims(cuts=(16, 20, 24, 28, 32, 36), window=WINDOW):
    cube, obs, dates = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")
    sim_r, _, days = K.interval_rates(cube, robs, dates)
    H = len(dates) - 1                                     # index of 2008-01-01
    rows = []
    for h in cuts:
        for method in ["naive"] + [f"{o}+{w}" for o, w in [("exp", "sat_exp"), ("arps", "sat_exp"), ("blend", "blend")]]:
            eo, ew = [], []
            for i in range(C.N_CASES):
                qo, qw = sim_r["oil"][i][:h], sim_r["wat"][i][:h]
                fut = np.arange(h, H)
                if method == "naive":
                    fo, fw = np.full(len(fut), qo[-1]), np.full(len(fut), qw[-1])
                else:
                    ok, wk = method.split("+")
                    fo, fw = extrapolate(qo, qw, fut, window, ok, wk)
                co, cw = np.sum(fo * days[h:H]), np.sum(fw * days[h:H])
                eo.append(abs(co - (cube["oil"][i][H] - cube["oil"][i][h])) / cube["oil"][i][H])
                ew.append(abs(cw - (cube["wat"][i][H] - cube["wat"][i][h])) / cube["wat"][i][H])
            rows.append({"hidden_quarters": H - h, "hidden_years": (H - h) / 4, "method": method,
                         "oil_median_err_%": 100 * np.median(eo), "oil_p90_err_%": 100 * np.percentile(eo, 90),
                         "wat_median_err_%": 100 * np.median(ew), "wat_p90_err_%": 100 * np.percentile(ew, 90)})
    return pd.DataFrame(rows)


def plot_backtest(T):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, v, t in zip(axs, ("oil", "wat"), ("Oil", "Water")):
        for m, g in T.groupby("method"):
            ax.plot(g["hidden_years"], g[f"{v}_median_err_%"], "o-", label=m)
        ax.set_xlabel("years hidden from the method"); ax.set_ylabel("median error of cumulative gained, % of total")
        ax.set_title(f"{t}: error grows with how far you extrapolate"); ax.set_yscale("log")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "50_backtest_sims.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Building a forecast from a history
# ---------------------------------------------------------------------------
def make_dates():
    fd = pd.date_range("2008-01-01", C.FORECAST_DATES[-1], freq="QS")        # 81 dates, incl. the anchor
    return fd, np.diff(fd.values.astype("datetime64[D]")).astype(float)       # 80 interval lengths


def rates_to_cum(cum0, rates, days):
    return cum0 + np.cumsum(rates * days)                                      # value at each later date


def forecast_from_history(qo_h, qw_h, obs_r40, oil_kind, wat_kind, window=WINDOW, anchor=True):
    """History rates r_0..r_{h-1} -> forecast of intervals h..119, then anchored to the OBSERVED r_40.
    Returns oil, water rate arrays for the 80 intervals starting at 2008-01-01."""
    h = len(qo_h)
    fut = np.arange(h, 120)
    fo, fw = extrapolate(qo_h, qw_h, fut, window, oil_kind, wat_kind)
    fo, fw = fo[40 - h:], fw[40 - h:]                                          # intervals 40..119 (80)
    if anchor:
        fo, fw = fo * obs_r40["oil"] / fo[0], fw * obs_r40["wat"] / fw[0]
    return fo, fw


# ---------------------------------------------------------------------------
# Back-test 2: full pipeline, out of sample, on the real observed field
# ---------------------------------------------------------------------------
def e2e_backtest(cut, n_steps=8000, burn=2500, n_draw=300, seed=C.SEED):
    """Pretend we are standing at date index `cut` (e.g. 24 = 2004-01-01): only data up to there is used."""
    U = L.load_uncertainty()
    X, x_obs, dates, lens = M.build_data()
    cube, obs, _ = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")
    _, obs_r, days = K.interval_rates(cube, robs, dates)
    theta = U[C.PARAMS].values
    bounds = (theta.min(0), theta.max(0))
    proxy = M.fit_final_proxy(theta, X, x_obs)
    keep = [cut + 1, cut + 1, cut + 1, cut + 1]                                # cum t_0..t_cut ; rates r_0..r_cut
    like0 = M.Likelihood(proxy, x_obs, lens, bounds, keep=keep)
    P0, best = M.initial_walkers(like0, 64, seed)
    rho = M.residual_lag1(proxy, x_obs, like0.slices, best)
    like = M.Likelihood(proxy, x_obs, lens, bounds, n_eff_factor=(1 - rho) / (1 + rho), keep=keep)
    chain, _, acc = M.aies(like, P0, n_steps, seed=seed)
    S = chain[burn::20].reshape(-1, chain.shape[2])
    S = S[np.random.default_rng(seed).choice(len(S), min(n_draw, len(S)), replace=False)]
    F = proxy.predict_curves(S[:, :4])
    H = 40
    sl = like.slices
    true_cum = {v: obs[v][H] for v in ("oil", "wat")}
    cum_at_cut = {v: obs[v][cut] for v in ("oil", "wat")}
    r_cut = {"oil": obs_r["oil"][cut], "wat": obs_r["wat"][cut]}               # known observed rate at the cut
    scale = {"oil": robs["oil"][-1], "wat": robs["wat"][-1]}
    out = {}

    def cum_to_H(fo, fw):                                                      # rates r_cut.. r_39 -> cum at t_40
        n = H - cut
        return {"oil": cum_at_cut["oil"] + np.sum(fo[:n] * days[cut:H]), "wat": cum_at_cut["wat"] + np.sum(fw[:n] * days[cut:H])}

    # naive
    out["naive: hold last rate"] = [cum_to_H(np.full(H - cut, r_cut["oil"]), np.full(H - cut, r_cut["wat"]))]
    # F1: observed tail
    qo, qw = obs_r["oil"][:cut + 1], obs_r["wat"][:cut + 1]
    fut = np.arange(cut + 1, H)
    res = []
    for ok, wk in [("exp", "sat_exp"), ("arps", "sat_exp"), ("blend", "blend")]:
        fo, fw = extrapolate(qo, qw, fut, WINDOW, ok, wk)
        res.append(cum_to_H(np.r_[r_cut["oil"], fo], np.r_[r_cut["wat"], fw]))
    out["F1: observed tail (blend)"] = [res[2]]
    # F2: posterior samples
    f2 = []
    for k in range(len(S)):
        qo_h = F[k, sl[2]] * scale["oil"]; qw_h = F[k, sl[3]] * scale["wat"]   # proxy-predicted rates r_0..r_cut
        qo_h, qw_h = qo_h[:cut], qw_h[:cut]                                    # history through r_{cut-1}
        fut2 = np.arange(cut, H)
        fo, fw = extrapolate(qo_h, qw_h, fut2, WINDOW, "blend", "blend")
        fo, fw = fo * r_cut["oil"] / fo[0], fw * r_cut["wat"] / fw[0]           # anchor to observed r_cut
        f2.append(cum_to_H(fo, fw))
    out["F2: posterior ensemble (median)"] = f2
    f1m = res[2]
    out["MIX: 50/50 of F1 and F2"] = [{"oil": 0.5 * (f1m["oil"] + d["oil"]), "wat": 0.5 * (f1m["wat"] + d["wat"])} for d in f2]
    # oracle: what the proxy itself says for 2008 (it has seen the full curves of other cases)
    out["oracle: proxy direct (not available in the real forecast)"] = [
        {"oil": F[k, sl[0].start + H] * obs["oil"][-1],
         "wat": F[k, sl[1].start + H] * obs["wat"][-1]} for k in range(len(S))]

    rows = []
    for name, lst in out.items():
        for v in ("oil", "wat"):
            vals = np.array([d[v] for d in lst])
            med = np.median(vals)
            rows.append({"cut": f"{dates[cut].year}", "hidden_years": (H - cut) / 4, "method": name, "variable": v,
                         "median_err_%": 100 * (med - true_cum[v]) / true_cum[v],
                         "band5-95_%": (f"[{100 * (np.percentile(vals, 5) - true_cum[v]) / true_cum[v]:+.2f}, "
                                        f"{100 * (np.percentile(vals, 95) - true_cum[v]) / true_cum[v]:+.2f}]") if len(vals) > 1 else "-",
                         "truth_inside_band": bool(np.percentile(vals, 5) <= true_cum[v] <= np.percentile(vals, 95)) if len(vals) > 1 else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The real forecast
# ---------------------------------------------------------------------------
def real_forecast(n_draw=800, seed=C.SEED):
    U = L.load_uncertainty()
    X, x_obs, dates, lens = M.build_data()
    cube, obs, _ = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")
    _, obs_r, _ = K.interval_rates(cube, robs, dates)
    fdates, fdays = make_dates()
    theta = U[C.PARAMS].values
    proxy = M.fit_final_proxy(theta, X, x_obs)
    like = M.Likelihood(proxy, x_obs, lens, (theta.min(0), theta.max(0)))
    S = np.load(C.CACHE_DIR / "posterior_mcmc.npy")
    rng = np.random.default_rng(seed)
    S = S[rng.choice(len(S), min(n_draw, len(S)), replace=False)]
    F = proxy.predict_curves(S[:, :4])
    r40 = {"oil": robs["oil"][-1], "wat": robs["wat"][-1], "gas": robs["gas"][-1]}
    goR = r40["gas"] / r40["oil"]
    scale = {"oil": robs["oil"][-1], "wat": robs["wat"][-1]}
    qo_all, qw_all = F[:, like.slices[2]] * scale["oil"], F[:, like.slices[3]] * scale["wat"]   # r_0..r_39

    combos = {"blend": ("blend", "blend"), "low-oil (exp)": ("exp", "sat_exp"), "high-oil (Arps)": ("arps", "sat_exp"),
              "alt-water (liq_exp)": ("exp", "liq_exp")}
    ens = {}
    for name, (ok, wk) in combos.items():
        rates = np.array([forecast_from_history(qo_all[k], qw_all[k], r40, ok, wk) for k in range(len(S))])   # (n,2,80)
        ens[name] = rates
    # observed-only forecast
    f1 = {}
    for name, (ok, wk) in combos.items():
        fo, fw = extrapolate(np.r_[obs_r["oil"], r40["oil"]], np.r_[obs_r["wat"], r40["wat"]], np.arange(41, 120), WINDOW, ok, wk)
        f1[name] = (np.r_[r40["oil"], fo], np.r_[r40["wat"], fw])
    f1b = np.array(f1["blend"])                                   # (2, 80)
    mix = 0.5 * (ens["blend"] + f1b[None, :, :])                  # per-sample 50/50 average of the two families
    return dict(S=S, F=F, ens=ens, f1=f1, mix=mix, fdates=fdates, fdays=fdays, r40=r40, gor=goR,
                cum0={v: obs[v][-1] for v in C.VARS}, obs=obs, robs=robs, dates=dates, obs_r=obs_r)


def to_cum(rates_oil, rates_wat, R):
    """Rates for the 80 intervals -> cumulative gas/oil/water at the 80 forecast dates (2008-04-01 .. 2028-01-01)."""
    co = rates_to_cum(R["cum0"]["oil"], rates_oil, R["fdays"])
    cw = rates_to_cum(R["cum0"]["wat"], rates_wat, R["fdays"])
    cg = rates_to_cum(R["cum0"]["gas"], rates_oil * R["gor"], R["fdays"])
    return cg, co, cw


def plot_forecast(R):
    fd = R["fdates"][1:]
    base = R["ens"]["blend"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 7))
    for j, (v, lab) in enumerate([("oil", "Oil"), ("wat", "Water"), ("gas", "Gas")]):
        ax_c, ax_r = axs[0, j], axs[1, j]
        # cumulative
        cums = np.array([to_cum(base[k, 0], base[k, 1], R)[{"gas": 0, "oil": 1, "wat": 2}[v]] for k in range(len(base))])
        lo, md, hi = np.percentile(cums, [5, 50, 95], axis=0)
        ax_c.plot(R["dates"], R["obs"][v], "k-", lw=2, label="observed history")
        ax_c.fill_between(fd, lo, hi, color="#4c78a8", alpha=0.3, label="F2 posterior 5-95%")
        ax_c.plot(fd, md, color="#4c78a8", label="F2 median")
        fo, fw = R["f1"]["blend"]
        f1c = to_cum(fo, fw, R)[{"gas": 0, "oil": 1, "wat": 2}[v]]
        ax_c.plot(fd, f1c, "r--", label="F1 observed-tail only")
        mixc = np.array([to_cum(R["mix"][k, 0], R["mix"][k, 1], R)[{"gas": 0, "oil": 1, "wat": 2}[v]] for k in range(len(base))])
        ax_c.plot(fd, np.median(mixc, axis=0), color="#2ca02c", lw=2, label="MIX (used in template)")
        ax_c.set_title(f"{lab} cumulative"); ax_c.axvline(R["dates"][-1], color="grey", ls=":")
        # rate
        idx = {"oil": 0, "wat": 1}.get(v)
        if v == "gas":
            rates = base[:, 0, :] * R["gor"]; obs_rate = R["obs_r"]["oil"] * R["gor"]; f1r = fo * R["gor"]
        else:
            rates = base[:, idx, :]; obs_rate = R["obs_r"][v]; f1r = (fo, fw)[idx]
        lo, md, hi = np.percentile(rates, [5, 50, 95], axis=0)
        ax_r.plot(R["dates"][:-1], obs_rate, "k-", lw=2)
        ax_r.fill_between(fd[:], lo, hi, color="#4c78a8", alpha=0.3)
        ax_r.plot(fd, md, color="#4c78a8"); ax_r.plot(fd, f1r, "r--")
        mr = R["mix"][:, 0, :] * (R["gor"] if v == "gas" else 1) if v != "wat" else R["mix"][:, 1, :]
        ax_r.plot(fd, np.median(mr, axis=0), color="#2ca02c", lw=2)
        ax_r.set_title(f"{lab} rate")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "51_forecast.png", dpi=130); plt.close(fig)


def plot_model_form(R):
    fd = R["fdates"][1:]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (v, ix, lab) in zip(axs, [("oil", 1, "oil cumulative"), ("wat", 2, "water cumulative")]):
        for name, rates in R["ens"].items():
            c = np.array([to_cum(rates[k, 0], rates[k, 1], R)[ix] for k in range(0, len(rates), 8)])
            ax.plot(fd, np.median(c, axis=0), label=name)
        ax.set_title(f"Median {lab} under different trend-model choices")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "52_model_form_spread.png", dpi=130); plt.close(fig)


def pick_template_cases(R, k=10, seed=C.SEED):
    """k diverse parameter sets from the posterior (k-means), ranked by how well each fits."""
    X, x_obs, dates, lens = M.build_data()
    U = L.load_uncertainty(); theta = U[C.PARAMS].values
    proxy = M.fit_final_proxy(theta, X, x_obs)
    like = M.Likelihood(proxy, x_obs, lens, (theta.min(0), theta.max(0)))
    S = R["S"]
    ll = like(S)
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(MinMaxScaler().fit_transform(S[:, :4]))
    idx = [int(np.where(lab == c)[0][np.argmax(ll[lab == c])]) for c in range(k)]
    idx = sorted(idx, key=lambda i: -ll[i])
    return idx


def envelope(R):
    """Honest uncertainty: pool every curve from every family (F1 and F2, all trend-model choices, MIX)
    and take 5/50/95% across ALL of them - includes model-choice uncertainty, not only parameters."""
    pools = []
    for rates in list(R["ens"].values()) + [R["mix"]]:
        pools += [np.array(to_cum(rates[k, 0], rates[k, 1], R)) for k in range(0, len(rates), 4)]
    for fo, fw in R["f1"].values():
        pools += [np.array(to_cum(fo, fw, R))] * 25                    # F1 has one curve per variant; weight it like a family
    return np.array(pools)                                              # (n, 3, 80)


def write_outputs(R, idx):
    base = R["mix"]
    fd = R["fdates"][1:]
    # 1) filled template (the file path is data/09_*.xlsx - the empty template the organisers gave)
    from .load_data import find_file
    wb = load_workbook(find_file("09"))
    params = []
    for n, i in enumerate(idx, start=1):
        cg, co, cw = to_cum(base[i, 0], base[i, 1], R)
        ws = wb[f"Case {n}"]
        assert ws.max_row - 1 == len(fd), f"template sheet Case {n} has {ws.max_row - 1} rows, expected {len(fd)}"
        for r in range(len(fd)):
            ws.cell(row=r + 2, column=2, value=float(cg[r]))
            ws.cell(row=r + 2, column=3, value=float(co[r]))
            ws.cell(row=r + 2, column=4, value=float(cw[r]))
        params.append({"template_case": n, **dict(zip(C.PARAMS, R["S"][i, :4]))})
    C.ensure_dirs()
    wb.save(C.OUT_DIR / "09_Template_Deliverable_filled.xlsx")
    pd.DataFrame(params).to_csv(C.TABLE_DIR / "template_case_parameters.csv", index=False)
    # 2) long CSV for Power BI: all variants + band
    rows = []
    for n, i in enumerate(idx, start=1):
        for name, rates in {**R["ens"], "MIX (template)": R["mix"]}.items():
            cg, co, cw = to_cum(rates[i, 0], rates[i, 1], R)
            for r in range(len(fd)):
                rows.append({"case": n, "variant": name, "date": fd[r], "gas_cum_MSCF": cg[r], "oil_cum_STB": co[r],
                             "water_cum_STB": cw[r], "oil_rate_STBd": rates[i, 0, r], "water_rate_STBd": rates[i, 1, r]})
    pd.DataFrame(rows).to_csv(C.TABLE_DIR / "forecast_long_for_powerbi.csv", index=False)
    cums = envelope(R)                                                                # (n,3,80)
    mixc = np.array([to_cum(base[k, 0], base[k, 1], R) for k in range(len(base))])
    band = {"date": fd}
    for j, v in enumerate(["gas_cum_MSCF", "oil_cum_STB", "water_cum_STB"]):
        lo, md, hi = np.percentile(cums[:, j, :], [5, 50, 95], axis=0)
        band.update({f"{v}_env_p5": lo, f"{v}_env_p50": md, f"{v}_env_p95": hi, f"{v}_mix_median": np.median(mixc[:, j, :], axis=0)})
    pd.DataFrame(band).to_csv(C.TABLE_DIR / "forecast_band.csv", index=False)
    return pd.DataFrame(params)


def check_outputs(path):
    """Sanity checks on the filled workbook."""
    wb = load_workbook(path, data_only=True)
    probs = []
    for ws in wb.worksheets:
        a = np.array([[c for c in row[1:4]] for row in ws.iter_rows(min_row=2, values_only=True)], dtype=float)
        if np.isnan(a).any(): probs.append(f"{ws.title}: NaN/empty cells")
        if (np.diff(a, axis=0) < 0).any(): probs.append(f"{ws.title}: cumulative decreases")
        if len(a) != 80: probs.append(f"{ws.title}: {len(a)} rows")
    return probs


# ---------------------------------------------------------------------------
def run_all():
    C.ensure_dirs()
    pd.set_option("display.width", 220)
    T = backtest_sims()
    T.to_csv(C.TABLE_DIR / "backtest_sims.csv", index=False)
    plot_backtest(T)
    print("BACK-TEST 1 (all 100 simulated cases): median / 90th-percentile error of the cumulative gained over the hidden years")
    print(T.round(2).to_string(index=False), "\n")

    E2 = pd.concat([e2e_backtest(c) for c in (24, 28, 32)], ignore_index=True)
    E2.to_csv(C.TABLE_DIR / "backtest_observed_e2e.csv", index=False)
    print("BACK-TEST 2 (the real observed field, full pipeline, only earlier years used). Error = forecast - truth, % of truth:")
    print(E2.round(2).to_string(index=False), "\n")

    R = real_forecast()
    plot_forecast(R); plot_model_form(R)
    idx = pick_template_cases(R)
    P = write_outputs(R, idx)
    print("Template cases (diverse posterior samples, best fit first):\n", P.round(4).to_string(index=False))
    probs = check_outputs(C.OUT_DIR / "09_Template_Deliverable_filled.xlsx")
    print("\nfilled-template checks:", "OK (80 rows, no gaps, cumulative never decreases)" if not probs else probs)
    cums = envelope(R)
    mixc = np.array([to_cum(R["mix"][k, 0], R["mix"][k, 1], R) for k in range(len(R["mix"]))])
    print("\n2028-01-01 cumulative:   template (MIX median)   |   envelope over ALL model choices 5 / 50 / 95 %")
    for j, nm in enumerate(["gas MSCF", "oil STB", "water STB"]):
        print(f"  {nm:10s} {np.median(mixc[:, j, -1]):.4g}   |   {np.percentile(cums[:, j, -1], 5):.4g} / {np.percentile(cums[:, j, -1], 50):.4g} / {np.percentile(cums[:, j, -1], 95):.4g}")
    return R


if __name__ == "__main__":
    run_all()
