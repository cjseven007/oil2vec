"""STEP 2 - Cleaning + putting every case on ONE common time grid.

Raw sheets have ~11,000 rows because each case has its OWN irregular simulator
time-steps (~165-200 points per case) and the sheet stacks all dates together. We:
  1. drop empty rows, sort, drop duplicate dates, force numeric
  2. run sanity checks (cumulative must not decrease, no negatives, no NaN)
  3. interpolate every case onto the quarterly OBSERVED dates (41 points,
     1998-01-01 .. 2008-01-01) so cases and observed data line up row by row

Output  = `cube`: dict  var -> ndarray (n_cases, n_times)   (cumulative curves)
          `dates`: DatetimeIndex of the common grid
          `obs`  : dict  var -> ndarray (n_times,)           (observed cumulative)

Run alone to test:   python -m src.clean
"""
import numpy as np
import pandas as pd

from . import config as C
from . import load_data as L


def tidy_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty/duplicate rows, sort by date, make numeric. Index = date."""
    case_cols = [c for c in df.columns if isinstance(c, int)] + ["observed"]
    df = df.dropna(subset=["date"]).copy()
    for c in case_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", subset=case_cols)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    return df.set_index("date")


def sanity_report(sheets: dict) -> pd.DataFrame:
    """One row per sheet with problem counts. Print it and eyeball it.

    NOTE: every case has its OWN simulator time-steps (the sheet holds the union
    of ~11,000 dates, so each case column is mostly empty). All checks therefore
    work on each case's own non-empty points, never on the stacked frame."""
    rows = []
    for name, df in sheets.items():
        n_pts, n_neg, n_dec, n_nonfinite = [], 0, 0, 0
        for c in [c for c in df.columns if isinstance(c, int)]:
            s = df[c].dropna()
            n_pts.append(len(s))
            n_neg += int((s < 0).sum())
            n_nonfinite += int((~np.isfinite(s.values)).sum())
            if name.startswith("cum"):
                tol = 1e-6 * max(s.abs().max(), 1.0)
                n_dec += int((s.diff().iloc[1:] < -tol).sum())
        rows.append({
            "sheet": name,
            "unique_dates": len(df),
            "pts_per_case_min": min(n_pts),
            "pts_per_case_max": max(n_pts),
            "n_negative": n_neg,
            "n_nonfinite": n_nonfinite,
            "n_decreasing_steps": n_dec if name.startswith("cum") else np.nan,
        })
    return pd.DataFrame(rows)


def to_common_grid(df: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    """Linear interpolation of every case column onto `grid`. Returns (n_cases, n_times)."""
    x = df.index.values.astype("datetime64[s]").astype(float)
    xg = grid.values.astype("datetime64[s]").astype(float)
    out = np.empty((C.N_CASES, len(grid)))
    for i in range(C.N_CASES):
        col = df[i + 1]
        m = col.notna().values
        out[i] = np.interp(xg, x[m], col.values[m])
    return out


def interval_rates(cube: dict, robs: dict, dates: pd.DatetimeIndex):
    """Put simulated and observed RATES on the same definition.

    DATA CONVENTION (verified): the observed cumulative equals  cumsum(rate * days)  exactly, i.e. the
    observed rate printed at date t is the average rate over the NEXT quarter [t, t+1).
    The simulator rate sheets, in contrast, store the instantaneous rate AT date t.  Comparing them
    directly shifts everything by about one quarter (and makes the 1998 ramp-up look like a huge error).
    Fix: for simulations use  (cum[t+1]-cum[t]) / days  - the same 'average over the next quarter'.
    Returns sim (100, n_t-1), obs (n_t-1,), days (n_t-1,)  per variable."""
    days = np.diff(dates.values.astype("datetime64[D]")).astype(float)
    sim = {v: np.diff(cube[v], axis=1) / days for v in C.VARS}
    obs = {v: robs[v][:-1] for v in C.VARS}
    return sim, obs, days


def build_cube(sheets_raw: dict | None = None, prefix: str = "cum"):
    """Main entry. prefix='cum' (cumulative curves) or 'rate'."""
    if sheets_raw is None:
        sheets_raw = L.load_curve_sheets()
    sheets = {k: tidy_sheet(v) for k, v in sheets_raw.items()}

    # common grid = the dates where observed data exists (identical in all sheets)
    obs_dates = sheets[f"{prefix}_oil"]["observed"].dropna().index
    for v in C.VARS:
        other = sheets[f"{prefix}_{v}"]["observed"].dropna().index
        assert obs_dates.equals(other), f"observed dates differ for {v}"

    cube, obs = {}, {}
    for v in C.VARS:
        df = sheets[f"{prefix}_{v}"]
        cube[v] = to_common_grid(df, obs_dates)
        obs[v] = df.loc[obs_dates, "observed"].values.astype(float)
    return cube, obs, obs_dates, sheets


def save_cache(cube, obs, dates, prefix="cum"):
    C.ensure_dirs()
    np.savez(C.CACHE_DIR / f"{prefix}_cube.npz",
             **{f"sim_{v}": cube[v] for v in C.VARS},
             **{f"obs_{v}": obs[v] for v in C.VARS},
             dates=dates.values.astype("datetime64[D]").astype(str))


def load_cache(prefix="cum"):
    z = np.load(C.CACHE_DIR / f"{prefix}_cube.npz", allow_pickle=False)
    cube = {v: z[f"sim_{v}"] for v in C.VARS}
    obs = {v: z[f"obs_{v}"] for v in C.VARS}
    dates = pd.DatetimeIndex(z["dates"])
    return cube, obs, dates


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raw = L.load_curve_sheets()
    tidy = {k: tidy_sheet(v) for k, v in raw.items()}
    print(sanity_report(tidy).to_string(index=False))

    for prefix in ("cum", "rate"):
        cube, obs, dates, _ = build_cube(raw, prefix)
        save_cache(cube, obs, dates, prefix)
        print(f"\n[{prefix}] common grid: {len(dates)} dates {dates[0].date()} -> {dates[-1].date()}")
        for v in C.VARS:
            print(f"  {v}: cube {cube[v].shape}, obs final={obs[v][-1]:.4g}, "
                  f"sim final min/max={cube[v][:, -1].min():.4g}/{cube[v][:, -1].max():.4g}")
    print("\ncache written to", C.CACHE_DIR.resolve())
