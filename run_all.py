"""Run the whole pipeline, or only some steps:

    python run_all.py                 # steps 1-11 (about 10 min; add --skip-lstm to save ~4 min)
    python run_all.py --steps 2 3     # only cleaning + EDA (uses cache from step 2)

Steps: 1 load check | 2 clean -> cache | 3 EDA figures | 4 PCA / oil2vec | 5 proxy + history match |
       6 deeper EDA (range vs effect, residuals, Sobol) | 7 history-match variants (ABC; sanity check)
       8 MCMC history match + discrepancy | 9 proxy comparison (GP / MLP / LSTM) |
       10 sequential design | 11 forecast + back-tests + filled template
       (11 needs the posterior written by 8)
(The optional 3-D grid conversion is separate:  python -m src.load_grids --workers 4)
"""
import argparse
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

from src import config as C, load_data as L, clean as K, eda, embed, proxy_baseline  # noqa: E402
from src import eda_advanced, match_v2, mcmc, proxy_compare, active_design, forecast  # noqa: E402


def step1():
    print("== 1. LOAD CHECK ==")
    print(L.load_uncertainty().groupby("split").size().to_dict())
    print({k: len(v) for k, v in L.load_curve_sheets().items()})


def step2():
    print("== 2. CLEAN ==")
    raw = L.load_curve_sheets()
    print(K.sanity_report({k: K.tidy_sheet(v) for k, v in raw.items()}).to_string(index=False))
    for prefix in ("cum", "rate"):
        cube, obs, dates, _ = K.build_cube(raw, prefix)
        K.save_cache(cube, obs, dates, prefix)
    print("cache written to", C.CACHE_DIR)


STEPS = {1: step1, 2: step2, 3: eda.run_all, 4: embed.run_all, 5: proxy_baseline.run_all,
         6: eda_advanced.run_all, 7: match_v2.run_all, 8: mcmc.run_all, 9: proxy_compare.run_all,
         10: active_design.run_all, 11: forecast.run_all}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="*", default=list(range(1, 12)))
    ap.add_argument("--skip-lstm", action="store_true", help="step 9 without the slow LSTM (saves ~4 min)")
    a = ap.parse_args()
    for s in a.steps:
        t = time.time()
        if s == 9:
            proxy_compare.run_all(include_lstm=not a.skip_lstm)
        else:
            STEPS[s]()
        print(f"-- step {s} took {time.time() - t:.1f}s\n")
