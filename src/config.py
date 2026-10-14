"""Central settings. Change paths / weights here only.

Default layout:   <project>/data/*.xlsx   (the organizers' files, original names)
To use another folder, set the environment variable DWC_DATA_DIR.
"""
import os
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Paths  (anchored to the project folder, so it works from any launch directory)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DWC_DATA_DIR", PROJECT_ROOT / "data"))   # put the .xlsx files in  <project>/data/
OUT_DIR = Path(os.environ.get("DWC_OUT_DIR", PROJECT_ROOT / "outputs"))
CACHE_DIR = OUT_DIR / "cache"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

# ----------------------------------------------------------------------------
# Problem definition (from the kick-off deck / README)
# ----------------------------------------------------------------------------
N_CASES = 100
TRAIN_CASES = list(range(1, 71))      # 70 train
VAL_CASES = list(range(71, 86))       # 15 validation
TEST_CASES = list(range(86, 101))     # 15 test

PARAMS = ["fault_tran", "poro_mult", "perm_mult", "aq_pv"]   # the 4 uncertainties

# Production variables used for matching / embedding (cumulative curves)
VARS = ["gas", "oil", "wat"]
CUM_SHEETS = {
    "gas": "Gas production cumulative",
    "oil": "Oil production cumulative",
    "wat": "Water production cumulative",
}
RATE_SHEETS = {
    "gas": "Gas production rate",
    "oil": "Oil production rate",
    "wat": "Water production rate",
}

# Weights used in the history-match mismatch (water is the most discriminating)
MISMATCH_WEIGHTS = {"gas": 1.0, "oil": 1.0, "wat": 2.0}

# Time axes
HIST_END = pd.Timestamp("2008-01-01")
FORECAST_DATES = pd.date_range("2008-04-01", "2028-01-01", freq="QS")  # 80 rows (template)

SEED = 42


def ensure_dirs():
    for d in (OUT_DIR, CACHE_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
