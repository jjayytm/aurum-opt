"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AURUM-OPT  v2.0 — Next-Generation Stope Shape Optimisation Tool             ║
║  "Mine smarter. Strike gold faster."                                         ║
║  Hackathon 2026                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PIPELINE                                                                    ║
║  1  data_loader          CSV ingestion with auto header detection            ║
║  2  preprocessing        Block attributes, grid indices, per-block waste     ║
║  3  build_3d_grids       3D prefix-sum arrays: gold · tonnes · waste         ║
║  4  stope_engine         Vectorised O(N) sliding-window scan                 ║
║  5  get_aligned_z_levels Gold-weighted Z elevation alignment (Memo Mar2026)  ║
║  6  economic_filter      Grade cutoff + Z compliance filter                  ║
║  7  ai_training          XGBoost classifier + regressor (10 features)        ║
║  8  ai_prediction        Full-grid economic zone prediction                  ║
║  9  ranking_module       Composite AI quality scoring                        ║
║  10 optimised_greedy     Exact DP column-based non-overlap selection         ║
║  11 iterative_improvement Spatial-index hill-climbing post-processor         ║
║  12 dxf_export           3D box prism DXF output                             ║
║  13 export_dashboard_json JSON bridge for live dashboard                     ║
║  14 reporting            Console results summary                             ║
║  15 live_reeval          Sub-30s Part 2 cutoff re-evaluation                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE                                                                       ║
║    python src/stope_optimizer.py "Block Model.csv" 10.0                     ║
║                                                                              ║
║  OUTPUTS  (all written to  outputs/  automatically)                          ║
║    outputs/stopes_optimised.dxf     ← submit to judges                      ║
║    outputs/stopes_all_candidates.dxf                                         ║
║    outputs/stopes_results.csv                                                ║
║    results.json                     ← dashboard data (project root)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    warnings.warn("ezdxf not found — DXF export disabled.  pip install ezdxf")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not found — AI layer disabled.  pip install scikit-learn")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS — all outputs go to outputs/ next to the project root
# ─────────────────────────────────────────────────────────────────────────────

def _get_output_dir() -> str:
    """
    Returns the absolute path to the outputs/ folder.
    Works whether the script is run from the project root, from src/, or anywhere.
    Creates the folder if it does not exist.
    """
    # __file__ resolves to  aurum-opt/src/stope_optimizer.py
    # going up one level gives aurum-opt/
    src_dir     = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.normpath(os.path.join(src_dir, ".."))
    out_dir     = os.path.join(project_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, project_dir

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH          = "Hackathon 2026 - Block Model.csv"
STOPE_HEIGHT_M    = 30.0        # Z direction  (fixed by rules)
STOPE_LENGTH_M    = 20.0        # X direction  (fixed by rules)
STOPE_THICKNESS_M =  5.0        # Y direction  (fixed by rules)
CUTOFF_GRADE      = 10.0        # g/t  — Part 1 default
FILL_GRADE        =  0.0        # missing block grade  (Technical Notes §4)
FILL_DENSITY      =  2.8        # missing block density t/m³
AI_SAMPLE_FRAC    =  0.10
AI_N_ESTIMATORS   =  150
AI_RANDOM_STATE   =  42
ITERATIVE_ITERS   =  100

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def data_loader(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the block-model CSV.
    Auto-detects the real header row — robust to any number of metadata lines
    that mining software may export before the column headers.
    """
    print(f"[data_loader]  Loading: {csv_path}")
    required = {"XC", "YC", "ZC", "XINC", "YINC", "ZINC", "AU", "DENSITY"}

    skiprows = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if required.issubset(c.strip().upper() for c in line.split(",")):
                skiprows = i
                break
            if i > 10:
                break

    df = pd.read_csv(csv_path, skiprows=skiprows)
    df.columns = df.columns.str.strip().str.upper()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"[data_loader]  {len(df):,} blocks loaded.  "
          f"Header at line {skiprows}.  "
          f"Columns: {list(df.columns)}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values, compute Volume / Tonnes / Gold per block,
    and assign integer grid indices (IX, IY, IZ) for prefix-sum mapping.
    """
    print("[preprocessing]  Computing block attributes ...")

    df["AU"]      = df["AU"].fillna(FILL_GRADE).clip(lower=0.0)
    df["DENSITY"] = df["DENSITY"].fillna(FILL_DENSITY).clip(lower=0.0)
    df["VOLUME"]  = df["XINC"] * df["YINC"] * df["ZINC"]
    df["TONNES"]  = df["VOLUME"] * df["DENSITY"]
    df["GOLD_G"]  = df["TONNES"] * df["AU"]

    xinc = df["XINC"].mode()[0]
    yinc = df["YINC"].mode()[0]
    zinc = df["ZINC"].mode()[0]

    df["IX"] = np.round((df["XC"] - df["XC"].min()) / xinc).astype(int)
    df["IY"] = np.round((df["YC"] - df["YC"].min()) / yinc).astype(int)
    df["IZ"] = np.round((df["ZC"] - df["ZC"].min()) / zinc).astype(int)

    df.attrs.update({
        "xinc": xinc, "yinc": yinc, "zinc": zinc,
        "x_min": df["XC"].min(),
        "y_min": df["YC"].min(),
        "z_min": df["ZC"].min(),
    })

    print(f"[preprocessing]  Block size : {xinc}m x {yinc}m x {zinc}m")
    print(f"[preprocessing]  Origin     : "
          f"X={df.attrs['x_min']:.1f}  "
          f"Y={df.attrs['y_min']:.1f}  "
          f"Z={df.attrs['z_min']:.1f}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — PREFIX SUM GRIDS
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_grids(df: pd.DataFrame):
    """
    Build three 3-D prefix-sum arrays: gold, total tonnes, waste tonnes.

    waste_grid[ix,iy,iz] holds the tonnes of that block only if its AU grade
    is below CUTOFF_GRADE — giving exact per-stope waste figures without any
    fixed-density approximation.

    Prefix-sum O(1) box query via inclusion-exclusion:
        S = P[x1,y1,z1] - P[x0,y1,z1] - P[x1,y0,z1] - P[x1,y1,z0]
          + P[x0,y0,z1] + P[x0,y1,z0] + P[x1,y0,z0] - P[x0,y0,z0]
    """
    print("[prefix_sum_module]  Building 3D grids ...")
    nx = int(df["IX"].max()) + 1
    ny = int(df["IY"].max()) + 1
    nz = int(df["IZ"].max()) + 1
    print(f"[prefix_sum_module]  Grid: {nx}x{ny}x{nz}  ({nx*ny*nz:,} cells)")

    ix_v = df["IX"].values
    iy_v = df["IY"].values
    iz_v = df["IZ"].values

    gold_grid  = np.zeros((nx, ny, nz), dtype=np.float64)
    tons_grid  = np.zeros((nx, ny, nz), dtype=np.float64)
    waste_grid = np.zeros((nx, ny, nz), dtype=np.float64)

    np.add.at(gold_grid,  (ix_v, iy_v, iz_v), df["GOLD_G"].values)
    np.add.at(tons_grid,  (ix_v, iy_v, iz_v), df["TONNES"].values)
    np.add.at(waste_grid, (ix_v, iy_v, iz_v),
              df["TONNES"].values * (df["AU"].values < CUTOFF_GRADE).astype(float))

    P_gold  = gold_grid.cumsum(0).cumsum(1).cumsum(2)
    P_tons  = tons_grid.cumsum(0).cumsum(1).cumsum(2)
    P_waste = waste_grid.cumsum(0).cumsum(1).cumsum(2)

    print("[prefix_sum_module]  Prefix-sum arrays built (gold, tonnes, waste).")
    return gold_grid, tons_grid, P_gold, P_tons, P_waste, (nx, ny, nz)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — STOPE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def stope_engine(df, P_gold, P_tons, P_waste, grid_shape):
    """
    Vectorised 3-D sliding-window scan using prefix sums.
    Complexity: O(Nx × Ny × Nz) — single pass, no Python inner loops.
    """
    xinc = df.attrs["xinc"]
    yinc = df.attrs["yinc"]
    zinc = df.attrs["zinc"]

    lb = max(1, int(round(STOPE_LENGTH_M    / xinc)))
    tb = max(1, int(round(STOPE_THICKNESS_M / yinc)))
    hb = max(1, int(round(STOPE_HEIGHT_M    / zinc)))

    nx, ny, nz = grid_shape
    nox = nx - lb + 1
    noy = ny - tb + 1
    noz = nz - hb + 1

    if nox <= 0 or noy <= 0 or noz <= 0:
        raise ValueError("Stope geometry exceeds block model extents.")

    print(f"[stope_engine]  Stope: L={lb} x T={tb} x H={hb}  "
          f"({lb*xinc:.0f}m x {tb*yinc:.0f}m x {hb*zinc:.0f}m)")
    print(f"[stope_engine]  Scanning {nox*noy*noz:,} positions ...")

    x0 = np.arange(nox);  x1 = x0 + lb - 1
    y0 = np.arange(noy);  y1 = y0 + tb - 1
    z0 = np.arange(noz);  z1 = z0 + hb - 1

    def _pad(P):
        Pp = np.zeros((P.shape[0]+1, P.shape[1]+1, P.shape[2]+1), dtype=np.float64)
        Pp[1:, 1:, 1:] = P
        return Pp

    Pg = _pad(P_gold);  Pt = _pad(P_tons);  Pw = _pad(P_waste)

    def _vbox(Pp, ax0, ax1, ay0, ay1, az0, az1):
        X0, X1 = ax0, ax1 + 1
        Y0, Y1 = ay0, ay1 + 1
        Z0, Z1 = az0, az1 + 1
        def _p(i, j, k):
            return Pp[i[:, None, None], j[None, :, None], k[None, None, :]]
        return (_p(X1,Y1,Z1) - _p(X0,Y1,Z1) - _p(X1,Y0,Z1) - _p(X1,Y1,Z0)
              + _p(X0,Y0,Z1) + _p(X0,Y1,Z0) + _p(X1,Y0,Z0) - _p(X0,Y0,Z0))

    gold_sum  = _vbox(Pg, x0, x1, y0, y1, z0, z1)
    tons_sum  = _vbox(Pt, x0, x1, y0, y1, z0, z1)
    waste_sum = _vbox(Pw, x0, x1, y0, y1, z0, z1)

    with np.errstate(invalid="ignore", divide="ignore"):
        wavg = np.where(tons_sum > 0, gold_sum / tons_sum, 0.0)

    print(f"[stope_engine]  Done.  Max wavg grade: {wavg.max():.2f} g/t")
    return wavg, gold_sum, tons_sum, waste_sum, (x0, y0, z0), (lb, tb, hb)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — Z ELEVATION ALIGNMENT  (Memo 02Mar2026 — MANDATORY)
# ─────────────────────────────────────────────────────────────────────────────

def get_aligned_z_levels(df, stope_blocks, grid_shape,
                          wavg=None, gold_sum=None, cutoff=CUTOFF_GRADE):
    """
    Return valid Z origin indices satisfying the mandatory elevation alignment
    rule (Memo 02Mar2026): all stopes must share consistent bottom Z elevations
    spaced exactly one stope height apart.

    Tries all possible starting offsets (0 to hb-1) and picks the one that
    maximises total captured gold above cutoff — not just stope count.
    """
    zinc = df.attrs["zinc"]
    lb, tb, hb = stope_blocks
    nx, ny, nz = grid_shape
    n_origins_z = nz - hb + 1

    if n_origins_z <= 0:
        return np.array([0])

    best_offset = 0
    best_score  = -1.0

    for offset in range(hb):
        levels = np.arange(offset, n_origins_z, hb)
        if wavg is not None and gold_sum is not None:
            z_mask         = np.zeros(wavg.shape[2], dtype=bool)
            z_mask[levels] = True
            econ           = wavg[:, :, z_mask] >= cutoff
            score          = float(gold_sum[:, :, z_mask][econ].sum())
        else:
            score = float(len(levels))

        if score > best_score:
            best_score  = score
            best_offset = offset

    aligned_iz = np.arange(best_offset, n_origins_z, hb)
    z_elevs    = df.attrs["z_min"] + aligned_iz * zinc

    print(f"[z_alignment]  Stope height = {hb} blocks ({hb*zinc:.0f}m)")
    print(f"[z_alignment]  Best offset  = {best_offset} blocks "
          f"({best_offset*zinc:.0f}m from model base)")
    print(f"[z_alignment]  Valid Z levels: {len(aligned_iz)}  |  "
          f"Elevations (m): {[f'{e:.1f}' for e in z_elevs]}")
    return aligned_iz

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — ECONOMIC FILTER
# ─────────────────────────────────────────────────────────────────────────────

def economic_filter(df, wavg, gold_sum, tons_sum, waste_sum,
                    origins, stope_blocks, grid_shape, cutoff=CUTOFF_GRADE):
    """
    Apply two filters:
      1. Grade filter   — wavg >= cutoff
      2. Z-level filter — origin must lie on the gold-maximising aligned grid
    Waste tonnes are read from the prefix-sum waste array for exact figures.
    """
    xinc  = df.attrs["xinc"];  yinc = df.attrs["yinc"];  zinc = df.attrs["zinc"]
    x_min = df.attrs["x_min"]; y_min = df.attrs["y_min"]; z_min = df.attrs["z_min"]
    x0_arr, y0_arr, z0_arr = origins
    lb, tb, hb = stope_blocks

    grade_mask   = wavg >= cutoff
    aligned_iz   = get_aligned_z_levels(df, stope_blocks, grid_shape,
                                         wavg=wavg, gold_sum=gold_sum,
                                         cutoff=cutoff)
    z_align_mask = np.zeros(wavg.shape, dtype=bool)
    z_align_mask[:, :, aligned_iz] = True
    mask = grade_mask & z_align_mask

    print(f"[economic_filter]  Cutoff = {cutoff} g/t")
    print(f"[economic_filter]  Passing grade only : {grade_mask.sum():,}")
    print(f"[economic_filter]  Passing grade + Z  : {mask.sum():,} "
          f"(Z-alignment enforced — Memo 02Mar2026)")

    if mask.sum() == 0:
        print("[economic_filter]  WARNING: No economic stopes found.")
        return pd.DataFrame()

    ix0, iy0, iz0 = np.where(mask)
    xc0 = x_min + x0_arr[ix0] * xinc
    yc0 = y_min + y0_arr[iy0] * yinc
    zc0 = z_min + z0_arr[iz0] * zinc

    stopes = pd.DataFrame({
        "IX0": ix0,       "IY0": iy0,       "IZ0": iz0,
        "X_ORIGIN": xc0,  "Y_ORIGIN": yc0,  "Z_ORIGIN": zc0,
        "X_END":  xc0 + lb * xinc,
        "Y_END":  yc0 + tb * yinc,
        "Z_END":  zc0 + hb * zinc,
        "WAVG_GRADE":   wavg[mask],
        "GOLD_G":       gold_sum[mask],
        "GOLD_OZ":      gold_sum[mask] / 31.1035,
        "ORE_TONNES":   tons_sum[mask],
        "WASTE_TONNES": waste_sum[mask].clip(min=0),
    })

    unique_z  = sorted(stopes["Z_ORIGIN"].unique())
    z_diffs   = np.diff(unique_z)
    hb_m      = hb * zinc
    compliant = len(z_diffs) == 0 or all(abs(d % hb_m) < 0.5 for d in z_diffs)

    print(f"[economic_filter]  Unique Z elevations : "
          f"{[f'{z:.1f}m' for z in unique_z]}")
    print(f"[economic_filter]  Z-alignment : "
          f"{'PASS' if compliant else 'FAIL'} (Memo 02Mar2026)")
    return stopes

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — AI TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def ai_training(df, wavg, gold_sum, tons_sum, waste_sum, grid_shape, stope_blocks):
    """
    Train XGBoost (or RandomForest fallback) classifier + regressor.

    10 features: wavg_grade, grade_std, total_gold, total_tonnes,
                 waste_ratio, gold_eff, depth_z, pos_x, pos_y,
                 grade×gold interaction
    """
    if not HAS_SKLEARN:
        print("[ai_training]  Skipped — scikit-learn not available.")
        return None, None, None

    print("[ai_training]  Generating training data ...")
    nx, ny, nz = grid_shape
    lb, tb, hb = stope_blocks
    nox = nx - lb + 1;  noy = ny - tb + 1;  noz = nz - hb + 1

    if nox <= 0 or noy <= 0 or noz <= 0:
        return None, None, None

    total_pos = nox * noy * noz
    n_sample  = min(max(1000, int(total_pos * AI_SAMPLE_FRAC)), total_pos)

    rng      = np.random.default_rng(AI_RANDOM_STATE)
    flat_idx = rng.choice(total_pos, size=n_sample, replace=False)
    iz_s = flat_idx % noz
    tmp  = flat_idx // noz
    iy_s = tmp % noy
    ix_s = tmp // noy

    g_vals  = wavg[ix_s, iy_s, iz_s]
    gold_v  = gold_sum[ix_s, iy_s, iz_s]
    tons_v  = tons_sum[ix_s, iy_s, iz_s]
    waste_v = waste_sum[ix_s, iy_s, iz_s]

    with np.errstate(invalid="ignore", divide="ignore"):
        gold_eff    = np.where(tons_v > 0, gold_v / tons_v,  0.0)
        waste_ratio = np.where(tons_v > 0, waste_v / tons_v, 1.0)

    def _nbr_std(arr, ix, iy, iz):
        sh = arr.shape;  stack = []
        for di in (-1, 0, 1):
            ni = np.clip(ix + di, 0, sh[0] - 1)
            for dj in (-1, 0, 1):
                nj = np.clip(iy + dj, 0, sh[1] - 1)
                for dk in (-1, 0, 1):
                    nk = np.clip(iz + dk, 0, sh[2] - 1)
                    stack.append(arr[ni, nj, nk])
        return np.std(np.array(stack), axis=0)

    grade_std = _nbr_std(wavg, ix_s, iy_s, iz_s)

    X = np.column_stack([
        g_vals, grade_std, gold_v, tons_v, waste_ratio, gold_eff,
        iz_s / max(noz - 1, 1),
        ix_s / max(nox - 1, 1),
        iy_s / max(noy - 1, 1),
        g_vals * gold_v / (gold_v.max() + 1e-9),
    ])
    y_clf = (g_vals >= CUTOFF_GRADE).astype(int)
    y_reg = g_vals

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    if HAS_XGB:
        clf = XGBClassifier(n_estimators=AI_N_ESTIMATORS, random_state=AI_RANDOM_STATE,
                            max_depth=6, learning_rate=0.1,
                            eval_metric="logloss", verbosity=0)
        reg = XGBRegressor(n_estimators=AI_N_ESTIMATORS, random_state=AI_RANDOM_STATE,
                           max_depth=6, learning_rate=0.1, verbosity=0)
        algo = "XGBoost"
    else:
        clf = RandomForestClassifier(n_estimators=AI_N_ESTIMATORS,
                                     random_state=AI_RANDOM_STATE, n_jobs=-1)
        reg = RandomForestRegressor(n_estimators=AI_N_ESTIMATORS,
                                    random_state=AI_RANDOM_STATE, n_jobs=-1)
        algo = "RandomForest"

    print(f"[ai_training]  Training {algo} on {n_sample:,} samples ...")
    clf.fit(X_sc, y_clf)
    reg.fit(X_sc, y_reg)
    print(f"[ai_training]  Done.  Economic fraction: {100*y_clf.mean():.1f}%")
    return clf, reg, scaler

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — AI PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def ai_prediction(clf, reg, scaler, wavg, gold_sum, tons_sum, waste_sum,
                  grid_shape, stope_blocks):
    """
    Run classifier over all positions to generate a priority mask.
    The mask never overrides cutoff or Z-alignment decisions.
    """
    if clf is None or not HAS_SKLEARN:
        print("[ai_prediction]  Skipped.")
        return None

    print("[ai_prediction]  Generating priority mask ...")
    nx, ny, nz = grid_shape
    lb, tb, hb = stope_blocks
    nox = nx - lb + 1;  noy = ny - tb + 1;  noz = nz - hb + 1

    ix_f, iy_f, iz_f = (a.ravel() for a in np.meshgrid(
        np.arange(nox), np.arange(noy), np.arange(noz), indexing="ij"))

    g_v = wavg[ix_f, iy_f, iz_f]
    g_g = gold_sum[ix_f, iy_f, iz_f]
    t_v = tons_sum[ix_f, iy_f, iz_f]
    w_v = waste_sum[ix_f, iy_f, iz_f]

    with np.errstate(invalid="ignore", divide="ignore"):
        gold_eff    = np.where(t_v > 0, g_g / t_v,  0.0)
        waste_ratio = np.where(t_v > 0, w_v / t_v,  1.0)

    X = np.column_stack([
        g_v, np.zeros(len(g_v)), g_g, t_v, waste_ratio, gold_eff,
        iz_f / max(noz - 1, 1),
        ix_f / max(nox - 1, 1),
        iy_f / max(noy - 1, 1),
        g_v * g_g / (g_g.max() + 1e-9),
    ])

    y_pred     = clf.predict(scaler.transform(X))
    mask       = np.zeros(wavg.shape, dtype=bool)
    mask[ix_f, iy_f, iz_f] = y_pred.astype(bool)

    true_econ = wavg >= CUTOFF_GRADE
    prec = (mask & true_econ).sum() / max(mask.sum(), 1)
    rec  = (mask & true_econ).sum() / max(true_econ.sum(), 1)
    print(f"[ai_prediction]  {mask.sum():,} predicted economic  "
          f"(precision={prec:.2%}, recall={rec:.2%})")
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 9 — RANKING MODULE
# ─────────────────────────────────────────────────────────────────────────────

def ranking_module(stopes: pd.DataFrame, reg=None, scaler=None) -> pd.DataFrame:
    """
    Score each stope with the AI regressor (or heuristic fallback).
    Uses the same 10-feature matrix as ai_training for consistency.
    """
    if stopes.empty:
        return stopes

    print(f"[ranking_module]  Ranking {len(stopes):,} stopes ...")
    stopes     = stopes.copy()
    max_grade  = stopes["WAVG_GRADE"].max()

    with np.errstate(invalid="ignore", divide="ignore"):
        stopes["GOLD_EFF"] = np.where(
            stopes["ORE_TONNES"] > 0,
            stopes["GOLD_G"] / stopes["ORE_TONNES"], 0.0)

    total_t = stopes["ORE_TONNES"] + stopes["WASTE_TONNES"]
    stopes["WASTE_RATIO"]  = np.where(total_t > 0,
                                       stopes["WASTE_TONNES"] / total_t, 0.0)
    stopes["DEPTH_FACTOR"] = 1.0 / (1.0 + stopes["Z_ORIGIN"].abs() / 1000.0)
    stopes["GRADE_NORM"]   = stopes["WAVG_GRADE"] / max(max_grade, 1e-9)

    if reg is not None and HAS_SKLEARN:
        max_gold_g = stopes["GOLD_G"].max()
        X_rank = np.column_stack([
            stopes["WAVG_GRADE"].values,
            np.zeros(len(stopes)),
            stopes["GOLD_G"].values,
            stopes["ORE_TONNES"].values,
            stopes["WASTE_RATIO"].values,
            stopes["GOLD_EFF"].values,
            stopes["DEPTH_FACTOR"].values,
            stopes["GRADE_NORM"].values,
            np.zeros(len(stopes)),
            stopes["WAVG_GRADE"].values * stopes["GOLD_G"].values / (max_gold_g + 1e-9),
        ])
        try:
            stopes["AI_SCORE"] = reg.predict(scaler.transform(X_rank))
        except Exception as exc:
            print(f"[ranking_module]  Regressor fallback ({exc})")
            stopes["AI_SCORE"] = stopes["GRADE_NORM"]
    else:
        ge_max = stopes["GOLD_EFF"].max()
        stopes["AI_SCORE"] = (
            0.50 * stopes["GRADE_NORM"]
          + 0.25 * stopes["GOLD_EFF"] / max(ge_max, 1e-9)
          + 0.15 * stopes["DEPTH_FACTOR"]
          - 0.10 * stopes["WASTE_RATIO"]
        )

    stopes = stopes.sort_values("AI_SCORE", ascending=False).reset_index(drop=True)
    stopes.index += 1
    stopes.index.name = "RANK"

    r = stopes.iloc[0]
    print(f"[ranking_module]  Top stope: {r['WAVG_GRADE']:.2f} g/t  "
          f"{r['GOLD_OZ']:.1f} oz  score={r['AI_SCORE']:.4f}")
    return stopes

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 10 — EXACT DP NON-OVERLAP SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_block_sets(stopes, xinc, yinc, zinc):
    """Pre-compute frozenset of block indices for every stope — called once."""
    block_sets = []
    for _, row in stopes.iterrows():
        ix0 = int(row["IX0"]); iy0 = int(row["IY0"]); iz0 = int(row["IZ0"])
        lb  = int(round((row["X_END"] - row["X_ORIGIN"]) / xinc))
        tb  = int(round((row["Y_END"] - row["Y_ORIGIN"]) / yinc))
        hb  = int(round((row["Z_END"] - row["Z_ORIGIN"]) / zinc))
        block_sets.append(frozenset(
            (ix0+dx, iy0+dy, iz0+dz)
            for dx in range(lb) for dy in range(tb) for dz in range(hb)
        ))
    return block_sets


def optimised_greedy_selection(stopes: pd.DataFrame, df: pd.DataFrame):
    """
    Exact 1-D interval scheduling DP — provably optimal non-overlap selection.

    Geometric insight:
      T = 1 block  →  no Y-overlaps possible
      Z levels spaced hb blocks apart  →  no Z-overlaps possible
      Only X-direction overlaps need resolving

    For each (IY, IZ) column we run weighted interval scheduling DP in
    O(n log n). This gives the mathematically proven maximum gold extraction.
    """
    print(f"\n[optimised_greedy]  Column-based exact DP optimisation ...")

    xinc = df.attrs["xinc"]
    yinc = df.attrs["yinc"]
    zinc = df.attrs["zinc"]
    lb   = int(round(STOPE_LENGTH_M / xinc))

    stopes_r   = stopes.reset_index(drop=True)
    block_sets = _precompute_block_sets(stopes_r, xinc, yinc, zinc)

    stopes_r["_IX0"] = stopes_r["IX0"].astype(int)
    stopes_r["_IY0"] = stopes_r["IY0"].astype(int)
    stopes_r["_IZ0"] = stopes_r["IZ0"].astype(int)

    selected_indices = []
    processed        = 0

    for (iz0, iy0), grp in stopes_r.groupby(["_IZ0", "_IY0"]):
        grp_s = grp.sort_values("_IX0").reset_index()
        n     = len(grp_s)

        if n == 0:
            continue
        if n == 1:
            selected_indices.append(grp_s.loc[0, "index"])
            processed += 1
            continue

        gold     = grp_s["GOLD_G"].values
        ix0s     = grp_s["_IX0"].values
        orig_idx = grp_s["index"].values
        dp       = gold.copy()
        prev     = [-1] * n

        for i in range(1, n):
            lo, hi, best_j = 0, i - 1, -1
            threshold = ix0s[i] - lb
            while lo <= hi:
                mid = (lo + hi) // 2
                if ix0s[mid] <= threshold:
                    best_j = mid;  lo = mid + 1
                else:
                    hi = mid - 1

            candidate = (dp[best_j] if best_j >= 0 else 0.0) + gold[i]
            if candidate > dp[i - 1]:
                dp[i] = candidate;  prev[i] = best_j
            else:
                dp[i] = dp[i - 1]; prev[i] = i - 1

        col_selected = []
        i = n - 1
        while i >= 0:
            best_j    = prev[i]
            prev_gold = dp[best_j] if best_j >= 0 else 0.0
            if abs(dp[i] - (prev_gold + gold[i])) < 1e-6:
                col_selected.append(orig_idx[i])
                i = best_j
            else:
                i -= 1

        selected_indices.extend(col_selected)
        processed += 1

    result = stopes_r.loc[selected_indices].copy()

    # Sanity-check: remove any residual overlaps
    mined = set()
    clean = []
    for i, row in result.iterrows():
        bs = block_sets[i]
        if bs.isdisjoint(mined):
            clean.append(i);  mined.update(bs)
    result = stopes_r.loc[clean].copy()

    total_gold = result["GOLD_G"].sum()
    wavg_r     = total_gold / result["ORE_TONNES"].sum()
    waste_r    = result["WASTE_TONNES"].sum() / (
        result["ORE_TONNES"].sum() + result["WASTE_TONNES"].sum() + 1e-9)

    print(f"  Columns processed : {processed:,}")
    print(f"  Stopes selected   : {len(result):,}")
    print(f"  Total gold        : {total_gold/31.1035:>12,.0f} oz")
    print(f"  Mean grade        : {wavg_r:.2f} g/t")
    print(f"  Waste ratio       : {100*waste_r:.1f}%")

    result = result.reset_index(drop=True)
    result.index += 1
    result.index.name = "RANK"
    return result, block_sets, stopes_r

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 11 — ITERATIVE IMPROVEMENT
# ─────────────────────────────────────────────────────────────────────────────

def iterative_improvement(stopes_all: pd.DataFrame, block_sets_all: list,
                           initial_result: pd.DataFrame, df: pd.DataFrame,
                           n_iterations: int = ITERATIVE_ITERS) -> pd.DataFrame:
    """
    Hill-climbing post-processor on top of the DP solution.
    Spatial index reduces candidate scan ~100x vs naive approach.
    """
    if n_iterations == 0:
        return initial_result

    print(f"\n[iterative]  Hill-climbing ({n_iterations} iterations) ...")
    t0   = time.perf_counter()
    xinc = df.attrs["xinc"]
    lb   = int(round(STOPE_LENGTH_M / xinc))

    spatial_idx = defaultdict(list)
    for i, row in stopes_all.iterrows():
        kx = int(row["IX0"]) // max(lb, 1)
        kz = int(row["IZ0"])
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                spatial_idx[(kz + dz, kx + dx)].append(i)
    spatial_idx = {k: list(set(v)) for k, v in spatial_idx.items()}

    current    = initial_result.copy().reset_index(drop=True)
    stopes_all = stopes_all.reset_index(drop=True)

    key_to_all = {
        (int(r["IX0"]), int(r["IY0"]), int(r["IZ0"])): i
        for i, r in stopes_all.iterrows()
    }

    def _get_idx(row):
        return key_to_all.get(
            (int(row["IX0"]), int(row["IY0"]), int(row["IZ0"])), -1)

    current_all_idx = [_get_idx(r) for _, r in current.iterrows()]
    matched = sum(1 for x in current_all_idx if x >= 0)
    print(f"[iterative]  Mapped {matched}/{len(current)} stopes to candidate pool")

    mined = set()
    for ai in current_all_idx:
        if ai >= 0:
            mined.update(block_sets_all[ai])

    initial_gold = current["GOLD_G"].sum()
    current_gold = initial_gold
    rng          = np.random.default_rng(AI_RANDOM_STATE)
    improved     = 0

    for iteration in range(n_iterations):
        if len(current) == 0:
            break

        gold_vals = current["GOLD_G"].values
        weights   = 1.0 / (gold_vals + 1.0)
        weights  /= weights.sum()
        rm_pos    = int(rng.choice(len(current), p=weights))

        rm_row  = current.iloc[rm_pos]
        rm_key  = (int(rm_row["IX0"]), int(rm_row["IY0"]), int(rm_row["IZ0"]))
        rm_ai   = key_to_all.get(rm_key, -1)
        if rm_ai < 0:
            continue

        rm_blocks  = block_sets_all[rm_ai]
        rm_gold    = rm_row["GOLD_G"]
        temp_mined = mined - rm_blocks

        iz_key = int(rm_row["IZ0"])
        ix_key = int(rm_row["IX0"]) // max(lb, 1)
        nearby = []
        for dk in (-1, 0, 1):
            nearby.extend(spatial_idx.get((iz_key, ix_key + dk), []))

        nearby_sorted = sorted(nearby,
                               key=lambda i: stopes_all.iloc[i]["GOLD_G"],
                               reverse=True)
        new_rows  = []
        new_gold  = 0.0
        temp_used = set()

        for ci in nearby_sorted:
            cb = block_sets_all[ci]
            if (not cb.isdisjoint(rm_blocks)
                    and cb.isdisjoint(temp_mined)
                    and cb.isdisjoint(temp_used)):
                new_rows.append(stopes_all.iloc[ci])
                new_gold += stopes_all.iloc[ci]["GOLD_G"]
                temp_used.update(cb)

        if new_gold > rm_gold + 1e-6:
            current = current.drop(index=rm_pos).reset_index(drop=True)
            current_all_idx.pop(rm_pos)
            new_df  = pd.DataFrame(new_rows)
            new_ais = [key_to_all.get(
                (int(r["IX0"]), int(r["IY0"]), int(r["IZ0"])), -1)
                for _, r in new_df.iterrows()]
            current         = pd.concat([current, new_df], ignore_index=True)
            current_all_idx += new_ais
            mined            = temp_mined | temp_used
            current_gold     = current["GOLD_G"].sum()
            improved        += 1
            print(f"  [{iteration:>4}]  "
                  f"+{(new_gold-rm_gold)/31.1035:>5.0f} oz  "
                  f"total={current_gold/31.1035:>10,.0f} oz  "
                  f"stopes={len(current):,}")

    elapsed = time.perf_counter() - t0
    print(f"\n[iterative]  Done in {elapsed:.1f}s  "
          f"({improved}/{n_iterations} accepted)")
    print(f"[iterative]  Initial : {initial_gold/31.1035:>12,.0f} oz")
    print(f"[iterative]  Final   : {current_gold/31.1035:>12,.0f} oz")
    print(f"[iterative]  Gain    : +{(current_gold-initial_gold)/31.1035:>10,.0f} oz")

    current = current.sort_values("GOLD_OZ", ascending=False).reset_index(drop=True)
    current.index += 1
    current.index.name = "RANK"
    return current

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 12 — DXF EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def dxf_export(stopes: pd.DataFrame, output_path: str):
    """Write stopes as closed 3D box prisms (6 × 3DFACE + 12 × LINE per stope)."""
    if stopes.empty:
        print("[dxf_export]  No stopes to export."); return
    if not HAS_EZDXF:
        print("[dxf_export]  Skipped — pip install ezdxf"); return

    print(f"[dxf_export]  Writing {len(stopes):,} stopes → {output_path} ...")
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    doc.layers.add("STOPES",  color=3)
    doc.layers.add("OUTLINE", color=7)

    for _, row in stopes.iterrows():
        x0, y0, z0 = row["X_ORIGIN"], row["Y_ORIGIN"], row["Z_ORIGIN"]
        x1, y1, z1 = row["X_END"],    row["Y_END"],    row["Z_END"]
        c = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
             (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),
                  (2,3,7,6),(1,2,6,5),(0,3,7,4)]:
            msp.add_3dface([c[i] for i in f], dxfattribs={"layer": "STOPES"})
        for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]:
            msp.add_line(c[i], c[j], dxfattribs={"layer": "OUTLINE"})

    doc.saveas(output_path)
    print(f"[dxf_export]  Saved: {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 13 — DASHBOARD JSON EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_dashboard_json(stopes: pd.DataFrame, cutoff: float,
                           elapsed_scan: float, elapsed_total: float,
                           path: str):
    """Write results.json consumed by dashboard.html."""
    if stopes.empty:
        return

    by_elev = (
        stopes.groupby("Z_ORIGIN")
        .agg(gold_oz=("GOLD_OZ", "sum"),
             count=("GOLD_OZ", "count"),
             avg_grade=("WAVG_GRADE", "mean"))
        .reset_index()
        .sort_values("Z_ORIGIN")
    )

    top20_cols = ["X_ORIGIN", "Y_ORIGIN", "Z_ORIGIN", "WAVG_GRADE",
                  "GOLD_OZ", "ORE_TONNES", "WASTE_TONNES", "WASTE_RATIO"]
    top20 = stopes.head(20)[top20_cols].round(2).to_dict(orient="records")

    tg = stopes["GOLD_G"].sum()
    ot = stopes["ORE_TONNES"].sum()
    wt = stopes["WASTE_TONNES"].sum()

    payload = {
        "cutoff":       cutoff,
        "gold_oz":      round(stopes["GOLD_OZ"].sum(), 1),
        "gold_grams":   round(tg, 1),
        "waste_pct":    round(100 * wt / (ot + wt + 1e-9), 2),
        "ore_tonnes":   round(ot, 0),
        "waste_tonnes": round(wt, 0),
        "stope_count":  len(stopes),
        "mean_grade":   round(stopes["WAVG_GRADE"].mean(), 2),
        "max_grade":    round(stopes["WAVG_GRADE"].max(), 2),
        "scan_time_s":  round(elapsed_scan, 2),
        "total_time_s": round(elapsed_total, 2),
        "z_elevations": sorted(stopes["Z_ORIGIN"].unique().tolist()),
        "by_elevation": [
            {"z":         round(float(r["Z_ORIGIN"]), 1),
             "gold_oz":   round(float(r["gold_oz"]), 1),
             "count":     int(r["count"]),
             "avg_grade": round(float(r["avg_grade"]), 2)}
            for _, r in by_elev.iterrows()
        ],
        "top20_stopes": top20,
    }

    # Write the main payload
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[dashboard]  Exported → {path}")

    # Also maintain a lightweight index of all cutoffs that have been run
    # so the frontend can discover them dynamically without hardcoding.
    try:
        base_dir = os.path.dirname(os.path.abspath(path))
        index_path = os.path.join(base_dir, "results_index.json")
        existing: dict | None = None
        if os.path.exists(index_path):
            with open(index_path, "r") as fh:
                existing = json.load(fh)
        if not isinstance(existing, dict):
            existing = {}
        cuts = set(existing.get("cutoffs", []))
        cuts.add(float(cutoff))
        existing["cutoffs"] = sorted(cuts)
        with open(index_path, "w") as fh:
            json.dump(existing, fh, indent=2)
        print(f"[dashboard]  Updated cutoff index → {index_path}")
    except Exception as exc:  # index is best-effort only
        print(f"[dashboard]  Warning: could not update results_index.json ({exc})")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 14 — REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def reporting(stopes: pd.DataFrame, cutoff: float,
              elapsed_scan: float, elapsed_total: float):
    bar = "=" * 68
    print(f"\n{bar}\n  AURUM-OPT  --  RESULTS SUMMARY\n{bar}")

    if stopes.empty:
        print("  No economic stopes found at this cutoff grade.")
        print(f"{bar}\n"); return

    tg = stopes["GOLD_G"].sum()
    ot = stopes["ORE_TONNES"].sum()
    wt = stopes["WASTE_TONNES"].sum()

    print(f"  Cutoff grade          : {cutoff:.2f} g/t")
    print(f"  Economic stopes       : {len(stopes):,}")
    print(f"  Total gold captured   : {tg:,.0f} g  ({tg/31.1035:,.1f} oz)")
    print(f"  Ore tonnes            : {ot:,.0f} t")
    print(f"  Waste tonnes          : {wt:,.0f} t")
    print(f"  Waste ratio           : {100*wt/(ot+wt+1e-9):.1f}%")
    print(f"  Mean wavg grade       : {stopes['WAVG_GRADE'].mean():.2f} g/t")
    print(f"  Max wavg grade        : {stopes['WAVG_GRADE'].max():.2f} g/t")
    print(f"  Unique Z elevations   : "
          f"{sorted(stopes['Z_ORIGIN'].unique().tolist())}")

    zinc_step = stopes["Z_END"].iloc[0] - stopes["Z_ORIGIN"].iloc[0]
    z_diffs   = np.diff(sorted(stopes["Z_ORIGIN"].unique()))
    compliant = len(z_diffs) == 0 or all(
        abs(d % zinc_step) < 0.5 for d in z_diffs)
    print(f"  Z-alignment           : "
          f"{'PASS' if compliant else 'FAIL'} (Memo 02Mar2026)")
    print(f"  Scan time             : {elapsed_scan:.2f} s")
    print(f"  Total runtime         : {elapsed_total:.2f} s")

    print(f"\n  TOP 10 STOPES BY RANK:")
    print(f"  {'Rk':>3}  {'X_ORIG':>9}  {'Y_ORIG':>9}  {'Z_ORIG':>8}  "
          f"{'WAVG':>7}  {'Gold_oz':>9}  {'Ore_t':>9}  {'Waste%':>6}")
    for rk, row in stopes.head(10).iterrows():
        wr = 100 * row["WASTE_TONNES"] / (
            row["ORE_TONNES"] + row["WASTE_TONNES"] + 1e-9)
        print(f"  {rk:>3}  {row['X_ORIGIN']:>9.1f}  {row['Y_ORIGIN']:>9.1f}  "
              f"{row['Z_ORIGIN']:>8.1f}  {row['WAVG_GRADE']:>7.2f}  "
              f"{row['GOLD_OZ']:>9.1f}  {row['ORE_TONNES']:>9.0f}  {wr:>5.1f}%")
    print(f"\n{bar}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 15 — LIVE CUTOFF RE-EVALUATION  (Part 2)
# ─────────────────────────────────────────────────────────────────────────────

def live_reeval(df, wavg, gold_sum, tons_sum, waste_sum,
                origins, stope_blocks, grid_shape,
                new_cutoff: float, reg=None, scaler=None):
    """
    Re-evaluate at a new cutoff in under 30 seconds.
    No re-scan. No prefix-sum rebuild.

    When judges announce the Part 2 cutoff:
      1. Uncomment the live_reeval block at the bottom of this file
      2. Set new_cutoff to the announced value
      3. Save and run — done in < 30 seconds
    """
    t0 = time.perf_counter()
    out_dir, project_dir = _get_output_dir()

    print(f"\n{'='*68}")
    print(f"  LIVE RE-EVALUATION  —  NEW CUTOFF = {new_cutoff} g/t")
    print(f"{'='*68}")

    stopes_all = economic_filter(
        df, wavg, gold_sum, tons_sum, waste_sum,
        origins, stope_blocks, grid_shape, cutoff=new_cutoff)

    if stopes_all.empty:
        print("[live_reeval]  No economic stopes at this cutoff.")
        return pd.DataFrame()

    stopes_all = ranking_module(stopes_all, reg=reg, scaler=scaler)
    stopes_opt, block_sets, stopes_reset = optimised_greedy_selection(
        stopes_all, df)
    stopes_opt = iterative_improvement(
        stopes_reset, block_sets, stopes_opt, df, n_iterations=50)

    dxf_path = os.path.join(out_dir, f"stopes_cutoff_{new_cutoff:.1f}gt.dxf")
    dxf_export(stopes_opt, output_path=dxf_path)

    # Write both a generic results.json (for the dashboard default)
    # and a cutoff-specific file, so new cutoffs like 6 g/t show up
    # automatically in the frontend without manual wiring.
    e = time.perf_counter() - t0
    cutoff_str = str(new_cutoff).rstrip("0").rstrip(".")
    for _path in [
        os.path.join(project_dir, "results.json"),
        os.path.join(project_dir, f"results_{cutoff_str}gt.json"),
    ]:
        export_dashboard_json(
            stopes_opt,
            new_cutoff,
            e,
            e,
            path=_path,
        )
    reporting(stopes_opt, new_cutoff, e, e)
    return stopes_opt

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path: str = CSV_PATH, cutoff: float = CUTOFF_GRADE) -> dict:
    t0 = time.perf_counter()

    # Resolve output paths up-front so every module uses them consistently
    out_dir, project_dir = _get_output_dir()

    print("\n" + "="*68)
    print("  AURUM-OPT  v2.0  —  DETERMINISTIC PIPELINE")
    print("="*68)
    print(f"[main]  Outputs directory : {out_dir}")

    # ── Load & preprocess ────────────────────────────────────────────────────
    df = data_loader(csv_path)
    df = preprocessing(df)

    # ── Build prefix-sum grids ───────────────────────────────────────────────
    _, _, P_gold, P_tons, P_waste, grid_shape = build_3d_grids(df)

    # ── Full vectorised scan ─────────────────────────────────────────────────
    ts = time.perf_counter()
    wavg, gold_sum, tons_sum, waste_sum, origins, stope_blocks = stope_engine(
        df, P_gold, P_tons, P_waste, grid_shape)
    te = time.perf_counter()

    # ── Economic filter + Z alignment ────────────────────────────────────────
    stopes_all = economic_filter(
        df, wavg, gold_sum, tons_sum, waste_sum,
        origins, stope_blocks, grid_shape, cutoff=cutoff)

    # ── AI Enhancement Layer ─────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  AURUM-OPT  v2.0  —  AI ENHANCEMENT LAYER")
    print("="*68)

    clf, reg, scaler = ai_training(
        df, wavg, gold_sum, tons_sum, waste_sum, grid_shape, stope_blocks)
    ai_prediction(clf, reg, scaler, wavg, gold_sum, tons_sum, waste_sum,
                  grid_shape, stope_blocks)
    stopes_all = ranking_module(stopes_all, reg=reg, scaler=scaler)

    # ── Exact DP non-overlap optimisation ────────────────────────────────────
    stopes_opt, block_sets, stopes_reset = optimised_greedy_selection(
        stopes_all, df)

    # ── Hill-climbing post-processor ─────────────────────────────────────────
    stopes_opt = iterative_improvement(
        stopes_reset, block_sets, stopes_opt, df,
        n_iterations=ITERATIVE_ITERS)

    # ── Final sort ───────────────────────────────────────────────────────────
    stopes_opt = stopes_opt.sort_values(
        "GOLD_OZ", ascending=False).reset_index(drop=True)
    stopes_opt.index += 1
    stopes_opt.index.name = "RANK"

    elapsed_total = time.perf_counter() - t0

    # ── All outputs → outputs/ folder ────────────────────────────────────────
    dxf_export(stopes_opt, output_path=os.path.join(out_dir, "stopes_optimised.dxf"))
    dxf_export(stopes_all, output_path=os.path.join(out_dir, "stopes_all_candidates.dxf"))

    csv_path_out = os.path.join(out_dir, "stopes_results.csv")
    stopes_opt.to_csv(csv_path_out, index=True)
    print(f"[main]  stopes_results.csv  → {csv_path_out}")

    # Write both results.json (latest) AND results_Xgt.json (cutoff-specific)
    cutoff_str = str(cutoff).rstrip("0").rstrip(".")
    for _path in [
        os.path.join(project_dir, "results.json"),
        os.path.join(project_dir, f"results_{cutoff_str}gt.json"),
    ]:
        export_dashboard_json(stopes_opt, cutoff,
                              elapsed_scan=te - ts,
                              elapsed_total=elapsed_total,
                              path=_path)

    reporting(stopes_opt, cutoff,
              elapsed_scan=te - ts,
              elapsed_total=elapsed_total)

    return dict(
        df=df, wavg=wavg, gold_sum=gold_sum, tons_sum=tons_sum,
        waste_sum=waste_sum, origins=origins, stope_blocks=stope_blocks,
        grid_shape=grid_shape, stopes_all=stopes_all,
        clf=clf, reg=reg, scaler=scaler,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _csv = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    _cut = float(sys.argv[2]) if len(sys.argv) > 2 else CUTOFF_GRADE

    state = main(csv_path=_csv, cutoff=_cut)

    # ── PART 2 — uncomment and set cutoff when judges announce it ────────────
    # live_reeval(
    #     state["df"],           state["wavg"],         state["gold_sum"],
    #     state["tons_sum"],     state["waste_sum"],    state["origins"],
    #     state["stope_blocks"], state["grid_shape"],
    #     new_cutoff = 8.0,      # ← change this number only
    #     reg    = state["reg"],
    #     scaler = state["scaler"],
    # )
