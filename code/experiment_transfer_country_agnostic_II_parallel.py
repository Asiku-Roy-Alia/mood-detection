# experiment_transfer_country_agnostic_II.py
# Country-Agnostic II: Train on pooled foreign countries (ALL - UG), test on UG.
# Parallel version: uses half of available CPUs; parallelizes site loading and per-fold-per-repeat evals.

import os
import math
import json
from typing import Optional, List, Tuple, Iterable, Dict
import numpy as np
import pandas as pd

# ------------------ CPU / Parallel config  ------------------

def _half_cpus() -> int:
    try:
        n = os.cpu_count() or 1
    except Exception:
        n = 1
    return max(1, n // 2)

# Fix outer parallelism workers
N_WORKERS = _half_cpus()

# Prevent BLAS/OpenMP oversubscription; keep 1 thread inside each worker
for _ev in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    # only set if not explicitly preset by caller
    os.environ.setdefault(_ev, "1")

# ---------------------------------------------------------------------------------------

from joblib import Parallel, delayed

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ----------------------------- utilities -----------------------------

def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def pick_best_datetime_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if any(t in c.lower() for t in
            ["instancetimestamp","time","timestamp","date","datetime","start","end","submitted"])]
    best, best_n = None, -1
    for c in cands:
        n = parse_dt(df[c]).notna().sum()
        if n > best_n: best, best_n = c, n
    if best is None:
        raise ValueError("No usable datetime-like column found in timediaries (override with --td_time_col).")
    return best

def pick_user_id_column(td: pd.DataFrame, feat_userids: pd.Series) -> str:
    feat = set(feat_userids.astype(str).unique())
    for c in ["userid","user_id","participantid","participant_id","subject","uid"]:
        if c in td.columns and len(set(td[c].astype(str).unique()) & feat) > 0: return c
    for c in td.columns:
        try:
            if len(set(td[c].astype(str).unique()) & feat) > 0: return c
        except: pass
    raise ValueError("No user id column in timediaries matches 'userid' (override with --td_user_col).")

def map_binary(y5: np.ndarray) -> np.ndarray:
    # 1,2 -> 0 ; 3,4,5 -> 1
    return (y5 >= 3).astype(int)

def map_three(y5: np.ndarray) -> np.ndarray:
    # 1,2 -> 0 ; 3 -> 1 ; 4,5 -> 2
    return np.where(y5 <= 2, 0, np.where(y5 == 3, 1, 2)).astype(int)

def class_counts(arr: np.ndarray) -> Dict[int,int]:
    vals, cnts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

# ------------------------ IO & alignment ------------------------

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "userid" not in df.columns: raise ValueError("Expected 'userid' in features.")
    if "start_interval" not in df.columns or "end_interval" not in df.columns:
        raise ValueError("Expected 'start_interval' and 'end_interval' in features.")
    df["start_interval"] = parse_dt(df["start_interval"]).dt.tz_localize(None)
    df["end_interval"]   = parse_dt(df["end_interval"]).dt.tz_localize(None)
    # bools -> ints for KNNImputer
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols): df[bool_cols] = df[bool_cols].astype(int)
    return df

def load_timediaries(path: str, td_time_col: Optional[str], td_user_col: Optional[str],
                     feat_userids: pd.Series) -> pd.DataFrame:
    if path.endswith('.xlsx'):
        td = pd.read_excel(path)
    else:
        if os.path.isdir(path):
            parts = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
            if not parts: raise FileNotFoundError(f"No .parquet found in folder: {path}")
            path = parts[0]
        try:
            td = pd.read_parquet(path)  # needs pyarrow or fastparquet
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {e}\nInstall: pip install pyarrow  (or fastparquet)")
    if "A6a" not in td.columns: raise ValueError("Timediaries missing 'A6a' (mood).")
    time_col = td_time_col or pick_best_datetime_column(td)
    user_col = td_user_col or pick_user_id_column(td, feat_userids)
    td = td[[user_col, time_col, "A6a"]].rename(columns={user_col:"td_userid", time_col:"td_time"})
    td["td_time"] = parse_dt(td["td_time"]).dt.tz_localize(None)
    td = td[td["A6a"].notna()].copy()
    td["td_userid"] = td["td_userid"].astype(str)
    return td

def align_features_with_mood(features: pd.DataFrame, td: pd.DataFrame,
                             tolerance_minutes: int) -> pd.DataFrame:
    f = features.copy()
    f["userid"] = f["userid"].astype(str)
    f.sort_values(["userid","end_interval"], inplace=True)
    td = td.sort_values(["td_userid","td_time"]).copy()
    tol = pd.Timedelta(minutes=tolerance_minutes)
    outs=[]
    for uid, grp in f.groupby("userid"):
        td_u = td[td["td_userid"]==uid]
        if td_u.empty: continue
        m = pd.merge_asof(
            grp.sort_values("end_interval"),
            td_u[["td_time","A6a"]].sort_values("td_time"),
            left_on="end_interval", right_on="td_time",
            direction="backward", allow_exact_matches=True, tolerance=tol
        )
        outs.append(m)
    if not outs:
        raise RuntimeError("No aligned rows; increase tolerance or check user IDs.")
    aligned = pd.concat(outs, ignore_index=True)
    aligned = aligned[aligned["A6a"].notna()].copy()
    aligned["A6a"] = aligned["A6a"].astype("Int64")
    return aligned

def load_and_align_site(site: str, country: str, continent: str,
                        feat_path: str, td_path: str,
                        tolerance_minutes: int,
                        td_time_col: Optional[str]=None, td_user_col: Optional[str]=None) -> pd.DataFrame:
    feat = load_features(feat_path)
    td   = load_timediaries(td_path, td_time_col, td_user_col, feat["userid"])
    df   = align_features_with_mood(feat, td, tolerance_minutes)
    df["site"] = site
    df["country"] = country
    df["continent"] = continent
    # prefix user ids with site to avoid collisions
    df["userid"] = (df["site"].astype(str) + ":" + df["userid"].astype(str)).astype(str)
    return df

# Wrapper for joblib parallel site load
def _load_site_entry(entry: Dict, tolerance_minutes: int) -> pd.DataFrame:
    return load_and_align_site(
        entry["site"], entry["country"], entry["continent"],
        entry["features"], entry["timediaries"],
        tolerance_minutes
    )

# ------------------------ models & helpers ------------------------

def make_rf_pipeline(k_impute=5):
    # n_jobs=1 to avoid nested parallelism; we parallelize at task level
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=k_impute)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=1))
    ])

def make_svc_pipeline(k_impute=5):
    # SVC has no n_jobs; BLAS threads already constrained by env vars above
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=k_impute)),
        ("scale", StandardScaler(with_mean=True)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"))
    ])

def ensure_proba_columns(y_score: np.ndarray, trained_classes: np.ndarray, full_labels: List[int]) -> np.ndarray:
    trained_classes = np.asarray(trained_classes).astype(int)
    idx_map = {c:i for i,c in enumerate(trained_classes)}
    out = np.zeros((y_score.shape[0], len(full_labels)), dtype=float)
    for j, c in enumerate(full_labels):
        if c in idx_map:
            out[:, j] = y_score[:, idx_map[c]]
        else:
            out[:, j] = 0.0
    rowsum = out.sum(axis=1, keepdims=True)
    zero_rows = rowsum.squeeze() == 0
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / len(full_labels)
    return out

# ---------------------- UG profiles & deterministic folds ----------------------

def user_class_profile(y: np.ndarray, users: np.ndarray, classes: Iterable[int]) -> pd.DataFrame:
    df = pd.DataFrame({"user": users, "y": y})
    prof = df.groupby(["user","y"]).size().unstack(fill_value=0)
    prof = prof.reindex(columns=sorted(classes), fill_value=0)
    prof["total"] = prof.sum(axis=1)
    return prof.reset_index()

def build_disjoint_folds_with_constraints(
    prof_ug: pd.DataFrame,
    folds: int,
    test_min_classes: int,
    classes_all: Iterable[int]
) -> List[np.ndarray]:
    users = prof_ug["user"].tolist()
    class_cols = [c for c in prof_ug.columns if isinstance(c, (int, np.integer))]
    class_totals = prof_ug[class_cols].sum(axis=0).sort_values().index.tolist()
    fold_users = [set() for _ in range(folds)]
    fold_cov   = [set() for _ in range(folds)]
    target_fold_size = int(round(len(users) / folds))
    assigned = set()

    # assign carriers of rarer classes first, round-robin
    for cls in class_totals:
        carriers = prof_ug.loc[prof_ug[cls] > 0, "user"].tolist()
        carriers = [u for u in carriers if u not in assigned]
        multicover = prof_ug.set_index("user").loc[carriers, class_cols].gt(0).sum(axis=1)
        carriers = [u for u,_ in sorted(zip(carriers, multicover), key=lambda t: (-t[1], str(t[0])))]
        f = 0
        for u in carriers:
            best = None
            for off in range(folds):
                k = (f + off) % folds
                if len(fold_users[k]) >= target_fold_size: 
                    continue
                best = k
                break
            if best is None:
                sizes = [len(s) for s in fold_users]
                best = int(np.argmin(sizes))
            fold_users[best].add(u)
            cov_here = {c for c in class_cols if prof_ug.loc[prof_ug["user"]==u, c].values[0] > 0}
            fold_cov[best].update(cov_here)
            assigned.add(u)
            f = (best + 1) % folds

    remaining = [u for u in users if u not in assigned]
    for u in remaining:
        sizes = np.array([len(s) for s in fold_users])
        candidates = np.where(sizes == sizes.min())[0].tolist()
        u_cov = {c for c in class_cols if prof_ug.loc[prof_ug["user"]==u, c].values[0] > 0}
        gains = [len((fold_cov[k] | u_cov)) - len(fold_cov[k]) for k in candidates]
        best = candidates[int(np.argmax(gains))]
        fold_users[best].add(u)
        fold_cov[best].update(u_cov)

    def fold_ok(k: int) -> bool:
        return len([c for c in class_cols if (prof_ug.loc[prof_ug["user"].isin(list(fold_users[k])), c].sum() > 0)]) >= test_min_classes

    for k in range(folds):
        if fold_ok(k):
            continue
        for kk in range(folds):
            if k == kk: continue
            swapped = False
            for u2 in list(fold_users[kk]):
                u2_cov = set([c for c in class_cols if prof_ug.loc[prof_ug["user"]==u2, c].values[0] > 0])
                for u1 in list(fold_users[k]):
                    u1_cov = set([c for c in class_cols if prof_ug.loc[prof_ug["user"]==u1, c].values[0] > 0])
                    new_k_users  = (fold_users[k]  - {u1}) | {u2}
                    new_kk_users = (fold_users[kk] - {u2}) | {u1}
                    def classes_of(us):
                        sub = prof_ug[prof_ug["user"].isin(list(us))]
                        return set([c for c in class_cols if sub[c].sum() > 0])
                    if len(classes_of(new_k_users))  >= test_min_classes and \
                       len(classes_of(new_kk_users)) >= test_min_classes:
                        fold_users[k].remove(u1); fold_users[k].add(u2)
                        fold_users[kk].remove(u2); fold_users[kk].add(u1)
                        swapped = True
                        break
                if swapped: break
        if not fold_ok(k):
            raise RuntimeError("Deterministic fold builder could not enforce class diversity. Reduce 'folds' or relax constraints.")

    return [np.array(sorted(list(s))) for s in fold_users]

# ---------------------- transfer evaluation (parallel tasks) ----------------------

def _eval_task(
    te_users: np.ndarray,
    df_src_pool: pd.DataFrame,
    df_ug: pd.DataFrame,
    feature_cols_intersection: List[str],
    pipe_plm,  # pre-trained pooled PLM model
    y_src_vec: np.ndarray,
    y_ug: np.ndarray,
    trained_classes_plm: np.ndarray,
    full_labels: List[int],
    multiclass: bool,
    seed: int
) -> Tuple[float, float]:
    """
    One (repeat, fold) evaluation task.
    Returns (plm_auc, hm_auc).
    """
    rng = np.random.RandomState(seed)
    ug_user_series = df_ug["userid"].astype(str)
    te_idx = np.where(ug_user_series.isin(te_users))[0]

    # ---------- PLM ----------
    X_te = df_ug[feature_cols_intersection].iloc[te_idx]
    y_te = y_ug[te_idx]
    proba = pipe_plm.predict_proba(X_te)
    proba = ensure_proba_columns(proba, trained_classes_plm, full_labels)
    if multiclass:
        plm_auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=full_labels)
    else:
        plm_auc = roc_auc_score(y_te, proba[:,1])

    # ---------- HM ----------
    hm_add = []
    for u in te_users:
        u_idx = np.where(ug_user_series.values == u)[0]
        u_idx = np.intersect1d(u_idx, te_idx, assume_unique=False)
        u_loc = u_idx.copy()
        rng.shuffle(u_loc)
        half = int(math.floor(len(u_loc) * 0.5))
        hm_add.extend(u_loc[:half].tolist())
    hm_add = np.array(sorted(hm_add))

    X_hm_train = pd.concat([df_src_pool[feature_cols_intersection],
                            df_ug[feature_cols_intersection].iloc[hm_add]], axis=0)
    y_hm_train = np.concatenate([y_src_vec, y_ug[hm_add]], axis=0)

    if len(y_hm_train) > len(y_src_vec):
        keep_idx, _, _, _ = train_test_split(
            np.arange(len(y_hm_train)), y_hm_train,
            train_size=len(y_src_vec), stratify=y_hm_train,
            random_state=int(rng.randint(0, 2**31-1))
        )
        X_hm_train = X_hm_train.iloc[keep_idx]
        y_hm_train = y_hm_train[keep_idx]

    # Train HM model (RF n_jobs=1; threads controlled by env)
    model_maker = make_rf_pipeline  # RF primary
    pipe_hm = model_maker()
    pipe_hm.fit(X_hm_train, y_hm_train)
    trained_classes_hm = pipe_hm.named_steps["clf"].classes_.astype(int)

    # evaluate on remaining half
    hm_te = []
    for u in te_users:
        u_idx = np.where(ug_user_series.values == u)[0]
        u_idx = np.intersect1d(u_idx, te_idx, assume_unique=False)
        rng.shuffle(u_idx)
        half = int(math.floor(len(u_idx) * 0.5))
        hm_te.extend(u_idx[half:].tolist())
    hm_te = np.array(sorted(hm_te))

    X_te_hm = df_ug[feature_cols_intersection].iloc[hm_te]
    y_te_hm = y_ug[hm_te]
    proba_hm = pipe_hm.predict_proba(X_te_hm)
    proba_hm = ensure_proba_columns(proba_hm, trained_classes_hm, full_labels)
    if multiclass:
        hm_auc = roc_auc_score(y_te_hm, proba_hm, multi_class="ovr", average="macro", labels=full_labels)
    else:
        hm_auc = roc_auc_score(y_te_hm, proba_hm[:,1])

    return float(plm_auc), float(hm_auc)

def run_transfer_II_pooled_parallel(
    df_src_pool: pd.DataFrame, df_ug: pd.DataFrame,
    feature_cols_intersection: List[str],
    folds: int, repeats: int,
    multiclass: bool,
    rng_seed: int = 42
) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Parallel task runner: builds deterministic UG folds once, trains pooled PLM once,
    then evaluates (repeat, fold) tasks in parallel using N_WORKERS.
    """
    rng = np.random.RandomState(rng_seed)

    # Labels and constraints
    if multiclass:
        y_src = df_src_pool["y_three"].values
        y_ug  = df_ug["y_three"].values
        classes_all = (0,1,2)
        test_min_classes = 2   # OVR AUROC requires ≥2 classes in y_true
        full_labels = [0,1,2]
    else:
        y_src = df_src_pool["y_bin"].values
        y_ug  = df_ug["y_bin"].values
        classes_all = (0,1)
        test_min_classes = 2   # binary AUROC requires both classes in y_true
        full_labels = [0,1]

    if len(np.unique(y_src)) < 2:
        raise RuntimeError("Pooled source has <2 classes for this task; cannot train a classifier.")

    # UG folds with deterministic, disjoint construction
    prof_ug = user_class_profile(y_ug, df_ug["userid"].values, classes_all)
    fold_user_sets = build_disjoint_folds_with_constraints(
        prof_ug, folds=folds, test_min_classes=test_min_classes, classes_all=classes_all
    )

    # Train pooled PLM model once (RF n_jobs=1)
    X_src = df_src_pool[feature_cols_intersection]; y_src_vec = y_src
    pipe_plm = make_rf_pipeline(); pipe_plm.fit(X_src, y_src_vec)
    trained_classes_plm = pipe_plm.named_steps["clf"].classes_.astype(int)

    # Build task list: one per (repeat, fold)
    tasks = []
    for rep in range(repeats):
        order = np.arange(len(fold_user_sets)); rng.shuffle(order)
        for idx in order:
            te_users = fold_user_sets[idx]
            # seed per task so splits are independent/reproducible
            seed = int(rng.randint(0, 2**31-1))
            tasks.append((te_users, seed))

    # Execute tasks in parallel
    results = Parallel(n_jobs=N_WORKERS, backend="loky", prefer="processes")(
        delayed(_eval_task)(
            te_users,
            df_src_pool, df_ug,
            feature_cols_intersection,
            pipe_plm,
            y_src_vec, y_ug,
            trained_classes_plm,
            full_labels,
            multiclass,
            seed
        )
        for (te_users, seed) in tasks
    )

    plm_scores = [r[0] for r in results]
    hm_scores  = [r[1] for r in results]

    return (float(np.mean(plm_scores)), float(np.std(plm_scores))), \
           (float(np.mean(hm_scores)),  float(np.std(hm_scores)))

# ----------------------------- main -----------------------------

def main():
    # Sites 
    sites = [
        {"site": "LuisPotosi", "country": "Mexico", "continent": "Latin America",
         "features": "../data/new_data/LuisPotosi/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/LuisPotosi/timediaries.parquet"},
        {"site": "Amrita", "country": "India", "continent": "Asia",
         "features": "../data/new_data/Amrita/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Amrita/timediaries.parquet"},
        {"site": "Asuncion", "country": "Paraguay", "continent": "Latin America",
         "features": "../data/new_data/Asuncion/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Asuncion/timediaries.parquet"},
        {"site": "Copenhagen", "country": "Denmark", "continent": "Europe",
         "features": "../data/new_data/Copenhagen/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Copenhagen/timediaries.parquet"},
        {"site": "Jilin", "country": "China", "continent": "Asia",
         "features": "../data/new_data/Jilin/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Jilin/timediaries.parquet"},
        {"site": "Ulaanbaatar", "country": "Mongolia", "continent": "Asia",
         "features": "../data/new_data/Ulaanbaatar/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Ulaanbataar/timediaries.parquet"},
        {"site": "Mak", "country": "Uganda", "continent": "Africa",
         "features": "../data/new_data/Mak/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Mak/timediaries.parquet"},
        {"site": "London", "country": "UnitedKingdom", "continent": "Europe",
         "features": "../data/new_data/London/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/London/timediaries.parquet"},
        {"site": "Trento", "country": "Italy", "continent": "Europe",
         "features": "../data/new_data/Trento/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Trento/timediaries.parquet"}
    ]

    outdir = "../results"
    os.makedirs(outdir, exist_ok=True)

    tolerance_minutes = 60
    folds = 2
    repeats = 5
    target_country = "Uganda"

    # ---------------- Parallel site loading ----------------
    print(f"=== Loading & aligning all sites (parallel, n_jobs={N_WORKERS}) ===")
    frames = Parallel(n_jobs=N_WORKERS, backend="loky", prefer="processes")(
        delayed(_load_site_entry)(s, tolerance_minutes) for s in sites
    )
    for s, df in zip(sites, frames):
        print(f"--- Loaded {s['site']} ({s['country']}/{s['continent']}) | "
              f"aligned={len(df)} users={df['userid'].nunique()} A6a={df['A6a'].value_counts(dropna=False).to_dict()}")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"Total rows={len(df_all)} users={df_all['userid'].nunique()} countries={df_all['country'].nunique()}")

    # Labels (single assign to avoid fragmentation warnings)
    a6 = df_all["A6a"].astype("int64")
    df_all = df_all.assign(
        valence5 = a6,
        y_bin    = map_binary(a6.values),
        y_three  = map_three(a6.values),
    )

    # Partition target vs pooled sources
    target_mask = df_all["country"].astype(str).str.upper() == target_country.upper()
    if not target_mask.any():
        raise ValueError(f"Target country {target_country} not found in data.")
    df_ug = df_all[target_mask].copy()
    df_src_pool = df_all[~target_mask].copy()

    # Feature-space harmonization: numeric features common to both pools
    key_cols = {"userid","experimentid","start_interval","end_interval","A6a","td_time",
                "valence5","y_bin","y_three","site","country","continent"}
    numeric_cols = [c for c in df_all.columns if c not in key_cols and pd.api.types.is_numeric_dtype(df_all[c])]
    feat_src = [c for c in numeric_cols if c in df_src_pool.columns]
    feat_tgt = [c for c in numeric_cols if c in df_ug.columns]
    feat_inter = sorted(list(set(feat_src).intersection(set(feat_tgt))))
    if len(feat_inter) == 0:
        raise RuntimeError("No shared numeric features between pooled source and UG.")
    print(f"Using {len(feat_inter)} shared features.")

    results = {}

    # ---------------- Binary ----------------
    try:
        print("\n=== Binary (neg vs non-neg) — parallel evaluation ===")
        print("Pooled Source Binary counts:", class_counts(df_src_pool["y_bin"].values))
        print("Target Binary counts:", class_counts(df_ug["y_bin"].values))
        (plm_bin_m, plm_bin_s), (hm_bin_m, hm_bin_s) = run_transfer_II_pooled_parallel(
            df_src_pool, df_ug, feat_inter,
            folds=folds, repeats=repeats, multiclass=False
        )
        results["binary"] = {"plm": {"mean": plm_bin_m, "std": plm_bin_s},
                             "hm":  {"mean": hm_bin_m,  "std": hm_bin_s}}
        print(f"PLM AUROC: {plm_bin_m:.3f} ± {plm_bin_s:.3f}")
        print(f"HM  AUROC: {hm_bin_m:.3f} ± {hm_bin_s:.3f}")
    except Exception as e:
        print(f"[ERROR] Binary (ALL-{target_country})→{target_country}: {e}")
        results["binary"] = {"error": str(e)}

    # ---------------- Three-class ----------------
    try:
        print("\n=== Three-class (neg/neutral/pos) — parallel evaluation ===")
        print("Pooled Source Three-class counts:", class_counts(df_src_pool["y_three"].values))
        print("Target Three-class counts:", class_counts(df_ug["y_three"].values))
        (plm_3_m, plm_3_s), (hm_3_m, hm_3_s) = run_transfer_II_pooled_parallel(
            df_src_pool, df_ug, feat_inter,
            folds=folds, repeats=repeats, multiclass=True
        )
        results["three_class"] = {"plm": {"mean": plm_3_m, "std": plm_3_s},
                                  "hm":  {"mean": hm_3_m,  "std": hm_3_s}}
        print(f"PLM macro-AUROC: {plm_3_m:.3f} ± {plm_3_s:.3f}")
        print(f"HM  macro-AUROC: {hm_3_m:.3f} ± {hm_3_s:.3f}")
    except Exception as e:
        print(f"[ERROR] Three-class (ALL-{target_country})→{target_country}: {e}")
        results["three_class"] = {"error": str(e)}

    # Save results
    out_path = os.path.join(outdir, f"transfer_II_ALL_except_{target_country.lower().replace(' ', '_')}_to_{target_country}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_path}")

if __name__ == "__main__":
    main()
