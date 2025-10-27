# experiment_country_specific_fixed.py
# Country-specific mood inference experiments for multiple sites
# PLM vs HM per the paper, kNN-impute, RF primary.

import argparse
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Iterable
from sklearn.model_selection import train_test_split 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# ----------------------------- parsing utils ---------------------------------

def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def pick_best_datetime_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if any(tok in c.lower() for tok in [
        "instancetimestamp","time","timestamp","date","datetime","start","end","submitted"
    ])]
    best_col, best_nonnull = None, -1
    for c in candidates:
        dt = parse_dt(df[c])
        nonnull = dt.notna().sum()
        if nonnull > best_nonnull:
            best_col, best_nonnull = c, nonnull
    if best_col is None:
        raise ValueError("No usable datetime-like column found in timediaries. "
                         "Use --timediaries_time_col to override.")
    return best_col

def pick_user_id_column(td: pd.DataFrame, feature_user_ids: pd.Series) -> str:
    commons = ["userid","user_id","participantid","participant_id","subject","uid"]
    feat_u = set(feature_user_ids.astype(str).unique())
    for c in commons:
        if c in td.columns:
            u = set(td[c].astype(str).unique())
            if len(u & feat_u) > 0:
                return c
    # fall back: any overlapping column
    for c in td.columns:
        try:
            u = set(td[c].astype(str).unique())
            if len(u & feat_u) > 0:
                return c
        except Exception:
            continue
    raise ValueError("No user id column in timediaries matches 'userid' in features. "
                     "Use --timediaries_user_col to override.")

def map_binary(y5: np.ndarray) -> np.ndarray:
    # 1,2 -> 0 (negative); 3,4,5 -> 1 (non-negative)
    return (y5 >= 3).astype(int)

def map_three(y5: np.ndarray) -> np.ndarray:
    # 1,2 -> 0 (neg); 3 -> 1 (neutral); 4,5 -> 2 (pos)
    y3 = np.where(y5 <= 2, 0, np.where(y5 == 3, 1, 2))
    return y3.astype(int)

def class_counts(arr: np.ndarray) -> Dict[int,int]:
    vals, cnts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}

# --------------------------- data loading -------------------------------------

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "userid" not in df.columns:
        raise ValueError("Expected 'userid' in features file.")
    if "start_interval" not in df.columns or "end_interval" not in df.columns:
        raise ValueError("Expected 'start_interval' and 'end_interval'.")
    df["start_interval"] = parse_dt(df["start_interval"])
    df["start_interval"] = df["start_interval"].dt.tz_localize(None)
    df["end_interval"]   = parse_dt(df["end_interval"])
    df["end_interval"] = df["end_interval"].dt.tz_localize(None) 
    return df

def load_timediaries(path: str, explicit_time_col: Optional[str], explicit_user_col: Optional[str],
                     feature_user_ids: pd.Series) -> Tuple[pd.DataFrame, str, str]:
    if os.path.isdir(path):
        parts = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        if not parts: 
            raise FileNotFoundError(f"No .parquet files inside folder: {path}")
        path = parts[0]
    try:
        td = pd.read_parquet(path)  
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet ({e}). Install pyarrow or fastparquet:\n"
            "  pip install pyarrow\n"
            "or\n"
            "  pip install fastparquet"
        )
    if "A6a" not in td.columns:
        raise ValueError("Timediaries missing 'A6a' column (mood).")
    time_col = explicit_time_col or pick_best_datetime_column(td)
    user_col = explicit_user_col or pick_user_id_column(td, feature_user_ids)
    td = td[[user_col, time_col, "A6a"]].copy()
    td.rename(columns={user_col: "td_userid", time_col: "td_time"}, inplace=True)
    td["td_time"] = parse_dt(td["td_time"])
    td["td_time"] = td["td_time"].dt.tz_localize(None) 
    td = td[td["A6a"].notna()].copy()
    td["td_userid"] = td["td_userid"].astype(str)
    return td, "td_userid", "td_time"

def align_features_with_mood(features: pd.DataFrame, td: pd.DataFrame, tolerance_minutes: int = 60) -> pd.DataFrame:
    f = features.copy()
    f["userid"] = f["userid"].astype(str)
    f.sort_values(["userid", "end_interval"], inplace=True)
    td = td.sort_values(["td_userid", "td_time"]).copy()
    out_frames = []
    tol = pd.Timedelta(minutes=tolerance_minutes)
    for uid, grp in f.groupby("userid"):
        td_u = td[td["td_userid"] == uid]
        if td_u.empty:
            continue
        merged = pd.merge_asof(
            grp.sort_values("end_interval"),
            td_u[["td_time", "A6a"]].sort_values("td_time"),
            left_on="end_interval",
            right_on="td_time",
            direction="backward",
            allow_exact_matches=True,
            tolerance=tol
        )
        out_frames.append(merged)
    if not out_frames:
        raise RuntimeError("No timediary entries matched any feature intervals. "
                           "Increase --tolerance_minutes or check user IDs.")
    aligned = pd.concat(out_frames, ignore_index=True)
    aligned = aligned[aligned["A6a"].notna()].copy()
    aligned["A6a"] = aligned["A6a"].astype("Int64")
    return aligned

# --------------------------- modeling utils -----------------------------------

def make_rf_pipeline(k_impute=5):
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=k_impute)),
        ("clf", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])

def make_svc_pipeline(k_impute=5):
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("scale", StandardScaler(with_mean=True)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"))
    ])

# ------------------------ stratified group splitting --------------------------

def user_class_profile(y: np.ndarray, users: np.ndarray, classes: Iterable[int]) -> pd.DataFrame:
    """Per-user counts of each class in y."""
    df = pd.DataFrame({"user": users, "y": y})
    prof = df.groupby(["user", "y"]).size().unstack(fill_value=0)
    # Ensure all classes columns exist
    prof = prof.reindex(columns=sorted(classes), fill_value=0)
    prof["total"] = prof.sum(axis=1)
    return prof.reset_index() 

def sample_users_with_class_constraints(
    prof: pd.DataFrame,
    test_user_frac: float,
    required_classes_test: Iterable[int],
    required_classes_train: Iterable[int],
    max_tries: int = 1000,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample a set of test users (~fraction of unique users) such that:
      - Test contains at least one sample from each class in required_classes_test
      - Train contains at least one sample from each class in required_classes_train
    """
    rng = rng or np.random.RandomState(42)
    users = prof["user"].values
    n_users = len(users)
    n_test = max(1, int(round(test_user_frac * n_users)))
    class_cols = [c for c in prof.columns if isinstance(c, (int, np.integer))]

    for _ in range(max_tries):
        te_users = rng.choice(users, size=n_test, replace=False)
        tr_users = np.setdiff1d(users, te_users)
        te_mask = prof["user"].isin(te_users)
        tr_mask = prof["user"].isin(tr_users)

        te_counts = prof.loc[te_mask, class_cols].sum().to_dict()
        tr_counts = prof.loc[tr_mask, class_cols].sum().to_dict()

        ok_test  = all(te_counts.get(c,0) > 0 for c in required_classes_test)
        ok_train = all(tr_counts.get(c,0) > 0 for c in required_classes_train)
        if ok_test and ok_train:
            return te_users, tr_users
    raise RuntimeError("Unable to find a valid user split that satisfies class constraints. "
                       "Try increasing data, lowering constraints, or reducing n_folds.")

def build_index_from_user_sets(all_users: np.ndarray, users_series: pd.Series,
                               te_users: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    te_idx = np.where(users_series.isin(te_users))[0]
    tr_idx = np.where(~users_series.isin(te_users))[0]
    return tr_idx, te_idx

# --------------------------- PLM/HM runners -----------------------------------

def run_country_specific_with_constraints(
    df: pd.DataFrame,
    feature_cols: List[str],
    labels: np.ndarray,
    users: np.ndarray,
    model_pipeline_maker,
    repeats: int = 5,
    folds: int = 5,
    classes_all: Iterable[int] = (0,1),
    require_all_classes_in_test: bool = True,
    hm_half_fraction: float = 0.5,
    random_state: int = 42,
    multiclass: bool = False
) -> Dict[str, Tuple[float,float]]:
    """
    Repeated constrained group splits (~80/20 by users), ensuring label presence in train/test.
    Returns mean±sd AUROC for PLM and HM.
    """
    rng = np.random.RandomState(random_state)
    users_series = pd.Series(users)
    prof = user_class_profile(labels, users, classes_all)
    # Temporary diagnostic: print user class profile
    mode = "Three-class" if multiclass else "Binary"
    #print(f"{mode} user class profile (counts per user):")
    #print(prof.to_string(index=False))
    rare_class = classes_all[-1]
    #print(f"Users with class {rare_class} (>0): {sum(prof[rare_class] > 0)}")
    #print(f"Total class {rare_class} samples: {prof[rare_class].sum()}")

    test_frac = 1.0 / folds
    plm_scores, hm_scores = [], []

    # Required classes in test/train
    if require_all_classes_in_test:
        req_test = tuple(classes_all)
    else:
        req_test = (classes_all[0],)  # require only majority class in test (class 0)
    req_train = tuple(classes_all)

    for rep in range(repeats):
        used_test_sets = set()
        for _fold in range(folds):
            # sample a valid test user set, avoid duplicates in the same rep
            for _try in range(1000):
                te_users, tr_users = sample_users_with_class_constraints(
                    prof, test_frac, req_test, req_train, rng=rng
                )
                key = tuple(sorted(te_users.tolist()))
                if key not in used_test_sets:
                    used_test_sets.add(key)
                    break
            tr_idx, te_idx = build_index_from_user_sets(
                prof["user"].values, users_series, te_users
            )

            # ---- PLM ----
            X_tr, y_tr = df[feature_cols].iloc[tr_idx], labels[tr_idx]
            X_te, y_te = df[feature_cols].iloc[te_idx], labels[te_idx]
            pipe = model_pipeline_maker()
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)

            if multiclass:
                auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
            else:
                auc = roc_auc_score(y_te, proba[:,1])
            plm_scores.append(auc)

            # ---- HM ----
            # for each test user, move half records to train; keep other half for test
            hm_tr = set(tr_idx.tolist())
            hm_te = []
            for u in te_users:
                u_idx = np.where(users_series.values == u)[0]
                # of those in test set only:
                u_idx = np.intersect1d(u_idx, te_idx, assume_unique=False)
                rng.shuffle(u_idx)
                half = int(math.floor(len(u_idx) * hm_half_fraction))
                hm_tr.update(u_idx[:half].tolist())
                hm_te.extend(u_idx[half:].tolist())
            hm_tr = np.array(sorted(list(hm_tr)))
            hm_te = np.array(sorted(hm_te))

            # Stratified downsample to preserve classes
            hm_tr_size = len(hm_tr)
            orig_tr_size = len(tr_idx)
            if hm_tr_size > orig_tr_size:
                # Use train_test_split to keep orig_tr_size stratified samples
                keep_mask, _, _, _ = train_test_split(
                    np.arange(hm_tr_size),
                    labels[hm_tr],  # Stratify on y (labels is y_three or y_bin)
                    train_size=orig_tr_size,
                    stratify=labels[hm_tr],
                    random_state=rng.randint(0, 2**31 - 1)
                )
                hm_tr = hm_tr[keep_mask]

            X_tr_hm, y_tr_hm = df[feature_cols].iloc[hm_tr], labels[hm_tr]
            X_te_hm, y_te_hm = df[feature_cols].iloc[hm_te], labels[hm_te]
            pipe_hm = model_pipeline_maker()
            pipe_hm.fit(X_tr_hm, y_tr_hm)
            proba_hm = pipe_hm.predict_proba(X_te_hm)

            if multiclass:
                auc_hm = roc_auc_score(y_te_hm, proba_hm, multi_class="ovr", average="macro")
            else:
                auc_hm = roc_auc_score(y_te_hm, proba_hm[:,1])
            hm_scores.append(auc_hm)

    key = "three" if multiclass else "bin"
    return {
        f"plm_{key}": (float(np.mean(plm_scores)), float(np.std(plm_scores))),
        f"hm_{key}":  (float(np.mean(hm_scores)),  float(np.std(hm_scores))),
    }

# ------------------------------- main -----------------------------------------

def process_site(site_name, features_path, timediaries_path, outdir, tolerance_minutes=60, folds=3, repeats=5, model="rf", timediaries_time_col=None, timediaries_user_col=None):
    os.makedirs(outdir, exist_ok=True)

    print(f"\n\n=== Processing site: {site_name} ===\n")

    print(f"=== Loading features from: {features_path}")
    feat = load_features(features_path)
    # Make booleans numeric for KNNImputer
    bool_cols = feat.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        feat[bool_cols] = feat[bool_cols].astype(int)

    print(f"Features shape: {feat.shape[0]} rows, {feat.shape[1]} cols")
    print(f"Feature sample cols: {feat.columns[:12].tolist()}")

    print(f"\n=== Loading timediaries from: {timediaries_path}")
    td, td_user_col, td_time_col = load_timediaries(
        timediaries_path, timediaries_time_col, timediaries_user_col, feat["userid"]
    )
    print(f"Timediaries (non-null A6a): {td.shape}; time={td_time_col}; user={td_user_col}")
    print(f"A6a uniques: {pd.array(td['A6a']).unique().tolist()}")

    print("\n=== Aligning (asof on end_interval per user) ...")
    aligned = align_features_with_mood(feat, td, tolerance_minutes=tolerance_minutes)
    print(f"Aligned shape: {aligned.shape}")
    print(f"A6a after join: {pd.array(aligned['A6a']).unique().tolist()}")

    aligned = aligned.copy()
    aligned["valence5"] = aligned["A6a"].astype("int64")
    aligned["y_bin"]    = map_binary(aligned["valence5"].values)
    aligned["y_three"]  = map_three(aligned["valence5"].values)

    n_users = aligned["userid"].astype(str).nunique()
    print(f"\n=== {site_name} country-specific dataset ===")
    print(f"Users: {n_users} | Samples: {len(aligned)}")
    print("Valence5 counts:", aligned["valence5"].value_counts(dropna=False).to_dict())
    print("Binary counts (0,1):", class_counts(aligned["y_bin"].values))
    print("Three-class counts (0,1,2):", class_counts(aligned["y_three"].values))

    key_cols = {"userid","experimentid","start_interval","end_interval","A6a","td_time","valence5","y_bin","y_three"}
    feature_cols = [c for c in aligned.columns if c not in key_cols and pd.api.types.is_numeric_dtype(aligned[c])]
    if len(feature_cols) == 0:
        raise ValueError("No numeric features found for modeling.")
    print(f"\nUsing {len(feature_cols)} numeric features.")

    model_maker = make_rf_pipeline if model == "rf" else make_svc_pipeline
    users = aligned["userid"].astype(str).values

    # ---------------- Binary ----------------
    print("\n=== Training: Binary (neg vs non-neg) with stratified user splits ===")
    res_bin = run_country_specific_with_constraints(
        aligned, feature_cols, aligned["y_bin"].values, users, model_maker,
        repeats=repeats, folds=folds,
        classes_all=(0,1),
        require_all_classes_in_test=False,
        multiclass=False
    )
    print(f"PLM (bin) AUROC: {res_bin['plm_bin'][0]:.3f} ± {res_bin['plm_bin'][1]:.3f}")
    print(f"HM  (bin) AUROC: {res_bin['hm_bin'][0]:.3f} ± {res_bin['hm_bin'][1]:.3f}")

    # ---------------- Three-class ----------------
    print("\n=== Training: Three-class (neg/neutral/pos) with stratified user splits ===")
    res_three = run_country_specific_with_constraints(
        aligned, feature_cols, aligned["y_three"].values, users, model_maker,
        repeats=repeats, folds=folds,
        classes_all=(0,1,2),
        require_all_classes_in_test=False,
        multiclass=True
    )
    print(f"PLM (3-class) macro-AUROC: {res_three['plm_three'][0]:.3f} ± {res_three['plm_three'][1]:.3f}")
    print(f"HM  (3-class) macro-AUROC: {res_three['hm_three'][0]:.3f} ± {res_three['hm_three'][1]:.3f}")

    # Save results
    out = {
        "site": site_name,
        "model": model,
        "folds": folds,
        "repeats": repeats,
        "tolerance_minutes": tolerance_minutes,
        **{k: {"mean": float(v[0]), "std": float(v[1])}
           for k, v in {**res_bin, **res_three}.items()}
    }
    out_path = os.path.join(outdir, f"{site_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_country_specific_results.json")
    pd.Series(out).to_json(out_path, indent=2)
    print(f"\nSaved results JSON → {out_path}")

def main():
    # Define all sites with their paths 
    '''
        {
            "site_name": "LuisPotosi (Mexico)",
            "features": "../data/new_data/LuisPotosi/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/LuisPotosi/timediaries.xlsx",
            "outdir": "../results"
        },
        {
            "site_name": "Amrita (India)",
            "features": "../data/new_data/Amrita/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/Amrita/timediaries.xlsx",
            "outdir": "../results"
        },
        {
            "site_name": "Asuncion (Paraguay)",
            "features": "../data/new_data/Asuncion/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/Asuncion/timediaries.xlsx",
            "outdir": "../results"
        },
        {
            "site_name": "Copenhagen (Denmark)",
            "features": "../data/new_data/Copenhagen/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/Copenhagen/timediaries.xlsx",
            "outdir": "../results"
        },
        {
            "site_name": "Jilin (China)",
            "features": "../data/new_data/Jilin/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/Jilin/timediaries.xlsx",
            "outdir": "../results"
        }, 
        {
            "site_name": "Ulaanbaatar (Mongolia)",
            "features": "../data/new_data/Ulaanbaatar/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/Ulaanbataar/timediaries.xlsx",  
            "outdir": "../results"
        },
        {
            "site_name": "Mak (Uganda)",
            "features": "../data/new_data/Mak/processed/joined_features.csv",
            "timediaries": "../data/Timediaries/Mak/timediaries.parquet",
            "outdir": "../results"
        },
                {
            "site_name": "London (United Kingdom)",
            "features": "../data/new_data/London/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/London/timediaries.xlsx",
            "outdir": "../results"
        }

    '''
    sites = [
        {
            "site_name": "LuisPotosi (Mexico)",
            "features": "../data/new_data/LuisPotosi/processed/joined_features.csv",
            "timediaries": "../data/Timediaries/LuisPotosi/timediaries.parquet",
            "outdir": "../results"
        },
        {
            "site_name": "Trento (Italy)",
            "features": "../data/new_data/Trento/processed/joined_features.csv",
            "timediaries": "../data/Timediaries/Trento/timediaries.parquet",
            "outdir": "../results"
        },
        { 
            "site_name": "Ulaanbaatar (Mongolia)",
            "features": "../data/new_data/Ulaanbaatar/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/Ulaanbataar/timediaries.xlsx",  
            "outdir": "../results"
        },
        {
            "site_name": "London (United Kingdom)",
            "features": "../data/new_data/London/processed/joined_features.csv",
            "timediaries": "../data/Timediaries_new/London/timediaries.xlsx",
            "outdir": "../results"
        }
    ]

    # Common parameters 
    tolerance_minutes = 60
    folds = 3  
    repeats = 5  
    model = "rf"

    for site in sites:
        try:
            process_site(
                site["site_name"],
                site["features"],
                site["timediaries"],
                site["outdir"],
                tolerance_minutes=tolerance_minutes,
                folds=folds,
                repeats=repeats,
                model=model
            )
        except Exception as e:
            print(f"Error processing {site['site_name']}: {str(e)}")

if __name__ == "__main__":
    main()