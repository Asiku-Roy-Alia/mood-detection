# experiment_transfer_country_agnostic_I.py
# Country-Agnostic I: Train on each single foreign country, test on Uganda.
# Outputs PLM  and HM AUROC for Binary and Three-class tasks.

import os
import math
import json
from typing import Optional, List, Tuple, Iterable, Dict
import numpy as np
import pandas as pd

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
    df["start_interval"] = parse_dt(df["start_interval"])
    df["start_interval"] = df["start_interval"].dt.tz_localize(None)
    df["end_interval"]   = parse_dt(df["end_interval"])
    df["end_interval"] = df["end_interval"].dt.tz_localize(None)
    # bools -> ints for KNNImputer
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols): df[bool_cols] = df[bool_cols].astype(int)
    return df

def load_timediaries(path: str, td_time_col: Optional[str], td_user_col: Optional[str],
                     feat_userids: pd.Series) -> pd.DataFrame:
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
    td["td_time"] = parse_dt(td["td_time"])
    td["td_time"] = td["td_time"].dt.tz_localize(None)
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

# ------------------------ models & helpers ------------------------

def make_rf_pipeline(k_impute=5):
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=k_impute)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))  # Reduced for speed
    ])

def make_svc_pipeline(k_impute=5):
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=k_impute)),
        ("scale", StandardScaler(with_mean=True)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"))
    ])

def ensure_proba_columns(y_score: np.ndarray, trained_classes: np.ndarray, full_labels: List[int]) -> np.ndarray:
    """
    Re-map predict_proba outputs to fixed column order (full_labels).
    If a class is absent in training, fill its column with zeros.
    """
    trained_classes = np.asarray(trained_classes).astype(int)
    idx_map = {c:i for i,c in enumerate(trained_classes)}
    out = np.zeros((y_score.shape[0], len(full_labels)), dtype=float)
    for j, c in enumerate(full_labels):
        if c in idx_map:
            out[:, j] = y_score[:, idx_map[c]]
        else:
            out[:, j] = 0.0
    # Re-normalize row-wise to avoid degenerate zeros-only vectors
    rowsum = out.sum(axis=1, keepdims=True)
    zero_rows = rowsum.squeeze() == 0
    if np.any(zero_rows):
        # fallback to uniform for missing all-trained-classes edge case 
        out[zero_rows] = 1.0 / len(full_labels)
    return out

# ---------------------- transfer evaluation ----------------------

def user_class_profile(y: np.ndarray, users: np.ndarray, classes: Iterable[int]) -> pd.DataFrame:
    df = pd.DataFrame({"user": users, "y": y})
    prof = df.groupby(["user","y"]).size().unstack(fill_value=0)
    prof = prof.reindex(columns=sorted(classes), fill_value=0)
    prof["total"] = prof.sum(axis=1)
    return prof.reset_index()

def sample_ug_test_users(prof_ug: pd.DataFrame, folds: int, rng: np.random.RandomState,
                         test_min_classes: int = 1) -> List[np.ndarray]:
    """
    Build 'folds' *disjoint* sets of UG test users sequentially from remaining pool,
    such that each fold's test set contains at least 'test_min_classes' distinct classes.
    """
    users = prof_ug["user"].values
    n_users = len(users)
    n_test = max(1, int(round(n_users / folds)))
    class_cols = [c for c in prof_ug.columns if isinstance(c, (int, np.integer))]

    remaining_users = set(users)
    folds_sets = []
    max_tries = 10000  # Increased for safety

    for fold_idx in range(folds):
        avail_users = np.array(list(remaining_users))
        n_avail = len(avail_users)
        if n_avail == 0:
            break
        n_test_this = min(n_test, n_avail)

        found = False
        for _ in range(max_tries):
            if n_avail < test_min_classes:  # Impossible if too few left
                break
            te_users = rng.choice(avail_users, size=n_test_this, replace=False)
            te_mask = prof_ug["user"].isin(te_users)
            te_present = [c for c in class_cols if prof_ug.loc[te_mask, c].sum() > 0]
            if len(te_present) >= test_min_classes:
                folds_sets.append(te_users)
                remaining_users -= set(te_users)
                found = True
                break

        if not found:
            raise RuntimeError(f"Unable to construct fold {fold_idx+1} from remaining {n_avail} users "
                               f"with >= {test_min_classes} classes. Try reducing folds or test_min_classes.")

    if len(folds_sets) < folds:
        raise RuntimeError("Incomplete folds due to insufficient remaining users.")
    return folds_sets

def run_transfer_I_for_source(
    df_src: pd.DataFrame, df_ug: pd.DataFrame,
    feature_cols_intersection: List[str],
    model_maker,
    folds: int, repeats: int,
    multiclass: bool,
    rng_seed: int = 42
) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Train on full source country; evaluate on UG with user-folds (PLM) and HM (few-shot personalization).
    Returns (PLM_mean±sd, HM_mean±sd) AUROC.
    """
    rng = np.random.RandomState(rng_seed)

    # Labels
    if multiclass:
        y_src = df_src["y_three"].values
        y_ug  = df_ug["y_three"].values
        classes_all = (0,1,2)
        test_min_classes = 1   # multiclass OVR requires >=2 in y_true
        full_labels = [0,1,2]
    else:
        y_src = df_src["y_bin"].values
        y_ug  = df_ug["y_bin"].values
        classes_all = (0,1)
        test_min_classes = 1
        full_labels = [0,1]

    # Ensure source has at least 2 classes to train a classifier
    if len(np.unique(y_src)) < 2:
        raise RuntimeError("Source country has <2 classes for this task; cannot train a classifier.")

    # Build UG user folds with class constraints
    prof_ug = user_class_profile(y_ug, df_ug["userid"].values, classes_all)
    ug_user_series = df_ug["userid"].astype(str)
    fold_user_sets = sample_ug_test_users(prof_ug, folds=folds, rng=rng, test_min_classes=test_min_classes)

    # Train once on source (fixed) — PLM baseline
    X_src = df_src[feature_cols_intersection]; y_src_vec = y_src
    pipe_plm = model_maker(); pipe_plm.fit(X_src, y_src_vec)
    trained_classes_plm = pipe_plm.named_steps["clf"].classes_.astype(int)

    plm_scores, hm_scores = [], []

    for rep in range(repeats):
        rng.shuffle(fold_user_sets)
        for te_users in fold_user_sets:
            te_idx = np.where(ug_user_series.isin(te_users))[0]
            tr_idx = np.where(~ug_user_series.isin(te_users))[0]  
            # ---------- PLM: evaluate source-only model on this UG fold ----------
            X_te = df_ug[feature_cols_intersection].iloc[te_idx]
            y_te = y_ug[te_idx]
            proba = pipe_plm.predict_proba(X_te)
            # remap proba to fixed label set for multiclass robustness
            if multiclass:
                proba = ensure_proba_columns(proba, trained_classes_plm, full_labels)
                auc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=full_labels)
            else:
                # ensure binary has both classes in y_true; guaranteed by fold construction
                proba = ensure_proba_columns(proba, trained_classes_plm, full_labels)
                auc = roc_auc_score(y_te, proba[:,1])
            plm_scores.append(auc)

            # ---------- HM: add 50% of each test user's UG samples to training ----------
            hm_add = []
            for u in te_users:
                u_idx = np.where(ug_user_series.values == u)[0]
                u_idx = np.intersect1d(u_idx, te_idx, assume_unique=False)
                u_loc = u_idx.copy()
                rng.shuffle(u_loc)
                half = int(math.floor(len(u_loc) * 0.5))
                hm_add.extend(u_loc[:half].tolist())
            hm_add = np.array(sorted(hm_add))

            # Build HM training: source data + hm_add from UG
            X_hm_train = pd.concat([df_src[feature_cols_intersection], df_ug[feature_cols_intersection].iloc[hm_add]], axis=0)
            y_hm_train = np.concatenate([y_src_vec, y_ug[hm_add]], axis=0)

            # Downsample back to original source train size, stratified by label 
            if len(y_hm_train) > len(y_src_vec):
                keep_idx, _, _, _ = train_test_split(
                    np.arange(len(y_hm_train)), y_hm_train,
                    train_size=len(y_src_vec), stratify=y_hm_train, random_state=int(rng.randint(0, 2**31-1))  
                )
                X_hm_train = X_hm_train.iloc[keep_idx]
                y_hm_train = y_hm_train[keep_idx]

            # Train HM model
            pipe_hm = model_maker(); pipe_hm.fit(X_hm_train, y_hm_train)
            trained_classes_hm = pipe_hm.named_steps["clf"].classes_.astype(int)

            # Evaluate HM on the remaining half of the UG fold
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

            if multiclass:
                proba_hm = ensure_proba_columns(proba_hm, trained_classes_hm, full_labels)
                auc_hm = roc_auc_score(y_te_hm, proba_hm, multi_class="ovr", average="macro", labels=full_labels)
            else:
                proba_hm = ensure_proba_columns(proba_hm, trained_classes_hm, full_labels)
                auc_hm = roc_auc_score(y_te_hm, proba_hm[:,1])
            hm_scores.append(auc_hm)

    return (float(np.mean(plm_scores)), float(np.std(plm_scores))), \
           (float(np.mean(hm_scores)),  float(np.std(hm_scores)))

# ----------------------------- main -----------------------------

def main():
    # Sites 
    sites = [
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
        {"site": "LuisPotosi", "country": "Mexico", "continent": "Latin America",
         "features": "../data/new_data/LuisPotosi/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/LuisPotosi/timediaries.parquet"},
        {"site": "Mak", "country": "Uganda", "continent": "Africa",
         "features": "../data/new_data/Mak/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Mak/timediaries.parquet"},
        {"site": "Ulaanbaatar", "country": "Mongolia", "continent": "Asia",
         "features": "../data/new_data/Ulaanbaatar/processed/joined_features.csv",
         "timediaries": "../data/Timediaries/Ulaanbataar/timediaries.parquet"},
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
    folds = 3  
    repeats = 5
    model = "rf"
    target_site = "Mak"  

    # Assemble all sites
    print("=== Loading & aligning all sites ===")
    frames = []
    for s in sites:
        print(f"--- Loading {s['site']} ({s['country']}/{s['continent']}) ---")
        df = load_and_align_site(s["site"], s["country"], s["continent"],
                                 s["features"], s["timediaries"],
                                 tolerance_minutes)
        print(f"    aligned={len(df)} users={df['userid'].nunique()} A6a={df['A6a'].value_counts(dropna=False).to_dict()}")
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    print(f"Total rows={len(df_all)} users={df_all['userid'].nunique()} countries={df_all['country'].nunique()}")

    # Labels
    df_all["valence5"] = df_all["A6a"].astype("int64")
    df_all["y_bin"]    = map_binary(df_all["valence5"].values)
    df_all["y_three"]  = map_three(df_all["valence5"].values)

    # Partition target vs sources
    target_mask = df_all["site"] == target_site
    if not target_mask.any():
        raise ValueError(f"Target site {target_site} not found.")
    df_target = df_all[target_mask].copy()
    df_sources = df_all[~target_mask].copy()

    # Feature-space: numeric common to all 
    key_cols = {"userid","experimentid","start_interval","end_interval","A6a","td_time",
                "valence5","y_bin","y_three","site","country","continent"}
    numeric_cols = [c for c in df_all.columns if c not in key_cols and pd.api.types.is_numeric_dtype(df_all[c])]
    print(f"Candidate numeric features: {len(numeric_cols)}")

    # Model factory
    model_maker = make_rf_pipeline if model=="rf" else make_svc_pipeline

    # Iterate each source country
    results = {}
    for src_country in sorted(df_sources["country"].unique()):
        try:
            df_src = df_sources[df_sources["country"]==src_country].copy()

            # Intersect feature columns robustly (in case of slight schema diffs across sites)
            feat_src = [c for c in numeric_cols if c in df_src.columns]
            feat_tgt = [c for c in numeric_cols if c in df_target.columns]
            feat_inter = sorted(list(set(feat_src).intersection(set(feat_tgt))))
            if len(feat_inter) == 0:
                print(f"[WARN] No shared numeric features for {src_country}→{target_site}. Skipping.")
                continue

            print(f"\n=== {src_country} → {target_site} (Country-Agnostic I) ===")
            print(f"Source rows={len(df_src)} users={df_src['userid'].nunique()} | Target rows={len(df_target)} users={df_target['userid'].nunique()}")
            print(f"Using {len(feat_inter)} shared features.")
            print("Source Binary counts:", class_counts(df_src["y_bin"].values))
            print("Target Binary counts:", class_counts(df_target["y_bin"].values))

            # Binary
            (plm_bin_m, plm_bin_s), (hm_bin_m, hm_bin_s) = run_transfer_I_for_source(
                df_src, df_target, feat_inter, model_maker,
                folds=folds, repeats=repeats, multiclass=False
            )
            bin_res = {"plm": {"mean": plm_bin_m, "std": plm_bin_s},
                       "hm":  {"mean": hm_bin_m,  "std": hm_bin_s}}

            # Three-class
            print("Source Three-class counts:", class_counts(df_src["y_three"].values))
            print("Target Three-class counts:", class_counts(df_target["y_three"].values))
            (plm_3_m, plm_3_s), (hm_3_m, hm_3_s) = run_transfer_I_for_source(
                df_src, df_target, feat_inter, model_maker,
                folds=folds, repeats=repeats, multiclass=True
            )
            three_res = {"plm": {"mean": plm_3_m, "std": plm_3_s},
                         "hm":  {"mean": hm_3_m,  "std": hm_3_s}}

            results[f"{src_country}->{target_site}"] = {
                "features": len(feat_inter),
                "binary": bin_res,
                "three_class": three_res
            }
            print(f"Binary PLM: {plm_bin_m:.3f} ± {plm_bin_s:.3f} | HM: {hm_bin_m:.3f} ± {hm_bin_s:.3f}")
            print(f"Three-class PLM: {plm_3_m:.3f} ± {plm_3_s:.3f} | HM: {hm_3_m:.3f} ± {hm_3_s:.3f}")

        except Exception as e:
            print(f"[ERROR] {src_country}→{target_site}: {e}")
            results[f"{src_country}->{target_site}"] = {"error": str(e)}

    # Save results
    out_path = os.path.join(outdir, f"transfer_I_to_{target_site.lower().replace(' ', '_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {out_path}")

if __name__ == "__main__":
    main()  