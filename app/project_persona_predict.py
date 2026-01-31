# streamlit_app.py
# Persona Predict (PDF-aligned) — Streamlit + Altair
# - Robust loader (repo scan + upload)
# - FE yang “nyambung” ke raw_data.csv kamu (termasuk Segmen_karir)
# - Clustering anti-NaN (imputer)
# - Supervised ranking (Top-K) + target selectable (karena label placement di CSV kamu bukan “placed” murni)
# - Session state: hasil clustering & supervised gak hilang tiap klik
# - Filters + Top-N + Zoom supaya semua chart “ketampung”

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Persona Predict (PDF-aligned)", layout="wide")
alt.data_transformers.disable_max_rows()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)

# domain mapping untuk FE Segmen_karir
DOMAIN_DATA = re.compile(r"(data|stat|matemat|informat|komputer|it|teknik|engineer|sistem)", re.I)
DOMAIN_DESIGN = re.compile(r"(ui|ux|design|d kv|dkv|visual|grafis|multimedia)", re.I)
DOMAIN_BIZ = re.compile(r"(bisnis|manajem|akunt|finance|marketing|ekonomi|business)", re.I)


# ----------------------------
# Repo file scan + loader
# ----------------------------
def _repo_root_from_file() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def _is_data_file(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))


def sniff_file_head(path: Path, n_lines: int = 12) -> str:
    try:
        name = path.name.lower()
        if name.endswith((".csv.gz", ".gz")):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                out = []
                for _ in range(n_lines):
                    try:
                        out.append(next(f))
                    except StopIteration:
                        break
                return "".join(out)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                out = []
                for _ in range(n_lines):
                    try:
                        out.append(next(f))
                    except StopIteration:
                        break
                return "".join(out)
    except Exception as e:
        return f"[Gagal baca head: {e}]"


def _detect_delimiter(path: Path) -> str:
    candidates = [",", ";", "\t", "|"]
    sample = ""
    try:
        name = path.name.lower()
        if name.endswith((".csv.gz", ".gz")):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                sample = f.read(1024 * 64)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(1024 * 64)
    except Exception:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=candidates)
        return dialect.delimiter
    except Exception:
        counts = {c: sample.count(c) for c in candidates}
        return max(counts, key=counts.get) if counts else ","


def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    if name.endswith((".csv", ".csv.gz", ".gz")):
        sep = _detect_delimiter(path)
        return pd.read_csv(
            path,
            sep=sep,
            engine="c",
            compression="gzip" if name.endswith((".csv.gz", ".gz")) else None,
            encoding_errors="replace",
            on_bad_lines="skip",
            low_memory=False,
        )

    raise ValueError(f"Format tidak didukung: {path.name}")


@st.cache_data(show_spinner=False)
def scan_repo_files() -> tuple[str, str, List[str]]:
    script_path = Path(__file__).resolve()
    repo_root = _repo_root_from_file()

    rels: List[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if not _is_data_file(p):
            continue
        rel = str(p.relative_to(repo_root))
        if any(part.startswith(".") for part in p.parts):
            continue
        if "venv" in rel or "__pycache__" in rel:
            continue
        rels.append(rel)

    rels = sorted(set(rels))
    return str(script_path), str(repo_root), rels


# ----------------------------
# Column helpers
# ----------------------------
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().strip())


def pick_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df.empty:
        return None
    norm_map = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        k = _norm_col(a)
        if k in norm_map:
            return norm_map[k]
    # contains fallback
    for c in df.columns:
        nc = _norm_col(c)
        for a in aliases:
            if _norm_col(a) in nc:
                return c
    return None


def ensure_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")


def to_int_safe(s: pd.Series, default: int = 0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(default).astype(int)


# ----------------------------
# TARGET
# ----------------------------
def make_target(df_in: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df_in.copy()
    col = pick_col(df, ["Penyaluran Kerja", "Penyaluran_kerja", "Penyaluran", "Placement", "Placed"])
    if not col:
        return df

    s = ensure_str_series(df[col]).str.lower().str.strip()

    if mode == "Interest = Tertarik (recommended untuk CSV ini)":
        df["Penyaluran_flag"] = s.str.contains(r"\btertarik\b", regex=True, na=False).astype(int)
    else:
        df["Penyaluran_flag"] = s.map(lambda x: 1 if YES_PATTERN.search(str(x)) else 0).astype(int)

    df["Penyaluran_label"] = np.where(df["Penyaluran_flag"] == 1, "Positive", "Negative")
    return df


# ----------------------------
# FEATURE ENGINEERING (PDF-ish, tapi disesuaikan ke raw_data.csv kamu)
# ----------------------------
def map_motivasi_cluster(val: str) -> str:
    v = str(val).lower()
    if "dapat kerja" in v or "bekerja" in v:
        return "Dapat kerja"
    if "belajar skill" in v:
        return "Belajar skill"
    if "upgrade diri" in v:
        return "Upgrade diri"
    if "freelance" in v:
        return "Freelance"
    if "switch career" in v or "switch karir" in v:
        return "Switch career"
    return "Lainnya"


def map_motivasi_risk_flag(cluster: str) -> str:
    return "High risk" if cluster == "Dapat kerja" else "Low risk"


def map_domain_pendidikan(val: str) -> str:
    v = str(val)
    if DOMAIN_DATA.search(v):
        return "Data/IT"
    if DOMAIN_DESIGN.search(v):
        return "Design"
    if DOMAIN_BIZ.search(v):
        return "Business"
    return "Other"


def map_domain_product(val: str) -> str:
    v = str(val).lower()
    if "data" in v or "ds" in v or "analytics" in v:
        return "Data/IT"
    if "ui" in v or "ux" in v or "design" in v:
        return "Design"
    if "marketing" in v or "business" in v or "product" in v:
        return "Business"
    return "Other"


def fe_core(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # strip object cols
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Umur + bin
    col_umur = pick_col(df, ["Umur", "Age"])
    if col_umur:
        df["Umur"] = pd.to_numeric(df[col_umur], errors="coerce")
        bins = [0, 22, 25, 30, 100]
        labels = ["<=22", "23-25", "26-30", "30+"]
        df["Umur_bin"] = pd.cut(df["Umur"], bins=bins, labels=labels, include_lowest=True)

    # Region (Kota Jabodetabek)
    col_kota = pick_col(df, ["Kota(Jabodetabek)", "Kota", "Domisili"])
    if col_kota:
        s = ensure_str_series(df[col_kota])
        df["Region"] = np.where(
            s.str.lower().str.contains("jabodetabek", na=False),
            "Jabodetabek",
            "Luar Jabodetabek",
        )

    # Batch_num + Batch_has_plus
    col_batch = pick_col(df, ["Batch"])
    if col_batch:
        s = ensure_str_series(df[col_batch])
        df["Batch_num"] = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
        df["Batch_has_plus"] = s.str.contains(r"\+", regex=True).astype(int)

    # Program_jobconnect_flag (Kategori mengandung "+")
    col_kategori = pick_col(df, ["Kategori", "Category"])
    if col_kategori:
        s = ensure_str_series(df[col_kategori])
        df["Program_jobconnect_flag"] = s.str.contains(r"\+", regex=True).astype(int)

    # Community_flag (kalau ada kolom “Community”)
    col_comm = pick_col(df, ["Community"])
    if col_comm:
        s = ensure_str_series(df[col_comm])
        df["Community_flag"] = (~s.str.lower().str.contains("tidak", na=False)).astype(int)

    # Event_flag (kalau ada kolom event)
    col_event = pick_col(df, ["Pernah ikut acara dibimbing/tidak", "Pernah ikut acara", "Event"])
    if col_event:
        s = ensure_str_series(df[col_event])
        df["Event_flag"] = s.str.lower().str.contains(r"(pernah|sudah|ya)", regex=True, na=False).astype(int)

    # Engagement_level (jika tersedia flag)
    if all(c in df.columns for c in ["Community_flag", "Event_flag", "Program_jobconnect_flag"]):
        active_count = df[["Community_flag", "Event_flag", "Program_jobconnect_flag"]].sum(axis=1)
        passive_count = (1 - df["Community_flag"]) + (1 - df["Event_flag"]) + (1 - df["Program_jobconnect_flag"])
        df["Engagement_level"] = np.select(
            [active_count >= 2, (active_count == 1) & (passive_count <= 2), passive_count >= 2],
            ["High", "Medium", "Low"],
            default="Low",
        )

    # Motivasi_cluster + risk_flag (kolom CSV kamu: “Motivasi mengikuti bootcamp”)
    col_motiv = pick_col(df, ["Motivasi mengikuti bootcamp", "Motivasi utama", "Motivasi", "Motivasi_utama"])
    if col_motiv:
        mc = ensure_str_series(df[col_motiv]).map(map_motivasi_cluster)
        df["Motivasi_cluster"] = mc
        df["Motivasi_risk_flag"] = mc.map(map_motivasi_risk_flag)

    # --- FE Segmen_karir (ini yang bikin kolom “Segmen_karir” jadi ADA) ---
    col_jurpend = pick_col(df, ["Jurusan pendidikan", "Jurusan Pendidikan", "Jurusan"])
    col_prod = pick_col(df, ["Product", "Produk"])

    if col_jurpend:
        df["Domain_pendidikan"] = ensure_str_series(df[col_jurpend]).map(map_domain_pendidikan)
    if col_prod:
        df["Domain_product"] = ensure_str_series(df[col_prod]).map(map_domain_product)

    if ("Domain_pendidikan" in df.columns) and ("Domain_product" in df.columns):
        dp = df["Domain_pendidikan"].astype("string").fillna("Other")
        dpr = df["Domain_product"].astype("string").fillna("Other")
        df["is_switcher"] = ((dp != dpr) & (dp != "Other") & (dpr != "Other")).astype(int)
        df["Segmen_karir"] = df["is_switcher"].map({0: "Upskiller", 1: "Career Switcher"}).astype("string")

    return df


# ----------------------------
# Sampling (KEEP ALL POSITIVES)
# ----------------------------
def sample_keep_all_positives(df: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    if target not in df.columns:
        return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)

    df = df.copy()
    y = to_int_safe(df[target], 0).clip(0, 1)

    pos = df[y == 1]
    neg = df[y == 0]

    if len(df) <= n:
        return df.reset_index(drop=True)

    remaining = max(n - len(pos), 0)
    if remaining <= 0:
        return pos.sample(min(n, len(pos)), random_state=seed).reset_index(drop=True)

    neg_s = neg.sample(min(remaining, len(neg)), random_state=seed)
    out = pd.concat([pos, neg_s], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# ----------------------------
# EDA helpers (Altair)
# ----------------------------
def freq_table(s: pd.Series) -> pd.DataFrame:
    x = s.astype("string").fillna("Unknown")
    vc = x.value_counts(dropna=False)
    out = vc.rename_axis("Kategori").reset_index(name="Jumlah")
    out["Persentase"] = (out["Jumlah"] / out["Jumlah"].sum()) * 100.0
    return out


def chart_dist_horizontal(df_freq: pd.DataFrame, title: str) -> alt.Chart:
    h = min(760, max(240, 28 * len(df_freq)))
    base = (
        alt.Chart(df_freq)
        .mark_bar()
        .encode(
            y=alt.Y("Kategori:N", sort="-x", title=None),
            x=alt.X("Persentase:Q", title="Persentase peserta (%)"),
            tooltip=["Kategori:N", "Jumlah:Q", alt.Tooltip("Persentase:Q", format=".2f")],
        )
        .properties(height=h, title=title)
    )
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("Persentase:Q", format=".1f"))
    return base + text


def placement_rate_table(df: pd.DataFrame, by_col: str, target_col: str) -> pd.DataFrame:
    g = df[[by_col, target_col]].copy()
    g[by_col] = g[by_col].astype("string").fillna("Unknown")
    g[target_col] = to_int_safe(g[target_col], 0).clip(0, 1)

    agg = g.groupby(by_col, dropna=False)[target_col].agg(["count", "sum"]).reset_index()
    agg = agg.rename(columns={by_col: "Group", "count": "Total", "sum": "Positives"})
    agg["Placement_rate_pct"] = (agg["Positives"] / agg["Total"]) * 100.0
    return agg.sort_values("Placement_rate_pct", ascending=False).reset_index(drop=True)


def placement_rate_bar(df: pd.DataFrame, by_col: str, target_col: str, title: str, zoom_max_pct: float, top_n: int) -> alt.Chart:
    tab = placement_rate_table(df, by_col, target_col).head(top_n)
    h = min(760, max(240, 28 * len(tab)))
    base = (
        alt.Chart(tab)
        .mark_bar()
        .encode(
            y=alt.Y("Group:N", sort="-x", title=None),
            x=alt.X(
                "Placement_rate_pct:Q",
                title="Placement rate (%)",
                scale=alt.Scale(domain=[0, float(zoom_max_pct)]),
            ),
            tooltip=["Group:N", "Total:Q", "Positives:Q", alt.Tooltip("Placement_rate_pct:Q", format=".3f")],
        )
        .properties(height=h, title=title)
    )
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("Placement_rate_pct:Q", format=".3f"))
    return base + text


# ----------------------------
# Clustering + Supervised
# ----------------------------
def split_num_cat(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", ohe)])

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def select_feature_cols(df: pd.DataFrame) -> List[str]:
    want = [
        # FE utama
        "Umur",
        "Umur_bin",
        "Region",
        "Batch_num",
        "Batch_has_plus",
        "Community_flag",
        "Event_flag",
        "Engagement_level",
        "Program_jobconnect_flag",
        "Motivasi_cluster",
        "Motivasi_risk_flag",
        "Domain_pendidikan",
        "Domain_product",
        "is_switcher",
        "Segmen_karir",
        # kolom raw yang sering informatif (kalau ada)
        "Product",
        "Kategori",
        "Level pendidikan",
        "Level Pekerjaan",
        "Kategori Pekerjaan",
        "Channel",
        "Month",
        "Negara",
    ]
    return [c for c in want if c in df.columns]


@dataclass
class ClusterOut:
    k_df: pd.DataFrame
    best_k: int
    labeled: pd.DataFrame
    svd2d: pd.DataFrame


@st.cache_data(show_spinner=False)
def fit_cluster(df_in: pd.DataFrame, feature_cols: List[str], k_min: int, k_max: int, seed: int, sil_sample: int) -> ClusterOut:
    df = df_in.copy()
    X = df[feature_cols].copy()
    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    rows = []
    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
        pipe = Pipeline([("prep", prep), ("km", km)])
        pipe.fit(X)

        Xt = pipe.named_steps["prep"].transform(X)
        labels = pipe.named_steps["km"].labels_

        # silhouette: sampling biar gak berat
        ss = min(int(sil_sample), Xt.shape[0])
        sil = silhouette_score(Xt, labels, sample_size=ss, random_state=seed)
        rows.append({"k": k, "silhouette": float(sil)})

    k_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_k = int(k_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])

    km = MiniBatchKMeans(n_clusters=best_k, random_state=seed, batch_size=2048, n_init="auto")
    pipe = Pipeline([("prep", prep), ("km", km)])
    pipe.fit(X)
    labels = pipe.named_steps["km"].labels_
    Xt = pipe.named_steps["prep"].transform(X)

    svd = TruncatedSVD(n_components=2, random_state=seed)
    xy = svd.fit_transform(Xt)
    svd2d = pd.DataFrame({"SVD1": xy[:, 0], "SVD2": xy[:, 1], "cluster_id": labels})

    labeled = df_in.copy()
    labeled["cluster_id"] = labels
    return ClusterOut(k_df=k_df, best_k=best_k, labeled=labeled, svd2d=svd2d)


def chart_silhouette(k_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(k_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="k"),
            y=alt.Y("silhouette:Q", title="Silhouette"),
            tooltip=["k", alt.Tooltip("silhouette:Q", format=".4f")],
        )
        .properties(height=240, title="Silhouette by k (PDF-aligned)")
    )


def chart_svd(df2d: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df2d)
        .mark_circle(size=35, opacity=0.75)
        .encode(
            x=alt.X("SVD1:Q"),
            y=alt.Y("SVD2:Q"),
            color=alt.Color("cluster_id:N", title="cluster_id"),
            tooltip=["cluster_id:N", alt.Tooltip("SVD1:Q", format=".3f"), alt.Tooltip("SVD2:Q", format=".3f")],
        )
        .properties(height=520, title="Cluster scatter (TruncatedSVD 2D) — PDF-aligned")
        .interactive()
    )


@dataclass
class SupOut:
    pr_auc: float
    pr_curve: pd.DataFrame
    score_df: pd.DataFrame
    topk_table: pd.DataFrame
    lift_table: pd.DataFrame
    model: Pipeline


def topk_rates_table(y_true, y_score, K_list: List[int]) -> Tuple[pd.DataFrame, int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        raise ValueError("total_positive = 0 di y_true. Top-K tidak bisa dihitung.")
    rows = []
    for K in K_list:
        K = int(K)
        K_eff = min(K, len(y_sorted))
        captured = int(y_sorted[:K_eff].sum())
        prec_k = captured / K_eff if K_eff > 0 else 0.0
        rec_k = captured / total_pos
        rows.append({"K": K_eff, "Positives captured": captured, "Precision@K": prec_k, "Recall@K": rec_k})
    return pd.DataFrame(rows), total_pos


@st.cache_data(show_spinner=False)
def fit_supervised(
    df_in: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    test_size: float,
    seed: int,
    use_grid: bool,
) -> SupOut:
    df = df_in.copy()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' tidak ada.")

    y = to_int_safe(df[target], 0).clip(0, 1)
    X = df[feature_cols].copy()

    vc = y.value_counts()
    if len(vc) < 2 or vc.min() < 2:
        raise ValueError(f"Class minoritas terlalu kecil untuk split (min_count={int(vc.min()) if len(vc) else 0}).")

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    base = Pipeline(
        [
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )

    if use_grid:
        grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
    else:
        best = base
        best.fit(Xtr, ytr)

    y_proba = best.predict_proba(Xte)[:, 1]
    pr_auc = float(average_precision_score(yte, y_proba))
    prec, rec, _ = precision_recall_curve(yte, y_proba)
    pr_curve = pd.DataFrame({"recall": rec, "precision": prec})

    score_df = pd.DataFrame({"y_true": np.asarray(yte).astype(int), "proba": y_proba}).sort_values(
        "proba", ascending=False
    ).reset_index(drop=True)

    N = len(score_df)
    K_list = sorted(
        set(
            [
                10,
                20,
                50,
                100,
                200,
                int(0.005 * N),
                int(0.01 * N),
                int(0.02 * N),
                int(0.05 * N),
                int(0.1 * N),
            ]
        )
    )
    K_list = [k for k in K_list if k > 0]
    topk_df, total_pos = topk_rates_table(score_df["y_true"], score_df["proba"], K_list)

    prevalence = total_pos / N
    lift = topk_df.copy()
    lift["Prevalence"] = prevalence
    lift["Lift@K"] = lift["Precision@K"] / prevalence
    lift["Random Recall"] = lift["K"] / N
    lift["Random Precision"] = prevalence
    lift["Random Lift"] = 1.0

    return SupOut(
        pr_auc=pr_auc,
        pr_curve=pr_curve,
        score_df=score_df,
        topk_table=topk_df,
        lift_table=lift,
        model=best,
    )


def chart_pr_curve(pr_curve: pd.DataFrame, pr_auc: float) -> alt.Chart:
    return (
        alt.Chart(pr_curve)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
        )
        .properties(height=320, title=f"Precision–Recall Curve (Holdout) | PR-AUC={pr_auc:.3f}")
        .interactive()
    )


def chart_topk_capture(score_df: pd.DataFrame, K_list: List[int]) -> alt.Chart:
    df = score_df.copy()
    df["rank"] = np.arange(1, len(df) + 1)
    df["cum_hits"] = df["y_true"].cumsum()
    kset = set(K_list)
    df_k = df[df["rank"].isin(kset)].copy()

    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("rank:Q", title="Top-K (jumlah peserta diprioritaskan)"),
            y=alt.Y("cum_hits:Q", title="Jumlah positif tertangkap (cumulative)"),
            tooltip=["rank", "cum_hits", "y_true", alt.Tooltip("proba:Q", format=".6f")],
        )
    )
    pts = alt.Chart(df_k).mark_circle(size=70).encode(x="rank:Q", y="cum_hits:Q")
    return (line + pts).properties(height=320, title="Top-K Capture (Holdout) — PDF-aligned").interactive()


def chart_precision_recall_at_k(topk: pd.DataFrame) -> alt.Chart:
    df = topk.copy()
    df_long = df.melt(id_vars=["K"], value_vars=["Recall@K", "Precision@K"], var_name="Metric", value_name="Rate")
    return (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("Rate:Q", title="Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title=None),
            tooltip=["K", "Metric", alt.Tooltip("Rate:Q", format=".3f")],
        )
        .properties(height=320, title="Top-K Capture Curve (Precision@K & Recall@K) — PDF-aligned")
        .interactive()
    )


def chart_lift(lift: pd.DataFrame) -> alt.Chart:
    df = lift.copy()
    base = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("Lift@K:Q", title="Lift@K"),
            tooltip=[
                "K",
                alt.Tooltip("Lift@K:Q", format=".2f"),
                alt.Tooltip("Recall@K:Q", format=".3f"),
                alt.Tooltip("Precision@K:Q", format=".3f"),
            ],
        )
        .properties(height=320, title="Lift / Gains (Lift@K) — PDF-aligned")
        .interactive()
    )
    baseline = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
    return base + baseline


# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose", ["Repo file", "Upload file (CSV/XLSX/GZ)"], index=0)

    df_raw: Optional[pd.DataFrame] = None
    found_path: Optional[str] = None

    if source == "Repo file":
        script_path, repo_root, rels = scan_repo_files()
        st.caption("Debug (repo detection)")
        st.code(f"__file__: {script_path}\nrepo_root: {repo_root}")

        if not rels:
            st.error("Tidak ada file data terdeteksi di repo. Pakai Upload.")
            st.stop()

        chosen = st.selectbox("Pilih file data di repo:", rels, index=0)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file")
        try:
            st.write("Path:", chosen)
            st.write("Size (bytes):", full_path.stat().st_size)
        except Exception as e:
            st.error(f"Gagal akses file: {e}")
            st.stop()

        if full_path.name.lower().endswith((".csv", ".csv.gz", ".gz")):
            st.code(sniff_file_head(full_path, n_lines=10))

        try:
            df_raw = read_table(full_path)
            found_path = str(full_path)
            st.success(f"Loaded: {chosen}")
        except Exception as e:
            st.error(f"Gagal baca file: {chosen}\nError: {e}")
            st.stop()

    else:
        f = st.file_uploader("Upload CSV/XLSX/GZ", type=["csv", "xlsx", "xls", "gz"])
        if f is not None:
            name = f.name.lower()
            try:
                if name.endswith(".csv"):
                    df_raw = pd.read_csv(f, encoding_errors="replace", on_bad_lines="skip")
                elif name.endswith((".xlsx", ".xls")):
                    df_raw = pd.read_excel(f)
                elif name.endswith((".gz", ".csv.gz")):
                    df_raw = pd.read_csv(f, compression="gzip", encoding_errors="replace", on_bad_lines="skip")
                else:
                    st.error("Upload CSV / XLSX / GZ.")
                    st.stop()
                found_path = "uploaded"
            except Exception as e:
                st.error(f"Gagal baca upload: {e}")
                st.stop()

    st.divider()
    st.header("Core controls")
    seed = st.number_input("random_state", value=42, step=1)

    target_mode = st.selectbox(
        "Target definition (Penyaluran_flag)",
        ["Interest = Tertarik (recommended untuk CSV ini)", "Strict placed (regex ya/sudah/placed)"],
        index=0,
    )

    use_sample = st.toggle("Use sample for training/plots?", value=True)
    sample_n = st.number_input("Sample size (keep all positives)", value=2500, min_value=500, step=250)

    st.divider()
    st.header("Filters (optional)")
    st.caption("Filter dipakai sebelum FE+model. Kosong = no filter.")
    # nanti kita isi options setelah df kebentuk; placeholder di sini

    st.divider()
    st.header("Pipelines")
    run_eda = st.toggle("Show EDA (PDF-ish)", value=True)
    run_cluster = st.toggle("Enable clustering tab", value=True)
    run_sup = st.toggle("Enable supervised tab", value=True)

    st.divider()
    st.header("Clustering options")
    sil_sample = st.number_input("Silhouette sample_size", value=2000, min_value=300, step=200)

    st.divider()
    st.header("Supervised options")
    test_size = st.slider("test_size", 0.05, 0.5, 0.2)
    use_grid = st.toggle("GridSearch C (slower, often better)", value=False)


# ----------------------------
# Main
# ----------------------------
st.title("Persona Predict (PDF-aligned)")
if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()

st.caption(f"Loaded from: {found_path}")

# Basic cleanup (safe)
df0 = df_raw.copy()
df0.columns = [str(c).strip() for c in df0.columns]

# Build target + FE
df1 = make_target(df0, mode=target_mode)
df1 = fe_core(df1)

# Filters UI (setelah df1 ada)
with st.sidebar:
    # Build options from existing raw columns if present
    def _ms_options(colname: str) -> Optional[List[str]]:
        if colname in df1.columns:
            return sorted(df1[colname].astype("string").fillna("Unknown").unique().tolist())
        return None

    f_product = None
    f_month = None
    f_kategori = None

    opts_prod = _ms_options("Product")
    if opts_prod:
        f_product = st.multiselect("Product", opts_prod)

    opts_month = _ms_options("Month")
    if opts_month:
        f_month = st.multiselect("Month", opts_month)

    opts_kat = _ms_options("Kategori")
    if opts_kat:
        f_kategori = st.multiselect("Kategori", opts_kat)

# Apply filters
mask = pd.Series(True, index=df1.index)
if f_product is not None and len(f_product) > 0:
    mask &= df1["Product"].astype("string").fillna("Unknown").isin(f_product)
if f_month is not None and len(f_month) > 0:
    mask &= df1["Month"].astype("string").fillna("Unknown").isin(f_month)
if f_kategori is not None and len(f_kategori) > 0:
    mask &= df1["Kategori"].astype("string").fillna("Unknown").isin(f_kategori)

df = df1.loc[mask].reset_index(drop=True)

# Sampling for work df
df_work = df
if use_sample and len(df_work) > int(sample_n):
    if "Penyaluran_flag" in df_work.columns:
        df_work = sample_keep_all_positives(df_work, "Penyaluran_flag", int(sample_n), int(seed))
        st.warning(f"Using sample (keep all positives): {len(df_work):,} rows (full: {len(df):,}).")
    else:
        df_work = df_work.sample(int(sample_n), random_state=int(seed)).reset_index(drop=True)
        st.warning(f"Using sample: {len(df_work):,} rows (full: {len(df):,}).")

# Tabs
tabs = st.tabs(["📌 Overview", "🔎 EDA", "🧩 Clustering", "🎯 Supervised (Top-K)"])

# ----------------------------
# Overview
# ----------------------------
with tabs[0]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Preview (work)")
        st.dataframe(df_work.head(30), use_container_width=True)
    with c2:
        st.subheader("Quick stats")
        st.metric("Rows (work)", f"{len(df_work):,}")
        st.metric("Rows (full)", f"{len(df):,}")
        st.metric("Columns", f"{df.shape[1]:,}")

        if "Penyaluran_flag" in df.columns:
            vc_full = df["Penyaluran_flag"].value_counts(dropna=False)
            vc_work = df_work["Penyaluran_flag"].value_counts(dropna=False) if "Penyaluran_flag" in df_work.columns else None
            st.write("Target counts (full):", vc_full.to_dict())
            if vc_work is not None:
                st.write("Target counts (work):", vc_work.to_dict())

        st.divider()
        st.write("Kolom FE yang penting:")
        show_cols = [c for c in ["Motivasi_cluster", "Motivasi_risk_flag", "Segmen_karir", "Domain_pendidikan", "Domain_product", "is_switcher"] if c in df.columns]
        st.write(show_cols if show_cols else "Belum kebentuk (cek mapping kolom raw).")

# ----------------------------
# EDA
# ----------------------------
with tabs[1]:
    if not run_eda:
        st.info("EDA dimatikan dari sidebar.")
    else:
        st.header("EDA (PDF-ish)")

        if "Penyaluran_flag" not in df_work.columns:
            st.warning("Target Penyaluran_flag belum ada (kolom Penyaluran Kerja tidak ketemu). EDA rate akan terbatas.")
        target = "Penyaluran_flag"

        cA, cB = st.columns([1.2, 1])
        with cA:
            st.subheader("Distribusi motivasi")
            if "Motivasi_cluster" in df_work.columns:
                f = freq_table(df_work["Motivasi_cluster"])
                st.altair_chart(chart_dist_horizontal(f, "Distribusi motivasi"), use_container_width=True)
                st.dataframe(f, use_container_width=True, hide_index=True)
            else:
                st.info("Motivasi_cluster tidak ada (cek kolom 'Motivasi mengikuti bootcamp').")

        with cB:
            st.subheader("Distribusi segmen karir (FE)")
            if "Segmen_karir" in df_work.columns:
                f = freq_table(df_work["Segmen_karir"])
                st.altair_chart(chart_dist_horizontal(f, "Distribusi Segmen_karir"), use_container_width=True)
                st.dataframe(f, use_container_width=True, hide_index=True)
            else:
                st.info("Segmen_karir belum kebentuk. Cek Jurusan pendidikan & Product ada/tidak.")

        st.divider()

        if target in df_work.columns:
            st.subheader("Placement/Target rate (zoom + top-N)")
            top_n = st.slider("Top-N categories per chart", 5, 40, 20)
            zoom = st.number_input("Zoom max (%)", value=1.2, min_value=0.05, step=0.1)

            candidates = [
                "Motivasi_cluster",
                "Motivasi_risk_flag",
                "Segmen_karir",
                "Region",
                "Umur_bin",
                "Program_jobconnect_flag",
                "Domain_pendidikan",
                "Domain_product",
                "Kategori",
                "Product",
                "Month",
                "Channel",
            ]
            candidates = [c for c in candidates if c in df_work.columns]

            if not candidates:
                st.info("Tidak ada kolom kategori yang siap untuk rate chart.")
            else:
                pick = st.selectbox("Pilih breakdown:", candidates, index=0)
                st.altair_chart(
                    placement_rate_bar(df_work, pick, target, f"Placement/Target rate by {pick}", float(zoom), int(top_n)),
                    use_container_width=True,
                )

                tab = placement_rate_table(df_work, pick, target).head(int(top_n))
                st.dataframe(tab, use_container_width=True, hide_index=True)
        else:
            st.info("Target Penyaluran_flag tidak ada, jadi placement-rate charts dilewati.")

# ----------------------------
# Clustering
# ----------------------------
with tabs[2]:
    if not run_cluster:
        st.info("Clustering dimatikan dari sidebar.")
    else:
        st.header("Persona clustering (PDF-aligned)")

        feat_cols = select_feature_cols(df_work)
        if not feat_cols:
            st.error("Fitur untuk clustering tidak cukup. Pastikan FE kebentuk.")
        else:
            with st.form("cluster_form"):
                c1, c2, c3 = st.columns([1, 1, 1.2])
                with c1:
                    kmin = st.number_input("k_min", value=2, min_value=2, step=1)
                with c2:
                    kmax = st.number_input("k_max", value=8, min_value=2, step=1)
                with c3:
                    submitted = st.form_submit_button("Run clustering", use_container_width=True)

            if submitted:
                if int(kmax) < int(kmin):
                    st.error("k_max harus >= k_min")
                else:
                    try:
                        cl = fit_cluster(df_work, feat_cols, int(kmin), int(kmax), int(seed), int(sil_sample))
                        st.session_state["cluster_out"] = cl
                        st.session_state["df_with_cluster"] = cl.labeled
                        st.success(f"Clustering done. Best k = {cl.best_k}")
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")

            # display stored result
            if "cluster_out" in st.session_state:
                cl: ClusterOut = st.session_state["cluster_out"]
                st.write(f"Best k (silhouette): **{cl.best_k}**")
                st.altair_chart(chart_silhouette(cl.k_df), use_container_width=True)
                st.altair_chart(chart_svd(cl.svd2d), use_container_width=True)

                st.subheader("Cluster counts")
                ct = pd.DataFrame(cl.labeled["cluster_id"].value_counts().rename_axis("cluster_id").reset_index(name="n"))
                st.dataframe(ct, use_container_width=True, hide_index=True)

# ----------------------------
# Supervised ranking (Top-K)
# ----------------------------
with tabs[3]:
    if not run_sup:
        st.info("Supervised dimatikan dari sidebar.")
    else:
        st.header("Supervised ranking (LogReg) — Top-K (PDF-aligned)")

        # Prefer df_with_cluster if exists
        df_sup = st.session_state.get("df_with_cluster", df_work)

        target = "Penyaluran_flag"
        if target not in df_sup.columns:
            st.error("Target 'Penyaluran_flag' tidak ada. Pastikan kolom 'Penyaluran Kerja' ter-detect, atau ganti dataset.")
        else:
            vc = df_sup[target].value_counts(dropna=False)
            st.write("Target counts (work/sup):", vc.to_dict())

            feat_cols = select_feature_cols(df_sup)
            if "cluster_id" in df_sup.columns and "cluster_id" not in feat_cols:
                feat_cols = feat_cols + ["cluster_id"]

            if not feat_cols:
                st.error("Fitur supervised tidak cukup. Pastikan FE kebentuk.")
            else:
                with st.form("sup_form"):
                    st.caption("Tips: kalau target positive sangat sedikit, model + Top-K kurang stabil (lebih baik target-mode 'Interest=Tertarik').")
                    submitted = st.form_submit_button("Run supervised ranking", use_container_width=True)

                if submitted:
                    try:
                        sup = fit_supervised(df_sup, target, feat_cols, float(test_size), int(seed), bool(use_grid))
                        st.session_state["sup_out"] = sup
                        st.success("Supervised ranking done.")
                    except Exception as e:
                        st.error(str(e))

                if "sup_out" in st.session_state:
                    sup: SupOut = st.session_state["sup_out"]

                    st.altair_chart(chart_pr_curve(sup.pr_curve, sup.pr_auc), use_container_width=True)

                    K_list = sup.topk_table["K"].astype(int).tolist()
                    st.altair_chart(chart_topk_capture(sup.score_df, K_list), use_container_width=True)

                    st.subheader("Top-K table (Holdout)")
                    show = sup.topk_table.copy()
                    show["Precision@K"] = (show["Precision@K"] * 100).map(lambda x: f"{x:.2f}%")
                    show["Recall@K"] = (show["Recall@K"] * 100).map(lambda x: f"{x:.2f}%")
                    st.dataframe(show[["K", "Positives captured", "Precision@K", "Recall@K"]], use_container_width=True, hide_index=True)

                    st.altair_chart(chart_precision_recall_at_k(sup.topk_table), use_container_width=True)
                    st.altair_chart(chart_lift(sup.lift_table), use_container_width=True)

                    pos_rank = sup.score_df.index[sup.score_df["y_true"] == 1].tolist()
                    if pos_rank:
                        st.success(f"Rank positif (1-based): {[r+1 for r in pos_rank]}")
                    else:
                        st.warning("Tidak ada positive di holdout (cek split / sample / target_mode).")

st.caption("Catatan: Hasil clustering & supervised disimpan di session_state, jadi tidak hilang setelah rerun/klik lain.")
