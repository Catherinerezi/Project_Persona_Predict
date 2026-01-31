# streamlit_app.py
# Persona Predict (PDF-aligned) — Streamlit + Altair
# - NO Google Drive / requests
# - NO LFS dependency (repo file scan + upload)
# - EDA + Clustering + Supervised mengikuti pola PDF vertopal

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Persona Predict (PDF-aligned)", layout="wide")
alt.data_transformers.disable_max_rows()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)

# ----------------------------
# Repo file scan + robust loader (NO drive)
# ----------------------------
def _repo_root_from_file() -> Path:
    # Streamlit Cloud biasanya /mount/src/<repo>/<subfolder>/<script>.py
    # script di /app/ => parents[1] adalah root repo
    script_path = Path(__file__).resolve()
    return script_path.parents[1]

def _is_data_file(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))

def sniff_file_head(path: Path, n_lines: int = 8) -> str:
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return "".join([next(f) for _ in range(n_lines)])
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return "".join([next(f) for _ in range(n_lines)])
    except Exception as e:
        return f"[Gagal baca head: {e}]"

def _detect_delimiter(path: Path) -> str:
    # Deteksi delimiter tanpa engine="python" (biar bisa pakai engine C dan low_memory)
    # Cukup ambil sample kecil.
    candidates = [",", ";", "\t", "|"]
    sample = ""
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
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
        # fallback: hitung frekuensi delimiter paling sering
        counts = {c: sample.count(c) for c in candidates}
        return max(counts, key=counts.get) if counts else ","

def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    if name.endswith((".csv", ".csv.gz", ".gz")):
        sep = _detect_delimiter(path)
        # pakai engine C untuk stabil + low_memory ok
        return pd.read_csv(
            path,
            sep=sep,
            engine="c",
            compression="gzip" if (name.endswith(".csv.gz") or name.endswith(".gz")) else None,
            encoding_errors="replace",
            on_bad_lines="skip",
            low_memory=False,
        )

    raise ValueError(f"Format tidak didukung: {path.name}")

@st.cache_data(show_spinner=True)
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
# Column helpers (match raw column names robustly)
# ----------------------------
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().strip())

def pick_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """
    Cari kolom df yang paling cocok dengan list alias.
    Match pakai normalisasi (hapus spasi/simbol).
    """
    if df.empty:
        return None
    norm_map = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        k = _norm_col(a)
        if k in norm_map:
            return norm_map[k]
    # fallback: contains
    for c in df.columns:
        nc = _norm_col(c)
        for a in aliases:
            if _norm_col(a) in nc:
                return c
    return None

def ensure_str_series(s: pd.Series) -> pd.Series:
    # aman buat fillna string di pandas versi baru
    return s.astype("string").fillna("")

def to_int_safe(s: pd.Series, default: int = 0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(default).astype(int)

# ----------------------------
# TARGET (sesuai PDF: Penyaluran_flag dari "Penyaluran Kerja")
# ----------------------------
def make_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Penyaluran_flag" in df.columns:
        # pastikan int 0/1
        df = df.copy()
        df["Penyaluran_flag"] = to_int_safe(df["Penyaluran_flag"], 0).clip(0, 1)
        return df

    col_pk = pick_col(df, ["Penyaluran Kerja", "Penyaluran_kerja", "Placement", "Placed"])
    if not col_pk:
        return df

    s = ensure_str_series(df[col_pk])
    out = df.copy()
    out["Penyaluran_flag"] = s.map(lambda x: 1 if YES_PATTERN.search(str(x)) else 0).astype(int)
    out["Penyaluran_label"] = np.where(out["Penyaluran_flag"] == 1, "Tersalur kerja", "Belum tersalur")
    return out

# ----------------------------
# FEATURE ENGINEERING (mengikuti potongan PDF yang kebaca)
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
    # sesuai PDF: high risk = "Dapat kerja"
    return "High risk" if cluster == "Dapat kerja" else "Low risk"

def fe_core(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # strip string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Umur + bin (PDF: [0,22,25,30,100])
    col_umur = pick_col(df, ["Umur", "Age"])
    if col_umur:
        df["Umur"] = pd.to_numeric(df[col_umur], errors="coerce")
        bins = [0, 22, 25, 30, 100]
        labels = ["<=22", "23-25", "26-30", "30+"]
        df["Umur_bin"] = pd.cut(df["Umur"], bins=bins, labels=labels, include_lowest=True)

    # Region dari "Kota(Jabodetabek)"
    col_kota = pick_col(df, ["Kota(Jabodetabek)", "Kota", "Domisili"])
    if col_kota:
        s = ensure_str_series(df[col_kota])
        df["Region"] = np.where(s.str.lower().str.contains("jabodetabek", na=False), "Jabodetabek", "Luar Jabodetabek")

    # Batch_num + Batch_has_plus dari "Batch"
    col_batch = pick_col(df, ["Batch"])
    if col_batch:
        s = ensure_str_series(df[col_batch])
        df["Batch_num"] = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
        df["Batch_has_plus"] = s.str.contains(r"\+", regex=True).astype(int)

    # Program_jobconnect_flag dari "Kategori" mengandung "+"
    col_kategori = pick_col(df, ["Kategori", "Category"])
    if col_kategori:
        s = ensure_str_series(df[col_kategori])
        df["Program_jobconnect_flag"] = s.str.contains(r"\+", regex=True).astype(int)

    # Community_flag dari "Community" tidak mengandung "tidak"
    col_comm = pick_col(df, ["Community"])
    if col_comm:
        s = ensure_str_series(df[col_comm])
        df["Community_flag"] = (~s.str.lower().str.contains("tidak", na=False)).astype(int)

    # Event_flag dari "Pernah ikut acara dibimbing/tidak" => pernah/sudah/ya
    col_event = pick_col(df, ["Pernah ikut acara dibimbing/tidak", "Pernah ikut acara", "Event"])
    if col_event:
        s = ensure_str_series(df[col_event])
        df["Event_flag"] = s.str.lower().str.contains(r"(pernah|sudah|ya)", regex=True, na=False).astype(int)

    # Engagement_level (PDF: active_count + passive_count logic)
    if all(c in df.columns for c in ["Community_flag", "Event_flag", "Program_jobconnect_flag"]):
        active_count = df[["Community_flag", "Event_flag", "Program_jobconnect_flag"]].sum(axis=1)
        passive_count = (
            (1 - df["Community_flag"]) + (1 - df["Event_flag"]) + (1 - df["Program_jobconnect_flag"])
        )
        df["Engagement_level"] = np.select(
            [
                active_count >= 2,
                (active_count == 1) & (passive_count <= 2),
                passive_count >= 2,
            ],
            ["High", "Medium", "Low"],
            default="Low",
        )

    # Motivasi_cluster + risk_flag dari "Motivasi utama"
    col_motiv = pick_col(df, ["Motivasi utama", "Motivasi", "Motivasi_utama"])
    if col_motiv:
        mc = df[col_motiv].map(map_motivasi_cluster)
        df["Motivasi_cluster"] = mc
        df["Motivasi_risk_flag"] = mc.map(map_motivasi_risk_flag)

    # Segmen_karir (kalau ada di raw, biarkan; kalau ada variasi nama, tarik)
    col_seg = pick_col(df, ["Segmen_karir", "Segmen karir", "Segmen Karir", "Career segment"])
    if col_seg and "Segmen_karir" not in df.columns:
        df["Segmen_karir"] = df[col_seg]

    return df

# ----------------------------
# Sampling (KEEP ALL POSITIVES) — biar supervised nggak mati gara2 sampling
# ----------------------------
def sample_keep_all_positives(df: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    if target not in df.columns:
        return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)

    df = df.copy()
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if len(df) <= n:
        return df.reset_index(drop=True)

    # keep all positives
    remaining = max(n - len(pos), 0)
    if remaining <= 0:
        return pos.sample(min(n, len(pos)), random_state=seed).reset_index(drop=True)

    neg_s = neg.sample(min(remaining, len(neg)), random_state=seed)
    out = pd.concat([pos, neg_s], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out

# ----------------------------
# EDA helpers (Altair) — match PDF style
# ----------------------------
def freq_table(s: pd.Series) -> pd.DataFrame:
    x = s.copy()
    x = x.astype("string").fillna("Unknown")
    vc = x.value_counts(dropna=False)
    df = vc.rename_axis("Kategori").reset_index(name="Jumlah")
    df["Persentase"] = (df["Jumlah"] / df["Jumlah"].sum()) * 100.0
    return df

def chart_dist_horizontal(df_freq: pd.DataFrame, title: str) -> alt.Chart:
    # dynamic height
    h = min(700, max(240, 28 * len(df_freq)))
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
    text = base.mark_text(align="left", dx=4).encode(
        text=alt.Text("Persentase:Q", format=".1f")
    )
    return (base + text)

def placement_rate_table(df: pd.DataFrame, by_col: str, target_col: str) -> pd.DataFrame:
    g = df[[by_col, target_col]].copy()
    # IMPORTANT: cast to string first so fillna("Unknown") never breaks
    g[by_col] = g[by_col].astype("string").fillna("Unknown")
    g[target_col] = to_int_safe(g[target_col], 0).clip(0, 1)

    agg = g.groupby(by_col, dropna=False)[target_col].agg(["count", "sum"]).reset_index()
    agg = agg.rename(columns={by_col: "Group", "count": "Total", "sum": "Positives"})
    agg["Placement_rate_pct"] = (agg["Positives"] / agg["Total"]) * 100.0
    agg = agg.sort_values("Placement_rate_pct", ascending=False).reset_index(drop=True)
    return agg

def placement_rate_bar(
    df: pd.DataFrame,
    by_col: str,
    target_col: str,
    title: str,
    zoom_max_pct: float,
) -> alt.Chart:
    tab = placement_rate_table(df, by_col, target_col)
    h = min(700, max(240, 28 * len(tab)))
    # domain 0..zoom_max_pct (PDF pakai zoom kecil < 1.2%)
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
            tooltip=[
                "Group:N",
                "Total:Q",
                "Positives:Q",
                alt.Tooltip("Placement_rate_pct:Q", format=".3f"),
            ],
        )
        .properties(height=h, title=title)
    )
    text = base.mark_text(align="left", dx=4).encode(
        text=alt.Text("Placement_rate_pct:Q", format=".3f")
    )
    return base + text

# ----------------------------
# Clustering + Supervised (PDF-aligned)
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

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", ohe)]), cat_cols),
        ],
        remainder="drop",
    )

def select_feature_cols(df: pd.DataFrame) -> List[str]:
    # mengikuti PDF: pakai fitur yang memang dibuat/ada
    want = [
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
        "Segmen_karir",
    ]
    return [c for c in want if c in df.columns]

@dataclass
class ClusterOut:
    k_df: pd.DataFrame
    best_k: int
    labeled: pd.DataFrame
    svd2d: pd.DataFrame

@st.cache_data(show_spinner=False)
def fit_cluster(df_in: pd.DataFrame, feature_cols: List[str], k_min: int, k_max: int, seed: int) -> ClusterOut:
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
        sil = silhouette_score(Xt, labels)
        rows.append({"k": k, "silhouette": float(sil)})
    k_df = pd.DataFrame(rows).sort_values("k")

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
    return ClusterOut(k_df=k_df.reset_index(drop=True), best_k=best_k, labeled=labeled, svd2d=svd2d)

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
        raise ValueError("total_positive = 0 di y_true. Recall@K tidak bisa dihitung.")
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
def fit_supervised(df_in: pd.DataFrame, target: str, feature_cols: List[str], test_size: float, seed: int) -> SupOut:
    df = df_in.copy()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' tidak ada.")

    y = to_int_safe(df[target], 0).clip(0, 1)
    X = df[feature_cols].copy()

    # Split: butuh minimal 2 per kelas untuk stratify aman
    vc = y.value_counts()
    if len(vc) < 2 or vc.min() < 2:
        raise ValueError(f"Class minoritas terlalu kecil untuk train/test split (min_count={int(vc.min()) if len(vc) else 0}).")

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    base = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))
    ])
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_

    y_proba = best.predict_proba(Xte)[:, 1]
    pr_auc = float(average_precision_score(yte, y_proba))
    prec, rec, _ = precision_recall_curve(yte, y_proba)
    pr_curve = pd.DataFrame({"recall": rec, "precision": prec})

    # score_df (holdout sorted)
    score_df = pd.DataFrame({"y_true": np.asarray(yte).astype(int), "proba": y_proba}).sort_values(
        "proba", ascending=False
    ).reset_index(drop=True)

    # pos rank info (PDF)
    pos_rank = score_df.index[score_df["y_true"] == 1].tolist()
    # Top-K list (PDF style)
    N = len(score_df)
    K_list = sorted(set([
        10, 20, 50, 100, 200,
        int(0.005 * N), int(0.01 * N), int(0.02 * N), int(0.05 * N), int(0.1 * N),
    ]))
    K_list = [k for k in K_list if k > 0]
    topk_df, total_pos = topk_rates_table(score_df["y_true"], score_df["proba"], K_list)

    prevalence = total_pos / N
    lift = topk_df.copy()
    lift["Prevalence"] = prevalence
    lift["Lift@K"] = lift["Precision@K"] / prevalence
    lift["Random Recall"] = lift["K"] / N
    lift["Random Precision"] = prevalence
    lift["Random Lift"] = 1.0

    # attach rank info as metadata rows (optional display)
    # (we will show it in UI text)
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
            tooltip=["K", alt.Tooltip("Lift@K:Q", format=".2f"), alt.Tooltip("Recall@K:Q", format=".3f"), alt.Tooltip("Precision@K:Q", format=".3f")],
        )
        .properties(height=320, title="Lift / Gains (Lift@K) — PDF-aligned")
        .interactive()
    )
    baseline = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(strokeDash=[6, 6])
        .encode(y="y:Q")
    )
    return base + baseline

# ----------------------------
# Sidebar UI (marketing-AB-ish layout: clean controls)
# ----------------------------
with st.sidebar:
    st.header("Data source")

    opts = ["Repo file", "Upload file (CSV/XLSX/GZ)"]
    source = st.radio("Choose", opts, index=0)

    df_raw: Optional[pd.DataFrame] = None
    found_path: Optional[str] = None

    if source == "Repo file":
        script_path, repo_root, rels = scan_repo_files()
        st.caption("Debug (repo detection)")
        st.code(f"__file__: {script_path}\nrepo_root: {repo_root}")

        if not rels:
            st.error(
                "Tidak ada file data (.csv/.csv.gz/.gz/.xlsx) terdeteksi di repo.\n\n"
                "Solusi:\n"
                "- Pakai **Upload file**, atau\n"
                "- Pastikan file ada di repo (misal: raw_data/)"
            )
            st.stop()

        chosen = st.selectbox("Pilih file data di repo:", rels, index=0)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file (bukti file kebaca bener)")
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
                elif name.endswith(".gz") or name.endswith(".csv.gz"):
                    df_raw = pd.read_csv(f, compression="gzip", encoding_errors="replace", on_bad_lines="skip")
                else:
                    st.error("Upload CSV / XLSX / GZ.")
                    st.stop()
                found_path = "uploaded"
            except Exception as e:
                st.error(f"Gagal baca upload: {e}")
                st.stop()

    st.divider()
    st.header("Controls")
    seed = st.number_input("random_state", value=42, step=1)
    use_sample = st.toggle("Use sample for training/plots?", value=True)
    sample_n = st.number_input("Sample size (keep all positives)", value=2000, min_value=500, step=250)

    st.divider()
    st.header("Pipelines")
    run_eda = st.toggle("Show EDA (PDF)", value=True)
    run_cluster = st.toggle("Run clustering", value=True)
    run_sup = st.toggle("Run supervised ranking", value=True)

    st.divider()
    test_size = st.slider("test_size (supervised)", 0.05, 0.5, 0.2)

# ----------------------------
# Main
# ----------------------------
if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()

st.title("Persona Predict (PDF-aligned)")
st.caption(f"Loaded from: {found_path}")

# build features like PDF
df = make_target(df_raw)
df = fe_core(df)

# sampling (keep positives)
df_work = df
if use_sample and len(df_work) > int(sample_n):
    if "Penyaluran_flag" in df_work.columns:
        df_work = sample_keep_all_positives(df_work, "Penyaluran_flag", int(sample_n), int(seed))
        st.warning(f"Using sample (keep all positives): {len(df_work):,} rows (full: {len(df):,}).")
    else:
        df_work = df_work.sample(int(sample_n), random_state=int(seed)).reset_index(drop=True)
        st.warning(f"Using sample: {len(df_work):,} rows (full: {len(df):,}).")

# Overview (clean)
c1, c2 = st.columns([1.4, 1])
with c1:
    st.subheader("Preview")
    st.dataframe(df_work.head(25), use_container_width=True)
with c2:
    st.subheader("Stats")
    st.metric("Rows (work)", f"{len(df_work):,}")
    st.metric("Rows (full)", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")
    if "Penyaluran_flag" in df.columns:
        st.write("Target counts (full)")
        st.write(df["Penyaluran_flag"].value_counts())

st.divider()

# ----------------------------
# EDA (match PDF charts)
# ----------------------------
if run_eda:
    st.header("EDA mengikuti PDF vertopal (motivasi + placement-rate zoom)")
    target = "Penyaluran_flag"

    # 1) Distribusi motivasi
    if "Motivasi_cluster" in df_work.columns:
        df_freq = freq_table(df_work["Motivasi_cluster"])
        st.altair_chart(chart_dist_horizontal(df_freq, "Distribusi motivasi utama peserta"), use_container_width=True)
        st.dataframe(df_freq, use_container_width=True, hide_index=True)
    else:
        st.info("Kolom 'Motivasi_cluster' tidak ditemukan. Pastikan ada kolom 'Motivasi utama' di raw_data.")

    # 2) Placement rate per motivasi (zoom default 0.7%)
    st.subheader("Placement rate per motivasi (zoom)")
    zoom_m = st.number_input("Zoom max (%) — motivasi", value=0.7, min_value=0.01, step=0.1)
    if ("Motivasi_cluster" in df_work.columns) and (target in df_work.columns):
        st.altair_chart(
            placement_rate_bar(df_work, "Motivasi_cluster", target, "Placement rate per motivasi", float(zoom_m)),
            use_container_width=True,
        )

    # 3) Placement rate per risk level (zoom default 0.5%)
    st.subheader("Placement rate per risk level (zoom)")
    zoom_r = st.number_input("Zoom max (%) — risk", value=0.5, min_value=0.01, step=0.1)
    if ("Motivasi_risk_flag" in df_work.columns) and (target in df_work.columns):
        st.altair_chart(
            placement_rate_bar(df_work, "Motivasi_risk_flag", target, "Placement rate per risk level", float(zoom_r)),
            use_container_width=True,
        )

    # 4) Placement rate per Segmen / Umur / Region / JobConnect (zoom default 1.2%)
    st.subheader("Placement rate per Segmen Karir / Umur / Region / JobConnect (zoom)")
    zoom_x = st.number_input("Zoom max (%) — segment/umur/region/jobconnect", value=1.2, min_value=0.05, step=0.1)

    grid = st.columns(2)

    # Segmen_karir
    with grid[0]:
        if "Segmen_karir" in df_work.columns and target in df_work.columns:
            st.altair_chart(
                placement_rate_bar(df_work, "Segmen_karir", target, "Placement rate per Segmen_karir", float(zoom_x)),
                use_container_width=True,
            )
        else:
            st.info("Segmen_karir tidak tersedia di raw_data.")

    # Region
    with grid[1]:
        if "Region" in df_work.columns and target in df_work.columns:
            st.altair_chart(
                placement_rate_bar(df_work, "Region", target, "Placement rate per Region", float(zoom_x)),
                use_container_width=True,
            )
        else:
            st.info("Region tidak tersedia (cek kolom Kota(Jabodetabek)).")

    grid2 = st.columns(2)

    # Umur_bin
    with grid2[0]:
        if "Umur_bin" in df_work.columns and target in df_work.columns:
            st.altair_chart(
                placement_rate_bar(df_work, "Umur_bin", target, "Placement rate per Umur_bin", float(zoom_x)),
                use_container_width=True,
            )
        else:
            st.info("Umur_bin tidak tersedia (cek kolom Umur).")

    # JobConnect flag
    with grid2[1]:
        if "Program_jobconnect_flag" in df_work.columns and target in df_work.columns:
            st.altair_chart(
                placement_rate_bar(df_work, "Program_jobconnect_flag", target, "Placement rate per JobConnect flag", float(zoom_x)),
                use_container_width=True,
            )
        else:
            st.info("Program_jobconnect_flag tidak tersedia (cek kolom Kategori).")

st.divider()

# ----------------------------
# Clustering (PDF-aligned: silhouette + SVD scatter; NO inertia chart)
# ----------------------------
if run_cluster:
    st.header("Persona clustering (PDF-aligned)")

    feat_cols = select_feature_cols(df_work)
    if not feat_cols:
        st.error("Fitur untuk clustering tidak cukup. Pastikan FE sukses (Umur/Region/Batch/flags/motivasi).")
    else:
        cA, cB, cC = st.columns([1, 1, 1.2])
        with cA:
            kmin = st.number_input("k_min", value=2, min_value=2, step=1)
        with cB:
            kmax = st.number_input("k_max", value=8, min_value=2, step=1)
        with cC:
            run_btn = st.button("Run clustering", use_container_width=True)

        if run_btn:
            try:
                cl = fit_cluster(df_work, feat_cols, int(kmin), int(kmax), int(seed))
                st.write(f"Best k (silhouette): **{cl.best_k}**")
                st.altair_chart(chart_silhouette(cl.k_df), use_container_width=True)
                st.altair_chart(chart_svd(cl.svd2d), use_container_width=True)
                df_work = cl.labeled
            except Exception as e:
                st.error(f"Clustering failed: {e}")

st.divider()

# ----------------------------
# Supervised ranking (PDF-aligned: PR curve + Top-K + Precision/Recall@K + Lift)
# ----------------------------
if run_sup:
    st.header("Supervised ranking (LogReg) — PDF-aligned")

    target = "Penyaluran_flag"
    feat_cols = select_feature_cols(df_work)
    if "cluster_id" in df_work.columns:
        feat_cols = feat_cols + ["cluster_id"]

    if target not in df_work.columns:
        st.error("Target 'Penyaluran_flag' tidak ada. Pastikan ada kolom 'Penyaluran Kerja' di raw_data.")
    elif not feat_cols:
        st.error("Fitur supervised tidak cukup. Pastikan FE sukses.")
    else:
        run_btn = st.button("Run supervised ranking", use_container_width=True)
        if run_btn:
            try:
                sup = fit_supervised(df_work, target, feat_cols, float(test_size), int(seed))

                st.altair_chart(chart_pr_curve(sup.pr_curve, sup.pr_auc), use_container_width=True)

                # Top-K capture plot uses K from table
                K_list = sup.topk_table["K"].astype(int).tolist()
                st.altair_chart(chart_topk_capture(sup.score_df, K_list), use_container_width=True)

                # Top-K table formatted like PDF
                show = sup.topk_table.copy()
                show["Precision@K"] = (show["Precision@K"] * 100).map(lambda x: f"{x:.2f}%")
                show["Recall@K"] = (show["Recall@K"] * 100).map(lambda x: f"{x:.2f}%")
                st.subheader("Top-K table (Holdout)")
                st.dataframe(show[["K", "Positives captured", "Precision@K", "Recall@K"]], use_container_width=True, hide_index=True)

                # Precision/Recall curve
                st.altair_chart(chart_precision_recall_at_k(sup.topk_table), use_container_width=True)

                # Lift
                st.altair_chart(chart_lift(sup.lift_table), use_container_width=True)

                # Bonus: rank positive
                pos_rank = sup.score_df.index[sup.score_df["y_true"] == 1].tolist()
                if pos_rank:
                    st.success(f"Rank positif (0-based): {pos_rank} | (1-based): {[r+1 for r in pos_rank]}")
                else:
                    st.warning("Tidak ada positive di holdout (cek split / sample / target).")

            except Exception as e:
                st.error(str(e))

st.caption("Catatan: Semua visualisasi & logika dibuat ngikutin pola yang muncul di PDF vertopal (Top-K fokus karena target sangat imbalanced).")
