# streamlit_app.py
# Persona Predict — Streamlit (Altair)
# Notes for large data:
# - For exploration/training in-app, use sampling (sidebar).
# - For production, precompute features & train offline, then load model artifacts.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Persona Predict", layout="wide")
alt.data_transformers.disable_max_rows()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)

import io
import requests

def normalize_drive_url(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

@st.cache_data(show_spinner=False)
def read_csv_from_url(url: str) -> pd.DataFrame:
    url = normalize_drive_url(url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False)
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Upload CSV / XLSX.")


def has_gsheets() -> bool:
    return "gcp_service_account" in st.secrets


def read_gsheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(sheet_id).worksheet(worksheet)
    return pd.DataFrame(ws.get_all_records())


# ----------------------------
# Prep
# ----------------------------
def make_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Penyaluran_flag" in df.columns:
        return df
    if "Penyaluran Kerja" not in df.columns:
        return df
    s = df["Penyaluran Kerja"].astype(str).fillna("")
    df = df.copy()
    df["Penyaluran_flag"] = s.map(lambda x: 1 if YES_PATTERN.search(x) else 0).astype(int)
    df["Penyaluran_label"] = np.where(df["Penyaluran_flag"] == 1, "Tersalur kerja", "Belum tersalur")
    return df


def safe_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal FE that won't explode file size."""
    df = df.copy()
    if "Umur" in df.columns and "Umur_bin" not in df.columns:
        umur = pd.to_numeric(df["Umur"], errors="coerce")
        bins = [-np.inf, 18, 22, 26, 30, 35, 45, np.inf]
        labels = ["<=18", "19-22", "23-26", "27-30", "31-35", "36-45", "46+"]
        df["Umur_bin"] = pd.cut(umur, bins=bins, labels=labels)
    if "Tanggal Gabungan" in df.columns and "Month" not in df.columns:
        d = pd.to_datetime(df["Tanggal Gabungan"], errors="coerce")
        df["Month"] = d.dt.to_period("M").astype(str)
    return df


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


def drop_identifier_like(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that often break models (id/email/name/raw dates)."""
    drop = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["email", "e-mail", "nama", "name", "phone", "telepon", "no hp", "no_hp", "hp"]):
            drop.append(c)
        if cl in ["id", "user_id", "id_user", "id_peserta"] or cl.endswith("_id") or cl.startswith("id_"):
            drop.append(c)
        if ("tanggal" in cl or "date" in cl) and c != "Month":
            drop.append(c)
    return df.drop(columns=sorted(set(drop)), errors="ignore")


# ----------------------------
# Clustering
# ----------------------------
@dataclass
class ClusterOut:
    k_df: pd.DataFrame
    best_k: int
    labeled: pd.DataFrame
    svd2d: pd.DataFrame


@st.cache_data(show_spinner=False)
def fit_cluster(df_in: pd.DataFrame, feature_cols: List[str], k_min: int, k_max: int, seed: int) -> ClusterOut:
    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X = drop_identifier_like(X)

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    rows = []
    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
        pipe = Pipeline([("prep", prep), ("km", km)])
        pipe.fit(X)
        Xt = pipe.named_steps["prep"].transform(X)
        labels = pipe.named_steps["km"].labels_
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(Xt, labels)
        rows.append({"k": k, "silhouette": float(sil), "inertia": float(pipe.named_steps["km"].inertia_)})
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


# ----------------------------
# Supervised (ranking)
# ----------------------------
@dataclass
class SupOut:
    summary: pd.DataFrame
    scored: pd.DataFrame
    model: Pipeline


@st.cache_data(show_spinner=False)
def fit_supervised(df_in: pd.DataFrame, target: str, feature_cols: List[str], test_size: float, seed: int) -> SupOut:
    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    y = df[target].astype(int)

    X = drop_identifier_like(X)
    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    base = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
    ])
    base.fit(Xtr, ytr)

    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_

    def metrics(name: str, m: Pipeline):
        p = m.predict_proba(Xte)[:, 1]
        return {
            "model": name,
            "PR-AUC": float(average_precision_score(yte, p)),
            "ROC-AUC": float(roc_auc_score(yte, p)),
            "LogLoss": float(log_loss(yte, p, labels=[0, 1])),
            "Brier": float(brier_score_loss(yte, p)),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
        }

    summary = pd.DataFrame([metrics("Baseline", base), metrics("Tuned", best)])

    pall = best.predict_proba(X)[:, 1]
    scored = df_in.copy()
    scored["placement_score"] = pall
    scored = scored.sort_values("placement_score", ascending=False).reset_index(drop=True)

    return SupOut(summary=summary, scored=scored, model=best)


# ----------------------------
# Charts (Altair)
# ----------------------------
def chart_k(k_df: pd.DataFrame) -> alt.Chart:
    a = alt.Chart(k_df).mark_line(point=True).encode(
        x="k:Q", y="silhouette:Q", tooltip=["k", "silhouette", "inertia"]
    ).properties(height=220, title="Silhouette by k")
    b = alt.Chart(k_df).mark_line(point=True).encode(
        x="k:Q", y="inertia:Q", tooltip=["k", "silhouette", "inertia"]
    ).properties(height=220, title="Inertia by k")
    return alt.vconcat(a, b).resolve_scale(y="independent")


def chart_svd(df2d: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df2d)
        .mark_circle(size=40, opacity=0.7)
        .encode(x="SVD1:Q", y="SVD2:Q", color="cluster_id:N", tooltip=["cluster_id:N", "SVD1:Q", "SVD2:Q"])
        .properties(height=520, title="Cluster scatter (TruncatedSVD 2D)")
        .interactive()
    )


def chart_topk(scored: pd.DataFrame, target: str) -> alt.Chart:
    df = scored.copy()
    df["rank"] = np.arange(1, len(df) + 1)
    df["y_true"] = df[target].astype(int)
    df["cum_hits"] = df["y_true"].cumsum()
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("rank:Q", title="Rank (1=highest score)"),
            y=alt.Y("cum_hits:Q", title="Cumulative positives captured"),
            tooltip=["rank", "cum_hits", "placement_score", "y_true"],
        )
        .properties(height=260, title="Top-K capture curve")
        .interactive()
    )


# ----------------------------
# UI
# ----------------------------
st.title("Persona Predict — Streamlit (Altair)")

with st.sidebar:
    st.header("Data source")

    opts = ["Upload file (CSV/XLSX)", "URL (CSV)"] + (["Google Sheets"] if has_gsheets() else [])
    source = st.radio("Choose", opts)

    df_raw: Optional[pd.DataFrame] = None

    if source == "Upload file (CSV/XLSX)":
        f = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
        if f is not None:
            df_raw = read_uploaded(f)

    elif source == "URL (CSV)":
        url = st.text_input("CSV direct URL (Drive/Dropbox/S3)")
        st.caption("Tip (Drive): use https://drive.google.com/uc?export=download&id=FILE_ID")
        if url:
            try:
                df_raw = read_csv_from_url(url)
            except Exception as e:
                st.error(f"Failed to load CSV from URL: {e}")

    else:  # Google Sheets
        st.caption("Set `gcp_service_account` in Streamlit secrets.")
        sheet_id = st.text_input("Sheet ID")
        worksheet = st.text_input("Worksheet", value="Sheet1")
        if sheet_id and worksheet:
            try:
                df_raw = read_gsheet(sheet_id, worksheet)
            except Exception as e:
                st.error(f"Failed to read Google Sheet: {e}")

    st.divider()
    st.header("Big data controls")
    sample_mode = st.selectbox("Use sample for training/plots?", ["No (full)", "Yes (sample)"], index=1)
    sample_n = st.number_input("Sample size", value=5000, min_value=500, step=500)

    st.divider()
    st.header("Pipeline")
    do_cluster = st.checkbox("Run clustering", value=True)
    do_supervised = st.checkbox("Run supervised ranking", value=True)

    st.divider()
    st.header("Settings")
    seed = st.number_input("random_state", value=42, step=1)
    test_size = st.slider("test_size", 0.05, 0.5, 0.2)

if df_raw is None:
    st.info("Upload dataset, paste URL CSV, or connect Google Sheets.")
    st.stop()

df = safe_fe(df_raw.copy())
df = make_target(df)

df_work = df
if sample_mode.startswith("Yes") and len(df) > int(sample_n):
    df_work = df.sample(int(sample_n), random_state=int(seed)).reset_index(drop=True)
    st.warning(f"Using sample: {len(df_work):,} rows (full data kept: {len(df):,}).")

# Overview
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

# Default features (light)
default_cols = [
    "Umur", "Umur_bin", "Region", "Batch_num", "Batch_has_plus",
    "Community_flag", "Event_flag", "Engagement_level",
    "Kategori_Pekerjaan_FE", "Level_Pekerjaan_FE",
    "Domain_pendidikan", "Domain_product",
    "Segmen_karir", "Level_pendidikan_FE",
    "Motivasi_cluster", "Motivasi_risk_flag",
    "Month", "Channel", "Product", "Kategori",
]
avail = set(df_work.columns)
feat_cols = [c for c in default_cols if c in avail]
if not feat_cols:
    feat_cols = [c for c in df_work.columns if c not in ["Penyaluran_flag", "Penyaluran_label"]][:25]

# Clustering
if do_cluster:
    st.subheader("1) Persona clustering")
    a, b = st.columns(2)
    with a:
        kmin = st.number_input("k_min", value=2, min_value=2, step=1)
    with b:
        kmax = st.number_input("k_max", value=8, min_value=2, step=1)
    try:
        cl = fit_cluster(df_work, feat_cols, int(kmin), int(kmax), int(seed))
        st.write(f"Best k (silhouette): **{cl.best_k}**")
        st.altair_chart(chart_k(cl.k_df), use_container_width=True)
        st.altair_chart(chart_svd(cl.svd2d), use_container_width=True)
        df_work = cl.labeled
    except Exception as e:
        st.error(f"Clustering failed: {e}")

st.divider()

# Supervised ranking
if do_supervised:
    st.subheader("2) Supervised ranking (Logistic Regression)")
    if "Penyaluran_flag" not in df_work.columns:
        st.warning("Need 'Penyaluran Kerja' or 'Penyaluran_flag' to run supervised model.")
    else:
        sup_cols = feat_cols + (["cluster_id"] if "cluster_id" in df_work.columns else [])
        try:
            sup = fit_supervised(df_work, "Penyaluran_flag", sup_cols, float(test_size), int(seed))
            st.dataframe(sup.summary, use_container_width=True, hide_index=True)
            st.altair_chart(chart_topk(sup.scored, "Penyaluran_flag"), use_container_width=True)

            topn = st.slider("Top-N", 10, min(500, len(sup.scored)), 50)
            st.dataframe(sup.scored.head(int(topn)), use_container_width=True)

            st.download_button(
                "Download Top-N (CSV)",
                data=sup.scored.head(int(topn)).to_csv(index=False).encode("utf-8"),
                file_name="persona_predict_topN.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Supervised modelling failed: {e}")

st.caption("For very large data: train offline, then deploy app to score + visualize only.")
