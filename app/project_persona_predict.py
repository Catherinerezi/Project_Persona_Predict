# streamlit_app.py
# Persona Predict — Streamlit (Marketing-AB style, EDA-first)
# - NO Google Drive / requests
# - NO LFS pointer logic (app reads what repo clone has)
# - Visual utama: sesuai kolom asli (distribution + rate) seperti marketing AB
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Optional ML (advanced)
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Persona Predict — EDA-first", layout="wide")
alt.data_transformers.disable_max_rows()

# Default pattern (bisa kamu edit di sidebar)
DEFAULT_YES_KEYWORDS = ["ya", "y", "yes", "sudah", "tersalur", "placed", "berhasil", "bekerja", "work"]


# ----------------------------
# Repo helpers
# ----------------------------
def _repo_root_from_file() -> Path:
    """
    Streamlit Cloud biasanya:
      /mount/src/<repo>/<subfolder>/<script>.py
    Kalau script kamu ada di /app/, repo_root = parents[1].
    """
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def _is_data_file(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))


def sniff_file_head(path: Path, n_lines: int = 10) -> str:
    """Baca sedikit aja (aman untuk file gede)."""
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
            import gzip

            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return "".join(itertools.islice(f, n_lines))
        if name.endswith((".xlsx", ".xls")):
            return "[Excel file] (preview di tab Overview)"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(itertools.islice(f, n_lines))
    except Exception as e:
        return f"[Gagal baca head: {e}]"


def read_table(path_or_buf) -> pd.DataFrame:
    """
    Loader robust: support CSV/XLSX/GZ.
    Penting: hindari low_memory param (di beberapa env + engine python bisa error).
    """
    name = getattr(path_or_buf, "name", str(path_or_buf)).lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path_or_buf)

    if name.endswith(".csv"):
        return pd.read_csv(
            path_or_buf,
            sep=None,
            engine="python",
            encoding_errors="replace",
            on_bad_lines="skip",
        )

    if name.endswith(".csv.gz") or name.endswith(".gz"):
        return pd.read_csv(
            path_or_buf,
            compression="gzip",
            sep=None,
            engine="python",
            encoding_errors="replace",
            on_bad_lines="skip",
        )

    raise ValueError("Format tidak didukung. Pakai CSV / CSV.GZ / GZ / XLSX.")


@st.cache_data(show_spinner=False)
def scan_repo_files() -> tuple[str, str, list[str]]:
    """Cari semua file data di repo supaya bisa dipilih dari UI."""
    script_path = Path(__file__).resolve()
    repo_root = _repo_root_from_file()

    rels: list[str] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and _is_data_file(p):
            rel = str(p.relative_to(repo_root))
            # skip noise
            if any(part.startswith(".") for part in p.parts):
                continue
            if "venv" in rel or "__pycache__" in rel:
                continue
            rels.append(rel)

    rels = sorted(set(rels))
    return str(script_path), str(repo_root), rels


# ----------------------------
# Minimal standardization (do NOT distort data)
# ----------------------------
def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_date_cols(df: pd.DataFrame) -> List[str]:
    # Kandidat tanggal berdasarkan nama kolom
    cand = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["tanggal", "date", "created", "joined", "gabung"]):
            cand.append(c)
    return cand


def infer_target_text_cols(df: pd.DataFrame) -> List[str]:
    cand = []
    for c in df.columns:
        cl = c.lower()
        if "penyaluran" in cl or "kerja" in cl or "placement" in cl:
            cand.append(c)
    return cand


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_target_from_text(df: pd.DataFrame, text_col: str, yes_keywords: List[str]) -> pd.DataFrame:
    """
    Bentuk target Penyaluran_flag dari kolom teks.
    Tujuannya: mirip 'converted' di marketing AB -> binary flag.
    """
    df = df.copy()
    pat = re.compile(r"\b(" + "|".join(map(re.escape, yes_keywords)) + r")\b", re.I)
    s = df[text_col].astype(str).fillna("")
    df["Penyaluran_flag"] = s.map(lambda x: 1 if pat.search(x) else 0).astype(int)
    df["Penyaluran_label"] = np.where(df["Penyaluran_flag"] == 1, "Tersalur kerja", "Belum tersalur")
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int, target_col: Optional[str]) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()

    if target_col and target_col in df.columns:
        y = df[target_col]
        # Kalau target kacau (semua sama), fallback random sample
        if y.nunique(dropna=True) >= 2:
            # stratified sampling manual
            rng = np.random.default_rng(seed)
            parts = []
            for val, g in df.groupby(target_col, dropna=False):
                take = max(1, int(round(n * (len(g) / len(df)))))
                idx = rng.choice(g.index.to_numpy(), size=min(take, len(g)), replace=False)
                parts.append(df.loc[idx])
            out = pd.concat(parts, axis=0).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
            return out

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ----------------------------
# Charts (Marketing-AB style: distribution + rate)
# ----------------------------
def bar_count(df: pd.DataFrame, col: str, top_n: int = 25) -> alt.Chart:
    vc = (
        df[col]
        .astype(str)
        .fillna("NA")
        .value_counts()
        .head(top_n)
        .rename_axis(col)
        .reset_index(name="count")
    )
    return (
        alt.Chart(vc)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{col}:N", sort="-x", title=col),
            tooltip=[col, "count"],
        )
        .properties(height=320, title=f"Top {min(top_n, len(vc))} values — {col}")
        .interactive()
    )


def hist_numeric(df: pd.DataFrame, col: str) -> alt.Chart:
    s = to_numeric_safe(df[col])
    tmp = pd.DataFrame({col: s}).dropna()
    return (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=col),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=320, title=f"Histogram — {col}")
        .interactive()
    )


def rate_by_category(df: pd.DataFrame, cat_col: str, target_col: str, top_n: int = 25) -> alt.Chart:
    tmp = df[[cat_col, target_col]].copy()
    tmp[cat_col] = tmp[cat_col].astype(str).fillna("NA")
    tmp[target_col] = to_numeric_safe(tmp[target_col])
    agg = (
        tmp.groupby(cat_col)[target_col]
        .agg(n="count", positives="sum", rate="mean")
        .reset_index()
        .sort_values("n", ascending=False)
        .head(top_n)
    )
    agg["rate_pct"] = agg["rate"] * 100.0
    return (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("rate_pct:Q", title=f"Rate {target_col} (%)"),
            y=alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
            tooltip=[cat_col, "n", "positives", alt.Tooltip("rate_pct:Q", format=".2f")],
        )
        .properties(height=340, title=f"Rate by {cat_col} (Top {min(top_n, len(agg))} by N)")
        .interactive()
    )


def rate_over_time(df: pd.DataFrame, date_col: str, target_col: str) -> alt.Chart:
    tmp = df[[date_col, target_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[target_col] = to_numeric_safe(tmp[target_col])
    tmp = tmp.dropna(subset=[date_col, target_col])
    if tmp.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
    agg = tmp.groupby("month")[target_col].agg(n="count", rate="mean").reset_index()
    agg["rate_pct"] = agg["rate"] * 100.0
    return (
        alt.Chart(agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:N", sort=None, title="Month"),
            y=alt.Y("rate_pct:Q", title=f"Rate {target_col} (%)"),
            tooltip=["month", "n", alt.Tooltip("rate_pct:Q", format=".2f")],
        )
        .properties(height=300, title=f"Rate over time — {date_col}")
        .interactive()
    )


# ----------------------------
# Advanced (Optional) ML — safe guards
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


def drop_identifier_like(df: pd.DataFrame) -> pd.DataFrame:
    """Buang kolom yang sering bikin model misleading (id, email, nama, nomor)."""
    drop = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["email", "e-mail", "nama", "name", "phone", "telepon", "no hp", "no_hp", "hp"]):
            drop.append(c)
        if cl in ["id", "user_id", "id_user", "id_peserta"] or cl.endswith("_id") or cl.startswith("id_"):
            drop.append(c)
    return df.drop(columns=sorted(set(drop)), errors="ignore")


@dataclass
class ClusterOut:
    k_df: pd.DataFrame
    best_k: int
    labeled: pd.DataFrame
    svd2d: Optional[pd.DataFrame]


@st.cache_data(show_spinner=False)
def fit_cluster(df_in: pd.DataFrame, feature_cols: List[str], k_min: int, k_max: int, seed: int, make_2d: bool) -> ClusterOut:
    from sklearn.metrics import silhouette_score

    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X = drop_identifier_like(X)

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    rows = []
    best_k = k_min
    best_sil = -1e9

    # fit per-k
    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
        pipe = Pipeline([("prep", prep), ("km", km)])
        pipe.fit(X)
        Xt = pipe.named_steps["prep"].transform(X)
        labels = pipe.named_steps["km"].labels_
        sil = float(silhouette_score(Xt, labels)) if k > 1 else float("nan")
        inertia = float(pipe.named_steps["km"].inertia_)
        rows.append({"k": k, "silhouette": sil, "inertia": inertia})
        if np.isfinite(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    k_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # fit best
    km = MiniBatchKMeans(n_clusters=best_k, random_state=seed, batch_size=2048, n_init="auto")
    pipe = Pipeline([("prep", prep), ("km", km)])
    pipe.fit(X)
    labels = pipe.named_steps["km"].labels_

    labeled = df_in.copy()
    labeled["cluster_id"] = labels

    svd2d = None
    if make_2d:
        Xt = pipe.named_steps["prep"].transform(X)
        svd = TruncatedSVD(n_components=2, random_state=seed)
        xy = svd.fit_transform(Xt)
        svd2d = pd.DataFrame({"SVD1": xy[:, 0], "SVD2": xy[:, 1], "cluster_id": labels})

    return ClusterOut(k_df=k_df, best_k=int(best_k), labeled=labeled, svd2d=svd2d)


@dataclass
class SupOut:
    summary: pd.DataFrame
    scored: pd.DataFrame
    model: Pipeline


@st.cache_data(show_spinner=False)
def fit_supervised(df_in: pd.DataFrame, target: str, feature_cols: List[str], test_size: float, seed: int) -> SupOut:
    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    y = to_numeric_safe(df[target]).fillna(0).astype(int)

    X = drop_identifier_like(X)
    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    # Guard: minimal class size
    vc = y.value_counts()
    min_count = int(vc.min()) if len(vc) else 0
    if y.nunique() < 2 or min_count < 2:
        raise ValueError(f"Target class terlalu kecil untuk train/test split. value_counts={vc.to_dict()}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    base = Pipeline(
        [
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
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
# Sidebar: Data source + Controls
# ----------------------------
with st.sidebar:
    st.header("📦 Data source (NO Drive)")
    source = st.radio("Choose", ["Repo file", "Upload file (CSV/XLSX/GZ)"], index=0)

    df_raw: Optional[pd.DataFrame] = None
    found_path: Optional[str] = None

    if source == "Repo file":
        script_path, repo_root, rels = scan_repo_files()
        st.caption("Debug (repo detection)")
        st.code(f"__file__: {script_path}\nrepo_root: {repo_root}")

        if not rels:
            st.error("Tidak ada file data (.csv/.gz/.xlsx) terdeteksi di repo.")
            st.stop()

        # Default prefer raw_data/raw_data.csv jika ada
        default_idx = 0
        for i, r in enumerate(rels):
            if r.replace("\\", "/").endswith("raw_data/raw_data.csv"):
                default_idx = i
                break

        chosen = st.selectbox("Pilih file data di repo:", rels, index=default_idx)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file (ini bukti file bener kebaca)")
        try:
            st.write("Path:", chosen)
            st.write("Size (bytes):", int(full_path.stat().st_size))
        except Exception as e:
            st.error(f"Gagal akses file: {e}")
            st.stop()

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
            try:
                df_raw = read_table(f)
                found_path = "uploaded"
                st.success("Loaded: uploaded")
            except Exception as e:
                st.error(f"Gagal baca upload: {e}")
                st.stop()

    st.divider()
    st.header("⚙️ Controls (ala Marketing AB)")
    seed = st.number_input("random_state", value=42, step=1)

    sample_on = st.selectbox("Use sample for plots/training?", ["Yes (recommended)", "No (full)"], index=0)
    sample_n = st.number_input("Sample size", value=5000, min_value=500, step=500)

    st.divider()
    st.header("🎯 Target (conversion-style)")
    # target builder controls
    yes_kw = st.text_input(
        "YES keywords (comma-separated) untuk bentuk Penyaluran_flag dari teks",
        value=", ".join(DEFAULT_YES_KEYWORDS),
    )
    yes_keywords = [x.strip() for x in yes_kw.split(",") if x.strip()]

    st.caption("Target dipakai untuk chart rate (mirip conversion rate marketing AB).")


# Stop if no data
if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()

df_raw = standardize_cols(df_raw)

# ----------------------------
# Main header
# ----------------------------
st.title("Persona Predict — EDA-first (Marketing AB style)")
st.caption(f"Loaded from: {found_path}")

# ----------------------------
# Tabs (Marketing AB style)
# ----------------------------
tab_overview, tab_eda, tab_target, tab_advanced = st.tabs(
    ["📌 Overview Data", "🔎 EDA", "🧪 Target & Rates", "🛠 Advanced (Optional)"]
)

# Work df (sampling) - keep raw for fidelity
df_full = df_raw.copy()

# Build/ensure target on FULL first (so sampling can be stratified)
target_text_cands = infer_target_text_cols(df_full)
date_cands = infer_date_cols(df_full)

# If Penyaluran_flag already exists use it; else try build from text candidates
df_full2 = df_full.copy()
if "Penyaluran_flag" not in df_full2.columns:
    if target_text_cands:
        df_full2 = make_target_from_text(df_full2, target_text_cands[0], yes_keywords)

target_col = "Penyaluran_flag" if "Penyaluran_flag" in df_full2.columns else None

# Sampling
df_work = df_full2.copy()
if sample_on.startswith("Yes") and len(df_work) > int(sample_n):
    df_work = stratified_sample(df_work, int(sample_n), int(seed), target_col).reset_index(drop=True)
    st.warning(f"Using sample: {len(df_work):,} rows (full: {len(df_full2):,})")

# ----------------------------
# TAB: Overview
# ----------------------------
with tab_overview:
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    with c1:
        st.subheader("Shape & Preview")
        st.write("Shape (work):", df_work.shape)
        st.write("Shape (full):", df_full2.shape)
        st.dataframe(df_work.head(25), use_container_width=True)

    with c2:
        st.subheader("Dtypes")
        st.json({c: str(t) for c, t in df_work.dtypes.items()})

    with c3:
        st.subheader("Missing (%)")
        miss = (df_work.isna().mean() * 100).sort_values(ascending=False).round(2).reset_index()
        miss.columns = ["column", "missing_pct"]
        st.dataframe(miss, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Basic sanity checks (biar output setia ke data)")
    st.write("Top columns:", list(df_work.columns)[:20])
    if target_col:
        st.write("Target counts (FULL):")
        st.write(df_full2[target_col].value_counts(dropna=False))
        st.write("Target counts (WORK):")
        st.write(df_work[target_col].value_counts(dropna=False))
    else:
        st.info("Target Penyaluran_flag belum terbentuk (kolom penyaluran tidak terdeteksi).")


# ----------------------------
# TAB: EDA (pure raw columns)
# ----------------------------
with tab_eda:
    st.subheader("EDA sesuai kolom asli (distribusi, bukan embedding)")

    # Pick columns
    cols = list(df_work.columns)
    num_cols = df_work.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]

    left, right = st.columns(2)
    with left:
        st.markdown("### Distribusi kategori (Top values)")
        if cat_cols:
            cat_pick = st.selectbox("Pilih kolom kategori:", cat_cols, index=0)
            topn = st.slider("Top-N", 5, 60, 25)
            st.altair_chart(bar_count(df_work, cat_pick, top_n=int(topn)), use_container_width=True)
        else:
            st.info("Tidak ada kolom kategorikal terdeteksi.")

    with right:
        st.markdown("### Distribusi numerik")
        if num_cols:
            num_pick = st.selectbox("Pilih kolom numerik:", num_cols, index=0)
            st.altair_chart(hist_numeric(df_work, num_pick), use_container_width=True)
        else:
            st.info("Tidak ada kolom numerik terdeteksi.")

    st.divider()
    st.markdown("### Table ringkas (Top categories)")
    if cat_cols:
        quick = []
        for c in cat_cols[:8]:
            vc = df_work[c].astype(str).fillna("NA").value_counts().head(5)
            quick.append(
                pd.DataFrame({"column": c, "value": vc.index.tolist(), "count": vc.values.tolist()})
            )
        st.dataframe(pd.concat(quick, axis=0), use_container_width=True, hide_index=True)


# ----------------------------
# TAB: Target & Rates (marketing AB style: conversion rate)
# ----------------------------
with tab_target:
    st.subheader("Target & Rates (mirip conversion rate marketing AB)")

    # Allow user choose target source explicitly
    st.markdown("#### 1) Target definition")
    target_mode = "Use existing Penyaluran_flag" if target_col else "Build from text column"
    modes = []
    if "Penyaluran_flag" in df_full2.columns:
        modes.append("Use existing Penyaluran_flag")
    if infer_target_text_cols(df_full2):
        modes.append("Build from text column")
    if not modes:
        modes = ["(No target detected)"]

    target_mode = st.radio("Mode:", modes, index=0)

    df_t = df_full2.copy()
    if target_mode == "Build from text column":
        text_cands = infer_target_text_cols(df_t)
        if not text_cands:
            st.error("Tidak ada kandidat kolom penyaluran/kerja terdeteksi untuk bikin target.")
        else:
            text_col = st.selectbox("Pilih kolom teks target:", text_cands, index=0)
            df_t = make_target_from_text(df_t, text_col, yes_keywords)
            target_col2 = "Penyaluran_flag"
    elif target_mode == "Use existing Penyaluran_flag":
        target_col2 = "Penyaluran_flag"
    else:
        target_col2 = None

    if not target_col2 or target_col2 not in df_t.columns:
        st.info("Target belum tersedia. Tab ini butuh target binary untuk rate charts.")
        st.stop()

    # Re-derive work sample stratified using the chosen target
    df_work_t = df_t.copy()
    if sample_on.startswith("Yes") and len(df_work_t) > int(sample_n):
        df_work_t = stratified_sample(df_work_t, int(sample_n), int(seed), target_col2)

    # Summary metrics like marketing AB
    x = int(to_numeric_safe(df_t[target_col2]).fillna(0).sum())
    n = int(df_t.shape[0])
    p = x / n if n else float("nan")

    cA, cB, cC = st.columns(3)
    cA.metric("Rows (FULL)", f"{n:,}")
    cB.metric("Positives (FULL)", f"{x:,}")
    cC.metric("Rate (FULL)", f"{p*100:.2f}%")

    st.divider()
    st.markdown("#### 2) Rate by dimension (kategori)")
    cols = list(df_work_t.columns)
    num_cols = df_work_t.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols and c not in [target_col2, "Penyaluran_label"]]

    if not cat_cols:
        st.info("Tidak ada kolom kategorikal untuk breakdown rate.")
    else:
        dim = st.selectbox("Pilih dimensi untuk breakdown rate:", cat_cols, index=0)
        topn = st.slider("Top-N by N", 5, 60, 25, key="rate_topn")
        st.altair_chart(rate_by_category(df_work_t, dim, target_col2, top_n=int(topn)), use_container_width=True)

    st.divider()
    st.markdown("#### 3) Rate over time (kalau ada kolom tanggal)")
    date_cands2 = infer_date_cols(df_work_t)
    if date_cands2:
        date_col = st.selectbox("Pilih kolom tanggal:", date_cands2, index=0)
        st.altair_chart(rate_over_time(df_work_t, date_col, target_col2), use_container_width=True)
    else:
        st.info("Tidak terdeteksi kolom tanggal (tanggal/date/joined/gabung).")


# ----------------------------
# TAB: Advanced (Optional) — not the main report
# ----------------------------
with tab_advanced:
    st.subheader("Advanced (Optional) — tidak wajib untuk report")
    st.caption("Bagian ini opsional. Kalau report kamu harus 100% setia ke data, fokus di tab EDA + Target & Rates.")

    # Use a *work* df here for performance
    df_adv = df_work.copy()

    # Feature selection
    cols = [c for c in df_adv.columns if c not in ["Penyaluran_label"]]
    st.markdown("### 1) Clustering (optional)")
    do_cluster = st.checkbox("Enable clustering", value=False)

    if do_cluster:
        # Recommend: exclude target columns and obvious ID-like columns automatically
        default_feats = [c for c in cols if c not in ["Penyaluran_flag", "cluster_id"]]
        feat_cols = st.multiselect("Feature columns", options=cols, default=default_feats[:25])

        kmin = st.number_input("k_min", value=2, min_value=2, step=1)
        kmax = st.number_input("k_max", value=6, min_value=2, step=1)
        show_2d = st.checkbox("Show 2D projection (SVD) — bisa misleading untuk report", value=False)

        if st.button("Run clustering"):
            try:
                out = fit_cluster(df_adv, feat_cols, int(kmin), int(kmax), int(seed), make_2d=bool(show_2d))
                st.success(f"Best k (silhouette): {out.best_k}")

                st.dataframe(out.k_df, use_container_width=True, hide_index=True)
                st.markdown("Cluster sizes:")
                st.write(out.labeled["cluster_id"].value_counts().sort_index())

                # Profile cluster using raw categories (this is more faithful than SVD scatter)
                st.markdown("#### Cluster profiling (raw columns)")
                cat_cols = out.labeled.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
                cat_cols = [c for c in cat_cols if c not in ["Penyaluran_label"]]
                pick_prof = st.selectbox("Profiling column:", options=cat_cols, index=0) if cat_cols else None
                if pick_prof:
                    prof = (
                        out.labeled.groupby(["cluster_id", pick_prof]).size().rename("n").reset_index()
                        .sort_values(["cluster_id", "n"], ascending=[True, False])
                    )
                    st.dataframe(prof.groupby("cluster_id").head(8), use_container_width=True, hide_index=True)

                if show_2d and out.svd2d is not None:
                    st.markdown("#### 2D projection (SVD) — hanya untuk eksplorasi")
                    chart = (
                        alt.Chart(out.svd2d)
                        .mark_circle(size=40, opacity=0.7)
                        .encode(
                            x="SVD1:Q",
                            y="SVD2:Q",
                            color="cluster_id:N",
                            tooltip=["cluster_id:N", "SVD1:Q", "SVD2:Q"],
                        )
                        .properties(height=420)
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Clustering failed: {e}")

    st.divider()
    st.markdown("### 2) Supervised ranking (optional)")
    do_sup = st.checkbox("Enable supervised ranking", value=False)

    if do_sup:
        if "Penyaluran_flag" not in df_adv.columns:
            st.warning("Target Penyaluran_flag tidak ada. Buat target di tab Target & Rates dulu.")
        else:
            # Use the same df used by target tab? keep simple:
            df_sup = df_adv.copy()
            feat_cols = st.multiselect(
                "Feature columns (supervised)",
                options=[c for c in cols if c not in ["Penyaluran_flag", "Penyaluran_label", "cluster_id"]],
                default=[c for c in cols if c not in ["Penyaluran_flag", "Penyaluran_label", "cluster_id"]][:25],
            )
            test_size = st.slider("test_size", 0.05, 0.5, 0.2)

            if st.button("Run supervised ranking"):
                try:
                    out = fit_supervised(df_sup, "Penyaluran_flag", feat_cols, float(test_size), int(seed))
                    st.dataframe(out.summary, use_container_width=True, hide_index=True)

                    topn = st.slider("Top-N", 10, min(500, len(out.scored)), 50)
                    st.dataframe(out.scored.head(int(topn)), use_container_width=True)

                    st.download_button(
                        "Download Top-N (CSV)",
                        data=out.scored.head(int(topn)).to_csv(index=False).encode("utf-8"),
                        file_name="persona_predict_topN.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Supervised failed: {e}")

st.caption("Kalau output harus 100% representatif untuk report: pakai tab Overview + EDA + Target & Rates.")
