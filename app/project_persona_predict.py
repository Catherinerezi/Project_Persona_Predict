# streamlit_app.py
# Persona Predict — Streamlit (Altair) — NO Drive/requests, NO LFS requirement
from __future__ import annotations

import re
import itertools
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Persona Predict", layout="wide")
alt.data_transformers.disable_max_rows()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)


# ----------------------------
# Repo-aware data loader (like marketing AB) — NO Drive/requests
# ----------------------------
def _repo_root_from_file() -> Path:
    script_path = Path(__file__).resolve()
    # kalau script ada di /app/, repo_root = parents[1]
    return script_path.parents[1]


def _is_probably_table(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))


def sniff_file_head(path: Path, n_lines: int = 10) -> str:
    """Baca sedikit aja biar aman buat file gede."""
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
            import gzip
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return "".join(itertools.islice(f, n_lines))
        elif name.endswith(".csv"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return "".join(itertools.islice(f, n_lines))
        else:
            return "[preview head only for CSV/GZ]"
    except Exception as e:
        return f"[Gagal baca head: {e}]"


def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    if name.endswith(".csv"):
        # delimiter auto detect -> engine python; jangan pakai low_memory (error di python engine)
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding_errors="replace",
            on_bad_lines="skip",
        )

    if name.endswith(".csv.gz") or name.endswith(".gz"):
        return pd.read_csv(
            path,
            compression="gzip",
            sep=None,
            engine="python",
            encoding_errors="replace",
            on_bad_lines="skip",
        )

    raise ValueError(f"Format tidak didukung: {path.name}")


@st.cache_data(show_spinner=False)
def scan_repo_files() -> tuple[str, str, list[str]]:
    script_path = Path(__file__).resolve()
    repo_root = _repo_root_from_file()

    rels: list[str] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and _is_probably_table(p):
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
# Prep / FE
# ----------------------------
def make_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Penyaluran_flag" in df.columns:
        return df
    if "Penyaluran Kerja" not in df.columns:
        return df
    s = df["Penyaluran Kerja"].astype(str).fillna("")
    out = df.copy()
    out["Penyaluran_flag"] = s.map(lambda x: 1 if YES_PATTERN.search(x) else 0).astype(int)
    out["Penyaluran_label"] = np.where(out["Penyaluran_flag"] == 1, "Tersalur kerja", "Belum tersalur")
    return out


def safe_fe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Umur" in out.columns and "Umur_bin" not in out.columns:
        umur = pd.to_numeric(out["Umur"], errors="coerce")
        bins = [-np.inf, 18, 22, 26, 30, 35, 45, np.inf]
        labels = ["<=18", "19-22", "23-26", "27-30", "31-35", "36-45", "46+"]
        out["Umur_bin"] = pd.cut(umur, bins=bins, labels=labels)
    if "Tanggal Gabungan" in out.columns and "Month" not in out.columns:
        d = pd.to_datetime(out["Tanggal Gabungan"], errors="coerce")
        out["Month"] = d.dt.to_period("M").astype(str)
    return out


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
# Models
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
    from sklearn.metrics import silhouette_score

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
        pipe = Pipeline([("prep", prep), ("km", km)])
        pipe.fit(X)
        Xt = pipe.named_steps["prep"].transform(X)
        labels = pipe.named_steps["km"].labels_
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

    base = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))])
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
# Charts
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
st.title("Persona Predict")
st.caption("NO Drive/requests. Data from repo or upload only.")

df_raw: Optional[pd.DataFrame] = None
found_path: Optional[str] = None

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose", ["Repo file", "Upload file (CSV/XLSX/GZ)"], index=0)

    if source == "Repo file":
        script_path, repo_root, rels = scan_repo_files()
        st.caption("Debug (repo detection)")
        st.code(f"__file__: {script_path}\nrepo_root: {repo_root}")

        if not rels:
            st.error("Tidak ada file data (.csv/.csv.gz/.gz/.xlsx) terdeteksi di repo.")
            st.stop()

        # pilih default = file terbesar biar gak kepilih yang 2 bytes
        sizes = []
        for r in rels:
            p = Path(repo_root) / r
            try:
                sizes.append((r, p.stat().st_size))
            except Exception:
                sizes.append((r, -1))
        sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
        default_rel = sizes_sorted[0][0] if sizes_sorted else rels[0]
        default_idx = rels.index(default_rel) if default_rel in rels else 0

        chosen = st.selectbox("Pilih file data di repo:", rels, index=default_idx)
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

        # guard: kalau file kecil banget, stop biar gak kebaca salah
        if full_path.stat().st_size < 1024:
            st.error(
                "File terlalu kecil (kemungkinan placeholder/hasil GitHub error).\n"
                "Pilih file lain yang ukurannya MB-an (lihat Size)."
            )
            st.stop()

        df_raw = read_table(full_path)
        found_path = str(full_path)

    else:
        f = st.file_uploader("Upload CSV/XLSX/GZ", type=["csv", "xlsx", "xls", "gz"])
        if f is None:
            st.info("Upload dulu filenya.")
            st.stop()

        name = f.name.lower()
        try:
            if name.endswith(".csv"):
                df_raw = pd.read_csv(f)
            elif name.endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(f)
            elif name.endswith(".gz"):
                df_raw = pd.read_csv(f, compression="gzip", sep=None, engine="python", on_bad_lines="skip")
            else:
                st.error("Upload CSV / XLSX / GZ.")
                st.stop()
            found_path = "uploaded"
        except Exception as e:
            st.error(f"Gagal baca upload: {e}")
            st.stop()

    st.divider()
    st.header("Big data controls")
    sample_mode = st.selectbox("Use sample for training/plots?", ["No (full)", "Yes (sample)"], index=1)
    sample_n = st.number_input("Sample size", value=5000, min_value=500, step=500)

    st.divider()
    st.header("Pipeline")
    seed = st.number_input("random_state", value=42, step=1)
    test_size = st.slider("test_size", 0.05, 0.5, 0.2)

# ---- main page must show something
st.caption(f"Loaded from: {found_path}")
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

# Buttons to avoid auto-freeze
colA, colB = st.columns(2)
run_cluster = colA.button("Run clustering")
run_supervised = colB.button("Run supervised ranking")

if run_cluster:
    st.subheader("1) Persona clustering")
    a, b = st.columns(2)
    kmin = a.number_input("k_min", value=2, min_value=2, step=1)
    kmax = b.number_input("k_max", value=8, min_value=2, step=1)

    with st.spinner("Fitting clustering..."):
        cl = fit_cluster(df_work, feat_cols, int(kmin), int(kmax), int(seed))
    st.write(f"Best k (silhouette): **{cl.best_k}**")
    st.altair_chart(chart_k(cl.k_df), use_container_width=True)
    st.altair_chart(chart_svd(cl.svd2d), use_container_width=True)

if run_supervised:
    st.subheader("2) Supervised ranking (Logistic Regression)")
    if "Penyaluran_flag" not in df_work.columns:
        st.warning("Need 'Penyaluran Kerja' or 'Penyaluran_flag' to run supervised model.")
    else:
        sup_cols = feat_cols
        with st.spinner("Training supervised model..."):
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

st.caption("Tip: kalau data gede, keep sample ON supaya Streamlit nggak nge-freeze.")
