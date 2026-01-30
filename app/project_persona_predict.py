# streamlit_app.py
# Persona Predict — Streamlit (Altair)
# NO Git LFS pointers. NO Google Drive. NO requests/URL download.
# Data source: Repo file (scan) / Upload file.

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

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Persona Predict", layout="wide")
alt.data_transformers.disable_max_rows()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)

# ----------------------------
# Repo detection & data loading (Marketing-AB style)
# ----------------------------
def _find_repo_root(script_path: Path) -> Path:
    """
    Heuristik repo_root:
    - Naik ke atas sampai ketemu marker file/folder yang umum ada di repo Streamlit.
    - Kalau gagal, fallback: parents[1] (umumnya /repo/app/script.py).
    """
    markers = {"requirements.txt", "package.txt", "pyproject.toml", ".gitignore"}
    for p in [script_path.parent] + list(script_path.parents):
        try:
            if any((p / m).exists() for m in markers):
                return p
            # kadang repo punya folder app/ atau raw_data/
            if (p / "app").exists() or (p / "raw_data").exists() or (p / "raw-data").exists():
                return p
        except Exception:
            pass
    # fallback
    try:
        return script_path.parents[1]
    except Exception:
        return script_path.parent


def _is_probably_table_file(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))


def sniff_file_head(path: Path, n_lines: int = 10) -> str:
    """Baca sedikit aja buat debug (aman untuk file besar)."""
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
            return "[Head preview hanya untuk CSV/GZ]"
    except Exception as e:
        return f"[Gagal baca head: {e}]"


def _is_probably_placeholder(p: Path) -> bool:
    """
    Placeholder biasanya ukuran super kecil (0–200 bytes) atau head kosong.
    Ini bukan “file besar”, ini indikasi file yang kebawa ke deploy memang kecil.
    """
    try:
        if not p.exists():
            return True
        if p.stat().st_size < 200:
            return True
        return False
    except Exception:
        return True


def read_table(path: Path) -> pd.DataFrame:
    """
    Loader robust:
    - Excel: read_excel
    - CSV: auto delimiter detect (engine=python)
    - GZ: read_csv compression=gzip
    CATATAN penting:
    - Jangan pakai low_memory saat engine="python" (bisa error di beberapa versi pandas).
    """
    name = path.name.lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    if name.endswith(".csv"):
        # 1) coba autodetect delimiter
        try:
            return pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding_errors="replace",
                on_bad_lines="skip",
            )
        except Exception:
            # 2) fallback comma dengan C engine
            return pd.read_csv(
                path,
                sep=",",
                engine="c",
                encoding_errors="replace",
                on_bad_lines="skip",
                low_memory=False,
            )

    if name.endswith(".csv.gz") or name.endswith(".gz"):
        try:
            return pd.read_csv(
                path,
                compression="gzip",
                sep=None,
                engine="python",
                encoding_errors="replace",
                on_bad_lines="skip",
            )
        except Exception:
            return pd.read_csv(
                path,
                compression="gzip",
                sep=",",
                engine="c",
                encoding_errors="replace",
                on_bad_lines="skip",
                low_memory=False,
            )

    raise ValueError(f"Format tidak didukung: {path.name}")


@st.cache_data(show_spinner=True)
def scan_repo_files() -> tuple[str, str, list[str]]:
    script_path = Path(__file__).resolve()
    repo_root = _find_repo_root(script_path)

    rels: list[str] = []
    try:
        for p in repo_root.rglob("*"):
            if not p.is_file():
                continue
            if not _is_probably_table_file(p):
                continue
            rel = str(p.relative_to(repo_root))
            # skip noise
            if any(part.startswith(".") for part in p.parts):
                continue
            if "venv" in rel or "__pycache__" in rel:
                continue
            rels.append(rel)
    except Exception:
        pass

    rels = sorted(set(rels))
    return str(script_path), str(repo_root), rels


@st.cache_data(show_spinner=False)
def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".gz") or name.endswith(".csv.gz"):
        return pd.read_csv(file, compression="gzip", sep=None, engine="python", on_bad_lines="skip")
    raise ValueError("Upload CSV / XLSX / GZ.")


# ----------------------------
# Prep / Feature Engineering
# ----------------------------
def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target dibuat dari kolom 'Penyaluran Kerja' kalau ada.
    Hasil:
      - Penyaluran_flag (0/1)
      - Penyaluran_label
    """
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
    """Minimal FE yang aman (nggak bikin data meledak)."""
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


def drop_identifier_like(df: pd.DataFrame) -> pd.DataFrame:
    """Drop kolom yang sering bikin model rusak / leakage / ID."""
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


# ----------------------------
# Clustering
# ----------------------------
@dataclass
class ClusterOut:
    k_df: pd.DataFrame
    best_k: int
    labeled: pd.DataFrame
    svd2d: pd.DataFrame


def fit_cluster(df_in: pd.DataFrame, feature_cols: List[str], k_min: int, k_max: int, seed: int) -> ClusterOut:
    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X = drop_identifier_like(X)

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    rows = []
    # IMPORTANT: silhouette_score bisa berat; pakai sample internal kalau datanya besar
    # tapi kita sudah punya sample_mode di UI (df_work).
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


def chart_k(k_df: pd.DataFrame) -> alt.Chart:
    a = (
        alt.Chart(k_df)
        .mark_line(point=True)
        .encode(x="k:Q", y="silhouette:Q", tooltip=["k", "silhouette", "inertia"])
        .properties(height=220, title="Silhouette by k")
    )
    b = (
        alt.Chart(k_df)
        .mark_line(point=True)
        .encode(x="k:Q", y="inertia:Q", tooltip=["k", "silhouette", "inertia"])
        .properties(height=220, title="Inertia by k")
    )
    return alt.vconcat(a, b).resolve_scale(y="independent")


def chart_svd(df2d: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df2d)
        .mark_circle(size=40, opacity=0.7)
        .encode(
            x="SVD1:Q",
            y="SVD2:Q",
            color=alt.Color("cluster_id:N", title="cluster_id"),
            tooltip=["cluster_id:N", "SVD1:Q", "SVD2:Q"],
        )
        .properties(height=520, title="Cluster scatter (TruncatedSVD 2D)")
        .interactive()
    )


# ----------------------------
# Supervised (ranking) — robust anti-crash
# ----------------------------
@dataclass
class SupOut:
    summary: pd.DataFrame
    scored: pd.DataFrame
    model: Pipeline


def _safe_split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """
    Biar nggak ValueError:
    - Kalau y cuma 1 kelas → skip.
    - Kalau kelas minoritas terlalu kecil untuk stratify → split tanpa stratify (warning).
    - Kalau test_size bikin minoritas nggak kebagian train/test → auto adjust atau non-stratify.
    """
    y = y.astype(int)
    vc = y.value_counts(dropna=False)
    if y.nunique() < 2:
        return None, "Target cuma 1 kelas (semua 0 atau semua 1). Supervised ranking di-skip."

    min_count = int(vc.min())
    # Syarat stratify: minimal 2 per kelas supaya bisa kebagi train+test aman
    if min_count < 2:
        return None, f"Class minoritas terlalu kecil (min_count={min_count}). Supervised ranking di-skip."

    # cek feasibility stratify dengan test_size
    # butuh minimal 1 sample per kelas di test dan train
    # n_test_class = floor(count * test_size) minimal 1
    # n_train_class = count - n_test_class minimal 1
    # cari test_size aman
    ts = float(test_size)
    lower = 1.0 / min_count
    upper = 1.0 - (1.0 / min_count)

    use_stratify = True
    if not (lower <= ts <= upper):
        # kalau out of bounds, coba clamp
        ts_clamped = min(max(ts, lower), upper)
        # kalau clamp bikin aneh juga, fallback non-stratify
        if ts_clamped <= 0.0 or ts_clamped >= 1.0:
            use_stratify = False
        else:
            ts = ts_clamped

    try:
        if use_stratify:
            return train_test_split(X, y, test_size=ts, random_state=seed, stratify=y), None
        else:
            return train_test_split(X, y, test_size=ts, random_state=seed, stratify=None), (
                "test_size/kelas tidak aman untuk stratify → split tanpa stratify (hasil metrik bisa lebih noisy)."
            )
    except Exception:
        # last fallback
        return train_test_split(X, y, test_size=ts, random_state=seed, stratify=None), (
            "Stratify gagal → split tanpa stratify."
        )


def fit_supervised(df_in: pd.DataFrame, target: str, feature_cols: List[str], test_size: float, seed: int) -> tuple[Optional[SupOut], Optional[str]]:
    df = df_in.copy()
    if target not in df.columns:
        return None, f"Target '{target}' tidak ada."

    X = df[[c for c in feature_cols if c in df.columns]].copy()
    y = df[target].astype(int)

    X = drop_identifier_like(X)
    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    split_res, warn = _safe_split(X, y, test_size=test_size, seed=seed)
    if split_res is None:
        return None, warn

    Xtr, Xte, ytr, yte = split_res

    base = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
    ])

    # Fit baseline
    base.fit(Xtr, ytr)

    # Grid search (kecil supaya nggak berat)
    grid = {"clf__C": [0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    try:
        gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
    except Exception:
        best = base

    def metrics(name: str, m: Pipeline):
        p = m.predict_proba(Xte)[:, 1]
        out = {
            "model": name,
            "PR-AUC": float(average_precision_score(yte, p)),
            "ROC-AUC": float(roc_auc_score(yte, p)),
            "LogLoss": float(log_loss(yte, p, labels=[0, 1])),
            "Brier": float(brier_score_loss(yte, p)),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
        }
        return out

    summary = pd.DataFrame([metrics("Baseline", base), metrics("Tuned", best)])

    # Score seluruh data (df_in)
    pall = best.predict_proba(X)[:, 1]
    scored = df_in.copy()
    scored["placement_score"] = pall
    scored = scored.sort_values("placement_score", ascending=False).reset_index(drop=True)

    return SupOut(summary=summary, scored=scored, model=best), warn


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
# UI — Data source
# ----------------------------
st.title("Persona Predict — Clustering + Supervised Ranking")
st.caption("Data source: Repo file / Upload. Tidak pakai Drive/URL/requests. Tidak pakai LFS pointer.")

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
            st.error("Tidak ada file data (.csv/.csv.gz/.gz/.xlsx) terdeteksi di repo.")
            st.stop()

        chosen = st.selectbox("Pilih file data di repo:", rels, index=0)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file (ini bukti file beneran kebaca)")
        try:
            st.write("Path:", chosen)
            st.write("Size (bytes):", int(full_path.stat().st_size))
        except Exception as e:
            st.error(f"Gagal akses file: {e}")
            st.stop()

        # optional head preview
        if full_path.name.lower().endswith((".csv", ".csv.gz", ".gz")):
            st.code(sniff_file_head(full_path, n_lines=10))

        # placeholder check (kalau benar2 kecil)
        if _is_probably_placeholder(full_path):
            st.error(
                "File yang kebaca di environment Streamlit terlihat terlalu kecil (indikasi placeholder/terpotong).\n"
                "Kalau kamu yakin file aslinya besar, berarti yang ter-deploy ke Streamlit belum kebawa file besar itu."
            )
            st.stop()

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
                df_raw = read_uploaded(f)
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
    do_cluster = st.checkbox("Run clustering", value=True)
    do_supervised = st.checkbox("Run supervised ranking", value=True)

    st.divider()
    st.header("Settings")
    seed = st.number_input("random_state", value=42, step=1)
    test_size = st.slider("test_size", 0.05, 0.5, 0.2)

# Must have data
if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()

st.caption(f"Loaded from: {found_path}")

# Prep
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
        st.write(df["Penyaluran_flag"].value_counts(dropna=False))

st.divider()

# Feature selection (biar cluster "sesuai")
st.subheader("Feature selection (buat cluster/model)")
default_cols = [
    "Umur", "Umur_bin", "Region", "Batch_num", "Batch_has_plus",
    "Community_flag", "Event_flag", "Engagement_level",
    "Kategori_Pekerjaan_FE", "Level_Pekerjaan_FE",
    "Domain_pendidikan", "Domain_product",
    "Segmen_karir", "Level_pendidikan_FE",
    "Motivasi_cluster", "Motivasi_risk_flag",
    "Month", "Channel", "Product", "Kategori",
]
avail = [c for c in default_cols if c in df_work.columns]
if not avail:
    avail = [c for c in df_work.columns if c not in ["Penyaluran_flag", "Penyaluran_label"]][:30]

feat_cols = st.multiselect("Pilih fitur untuk clustering/supervised:", options=sorted(df_work.columns), default=avail)

if not feat_cols:
    st.warning("Pilih minimal 1 fitur.")
    st.stop()

st.divider()

# Clustering
if do_cluster:
    st.subheader("1) Persona clustering")
    a, b, c = st.columns([1, 1, 1.2])
    with a:
        kmin = st.number_input("k_min", value=2, min_value=2, step=1)
    with b:
        kmax = st.number_input("k_max", value=8, min_value=2, step=1)
    with c:
        run_cluster = st.button("Run clustering", use_container_width=True)

    if run_cluster:
        try:
            cl = fit_cluster(df_work, feat_cols, int(kmin), int(kmax), int(seed))
            st.write(f"Best k (silhouette): **{cl.best_k}**")
            st.altair_chart(chart_k(cl.k_df), use_container_width=True)
            st.altair_chart(chart_svd(cl.svd2d), use_container_width=True)

            # cluster size + contoh profil ringan
            st.subheader("Cluster sizes")
            st.dataframe(cl.labeled["cluster_id"].value_counts().rename("n").reset_index().rename(columns={"index": "cluster_id"}))

            df_work = cl.labeled
        except Exception as e:
            st.error(f"Clustering failed: {e}")

st.divider()

# Supervised ranking
if do_supervised:
    st.subheader("2) Supervised ranking (Logistic Regression)")
    if "Penyaluran_flag" not in df_work.columns:
        st.warning("Butuh kolom 'Penyaluran Kerja' atau 'Penyaluran_flag' untuk supervised.")
    else:
        sup_cols = feat_cols + (["cluster_id"] if "cluster_id" in df_work.columns else [])
        run_sup = st.button("Run supervised ranking", use_container_width=True)

        if run_sup:
            try:
                sup, warn = fit_supervised(df_work, "Penyaluran_flag", sup_cols, float(test_size), int(seed))
                if warn:
                    st.warning(warn)

                if sup is None:
                    st.warning("Supervised ranking di-skip karena kondisi target/data tidak memenuhi.")
                else:
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

st.caption("Tip: Kalau data besar, keep sample ON supaya Streamlit tidak freeze. Untuk produksi: training offline, app untuk scoring+viz.")
