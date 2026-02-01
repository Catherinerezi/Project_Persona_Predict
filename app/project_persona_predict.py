# app.py
# Persona Segmentation & Placement Prediction (PDF-aligned) — Single-file Streamlit
# - Charts: Altair only (NO Plotly)
# - Clustering: 3 clusters (persona naming + profiling) — NO random_state slider
# - Supervised: ranking Top-K + PR-AUC + Precision@K/Recall@K/Lift@K + downloadable Top-K
# - Dashboard Akhir (Bisnis): ringkas & actionable untuk tujuan proyek
#
# Requirements (if you need requirements.txt on Streamlit Cloud):
# streamlit
# pandas
# numpy
# altair
# scikit-learn

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Persona Segmentation & Placement Prediction (PDF-aligned)",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

# ----------------------------
# Constants (keep stable)
# ----------------------------
N_CLUSTERS = 3
# Note: KMeans family uses randomness for init; we FIX it internally for reproducibility
# (but we DO NOT expose it to UI and we keep n_init high so hasilnya stabil).
_INTERNAL_RANDOM_STATE = 42

# ----------------------------
# Utility helpers
# ----------------------------
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def _safe_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if isinstance(c, str)]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _maybe_parse_date(s: pd.Series) -> pd.Series:
    # robust date parsing
    try:
        out = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        return out
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(s)), errors="coerce")

def _ensure_month_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'Month' missing, try to create it from known date columns.
    """
    df = df.copy()
    if "Month" in df.columns:
        return df

    date_candidates = [
        "Tanggal Gabungan",
        "Tanggal_Gabungan",
        "Tanggal Gabung",
        "Tanggal_Gabung",
        "Tanggal",
        "Join Date",
        "Tanggal Join",
        "Tgl Gabungan",
    ]
    found = None
    for c in date_candidates:
        if c in df.columns:
            found = c
            break

    if found:
        dt = _maybe_parse_date(df[found])
        df["Month"] = dt.dt.to_period("M").astype(str)
    return df

def load_data(uploaded_file, repo_path: Optional[str]):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "upload"
        return df, source
    if repo_path:
        df = pd.read_csv(repo_path)
        source = "repo"
        return df, source
    return None, None

def apply_global_filters(df: pd.DataFrame, filter_cols: List[str], selections: Dict[str, List[str]]):
    out = df.copy()
    for c in filter_cols:
        if c not in out.columns:
            continue
        chosen = selections.get(c, [])
        if chosen:
            out = out[out[c].astype(str).isin([str(x) for x in chosen])]
    return out

def infer_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    PDF-aligned feature candidates (robust to missing columns).
    """
    cat_candidates = [
        "Batch",
        "Product",
        "Kategori",
        "Channel",
        "Level pendidikan",
        "Kategori Kesibukan",
        "Level Pekerjaan",
        "Kategori Pekerjaan",
        "Domisili",
        "Provinsi",
        "Negara",
        "Region",
        "Segmen_karir",
        "Motivasi_cluster",
        "Sumber",
        "Month",
    ]
    num_candidates = ["Umur"]

    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in num_candidates if c in df.columns]
    return cat_cols, num_cols

def make_target(
    df: pd.DataFrame,
    target_col: str = "Penyaluran Kerja",
    mode: str = "TERTARIK_AS_POSITIVE",
    manual_positive_value: Optional[str] = None,
):
    """
    Returns y (0/1) and desc.
    mode:
      - TERTARIK_AS_POSITIVE: 1 if contains "Tertarik"
      - TIDAK_TERTARIK_AS_POSITIVE: 1 if contains "Tidak"
      - MANUAL_VALUE_AS_POSITIVE: 1 if equals manual_positive_value
    """
    if target_col not in df.columns:
        return None, f"Target column '{target_col}' tidak ditemukan."

    s = df[target_col].astype(str).fillna("")

    if mode == "TERTARIK_AS_POSITIVE":
        y = s.str.contains("Tertarik", case=False, regex=False).astype(int)
        desc = "Target=1 jika mengandung 'Tertarik'"
        return y, desc

    if mode == "TIDAK_TERTARIK_AS_POSITIVE":
        y = s.str.contains("Tidak", case=False, regex=False).astype(int)
        desc = "Target=1 jika mengandung 'Tidak' (minoritas / at-risk / intervensi)"
        return y, desc

    if mode == "MANUAL_VALUE_AS_POSITIVE":
        if manual_positive_value is None:
            return None, "Manual mapping dipilih tapi nilai POSITIF belum ditentukan."
        y = (s == str(manual_positive_value)).astype(int)
        desc = f"Target=1 jika value == '{manual_positive_value}' (manual mapping)"
        return y, desc

    return None, "Mode target tidak dikenali."

def target_summary(y: pd.Series):
    n = int(len(y))
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_rate = (pos / n) if n else 0.0
    return n, pos, neg, pos_rate

def top_rate_by_group(df: pd.DataFrame, y: pd.Series, group_col: str, min_count=30, top_n=30):
    if group_col not in df.columns:
        return pd.DataFrame(columns=["Group", "Total", "Positives", "rate", "rate_pct"])

    tmp = pd.DataFrame({"Group": df[group_col].astype(str), "_y": y.astype(int).values})
    g = tmp.groupby("Group", dropna=False)["_y"].agg(["count", "sum"]).reset_index()
    g = g.rename(columns={"count": "Total", "sum": "Positives"})
    g["rate"] = np.where(g["Total"] > 0, g["Positives"] / g["Total"], 0.0)
    g = g[g["Total"] >= int(min_count)].sort_values("rate", ascending=False).head(int(top_n))
    g["rate_pct"] = (g["rate"] * 100).round(2)
    return g

def alt_bar_rate(df_rate: pd.DataFrame, title: str):
    if df_rate is None or df_rate.empty:
        return None

    base = alt.Chart(df_rate).mark_bar().encode(
        y=alt.Y("Group:N", sort="-x", title="Group"),
        x=alt.X("rate_pct:Q", title="Target rate (%)"),
        tooltip=[
            alt.Tooltip("Group:N"),
            alt.Tooltip("Total:Q", format=",.0f"),
            alt.Tooltip("Positives:Q", format=",.0f"),
            alt.Tooltip("rate_pct:Q", format=".2f"),
        ],
    ).properties(title=title, height=min(560, 24 * len(df_rate) + 90))

    text = alt.Chart(df_rate).mark_text(align="left", dx=6).encode(
        y=alt.Y("Group:N", sort="-x"),
        x=alt.X("rate_pct:Q"),
        text=alt.Text("rate_pct:Q", format=".2f"),
    )
    return base + text

def _make_ohe():
    # sklearn compatibility: sparse_output (new) vs sparse (old)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def build_preprocess(cat_cols: List[str], num_cols: List[str]):
    transformers = []
    if cat_cols:
        transformers.append(("cat", _make_ohe(), cat_cols))
    if num_cols:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def prepare_X(df_in: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    """
    Prevent ValueError in ColumnTransformer/OneHotEncoder:
    - cat: fillna + cast to str
    - num: to_numeric + fillna
    """
    cols = [c for c in (cat_cols + num_cols) if c in df_in.columns]
    X = df_in[cols].copy()

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna("Unknown").astype(str)

    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X

def persona_name_from_profiles(df: pd.DataFrame, cluster_col: str) -> Dict[int, str]:
    """
    Heuristic naming to approximate PDF labels:
    - Fresh Graduate Explorer
    - High Engagement Career Switcher
    - Working Professional Upskiller
    """
    clusters = sorted([int(x) for x in df[cluster_col].dropna().unique().tolist()])
    fallback = [
        "Fresh Graduate Explorer",
        "High Engagement Career Switcher",
        "Working Professional Upskiller",
    ]
    if len(clusters) == 0:
        return {}

    if ("Segmen_karir" not in df.columns) and ("Motivasi_cluster" not in df.columns):
        return {c: fallback[i % len(fallback)] for i, c in enumerate(clusters)}

    rows = []
    for c in clusters:
        sub = df[df[cluster_col] == c]
        seg_top = ""
        mot_top = ""
        if "Segmen_karir" in sub.columns and len(sub) > 0:
            try:
                seg_top = sub["Segmen_karir"].astype(str).value_counts().idxmax()
            except Exception:
                seg_top = ""
        if "Motivasi_cluster" in sub.columns and len(sub) > 0:
            try:
                mot_top = sub["Motivasi_cluster"].astype(str).value_counts().idxmax()
            except Exception:
                mot_top = ""
        rows.append((c, _safe_str(seg_top), _safe_str(mot_top), len(sub)))

    prof = pd.DataFrame(rows, columns=["cluster", "seg_top", "mot_top", "n"]).sort_values("n", ascending=False)

    used = set()
    names: Dict[int, str] = {}

    def pick_cluster(mask: pd.Series) -> Optional[int]:
        cand = prof[mask].sort_values("n", ascending=False)
        for cl in cand["cluster"].tolist():
            if cl not in used:
                used.add(cl)
                return int(cl)
        return None

    c_fresh = pick_cluster(prof["seg_top"].str.contains("Fresh|Graduate|Mahasiswa|Pelajar", case=False, na=False))
    c_switch = pick_cluster(
        prof["mot_top"].str.contains("Switch|career|Pindah", case=False, na=False)
        | prof["seg_top"].str.contains("Switch|career|Pindah", case=False, na=False)
    )
    c_work = pick_cluster(
        prof["seg_top"].str.contains("Working|Professional|Karyawan|Employee|Staff", case=False, na=False)
        | prof["mot_top"].str.contains("Upgrade|Upskill|Skill|Naik", case=False, na=False)
    )

    order = [c_fresh, c_switch, c_work]
    labels = [
        "Fresh Graduate Explorer",
        "High Engagement Career Switcher",
        "Working Professional Upskiller",
    ]
    for cl, lab in zip(order, labels):
        if cl is not None:
            names[int(cl)] = lab

    for cl in clusters:
        if int(cl) not in names:
            # fill remaining unique label
            for lab in fallback:
                if lab not in names.values():
                    names[int(cl)] = lab
                    break
            if int(cl) not in names:
                names[int(cl)] = f"Persona {int(cl)}"

    return names

def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    idx = np.argsort(-y_score)
    top = idx[: int(k)]
    y_top = y_true[top]
    precision = float(y_top.sum() / max(1, len(y_top)))
    recall = float(y_top.sum() / max(1, y_true.sum())) if y_true.sum() > 0 else 0.0
    return precision, recall

def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    base = float(y_true.mean()) if len(y_true) else 0.0
    prec, _ = precision_recall_at_k(y_true, y_score, int(k))
    return float((prec / base) if base > 0 else 0.0)

def signature(df_in: pd.DataFrame) -> Tuple:
    idx = df_in.index.to_numpy()
    head = tuple(idx[:5].tolist()) if len(idx) else ()
    tail = tuple(idx[-5:].tolist()) if len(idx) else ()
    return (len(df_in), head, tail)

def _chart_distribution_target(y: pd.Series, title="Distribusi target"):
    df_t = pd.DataFrame(
        {"target": ["0", "1"], "count": [int((y == 0).sum()), int((y == 1).sum())]}
    )
    ch = alt.Chart(df_t).mark_bar().encode(
        x=alt.X("target:N", title="Target"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=[alt.Tooltip("target:N"), alt.Tooltip("count:Q", format=",.0f")],
    ).properties(height=240, title=title)
    return ch

def _chart_count_by_col(df: pd.DataFrame, col: str, top_n=20, title=None):
    if col not in df.columns:
        return None
    vc = df[col].astype(str).value_counts().head(int(top_n)).reset_index()
    vc.columns = [col, "count"]
    ch = alt.Chart(vc).mark_bar().encode(
        y=alt.Y(f"{col}:N", sort="-x", title=col),
        x=alt.X("count:Q", title="Count"),
        tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("count:Q", format=",.0f")],
    ).properties(height=min(420, 18 * len(vc) + 80), title=title or f"Top {top_n} {col} (count)")
    return ch

# ----------------------------
# Title
# ----------------------------
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

# ----------------------------
# Sidebar: data source
# ----------------------------
with st.sidebar:
    st.header("Data source")
    source_choice = st.radio("Choose", ["Upload file (CSV)", "Repo file (path)"], key="src_choice")

    uploaded = None
    repo_path = None

    if source_choice == "Upload file (CSV)":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
    else:
        repo_path = st.text_input("Path (contoh: raw_data/raw_data.csv)", value="", key="repo_path_input")

df, src = load_data(uploaded, repo_path)
if df is None:
    st.info("Upload CSV atau isi path repo dulu.")
    st.stop()

df = _normalize_cols(df)
df = _ensure_month_col(df)

# ----------------------------
# Target settings (critical)
# ----------------------------
with st.sidebar:
    st.header("Target setup (biar match PDF)")

    target_col_default = "Penyaluran Kerja" if "Penyaluran Kerja" in df.columns else df.columns[0]
    target_col = st.selectbox(
        "Target column",
        options=_safe_cols(df),
        index=int(df.columns.get_loc(target_col_default)) if target_col_default in df.columns else 0,
        key="target_col_sel",
    )

    target_mode_ui = st.radio(
        "Target Mode",
        options=["Tertarik = 1", "Tidak Tertarik = 1", "Manual (pilih value positif)"],
        key="target_mode_radio",
        help=(
            "Kalau PDF hasilnya timpang (positif kecil), biasanya target=minoritas (contoh: Tidak Tertarik=1 untuk intervensi). "
            "Kalau mapping string bikin hasil 0 semua, pakai Manual."
        ),
    )

manual_pos_val = None
mode = "TERTARIK_AS_POSITIVE"
if target_mode_ui == "Tertarik = 1":
    mode = "TERTARIK_AS_POSITIVE"
elif target_mode_ui == "Tidak Tertarik = 1":
    mode = "TIDAK_TERTARIK_AS_POSITIVE"
else:
    mode = "MANUAL_VALUE_AS_POSITIVE"
    uniq = sorted(df[target_col].dropna().astype(str).unique().tolist()) if target_col in df.columns else []
    if len(uniq) == 0:
        st.sidebar.warning("Tidak ada value unik untuk manual mapping.")
    else:
        manual_pos_val = st.sidebar.selectbox("Nilai POSITIF (jadi 1)", options=uniq, key="manual_pos_val")

y, y_desc = make_target(df, target_col=target_col, mode=mode, manual_positive_value=manual_pos_val)
if y is None:
    st.error(y_desc)
    st.stop()

n_all, n_pos, n_neg, pos_rate = target_summary(y)
st.caption(f"**{y_desc}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{n_all:,}")
c2.metric("Positives (1)", f"{n_pos:,}")
c3.metric("Negatives (0)", f"{n_neg:,}")
c4.metric("Positive rate", f"{pos_rate*100:.2f}%")

# ----------------------------
# Global filters
# ----------------------------
with st.sidebar:
    st.header("Global filters")
    st.caption("Filter ini mempengaruhi EDA / Clustering / Supervised / Dashboard")

    # eligible filter cols (object-like + known)
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    known_defaults = [c for c in ["Product", "Kategori", "Month", "Segmen_karir", "Motivasi_cluster", "Region"] if c in df.columns]
    default_filter_cols = known_defaults if len(known_defaults) else obj_cols[:3]

    filter_cols = st.multiselect(
        "Pilih kolom untuk filter",
        options=obj_cols,
        default=default_filter_cols,
        key="filter_cols_multi",
    )

    filter_selections: Dict[str, List[str]] = {}
    for c in filter_cols:
        vals = sorted(df[c].dropna().astype(str).unique().tolist())
        filter_selections[c] = st.multiselect(c, options=vals, default=[], key=f"filt_{c}")

df_f = apply_global_filters(df, filter_cols, filter_selections)
y_f, _ = make_target(df_f, target_col=target_col, mode=mode, manual_positive_value=manual_pos_val)

# reset session caches if filter changes
sig_f = signature(df_f)
if st.session_state.get("_sig_f_prev") != sig_f:
    st.session_state["_sig_f_prev"] = sig_f
    for k in [
        "cluster_done", "cluster_sig", "df_clustered", "persona_map",
        "sup_done", "sup_sig", "sup_pipe", "sup_metrics", "sup_curves", "sup_top",
    ]:
        if k in st.session_state:
            del st.session_state[k]

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_eda, tab_cluster, tab_sup, tab_dash = st.tabs(
    ["Overview", "EDA (Target-driven)", "Clustering (Persona)", "Supervised (Top-K Ranking)", "Dashboard Akhir (Bisnis)"]
)

# ----------------------------
# Overview
# ----------------------------
with tab_overview:
    st.subheader("Goal alignment (harus nyambung ke tujuan proyek)")
    st.write(
        """
Proyek ini harus menjawab **3 hal**:
1) **Memahami pola peserta** → EDA target-driven, bisa difilter dan dibreakdown.
2) **Membangun segmentasi persona yang akurat** → clustering 3 persona + profiling + target-rate per persona.
3) **Model prediktif peluang penyaluran kerja** → ranking Top-K + trade-off K + output actionable untuk bisnis
   (strategi akuisisi, desain program, intervensi peserta).
"""
    )

    st.subheader("Data snapshot (setelah filter)")
    st.write(f"Sumber data: **{src}**  | Rows aktif: **{len(df_f):,}**")
    st.dataframe(df_f.head(30), use_container_width=True)

    st.subheader("Distribusi target (setelah filter)")
    st.altair_chart(_chart_distribution_target(y_f, "Distribusi target (filtered)"), use_container_width=True)

    # Quick count charts (optional)
    colA, colB = st.columns(2)
    with colA:
        if "Product" in df_f.columns:
            ch = _chart_count_by_col(df_f, "Product", top_n=15, title="Top Product (count)")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
    with colB:
        if "Segmen_karir" in df_f.columns:
            ch = _chart_count_by_col(df_f, "Segmen_karir", top_n=15, title="Top Segmen_karir (count)")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)

# ----------------------------
# EDA (Target-driven)
# ----------------------------
with tab_eda:
    st.subheader("EDA target-driven (buat memahami pola peserta)")

    obj_cols_f = [c for c in df_f.columns if df_f[c].dtype == "object"]
    if len(obj_cols_f) == 0:
        st.warning("Tidak ada kolom kategorikal (object) setelah filter. EDA breakdown terbatas.")
        st.stop()

    left, right = st.columns([1, 1])

    with left:
        breakdown_col = st.selectbox(
            "Pilih breakdown (groupby)",
            options=obj_cols_f,
            index=0,
            key="eda_breakdown",
        )
        min_count = st.slider("Min count per group", 5, 300, 30, key="eda_min_count")
        top_n = st.slider("Show top-N groups", 5, 80, 30, key="eda_topn")

    df_rate = top_rate_by_group(df_f, y_f, breakdown_col, min_count=min_count, top_n=top_n)
    chart = alt_bar_rate(df_rate, f"Target rate by {breakdown_col} (Top {top_n}, min_count={min_count})")

    if chart is None:
        st.warning("Tidak ada group yang lolos filter min_count. Turunkan min_count atau ubah filter.")
    else:
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            df_rate.rename(columns={"rate_pct": "target_rate_pct"}).drop(columns=["rate"]),
            use_container_width=True,
        )

    st.subheader("Target rate quick views (umum dipakai bisnis)")
    qa, qb, qc = st.columns(3)

    with qa:
        if "Product" in df_f.columns:
            d = top_rate_by_group(df_f, y_f, "Product", min_count=30, top_n=15)
            ch = alt_bar_rate(d, "Target rate by Product (Top 15)")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)

    with qb:
        if "Kategori" in df_f.columns:
            d = top_rate_by_group(df_f, y_f, "Kategori", min_count=30, top_n=15)
            ch = alt_bar_rate(d, "Target rate by Kategori (Top 15)")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)

    with qc:
        if "Region" in df_f.columns:
            d = top_rate_by_group(df_f, y_f, "Region", min_count=30, top_n=15)
            ch = alt_bar_rate(d, "Target rate by Region (Top 15)")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)

    st.subheader("Distribusi target (setelah filter)")
    st.altair_chart(_chart_distribution_target(y_f, "Distribusi target (filtered)"), use_container_width=True)

# ----------------------------
# Clustering (Persona)
# ----------------------------
with tab_cluster:
    st.subheader("Persona clustering (3 cluster, PDF-aligned)")

    cat_cols, num_cols = infer_feature_sets(df_f)
    st.write("**Fitur yang dipakai (sesuai PDF, kalau kolomnya ada):**")
    st.code(f"Categorical: {cat_cols}\nNumeric: {num_cols}", language="text")

    if len(cat_cols) + len(num_cols) == 0:
        st.error("Tidak ada feature PDF-aligned yang ditemukan di dataset ini.")
        st.stop()

    if len(df_f) < 50:
        st.warning("Data terlalu sedikit setelah filter untuk clustering yang stabil. Longgarkan filter.")
        st.stop()

    run_cluster = st.button("Run clustering", key="btn_cluster_run")

    cluster_sig = ("cluster", sig_f, tuple(cat_cols), tuple(num_cols))
    if run_cluster or (st.session_state.get("cluster_done") and st.session_state.get("cluster_sig") == cluster_sig):
        if run_cluster or (not st.session_state.get("cluster_done")) or st.session_state.get("cluster_sig") != cluster_sig:
            preprocess = build_preprocess(cat_cols, num_cols)

            X = prepare_X(df_f, cat_cols, num_cols)
            if X.shape[1] == 0:
                st.error("Feature matrix kosong setelah prepare_X.")
                st.stop()

            pipe = Pipeline([("prep", preprocess)])
            X_enc = pipe.fit_transform(X)

            # 2D embedding (PDF-aligned)
            svd = TruncatedSVD(n_components=2, random_state=_INTERNAL_RANDOM_STATE)
            X_2d = svd.fit_transform(X_enc)

            # Clustering: stable, no random slider
            km = MiniBatchKMeans(
                n_clusters=N_CLUSTERS,
                random_state=_INTERNAL_RANDOM_STATE,
                n_init=20,
                batch_size=1024,
            )
            cluster_id = km.fit_predict(X_enc)

            dfc = df_f.copy()
            dfc["_cluster_id"] = cluster_id
            dfc["_svd1"] = X_2d[:, 0]
            dfc["_svd2"] = X_2d[:, 1]

            mapping = persona_name_from_profiles(dfc, "_cluster_id")
            dfc["Persona"] = dfc["_cluster_id"].map(mapping).astype(str)

            st.session_state["cluster_done"] = True
            st.session_state["cluster_sig"] = cluster_sig
            st.session_state["df_clustered"] = dfc
            st.session_state["persona_map"] = mapping

        dfc = st.session_state["df_clustered"]
        mapping = st.session_state["persona_map"]

        st.success(f"Clustering selesai. Persona map: {mapping}")

        st.subheader("Cluster visualization (TruncatedSVD 2D) — Persona Named")
        tooltip_cols = ["Persona:N"]
        if "Product" in dfc.columns:
            tooltip_cols.append("Product:N")
        if "Kategori" in dfc.columns:
            tooltip_cols.append("Kategori:N")
        if "Segmen_karir" in dfc.columns:
            tooltip_cols.append("Segmen_karir:N")

        scatter = alt.Chart(dfc).mark_circle(size=22, opacity=0.65).encode(
            x=alt.X("_svd1:Q", title="SVD-1"),
            y=alt.Y("_svd2:Q", title="SVD-2"),
            color=alt.Color("Persona:N", legend=alt.Legend(title="Persona")),
            tooltip=tooltip_cols,
        ).properties(height=540)
        st.altair_chart(scatter, use_container_width=True)

        st.subheader("Cluster profiling (untuk akurasi persona & action bisnis)")
        prof_cols = [c for c in ["Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region"] if c in dfc.columns]
        if len(prof_cols) == 0:
            st.info("Kolom profiling (Segmen_karir/Motivasi_cluster/...) tidak tersedia di data ini.")
        else:
            blocks = []
            for p in sorted(dfc["Persona"].astype(str).unique().tolist()):
                sub = dfc[dfc["Persona"].astype(str) == str(p)]
                row = {"Persona": p, "Size": int(len(sub))}
                y_sub, _ = make_target(sub, target_col=target_col, mode=mode, manual_positive_value=manual_pos_val)
                row["Target rate (%)"] = round(float(y_sub.mean()) * 100, 2) if (y_sub is not None and len(sub)) else 0.0
                for c in prof_cols:
                    if c == "Persona":
                        continue
                    try:
                        vc = sub[c].astype(str).value_counts()
                        row[f"Top {c}"] = vc.index[0] if len(vc) else ""
                    except Exception:
                        row[f"Top {c}"] = ""
                blocks.append(row)
            prof_df = pd.DataFrame(blocks).sort_values("Size", ascending=False)
            st.dataframe(prof_df, use_container_width=True)

        st.subheader("Target rate by Persona (validasi bahwa persona nyambung target)")
        y_c, _ = make_target(dfc, target_col=target_col, mode=mode, manual_positive_value=manual_pos_val)
        if y_c is not None:
            dr = top_rate_by_group(dfc, y_c, "Persona", min_count=10, top_n=20)
            ch = alt_bar_rate(dr, "Target rate by Persona (min_count=10)")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)

# ----------------------------
# Supervised (Top-K Ranking)
# ----------------------------
with tab_sup:
    st.subheader("Supervised ranking — Top-K (PDF-aligned, untuk keputusan bisnis)")

    # Join persona if available
    dfm = df_f.copy()
    if st.session_state.get("cluster_done"):
        dfc = st.session_state["df_clustered"]
        # merge by index (safe)
        dfm = dfm.merge(dfc[["_cluster_id", "Persona"]], left_index=True, right_index=True, how="left")

    cat_cols, num_cols = infer_feature_sets(dfm)
    if "Persona" in dfm.columns and "Persona" not in cat_cols:
        cat_cols = cat_cols + ["Persona"]

    if len(cat_cols) + len(num_cols) == 0:
        st.error("Tidak ada feature untuk supervised (cek kolom dataset).")
        st.stop()

    y_m, _desc = make_target(dfm, target_col=target_col, mode=mode, manual_positive_value=manual_pos_val)
    if y_m is None:
        st.error("Target tidak valid untuk supervised.")
        st.stop()

    if y_m.nunique() < 2:
        st.error("Target setelah filter hanya punya 1 kelas. Ubah filter atau ganti Target Mode / Manual mapping.")
        st.stop()

    n_all_m, n_pos_m, n_neg_m, pos_rate_m = target_summary(y_m)
    st.write(f"Rows: **{n_all_m:,}** | Positives: **{n_pos_m:,}** | Positive rate: **{pos_rate_m*100:.2f}%**")

    k_cap = st.slider(
        "Business capacity K (berapa peserta yang bisa diintervensi / di-follow up)",
        min_value=50,
        max_value=min(2000, max(50, int(len(dfm) * 0.8))),
        value=min(200, max(50, int(len(dfm) * 0.1))),
        step=10,
        key="k_capacity",
    )

    colA, colB, colC = st.columns(3)
    with colA:
        test_size = st.slider("test_size (holdout)", 0.1, 0.4, 0.2, 0.05, key="sup_test_size")
    with colB:
        class_weight = st.selectbox("class_weight", options=["balanced", "none"], index=0, key="sup_class_weight")
    with colC:
        c_reg = st.slider("C (regularization)", 0.1, 5.0, 1.0, 0.1, key="sup_C")

    preprocess = build_preprocess(cat_cols, num_cols)
    model = LogisticRegression(
        max_iter=3000,
        class_weight=("balanced" if class_weight == "balanced" else None),
        C=float(c_reg),
        solver="liblinear",
    )
    pipe = Pipeline([("prep", preprocess), ("clf", model)])

    X = prepare_X(dfm, cat_cols, num_cols)
    y_arr = y_m.values.astype(int)

    # guard
    min_class = min(int((y_arr == 0).sum()), int((y_arr == 1).sum()))
    if min_class < 10:
        st.warning(
            "Kelas minoritas terlalu kecil untuk evaluasi yang stabil. "
            "Longgarkan filter / ganti target mode / pastikan dataset sama seperti PDF."
        )
        st.stop()

    run_sup = st.button("Run supervised ranking", key="btn_sup_run")

    sup_sig = ("sup", sig_f, tuple(cat_cols), tuple(num_cols), float(test_size), str(class_weight), float(c_reg), int(k_cap), mode, str(manual_pos_val))
    if run_sup or (st.session_state.get("sup_done") and st.session_state.get("sup_sig") == sup_sig):
        if run_sup or (not st.session_state.get("sup_done")) or st.session_state.get("sup_sig") != sup_sig:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_arr,
                test_size=float(test_size),
                random_state=_INTERNAL_RANDOM_STATE,
                stratify=y_arr,
            )

            pipe.fit(X_train, y_train)
            p_test = pipe.predict_proba(X_test)[:, 1]

            pr_auc = float(average_precision_score(y_test, p_test))
            try:
                roc_auc = float(roc_auc_score(y_test, p_test))
            except Exception:
                roc_auc = float("nan")

            # curves
            max_k = int(min(len(y_test), 2000))
            ks = np.unique(np.clip(np.linspace(50, max(50, max_k), 30).astype(int), 1, None))
            precs, recs, lifts = [], [], []
            for k in ks:
                pr_k, rc_k = precision_recall_at_k(y_test, p_test, int(k))
                lf_k = lift_at_k(y_test, p_test, int(k))
                precs.append(pr_k)
                recs.append(rc_k)
                lifts.append(lf_k)

            # Top-K full
            p_full = pipe.predict_proba(X)[:, 1]
            order = np.argsort(-p_full)
            top_idx = order[: int(k_cap)]
            df_top = dfm.iloc[top_idx].copy()
            df_top["score"] = p_full[top_idx]
            df_top["pred_rank"] = np.arange(1, len(df_top) + 1)

            st.session_state["sup_done"] = True
            st.session_state["sup_sig"] = sup_sig
            st.session_state["sup_pipe"] = pipe
            st.session_state["sup_metrics"] = {"pr_auc": pr_auc, "roc_auc": roc_auc}
            st.session_state["sup_curves"] = pd.DataFrame(
                {"k": ks, "precision_at_k": precs, "recall_at_k": recs, "lift_at_k": lifts}
            )
            st.session_state["sup_top"] = df_top.sort_values("pred_rank")

        # Render results
        met = st.session_state["sup_metrics"]
        curves = st.session_state["sup_curves"]
        df_top = st.session_state["sup_top"]

        st.subheader("Model quality (ranking-oriented)")
        m1, m2 = st.columns(2)
        m1.metric("PR-AUC (Average Precision)", f"{met['pr_auc']:.4f}")
        roc = met.get("roc_auc", np.nan)
        roc_text = "N/A" if (roc is None or (isinstance(roc, float) and np.isnan(roc))) else f"{roc:.4f}"
        m2.metric("ROC-AUC (opsional)", roc_text)

        st.caption("Untuk Top-K decision, fokus utama: PR-AUC + Precision@K/Recall@K/Lift@K (lebih nyambung ke kapasitas bisnis).")

        st.subheader("Trade-off curve: Precision@K, Recall@K, Lift@K")
        c_left, c_right = st.columns(2)

        ch1 = alt.Chart(curves).mark_line(point=True).encode(
            x=alt.X("k:Q", title="K"),
            y=alt.Y("precision_at_k:Q", title="Precision@K"),
            tooltip=[alt.Tooltip("k:Q"), alt.Tooltip("precision_at_k:Q", format=".3f")],
        ).properties(height=260, title="Precision@K")
        c_left.altair_chart(ch1, use_container_width=True)

        ch2 = alt.Chart(curves).mark_line(point=True).encode(
            x=alt.X("k:Q", title="K"),
            y=alt.Y("recall_at_k:Q", title="Recall@K"),
            tooltip=[alt.Tooltip("k:Q"), alt.Tooltip("recall_at_k:Q", format=".3f")],
        ).properties(height=260, title="Recall@K")
        c_right.altair_chart(ch2, use_container_width=True)

        ch3 = alt.Chart(curves).mark_line(point=True).encode(
            x=alt.X("k:Q", title="K"),
            y=alt.Y("lift_at_k:Q", title="Lift@K"),
            tooltip=[alt.Tooltip("k:Q"), alt.Tooltip("lift_at_k:Q", format=".2f")],
        ).properties(height=260, title="Lift@K (efektivitas vs baseline)")
        st.altair_chart(ch3, use_container_width=True)

        st.subheader("Top-K output (untuk eksekusi bisnis)")
        show_cols = [c for c in ["pred_rank", "score", "Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region", "Umur", "Month"] if c in df_top.columns]
        if len(show_cols) == 0:
            show_cols = ["pred_rank", "score"]

        st.dataframe(df_top[show_cols].head(500), use_container_width=True)

        csv_bytes = df_top[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Top-K CSV",
            data=csv_bytes,
            file_name="topk_ranking.csv",
            mime="text/csv",
            key="dl_topk",
        )

    else:
        st.info("Klik **Run supervised ranking** untuk membangun model ranking dan Top-K output.")

# ----------------------------
# Dashboard Akhir (Bisnis)
# ----------------------------
with tab_dash:
    st.subheader("Dashboard Akhir (Bisnis) — ringkas & actionable")

    # Build df for dashboard: include persona if available
    df_dash = df_f.copy()
    if st.session_state.get("cluster_done"):
        dfc = st.session_state["df_clustered"]
        df_dash = df_dash.merge(dfc[["_cluster_id", "Persona"]], left_index=True, right_index=True, how="left")

    y_dash, _ = make_target(df_dash, target_col=target_col, mode=mode, manual_positive_value=manual_pos_val)
    if y_dash is None:
        st.error("Target tidak valid untuk dashboard.")
        st.stop()

    n_all_d, n_pos_d, n_neg_d, base_rate = target_summary(y_dash)

    st.write("### KPI utama")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows aktif", f"{n_all_d:,}")
    k2.metric("Positives", f"{n_pos_d:,}")
    k3.metric("Positive rate", f"{base_rate*100:.2f}%")
    k4.metric("K (capacity)", f"{int(st.session_state.get('k_capacity', 200)):,}")
    if st.session_state.get("sup_done"):
        met = st.session_state["sup_metrics"]
        k5.metric("PR-AUC", f"{met['pr_auc']:.4f}")
    else:
        k5.metric("PR-AUC", "Run supervised dulu")

    st.divider()

    st.write("### 1) Pola peserta yang paling berdampak (target rate breakdown)")
    a, b = st.columns(2)

    with a:
        if "Persona" in df_dash.columns:
            d = top_rate_by_group(df_dash, y_dash, "Persona", min_count=10, top_n=20)
            ch = alt_bar_rate(d, "Target rate by Persona")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
        elif st.session_state.get("cluster_done") is False:
            st.info("Jalankan clustering dulu kalau mau breakdown by Persona.")

    with b:
        # pick the most relevant available breakdown automatically
        priority_cols = ["Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Region", "Channel", "Month"]
        pick = None
        for c in priority_cols:
            if c in df_dash.columns:
                pick = c
                break
        if pick is not None:
            d = top_rate_by_group(df_dash, y_dash, pick, min_count=30, top_n=20)
            ch = alt_bar_rate(d, f"Target rate by {pick}")
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Tidak ada kolom breakdown populer yang tersedia di dataset ini.")

    st.write("### 2) Output keputusan bisnis (Top-K ranking & distribusi)")
    if st.session_state.get("sup_done"):
        df_top = st.session_state["sup_top"].copy()
        k_cap_now = int(st.session_state.get("k_capacity", 200))

        # nearest curve stats
        curves = st.session_state["sup_curves"]
        # nearest K point
        nearest_idx = (curves["k"] - k_cap_now).abs().idxmin()
        nearest = curves.loc[nearest_idx]

        st.markdown(
            f"""
- **K = {k_cap_now:,}** = kapasitas intervensi / follow-up.
- Baseline positive rate = **{base_rate*100:.2f}%**.
- Estimasi di sekitar K≈{int(nearest["k"]):,}:  
  - Precision@K ≈ **{nearest["precision_at_k"]*100:.2f}%**  
  - Recall@K ≈ **{nearest["recall_at_k"]*100:.2f}%**  
  - Lift@K ≈ **{nearest["lift_at_k"]:.2f}×** (efektivitas vs random pick)
"""
        )

        # Distribution of Top-K
        c1, c2 = st.columns(2)

        with c1:
            if "Persona" in df_top.columns:
                dist = df_top["Persona"].astype(str).value_counts().reset_index()
                dist.columns = ["Persona", "count"]
                ch = alt.Chart(dist).mark_bar().encode(
                    y=alt.Y("Persona:N", sort="-x"),
                    x=alt.X("count:Q", title="Count"),
                    tooltip=[alt.Tooltip("Persona:N"), alt.Tooltip("count:Q", format=",.0f")],
                ).properties(height=260, title="Distribusi Top-K by Persona")
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("Top-K belum punya Persona (jalankan clustering sebelum supervised untuk persona-aware ranking).")

        with c2:
            # choose a business dimension for top-k distribution
            dim = None
            for c in ["Product", "Kategori", "Segmen_karir", "Region", "Channel", "Month"]:
                if c in df_top.columns:
                    dim = c
                    break
            if dim is not None:
                dist = df_top[dim].astype(str).value_counts().head(20).reset_index()
                dist.columns = [dim, "count"]
                ch = alt.Chart(dist).mark_bar().encode(
                    y=alt.Y(f"{dim}:N", sort="-x"),
                    x=alt.X("count:Q", title="Count"),
                    tooltip=[alt.Tooltip(f"{dim}:N"), alt.Tooltip("count:Q", format=",.0f")],
                ).properties(height=320, title=f"Distribusi Top-K by {dim} (Top 20)")
                st.altair_chart(ch, use_container_width=True)
            else:
                st.info("Tidak ada dimensi bisnis populer di dataset untuk distribusi Top-K.")

        st.write("### 3) Daftar prioritas (Top-K) — ringkas")
        show_cols = [c for c in ["pred_rank", "score", "Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region", "Month"] if c in df_top.columns]
        if len(show_cols) == 0:
            show_cols = ["pred_rank", "score"]
        st.dataframe(df_top[show_cols].head(200), use_container_width=True)

        csv_bytes = df_top[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Top-K (dashboard) CSV",
            data=csv_bytes,
            file_name="topk_dashboard.csv",
            mime="text/csv",
            key="dl_topk_dash",
        )

        st.success(
            "Dashboard akhir ini sudah nyambung ke tujuan proyek: pola peserta → persona → ranking Top-K → output untuk strategi akuisisi/program/intervensi."
        )
    else:
        st.warning("Dashboard bisnis butuh hasil supervised. Jalankan tab **Supervised (Top-K Ranking)** dulu.")

# ----------------------------
# Footer: mismatch guidance
# ----------------------------
st.caption(
    "Catatan: Kalau hasil masih beda dengan PDF, biasanya karena (1) dataset tidak identik, (2) definisi target/mapping berbeda, "
    "(3) filter aktif berbeda, atau (4) kolom PDF-aligned tidak lengkap. "
    "Di app ini kamu bisa lihat distribusi target & breakdown untuk cepat deteksi mismatch."
)
