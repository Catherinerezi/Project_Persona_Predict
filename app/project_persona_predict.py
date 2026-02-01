# app.py
# Persona Segmentation & Placement Prediction (PDF-aligned) — single-file Streamlit
# Charts: Altair only (no Plotly). Clustering: 3 clusters + persona naming. Supervised: Top-K dashboard + Dashboard Akhir (Bisnis).

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# Page config
st.set_page_config(
    page_title="Persona Segmentation & Placement Prediction",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

# Altair config (biar chart nggak kepotong / aman di dark mode streamlit)
def _apply_altair_theme():
    # Make charts more robust in Streamlit columns + long labels
    theme = {
        "config": {
            "view": {"strokeOpacity": 0},
            "axis": {
                "labelLimit": 260,
                "titlePadding": 10,
                "labelPadding": 6,
                "tickSize": 3,
            },
            "title": {"fontSize": 16, "anchor": "start"},
            "legend": {"labelLimit": 260},
        }
    }
    alt.themes.register("streamlit_safe", lambda: theme)
    alt.themes.enable("streamlit_safe")

_apply_altair_theme()

# Helpers
def _safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def _norm_str_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .fillna("")
        .str.strip()
        .str.lower()
    )

def load_data(uploaded_file, repo_path: str | None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "upload"
    else:
        if not repo_path:
            return None, None
        df = pd.read_csv(repo_path)
        source = "repo"
    return df, source

def apply_global_filters(df: pd.DataFrame, filter_cols: list[str], selections: dict[str, list]):
    out = df.copy()
    for c in filter_cols:
        chosen = selections.get(c, [])
        if chosen:
            out = out[out[c].astype(str).isin([str(v) for v in chosen])]
    return out

def infer_feature_sets(df: pd.DataFrame):
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

def _is_binary_numeric_target(s: pd.Series) -> bool:
    try:
        vals = pd.to_numeric(s, errors="coerce").dropna().unique()
        vals = set(vals.tolist())
        return vals.issubset({0, 1}) and len(vals) > 0
    except Exception:
        return False

def make_target(
    df: pd.DataFrame,
    target_col="Penyaluran Kerja",
    mode="TERTARIK_AS_POSITIVE",
    manual_positive_value: str | None = None,
):
    """
    Returns y (0/1) and description.

    Critical FIX:
    - "Tidak Tertarik" contains substring "Tertarik".
      So we must NOT use naive contains("Tertarik") for positives.
    """
    if target_col not in df.columns:
        return None, f"Target column '{target_col}' tidak ditemukan."

    s_raw = df[target_col]

    # Case 1: numeric binary already
    if _is_binary_numeric_target(s_raw):
        y = pd.to_numeric(s_raw, errors="coerce").fillna(0).astype(int)
        desc = f"Target numeric (0/1) dari kolom '{target_col}'"
        return y, desc

    s = _norm_str_series(s_raw)

    # Case 2: manual mapping (exact match after normalization)
    if mode == "MANUAL":
        if manual_positive_value is None:
            return None, "Mode MANUAL butuh manual_positive_value."
        mv = str(manual_positive_value).strip().lower()
        y = (s == mv).astype(int)
        desc = f"Target=1 jika '{manual_positive_value}' (manual exact match)"
        return y, desc

    # Case 3: string semantic mapping (safe)
    # Define:
    # - "tidak tertarik" / contains "tidak" => negative for TERTARIK mode, positive for TIDAK mode
    # - contains "tertarik" AND NOT contains "tidak" => positive for TERTARIK mode
    has_tidak = s.str.contains("tidak", regex=False)
    has_tertarik = s.str.contains("tertarik", regex=False)

    if mode == "TERTARIK_AS_POSITIVE":
        y = (has_tertarik & (~has_tidak)).astype(int)
        desc = "Target=1 jika 'Tertarik' (dan bukan 'Tidak ...') — safe mapping"
        return y, desc

    if mode == "TIDAK_TERTARIK_AS_POSITIVE":
        y = (has_tidak).astype(int)
        desc = "Target=1 jika mengandung 'Tidak' (at-risk/intervensi) — safe mapping"
        return y, desc

    return None, f"Mode target tidak dikenal: {mode}"

def target_summary(y: pd.Series):
    vc = y.value_counts(dropna=False).to_dict()
    n = int(y.shape[0])
    pos = int(vc.get(1, 0))
    neg = int(vc.get(0, 0))
    pos_rate = (pos / n) if n else 0.0
    return n, pos, neg, pos_rate

def top_rate_by_group(df: pd.DataFrame, y: pd.Series, group_col: str, min_count=30, top_n=30):
    tmp = df[[group_col]].copy()
    tmp["_y"] = y.values
    g = tmp.groupby(group_col, dropna=False)["_y"].agg(["count", "sum"]).reset_index()
    g = g.rename(columns={group_col: "Group", "count": "Total", "sum": "Positives"})
    g["rate"] = np.where(g["Total"] > 0, g["Positives"] / g["Total"], 0.0)
    g = g[g["Total"] >= min_count].sort_values("rate", ascending=False).head(top_n)
    g["rate_pct"] = (g["rate"] * 100).round(2)
    return g

def alt_bar_rate(df_rate: pd.DataFrame, title: str, max_height=520):
    if df_rate is None or df_rate.empty:
        return None
    h = min(max_height, 26 * len(df_rate) + 90)
    base = (
        alt.Chart(df_rate)
        .mark_bar()
        .encode(
            y=alt.Y("Group:N", sort="-x", title="Group"),
            x=alt.X("rate_pct:Q", title="Target rate (%)"),
            tooltip=["Group:N", "Total:Q", "Positives:Q", "rate_pct:Q"],
        )
        .properties(title=title, height=h)
    )
    text = alt.Chart(df_rate).mark_text(align="left", dx=6).encode(
        y=alt.Y("Group:N", sort="-x"),
        x=alt.X("rate_pct:Q"),
        text=alt.Text("rate_pct:Q", format=".2f"),
    )
    return (base + text).properties(autosize={"type": "fit", "contains": "padding"})

def alt_dist_target(y: pd.Series, title="Distribusi target"):
    df_t = pd.DataFrame({"target": [0, 1], "count": [int((y == 0).sum()), int((y == 1).sum())]})
    ch = (
        alt.Chart(df_t)
        .mark_bar()
        .encode(
            x=alt.X("target:N", title="Target"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["target:N", "count:Q"],
        )
        .properties(height=260, title=title, autosize={"type": "fit", "contains": "padding"})
    )
    return ch

def _make_ohe():
    # sklearn compatibility: sparse_output (new) vs sparse (old)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def build_preprocess(cat_cols: list[str], num_cols: list[str]):
    transformers = []
    if cat_cols:
        transformers.append(("cat", _make_ohe(), cat_cols))
    if num_cols:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def prepare_X(df_in: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    """
    Prevent ValueError in ColumnTransformer/OneHotEncoder:
    - cat: fillna + cast to str
    - num: to_numeric + fillna
    """
    cols = list(cat_cols) + list(num_cols)
    X = df_in[cols].copy()

    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X

def persona_name_from_profiles(df: pd.DataFrame, cluster_col: str):
    """
    Heuristic naming to approximate PDF labels:
    - Fresh Graduate Explorer
    - High Engagement Career Switcher
    - Working Professional Upskiller
    """
    names = {}
    clusters = sorted(df[cluster_col].dropna().unique().tolist())

    fallback = [
        "Fresh Graduate Explorer",
        "High Engagement Career Switcher",
        "Working Professional Upskiller",
    ]

    if "Segmen_karir" not in df.columns and "Motivasi_cluster" not in df.columns:
        return {c: fallback[i % len(fallback)] for i, c in enumerate(clusters)}

    rows = []
    for c in clusters:
        sub = df[df[cluster_col] == c]
        seg_top = sub["Segmen_karir"].value_counts().idxmax() if ("Segmen_karir" in sub.columns and len(sub)) else ""
        mot_top = sub["Motivasi_cluster"].value_counts().idxmax() if ("Motivasi_cluster" in sub.columns and len(sub)) else ""
        rows.append((c, _safe_str(seg_top), _safe_str(mot_top), len(sub)))
    prof = pd.DataFrame(rows, columns=["cluster", "seg_top", "mot_top", "n"])

    used = set()

    def pick_cluster(mask):
        cand = prof[mask].sort_values("n", ascending=False)
        for cl in cand["cluster"].tolist():
            if cl not in used:
                used.add(cl)
                return cl
        return None

    c_fresh = pick_cluster(prof["seg_top"].str.contains("fresh|graduate|mahasiswa|pelajar", case=False, na=False))
    c_switch = pick_cluster(
        prof["mot_top"].str.contains("switch|career|pindah", case=False, na=False)
        | prof["seg_top"].str.contains("switch|career|pindah", case=False, na=False)
    )
    c_work = pick_cluster(
        prof["seg_top"].str.contains("working|professional|karyawan|employee", case=False, na=False)
        | prof["mot_top"].str.contains("upgrade|upskill|skill", case=False, na=False)
    )

    order = [c_fresh, c_switch, c_work]
    labels = [
        "Fresh Graduate Explorer",
        "High Engagement Career Switcher",
        "Working Professional Upskiller",
    ]
    for cl, lab in zip(order, labels):
        if cl is not None:
            names[cl] = lab

    for cl in clusters:
        if cl not in names:
            for lab in fallback:
                if lab not in names.values():
                    names[cl] = lab
                    break
            if cl not in names:
                names[cl] = f"Persona {cl}"

    return names

def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    idx = np.argsort(-y_score)
    top = idx[:k]
    y_top = y_true[top]
    precision = (y_top.sum() / max(1, len(y_top)))
    recall = (y_top.sum() / max(1, y_true.sum())) if y_true.sum() > 0 else 0.0
    return float(precision), float(recall)

def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    base = y_true.mean() if len(y_true) else 0.0
    prec, _ = precision_recall_at_k(y_true, y_score, k)
    return float((prec / base) if base > 0 else 0.0)

def _signature(df_in: pd.DataFrame):
    idx = df_in.index.to_numpy()
    head = tuple(idx[:5].tolist()) if len(idx) else ()
    tail = tuple(idx[-5:].tolist()) if len(idx) else ()
    return (len(df_in), head, tail)

def _nan_safe_metric_value(x):
    if x is None:
        return "N/A"
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return "N/A"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "N/A"

# UI: Title + Sidebar Data Source
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

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

df = df.copy()
df.columns = [c.strip() for c in df.columns]

# Target settings
with st.sidebar:
    st.header("Target setup (biar match PDF)")

    target_col = st.selectbox(
        "Target column",
        options=[c for c in df.columns],
        index=(df.columns.get_loc("Penyaluran Kerja") if "Penyaluran Kerja" in df.columns else 0),
        key="target_col_sel",
    )

    target_mode = st.radio(
        "Target Mode",
        options=["Tertarik = 1", "Tidak Tertarik = 1", "Manual (pilih value positif)"],
        key="target_mode_radio",
        help=(
            "PENTING: 'Tidak Tertarik' mengandung substring 'Tertarik'. "
            "Di app ini mapping sudah SAFE (tidak ketipu substring)."
        ),
    )

manual_pos = None
mode = "TERTARIK_AS_POSITIVE"
if target_mode == "Tertarik = 1":
    mode = "TERTARIK_AS_POSITIVE"
elif target_mode == "Tidak Tertarik = 1":
    mode = "TIDAK_TERTARIK_AS_POSITIVE"
else:
    mode = "MANUAL"
    with st.sidebar:
        uniq = sorted(df[target_col].dropna().astype(str).unique().tolist()) if target_col in df.columns else []
        if uniq:
            manual_pos = st.selectbox("Nilai POSITIF (jadi 1)", options=uniq, key="manual_pos_val")
        else:
            st.warning("Kolom target kosong / tidak ada value unik untuk dipilih.")

y, y_desc = make_target(df, target_col=target_col, mode=mode, manual_positive_value=manual_pos)
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

# Global filters
with st.sidebar:
    st.header("Global filters")
    st.caption("Pilih kolom untuk filter → pilih value-nya (mempengaruhi EDA/Clustering/Supervised).")

    # Offer filter columns: object/category + some common dimension columns even if not object
    cand_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            cand_cols.append(c)
        elif c in ["Month", "Batch", "Product", "Kategori", "Channel", "Region", "Segmen_karir", "Motivasi_cluster", "Sumber"]:
            cand_cols.append(c)

    cand_cols = sorted(list(dict.fromkeys(cand_cols)))

    default_filter_cols = [c for c in ["Product", "Kategori", "Month", "Segmen_karir", "Motivasi_cluster", "Region"] if c in cand_cols]

    filter_cols = st.multiselect(
        "Pilih kolom untuk filter",
        options=cand_cols,
        default=default_filter_cols,
        key="filter_cols_multi",
    )

    filter_selections = {}
    for c in filter_cols:
        opts = sorted(df[c].dropna().astype(str).unique().tolist())
        filter_selections[c] = st.multiselect(
            f"{c}",
            options=opts,
            default=[],
            key=f"filt_{c}",
        )

df_f = apply_global_filters(df, filter_cols, filter_selections)
y_f, _ = make_target(df_f, target_col=target_col, mode=mode, manual_positive_value=manual_pos)

st.divider()


# Reset caches if filters change
sig_f = _signature(df_f)
if st.session_state.get("sig_f_prev") != sig_f:
    st.session_state["sig_f_prev"] = sig_f
    for k in [
        "cluster_done", "cluster_sig", "df_clustered", "persona_map",
        "sup_done", "sup_sig", "sup_pipe", "sup_metrics", "sup_curves", "sup_top"
    ]:
        if k in st.session_state:
            del st.session_state[k]

# Tabs
tab_overview, tab_eda, tab_cluster, tab_sup, tab_dash = st.tabs(
    ["Overview", "EDA (Target-driven)", "Clustering (Persona)", "Supervised (Top-K Ranking)", "Dashboard Akhir (Bisnis)"]
)

# Overview
with tab_overview:
    st.subheader("Project goal alignment")
    st.write(
        """
Proyek ini harus menjawab 3 hal:
1) **Memahami pola peserta** (EDA yang bisa di-breakdown & difilter),
2) **Membangun segmentasi persona yang akurat** (clustering 3 persona + profiling),
3) **Model prediktif peluang penyaluran kerja untuk keputusan bisnis** (Top-K ranking + dashboard action: siapa diprioritaskan, trade-off K, dan distribusi per segmen/persona).
"""
    )

    st.subheader("Data snapshot (setelah filter)")
    st.write(f"Sumber data: **{src}**  | Rows aktif: **{len(df_f):,}**")
    st.dataframe(df_f.head(30), use_container_width=True)

    st.subheader("Distribusi target (setelah filter) — sanity check mismatch PDF")
    st.altair_chart(alt_dist_target(y_f, "Distribusi target (setelah filter)"), use_container_width=True)

# EDA (Target-driven)
with tab_eda:
    st.subheader("EDA yang selaras tujuan (target-driven)")

    # Quick views (yang biasanya dipakai bisnis)
    st.markdown("### Target rate quick views")
    q1, q2, q3 = st.columns(3)

    def _quick_rate(col, title, min_count=30, top_n=15):
        if col in df_f.columns:
            g = top_rate_by_group(df_f, y_f, col, min_count=min_count, top_n=top_n)
            ch = alt_bar_rate(g, title, max_height=420)
            return g, ch
        return None, None

    g_prod, ch_prod = _quick_rate("Product", "Target rate by Product (Top 15)", min_count=30, top_n=15)
    g_kat, ch_kat = _quick_rate("Kategori", "Target rate by Kategori (Top 15)", min_count=30, top_n=15)
    g_chan, ch_chan = _quick_rate("Channel", "Target rate by Channel (Top 15)", min_count=30, top_n=15)

    with q1:
        if ch_prod is not None:
            st.altair_chart(ch_prod, use_container_width=True)
        else:
            st.info("Kolom 'Product' tidak ada / tidak lolos min_count.")
    with q2:
        if ch_kat is not None:
            st.altair_chart(ch_kat, use_container_width=True)
        else:
            st.info("Kolom 'Kategori' tidak ada / tidak lolos min_count.")
    with q3:
        if ch_chan is not None:
            st.altair_chart(ch_chan, use_container_width=True)
        else:
            st.info("Kolom 'Channel' tidak ada / tidak lolos min_count.")

    st.divider()

    # Flexible breakdown
    st.markdown("### Breakdown interaktif (pilih groupby sendiri)")
    obj_cols = [c for c in df_f.columns if df_f[c].dtype == "object"] or list(df_f.columns)
    breakdown_col = st.selectbox("Pilih breakdown (groupby)", options=obj_cols, key="eda_breakdown")

    left, right = st.columns([1, 1])
    with left:
        min_count = st.slider("Min count per group", 5, 300, 30, key="eda_min_count")
    with right:
        top_n = st.slider("Show top-N groups", 5, 80, 30, key="eda_topn")

    if breakdown_col:
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

    st.subheader("Distribusi target (setelah filter)")
    st.altair_chart(alt_dist_target(y_f, "Distribusi target (setelah filter)"), use_container_width=True)

# Clustering (Persona)
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

            # Encode
            enc_pipe = Pipeline([("prep", preprocess)])
            X_enc = enc_pipe.fit_transform(X)

            # 2D embedding (for viz)
            svd = TruncatedSVD(n_components=2, random_state=42)
            X_2d = svd.fit_transform(X_enc)

            # KMeans 3 clusters (no random slider; stable)
            km = MiniBatchKMeans(n_clusters=3, n_init=20, batch_size=1024, random_state=42)
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

        st.markdown("### Cluster visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)")
        tooltip_cols = ["Persona:N"]
        if "Product" in dfc.columns:
            tooltip_cols.append("Product:N")
        if "Kategori" in dfc.columns:
            tooltip_cols.append("Kategori:N")

        scatter = (
            alt.Chart(dfc)
            .mark_circle(size=26, opacity=0.7)
            .encode(
                x=alt.X("_svd1:Q", title="SVD-1"),
                y=alt.Y("_svd2:Q", title="SVD-2"),
                color=alt.Color("Persona:N", legend=alt.Legend(title="Persona")),
                tooltip=tooltip_cols,
            )
            .properties(height=560, autosize={"type": "fit", "contains": "padding"})
        )
        st.altair_chart(scatter, use_container_width=True)

        st.divider()

        st.markdown("### Cluster profiling (untuk akurasi persona & action bisnis)")
        prof_cols = [c for c in ["Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel"] if c in dfc.columns]

        blocks = []
        for p in sorted(dfc["Persona"].unique().tolist()):
            sub = dfc[dfc["Persona"] == p]
            row = {"Persona": p, "Size": int(len(sub))}
            y_sub, _ = make_target(sub, target_col=target_col, mode=mode, manual_positive_value=manual_pos)
            row["Target rate (%)"] = round(float(y_sub.mean() * 100), 2) if len(sub) else 0.0

            for c in prof_cols:
                if c == "Persona":
                    continue
                vc = sub[c].astype(str).value_counts()
                row[f"Top {c}"] = vc.index[0] if len(vc) else ""
            blocks.append(row)

        prof_df = pd.DataFrame(blocks).sort_values("Size", ascending=False)
        st.dataframe(prof_df, use_container_width=True)

        st.markdown("### Target rate by Persona (validasi persona nyambung target)")
        y_c, _ = make_target(dfc, target_col=target_col, mode=mode, manual_positive_value=manual_pos)
        rate_p = top_rate_by_group(dfc, y_c, "Persona", min_count=10, top_n=10)
        chp = alt_bar_rate(rate_p, "Target rate by Persona (min_count=10)")
        if chp is not None:
            st.altair_chart(chp, use_container_width=True)

# Supervised (Top-K Ranking)
with tab_sup:
    st.subheader("Supervised ranking — Top-K (PDF-aligned)")

    dfm = df_f.copy()
    if st.session_state.get("cluster_done"):
        dfc = st.session_state["df_clustered"]
        dfm = dfm.merge(dfc[["_cluster_id", "Persona"]], left_index=True, right_index=True, how="left")

    cat_cols, num_cols = infer_feature_sets(dfm)
    if "Persona" in dfm.columns and "Persona" not in cat_cols:
        cat_cols = cat_cols + ["Persona"]

    if len(cat_cols) + len(num_cols) == 0:
        st.error("Tidak ada feature untuk supervised (cek kolom dataset).")
        st.stop()

    y_m, _ = make_target(dfm, target_col=target_col, mode=mode, manual_positive_value=manual_pos)
    if y_m is None:
        st.error("Target invalid untuk supervised.")
        st.stop()

    if y_m.nunique() < 2:
        st.error("Target setelah filter hanya punya 1 kelas. Ubah filter atau ganti Target Mode.")
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

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        test_size = st.slider("test_size (holdout)", 0.1, 0.4, 0.2, 0.05, key="sup_test_size")
    with colB:
        class_weight = st.selectbox("class_weight", options=["balanced", "none"], index=0, key="sup_class_weight")
    with colC:
        c_reg = st.slider("C (regularization)", 0.1, 5.0, 1.0, 0.1, key="sup_C")

    preprocess = build_preprocess(cat_cols, num_cols)
    model = LogisticRegression(
        max_iter=4000,
        class_weight=("balanced" if class_weight == "balanced" else None),
        C=float(c_reg),
        solver="liblinear",
    )
    pipe = Pipeline([("prep", preprocess), ("clf", model)])

    X = prepare_X(dfm, cat_cols, num_cols)
    y_arr = y_m.values.astype(int)

    min_class = min((y_arr == 0).sum(), (y_arr == 1).sum())
    if min_class < 10:
        st.warning(
            "Kelas minoritas terlalu kecil untuk split+evaluasi yang stabil. "
            "Longgarkan filter / ganti Target Mode / gunakan dataset yang sama seperti PDF."
        )
        st.stop()

    run_sup = st.button("Run supervised ranking", key="btn_sup_run")

    sup_sig = ("sup", sig_f, tuple(cat_cols), tuple(num_cols), float(test_size), class_weight, float(c_reg), int(k_cap), mode, str(manual_pos))
    if run_sup or (st.session_state.get("sup_done") and st.session_state.get("sup_sig") == sup_sig):
        if run_sup or (not st.session_state.get("sup_done")) or st.session_state.get("sup_sig") != sup_sig:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_arr,
                test_size=float(test_size),
                random_state=42,
                stratify=y_arr,
            )

            pipe.fit(X_train, y_train)
            p_test = pipe.predict_proba(X_test)[:, 1]

            pr_auc = float(average_precision_score(y_test, p_test))
            try:
                roc_auc = float(roc_auc_score(y_test, p_test))
            except Exception:
                roc_auc = np.nan

            # Curves for K
            max_k = int(min(len(y_test), 2000))
            ks = np.unique(np.clip(np.linspace(50, max_k, 30).astype(int), 1, None))
            precs, recs, lifts = [], [], []
            for k in ks:
                pr_k, rc_k = precision_recall_at_k(y_test, p_test, int(k))
                lf_k = lift_at_k(y_test, p_test, int(k))
                precs.append(pr_k)
                recs.append(rc_k)
                lifts.append(lf_k)

            # Top-K table on FULL data (action)
            p_full = pipe.predict_proba(X)[:, 1]
            order = np.argsort(-p_full)
            top_idx = order[:int(k_cap)]
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

        met = st.session_state["sup_metrics"]
        curves = st.session_state["sup_curves"]
        df_top = st.session_state["sup_top"]

        st.subheader("Model quality (ranking-oriented)")
        m1, m2 = st.columns(2)
        m1.metric("PR-AUC (Average Precision)", _nan_safe_metric_value(met.get("pr_auc")))
        m2.metric("ROC-AUC (opsional)", _nan_safe_metric_value(met.get("roc_auc")))

        st.caption("Fokus utama untuk Top-K biasanya PR-AUC + Precision@K/Recall@K/Lift@K.")

        st.subheader("Trade-off curve: Precision@K, Recall@K, Lift@K")
        c_left, c_right = st.columns(2)

        ch1 = (
            alt.Chart(curves)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="K"),
                y=alt.Y("precision_at_k:Q", title="Precision@K"),
                tooltip=["k:Q", alt.Tooltip("precision_at_k:Q", format=".3f")],
            )
            .properties(height=280, title="Precision@K", autosize={"type": "fit", "contains": "padding"})
        )
        c_left.altair_chart(ch1, use_container_width=True)

        ch2 = (
            alt.Chart(curves)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="K"),
                y=alt.Y("recall_at_k:Q", title="Recall@K"),
                tooltip=["k:Q", alt.Tooltip("recall_at_k:Q", format=".3f")],
            )
            .properties(height=280, title="Recall@K", autosize={"type": "fit", "contains": "padding"})
        )
        c_right.altair_chart(ch2, use_container_width=True)

        ch3 = (
            alt.Chart(curves)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="K"),
                y=alt.Y("lift_at_k:Q", title="Lift@K"),
                tooltip=["k:Q", alt.Tooltip("lift_at_k:Q", format=".2f")],
            )
            .properties(height=280, title="Lift@K (vs baseline)", autosize={"type": "fit", "contains": "padding"})
        )
        st.altair_chart(ch3, use_container_width=True)

        st.subheader("Top-K output (untuk eksekusi bisnis)")
        show_cols = [c for c in ["pred_rank", "score", "Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region", "Umur"] if c in df_top.columns]
        if not show_cols:
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

# Dashboard Akhir (Bisnis)
with tab_dash:
    st.subheader("Dashboard Akhir (Bisnis) — ringkas & actionable")
    st.caption("Tab ini menjawab tujuan proyek: pola peserta → persona → model Top-K → rekomendasi aksi bisnis.")

    # Build base (with persona if exists)
    df_dash = df_f.copy()
    if st.session_state.get("cluster_done"):
        dfc = st.session_state["df_clustered"]
        df_dash = df_dash.merge(dfc[["_cluster_id", "Persona"]], left_index=True, right_index=True, how="left")

    y_dash, _ = make_target(df_dash, target_col=target_col, mode=mode, manual_positive_value=manual_pos)
    if y_dash is None:
        st.error("Target tidak valid untuk dashboard.")
        st.stop()

    n_all_d, n_pos_d, n_neg_d, base_rate = target_summary(y_dash)

    # KPI row
    st.markdown("### KPI utama")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows aktif", f"{n_all_d:,}")
    k2.metric("Positives", f"{n_pos_d:,}")
    k3.metric("Positive rate", f"{base_rate*100:.2f}%")
    k4.metric("K (capacity)", f"{int(st.session_state.get('k_capacity', 200)):,}")

    if st.session_state.get("sup_done"):
        met = st.session_state["sup_metrics"]
        k5.metric("PR-AUC", _nan_safe_metric_value(met.get("pr_auc")))
    else:
        k5.metric("PR-AUC", "Run supervised")

    st.divider()

    # 1) Pola peserta (target rate breakdown)
    st.markdown("## 1) Pola peserta yang paling berdampak (target rate breakdown)")
    a, b = st.columns(2)

    # Persona rate (if exists)
    if "Persona" in df_dash.columns:
        rate_persona = top_rate_by_group(df_dash, y_dash, "Persona", min_count=10, top_n=10)
        ch_persona = alt_bar_rate(rate_persona, "Target rate by Persona", max_height=420)
    else:
        rate_persona, ch_persona = None, None

    rate_prod = top_rate_by_group(df_dash, y_dash, "Product", min_count=30, top_n=15) if "Product" in df_dash.columns else None
    ch_prod2 = alt_bar_rate(rate_prod, "Target rate by Product (Top 15)", max_height=520) if rate_prod is not None else None

    with a:
        if ch_persona is not None:
            st.altair_chart(ch_persona, use_container_width=True)
        else:
            st.info("Persona belum ada. Jalankan tab Clustering dulu untuk dapat Persona.")
    with b:
        if ch_prod2 is not None:
            st.altair_chart(ch_prod2, use_container_width=True)
        else:
            st.info("Kolom 'Product' tidak tersedia / tidak lolos min_count.")

    c, d = st.columns(2)
    rate_kat2 = top_rate_by_group(df_dash, y_dash, "Kategori", min_count=30, top_n=15) if "Kategori" in df_dash.columns else None
    ch_kat2 = alt_bar_rate(rate_kat2, "Target rate by Kategori (Top 15)", max_height=520) if rate_kat2 is not None else None

    rate_chan2 = top_rate_by_group(df_dash, y_dash, "Channel", min_count=30, top_n=15) if "Channel" in df_dash.columns else None
    ch_chan2 = alt_bar_rate(rate_chan2, "Target rate by Channel (Top 15)", max_height=520) if rate_chan2 is not None else None

    with c:
        if ch_kat2 is not None:
            st.altair_chart(ch_kat2, use_container_width=True)
        else:
            st.info("Kolom 'Kategori' tidak tersedia / tidak lolos min_count.")
    with d:
        if ch_chan2 is not None:
            st.altair_chart(ch_chan2, use_container_width=True)
        else:
            st.info("Kolom 'Channel' tidak tersedia / tidak lolos min_count.")

    st.markdown("### Distribusi target (setelah filter)")
    st.altair_chart(alt_dist_target(y_dash, "Distribusi target (setelah filter)"), use_container_width=True)

    st.divider()

    # 2) Segmentasi persona (profiling)
    st.markdown("## 2) Segmentasi persona (akurasi persona & implikasi bisnis)")
    if "Persona" in df_dash.columns:
        blocks = []
        prof_cols = [c for c in ["Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region"] if c in df_dash.columns]
        for p in sorted(df_dash["Persona"].dropna().unique().tolist()):
            sub = df_dash[df_dash["Persona"] == p]
            row = {"Persona": p, "Size": int(len(sub))}
            y_sub, _ = make_target(sub, target_col=target_col, mode=mode, manual_positive_value=manual_pos)
            row["Target rate (%)"] = round(float(y_sub.mean() * 100), 2) if len(sub) else 0.0
            for ccol in prof_cols:
                vc = sub[ccol].astype(str).value_counts()
                row[f"Top {ccol}"] = vc.index[0] if len(vc) else ""
            blocks.append(row)
        prof_df = pd.DataFrame(blocks).sort_values("Size", ascending=False)
        st.dataframe(prof_df, use_container_width=True)
    else:
        st.info("Persona belum ada. Jalankan tab Clustering dulu.")

    st.divider()

    # 3) Model & Top-K (action)
    st.markdown("## 3) Model Top-K untuk keputusan bisnis (siapa diprioritaskan)")
    if not st.session_state.get("sup_done"):
        st.warning("Supervised belum dijalankan. Buka tab **Supervised (Top-K Ranking)** lalu klik **Run supervised ranking**.")
    else:
        met = st.session_state["sup_metrics"]
        curves = st.session_state["sup_curves"]
        df_top = st.session_state["sup_top"]
        k_cap_now = int(st.session_state.get("k_capacity", 200))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC", _nan_safe_metric_value(met.get("pr_auc")))
        m2.metric("ROC-AUC", _nan_safe_metric_value(met.get("roc_auc")))
        m3.metric("K (capacity)", f"{k_cap_now:,}")
        m4.metric("Baseline rate", f"{base_rate*100:.2f}%")

        # nearest curve point
        nearest = curves.iloc[(curves["k"] - k_cap_now).abs().argsort()[:1]]
        if len(nearest):
            nk = int(nearest["k"].iloc[0])
            prec_k = float(nearest["precision_at_k"].iloc[0])
            rec_k = float(nearest["recall_at_k"].iloc[0])
            lift_k = float(nearest["lift_at_k"].iloc[0])

            st.markdown(
                f"""
**Interpretasi bisnis (sekitar K≈{nk}):**
- Precision@K ≈ **{prec_k*100:.2f}%** → dari K yang di-follow up, segini yang benar-benar positif.
- Recall@K ≈ **{rec_k*100:.2f}%** → dari semua positif, segini yang tertangkap Top-K.
- Lift@K ≈ **{lift_k:.2f}×** → efisiensi dibanding random pick.
"""
            )

        st.markdown("### Distribusi Top-K (untuk strategi akuisisi / desain program / intervensi)")
        dist_cols = []
        for ccol in ["Persona", "Product", "Kategori", "Channel", "Region", "Segmen_karir"]:
            if ccol in df_top.columns:
                dist_cols.append(ccol)

        if dist_cols:
            # show up to 3 distributions
            cols = st.columns(min(3, len(dist_cols)))
            for i, ccol in enumerate(dist_cols[:3]):
                vc = df_top[ccol].astype(str).value_counts().head(15).reset_index()
                vc.columns = [ccol, "count"]
                ch = (
                    alt.Chart(vc)
                    .mark_bar()
                    .encode(
                        y=alt.Y(f"{ccol}:N", sort="-x", title=ccol),
                        x=alt.X("count:Q", title="Top-K count"),
                        tooltip=[f"{ccol}:N", "count:Q"],
                    )
                    .properties(height=380, title=f"Top-K count by {ccol} (Top 15)", autosize={"type": "fit", "contains": "padding"})
                )
                cols[i].altair_chart(ch, use_container_width=True)

        st.markdown("### Top-K list (ringkas)")
        show_cols = [c for c in ["pred_rank", "score", "Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region", "Umur"] if c in df_top.columns]
        if not show_cols:
            show_cols = ["pred_rank", "score"]
        st.dataframe(df_top[show_cols].head(200), use_container_width=True)

        st.success(
            "Dashboard Akhir ini sudah nyambung ke tujuan proyek: pola peserta → persona → Top-K untuk keputusan bisnis."
        )
