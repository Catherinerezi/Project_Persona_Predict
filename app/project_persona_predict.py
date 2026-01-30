import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Marketing AB - Persona Dashboard", layout="wide")
alt.data_transformers.disable_max_rows()

# -----------------------------
# Path resolver (anti Streamlit-cloud cwd ngaco)
# -----------------------------
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent          # .../app
ROOT_DIR = APP_DIR.parent          # repo root

def resolve_csv_path(user_input: str) -> Path:
    """
    Accept:
      - 'raw_data/raw_data.csv'
      - 'raw_data.csv'
      - '../raw_data/raw_data.csv'
    Will try:
      1) as-is relative to ROOT_DIR
      2) as-is relative to APP_DIR
      3) raw_data/<name> relative to ROOT_DIR and APP_DIR
    """
    s = (user_input or "").strip()
    if not s:
        return None

    candidates = []

    # 1) interpret as relative to repo root
    candidates.append((ROOT_DIR / s).resolve())
    # 2) interpret as relative to app dir
    candidates.append((APP_DIR / s).resolve())

    # 3) if user types only filename, try inside raw_data/
    name = Path(s).name
    candidates.append((ROOT_DIR / "raw_data" / name).resolve())
    candidates.append((APP_DIR / "raw_data" / name).resolve())

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # last try: maybe already absolute
    p_abs = Path(s)
    if p_abs.exists() and p_abs.is_file():
        return p_abs.resolve()

    return None

# -----------------------------
# Utils
# -----------------------------
def require_cols(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"[{where}] Kolom hilang: {missing}")
        st.caption("Kolom yang tersedia:")
        st.code(", ".join(df.columns.tolist()))
        st.stop()

def safe_col(df: pd.DataFrame, name: str):
    return name if name in df.columns else None

def to_long_rate(df, group_col, target_col):
    d = df.groupby(group_col, dropna=False)[target_col].agg(["mean", "count"]).reset_index()
    d["rate"] = d["mean"]
    d["n"] = d["count"]
    d[group_col] = d[group_col].astype(str)
    return d[[group_col, "rate", "n"]]

def alt_bar_rate(d, group_col, title):
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y(f"{group_col}:N", sort="-x", title=None),
            x=alt.X("rate:Q", title="Placement rate"),
            tooltip=[group_col, alt.Tooltip("rate:Q", format=".2%"), "n:Q"],
        )
        .properties(height=min(450, 18 * max(8, len(d))), title=title)
    )

def alt_hist_by_class(df, score_col, y_col, title, zoom_max=None):
    d = df[[score_col, y_col]].dropna().copy()
    if zoom_max is not None:
        d = d[d[score_col] <= zoom_max]
    d[y_col] = d[y_col].astype(int).astype(str)
    return (
        alt.Chart(d)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X(f"{score_col}:Q", bin=alt.Bin(maxbins=60), title=score_col),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color(f"{y_col}:N", title=y_col),
            tooltip=[alt.Tooltip("count():Q", title="count")],
        )
        .properties(height=300, title=title)
    )

def topk_table(y_true: np.ndarray, y_score: np.ndarray, k_list: list[int]) -> pd.DataFrame:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    N = len(y_true)
    prevalence = y_true.mean() if N else 0.0

    rows = []
    total_pos = int(y_true.sum())
    for k in k_list:
        k = int(k)
        if k <= 0 or k > N:
            continue
        captured = int(y_sorted[:k].sum())
        precision = captured / k if k else 0.0
        recall = captured / total_pos if total_pos else 0.0
        lift = (precision / prevalence) if prevalence > 0 else 0.0
        rows.append(
            {"K": k, "Positives captured": captured, "Precision@K": precision, "Recall@K": recall, "Lift@K": lift}
        )
    return pd.DataFrame(rows)

def cdf_percentile_among_negatives(df, score_col, y_col):
    d = df[[score_col, y_col]].dropna().copy()
    neg = d[d[y_col] == 0][score_col].values
    pos = d[d[y_col] == 1][score_col].values
    if len(neg) == 0 or len(pos) == 0:
        return None

    pos_score = float(np.min(pos))  # "pos#1 score=..."
    pct = float((neg <= pos_score).mean())  # percentile among negatives

    neg_sorted = np.sort(neg)
    cdf = np.arange(1, len(neg_sorted) + 1) / len(neg_sorted)
    out = pd.DataFrame({"score": neg_sorted, "cdf": cdf})
    return out, pos_score, pct

@st.cache_data(show_spinner=False)
def load_df_from_path(path_str: str) -> pd.DataFrame:
    p = resolve_csv_path(path_str)
    if p is None:
        raise FileNotFoundError(path_str)
    return pd.read_csv(p)

# -----------------------------
# Header
# -----------------------------
st.title("Persona / Marketing AB Dashboard (Altair + Streamlit)")
st.caption("Goal: semua visualisasi notebook muncul, tapi user bisa filter super gampang dari sidebar.")

# -----------------------------
# Sidebar: pilih CSV (nggak ngetik path lagi)
# -----------------------------
raw_dir_root = ROOT_DIR / "raw_data"
raw_dir_app  = APP_DIR / "raw_data"

csv_choices = []
for base in [raw_dir_root, raw_dir_app]:
    if base.exists():
        csv_choices += sorted([str(p.relative_to(ROOT_DIR)) if p.is_relative_to(ROOT_DIR) else str(p) for p in base.glob("*.csv")])

default_choice = "raw_data/raw_data.csv" if "raw_data/raw_data.csv" in csv_choices else (csv_choices[0] if csv_choices else "raw_data/raw_data.csv")

with st.sidebar:
    st.header("Data")
    if csv_choices:
        data_path = st.selectbox("Pilih CSV", csv_choices, index=csv_choices.index(default_choice) if default_choice in csv_choices else 0)
        st.caption("File diambil dari folder `raw_data/`.")
    else:
        data_path = st.text_input("Path CSV", value=default_choice)
        st.caption("Folder `raw_data/` tidak ketemu / kosong. Pastikan ada `raw_data/*.csv` di repo.")

# -----------------------------
# Load
# -----------------------------
try:
    df = load_df_from_path(data_path)
except Exception as e:
    st.error("Gagal load CSV. Ini murni masalah path/struktur repo.")
    st.code(f"Input path: {data_path}")
    st.code(f"APP_DIR : {APP_DIR}")
    st.code(f"ROOT_DIR: {ROOT_DIR}")
    st.exception(e)
    st.stop()

# -----------------------------
# WAJIB: segmentasi_karir harus ada (karena kamu bilang harus ada)
# -----------------------------
if "segmentasi_karir" not in df.columns:
    st.error("Kolom WAJIB `segmentasi_karir` tidak ada di CSV yang kamu load.")
    st.caption("Berarti: (1) file yang keload bukan raw_data.csv yang kamu maksud, atau (2) kolomnya beda nama.")
    st.code(", ".join(df.columns.tolist()))
    st.stop()

# -----------------------------
# Auto-detect target & score, tapi user bisa override 1 klik
# -----------------------------
auto_target = safe_col(df, "is_placed") or safe_col(df, "placed") or safe_col(df, "y")
auto_score  = safe_col(df, "y_proba") or safe_col(df, "score") or safe_col(df, "pred_proba")

with st.sidebar:
    st.divider()
    st.header("Kolom model (1 klik)")
    TARGET = st.selectbox("Target (0/1)", options=df.columns.tolist(), index=df.columns.tolist().index(auto_target) if auto_target in df.columns else 0)
    SCORE  = st.selectbox("Score/proba", options=df.columns.tolist(), index=df.columns.tolist().index(auto_score) if auto_score in df.columns else 0)

# validate basic types
require_cols(df, [TARGET, SCORE, "segmentasi_karir"], "Global")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df[SCORE]  = pd.to_numeric(df[SCORE], errors="coerce")

# -----------------------------
# Segment columns (optional)
# -----------------------------
SEGMENT_KARIR = "segmentasi_karir"
UMUR          = safe_col(df, "umur") or safe_col(df, "age")
REGION        = safe_col(df, "region")
UMUR_BIN      = safe_col(df, "umur_bin")
DOMAIN        = safe_col(df, "domain_pendidikan") or safe_col(df, "domain")
JOB_CAT       = safe_col(df, "kategori_pekerjaan") or safe_col(df, "job_category")

# -----------------------------
# Sidebar filters (super gampang)
# -----------------------------
with st.sidebar:
    st.divider()
    st.header("Filter (1 klik)")
    st.caption("Filter ini berlaku ke SEMUA tab & visualisasi.")

    def sb(colname, label):
        if colname is None:
            st.caption(f"({label}) kolom tidak ada")
            return None, None
        vals = ["All"] + sorted(df[colname].dropna().astype(str).unique().tolist())
        v = st.selectbox(label, vals, index=0)
        return colname, v

    f_cols = []
    for c, label in [
        (SEGMENT_KARIR, "segmentasi_karir"),
        (UMUR_BIN, "umur_bin"),
        (REGION, "region"),
        (DOMAIN, "domain_pendidikan/domain"),
        (JOB_CAT, "kategori_pekerjaan/job_category"),
    ]:
        col, val = sb(c, label)
        if col is not None and val is not None:
            f_cols.append((col, val))

    st.divider()
    show_all_eda = st.checkbox("Tampilkan SEMUA EDA (berat)", value=False)
    st.caption("Kalau OFF: EDA bisa dipilih via dropdown/multiselect (biar user nggak pusing).")

# apply filters
df_f = df.copy()
for col, val in f_cols:
    if val != "All":
        df_f = df_f[df_f[col].astype(str) == val]

# -----------------------------
# Tabs (Marketing AB style)
# -----------------------------
tab_overview, tab_eda, tab_cluster, tab_supervised, tab_debug = st.tabs(
    ["Overview", "EDA", "Clustering", "Supervised (Top-K)", "Debug/QA"]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Ringkasan Filter")
        st.write(f"Rows (filtered): **{len(df_f):,}** / total **{len(df):,}**")
        pos = int(df_f[TARGET].fillna(0).astype(int).sum())
        prev = pos / len(df_f) if len(df_f) else 0
        st.write(f"Positives: **{pos}** (prevalence **{prev:.4%}**)")
        st.write(f"Kolom target: `{TARGET}` | Kolom score: `{SCORE}`")
    with right:
        st.subheader("Jaminan output (nggak dipangkas)")
        st.markdown(
            """
- **EDA**: banyak chart placement rate (bisa semua/selected)
- **Clustering**: scatter SVD 2D + legend cluster_id (kalau ada kolomnya)
- **Supervised**: Top-K table + Recall/Precision curve + Gains/Lift + Score dist (zoom & full) + CDF percentile
- **Composite dashboard**: summary + Lift@K bar + Precision@K bar + Top-K table
"""
        )

# -----------------------------
# EDA
# -----------------------------
with tab_eda:
    st.subheader("EDA (Placement Rate)")
    # kandidat kolom kategorikal untuk EDA
    candidates = []
    for c in [SEGMENT_KARIR, UMUR_BIN, UMUR, REGION, DOMAIN, JOB_CAT]:
        if c and c in df.columns:
            candidates.append(c)
    if len(candidates) == 0:
        candidates = [c for c in df.columns if df[c].dtype == "object"][:20]

    if show_all_eda:
        st.info("Mode SEMUA EDA ON: semua chart akan dirender (bisa berat).")
        for c in candidates:
            with st.expander(f"Placement rate by {c}", expanded=False):
                d = to_long_rate(df_f, c, TARGET)
                st.altair_chart(alt_bar_rate(d, c, f"Placement rate by {c}"), use_container_width=True)
    else:
        pick = st.selectbox("Pilih dimensi EDA", candidates, index=0)
        d = to_long_rate(df_f, pick, TARGET)
        st.altair_chart(alt_bar_rate(d, pick, f"Placement rate by {pick}"), use_container_width=True)

        with st.expander("Show additional EDA charts (multi)", expanded=False):
            multi = st.multiselect("Pilih beberapa dimensi", candidates, default=candidates[:3])
            for c in multi:
                d = to_long_rate(df_f, c, TARGET)
                st.altair_chart(alt_bar_rate(d, c, f"Placement rate by {c}"), use_container_width=True)

# -----------------------------
# Clustering
# -----------------------------
with tab_cluster:
    st.subheader("Clustering (SVD 2D Scatter)")

    svd1 = safe_col(df_f, "SVD-1") or safe_col(df_f, "svd_1") or safe_col(df_f, "svd1")
    svd2 = safe_col(df_f, "SVD-2") or safe_col(df_f, "svd_2") or safe_col(df_f, "svd2")
    cluster_id = safe_col(df_f, "cluster_id") or safe_col(df_f, "cluster")

    if svd1 is None or svd2 is None:
        st.warning("Kolom SVD tidak ketemu (SVD-1/SVD-2 atau svd_1/svd_2).")
        st.stop()

    cols = [svd1, svd2] + ([cluster_id] if cluster_id else [])
    base = df_f[cols].dropna().copy()

    if cluster_id:
        base[cluster_id] = base[cluster_id].astype(str)
        chart = (
            alt.Chart(base)
            .mark_circle(size=40, opacity=0.65)
            .encode(
                x=alt.X(f"{svd1}:Q", title="SVD-1"),
                y=alt.Y(f"{svd2}:Q", title="SVD-2"),
                color=alt.Color(f"{cluster_id}:N", title="cluster_id"),
                tooltip=[svd1, svd2, cluster_id],
            )
            .properties(height=520, title="Cluster Visualization (TruncatedSVD 2D)")
        )
    else:
        chart = (
            alt.Chart(base)
            .mark_circle(size=40, opacity=0.65)
            .encode(
                x=alt.X(f"{svd1}:Q", title="SVD-1"),
                y=alt.Y(f"{svd2}:Q", title="SVD-2"),
                tooltip=[svd1, svd2],
            )
            .properties(height=520, title="SVD 2D Scatter (no cluster_id column)")
        )

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Supervised (Top-K)
# -----------------------------
with tab_supervised:
    st.subheader("Supervised Ranking (Top-K)")

    df_s = df_f[[TARGET, SCORE]].dropna().copy()
    df_s[TARGET] = df_s[TARGET].astype(int)

    y_true = df_s[TARGET].values
    y_score = df_s[SCORE].astype(float).values

    N = len(df_s)
    total_pos = int(y_true.sum())
    prevalence = (total_pos / N) if N else 0.0

    if N == 0:
        st.error("Setelah filter, data kosong. Longgarkan filter.")
        st.stop()

    k_default = min(50, N)
    k_bisnis = st.slider("K bisnis (misal Top-50)", 1, N, k_default)

    k_list = sorted(set([5, 10, 20, 30, 50, 60, 100, k_bisnis]))
    k_list = [k for k in k_list if k <= N]
    tbl = topk_table(y_true, y_score, k_list)

    st.markdown("### Top-K Summary Table")
    st.dataframe(
        tbl.assign(
            **{
                "Precision@K (%)": (tbl["Precision@K"] * 100).round(2),
                "Recall@K (%)": (tbl["Recall@K"] * 100).round(2),
                "Lift@K": tbl["Lift@K"].round(2),
            }
        )[["K", "Positives captured", "Precision@K (%)", "Recall@K (%)", "Lift@K"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Recall@K & Precision@K")
    curve = tbl.copy()
    d_long = curve.melt(id_vars=["K"], value_vars=["Recall@K", "Precision@K"], var_name="metric", value_name="value")
    line = (
        alt.Chart(d_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("value:Q", title="Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", title=None),
            tooltip=["K:Q", "metric:N", alt.Tooltip("value:Q", format=".2%")],
        )
        .properties(height=320, title="Top-K Capture Curve (Rate) — Recall@K & Precision@K")
    )
    st.altair_chart(line, use_container_width=True)

    st.markdown("### Cumulative Gains & Lift vs Random")
    g = curve.copy()
    g["Gains (Recall@K)"] = g["Recall@K"]
    g["Random baseline (K/N)"] = g["K"] / N
    g["Cumulative Lift@K"] = np.where(g["Random baseline (K/N)"] > 0, g["Gains (Recall@K)"] / g["Random baseline (K/N)"], 0.0)
    g["Random lift (=1)"] = 1.0

    g_long = g[["K", "Gains (Recall@K)", "Random baseline (K/N)", "Cumulative Lift@K", "Random lift (=1)"]].melt(
        id_vars=["K"], var_name="metric", value_name="value"
    )
    gains_chart = (
        alt.Chart(g_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("metric:N", title=None),
            tooltip=["K:Q", "metric:N", "value:Q"],
        )
        .properties(height=320, title="Cumulative Gains & Lift vs Random")
    )
    st.altair_chart(gains_chart, use_container_width=True)

    st.markdown("### Score Distribution (zoom & full)")
    h1, h2 = st.columns(2)
    with h1:
        st.altair_chart(alt_hist_by_class(df_f, SCORE, TARGET, "Zoom-in (0 to 0.1)", zoom_max=0.1), use_container_width=True)
    with h2:
        st.altair_chart(alt_hist_by_class(df_f, SCORE, TARGET, "Full range (0 to 1)", zoom_max=1.0), use_container_width=True)

    st.markdown("### CDF Negatif + posisi skor positif (percentile among negatives)")
    cdf_out = cdf_percentile_among_negatives(df_f, SCORE, TARGET)
    if cdf_out is None:
        st.warning("Tidak bisa bikin CDF: butuh y=0 dan y=1 setelah filter.")
    else:
        cdf_df, pos_score, pct = cdf_out
        cdf_chart = (
            alt.Chart(cdf_df)
            .mark_line()
            .encode(
                x=alt.X("score:Q", title=f"Predicted probability ({SCORE})"),
                y=alt.Y("cdf:Q", title="CDF (proporsi negatif ≤ skor)"),
                tooltip=[alt.Tooltip("score:Q", format=".6f"), alt.Tooltip("cdf:Q", format=".2%")],
            )
            .properties(height=320, title="CDF Negatif + posisi skor positif")
        )
        vline = alt.Chart(pd.DataFrame({"x": [pos_score]})).mark_rule(strokeDash=[6, 4]).encode(x="x:Q")
        st.altair_chart(cdf_chart + vline, use_container_width=True)
        st.write(f"Ringkasan: pos#1 score = **{pos_score:.6f}** → percentile among negatives ≈ **{pct:.2%}**")

    st.markdown("## Model Ranking Dashboard (Top-K) — composite")
    t_biz = topk_table(y_true, y_score, [k_bisnis]).iloc[0].to_dict()
    lift_k = float(t_biz["Lift@K"])
    prec_k = float(t_biz["Precision@K"])
    captured = int(t_biz["Positives captured"])

    left, mid, right = st.columns([1, 1, 1])
    with left:
        st.markdown("### Summary")
        st.write(f"Holdout N = **{N}**")
        st.write(f"Positives = **{total_pos}** (prevalence = **{prevalence:.4f}**)")
        st.write(f"K bisnis = **{k_bisnis}**")
        st.markdown(f"**Lift@{k_bisnis} = {lift_k:.2f}x**")
        st.markdown(f"Precision@{k_bisnis} = **{prec_k*100:.2f}%**")
        st.write(f"Positives captured @K = **{captured}**")

    # Lift bar (top-k only)
    b = tbl.copy()
    b["Lift@K"] = np.where(prevalence > 0, b["Precision@K"] / prevalence, 0.0)
    lift_bar = (
        alt.Chart(b[["K", "Lift@K"]])
        .mark_bar()
        .encode(x=alt.X("K:Q", title="K"), y=alt.Y("Lift@K:Q", title="Lift@K"), tooltip=["K:Q", alt.Tooltip("Lift@K:Q", format=".2f")])
        .properties(height=320, title="Lift@K (Top-K only)")
    )
    # Precision bar
    p = tbl.copy()
    p["Precision@K_pct"] = p["Precision@K"] * 100
    prec_bar = (
        alt.Chart(p[["K", "Precision@K_pct"]])
        .mark_bar()
        .encode(x=alt.X("K:Q", title="K"), y=alt.Y("Precision@K_pct:Q", title="Precision@K (%)"), tooltip=["K:Q", alt.Tooltip("Precision@K_pct:Q", format=".2f")])
        .properties(height=320, title=f"Precision@K (baseline prev={prevalence:.2%})")
    )

    with mid:
        st.altair_chart(lift_bar, use_container_width=True)
    with right:
        st.altair_chart(prec_bar, use_container_width=True)

# -----------------------------
# Debug/QA
# -----------------------------
with tab_debug:
    st.subheader("Debug / QA")
    st.write("Kalau ada yang hilang, ini tempat paling cepat buat lihat: data kebaca apa nggak & kolom apa yang ada.")

    st.markdown("### Lokasi & file yang kebaca")
    st.code(f"APP_FILE: {APP_FILE}")
    st.code(f"APP_DIR : {APP_DIR}")
    st.code(f"ROOT_DIR: {ROOT_DIR}")
    p_res = resolve_csv_path(data_path)
    st.code(f"RESOLVED CSV: {p_res}")

    st.markdown("### Kolom penting")
    st.json(
        {
            "TARGET": TARGET,
            "SCORE": SCORE,
            "segmentasi_karir": SEGMENT_KARIR,
            "umur": UMUR,
            "umur_bin": UMUR_BIN,
            "region": REGION,
            "domain": DOMAIN,
            "job_category": JOB_CAT,
        }
    )

    st.markdown("### Sample data (filtered)")
    st.dataframe(df_f.head(50), use_container_width=True)
