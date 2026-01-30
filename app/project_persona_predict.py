import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Marketing AB - Persona Dashboard", layout="wide")

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

def safe_col(df, name):
    return name if name in df.columns else None

def to_long_rate(df, group_col, target_col):
    # rate = mean(target)
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
            tooltip=[alt.Tooltip("count():Q", title="count")]
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
        rows.append({"K": k, "Positives captured": captured, "Precision@K": precision, "Recall@K": recall, "Lift@K": lift})
    return pd.DataFrame(rows)

def cdf_percentile_among_negatives(df, score_col, y_col):
    d = df[[score_col, y_col]].dropna().copy()
    neg = d[d[y_col] == 0][score_col].values
    pos = d[d[y_col] == 1][score_col].values
    if len(neg) == 0 or len(pos) == 0:
        return None

    pos_score = float(np.min(pos))  # mengikuti screenshot "pos#1 score=..."
    pct = float((neg <= pos_score).mean())  # percentile among negatives
    # build CDF curve
    neg_sorted = np.sort(neg)
    cdf = np.arange(1, len(neg_sorted) + 1) / len(neg_sorted)
    out = pd.DataFrame({"score": neg_sorted, "cdf": cdf})
    return out, pos_score, pct

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data
def load_data(path="raw_data.csv"):
    df = pd.read_csv(path)
    return df

# -----------------------------
# Header
# -----------------------------
st.title("Persona / Marketing AB Dashboard (Altair + Streamlit)")
st.caption("Goal: semua visualisasi notebook muncul, tapi user bisa filter super gampang dari sidebar.")

# -----------------------------
# Load
# -----------------------------
import pandas as pd
import streamlit as st
from pathlib import Path

DEFAULT_PATH = "raw_data/raw_data.csv"

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("Path CSV", value=DEFAULT_PATH)

p = Path(data_path)

if not p.exists():
    st.error(
        f"File tidak ditemukan: `{data_path}`.\n\n"
        "Pastikan CSV ada di repo sesuai path itu. Contoh yang benar: `raw_data/raw_data.csv`"
    )
    st.stop()

df = pd.read_csv(p)

# -----------------------------
# Global config: detect columns
# -----------------------------
# WAJIB kamu sesuaikan kalau nama kolommu beda:
TARGET = safe_col(df, "is_placed") or safe_col(df, "placed") or safe_col(df, "y")  # target binary
SCORE  = safe_col(df, "y_proba") or safe_col(df, "score") or safe_col(df, "pred_proba")

# kolom segment yang sering dipakai (opsional, auto-detect)
SEGMENT_KARIR = safe_col(df, "segmentasi_karir")
UMUR          = safe_col(df, "umur") or safe_col(df, "age")
REGION        = safe_col(df, "region")
UMUR_BIN      = safe_col(df, "umur_bin")
DOMAIN        = safe_col(df, "domain_pendidikan") or safe_col(df, "domain")
JOB_CAT       = safe_col(df, "kategori_pekerjaan") or safe_col(df, "job_category")

# -----------------------------
# Sidebar filters (super gampang)
# -----------------------------
with st.sidebar:
    st.header("Filter (1 klik)")
    st.caption("Filter ini berlaku ke SEMUA tab & visualisasi.")

    # helper to build selectbox with "All"
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
    st.caption("Kalau OFF: EDA ditampilkan via dropdown/expander biar user nggak pusing.")

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
        if TARGET:
            pos = int(df_f[TARGET].fillna(0).astype(int).sum())
            prev = pos / len(df_f) if len(df_f) else 0
            st.write(f"Positives: **{pos}** (prevalence **{prev:.4%}**)")
        else:
            st.warning("Target column belum terdeteksi (is_placed/placed/y). Set manual di kode.")
        if SEGMENT_KARIR is None:
            st.error("Kolom `segmentasi_karir` TIDAK ADA di data yang kebaca. Ini harusnya ada kalau memang ada di raw_data.csv.")
    with right:
        st.subheader("Apa yang dijamin tampil di app ini")
        st.markdown(
            """
- **EDA**: placement rate by segment (banyak chart), auto via dropdown/expander
- **Clustering**: scatter 2D (SVD-1 vs SVD-2) + legend cluster_id (kalau kolom ada)
- **Supervised**: Top-K table + Recall@K & Precision@K curve + Gains/Lift + Score distribution (zoom & full) + CDF percentile
- **Dashboard komposit**: panel summary kiri + Lift@K bar + Precision@K bar + Top-K summary table
"""
        )

# -----------------------------
# EDA (banyak chart, tapi controllable)
# -----------------------------
with tab_eda:
    st.subheader("EDA (Placement Rate)")

    if TARGET is None:
        st.error("Target column (is_placed/placed/y) tidak ketemu. Tidak bisa hitung placement rate.")
        st.stop()

    # daftar kandidat kolom kategori untuk EDA
    candidates = []
    for c in [SEGMENT_KARIR, UMUR_BIN, UMUR, REGION, DOMAIN, JOB_CAT]:
        if c and c in df.columns:
            candidates.append(c)

    # fallback: auto-detect categorical columns (limited)
    if len(candidates) == 0:
        candidates = [c for c in df.columns if df[c].dtype == "object"][:15]

    if show_all_eda:
        st.info("Mode SEMUA EDA ON: semua chart akan dirender (bisa berat).")
        for c in candidates:
            with st.expander(f"Placement rate by {c}", expanded=False):
                d = to_long_rate(df_f, c, TARGET)
                st.altair_chart(alt_bar_rate(d, c, f"Placement rate by {c}"), use_container_width=True)
    else:
        pick = st.selectbox("Pilih dimensi EDA", candidates)
        d = to_long_rate(df_f, pick, TARGET)
        st.altair_chart(alt_bar_rate(d, pick, f"Placement rate by {pick}"), use_container_width=True)

        st.caption("Butuh banyak chart sekaligus? Buka expander di bawah.")
        with st.expander("Show additional EDA charts (multi)", expanded=False):
            multi = st.multiselect("Pilih beberapa dimensi", candidates, default=candidates[:3])
            for c in multi:
                d = to_long_rate(df_f, c, TARGET)
                st.altair_chart(alt_bar_rate(d, c, f"Placement rate by {c}"), use_container_width=True)

# -----------------------------
# Clustering (SVD scatter) - mengikuti screenshotmu
# -----------------------------
with tab_cluster:
    st.subheader("Clustering (SVD 2D Scatter)")

    svd1 = safe_col(df_f, "SVD-1") or safe_col(df_f, "svd_1") or safe_col(df_f, "svd1")
    svd2 = safe_col(df_f, "SVD-2") or safe_col(df_f, "svd_2") or safe_col(df_f, "svd2")
    cluster_id = safe_col(df_f, "cluster_id") or safe_col(df_f, "cluster")

    if svd1 is None or svd2 is None:
        st.warning("Kolom SVD tidak ketemu (SVD-1/SVD-2 atau svd_1/svd_2). Scatter tidak bisa dibuat.")
        st.stop()

    base = df_f[[svd1, svd2] + ([cluster_id] if cluster_id else [])].dropna().copy()
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
# Supervised (Top-K) - mengikuti screenshotmu
# -----------------------------
with tab_supervised:
    st.subheader("Supervised Ranking (Top-K)")

    if TARGET is None or SCORE is None:
        st.error("Butuh kolom target + score. Pastikan ada is_placed/placed/y dan y_proba/score/pred_proba.")
        st.stop()

    df_s = df_f[[TARGET, SCORE]].dropna().copy()
    y_true = df_s[TARGET].astype(int).values
    y_score = df_s[SCORE].astype(float).values

    N = len(df_s)
    total_pos = int(y_true.sum())
    prevalence = (total_pos / N) if N else 0.0

    k_default = min(50, N) if N else 50
    k_bisnis = st.slider("K bisnis (misal Top-50)", min_value=1, max_value=max(1, N), value=k_default)

    # ---- Top-K Summary table
    st.markdown("### Top-K Summary Table")
    k_list = sorted(set([5, 10, 20, 30, 50, 60, 100, k_bisnis]))
    k_list = [k for k in k_list if k <= N]
    tbl = topk_table(y_true, y_score, k_list)

    st.dataframe(
        tbl.assign(
            **{
                "Precision@K": (tbl["Precision@K"] * 100).round(2),
                "Recall@K": (tbl["Recall@K"] * 100).round(2),
                "Lift@K": tbl["Lift@K"].round(2),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Curves: Recall@K & Precision@K (mirip screenshot)
    st.markdown("### Recall@K & Precision@K")
    curve = tbl.copy()
    curve["Recall@K"] = curve["Recall@K"].astype(float)
    curve["Precision@K"] = curve["Precision@K"].astype(float)

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

    # ---- Gains & Lift vs Random
    st.markdown("### Cumulative Gains & Lift vs Random")
    g = curve.copy()
    # gains = recall@k
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

    # ---- Lift@K bar + Precision@K bar (Top-K only) + baseline
    st.markdown("### Lift@K & Precision@K (Top-K only)")
    bars_left, bars_right = st.columns(2)

    with bars_left:
        b = curve.copy()
        b["Lift@K"] = np.where(prevalence > 0, b["Precision@K"] / prevalence, 0.0)
        b = b[["K", "Lift@K"]]
        lift_bar = (
            alt.Chart(b)
            .mark_bar()
            .encode(
                x=alt.X("K:Q", title="K"),
                y=alt.Y("Lift@K:Q", title="Lift@K"),
                tooltip=["K:Q", alt.Tooltip("Lift@K:Q", format=".2f")],
            )
            .properties(height=320, title="Holdout: Lift@K (Top-K)")
        )
        st.altair_chart(lift_bar, use_container_width=True)

    with bars_right:
        p = curve.copy()
        p["Precision@K_pct"] = p["Precision@K"] * 100
        prec_bar = (
            alt.Chart(p)
            .mark_bar()
            .encode(
                x=alt.X("K:Q", title="K"),
                y=alt.Y("Precision@K_pct:Q", title="Precision@K (%)"),
                tooltip=["K:Q", alt.Tooltip("Precision@K_pct:Q", format=".2f")],
            )
            .properties(height=320, title=f"Holdout: Precision@K (baseline prev={prevalence:.2%})")
        )
        st.altair_chart(prec_bar, use_container_width=True)

    # ---- Score distribution (zoom + full)
    st.markdown("### Score Distribution (y_proba by class) — zoom & full")
    if len(df_f) > 0:
        h1, h2 = st.columns(2)
        with h1:
            st.altair_chart(
                alt_hist_by_class(df_f, SCORE, TARGET, "Zoom-in (0 to 0.1)", zoom_max=0.1),
                use_container_width=True,
            )
        with h2:
            st.altair_chart(
                alt_hist_by_class(df_f, SCORE, TARGET, "Full range (0 to 1)", zoom_max=1.0),
                use_container_width=True,
            )

    # ---- CDF percentile among negatives (mengikuti screenshot)
    st.markdown("### CDF Negatif + posisi skor positif (percentile among negatives)")
    cdf_out = cdf_percentile_among_negatives(df_f, SCORE, TARGET)
    if cdf_out is None:
        st.warning("Tidak bisa bikin CDF: butuh data negatif dan positif (y=0 dan y=1) setelah filter.")
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

    # ---- Composite Dashboard (yang kamu tunjukin)
    st.markdown("## Model Ranking Dashboard (Top-K) — composite")
    # hit for K bisnis
    t_biz = topk_table(y_true, y_score, [k_bisnis]).iloc[0].to_dict()
    lift50 = float(t_biz["Lift@K"])
    prec50 = float(t_biz["Precision@K"])
    captured = int(t_biz["Positives captured"])

    left, mid, right = st.columns([1, 1, 1])

    with left:
        st.markdown("### Summary")
        st.write(f"Holdout N = **{N}**")
        st.write(f"Positives = **{total_pos}** (prevalence = **{prevalence:.4f}**)")
        st.write(f"K bisnis = **{k_bisnis}**")
        st.markdown(f"**Lift@{k_bisnis} = {lift50:.2f}x**")
        st.markdown(f"Precision@{k_bisnis} = **{prec50*100:.2f}%**")
        st.write(f"Positives captured @K = **{captured}**")

    with mid:
        st.altair_chart(lift_bar.properties(title=f"Lift@K (Top-K only)"), use_container_width=True)

    with right:
        st.altair_chart(prec_bar.properties(title=f"Precision@K (Top-K only)"), use_container_width=True)

    st.markdown("### Top-K Summary Table (di bawah dashboard)")
    st.dataframe(
        tbl.assign(
            **{
                "Precision@K": (tbl["Precision@K"] * 100).round(2),
                "Recall@K": (tbl["Recall@K"] * 100).round(2),
                "Lift@K": tbl["Lift@K"].round(2),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# Debug/QA (buat nangkep error + mismatch kolom)
# -----------------------------
with tab_debug:
    st.subheader("Debug / QA")
    st.write("Ini sengaja ada supaya kamu bisa ngecek cepat: masalahnya di data atau di app.")
    st.markdown("### Kolom terdeteksi")
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
    st.dataframe(df_f.head(30), use_container_width=True)
