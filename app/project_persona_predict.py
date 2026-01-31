
# app.py
# Streamlit Persona Segmentation + Placement Prediction (PDF-aligned, 1-file version)

import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression


# =========================
# Config
# =========================
st.set_page_config(page_title="Persona Predict Dashboard", layout="wide")

YES_PLACEMENT_RE = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)
YES_GENERAL_RE = re.compile(r"\b(ya|iya|y|yes|pernah|sudah)\b", re.I)
NO_GENERAL_RE = re.compile(r"\b(tidak|gak|ga|nggak|belum|no)\b", re.I)


# =========================
# Utilities
# =========================
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _to_num(x):
    try:
        return float(str(x).strip())
    except Exception:
        return np.nan


def _is_dark_theme() -> bool:
    try:
        return (st.get_option("theme.base") or "").lower() == "dark"
    except Exception:
        return True


def _configure_altair_dark(chart: alt.Chart) -> alt.Chart:
    """Make charts readable in dark theme."""
    if _is_dark_theme():
        return (
            chart
            .configure_view(strokeOpacity=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="#333333",
                tickColor="#666666"
            )
            .configure_legend(
                labelColor="white",
                titleColor="white"
            )
            .configure_title(color="white")
        )
    return chart


def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================
# FE (PDF-aligned)
# =========================
def map_domain_pendidikan(text: str) -> str:
    t = _safe_str(text).lower()
    if t == "":
        return "Other"
    if any(k in t for k in ["informatika", "computer", "it", "sistem informasi", "teknik informatika", "teknik komputer"]):
        return "Tech / IT"
    if any(k in t for k in ["stat", "matemat", "math", "fisika", "kimia", "biologi"]):
        return "STEM"
    if any(k in t for k in ["akunt", "finance", "manajemen", "bisnis", "ekonomi", "accounting"]):
        return "Business"
    if any(k in t for k in ["komunikasi", "marketing", "humas", "pr", "ilmu komunikasi"]):
        return "Comms / Marketing"
    if any(k in t for k in ["hukum", "law"]):
        return "Law"
    return "Other"


def map_domain_product(text: str) -> str:
    t = _safe_str(text).lower()
    if t == "":
        return "Other"
    if any(k in t for k in ["data", "analytics", "analyst", "science", "machine learning", "ml", "ai"]):
        return "Data"
    if any(k in t for k in ["product", "pm"]):
        return "Product"
    if any(k in t for k in ["ui/ux", "ux", "ui", "design"]):
        return "Design"
    if any(k in t for k in ["marketing", "digital marketing", "seo", "ads"]):
        return "Marketing"
    if any(k in t for k in ["backend", "front", "fullstack", "web", "golang", "java", "python", "software", "developer"]):
        return "Engineering"
    return "Other"


def map_level_pendidikan(text: str) -> str:
    t = _safe_str(text).lower()
    if t == "":
        return "Other"
    if any(k in t for k in ["sma", "smk", "slta"]):
        return "SMA/SMK"
    if any(k in t for k in ["d1", "d2", "d3", "d4", "diploma"]):
        return "Diploma"
    if any(k in t for k in ["s1", "sarjana", "undergraduate"]):
        return "S1"
    if any(k in t for k in ["s2", "magister", "master"]):
        return "S2+"
    return "Other"


def map_kategori_pekerjaan(text: str) -> str:
    t = _safe_str(text).lower()
    if t == "":
        return "Other"
    if any(k in t for k in ["it", "tech", "developer", "engineer", "data", "software"]):
        return "Tech"
    if any(k in t for k in ["finance", "akunt", "account"]):
        return "Finance"
    if any(k in t for k in ["marketing", "sales", "bd", "business development"]):
        return "Sales/Marketing"
    if any(k in t for k in ["hr", "human resource"]):
        return "HR"
    return "Other"


def map_level_pekerjaan(text: str) -> str:
    t = _safe_str(text).lower()
    if t == "":
        return "Other"
    if any(k in t for k in ["intern", "magang"]):
        return "Intern"
    if any(k in t for k in ["junior", "entry"]):
        return "Junior"
    if any(k in t for k in ["mid", "associate"]):
        return "Mid"
    if any(k in t for k in ["senior", "lead", "manager", "head"]):
        return "Senior+"
    return "Other"


def map_motivasi_cluster(text: str) -> str:
    # dari PDF (rule-based, interpretable)
    t = _safe_str(text).lower().replace("nan", " ").strip()
    if t == "":
        return "Tidak disebutkan"

    if any(k in t for k in [
        "ganti karir", "switch karir", "career switch",
        "pindah karir", "pindah bidang", "alih karir", "transition career"
    ]):
        return "Ganti karir"

    if any(k in t for k in ["upskill", "upgrade", "menambah skill", "nambah skill", "skill baru", "improve"]):
        return "Upgrade diri"

    if any(k in t for k in ["fresh graduate", "freshgrad", "baru lulus", "baru wisuda", "belum punya pengalaman"]):
        return "Fresh graduate cari kerja"

    if any(k in t for k in ["naik gaji", "gaji lebih", "penghasilan", "income", "promosi", "jenjang karir"]):
        return "Naik gaji / karir"

    if any(k in t for k in [
        "cepat kerja", "cepat dapat kerja", "cepat dapet kerja",
        "langsung kerja", "jaminan kerja", "pasti kerja", "guarantee job"
    ]):
        return "Ingin cepat kerja & hasil tinggi"

    if any(k in t for k in ["sertifikat", "certificate", "sertif", "cv", "portofolio", "portfolio"]):
        return "Sertifikat / CV / Portofolio"

    if any(k in t for k in ["trend", "tren", "penasaran", "coba", "tiktok", "lagi rame", "lagi ramai"]):
        return "Ikut tren / penasaran"

    if any(k in t for k in ["disuruh", "diajak", "temen", "rekomendasi", "direkomendasikan"]):
        return "Disuruh / ikut orang"

    # fallback
    return "Lainnya"


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- numeric umur
    if "Umur" in df.columns:
        df["Umur"] = df["Umur"].apply(_to_num)

    # --- target placement (PDF-aligned)
    target_col = pick_first_existing(df, ["Penyaluran Kerja", "Penyaluran_kerja", "Placement", "Placed"])
    if target_col:
        df["Penyaluran_flag"] = df[target_col].apply(lambda x: 1 if YES_PLACEMENT_RE.search(_safe_str(x)) else 0)
    else:
        df["Penyaluran_flag"] = 0

    # --- region
    region_col = pick_first_existing(df, ["Kota(Jabodetabek)", "Domisili", "Provinsi", "Region"])
    if region_col:
        df["Region"] = df[region_col].astype(str).replace({"nan": ""}).fillna("Unknown")
    else:
        df["Region"] = "Unknown"

    # --- batch parsing
    if "Batch" in df.columns:
        df["Batch_str"] = df["Batch"].astype(str).fillna("")
        df["Batch_has_plus"] = df["Batch_str"].str.contains(r"\+", regex=True).astype(int)
        df["Batch_num"] = df["Batch_str"].str.extract(r"(\d+)")[0].apply(_to_num)
    else:
        df["Batch_str"] = ""
        df["Batch_has_plus"] = 0
        df["Batch_num"] = np.nan

    # --- flags: jobconnect, community, event
    job_col = pick_first_existing(df, ["Job Connect", "Program Job Connect", "Program_jobconnect"])
    if job_col:
        df["Program_jobconnect_flag"] = df[job_col].apply(
            lambda x: 0 if NO_GENERAL_RE.search(_safe_str(x).lower()) else (1 if YES_GENERAL_RE.search(_safe_str(x).lower()) else 0)
        )
    else:
        df["Program_jobconnect_flag"] = 0

    if "Community" in df.columns:
        df["Community_flag"] = df["Community"].apply(lambda x: 0 if "tidak" in _safe_str(x).lower() else 1)
    else:
        df["Community_flag"] = 0

    ev_col = pick_first_existing(df, ["Pernah ikut acara dibimbing/tidak", "Event", "Pernah ikut acara"])
    if ev_col:
        df["Event_flag"] = df[ev_col].apply(lambda x: 1 if YES_GENERAL_RE.search(_safe_str(x).lower()) else 0)
    else:
        df["Event_flag"] = 0

    # --- engagement level
    def engagement_level(row):
        jc = row.get("Program_jobconnect_flag", 0)
        cm = row.get("Community_flag", 0)
        ev = row.get("Event_flag", 0)
        if jc == 1 and cm == 1 and ev == 1:
            return "High"
        if (jc + cm + ev) >= 2:
            return "Medium"
        if (jc + cm + ev) == 1:
            return "Low"
        return "Non-engaged"

    df["Engagement_level"] = df.apply(engagement_level, axis=1).astype("category")

    # --- text merge for motivasi mapping (PDF)
    text_cols = [
        "Motivasi mengikuti bootcamp",
        "Alasan mengambil kategori Bootcamp",
        "Mengapa memilih Dibimbing"
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
        else:
            df[c] = ""

    df["Motivasi_raw_all"] = (df[text_cols[0]] + " " + df[text_cols[1]] + " " + df[text_cols[2]]).astype(str)
    df["Motivasi_cluster"] = df["Motivasi_raw_all"].apply(map_motivasi_cluster).astype("category")

    # --- switcher
    alasan = df["Alasan mengambil kategori Bootcamp"].astype(str).str.lower()
    df["is_switcher"] = alasan.apply(
        lambda t: 1 if any(k in t for k in ["switch", "alih", "pindah", "ganti karir", "career switch", "pindah bidang"]) else 0
    )

    # --- segmen karir
    df["Segmen_karir"] = df["is_switcher"].map({1: "Career switcher", 0: "Upskiller"}).astype("category")

    # --- domain pendidikan / product
    edu_col = pick_first_existing(df, ["Background pendidikan", "Jurusan pendidikan", "Pendidikan"])
    if edu_col:
        df["Domain_pendidikan"] = df[edu_col].apply(map_domain_pendidikan).astype("category")
    else:
        df["Domain_pendidikan"] = "Other"

    prod_col = pick_first_existing(df, ["Product", "Produk", "Program"])
    if prod_col:
        df["Domain_product"] = df[prod_col].apply(map_domain_product).astype("category")
    else:
        df["Domain_product"] = "Other"

    # --- level pendidikan FE
    lp_col = pick_first_existing(df, ["Level pendidikan", "Pendidikan terakhir", "Level Pendidikan"])
    if lp_col:
        df["Level_pendidikan_FE"] = df[lp_col].apply(map_level_pendidikan).astype("category")
    else:
        df["Level_pendidikan_FE"] = "Other"

    # --- pekerjaan FE
    kp_col = pick_first_existing(df, ["Kategori Pekerjaan", "Kategori pekerjaan"])
    if kp_col:
        df["Kategori_Pekerjaan_FE"] = df[kp_col].apply(map_kategori_pekerjaan).astype("category")
    else:
        df["Kategori_Pekerjaan_FE"] = "Other"

    lvlp_col = pick_first_existing(df, ["Level Pekerjaan", "Level pekerjaan"])
    if lvlp_col:
        df["Level_Pekerjaan_FE"] = df[lvlp_col].apply(map_level_pekerjaan).astype("category")
    else:
        df["Level_Pekerjaan_FE"] = "Other"

    # --- umur bin
    if "Umur" in df.columns:
        df["Umur_bin"] = pd.cut(
            df["Umur"],
            bins=[0, 18, 23, 28, 35, 100],
            labels=["<=18", "19-23", "24-28", "29-35", "36+"],
            include_lowest=True
        ).astype("category")
    else:
        df["Umur_bin"] = "Unknown"

    # --- label untuk dashboard
    df["Penyaluran_label"] = df["Penyaluran_flag"].map({0: "Belum tersalur", 1: "Tersalur kerja"}).astype("category")

    # --- motivasi risk flag (PDF style)
    def flag_misalignment(row):
        mot = _safe_str(row.get("Motivasi_cluster", ""))
        eng = _safe_str(row.get("Engagement_level", ""))
        is_sw = int(row.get("is_switcher", 0))
        risk = 0
        if mot in ["Ikut tren / penasaran", "Disuruh / ikut orang"]:
            risk += 1
        if mot == "Sertifikat / CV / Portofolio":
            risk += 1
        if mot == "Ingin cepat kerja & hasil tinggi":
            risk += 2
        if is_sw == 1:
            risk += 1
        if eng == "Non-engaged":
            risk += 2
        elif eng == "Low":
            risk += 1

        if risk >= 4:
            return "High Risk"
        if risk >= 2:
            return "Medium Risk"
        return "Low Risk"

    df["Motivasi_risk_flag"] = df.apply(flag_misalignment, axis=1).astype("category")

    return df


def make_preprocessor(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def stratified_sample(df: pd.DataFrame, y_col: str, n: int, seed: int = 42, min_per_class: int = 30):
    if n >= len(df):
        return df

    y = df[y_col].astype(int)
    pos_idx = df.index[y == 1].tolist()
    neg_idx = df.index[y == 0].tolist()

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # can't stratify, fallback random
        return df.sample(n=n, random_state=seed)

    # ensure both classes appear
    n_pos = min(max(min_per_class, int(n * 0.2)), len(pos_idx), n - 1)
    n_neg = n - n_pos
    n_neg = min(n_neg, len(neg_idx))
    n_pos = n - n_neg  # adjust if neg limited

    rs = np.random.RandomState(seed)
    pick_pos = rs.choice(pos_idx, size=n_pos, replace=False)
    pick_neg = rs.choice(neg_idx, size=n_neg, replace=False)

    return df.loc[np.concatenate([pick_pos, pick_neg])]


def persona_name_from_cluster(profile: pd.DataFrame) -> str:
    """Heuristic mapping: align to PDF persona names."""
    # profile is a 1-row summary table we build; but easier: pass dict-like
    # We'll infer from dominant Kategori Kesibukan + engagement + is_switcher.
    # If missing, fallback.
    try:
        dom_kesibukan = str(profile.get("Kategori Kesibukan_mode", ""))
        dom_eng = str(profile.get("Engagement_level_mode", ""))
        switch_rate = float(profile.get("is_switcher_mean", 0.0))
        umur_mean = float(profile.get("Umur_mean", np.nan))
    except Exception:
        return "Persona"

    if "Mahasiswa" in dom_kesibukan or "Fresh" in dom_kesibukan or "Siswa" in dom_kesibukan:
        return "Fresh Graduate Explorer"

    if "Pekerja" in dom_kesibukan or "Usaha" in dom_kesibukan:
        return "Working Professional Upskiller"

    # job seeker / others
    if dom_eng in ["High", "Medium"] and switch_rate >= 0.4:
        return "High Engagement Career Switcher"

    # tie-break
    if not np.isnan(umur_mean) and umur_mean >= 28:
        return "Working Professional Upskiller"
    return "High Engagement Career Switcher"


# =========================
# Load Data
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file, encoding_errors="replace")


# =========================
# UI: Sidebar Data Source
# =========================
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

with st.sidebar:
    st.header("Data source")

    source_mode = st.radio(
        "Choose",
        ["Upload file (CSV/XLSX/GZ)", "Repo file (local path)"],
        index=1,
        key="data_source_mode"
    )

    df_raw = None
    if source_mode.startswith("Upload"):
        up = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
        if up is not None:
            df_raw = load_csv(up)
    else:
        default_path = "raw_data.csv"
        path = st.text_input("Path", value=default_path, key="repo_path")
        try:
            df_raw = load_csv(path)
        except Exception as e:
            st.error(f"Can't read path: {e}")

    st.caption("Tips: kalau deploy Streamlit Cloud, pastikan path repo benar (mis. raw_data/raw_data.csv).")

if df_raw is None:
    st.info("Upload CSV atau isi path repo dulu.")
    st.stop()

df = build_features(df_raw)

# =========================
# Global Filters (EDA-friendly)
# =========================
with st.sidebar:
    st.header("Global filters (EDA)")

    # build filters safely
    def multi_filter(label, col, key):
        if col in df.columns:
            opts = sorted([x for x in df[col].dropna().unique().tolist() if str(x) != "nan"])
            sel = st.multiselect(label, opts, default=[], key=key)
            return sel
        return []

    f_channel = multi_filter("Channel", "Channel", "f_channel")
    f_month = multi_filter("Month", "Month", "f_month")
    f_product = multi_filter("Product", "Product", "f_product")
    f_region = multi_filter("Region", "Region", "f_region")
    f_kesibukan = multi_filter("Kategori Kesibukan", "Kategori Kesibukan", "f_kesibukan")
    f_eng = multi_filter("Engagement_level", "Engagement_level", "f_eng")
    f_motiv = multi_filter("Motivasi_cluster", "Motivasi_cluster", "f_motiv")
    f_segkar = multi_filter("Segmen_karir", "Segmen_karir", "f_segkar")

def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    def _apply(col, sel):
        nonlocal d
        if sel and col in d.columns:
            d = d[d[col].isin(sel)]
    _apply("Channel", f_channel)
    _apply("Month", f_month)
    _apply("Product", f_product)
    _apply("Region", f_region)
    _apply("Kategori Kesibukan", f_kesibukan)
    _apply("Engagement_level", f_eng)
    _apply("Motivasi_cluster", f_motiv)
    _apply("Segmen_karir", f_segkar)
    return d

df_f = apply_filters(df)

# =========================
# Tabs
# =========================
tab_overview, tab_eda, tab_cluster, tab_supervised = st.tabs(
    ["Overview", "EDA", "Clustering (Persona)", "Supervised (Top-K)"]
)

# =========================
# Overview
# =========================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    total = len(df_f)
    pos = int(df_f["Penyaluran_flag"].sum()) if "Penyaluran_flag" in df_f.columns else 0
    neg = total - pos
    rate = (pos / total * 100) if total else 0.0

    c1.metric("Rows (after filters)", f"{total:,}")
    c2.metric("Tersalur kerja (1)", f"{pos:,}")
    c3.metric("Belum tersalur (0)", f"{neg:,}")
    c4.metric("Target rate", f"{rate:.2f}%")

    st.subheader("Sample data (filtered)")
    st.dataframe(df_f.head(30), use_container_width=True)


# =========================
# EDA
# =========================
with tab_eda:
    st.subheader("EDA — Pattern peserta & target rate (selaras tujuan proyek)")

    left, right = st.columns([1, 1])

    with left:
        breakdown_candidates = [
            "Channel", "Month", "Product", "Domain_product", "Domain_pendidikan",
            "Region", "Engagement_level", "Motivasi_cluster", "Motivasi_risk_flag",
            "Segmen_karir", "Kategori Kesibukan", "Level_pendidikan_FE"
        ]
        breakdown_candidates = [c for c in breakdown_candidates if c in df_f.columns]

        breakdown = st.selectbox(
            "Pilih breakdown untuk target-rate",
            breakdown_candidates,
            index=0,
            key="eda_breakdown"
        )
        top_n = st.slider("Top-N categories", 5, 60, 20, key="eda_topn")

    with right:
        st.caption("Catatan: target placement sangat minoritas di data PDF (≈0.2%). Grafik rate lebih meaningful kalau lihat count + rate barengan.")
        show_min_count = st.slider("Min count per category (filter noise)", 1, 200, 10, key="eda_mincount")

    # aggregate
    agg = (
        df_f
        .groupby(breakdown, dropna=False)["Penyaluran_flag"]
        .agg(total="count", positives="sum")
        .reset_index()
    )
    agg["placement_rate_pct"] = np.where(agg["total"] > 0, agg["positives"] / agg["total"] * 100.0, 0.0)

    agg = agg[agg["total"] >= show_min_count].copy()
    agg = agg.sort_values(["placement_rate_pct", "total"], ascending=[False, False]).head(top_n)

    # bar chart: rate
    base = alt.Chart(agg).encode(
        y=alt.Y(f"{breakdown}:N", sort="-x", title=None),
        x=alt.X("placement_rate_pct:Q", title="Placement rate (%)")
    )

    bars = base.mark_bar().encode(
        tooltip=[breakdown, "total", "positives", alt.Tooltip("placement_rate_pct:Q", format=".2f")]
    )

    text = base.mark_text(align="left", dx=3).encode(
        text=alt.Text("placement_rate_pct:Q", format=".1f")
    )

    chart = (bars + text).properties(height=520, title=f"Target rate by {breakdown}")
    chart = _configure_altair_dark(chart)

    st.altair_chart(chart, use_container_width=True)

    st.subheader("Table")
    st.dataframe(agg, use_container_width=True)

    st.subheader("Distribusi variabel inti (untuk segmentasi & strategi)")
    dist_cols = [c for c in ["Engagement_level", "Motivasi_cluster", "Segmen_karir", "Kategori Kesibukan", "Domain_product"] if c in df_f.columns]
    if dist_cols:
        colA, colB = st.columns(2)
        for i, col in enumerate(dist_cols[:4]):
            counts = df_f[col].astype(str).value_counts().reset_index()
            counts.columns = [col, "count"]
            ch = alt.Chart(counts).mark_bar().encode(
                y=alt.Y(f"{col}:N", sort="-x", title=None),
                x=alt.X("count:Q", title="Count"),
                tooltip=[col, "count"]
            ).properties(height=260, title=f"Distribusi {col}")
            ch = _configure_altair_dark(ch)
            (colA if i % 2 == 0 else colB).altair_chart(ch, use_container_width=True)
    else:
        st.info("Kolom distribusi inti belum tersedia di data.")


# =========================
# Clustering
# =========================
with tab_cluster:
    st.subheader("Clustering — Persona (3 cluster, named, PDF-aligned)")

    st.caption("Ini align ke PDF: MiniBatchKMeans + SVD 2D untuk plot. Default K=3 dan persona diberi nama.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        k = st.slider("Jumlah cluster (K)", 2, 6, 3, key="clust_k")
    with col2:
        random_state = st.number_input("Random seed", 0, 9999, 42, key="clust_seed")
    with col3:
        use_filtered = st.checkbox("Gunakan data hasil filter global", value=True, key="clust_use_filtered")

    df_c = df_f.copy() if use_filtered else df.copy()

    # feature cols (PDF list-ish)
    feature_candidates = [
        "Umur", "Umur_bin", "Region",
        "Batch_num", "Batch_has_plus",
        "Community_flag", "Event_flag", "Engagement_level",
        "Program_jobconnect_flag",
        "Motivasi_cluster", "Motivasi_risk_flag",
        "Domain_pendidikan", "Domain_product",
        "is_switcher", "Segmen_karir",
        "Product", "Kategori",
        "Month", "Channel",
        "Level_pendidikan_FE",
        "Kategori Kesibukan",
        "Level_Pekerjaan_FE", "Kategori_Pekerjaan_FE",
    ]
    feature_candidates = [c for c in feature_candidates if c in df_c.columns]

    selected_features = st.multiselect(
        "Feature cols (clustering)",
        feature_candidates,
        default=feature_candidates,
        key="clust_feat_cols"  # FIX duplicate id
    )

    run = st.button("Run clustering", key="btn_run_clustering")
    if run:
        if len(selected_features) < 2:
            st.error("Pilih minimal 2 feature.")
            st.stop()

        pre = make_preprocessor(df_c, selected_features)
        X = pre.fit_transform(df_c[selected_features])

        # SVD for plotting + stable clustering input (works for sparse)
        svd = TruncatedSVD(n_components=50 if X.shape[1] > 50 else min(10, X.shape[1]-1), random_state=random_state)
        X_red = svd.fit_transform(X)

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1024, n_init="auto")
        cluster_id = kmeans.fit_predict(X_red)

        df_out = df_c.copy()
        df_out["cluster_id"] = cluster_id

        # Build simple profile to name personas
        # We compute mode for key columns per cluster + means
        profiles = []
        for cid, g in df_out.groupby("cluster_id"):
            row = {"cluster_id": cid, "n": len(g)}
            if "Umur" in g.columns:
                row["Umur_mean"] = float(pd.to_numeric(g["Umur"], errors="coerce").mean())
            row["is_switcher_mean"] = float(pd.to_numeric(g.get("is_switcher", 0), errors="coerce").mean())

            for col in ["Kategori Kesibukan", "Engagement_level", "Motivasi_cluster", "Segmen_karir", "Domain_product"]:
                if col in g.columns:
                    mode = g[col].astype(str).value_counts().index[0]
                    row[f"{col}_mode"] = mode
            profiles.append(row)

        prof_df = pd.DataFrame(profiles)
        prof_df["persona_name"] = prof_df.apply(lambda r: persona_name_from_cluster(r.to_dict()), axis=1)

        # map cluster -> persona name
        persona_map = dict(zip(prof_df["cluster_id"], prof_df["persona_name"]))
        df_out["Persona"] = df_out["cluster_id"].map(persona_map).fillna(df_out["cluster_id"].astype(str))

        # SVD 2D for visualization (PDF-like)
        svd2 = TruncatedSVD(n_components=2, random_state=random_state)
        X2 = svd2.fit_transform(X)

        plot_df = pd.DataFrame({
            "SVD1": X2[:, 0],
            "SVD2": X2[:, 1],
            "Persona": df_out["Persona"].astype(str),
        })

        st.success("Clustering selesai.")
        st.write("Persona mapping:")
        st.dataframe(prof_df.sort_values("n", ascending=False), use_container_width=True)

        ch = alt.Chart(plot_df).mark_circle(size=35, opacity=0.65).encode(
            x=alt.X("SVD1:Q", title="SVD-1"),
            y=alt.Y("SVD2:Q", title="SVD-2"),
            color=alt.Color("Persona:N", title="Persona"),
            tooltip=["Persona"]
        ).properties(height=520, title="Cluster Visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)")
        st.altair_chart(_configure_altair_dark(ch), use_container_width=True)

        st.subheader("Cluster profiling (ringkas)")
        # quick profiling table
        prof_cols = [c for c in ["Persona", "Engagement_level", "Motivasi_cluster", "Segmen_karir", "Kategori Kesibukan", "Domain_product"] if c in df_out.columns]
        if prof_cols:
            prof2 = (
                df_out
                .groupby("Persona")[["Penyaluran_flag"]].agg(total="count", positives="sum").reset_index()
            )
            prof2["placement_rate_pct"] = prof2["positives"] / prof2["total"] * 100.0
            st.dataframe(prof2.sort_values("placement_rate_pct", ascending=False), use_container_width=True)

        st.subheader("Preview data + persona")
        st.dataframe(df_out[["Persona"] + [c for c in selected_features if c in df_out.columns] + ["Penyaluran_flag"]].head(50),
                     use_container_width=True)


# =========================
# Supervised (Top-K)
# =========================
with tab_supervised:
    st.subheader("Supervised ranking — Top-K (PDF-aligned)")
    st.caption("Tujuan: ranking peserta paling berpeluang tersalur kerja (Top-K sesuai kapasitas bisnis).")

    left, right = st.columns([1, 1])

    with left:
        use_filtered_sup = st.checkbox("Gunakan data hasil filter global", value=True, key="sup_use_filtered")
        test_size = st.slider("test_size (holdout)", 0.1, 0.4, 0.2, 0.05, key="sup_test_size")
        seed = st.number_input("Random seed", 0, 9999, 42, key="sup_seed")

    with right:
        k_cap = st.number_input("Business capacity K (berapa peserta yang bisa diintervensi)", 10, 2000, 200, key="sup_kcap")
        do_sample = st.checkbox("Gunakan sampling (stratified) untuk cepat", value=True, key="sup_do_sample")
        sample_n = st.number_input("Sample size", 500, 20000, 3000, key="sup_sample_n")

    df_s = df_f.copy() if use_filtered_sup else df.copy()

    # show target distribution BEFORE sampling
    total = len(df_s)
    pos = int(df_s["Penyaluran_flag"].sum())
    neg = total - pos
    st.write(f"Target counts (work/sup): **0={neg}**, **1={pos}**, total={total}")

    if do_sample:
        df_s = stratified_sample(df_s, "Penyaluran_flag", n=min(int(sample_n), len(df_s)), seed=int(seed), min_per_class=30)
        st.info(f"Using stratified sample: {len(df_s):,} rows (pos={int(df_s['Penyaluran_flag'].sum())}, neg={len(df_s)-int(df_s['Penyaluran_flag'].sum())})")

    # feature cols (supervised)
    feature_candidates = [
        "Umur", "Umur_bin", "Region",
        "Batch_num", "Batch_has_plus",
        "Community_flag", "Event_flag", "Engagement_level",
        "Program_jobconnect_flag",
        "Motivasi_cluster", "Motivasi_risk_flag",
        "Domain_pendidikan", "Domain_product",
        "is_switcher", "Segmen_karir",
        "Product", "Kategori", "Month", "Channel",
        "Level_pendidikan_FE", "Kategori Kesibukan",
        "Level_Pekerjaan_FE", "Kategori_Pekerjaan_FE",
    ]
    feature_candidates = [c for c in feature_candidates if c in df_s.columns]

    feat_cols = st.multiselect(
        "Feature cols (supervised)",
        feature_candidates,
        default=feature_candidates,
        key="sup_feat_cols"  # FIX duplicate id
    )

    if st.button("Run supervised ranking", key="btn_run_supervised"):
        if len(feat_cols) < 2:
            st.error("Pilih minimal 2 feature.")
            st.stop()

        y = df_s["Penyaluran_flag"].astype(int).values
        Xdf = df_s[feat_cols].copy()

        pos = int(y.sum())
        neg = int((y == 0).sum())

        # must have both classes
        if pos < 2 or neg < 2:
            st.error(
                f"Data terlalu imbalanced untuk split & train (pos={pos}, neg={neg}). "
                "Coba gunakan data full (matikan filter/sampling) atau cek label target."
            )
            st.stop()

        # split with stratify, try ensure test has at least 1 positive
        ok = False
        for rs in [int(seed), int(seed)+1, int(seed)+2, int(seed)+3, int(seed)+4]:
            X_train, X_test, y_train, y_test = train_test_split(
                Xdf, y, test_size=float(test_size), random_state=rs, stratify=y
            )
            if y_test.sum() >= 1 and y_train.sum() >= 1:
                ok = True
                break

        if not ok:
            st.warning("Tidak bisa menjamin ada positive di test. Tetap lanjut, tapi metrik bisa misleading.")

        pre = make_preprocessor(df_s, feat_cols)

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        )

        model = Pipeline(steps=[
            ("pre", pre),
            ("clf", clf)
        ])

        model.fit(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]

        # metrics
        try:
            roc = roc_auc_score(y_test, proba_test)
        except Exception:
            roc = np.nan
        try:
            pr = average_precision_score(y_test, proba_test)
        except Exception:
            pr = np.nan

        st.write(f"ROC-AUC: **{roc:.4f}** | PR-AUC: **{pr:.4f}**")
        y_pred = (proba_test >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion matrix (threshold=0.5):")
        st.write(cm)

        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred, digits=4))

        # Top-K on ALL (scoring full df_s)
        proba_all = model.predict_proba(Xdf)[:, 1]
        df_rank = df_s.copy()
        df_rank["score"] = proba_all
        df_rank = df_rank.sort_values("score", ascending=False).reset_index(drop=True)

        K = int(min(k_cap, len(df_rank)))
        topk = df_rank.head(K)
        capture = float(topk["Penyaluran_flag"].sum())
        total_pos = float(df_rank["Penyaluran_flag"].sum())
        capture_rate = (capture / total_pos * 100.0) if total_pos > 0 else 0.0

        st.subheader("Top-K results")
        st.write(f"K={K:,} | positives captured in Top-K: **{int(capture)} / {int(total_pos)}** ({capture_rate:.2f}%)")

        show_cols = [c for c in ["score", "Penyaluran_flag", "Persona", "Product", "Channel", "Month", "Region", "Engagement_level", "Motivasi_cluster"] if c in df_rank.columns]
        st.dataframe(topk[show_cols].head(200), use_container_width=True)

        # feature importance (logreg coeff)
        try:
            ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
            cat_cols = model.named_steps["pre"].transformers_[1][2]
            num_cols = model.named_steps["pre"].transformers_[0][2]

            cat_names = list(ohe.get_feature_names_out(cat_cols))
            feat_names = list(num_cols) + cat_names
            coef = model.named_steps["clf"].coef_[0]
            imp = pd.DataFrame({"feature": feat_names, "coef": coef})
            imp["abs"] = imp["coef"].abs()
            imp = imp.sort_values("abs", ascending=False).head(30)

            st.subheader("Top drivers (|coef| terbesar)")
            st.dataframe(imp[["feature", "coef"]], use_container_width=True)
        except Exception as e:
            st.info(f"Tidak bisa tampilkan feature importance detail (reason: {e})")

        # score distribution plot
        hist = alt.Chart(df_rank).mark_bar().encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Score"),
            y=alt.Y("count():Q", title="Count")
        ).properties(height=260, title="Score distribution")
        st.altair_chart(_configure_altair_dark(hist), use_container_width=True)
