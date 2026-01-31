# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Persona Segmentation & Placement Prediction (PDF-aligned)",
    layout="wide",
)

PLOTLY_TEMPLATE = "plotly_dark"

# =========================
# Helpers
# =========================
def _safe_lower(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def make_target_from_penyaluran(x: object) -> float:
    """
    Default target: Penyaluran Kerja (Interest) -> 1 if 'tertarik', 0 if 'tidak tertarik'.
    Unknown -> NaN.

    NOTE: Data kamu sangat imbalanced (positif dominan).
    Supervised di app ini dibuat robust via CV, bukan train/test yang gampang gagal.
    """
    t = _safe_lower(x)
    if not t or t in {"-", "nan", "none"}:
        return np.nan
    # Negative if contains "tidak" (even if contains tertarik)
    if "tidak" in t or "gak" in t or "ga " in t or t.startswith("ga"):
        return 0.0
    # Positive if tertarik or yes-like
    if "tertarik" in t or t == "ya" or t.startswith("ya "):
        return 1.0
    # fallback keywords
    if "tersalur" in t or "placed" in t or "berhasil" in t or "sudah" in t:
        return 1.0
    return np.nan

def infer_motivasi_cluster(text: object) -> str:
    t = _safe_lower(text)
    if not t:
        return "Lainnya"

    # Order matters (more specific first)
    if re.search(r"\bfreelance\b|\bsampingan\b|\bside\b", t):
        return "Freelance"
    if re.search(r"\bswitch\b|\bpindah\b|\bberalih\b|\bswitch career\b|\bcareer switch\b", t):
        return "Switch career"
    if re.search(r"\bcari kerja\b|\bjob\b|\bkerja\b|\bjob seeker\b|\bpenyaluran\b", t):
        return "Dapat kerja"
    if re.search(r"\bupgrade\b|\bimprove\b|\bmeningkatkan\b|\bnaik\b|\bportfolio\b|\bcv\b", t):
        return "Upgrade diri"
    if re.search(r"\bbelajar\b|\bmempelajari\b|\bmendalami\b|\bmengasah\b|\bdari nol\b|\bfrom scratch\b", t):
        return "Belajar skill"
    return "Lainnya"

def infer_segmen_karir(row: pd.Series) -> str:
    """
    Heuristik "Segmen karir" biar EDA & dashboard selaras dengan tujuan proyek.
    """
    kesibukan = _safe_lower(row.get("Kategori Kesibukan", ""))
    level_pekerjaan = _safe_lower(row.get("Level Pekerjaan", ""))
    kategori_pekerjaan = _safe_lower(row.get("Kategori Pekerjaan", ""))
    motivasi = _safe_lower(row.get("Motivasi mengikuti bootcamp", ""))

    # Fresh graduate / student
    if any(k in kesibukan for k in ["mahasiswa", "student", "kuliah", "pelajar"]):
        return "Fresh Graduate Explorer"

    # Job seeker (not working)
    if any(k in kesibukan for k in ["job seeker", "unemployed", "belum bekerja", "nganggur"]):
        return "Fresh Graduate Explorer"

    # Career switcher
    if re.search(r"\bswitch\b|\bpindah\b|\bberalih\b|\bcareer\b", motivasi):
        return "High Engagement Career Switcher"

    # Working professional
    if any(k in kesibukan for k in ["karyawan", "pegawai", "bekerja", "full time", "full-time", "freelancer", "wirausaha"]):
        return "Working Professional Upskiller"
    if any(k in level_pekerjaan for k in ["staff", "manager", "lead", "senior", "junior", "intern"]):
        return "Working Professional Upskiller"
    if kategori_pekerjaan:
        return "Working Professional Upskiller"

    return "Working Professional Upskiller"

def bool_from_yes_like(x: object) -> int:
    t = _safe_lower(x)
    if not t:
        return 0
    if re.search(r"\bya\b|\byes\b|\by\b|\bsudah\b|\bpernah\b|\baktif\b", t):
        return 1
    return 0

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # keep original names (because your dataset uses spaces), but ensure column presence safely
    return df

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Target
    if "Penyaluran Kerja" in out.columns:
        out["target"] = out["Penyaluran Kerja"].apply(make_target_from_penyaluran)
    else:
        out["target"] = np.nan

    # Motivasi cluster
    if "Motivasi mengikuti bootcamp" in out.columns:
        out["Motivasi_cluster"] = out["Motivasi mengikuti bootcamp"].apply(infer_motivasi_cluster)
    else:
        out["Motivasi_cluster"] = "Lainnya"

    # Segmen karir
    out["Segmen_karir"] = out.apply(infer_segmen_karir, axis=1)

    # Engagement signals (best-effort based on available columns)
    for col, newcol in [
        ("Blog dibimbing", "blog_flag"),
        ("Community", "community_flag"),
        ("Pernah ikut acara dibimbing/tidak", "event_flag"),
        ("Mendapat informasi melalui program job connector?", "program_jobconnect_flag"),
    ]:
        if col in out.columns:
            out[newcol] = out[col].apply(bool_from_yes_like)
        else:
            out[newcol] = 0

    out["engagement_score"] = out[["blog_flag", "community_flag", "event_flag", "program_jobconnect_flag"]].sum(axis=1)

    # Date parsing (best-effort)
    for date_col in ["Tanggal Gabungan", "Tanggal Aktivasi"]:
        if date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
    if "Tanggal Gabungan" in out.columns:
        out["Month"] = out["Tanggal Gabungan"].dt.to_period("M").astype(str)
    else:
        out["Month"] = "Unknown"

    return out

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = standardize_columns(df)
    return df

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_columns(df)
    return df

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, vals in filters.items():
        if col not in out.columns:
            continue
        if not vals:
            continue
        out = out[out[col].astype(str).isin([str(v) for v in vals])]
    return out

def pick_numeric_and_categorical(df: pd.DataFrame, exclude: set) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in exclude]
    numeric = []
    categorical = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical

def plot_target_rate_bar(group_df: pd.DataFrame, title: str, top_n: int = 30):
    """
    group_df columns required: Group, Total, Positives, target_rate_pct
    """
    g = group_df.sort_values(["target_rate_pct", "Total"], ascending=[False, False]).head(top_n)
    fig = px.bar(
        g,
        y="Group",
        x="target_rate_pct",
        orientation="h",
        text=g["target_rate_pct"].map(lambda v: f"{v:.1f}%"),
        hover_data=["Total", "Positives"],
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Target rate (%)", yaxis_title="")
    return fig

def make_group_table(df: pd.DataFrame, group_col: str, target_col: str = "target", min_count: int = 20) -> pd.DataFrame:
    tmp = df[[group_col, target_col]].copy()
    tmp = tmp.dropna(subset=[target_col])
    if tmp.empty:
        return pd.DataFrame(columns=["Group", "Total", "Positives", "target_rate_pct"])
    agg = tmp.groupby(group_col)[target_col].agg(["count", "sum"]).reset_index()
    agg.columns = ["Group", "Total", "Positives"]
    agg = agg[agg["Total"] >= min_count]
    agg["target_rate_pct"] = (agg["Positives"] / agg["Total"]) * 100.0
    agg = agg.sort_values(["target_rate_pct", "Total"], ascending=[False, False])
    return agg

# =========================
# Sidebar: Data source
# =========================
st.sidebar.title("Data source")

source_mode = st.sidebar.radio(
    "Choose",
    ["Upload file (CSV)", "Repo file (path)"],
    key="source_mode",
)

df_raw = None
if source_mode == "Upload file (CSV)":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
    if up is None:
        st.info("Upload CSV dulu untuk mulai.")
        st.stop()
    df_raw = load_data_from_upload(up)
else:
    default_path = "raw_data/raw_data.csv"
    path = st.sidebar.text_input("Path", value=default_path, key="repo_path")
    try:
        df_raw = load_data_from_path(path)
    except Exception as e:
        st.error(f"Can't read path: {e}")
        st.caption("Tips: kalau di repo kamu path-nya biasanya seperti `raw_data/raw_data.csv`.")
        st.stop()

df = ensure_features(df_raw)

# =========================
# Global filters
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Global filters")

# Candidate filter columns (categorical-ish)
candidate_filters = []
for c in df.columns:
    if c in {"target"}:
        continue
    if df[c].nunique(dropna=True) <= 60 and not pd.api.types.is_numeric_dtype(df[c]):
        candidate_filters.append(c)

default_filter_cols = [c for c in ["Product", "Kategori", "Month", "Segmen_karir", "Motivasi_cluster"] if c in df.columns]

filter_cols = st.sidebar.multiselect(
    "Pilih kolom untuk filter",
    options=sorted(candidate_filters),
    default=default_filter_cols,
    key="filter_cols",
)

filters = {}
for c in filter_cols:
    opts = sorted([x for x in df[c].dropna().astype(str).unique().tolist()])
    default_vals = []
    # keep default empty (no filter) unless user chooses
    filters[c] = st.sidebar.multiselect(
        f"{c}",
        options=opts,
        default=default_vals,
        key=f"filter_{c}",
    )

df_f = apply_filters(df, filters)

# =========================
# Header / Overview
# =========================
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows (filtered)", f"{len(df_f):,}")
with c2:
    known_target = df_f["target"].notna().sum()
    st.metric("Target known", f"{known_target:,}")
with c3:
    pos = int((df_f["target"] == 1).sum())
    st.metric("Target positives", f"{pos:,}")
with c4:
    if known_target > 0:
        rate = (df_f["target"].mean()) * 100
        st.metric("Base target rate", f"{rate:.1f}%")
    else:
        st.metric("Base target rate", "-")

st.caption(
    "Tujuan: memahami pola peserta, segmentasi persona yang akurat, dan model prediktif peluang penyaluran kerja "
    "untuk mendukung keputusan bisnis (akuisisi, desain program, intervensi)."
)

# =========================
# Tabs
# =========================
tab_overview, tab_eda, tab_cluster, tab_sup = st.tabs(["📌 Overview", "🔎 EDA", "🧩 Clustering", "🎯 Supervised (Top-K)"])

# -------------------------
# Overview
# -------------------------
with tab_overview:
    st.subheader("Data preview")
    st.dataframe(df_f.head(50), use_container_width=True)

    st.subheader("Target definition")
    st.markdown(
        """
Default target menggunakan kolom **Penyaluran Kerja**:
- **1** = mengandung *“tertarik”* (dan tidak mengandung *“tidak”*)
- **0** = mengandung *“tidak”* / *“tidak tertarik”*
- selain itu = **NaN** (unknown)

Karena data sangat **imbalanced** (positif jauh lebih banyak), bagian supervised dibuat robust pakai **Stratified CV** (bukan split yang gampang gagal).
        """.strip()
    )

# -------------------------
# EDA
# -------------------------
with tab_eda:
    st.subheader("EDA — fleksibel (filter + breakdown apa pun)")

    left, right = st.columns([0.35, 0.65])
    with left:
        # pick breakdown column from categorical candidates + engineered columns
        breakdown_candidates = sorted(
            list(set(candidate_filters + ["Segmen_karir", "Motivasi_cluster"]))
        )
        default_breakdown = "Segmen_karir" if "Segmen_karir" in df_f.columns else breakdown_candidates[0]
        breakdown_col = st.selectbox(
            "Pilih breakdown column",
            options=breakdown_candidates,
            index=breakdown_candidates.index(default_breakdown) if default_breakdown in breakdown_candidates else 0,
            key="eda_breakdown",
        )
        min_count = st.slider("Min count per group", 1, 300, 30, key="eda_min_count")
        top_n = st.slider("Show top-N groups", 5, 60, 30, key="eda_topn")

    with right:
        group_tbl = make_group_table(df_f, breakdown_col, "target", min_count=min_count)
        if group_tbl.empty:
            st.warning("Tidak ada group yang memenuhi syarat (target kosong atau min_count terlalu tinggi).")
        else:
            fig = plot_target_rate_bar(group_tbl, f"Target rate by {breakdown_col}", top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(group_tbl.head(top_n), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribusi & volume (biar nggak ketipu 100% karena sampel kecil)")

    colA, colB = st.columns(2)
    with colA:
        # volume by breakdown
        vol = df_f[breakdown_col].astype(str).value_counts().reset_index()
        vol.columns = ["Group", "Count"]
        vol = vol.head(top_n)
        figv = px.bar(
            vol,
            y="Group",
            x="Count",
            orientation="h",
            title=f"Volume by {breakdown_col} (Top-{top_n})",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(figv, use_container_width=True)

    with colB:
        # target distribution
        tmp = df_f["target"].dropna()
        if tmp.empty:
            st.info("Target kosong setelah filter.")
        else:
            dist = tmp.value_counts().reset_index()
            dist.columns = ["target", "count"]
            dist["target"] = dist["target"].map({0.0: "0 (Tidak)", 1.0: "1 (Tertarik)"})
            figd = px.pie(dist, values="count", names="target", title="Target distribution", template=PLOTLY_TEMPLATE)
            st.plotly_chart(figd, use_container_width=True)

# -------------------------
# Clustering
# -------------------------
with tab_cluster:
    st.subheader("Clustering (PDF-aligned) — default 3 persona + penamaan")

    # Features for clustering
    exclude = {
        "target",
        "Penyaluran Kerja",
    }
    # drop high-cardinality text columns (like full motivasi narrative) by default
    high_text_cols = []
    for c in df_f.columns:
        if not pd.api.types.is_numeric_dtype(df_f[c]) and df_f[c].nunique(dropna=True) > 200:
            high_text_cols.append(c)

    suggested_exclude = exclude.union(set(high_text_cols))

    st.caption(
        "Catatan: kolom teks panjang (high cardinality) otomatis di-exclude biar clustering stabil & selaras PDF."
    )

    numeric_cols, categorical_cols = pick_numeric_and_categorical(df_f, suggested_exclude)

    # ensure engineered categoricals exist
    for c in ["Segmen_karir", "Motivasi_cluster"]:
        if c in df_f.columns and c not in categorical_cols and c not in suggested_exclude:
            categorical_cols.append(c)

    with st.expander("Advanced: pilih feature untuk clustering", expanded=False):
        use_num = st.multiselect(
            "Numeric features",
            options=sorted(numeric_cols),
            default=sorted(numeric_cols),
            key="clust_num_feats",
        )
        use_cat = st.multiselect(
            "Categorical features",
            options=sorted(categorical_cols),
            default=sorted(categorical_cols),
            key="clust_cat_feats",
        )

    k = st.slider("Jumlah cluster (k)", 2, 6, 3, key="clust_k")
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="clust_rs")

    run_clust = st.button("Run clustering", key="run_clustering_btn")

    @st.cache_resource(show_spinner=False)
    def build_cluster_pipeline(use_num_cols: tuple, use_cat_cols: tuple, k: int, random_state: int):
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, list(use_num_cols)),
                ("cat", cat_pipe, list(use_cat_cols)),
            ],
            remainder="drop",
        )
        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("svd", TruncatedSVD(n_components=2, random_state=random_state)),
                ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init="auto")),
            ]
        )
        return pipe

    def assign_persona_names(df_local: pd.DataFrame, cluster_col: str) -> dict:
        """
        Map cluster_id -> persona name with simple scoring rules.
        Ensures unique mapping by greedy assignment.
        """
        # cluster profiles
        prof = df_local.groupby(cluster_col).agg(
            avg_age=("Umur", "mean") if "Umur" in df_local.columns else ("engagement_score", "mean"),
            engagement=("engagement_score", "mean"),
            switch_rate=("Motivasi_cluster", lambda s: np.mean(s.astype(str) == "Switch career")) if "Motivasi_cluster" in df_local.columns else ("engagement_score", "mean"),
            student_rate=("Kategori Kesibukan", lambda s: np.mean(s.astype(str).str.lower().str.contains("mahasiswa|student|kuliah|pelajar", regex=True))) if "Kategori Kesibukan" in df_local.columns else ("engagement_score", "mean"),
        ).reset_index()

        persona_list = [
            "Fresh Graduate Explorer",
            "Working Professional Upskiller",
            "High Engagement Career Switcher",
        ]

        # score each cluster for each persona
        scores = {}
        for _, r in prof.iterrows():
            cid = int(r[cluster_col])
            scores[cid] = {}

            # Fresh grad: high student_rate, lower age (if exists)
            s_fg = 2.5 * float(r.get("student_rate", 0)) + (0.5 if float(r.get("avg_age", 30)) < 25 else 0.0)

            # Working pro: lower student rate, moderate engagement
            s_wp = 1.0 * (1.0 - float(r.get("student_rate", 0))) + 0.6 * float(r.get("engagement", 0)) / 4.0

            # Career switcher: high switch_rate + high engagement
            s_cs = 2.0 * float(r.get("switch_rate", 0)) + 0.8 * float(r.get("engagement", 0)) / 4.0

            scores[cid]["Fresh Graduate Explorer"] = s_fg
            scores[cid]["Working Professional Upskiller"] = s_wp
            scores[cid]["High Engagement Career Switcher"] = s_cs

        # greedy unique assignment
        mapping = {}
        remaining_personas = set(persona_list)
        remaining_clusters = set(scores.keys())

        for _ in range(min(len(remaining_clusters), len(remaining_personas))):
            best = None
            for cid in remaining_clusters:
                for p in remaining_personas:
                    val = scores[cid][p]
                    if best is None or val > best[0]:
                        best = (val, cid, p)
            if best is None:
                break
            _, cid, p = best
            mapping[cid] = p
            remaining_clusters.remove(cid)
            remaining_personas.remove(p)

        # fallback names if k != 3
        for cid in scores.keys():
            if cid not in mapping:
                mapping[cid] = f"Persona {cid}"

        return mapping

    if run_clust:
        # minimal rows for clustering
        feat_df = df_f.copy()

        # columns might not exist in filtered df
        use_num_cols = [c for c in use_num if c in feat_df.columns]
        use_cat_cols = [c for c in use_cat if c in feat_df.columns]

        # If user unselects everything, stop
        if len(use_num_cols) + len(use_cat_cols) == 0:
            st.error("Pilih minimal 1 feature untuk clustering.")
            st.stop()

        pipe = build_cluster_pipeline(tuple(use_num_cols), tuple(use_cat_cols), k=k, random_state=int(random_state))
        X = feat_df[use_num_cols + use_cat_cols].copy()

        with st.spinner("Fitting clustering pipeline..."):
            svd_xy = pipe.fit_transform(X)
            labels = pipe.named_steps["kmeans"].labels_

        feat_df = feat_df.copy()
        feat_df["cluster_id"] = labels
        feat_df["SVD1"] = svd_xy[:, 0]
        feat_df["SVD2"] = svd_xy[:, 1]

        # persona naming
        persona_map = assign_persona_names(feat_df, "cluster_id")
        feat_df["Persona"] = feat_df["cluster_id"].map(persona_map)

        # scatter
        fig = px.scatter(
            feat_df,
            x="SVD1",
            y="SVD2",
            color="Persona" if k == 3 else "cluster_id",
            hover_data=["Segmen_karir", "Motivasi_cluster", "Product", "Kategori"] if "Product" in feat_df.columns else ["Segmen_karir", "Motivasi_cluster"],
            title="Cluster visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)",
            template=PLOTLY_TEMPLATE,
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster profiling (untuk penamaan persona)")
        prof_cols = []
        for c in ["Segmen_karir", "Motivasi_cluster", "Kategori Kesibukan", "Level Pekerjaan", "Product", "Kategori"]:
            if c in feat_df.columns:
                prof_cols.append(c)

        # show top categories per cluster/persona
        for persona in sorted(feat_df["Persona"].unique()):
            st.markdown(f"### {persona}")
            sub = feat_df[feat_df["Persona"] == persona]
            st.write(f"Rows: {len(sub):,} | Avg engagement_score: {sub['engagement_score'].mean():.2f}")

            for c in prof_cols[:4]:
                vc = sub[c].astype(str).value_counts().head(10).reset_index()
                vc.columns = [c, "count"]
                figc = px.bar(vc, x="count", y=c, orientation="h", title=f"Top {c}", template=PLOTLY_TEMPLATE)
                st.plotly_chart(figc, use_container_width=True)

        # store in session for supervised option
        st.session_state["clustered_df"] = feat_df
        st.session_state["cluster_persona_map"] = persona_map

    else:
        st.info("Klik **Run clustering** untuk generate cluster & persona.")

# -------------------------
# Supervised Top-K
# -------------------------
with tab_sup:
    st.subheader("Supervised ranking — Top-K (robust, PDF-aligned)")

    # Allow using clustered_df if available
    use_cluster_feature = st.toggle("Use cluster_id as feature (if available)", value=True, key="sup_use_cluster")
    test_size_dummy = st.slider("Holdout fraction (info only; CV used)", 0.1, 0.4, 0.2, 0.05, key="sup_test_size")

    # Target counts
    y = df_f["target"].copy()
    y_known = y.dropna()

    if y_known.empty:
        st.error("Target kosong setelah filter. Coba longgarkan filter atau cek kolom target.")
        st.stop()

    vc = y_known.value_counts()
    pos = int(vc.get(1.0, 0))
    neg = int(vc.get(0.0, 0))

    st.write("Target counts (known):")
    st.code({"1 (positive)": pos, "0 (negative)": neg})

    if neg < 2:
        st.error(
            "Kelas minoritas terlalu kecil untuk supervised (negatives < 2). "
            "Ini biasanya karena filter terlalu ketat. Longgarkan filter dulu."
        )
        st.stop()

    # Feature selection
    base_exclude = {"target", "Penyaluran Kerja"}
    # Use clustered df if exists
    df_sup = df_f.copy()
    if use_cluster_feature and "clustered_df" in st.session_state:
        # align by index length: safest is to merge on same row order (since df_f is filtered, clustering df may differ)
        # We'll only use cluster_id if shapes match; otherwise skip.
        cdf = st.session_state["clustered_df"]
        if len(cdf) == len(df_sup) and "cluster_id" in cdf.columns:
            df_sup = df_sup.copy()
            df_sup["cluster_id"] = cdf["cluster_id"].values
        else:
            st.warning("cluster_id tidak dipakai (data clustering tidak match dengan filtered df).")

    # Remove huge text columns
    high_text_cols = []
    for c in df_sup.columns:
        if c in base_exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df_sup[c]) and df_sup[c].nunique(dropna=True) > 250:
            high_text_cols.append(c)

    exclude = base_exclude.union(set(high_text_cols))

    numeric_cols, categorical_cols = pick_numeric_and_categorical(df_sup, exclude)

    # Candidate feature cols shown to user (avoid too many)
    feature_candidates = sorted(numeric_cols + categorical_cols)

    default_features = [c for c in [
        "Umur", "Region", "Batch_num", "Batch_has_plus",
        "engagement_score", "Segmen_karir", "Motivasi_cluster",
        "Domain pendidikan", "Domain product", "Product", "Kategori",
        "Channel", "Level pendidikan", "Kategori Kesibukan",
        "Level Pekerjaan", "Kategori Pekerjaan",
    ] if c in feature_candidates]

    feat_cols = st.multiselect(
        "Feature cols",
        options=feature_candidates,
        default=default_features if default_features else feature_candidates[:15],
        key="sup_feat_cols",
    )

    if not feat_cols:
        st.error("Pilih minimal 1 feature untuk supervised.")
        st.stop()

    k_business = st.number_input(
        "Business capacity K (berapa peserta yang bisa diintervensi)",
        min_value=1,
        max_value=min(2000, len(y_known)),
        value=min(200, len(y_known)),
        step=10,
        key="sup_k_business",
    )

    run_sup = st.button("Run supervised ranking", key="sup_run_btn")

    @st.cache_resource(show_spinner=False)
    def build_supervised_pipeline(num_cols: tuple, cat_cols: tuple):
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, list(num_cols)),
                ("cat", cat_pipe, list(cat_cols)),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
        )
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        return pipe

    if run_sup:
        data = df_sup[feat_cols + ["target"]].copy()
        data = data.dropna(subset=["target"])
        X = data[feat_cols]
        y2 = data["target"].astype(int)

        # split cols into num/cat based on dtype in X
        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in feat_cols if c not in num_cols]

        pipe = build_supervised_pipeline(tuple(num_cols), tuple(cat_cols))

        # robust CV folds
        min_class = min(int((y2 == 0).sum()), int((y2 == 1).sum()))
        n_splits = int(min(5, min_class))
        if n_splits < 2:
            st.error("Data terlalu sedikit untuk CV (min class < 2). Longgarkan filter.")
            st.stop()

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        with st.spinner("Training (CV) & scoring..."):
            # out-of-fold predicted probabilities for class 1
            proba = cross_val_predict(pipe, X, y2, cv=cv, method="predict_proba")[:, 1]

        scored = data.copy()
        scored["p_positive"] = proba
        scored = scored.sort_values("p_positive", ascending=False).reset_index(drop=True)

        # Top-K metrics
        K = int(k_business)
        topk = scored.head(K)
        base_rate = y2.mean()
        topk_rate = topk["target"].mean()

        st.subheader("Ringkasan Top-K (intervensi)")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Base target rate", f"{base_rate*100:.1f}%")
        with m2:
            st.metric(f"Top-{K} target rate", f"{topk_rate*100:.1f}%")
        with m3:
            lift = (topk_rate / base_rate) if base_rate > 0 else np.nan
            st.metric("Lift", f"{lift:.2f}x" if np.isfinite(lift) else "-")

        # Precision@K / Recall@K curve
        st.subheader("Precision@K / Recall@K")
        ks = np.unique(np.clip(np.linspace(20, min(1000, len(scored)), 25).astype(int), 1, len(scored)))
        prec = []
        rec = []
        total_pos = int((scored["target"] == 1).sum())
        for kk in ks:
            tmp = scored.head(int(kk))
            tp = int((tmp["target"] == 1).sum())
            prec.append(tp / kk)
            rec.append(tp / total_pos if total_pos > 0 else 0.0)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=ks, y=prec, mode="lines+markers", name="Precision@K"))
        fig_pr.add_trace(go.Scatter(x=ks, y=rec, mode="lines+markers", name="Recall@K"))
        fig_pr.update_layout(
            title="Precision@K & Recall@K (CV-based ranking)",
            xaxis_title="K",
            yaxis_title="Score",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.subheader(f"Top-{K} rows (ranked)")
        show_cols = []
        for c in ["p_positive", "target", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region", "Batch_num", "engagement_score"]:
            if c in scored.columns:
                show_cols.append(c)

        st.dataframe(topk[show_cols].head(200), use_container_width=True)

        st.markdown("---")
        st.subheader("Breakdown Top-K vs Baseline")
        bd_candidates = [c for c in ["Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region"] if c in scored.columns]
        bd_col = st.selectbox("Pilih breakdown", options=bd_candidates, key="sup_breakdown")
        min_count_bd = st.slider("Min count per group (breakdown)", 1, 300, 20, key="sup_min_bd")

        def compare_topk_baseline(df_all, df_topk, col):
            base_tbl = make_group_table(df_all, col, "target", min_count=min_count_bd)
            top_tbl = make_group_table(df_topk, col, "target", min_count=min_count_bd)
            base_tbl = base_tbl.rename(columns={"target_rate_pct": "baseline_rate_pct", "Total": "baseline_total", "Positives": "baseline_pos"})
            top_tbl = top_tbl.rename(columns={"target_rate_pct": "topk_rate_pct", "Total": "topk_total", "Positives": "topk_pos"})
            merged = pd.merge(base_tbl, top_tbl, on="Group", how="outer").fillna(0)
            merged["lift"] = np.where(merged["baseline_rate_pct"] > 0, merged["topk_rate_pct"] / merged["baseline_rate_pct"], np.nan)
            return merged.sort_values(["lift", "topk_total"], ascending=[False, False])

        merged = compare_topk_baseline(scored, topk, bd_col)
        st.dataframe(merged.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Kenapa memilih Top-K segini?")
        st.markdown(
            f"""
- **K = {K}** merepresentasikan *kapasitas intervensi bisnis* (berapa peserta yang bisa di-follow-up).
- Model memberi skor peluang (CV-based) → kamu ambil **Top-K** untuk prioritas outreach.
- Kamu bisa lihat trade-off di kurva **Precision@K / Recall@K**:
  - K kecil → precision tinggi (lebih “tepat sasaran”)
  - K besar → recall naik (lebih banyak positif terjangkau)
            """.strip()
        )

    else:
        st.info("Klik **Run supervised ranking** untuk hasil Top-K, kurva, dan breakdown.")
