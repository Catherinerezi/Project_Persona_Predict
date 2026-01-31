# app.py (NO-PLOTLY VERSION)
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

# -------------------------
# Helpers
# -------------------------
def _safe_lower(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def make_target_from_penyaluran(x: object) -> float:
    t = _safe_lower(x)
    if not t or t in {"-", "nan", "none"}:
        return np.nan
    if "tidak" in t or "gak" in t or "ga " in t or t.startswith("ga"):
        return 0.0
    if "tertarik" in t or t == "ya" or t.startswith("ya "):
        return 1.0
    if "tersalur" in t or "placed" in t or "berhasil" in t:
        return 1.0
    return np.nan

def infer_motivasi_cluster(text: object) -> str:
    t = _safe_lower(text)
    if not t:
        return "Lainnya"
    if re.search(r"\bfreelance\b|\bsampingan\b|\bside\b", t):
        return "Freelance"
    if re.search(r"\bswitch\b|\bpindah\b|\bberalih\b|\bswitch career\b|\bcareer switch\b", t):
        return "Switch career"
    if re.search(r"\bcari kerja\b|\bjob\b|\bkerja\b|\bjob seeker\b|\bpenyaluran\b", t):
        return "Dapat kerja"
    if re.search(r"\bupgrade\b|\bimprove\b|\bmeningkatkan\b|\bportfolio\b|\bcv\b", t):
        return "Upgrade diri"
    if re.search(r"\bbelajar\b|\bmempelajari\b|\bmendalami\b|\bmengasah\b|\bdari nol\b|\bfrom scratch\b", t):
        return "Belajar skill"
    return "Lainnya"

def infer_segmen_karir(row: pd.Series) -> str:
    kesibukan = _safe_lower(row.get("Kategori Kesibukan", ""))
    level_pekerjaan = _safe_lower(row.get("Level Pekerjaan", ""))
    kategori_pekerjaan = _safe_lower(row.get("Kategori Pekerjaan", ""))
    motivasi = _safe_lower(row.get("Motivasi mengikuti bootcamp", ""))

    if any(k in kesibukan for k in ["mahasiswa", "student", "kuliah", "pelajar"]):
        return "Fresh Graduate Explorer"
    if any(k in kesibukan for k in ["job seeker", "unemployed", "belum bekerja", "nganggur"]):
        return "Fresh Graduate Explorer"
    if re.search(r"\bswitch\b|\bpindah\b|\bberalih\b|\bcareer\b", motivasi):
        return "High Engagement Career Switcher"
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

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Penyaluran Kerja" in out.columns:
        out["target"] = out["Penyaluran Kerja"].apply(make_target_from_penyaluran)
    else:
        out["target"] = np.nan

    if "Motivasi mengikuti bootcamp" in out.columns:
        out["Motivasi_cluster"] = out["Motivasi mengikuti bootcamp"].apply(infer_motivasi_cluster)
    else:
        out["Motivasi_cluster"] = "Lainnya"

    out["Segmen_karir"] = out.apply(infer_segmen_karir, axis=1)

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

    if "Tanggal Gabungan" in out.columns:
        out["Tanggal Gabungan"] = pd.to_datetime(out["Tanggal Gabungan"], errors="coerce", dayfirst=True)
        out["Month"] = out["Tanggal Gabungan"].dt.to_period("M").astype(str)
    else:
        out["Month"] = "Unknown"

    return out

@st.cache_data(show_spinner=False)
def load_csv_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, vals in filters.items():
        if col not in out.columns:
            continue
        if not vals:
            continue
        out = out[out[col].astype(str).isin([str(v) for v in vals])]
    return out

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

def barh_with_labels(df_tbl: pd.DataFrame, title: str, top_n: int = 30):
    g = df_tbl.head(top_n).copy()
    if g.empty:
        st.warning("Tidak ada group yang memenuhi syarat.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(g))))
    y = np.arange(len(g))
    ax.barh(y, g["target_rate_pct"].values)
    ax.set_yticks(y)
    ax.set_yticklabels(g["Group"].astype(str).values)
    ax.invert_yaxis()
    ax.set_xlabel("Target rate (%)")
    ax.set_title(title)

    for i, v in enumerate(g["target_rate_pct"].values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

    st.pyplot(fig, clear_figure=True)

def scatter_svd(df_xy: pd.DataFrame, color_col: str, title: str):
    # simple matplotlib scatter by group
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = df_xy[color_col].astype(str).unique().tolist()
    for g in groups:
        sub = df_xy[df_xy[color_col].astype(str) == g]
        ax.scatter(sub["SVD1"], sub["SVD2"], s=10, alpha=0.6, label=str(g))
    ax.set_xlabel("SVD1")
    ax.set_ylabel("SVD2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig, clear_figure=True)

def pick_numeric_and_categorical(df: pd.DataFrame, exclude: set) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in exclude]
    numeric, categorical = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical

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
    df_raw = load_csv_upload(up)
else:
    default_path = "raw_data/raw_data.csv"
    path = st.sidebar.text_input("Path", value=default_path, key="repo_path")
    try:
        df_raw = load_csv_path(path)
    except Exception as e:
        st.error(f"Can't read path: {e}")
        st.caption("Tips: path repo biasanya `raw_data/raw_data.csv`.")
        st.stop()

df = ensure_features(df_raw)

# =========================
# Global filters
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Global filters")

candidate_filters = []
for c in df.columns:
    if c in {"target"}:
        continue
    if (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique(dropna=True) <= 60:
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
    opts = sorted(df[c].dropna().astype(str).unique().tolist())
    filters[c] = st.sidebar.multiselect(
        f"{c}",
        options=opts,
        default=[],
        key=f"filter_{c}",
    )

df_f = apply_filters(df, filters)

# =========================
# Header / Overview
# =========================
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")
st.caption(
    "Tujuan: memahami pola peserta, segmentasi persona yang akurat, dan model prediktif peluang penyaluran kerja "
    "untuk mendukung keputusan bisnis (akuisisi, desain program, intervensi)."
)

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

tabs = st.tabs(["📌 Overview", "🔎 EDA", "🧩 Clustering", "🎯 Supervised (Top-K)"])

# =========================
# Overview
# =========================
with tabs[0]:
    st.subheader("Data preview")
    st.dataframe(df_f.head(50), use_container_width=True)

    st.subheader("Target definition")
    st.markdown(
        """
Default target menggunakan kolom **Penyaluran Kerja**:
- **1** = mengandung *“tertarik”* (dan tidak mengandung *“tidak”*)
- **0** = mengandung *“tidak”* / *“tidak tertarik”*
- selain itu = **NaN** (unknown)

Supervised dibuat robust pakai **Stratified CV** (bukan split yang gampang gagal di data imbalanced).
        """.strip()
    )

# =========================
# EDA
# =========================
with tabs[1]:
    st.subheader("EDA — fleksibel (filter + breakdown apa pun)")

    breakdown_candidates = sorted(list(set(candidate_filters + ["Segmen_karir", "Motivasi_cluster"])))
    if not breakdown_candidates:
        st.warning("Tidak ada kolom breakdown yang cocok (categorical <=60 unique).")
        st.stop()

    default_breakdown = "Segmen_karir" if "Segmen_karir" in df_f.columns else breakdown_candidates[0]
    breakdown_col = st.selectbox(
        "Pilih breakdown column",
        options=breakdown_candidates,
        index=breakdown_candidates.index(default_breakdown) if default_breakdown in breakdown_candidates else 0,
        key="eda_breakdown",
    )
    min_count = st.slider("Min count per group", 1, 300, 30, key="eda_min_count")
    top_n = st.slider("Show top-N groups", 5, 60, 30, key="eda_topn")

    group_tbl = make_group_table(df_f, breakdown_col, "target", min_count=min_count)
    barh_with_labels(group_tbl, f"Target rate by {breakdown_col}", top_n=top_n)
    st.dataframe(group_tbl.head(top_n), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribusi target")
    tmp = df_f["target"].dropna()
    if tmp.empty:
        st.info("Target kosong setelah filter.")
    else:
        dist = tmp.value_counts().rename_axis("target").reset_index(name="count")
        dist["target"] = dist["target"].map({0.0: "0 (Tidak)", 1.0: "1 (Tertarik)"})
        st.bar_chart(dist.set_index("target")["count"])

# =========================
# Clustering
# =========================
with tabs[2]:
    st.subheader("Clustering (PDF-aligned) — default 3 persona + penamaan")

    exclude = {"target", "Penyaluran Kerja"}
    high_text_cols = []
    for c in df_f.columns:
        if c in exclude:
            continue
        if (not pd.api.types.is_numeric_dtype(df_f[c])) and df_f[c].nunique(dropna=True) > 250:
            high_text_cols.append(c)
    exclude = exclude.union(set(high_text_cols))

    numeric_cols, categorical_cols = pick_numeric_and_categorical(df_f, exclude)

    for c in ["Segmen_karir", "Motivasi_cluster"]:
        if c in df_f.columns and c not in categorical_cols and c not in exclude:
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
        prof = df_local.groupby(cluster_col).agg(
            engagement=("engagement_score", "mean"),
            switch_rate=("Motivasi_cluster", lambda s: np.mean(s.astype(str) == "Switch career")) if "Motivasi_cluster" in df_local.columns else ("engagement_score", "mean"),
            student_rate=("Kategori Kesibukan", lambda s: np.mean(s.astype(str).str.lower().str.contains("mahasiswa|student|kuliah|pelajar", regex=True))) if "Kategori Kesibukan" in df_local.columns else ("engagement_score", "mean"),
        ).reset_index()

        personas = ["Fresh Graduate Explorer", "Working Professional Upskiller", "High Engagement Career Switcher"]

        scores = {}
        for _, r in prof.iterrows():
            cid = int(r[cluster_col])
            scores[cid] = {}
            s_fg = 2.5 * float(r.get("student_rate", 0))
            s_wp = 1.0 * (1.0 - float(r.get("student_rate", 0))) + 0.6 * float(r.get("engagement", 0)) / 4.0
            s_cs = 2.0 * float(r.get("switch_rate", 0)) + 0.8 * float(r.get("engagement", 0)) / 4.0
            scores[cid]["Fresh Graduate Explorer"] = s_fg
            scores[cid]["Working Professional Upskiller"] = s_wp
            scores[cid]["High Engagement Career Switcher"] = s_cs

        mapping = {}
        remaining_p = set(personas)
        remaining_c = set(scores.keys())
        for _ in range(min(len(remaining_c), len(remaining_p))):
            best = None
            for cid in remaining_c:
                for p in remaining_p:
                    val = scores[cid][p]
                    if best is None or val > best[0]:
                        best = (val, cid, p)
            if best is None:
                break
            _, cid, p = best
            mapping[cid] = p
            remaining_c.remove(cid)
            remaining_p.remove(p)

        for cid in scores.keys():
            if cid not in mapping:
                mapping[cid] = f"Persona {cid}"
        return mapping

    if run_clust:
        if len(use_num) + len(use_cat) == 0:
            st.error("Pilih minimal 1 feature untuk clustering.")
            st.stop()

        pipe = build_cluster_pipeline(tuple(use_num), tuple(use_cat), k=k, random_state=int(random_state))
        X = df_f[use_num + use_cat].copy()

        with st.spinner("Fitting clustering pipeline..."):
            svd_xy = pipe.fit_transform(X)
            labels = pipe.named_steps["kmeans"].labels_

        feat_df = df_f.copy()
        feat_df["cluster_id"] = labels
        feat_df["SVD1"] = svd_xy[:, 0]
        feat_df["SVD2"] = svd_xy[:, 1]

        persona_map = assign_persona_names(feat_df, "cluster_id")
        feat_df["Persona"] = feat_df["cluster_id"].map(persona_map)

        scatter_svd(
            feat_df,
            color_col="Persona" if k == 3 else "cluster_id",
            title="Cluster visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)",
        )

        st.subheader("Cluster profiling (ringkas)")
        st.dataframe(
            feat_df.groupby(["Persona"]).agg(
                rows=("Persona", "count"),
                avg_engagement=("engagement_score", "mean"),
            ).reset_index(),
            use_container_width=True,
        )

        st.session_state["clustered_df"] = feat_df

    else:
        st.info("Klik **Run clustering** untuk generate cluster & persona.")

# =========================
# Supervised Top-K
# =========================
with tabs[3]:
    st.subheader("Supervised ranking — Top-K (robust, PDF-aligned)")

    use_cluster_feature = st.toggle("Use cluster_id as feature (if available)", value=True, key="sup_use_cluster")

    y = df_f["target"].copy()
    y_known = y.dropna()

    if y_known.empty:
        st.error("Target kosong setelah filter. Longgarkan filter atau cek kolom target.")
        st.stop()

    vc = y_known.value_counts()
    pos = int(vc.get(1.0, 0))
    neg = int(vc.get(0.0, 0))
    st.write("Target counts (known):")
    st.code({"1 (positive)": pos, "0 (negative)": neg})

    if neg < 2:
        st.error("Kelas minoritas terlalu kecil untuk supervised (negatives < 2). Longgarkan filter dulu.")
        st.stop()

    df_sup = df_f.copy()
    if use_cluster_feature and "clustered_df" in st.session_state:
        cdf = st.session_state["clustered_df"]
        if len(cdf) == len(df_sup) and "cluster_id" in cdf.columns:
            df_sup = df_sup.copy()
            df_sup["cluster_id"] = cdf["cluster_id"].values
        else:
            st.warning("cluster_id tidak dipakai (data clustering tidak match dengan filtered df).")

    base_exclude = {"target", "Penyaluran Kerja"}
    high_text_cols = []
    for c in df_sup.columns:
        if c in base_exclude:
            continue
        if (not pd.api.types.is_numeric_dtype(df_sup[c])) and df_sup[c].nunique(dropna=True) > 250:
            high_text_cols.append(c)
    exclude = base_exclude.union(set(high_text_cols))

    numeric_cols, categorical_cols = pick_numeric_and_categorical(df_sup, exclude)
    feature_candidates = sorted(numeric_cols + categorical_cols)

    default_features = [c for c in [
        "Umur", "Region", "Batch_num",
        "engagement_score", "Segmen_karir", "Motivasi_cluster",
        "Domain pendidikan", "Domain product", "Product", "Kategori",
        "Channel", "Level pendidikan", "Kategori Kesibukan",
        "Level Pekerjaan", "Kategori Pekerjaan",
        "cluster_id",
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

    K = st.number_input(
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
        return Pipeline(steps=[("pre", pre), ("clf", clf)])

    if run_sup:
        data = df_sup[feat_cols + ["target"]].copy()
        data = data.dropna(subset=["target"])
        X = data[feat_cols]
        y2 = data["target"].astype(int)

        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in feat_cols if c not in num_cols]

        pipe = build_supervised_pipeline(tuple(num_cols), tuple(cat_cols))

        min_class = min(int((y2 == 0).sum()), int((y2 == 1).sum()))
        n_splits = int(min(5, min_class))
        if n_splits < 2:
            st.error("Data terlalu sedikit untuk CV (min class < 2). Longgarkan filter.")
            st.stop()

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        with st.spinner("Training (CV) & scoring..."):
            proba = cross_val_predict(pipe, X, y2, cv=cv, method="predict_proba")[:, 1]

        scored = data.copy()
        scored["p_positive"] = proba
        scored = scored.sort_values("p_positive", ascending=False).reset_index(drop=True)

        topk = scored.head(int(K))
        base_rate = y2.mean()
        topk_rate = topk["target"].mean()

        st.subheader("Ringkasan Top-K (intervensi)")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Base target rate", f"{base_rate*100:.1f}%")
        with m2:
            st.metric(f"Top-{int(K)} target rate", f"{topk_rate*100:.1f}%")
        with m3:
            lift = (topk_rate / base_rate) if base_rate > 0 else np.nan
            st.metric("Lift", f"{lift:.2f}x" if np.isfinite(lift) else "-")

        st.subheader("Precision@K / Recall@K")
        ks = np.unique(np.clip(np.linspace(20, min(1000, len(scored)), 25).astype(int), 1, len(scored)))
        prec, rec = [], []
        total_pos = int((scored["target"] == 1).sum())

        for kk in ks:
            tmp = scored.head(int(kk))
            tp = int((tmp["target"] == 1).sum())
            prec.append(tp / kk)
            rec.append(tp / total_pos if total_pos > 0 else 0.0)

        pr_df = pd.DataFrame({"K": ks, "Precision@K": prec, "Recall@K": rec}).set_index("K")
        st.line_chart(pr_df)

        st.subheader(f"Top-{int(K)} rows (ranked)")
        show_cols = [c for c in ["p_positive", "target", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region", "Batch_num", "engagement_score"] if c in scored.columns]
        st.dataframe(topk[show_cols].head(200), use_container_width=True)

        st.markdown("---")
        st.subheader("Breakdown Top-K vs Baseline")
        bd_candidates = [c for c in ["Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Region"] if c in scored.columns]
        if bd_candidates:
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
- **K = {int(K)}** = kapasitas intervensi (berapa peserta bisa di-follow-up).
- Model memberi skor peluang → kamu ambil **Top-K** untuk prioritas outreach.
- Kurva **Precision@K / Recall@K** nunjukkin trade-off:
  - K kecil → precision tinggi (lebih tepat sasaran)
  - K besar → recall naik (lebih banyak positif terjangkau)
            """.strip()
        )

    else:
        st.info("Klik **Run supervised ranking** untuk hasil Top-K & kurva.")
