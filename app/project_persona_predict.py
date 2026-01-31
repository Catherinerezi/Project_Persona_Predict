# app.py
# Persona Segmentation & Placement Prediction — Single-file Streamlit app (PDF-aligned, no Plotly/Altair)

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Persona Segmentation & Placement Prediction (PDF-aligned)",
    layout="wide"
)


# -------------------------
# Helpers
# -------------------------
def _norm_col(c: str) -> str:
    c = str(c).strip()
    c = c.replace("\n", " ").replace("\t", " ")
    c = " ".join(c.split())
    return c


def make_target_from_penyaluran(v):
    """
    Binarize target dari kolom 'Penyaluran Kerja' (raw data).
    - 1: Tertarik
    - 0: Tidak tertarik / belum / masih / dipertimbangkan
    - NaN: unknown / '-' / kosong
    """
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ["", "-", "nan", "none"]:
        return np.nan

    # negatives
    if "tidak" in s:
        return 0
    if "belum" in s or "masih" in s or "dipertimbangkan" in s or "perlu" in s:
        return 0

    # positives
    if "tertarik" in s:
        return 1

    # fallback unknown
    return np.nan


def safe_value_counts(series: pd.Series):
    vc = series.value_counts(dropna=False)
    out = vc.rename_axis("value").reset_index(name="count")
    out["value"] = out["value"].astype(str)
    return out


def build_preprocess(cat_cols, num_cols):
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    prep = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return prep


def plot_barh(values, labels, title, xlabel, annotate_fmt="{:.1f}%"):
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.35 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    # annotate
    for i, v in enumerate(values):
        ax.text(v + (max(values) * 0.01 if max(values) > 0 else 0.01), i, annotate_fmt.format(v),
                va="center", fontsize=10)

    fig.tight_layout()
    return fig


def plot_scatter_2d(x, y, labels, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    unique = pd.unique(labels)
    for u in unique:
        mask = (labels == u)
        ax.scatter(x[mask], y[mask], s=14, alpha=0.75, label=str(u))
    ax.set_title(title)
    ax.set_xlabel("SVD-1")
    ax.set_ylabel("SVD-2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def precision_recall_at_k(y_true, y_score, ks):
    # sort by score desc
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    total_pos = (y_sorted == 1).sum()
    out = []
    for k in ks:
        k = int(k)
        k = max(1, min(k, len(y_sorted)))
        topk = y_sorted[:k]
        tp = (topk == 1).sum()
        prec = tp / k
        rec = tp / total_pos if total_pos > 0 else np.nan
        out.append((k, prec, rec))
    return pd.DataFrame(out, columns=["k", "precision_at_k", "recall_at_k"])


def group_lift_table(df_eval, group_col, k):
    """
    df_eval harus punya kolom:
      - y_true (0/1)
      - y_score (prob)
      - group_col (kategori)
    """
    d = df_eval.copy()
    d = d.dropna(subset=[group_col])

    baseline = d["y_true"].mean()
    if pd.isna(baseline) or baseline == 0:
        baseline = np.nan

    # global top-k threshold
    k = int(max(1, min(k, len(d))))
    thr = np.sort(d["y_score"].values)[-k]  # score cutoff (approx)
    d["is_topk"] = d["y_score"] >= thr

    g = d.groupby(group_col, dropna=False).agg(
        baseline_total=("y_true", "size"),
        baseline_pos=("y_true", "sum"),
        baseline_rate=("y_true", "mean"),
        topk_total=("is_topk", "sum"),
        topk_pos=("y_true", lambda x: int(((d.loc[x.index, "is_topk"]) & (x == 1)).sum())),
    ).reset_index()

    g["topk_rate"] = np.where(g["topk_total"] > 0, g["topk_pos"] / g["topk_total"], np.nan)
    g["lift"] = g["topk_rate"] / g["baseline_rate"].replace({0: np.nan})

    # sort by lift desc, then baseline_total
    g = g.sort_values(["lift", "baseline_total"], ascending=[False, False])
    return g


# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    if hasattr(file_or_path, "read"):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # standardize key columns (keep original names, add snake-ish aliases)
    # NOTE: match kolom di raw_data.csv yang kamu pakai
    if "Motivasi Cluster" in d.columns:
        d["Motivasi_cluster"] = d["Motivasi Cluster"]
    if "Segmentasi Karir" in d.columns:
        d["Segmen_karir"] = d["Segmentasi Karir"]
    if "Domain" in d.columns:
        d["Domain_pendidikan"] = d["Domain"]
    if "Sumber" in d.columns:
        d["Channel"] = d["Sumber"]

    # month from "Tanggal Gabungan"
    if "Tanggal Gabungan" in d.columns:
        dt = pd.to_datetime(d["Tanggal Gabungan"], errors="coerce", dayfirst=True)
        d["Month"] = dt.dt.to_period("M").astype(str)

    # binary target for supervised
    if "Penyaluran Kerja" in d.columns:
        d["target"] = d["Penyaluran Kerja"].apply(make_target_from_penyaluran)

    return d


# -------------------------
# Sidebar: data source
# -------------------------
st.sidebar.header("Data source")

source_mode = st.sidebar.radio(
    "Choose",
    options=["Upload file (CSV)", "Repo/local path"],
    index=0,
    key="data_source_mode"
)

uploaded = None
path_input = None

if source_mode == "Upload file (CSV)":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
else:
    path_input = st.sidebar.text_input(
        "Path",
        value="raw_data/raw_data.csv",
        help="Contoh (deploy Streamlit Cloud): raw_data/raw_data.csv",
        key="path_input"
    )

df_raw = None
load_err = None
try:
    if uploaded is not None:
        df_raw = load_csv(uploaded)
    elif path_input:
        df_raw = load_csv(path_input)
except Exception as e:
    load_err = str(e)

if df_raw is None:
    st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")
    if load_err:
        st.error(f"Can't read data: {load_err}")
    st.info("Upload CSV dulu (sidebar) atau isi path repo yang benar.")
    st.stop()

df = add_derived_columns(df_raw)

# -------------------------
# Sidebar: Global filters
# -------------------------
st.sidebar.header("Global filters")

filter_candidates = [
    c for c in [
        "Product", "Kategori", "Month", "Channel",
        "Segmen_karir", "Motivasi_cluster", "Domain_pendidikan",
        "Region", "Domisili", "Provinsi", "Negara",
        "Program", "Level pendidikan", "Kategori Kesibukan",
        "Level Pekerjaan", "Kategori Pekerjaan",
        "Status switcher", "Engagement", "Community flag", "Event flag",
    ] if c in df.columns
]

selected_filter_cols = st.sidebar.multiselect(
    "Pilih kolom untuk filter",
    options=filter_candidates,
    default=[c for c in ["Product", "Kategori", "Month", "Segmen_karir"] if c in filter_candidates],
    key="filter_cols_multi"
)

df_f = df.copy()
for col in selected_filter_cols:
    # build options
    opts = pd.Series(df_f[col].dropna().unique()).astype(str).sort_values().tolist()
    if len(opts) == 0:
        continue
    chosen = st.sidebar.multiselect(
        f"{col}",
        options=opts,
        default=[],
        key=f"filter_{col}"  # IMPORTANT: unique key
    )
    if chosen:
        df_f = df_f[df_f[col].astype(str).isin(chosen)]

# -------------------------
# Header + KPI
# -------------------------
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

left, right = st.columns([1.2, 1])
with left:
    st.markdown(
        """
**Tujuan proyek (selaras dengan PDF/vertopal):**
- Memahami pola peserta Dibimbing  
- Membangun segmentasi peserta yang akurat (persona)  
- Mengembangkan model prediktif peluang penyaluran kerja untuk keputusan bisnis (akuisisi, desain program, intervensi)
"""
    )

with right:
    n_rows = len(df_f)
    n_cols = df_f.shape[1]
    pos_rate = df_f["target"].mean() if "target" in df_f.columns else np.nan
    a, b, c = st.columns(3)
    a.metric("Rows", f"{n_rows:,}")
    b.metric("Columns", f"{n_cols:,}")
    c.metric("Target mean", "-" if pd.isna(pos_rate) else f"{pos_rate*100:.1f}%")

st.divider()

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_eda, tab_cluster, tab_supervised = st.tabs(
    ["Overview", "EDA (Target-driven)", "Clustering (Persona)", "Supervised (Top-K Ranking)"]
)

# -------------------------
# OVERVIEW
# -------------------------
with tab_overview:
    st.subheader("Quick dataset check")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Contoh 10 baris (setelah filter):")
        st.dataframe(df_f.head(10), use_container_width=True)
    with c2:
        if "target" in df_f.columns:
            st.write("Distribusi target (dari 'Penyaluran Kerja'):")
            st.dataframe(safe_value_counts(df_f["target"]), use_container_width=True)
        else:
            st.warning("Kolom target belum terbuat (butuh kolom 'Penyaluran Kerja').")

    st.info(
        "Catatan penting: untuk supervised, data yang target-nya NaN akan di-drop (khusus supervised). "
        "EDA & clustering tetap bisa jalan di seluruh data."
    )

# -------------------------
# EDA (TARGET-DRIVEN)
# -------------------------
with tab_eda:
    st.subheader("Target rate dashboard (bisa pilih breakdown apa saja)")

    if "target" not in df_f.columns:
        st.error("Tidak ada kolom target. Pastikan CSV punya kolom 'Penyaluran Kerja'.")
        st.stop()

    # breakdown candidates for target-rate
    breakdown_cols = [c for c in [
        "Product", "Kategori", "Channel", "Month",
        "Segmen_karir", "Motivasi_cluster", "Domain_pendidikan",
        "Region", "Domisili", "Provinsi", "Negara",
        "Program", "Level pendidikan", "Kategori Kesibukan",
        "Level Pekerjaan", "Kategori Pekerjaan",
        "Status switcher",
    ] if c in df_f.columns]

    colA, colB, colC = st.columns([1.2, 0.8, 0.8])
    with colA:
        breakdown = st.selectbox("Pilih breakdown", options=breakdown_cols, index=0, key="eda_breakdown")
    with colB:
        show_top_n = st.slider("Show top-N groups", 5, 60, 30, key="eda_topn")
    with colC:
        min_count = st.slider("Min count per group", 1, 200, 20, key="eda_mincount")

    d_eda = df_f.dropna(subset=["target"]).copy()
    if d_eda.empty:
        st.warning("Semua target NaN setelah filter. Longgarkan filter atau cek mapping target.")
        st.stop()

    g = d_eda.groupby(breakdown, dropna=False).agg(
        total=("target", "size"),
        positives=("target", "sum"),
    ).reset_index()

    g["target_rate_pct"] = np.where(g["total"] > 0, 100 * g["positives"] / g["total"], np.nan)
    g = g[g["total"] >= min_count].copy()
    g = g.sort_values("target_rate_pct", ascending=False).head(int(show_top_n))

    if g.empty:
        st.warning("Tidak ada group yang memenuhi min_count setelah filter.")
        st.stop()

    fig = plot_barh(
        values=g["target_rate_pct"].values,
        labels=g[breakdown].astype(str).values,
        title=f"Target rate by {breakdown}",
        xlabel="Target rate (%)",
        annotate_fmt="{:.1f}%"
    )
    st.pyplot(fig, use_container_width=True)

    st.write("Tabel ringkas:")
    st.dataframe(g.rename(columns={breakdown: "group"}), use_container_width=True)

    st.subheader("Distribusi target")
    vc = d_eda["target"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(vc.index.astype(str), vc.values)
    ax2.set_title("Distribusi target (0/1)")
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Count")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# -------------------------
# CLUSTERING (PERSONA)
# -------------------------
with tab_cluster:
    st.subheader("Persona clustering (3 cluster, PDF-aligned)")

    # columns based on the PDF notebook
    cat_cols = [c for c in [
        "Region", "Batch", "Sumber", "Domain", "Program", "Product", "Kategori", "Channel",
        "Level pendidikan", "Kategori Kesibukan", "Level Pekerjaan", "Kategori Pekerjaan",
        "Domisili", "Provinsi", "Negara", "Segmentasi Karir", "Motivasi Cluster", "Status switcher",
        "Community flag", "Event flag",
    ] if c in df_f.columns]

    num_cols = [c for c in [
        "Engagement", "Umur"
    ] if c in df_f.columns]

    if len(cat_cols) + len(num_cols) == 0:
        st.error("Tidak menemukan kolom fitur untuk clustering. Cek nama kolom di CSV.")
        st.stop()

    st.write("Fitur yang dipakai (sesuai PDF):")
    st.code(f"Categorical: {cat_cols}\nNumeric: {num_cols}")

    run_cluster = st.button("Run clustering", type="primary", key="btn_run_cluster")

    if run_cluster:
        with st.spinner("Running clustering..."):
            X_df = df_f[cat_cols + num_cols].copy()

            prep = build_preprocess(cat_cols, num_cols)

            # PDF aligned: MiniBatchKMeans + TruncatedSVD 2D
            # IMPORTANT: random_state dikunci (tidak ditaruh di UI) supaya konsisten seperti PDF.
            k = 3
            cluster_model = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                batch_size=512
            )

            pipe = Pipeline(steps=[
                ("prep", prep),
                ("cluster", cluster_model)
            ])

            pipe.fit(X_df)
            cluster_id = pipe.named_steps["cluster"].labels_

            # persona mapping from PDF
            persona_map = {
                0: "High Engagement Career Switcher",
                1: "Fresh Graduate Explorer",
                2: "Working Professional Upskiller"
            }

            out = df_f.copy()
            out["cluster_id"] = cluster_id
            out["cluster_name"] = out["cluster_id"].map(persona_map).fillna(
                "Cluster " + (out["cluster_id"] + 1).astype(str)
            )

            # 2D reduction (fit on transformed feature space)
            X_trans = pipe.named_steps["prep"].transform(X_df)
            svd = TruncatedSVD(n_components=2, random_state=42)
            X_2d = svd.fit_transform(X_trans)
            out["svd1"] = X_2d[:, 0]
            out["svd2"] = X_2d[:, 1]

            # store in session_state for supervised tab
            st.session_state["cluster_out"] = out
            st.session_state["cluster_pipe"] = pipe

        st.success("Clustering selesai. (3 persona sesuai PDF)")

    if "cluster_out" in st.session_state:
        out = st.session_state["cluster_out"]

        st.subheader("Cluster visualization (TruncatedSVD 2D) — Persona Named")
        fig_sc = plot_scatter_2d(
            x=out["svd1"].values,
            y=out["svd2"].values,
            labels=out["cluster_name"].astype(str).values,
            title="Cluster Visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)"
        )
        st.pyplot(fig_sc, use_container_width=True)

        st.subheader("Cluster profiling (ringkas)")
        prof_cols = [c for c in ["cluster_name", "target", "Engagement", "Umur", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel"] if c in out.columns]
        base = out[prof_cols].copy()

        grp = base.groupby("cluster_name", dropna=False).agg(
            total=("cluster_name", "size"),
            target_rate=("target", "mean"),
            avg_engagement=("Engagement", "mean") if "Engagement" in base.columns else ("cluster_name", "size"),
            avg_umur=("Umur", "mean") if "Umur" in base.columns else ("cluster_name", "size"),
        ).reset_index()

        grp["target_rate_pct"] = grp["target_rate"] * 100
        st.dataframe(grp.sort_values("total", ascending=False), use_container_width=True)

        st.caption("Catatan: Profiling detail bisa kamu tambah (mis. top kategori/program per persona) kalau dibutuhkan.")

    else:
        st.info("Klik **Run clustering** dulu untuk menghasilkan persona & visualisasi.")

# -------------------------
# SUPERVISED (TOP-K)
# -------------------------
with tab_supervised:
    st.subheader("Supervised ranking — Top-K (PDF-aligned)")

    if "target" not in df_f.columns:
        st.error("Tidak ada kolom target. Pastikan CSV punya kolom 'Penyaluran Kerja'.")
        st.stop()

    # use clustering result if available (optional)
    if "cluster_out" in st.session_state:
        df_sup_base = st.session_state["cluster_out"].copy()
    else:
        df_sup_base = df_f.copy()

    # prepare supervised dataset (drop unknown target)
    df_sup = df_sup_base.dropna(subset=["target"]).copy()
    df_sup["target"] = df_sup["target"].astype(int)

    # feature columns similar to PDF (plus optional cluster_id)
    feat_candidates = [c for c in [
        "Umur", "Region", "Batch", "Sumber", "Domain_pendidikan", "Program", "Product", "Kategori", "Channel",
        "Motivasi_cluster", "Segmen_karir", "Status switcher", "Community flag", "Event flag", "Engagement",
        "Month", "Level pendidikan", "Kategori Kesibukan", "Level Pekerjaan", "Kategori Pekerjaan",
        "Domisili", "Provinsi", "Negara",
        "cluster_id", "cluster_name"
    ] if c in df_sup.columns]

    if len(feat_candidates) == 0:
        st.error("Tidak ada kolom fitur untuk supervised. Cek nama kolom di CSV.")
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        k_capacity = st.slider("Business capacity K (berapa peserta bisa diintervensi)", 20, 800, 200, step=10, key="sup_k")
    with col2:
        test_size = st.slider("test_size (holdout)", 0.1, 0.4, 0.2, step=0.05, key="sup_test_size")
    with col3:
        use_cluster_id = st.checkbox("Use cluster_id as feature (jika ada)", value=True, key="sup_use_cluster")

    if not use_cluster_id:
        feat_candidates = [c for c in feat_candidates if c not in ["cluster_id", "cluster_name"]]

    st.write("Fitur supervised dipakai:")
    st.code(", ".join(feat_candidates))

    # check class counts
    vc = df_sup["target"].value_counts()
    st.write("Target counts (0/1):", dict(vc.to_dict()))

    # safety checks
    if vc.min() < 10:
        st.warning(
            "Class minoritas terlalu kecil untuk split & evaluasi yang stabil. "
            "Coba longgarkan filter atau pastikan mapping target benar."
        )

    run_sup = st.button("Run supervised ranking", type="primary", key="btn_run_supervised")

    if run_sup:
        with st.spinner("Training supervised model + evaluating Top-K..."):
            X = df_sup[feat_candidates].copy()
            y = df_sup["target"].values

            # split stratified if possible
            can_strat = (len(np.unique(y)) == 2) and (min(vc.values) >= 2)
            if can_strat:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=float(test_size),
                    random_state=42,
                    stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y

            # split columns by type
            cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"]
            num_cols = [c for c in X.columns if c not in cat_cols]

            prep = build_preprocess(cat_cols, num_cols)

            clf = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=None
            )

            sup_pipe = Pipeline(steps=[
                ("prep", prep),
                ("clf", clf)
            ])

            sup_pipe.fit(X_train, y_train)

            # scores on test
            y_score = sup_pipe.predict_proba(X_test)[:, 1]

            # Top-K metrics curve
            ks = np.unique(np.clip(np.linspace(20, min(len(y_score), 800), 15).astype(int), 1, len(y_score)))
            pr_df = precision_recall_at_k(y_test, y_score, ks)

            # plot curves
            fig_pr, ax_pr = plt.subplots(figsize=(8, 4))
            ax_pr.plot(pr_df["k"], pr_df["precision_at_k"], marker="o")
            ax_pr.set_title("Precision@K curve")
            ax_pr.set_xlabel("K")
            ax_pr.set_ylabel("Precision@K")
            fig_pr.tight_layout()

            fig_rc, ax_rc = plt.subplots(figsize=(8, 4))
            ax_rc.plot(pr_df["k"], pr_df["recall_at_k"], marker="o")
            ax_rc.set_title("Recall@K curve")
            ax_rc.set_xlabel("K")
            ax_rc.set_ylabel("Recall@K")
            fig_rc.tight_layout()

            st.session_state["sup_pipe"] = sup_pipe
            st.session_state["sup_eval"] = {
                "X_test": X_test,
                "y_test": y_test,
                "y_score": y_score,
                "pr_df": pr_df,
                "k_capacity": int(k_capacity)
            }
            st.session_state["sup_fig_pr"] = fig_pr
            st.session_state["sup_fig_rc"] = fig_rc

        st.success("Supervised selesai. Scroll untuk dashboard evaluasi & Top-K output.")

    if "sup_eval" in st.session_state:
        ev = st.session_state["sup_eval"]
        k_capacity = int(ev["k_capacity"])
        X_test = ev["X_test"]
        y_test = ev["y_test"]
        y_score = ev["y_score"]
        pr_df = ev["pr_df"]

        st.subheader("Kenapa memilih Top-K segini?")
        st.markdown(
            f"""
- **K = {k_capacity}** = kapasitas intervensi (berapa peserta yang bisa di-follow-up).
- Model memberi **skor peluang** (probabilitas target=1), lalu kita ambil **Top-K** untuk prioritas.
- Kurva **Precision@K** / **Recall@K** menunjukkan trade-off:
  - K kecil → precision biasanya lebih tinggi (lebih “tepat sasaran”)
  - K besar → recall naik (lebih banyak positif yang terjangkau)
"""
        )

        colp, colr = st.columns(2)
        with colp:
            st.pyplot(st.session_state["sup_fig_pr"], use_container_width=True)
        with colr:
            st.pyplot(st.session_state["sup_fig_rc"], use_container_width=True)

        st.write("Tabel Precision@K / Recall@K:")
        pr_show = pr_df.copy()
        pr_show["precision_at_k"] = (pr_show["precision_at_k"] * 100).round(2)
        pr_show["recall_at_k"] = (pr_show["recall_at_k"] * 100).round(2)
        st.dataframe(pr_show, use_container_width=True)

        st.subheader("Lift & Top-K performance by group (dashboard)")
        df_eval = X_test.copy()
        df_eval["y_true"] = y_test
        df_eval["y_score"] = y_score

        group_candidates = [c for c in [
            "cluster_name", "cluster_id",
            "Segmen_karir", "Motivasi_cluster",
            "Product", "Kategori", "Channel", "Month", "Region"
        ] if c in df_eval.columns]

        gcol = st.selectbox("Pilih group breakdown", options=group_candidates, index=0, key="sup_group")
        lift_tbl = group_lift_table(df_eval, gcol, k_capacity)

        # nicer % columns
        lift_tbl = lift_tbl.rename(columns={gcol: "group"})
        lift_tbl["baseline_rate_pct"] = (lift_tbl["baseline_rate"] * 100).round(2)
        lift_tbl["topk_rate_pct"] = (lift_tbl["topk_rate"] * 100).round(2)
        st.dataframe(lift_tbl, use_container_width=True)

        st.subheader("Daftar kandidat Top-K (untuk action)")
        order = np.argsort(-y_score)
        k = int(max(1, min(k_capacity, len(order))))
        top_idx = order[:k]
        top_df = df_eval.iloc[top_idx].copy()
        top_df["rank"] = np.arange(1, len(top_df) + 1)
        cols_show = ["rank", "y_score"]
        # add some readable columns if present
        for c in ["Product", "Kategori", "Segmen_karir", "Motivasi_cluster", "cluster_name", "Region", "Channel", "Month"]:
            if c in top_df.columns and c not in cols_show:
                cols_show.append(c)

        st.dataframe(top_df[cols_show].sort_values("rank"), use_container_width=True)

    else:
        st.info("Klik **Run supervised ranking** dulu untuk menampilkan dashboard Top-K.")
