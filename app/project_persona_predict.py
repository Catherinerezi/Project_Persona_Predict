# app.py
# Persona Segmentation & Placement Prediction (PDF-aligned) — single-file Streamlit
# Charts: Altair only (no Plotly). Clustering: 3 clusters + persona naming. Supervised: Top-K dashboard.

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
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Persona Segmentation & Placement Prediction (PDF-aligned)",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

# ----------------------------
# Helpers
# ----------------------------
def _safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

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
            out = out[out[c].isin(chosen)]
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

def make_target(df: pd.DataFrame, target_col="Penyaluran Kerja", mode="TERTARIK_AS_POSITIVE"):
    """
    Returns y (0/1) and a pretty label description.
    mode:
      - TERTARIK_AS_POSITIVE: 1 if value contains "Tertarik"
      - TIDAK_TERTARIK_AS_POSITIVE: 1 if value contains "Tidak"
    """
    if target_col not in df.columns:
        return None, f"Target column '{target_col}' tidak ditemukan."

    s = df[target_col].astype(str).fillna("")

    if mode == "TERTARIK_AS_POSITIVE":
        y = s.str.contains("Tertarik", case=False, regex=False).astype(int)
        desc = "Target=1 jika 'Tertarik' (propensity tertarik penyaluran)"
    else:
        y = s.str.contains("Tidak", case=False, regex=False).astype(int)
        desc = "Target=1 jika 'Tidak Tertarik' (at-risk: butuh intervensi)"

    return y, desc

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

def alt_bar_rate(df_rate: pd.DataFrame, title: str):
    if df_rate.empty:
        return None
    base = alt.Chart(df_rate).mark_bar().encode(
        y=alt.Y("Group:N", sort="-x", title="Group"),
        x=alt.X("rate_pct:Q", title="Target rate (%)"),
        tooltip=["Group:N", "Total:Q", "Positives:Q", "rate_pct:Q"],
    ).properties(title=title, height=min(520, 24 * len(df_rate) + 80))
    text = alt.Chart(df_rate).mark_text(align="left", dx=6).encode(
        y=alt.Y("Group:N", sort="-x"),
        x=alt.X("rate_pct:Q"),
        text=alt.Text("rate_pct:Q", format=".2f"),
    )
    return (base + text)

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
        # numeric safe scaler
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def prepare_X(df_in: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    """
    Prevent ValueError in ColumnTransformer/OneHotEncoder:
    - cat: fillna + cast to str
    - num: to_numeric + fillna
    """
    X = df_in[cat_cols + num_cols].copy()

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
        seg_top = sub["Segmen_karir"].value_counts().idxmax() if "Segmen_karir" in sub.columns and len(sub) else ""
        mot_top = sub["Motivasi_cluster"].value_counts().idxmax() if "Motivasi_cluster" in sub.columns and len(sub) else ""
        rows.append((c, _safe_str(seg_top), _safe_str(mot_top), len(sub)))
    prof = pd.DataFrame(rows, columns=["cluster", "seg_top", "mot_top", "n"])

    used = set()

    def pick_cluster(cond):
        cand = prof[cond].sort_values("n", ascending=False)
        for cl in cand["cluster"].tolist():
            if cl not in used:
                used.add(cl)
                return cl
        return None

    c_fresh = pick_cluster(prof["seg_top"].str.contains("Fresh|Graduate|Mahasiswa|Pelajar", case=False, na=False))
    c_switch = pick_cluster(
        prof["mot_top"].str.contains("Switch|career|Pindah", case=False, na=False)
        | prof["seg_top"].str.contains("Switch|career|Pindah", case=False, na=False)
    )
    c_work = pick_cluster(
        prof["seg_top"].str.contains("Working|Professional|Karyawan|Employee", case=False, na=False)
        | prof["mot_top"].str.contains("Upgrade|Upskill|Skill", case=False, na=False)
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
    return precision, recall

def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    base = y_true.mean() if len(y_true) else 0.0
    prec, _ = precision_recall_at_k(y_true, y_score, k)
    return (prec / base) if base > 0 else 0.0

def _signature(df_in: pd.DataFrame):
    # small signature so cache resets when filter changes
    idx = df_in.index.to_numpy()
    head = tuple(idx[:5].tolist()) if len(idx) else ()
    tail = tuple(idx[-5:].tolist()) if len(idx) else ()
    return (len(df_in), head, tail)

# ----------------------------
# Sidebar: data source
# ----------------------------
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

with st.sidebar:
    st.header("Data source")

    source_choice = st.radio(
        "Choose",
        ["Upload file (CSV)", "Repo file (path)"],
        key="src_choice",
    )

    uploaded = None
    repo_path = None

    if source_choice == "Upload file (CSV)":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
    else:
        repo_path = st.text_input(
            "Path (contoh: raw_data/raw_data.csv)",
            value="",
            key="repo_path_input",
        )

df, src = load_data(uploaded, repo_path)

if df is None:
    st.info("Upload CSV atau isi path repo dulu.")
    st.stop()

df = df.copy()
df.columns = [c.strip() for c in df.columns]

# ----------------------------
# Target settings (critical to match PDF)
# ----------------------------
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
        options=["Tertarik = 1", "Tidak Tertarik = 1"],
        key="target_mode_radio",
        help="Kalau hasil PDF timpang (positif kecil), biasanya pakai target minoritas (mis. Tidak Tertarik=1 untuk intervensi).",
    )

mode = "TERTARIK_AS_POSITIVE" if target_mode == "Tertarik = 1" else "TIDAK_TERTARIK_AS_POSITIVE"
y, y_desc = make_target(df, target_col=target_col, mode=mode)
if y is None:
    st.error(y_desc)
    st.stop()

# Fallback: kalau mapping string bikin 0 semua, kasih opsi pilih nilai positif manual
if int(y.sum()) == 0 and target_col in df.columns:
    with st.sidebar:
        st.warning("Target jadi 0 semua. Pilih nilai yang dianggap POSITIF agar sesuai data kamu.")
        uniq = sorted(df[target_col].dropna().astype(str).unique().tolist())
        if uniq:
            pos_val = st.selectbox("Nilai POSITIF (jadi 1)", options=uniq, key="manual_pos_val")
            y = (df[target_col].astype(str) == str(pos_val)).astype(int)
            y_desc = f"Target=1 jika '{pos_val}' (manual mapping)"

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
    cat_like = [c for c in df.columns if df[c].dtype == "object"]
    default_filter_cols = [c for c in ["Product", "Kategori", "Month", "Segmen_karir", "Motivasi_cluster"] if c in df.columns]

    filter_cols = st.multiselect(
        "Pilih kolom untuk filter",
        options=cat_like,
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
y_f, _ = make_target(df_f, target_col=target_col, mode=mode)

st.divider()

# Reset caches if filters change
sig_f = _signature(df_f)
if st.session_state.get("sig_f_prev") != sig_f:
    st.session_state["sig_f_prev"] = sig_f
    # clear cached model outputs tied to old filtered data
    for k in ["cluster_done", "df_clustered", "persona_map", "cluster_sig", "sup_done", "sup_sig",
              "sup_pipe", "sup_metrics", "sup_curves", "sup_top"]:
        if k in st.session_state:
            del st.session_state[k]

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_eda, tab_cluster, tab_sup = st.tabs(
    ["Overview", "EDA (Target-driven)", "Clustering (Persona)", "Supervised (Top-K Ranking)"]
)

# ----------------------------
# Overview
# ----------------------------
with tab_overview:
    st.subheader("Project goal alignment")
    st.write(
        """
Proyek ini harus menjawab 3 hal:
1) **Memahami pola peserta** (EDA yang bisa di-breakdown & difilter),
2) **Membangun segmentasi persona yang akurat** (clustering 3 persona + profiling),
3) **Model prediktif untuk keputusan bisnis** (Top-K ranking + dashboard action: siapa diprioritaskan, trade-off K, dan dampak per segmen/persona).
"""
    )

    st.subheader("Data snapshot (setelah filter)")
    st.write(f"Sumber data: **{src}**  | Rows aktif: **{len(df_f):,}**")
    st.dataframe(df_f.head(30), use_container_width=True)

# ----------------------------
# EDA (Target-driven)
# ----------------------------
with tab_eda:
    st.subheader("EDA yang selaras tujuan (target-driven)")
    left, _right = st.columns([1, 1])

    with left:
        breakdown_col = st.selectbox(
            "Pilih breakdown (groupby)",
            options=[c for c in df_f.columns if df_f[c].dtype == "object"],
            index=0 if any(df_f[c].dtype == "object" for c in df_f.columns) else 0,
            key="eda_breakdown",
        )
        min_count = st.slider("Min count per group", 5, 300, 30, key="eda_min_count")
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
    df_t = pd.DataFrame(
        {"target": [0, 1], "count": [int((y_f == 0).sum()), int((y_f == 1).sum())]}
    )
    dist = alt.Chart(df_t).mark_bar().encode(
        x=alt.X("target:N", title="Target"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["target:N", "count:Q"],
    ).properties(height=240)
    st.altair_chart(dist, use_container_width=True)

# ----------------------------
# Clustering
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
    else:
        run_cluster = st.button("Run clustering", key="btn_cluster_run")

        # cache tied to current filtered signature
        cluster_sig = ("cluster", sig_f, tuple(cat_cols), tuple(num_cols))
        if run_cluster or (st.session_state.get("cluster_done") and st.session_state.get("cluster_sig") == cluster_sig):
            if run_cluster or not st.session_state.get("cluster_done") or st.session_state.get("cluster_sig") != cluster_sig:
                preprocess = build_preprocess(cat_cols, num_cols)

                X = prepare_X(df_f, cat_cols, num_cols)
                pipe = Pipeline([("prep", preprocess)])
                X_enc = pipe.fit_transform(X)

                svd = TruncatedSVD(n_components=2, random_state=42)
                X_2d = svd.fit_transform(X_enc)

                # MiniBatchKMeans works with sparse matrices safely
                km = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=20, batch_size=1024)
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

            st.subheader("Cluster visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)")
            tooltip_cols = ["Persona:N"]
            if "Product" in dfc.columns:
                tooltip_cols.append("Product:N")
            if "Kategori" in dfc.columns:
                tooltip_cols.append("Kategori:N")

            scatter = alt.Chart(dfc).mark_circle(size=22, opacity=0.65).encode(
                x=alt.X("_svd1:Q", title="SVD-1"),
                y=alt.Y("_svd2:Q", title="SVD-2"),
                color=alt.Color("Persona:N", legend=alt.Legend(title="Persona")),
                tooltip=tooltip_cols,
            ).properties(height=520)
            st.altair_chart(scatter, use_container_width=True)

            st.subheader("Cluster profiling (untuk penamaan persona)")
            prof_cols = [c for c in ["Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel"] if c in dfc.columns]
            if prof_cols:
                blocks = []
                for p in dfc["Persona"].unique():
                    sub = dfc[dfc["Persona"] == p]
                    row = {"Persona": p, "Size": len(sub)}
                    y_sub, _ = make_target(sub, target_col=target_col, mode=mode)
                    row["Target rate (%)"] = round(y_sub.mean() * 100, 2) if len(sub) else 0.0
                    for c in prof_cols:
                        if c == "Persona":
                            continue
                        vc = sub[c].astype(str).value_counts()
                        row[f"Top {c}"] = vc.index[0] if len(vc) else ""
                    blocks.append(row)
                prof_df = pd.DataFrame(blocks).sort_values("Size", ascending=False)
                st.dataframe(prof_df, use_container_width=True)
            else:
                st.info("Kolom profiling (Segmen_karir/Motivasi_cluster/...) tidak tersedia di data ini.")

# ----------------------------
# Supervised (Top-K)
# ----------------------------
with tab_sup:
    st.subheader("Supervised ranking — Top-K (PDF-aligned)")

    dfm = df_f.copy()
    if st.session_state.get("cluster_done"):
        dfc = st.session_state["df_clustered"]
        dfm = dfm.merge(
            dfc[["_cluster_id", "Persona"]],
            left_index=True,
            right_index=True,
            how="left",
        )

    cat_cols, num_cols = infer_feature_sets(dfm)
    if "Persona" in dfm.columns and "Persona" not in cat_cols:
        cat_cols = cat_cols + ["Persona"]

    if len(cat_cols) + len(num_cols) == 0:
        st.error("Tidak ada feature untuk supervised (cek kolom dataset).")
        st.stop()

    y_m, _ = make_target(dfm, target_col=target_col, mode=mode)
    if y_m.nunique() < 2:
        st.error("Target setelah filter hanya punya 1 kelas. Ubah filter atau ganti Target Mode.")
        st.stop()

    n_all, n_pos, n_neg, pos_rate = target_summary(y_m)
    st.write(f"Rows: **{n_all:,}** | Positives: **{n_pos:,}** | Positive rate: **{pos_rate*100:.2f}%**")

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

    # solver liblinear is robust for sparse + binary
    model = LogisticRegression(
        max_iter=3000,
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

    sup_sig = ("sup", sig_f, tuple(cat_cols), tuple(num_cols), float(test_size), class_weight, float(c_reg), int(k_cap))
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

            pr_auc = average_precision_score(y_test, p_test)
            try:
                roc_auc = roc_auc_score(y_test, p_test)
            except Exception:
                roc_auc = np.nan

            ks = np.unique(np.clip(np.linspace(50, min(len(y_test), 2000), 30).astype(int), 1, None))
            precs, recs, lifts = [], [], []
            for k in ks:
                pr_k, rc_k = precision_recall_at_k(y_test, p_test, int(k))
                lf_k = lift_at_k(y_test, p_test, int(k))
                precs.append(pr_k)
                recs.append(rc_k)
                lifts.append(lf_k)

            p_full = pipe.predict_proba(X)[:, 1]
            order = np.argsort(-p_full)
            top_idx = order[:int(k_cap)]
            df_top = dfm.iloc[top_idx].copy()
            df_top["score"] = p_full[top_idx]
            df_top["pred_rank"] = np.arange(1, len(df_top) + 1)

            st.session_state["sup_done"] = True
            st.session_state["sup_sig"] = sup_sig
            st.session_state["sup_pipe"] = pipe
            st.session_state["sup_metrics"] = {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)}
            st.session_state["sup_curves"] = pd.DataFrame(
                {"k": ks, "precision_at_k": precs, "recall_at_k": recs, "lift_at_k": lifts}
            )
            st.session_state["sup_top"] = df_top.sort_values("pred_rank")

        met = st.session_state["sup_metrics"]
        curves = st.session_state["sup_curves"]
        df_top = st.session_state["sup_top"]

        st.subheader("Model quality (ranking-oriented)")
        m1, m2 = st.columns(2)
        m1.metric("PR-AUC (Average Precision)", f"{met['pr_auc']:.4f}")
        m2.metric("ROC-AUC (opsional)", f"{met['roc_auc']:.4f}" if not np.isnan(met["roc_auc"]) else "N/A")

        st.caption("Fokus utama untuk Top-K biasanya PR-AUC + Precision@K/Recall@K/Lift@K (lebih nyambung ke kapasitas bisnis).")

        st.subheader("Trade-off curve: Precision@K, Recall@K, Lift@K")
        c_left, c_right = st.columns(2)

        ch1 = alt.Chart(curves).mark_line(point=True).encode(
            x=alt.X("k:Q", title="K"),
            y=alt.Y("precision_at_k:Q", title="Precision@K"),
            tooltip=["k:Q", alt.Tooltip("precision_at_k:Q", format=".3f")],
        ).properties(height=260, title="Precision@K")
        c_left.altair_chart(ch1, use_container_width=True)

        ch2 = alt.Chart(curves).mark_line(point=True).encode(
            x=alt.X("k:Q", title="K"),
            y=alt.Y("recall_at_k:Q", title="Recall@K"),
            tooltip=["k:Q", alt.Tooltip("recall_at_k:Q", format=".3f")],
        ).properties(height=260, title="Recall@K")
        c_right.altair_chart(ch2, use_container_width=True)

        ch3 = alt.Chart(curves).mark_line(point=True).encode(
            x=alt.X("k:Q", title="K"),
            y=alt.Y("lift_at_k:Q", title="Lift@K"),
            tooltip=["k:Q", alt.Tooltip("lift_at_k:Q", format=".2f")],
        ).properties(height=260, title="Lift@K (efektivitas vs baseline)")
        st.altair_chart(ch3, use_container_width=True)

        st.subheader("Top-K output (untuk eksekusi bisnis)")
        show_cols = [c for c in ["pred_rank", "score", "Persona", "Segmen_karir", "Motivasi_cluster", "Product", "Kategori", "Channel", "Umur"] if c in df_top.columns]
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

        st.subheader("Dashboard akhir (actionable summary untuk tujuan proyek)")
        st.write("**1) Segmentasi & pola (target rate by persona/segmen)**")
        if "Persona" in dfm.columns:
            df_rate_p = top_rate_by_group(dfm, y_m, "Persona", min_count=10, top_n=20)
            chp = alt_bar_rate(df_rate_p, "Target rate by Persona")
            if chp is not None:
                st.altair_chart(chp, use_container_width=True)

        if "Segmen_karir" in dfm.columns:
            df_rate_s = top_rate_by_group(dfm, y_m, "Segmen_karir", min_count=30, top_n=30)
            chs = alt_bar_rate(df_rate_s, "Target rate by Segmen_karir (Top, min_count=30)")
            if chs is not None:
                st.altair_chart(chs, use_container_width=True)

        st.write("**2) Kenapa memilih Top-K segini? (trade-off & kapasitas)**")
        base_rate = y_m.mean()
        nearest = curves.iloc[(curves["k"] - int(k_cap)).abs().argsort()[:1]]
        if len(nearest):
            nk = int(nearest["k"].iloc[0])
            st.markdown(
                f"""
- **K = {k_cap}** = kapasitas intervensi (berapa peserta yang bisa di-follow up).
- Baseline positive rate = **{base_rate*100:.2f}%**.
- Perkiraan di sekitar K≈{nk}:  
  - Precision@K ≈ **{nearest['precision_at_k'].iloc[0]*100:.2f}%**  
  - Recall@K ≈ **{nearest['recall_at_k'].iloc[0]*100:.2f}%**  
  - Lift@K ≈ **{nearest['lift_at_k'].iloc[0]:.2f}×**
"""
            )

        st.write("**3) Distribusi Top-K (untuk strategi akuisisi/desain program/intervensi)**")
        if "Persona" in df_top.columns:
            dist_top = df_top["Persona"].value_counts().reset_index()
            dist_top.columns = ["Persona", "count"]
            chd = alt.Chart(dist_top).mark_bar().encode(
                y=alt.Y("Persona:N", sort="-x"),
                x=alt.X("count:Q"),
                tooltip=["Persona:N", "count:Q"],
            ).properties(height=220, title="Top-K count by Persona")
            st.altair_chart(chd, use_container_width=True)

        if "Product" in df_top.columns:
            dist_prod = df_top["Product"].astype(str).value_counts().head(20).reset_index()
            dist_prod.columns = ["Product", "count"]
            chp2 = alt.Chart(dist_prod).mark_bar().encode(
                y=alt.Y("Product:N", sort="-x"),
                x=alt.X("count:Q"),
                tooltip=["Product:N", "count:Q"],
            ).properties(height=320, title="Top-K count by Product (Top 20)")
            st.altair_chart(chp2, use_container_width=True)

        st.success(
            "Kalau mau hasil identik PDF: pastikan dataset sama + definisi target sama. "
            "Di file ini mismatch akan kelihatan dari distribusi target & PR-AUC/@K."
        )
