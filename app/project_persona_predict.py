import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Persona Segmentation & Placement Prediction", layout="wide")
alt.data_transformers.disable_max_rows()

# ----------------------------
# Helpers
# ----------------------------
def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def _norm_lower(x) -> str:
    return _norm_str(x).lower()

def make_flag_yes(x) -> int:
    s = _norm_lower(x)
    if s == "":
        return 0
    yes = ["ya", "y", "yes", "iya", "ikut", "pernah", "sudah", "hadir", "join", "gabung", "aktif", "sering"]
    no = ["tidak", "nggak", "gak", "ga", "belum", "never", "no", "tdk"]
    # token-level check
    toks = set(re.split(r"\s+", s))
    if toks & set(yes) and not (toks & set(no)):
        return 1
    if "pernah" in s and all(k not in s for k in ["tidak", "nggak", "gak", "ga", "belum"]):
        return 1
    return 0

def map_domain_product(product: str) -> str:
    s = _norm_lower(product)
    if any(k in s for k in ["data", "analytics", "analyst", "science", "machine learning", "ml", "ai"]):
        return "Data"
    if any(k in s for k in ["backend", "front", "fullstack", "web", "golang", "python", "software", "dev", "engineer"]):
        return "Tech"
    if any(k in s for k in ["business", "product", "pm", "management", "hr", "finance", "audit", "investment", "accounting"]):
        return "Business"
    if any(k in s for k in ["ui", "ux", "design", "dkv", "graphic", "visual"]):
        return "Design"
    if any(k in s for k in ["marketing", "digital marketing", "seo", "ads", "branding", "pr", "public relation"]):
        return "Marketing"
    return "Other"

def map_domain_pendidikan(jurusan: str, background: str = "") -> str:
    s = f"{_norm_lower(jurusan)} {_norm_lower(background)}"
    if any(k in s for k in ["informatika", "komputer", "sistem informasi", "teknik informatika", "it", "software", "computer"]):
        return "Tech"
    if any(k in s for k in ["statistika", "matematika", "data", "fisika", "kimia", "biologi", "science", "sains"]):
        return "Data"
    if any(k in s for k in ["manajemen", "akuntansi", "ekonomi", "bisnis", "marketing", "keuangan", "finance", "accounting"]):
        return "Business"
    if any(k in s for k in ["desain", "dkv", "arsitektur", "seni", "visual", "interior"]):
        return "Design"
    if any(k in s for k in ["komunikasi", "hukum", "psikologi", "sosiologi", "hubungan internasional", "administrasi publik", "politik"]):
        return "Social"
    return "Other"

def map_motivasi_cluster(motivasi: str) -> str:
    s = _norm_lower(motivasi)
    if s == "":
        return "Lainnya"
    if any(k in s for k in ["freelance", "freelancer"]):
        return "Freelance"
    if any(k in s for k in ["switch", "pindah", "ganti karir", "career switch", "alih karir"]):
        return "Switch career"
    if any(k in s for k in ["kerja", "job", "karir", "penyaluran", "tersalur", "placement"]):
        return "Dapat kerja"
    if any(k in s for k in ["belajar", "skill", "upskill", "reskill", "upgrade", "ilmu", "kompetensi"]):
        return "Belajar skill"
    return "Lainnya"

def make_target_penyaluran(x) -> int:
    """
    Target robust (biar nggak "100% semua" karena salah parsing):
    - kalau jawabannya "Ya/Sudah/Tersalur/Placed" => 1
    - kalau jawabannya "Tertarik" => 1
    - kalau "Tidak/Belum/Gak" => 0
    """
    s = _norm_lower(x)
    if s == "" or s in ["-", "nan"]:
        return 0

    # negatives
    if any(s.startswith(p) for p in ["tidak", "nggak", "gak", "ga", "belum"]):
        return 0
    if "tidak" in s and ("tertarik" in s or "tetarik" in s):
        return 0

    # positives: placement-like
    if any(k in s for k in ["ya", "sudah", "tersalur", "placed", "berhasil", "udah", "sukses"]):
        return 1

    # positives: interest-like
    if "tertarik" in s or "tetarik" in s or re.search(r"\btrt?arik\b", s):
        return 1

    return 0

def bucket_umur(u):
    try:
        u = float(u)
    except Exception:
        return "Unknown"
    if u < 20: return "<20"
    if u < 25: return "20-24"
    if u < 30: return "25-29"
    if u < 35: return "30-34"
    return "35+"

def make_engagement_level(row: pd.Series) -> str:
    score = 0
    score += make_flag_yes(row.get("Blog dibimbing", ""))
    score += make_flag_yes(row.get("Community", ""))
    score += make_flag_yes(row.get("Pernah ikut acara dibimbing/tidak", ""))
    prod = _norm_lower(row.get("Product", ""))
    if "job" in prod and "connect" in prod:
        score += 1

    if score >= 3: return "High"
    if score == 2: return "Medium"
    if score == 1: return "Low"
    return "None"

def apply_global_filters(df: pd.DataFrame, filter_cols: list[str], key_prefix: str="flt"):
    out = df.copy()
    for col in filter_cols:
        if col not in out.columns:
            continue
        vals = sorted([v for v in out[col].dropna().astype(str).unique().tolist()])
        if not vals:
            continue
        chosen = st.sidebar.multiselect(col, vals, default=vals, key=f"{key_prefix}_{col}")
        out = out[out[col].astype(str).isin(chosen)]
    return out

def target_rate_table(df: pd.DataFrame, group_col: str, target_col: str, min_n: int = 30):
    g = (
        df.groupby(group_col, dropna=False)[target_col]
        .agg(total="count", positives="sum")
        .reset_index()
    )
    g["rate_pct"] = np.where(g["total"] > 0, 100 * g["positives"] / g["total"], np.nan)
    g = g[g["total"] >= min_n].copy()
    g = g.sort_values(["rate_pct", "total"], ascending=[False, False])
    return g

def bar_rate_chart(g: pd.DataFrame, group_col: str, title: str, top_n: int = 20):
    if g.empty:
        return None
    gg = g.head(top_n).copy()
    gg[group_col] = gg[group_col].astype(str)

    base = alt.Chart(gg).mark_bar().encode(
        y=alt.Y(f"{group_col}:N", sort="-x", title=None),
        x=alt.X("rate_pct:Q", title="Target rate (%)"),
        tooltip=[group_col, "total:Q", "positives:Q", alt.Tooltip("rate_pct:Q", format=".2f")],
    ).properties(title=title, height=min(600, 25 * len(gg) + 60))

    text = alt.Chart(gg).mark_text(align="left", dx=5).encode(
        y=alt.Y(f"{group_col}:N", sort="-x"),
        x=alt.X("rate_pct:Q"),
        text=alt.Text("rate_pct:Q", format=".1f"),
    )
    return base + text

def build_preprocessor(cat_cols, num_cols):
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def name_persona_from_profile(profile: pd.DataFrame) -> dict:
    """
    Deterministic naming (biar sesuai PDF & stabil):
    - termuda / paling "fresh grad/intern" => Fresh Graduate Explorer
    - paling switcher + engagement tinggi => High Engagement Career Switcher
    - sisanya => Working Professional Upskiller
    """
    profile = profile.copy()
    fresh_cluster = int(profile.sort_values(["share_internish", "mean_umur"], ascending=[False, True]).iloc[0]["cluster_id"])
    switch_cluster = int(profile.sort_values(["share_switcher", "engagement_score"], ascending=[False, False]).iloc[0]["cluster_id"])
    remaining = [c for c in profile["cluster_id"].tolist() if c not in {fresh_cluster, switch_cluster}]
    work_cluster = int(remaining[0]) if remaining else switch_cluster
    return {
        fresh_cluster: "Fresh Graduate Explorer",
        switch_cluster: "High Engagement Career Switcher",
        work_cluster: "Working Professional Upskiller",
    }

def precision_recall_at_k(y_true, y_score, k):
    k = int(k)
    if k <= 0:
        return np.nan, np.nan
    order = np.argsort(-y_score)
    topk = order[:k]
    y_top = y_true.iloc[topk].values
    tp = y_top.sum()
    precision = tp / k
    recall = tp / max(1, y_true.sum())
    return precision, recall

# ----------------------------
# App
# ----------------------------
st.title("Persona Segmentation & Placement Prediction (PDF-aligned, single-file)")

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="uploader_csv")
path_text = st.sidebar.text_input("Atau isi path lokal (opsional)", value="", key="path_input")

df_raw = None
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
elif path_text.strip():
    try:
        df_raw = pd.read_csv(path_text.strip())
    except Exception as e:
        st.sidebar.error(f"Can't read path: {e}")

if df_raw is None:
    st.info("Upload CSV dulu untuk mulai.")
    st.stop()

df0 = df_raw.copy()
df0.columns = [c.strip() for c in df0.columns]

# Feature engineering (PDF-aligned style)
df0["Motivasi"] = df0.get("Motivasi mengikuti bootcamp", df0.get("Motivasi", "")).astype(str)
df0["Motivasi_cluster"] = df0["Motivasi"].apply(map_motivasi_cluster)

df0["Domain_pendidikan"] = [
    map_domain_pendidikan(j, b)
    for j, b in zip(df0.get("Jurusan pendidikan", ""), df0.get("Background pendidikan", ""))
]
df0["Domain_product"] = df0.get("Product", "").apply(map_domain_product)
df0["is_switcher"] = (df0["Domain_pendidikan"] != df0["Domain_product"]).astype(int)

df0["Engagement_level"] = df0.apply(make_engagement_level, axis=1)

df0["Umur_num"] = pd.to_numeric(df0.get("Umur", np.nan), errors="coerce")
df0["Umur_bin"] = df0["Umur_num"].apply(bucket_umur)

def segmen_karir_baseline(row):
    if int(row.get("is_switcher", 0)) == 1:
        return "Switch career"
    kp = _norm_lower(row.get("Kategori Pekerjaan", ""))
    lp = _norm_lower(row.get("Level Pekerjaan", ""))
    kk = _norm_lower(row.get("Kategori Kesibukan", ""))
    if re.search(r"(intern|magang|fresh)", kp) or "fresh" in lp or "mahasiswa" in kk:
        return "Fresh Graduate Explorer"
    return "Working Professional Upskiller"

df0["Segmen_karir"] = df0.apply(segmen_karir_baseline, axis=1)

# Target
df0["target"] = df0.get("Penyaluran Kerja", "").apply(make_target_penyaluran).astype(int)

# ----------------------------
# Filters
# ----------------------------
st.sidebar.header("Global filters")
filterable_cols = ["Product", "Kategori", "Channel", "Month", "Segmen_karir", "Motivasi_cluster", "Engagement_level", "Umur_bin"]
existing_filter_cols = [c for c in filterable_cols if c in df0.columns]

chosen_filter_cols = st.sidebar.multiselect(
    "Pilih kolom untuk filter",
    existing_filter_cols,
    default=[c for c in ["Product", "Kategori"] if c in existing_filter_cols],
    key="filter_cols_pick",
)

df = apply_global_filters(df0, chosen_filter_cols, key_prefix="gflt")

tabs = st.tabs(["Overview", "EDA (Target-driven)", "Clustering (Persona)", "Supervised (Top-K Ranking)"])

# ----------------------------
# Overview
# ----------------------------
with tabs[0]:
    st.subheader("Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    n = len(df)
    pos = int(df["target"].sum())
    rate = (pos / n * 100) if n else 0
    c1.metric("Rows (after filter)", f"{n:,}")
    c2.metric("Target positives", f"{pos:,}")
    c3.metric("Target rate", f"{rate:.2f}%")
    c4.metric("Distinct products", f"{df['Product'].nunique() if 'Product' in df.columns else 0:,}")

    st.caption(
        "Catatan: group bisa 100% kalau sample group kecil / target dominan. "
        "Di EDA kamu bisa atur minimum group size untuk ngefilter noise."
    )
    st.dataframe(df.head(30), use_container_width=True)

# ----------------------------
# EDA
# ----------------------------
with tabs[1]:
    st.subheader("EDA yang selaras dengan tujuan: pola peserta → segmentasi → prediksi penyaluran")

    dist = df["target"].value_counts().rename_axis("target").reset_index(name="count")
    dist["label"] = dist["target"].map({0: "0 (Tidak)", 1: "1 (Tertarik/Tersalur)"})
    ch = alt.Chart(dist).mark_bar().encode(
        x=alt.X("label:N", title=None),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["label:N", "count:Q"],
    ).properties(title="Distribusi target", height=250)
    st.altair_chart(ch, use_container_width=True)

    st.markdown("### Target rate by breakdown")
    breakdown_candidates = [c for c in [
        "Product", "Kategori", "Channel", "Motivasi_cluster", "Segmen_karir",
        "Engagement_level", "Umur_bin", "Domain_product", "Domain_pendidikan", "is_switcher"
    ] if c in df.columns]

    breakdown_col = st.selectbox("Pilih breakdown", breakdown_candidates, index=0, key="eda_breakdown")
    min_n = st.slider("Minimum ukuran group (filter noise)", 1, 300, 30, key="eda_min_n")
    top_n = st.slider("Show top-N groups", 5, 60, 20, key="eda_top_n")

    g = target_rate_table(df, breakdown_col, "target", min_n=min_n)
    chart = bar_rate_chart(g, breakdown_col, f"Target rate by {breakdown_col} (min_n={min_n})", top_n=top_n)

    if chart is None:
        st.warning("Tidak ada group yang memenuhi minimum ukuran group. Turunkan min_n atau ganti breakdown.")
    else:
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(g.head(top_n), use_container_width=True)

# ----------------------------
# Clustering
# ----------------------------
with tabs[2]:
    st.subheader("Persona clustering (3 cluster, PDF-aligned)")

    cat_cols_pref = [
        "Batch", "Product", "Kategori", "Channel",
        "Level pendidikan", "Kategori Kesibukan", "Level Pekerjaan", "Kategori Pekerjaan",
        "Domain_pendidikan", "Domain_product", "Motivasi_cluster", "Engagement_level", "Umur_bin", "is_switcher"
    ]
    cat_cols = [c for c in cat_cols_pref if c in df0.columns]
    num_cols = ["Umur_num"]

    st.caption("Fitur yang dipakai (sesuai PDF, disesuaikan dengan kolom yang tersedia):")
    st.code(f"Categorical: {cat_cols}\nNumeric: {num_cols}")

    run = st.button("Run clustering", key="btn_cluster")
    if run:
        try:
            X = df0[cat_cols + num_cols].copy()

            pre = build_preprocessor(cat_cols, num_cols)
            kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)  # fixed, no UI knob
            pipe = Pipeline([("pre", pre), ("kmeans", kmeans)])

            labels = pipe.fit_predict(X)

            Xt = pipe.named_steps["pre"].transform(X)
            svd = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(Xt)

            tmp = df0.copy()
            tmp["cluster_id"] = labels
            tmp["svd1"] = coords[:, 0]
            tmp["svd2"] = coords[:, 1]

            tmp["_internish"] = (
                tmp.get("Kategori Pekerjaan", "").astype(str).str.lower().str.contains("intern|magang|fresh", regex=True, na=False).astype(int)
                if "Kategori Pekerjaan" in tmp.columns else 0
            )
            tmp["_eng_score"] = tmp["Engagement_level"].map({"None": 0, "Low": 1, "Medium": 2, "High": 3}).fillna(0).astype(float)

            profile = tmp.groupby("cluster_id").agg(
                mean_umur=("Umur_num", "mean"),
                share_switcher=("is_switcher", "mean"),
                share_internish=("_internish", "mean"),
                engagement_score=("_eng_score", "mean"),
                n=("cluster_id", "count"),
            ).reset_index()

            mapping = name_persona_from_profile(profile)
            tmp["Persona"] = tmp["cluster_id"].map(mapping).fillna(tmp["cluster_id"].astype(str))

            st.session_state["cluster_result"] = {
                "df": tmp,
                "pipe": pipe,
                "persona_map": mapping,
                "profile": profile,
            }
            st.success("Clustering selesai. Persona sudah dinamai (PDF-aligned).")

        except Exception as e:
            st.error(f"Gagal clustering: {e}")
            st.stop()

    if "cluster_result" in st.session_state and st.session_state["cluster_result"] is not None:
        tmp = st.session_state["cluster_result"]["df"]
        profile = st.session_state["cluster_result"]["profile"]
        mapping = st.session_state["cluster_result"]["persona_map"]

        st.markdown("### Cluster visualization (TruncatedSVD 2D) — Altair")
        chart = alt.Chart(tmp).mark_circle(size=30, opacity=0.6).encode(
            x=alt.X("svd1:Q", title="SVD-1"),
            y=alt.Y("svd2:Q", title="SVD-2"),
            color=alt.Color("Persona:N", legend=alt.Legend(title="Persona")),
            tooltip=[
                "Persona:N", "Product:N", "Kategori:N", "Motivasi_cluster:N", "Engagement_level:N",
                alt.Tooltip("Umur_num:Q", format=".0f"),
            ],
        ).properties(height=520)
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Profiling ringkas per persona")
        profile2 = profile.copy()
        profile2["Persona"] = profile2["cluster_id"].map(mapping)
        st.dataframe(
            profile2[["Persona", "n", "mean_umur", "share_switcher", "share_internish", "engagement_score"]]
            .sort_values("n", ascending=False),
            use_container_width=True,
        )

# ----------------------------
# Supervised
# ----------------------------
with tabs[3]:
    st.subheader("Supervised ranking (LogReg) — Top-K (PDF-aligned)")

    df_sup = df0.copy()
    if "cluster_result" in st.session_state and st.session_state["cluster_result"] is not None:
        df_sup = st.session_state["cluster_result"]["df"].copy()

    n = len(df_sup)
    pos = int(df_sup["target"].sum())
    neg = n - pos
    st.write(f"Target counts: 1={pos:,} | 0={neg:,}")

    if pos < 5 or neg < 5:
        st.warning("Data target terlalu tidak seimbang / terlalu sedikit untuk supervised yang stabil. Cek definisi target atau pakai data lebih lengkap.")
        st.stop()

    feat_candidates = [
        "Umur_num", "Umur_bin", "Domisili", "Provinsi", "Negara",
        "Batch", "Product", "Kategori", "Channel",
        "Level pendidikan", "Kategori Kesibukan", "Level Pekerjaan", "Kategori Pekerjaan",
        "Motivasi_cluster", "Domain_pendidikan", "Domain_product", "is_switcher", "Engagement_level"
    ]
    if "Persona" in df_sup.columns:
        feat_candidates.append("Persona")

    feat_cols = [c for c in feat_candidates if c in df_sup.columns]

    with st.expander("Feature columns (klik untuk lihat)"):
        st.write(feat_cols)

    test_size = st.slider("Holdout test_size", 0.1, 0.4, 0.2, 0.05, key="sup_test_size")
    k_capacity = st.slider("Business capacity K (berapa peserta yang bisa diintervensi)", 50, min(2000, n), 200, 50, key="sup_k")

    run_sup = st.button("Run supervised ranking", key="btn_sup")
    if run_sup:
        try:
            X = df_sup[feat_cols].copy()
            y = df_sup["target"].astype(int)

            num_cols = ["Umur_num"] if "Umur_num" in feat_cols else []
            cat_cols = [c for c in feat_cols if c not in num_cols]

            pre = build_preprocessor(cat_cols, num_cols)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            model = Pipeline([("pre", pre), ("clf", clf)])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=42, stratify=y
            )

            model.fit(X_train, y_train)
            p_test = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, p_test)
            ap = average_precision_score(y_test, p_test)

            st.session_state["sup_result"] = {
                "model": model,
                "feat_cols": feat_cols,
                "X": X,
                "y": y,
                "X_test": X_test,
                "y_test": y_test,
                "p_test": p_test,
                "auc": auc,
                "ap": ap,
            }
            st.success("Model trained.")

        except Exception as e:
            st.error(f"Gagal supervised: {e}")
            st.stop()

    if "sup_result" in st.session_state:
        res = st.session_state["sup_result"]
        st.metric("ROC-AUC (holdout)", f"{res['auc']:.3f}")
        st.metric("Average Precision (PR-AUC)", f"{res['ap']:.3f}")

        y_test = res["y_test"].reset_index(drop=True)
        p_test = pd.Series(res["p_test"]).reset_index(drop=True)

        max_k = min(len(y_test), 2000)
        ks = np.unique(np.linspace(50, max_k, 25).astype(int))
        rows = []
        for k in ks:
            prec, rec = precision_recall_at_k(y_test, p_test, k)
            rows.append({"K": int(k), "precision_at_k": float(prec), "recall_at_k": float(rec)})
        curve = pd.DataFrame(rows)

        prec_chart = alt.Chart(curve).mark_line().encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("precision_at_k:Q", title="Precision@K"),
            tooltip=["K:Q", alt.Tooltip("precision_at_k:Q", format=".3f")],
        ).properties(height=250, title="Precision@K curve (holdout)")

        rec_chart = alt.Chart(curve).mark_line().encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("recall_at_k:Q", title="Recall@K"),
            tooltip=["K:Q", alt.Tooltip("recall_at_k:Q", format=".3f")],
        ).properties(height=250, title="Recall@K curve (holdout)")

        st.altair_chart(prec_chart, use_container_width=True)
        st.altair_chart(rec_chart, use_container_width=True)

        prec_k, rec_k = precision_recall_at_k(y_test, p_test, k_capacity)
        baseline = y_test.mean()

        st.markdown("### Ringkasan untuk K yang dipilih")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline target rate", f"{baseline*100:.2f}%")
        c2.metric("Precision@K", f"{prec_k*100:.2f}%")
        c3.metric("Recall@K", f"{rec_k*100:.2f}%")

        st.markdown("### Dashboard akhir (sesuai tujuan proyek)")
        st.markdown(
            f"""
Proyek ini bertujuan untuk:
1) memahami pola peserta Dibimbing,  
2) membangun segmentasi peserta yang akurat (persona), dan  
3) mengembangkan model prediktif peluang penyaluran kerja untuk keputusan bisnis.

**Interpretasi Top-K:**
- **K = {k_capacity}** = kapasitas intervensi (jumlah peserta yang bisa di-follow-up).
- Model memberi skor peluang → kita ambil **Top-K** untuk prioritas outreach.
- Kurva **Precision@K / Recall@K** menunjukkan trade-off:
  - K kecil → precision lebih tinggi (lebih tepat sasaran)
  - K besar → recall naik (lebih banyak positif terjangkau)
            """
        )

        # Uplift table (lebih aman: pakai subset topk_df)
        model = res["model"]
        X_all = res["X"]
        p_all = model.predict_proba(X_all)[:, 1]

        full = df_sup.copy()
        full["_score"] = p_all
        full_sorted = full.sort_values("_score", ascending=False).reset_index(drop=True)
        full_sorted["_is_topk"] = (np.arange(len(full_sorted)) < k_capacity).astype(int)

        group_candidates = [c for c in ["Persona", "Product", "Kategori", "Channel", "Motivasi_cluster", "Segmen_karir", "Engagement_level"] if c in full_sorted.columns]
        group_col = st.selectbox("Breakdown uplift by", group_candidates, index=0, key="uplift_group")

        base_g = full_sorted.groupby(group_col).agg(
            baseline_total=("target", "count"),
            baseline_pos=("target", "sum"),
        ).reset_index()

        topk_df = full_sorted[full_sorted["_is_topk"] == 1].copy()
        topk_g = topk_df.groupby(group_col).agg(
            topk_total=("target", "count"),
            topk_pos=("target", "sum"),
        ).reset_index()

        g = base_g.merge(topk_g, on=group_col, how="left").fillna({"topk_total": 0, "topk_pos": 0})
        g["baseline_rate_pct"] = 100 * g["baseline_pos"] / g["baseline_total"]
        g["topk_rate_pct"] = np.where(g["topk_total"] > 0, 100 * g["topk_pos"] / g["topk_total"], np.nan)
        g["lift"] = g["topk_rate_pct"] / (g["baseline_rate_pct"] + 1e-9)
        g = g.sort_values(["lift", "topk_total"], ascending=[False, False])

        st.markdown("### Uplift table (Top-K vs baseline)")
        st.dataframe(g.head(30), use_container_width=True)

        st.markdown("### Rekomendasi aksi (langsung bisa dipakai bisnis)")
        top_actions = g[g["topk_total"] > 0].head(10).copy()
        st.write("Prioritas intervensi (Top-10 group):")
        st.dataframe(top_actions[[group_col, "topk_total", "topk_pos", "topk_rate_pct", "lift"]], use_container_width=True)
