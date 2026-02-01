# app.py
# Persona Segmentation & Placement Prediction (PDF-aligned) — single-file Streamlit
# - Target utama: Penyaluran Kerja (placement) -> Penyaluran_flag
# - EDA: target-driven
# - Clustering: 3 clusters (MiniBatchKMeans), persona naming (PDF-aligned)
# - Supervised: LogisticRegression + GridSearchCV (scoring=average_precision / PR-AUC), report ROC-AUC too
# - Final dashboard: Top-K ranking + segment lift + business-ready recommendations

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import altair as alt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(
    page_title="Persona Segmentation & Placement Prediction (PDF-aligned)",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

st.markdown(
    """
<style>
/* Make tables nicer */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
/* Slightly tighter main padding */
.main .block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers (PDF-aligned FE)
# ----------------------------
def _col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find best matching column name (exact or case/strip)."""
    if name in df.columns:
        return name
    low = {c.strip().lower(): c for c in df.columns}
    return low.get(name.strip().lower())


def _make_ohe():
    """Sklearn compatibility: sparse_output vs sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_age_to_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    m = re.search(r"(\d{1,2})", s)
    if not m:
        return None
    try:
        v = int(m.group(1))
        if v <= 0 or v > 80:
            return None
        return v
    except Exception:
        return None


def age_bin(v: Optional[int]) -> str:
    # PDF snippet uses: bins=[0,22,25,30,100], labels
    if v is None:
        return "Unknown"
    if v <= 22:
        return "≤22"
    if 23 <= v <= 25:
        return "23–25"
    if 26 <= v <= 30:
        return "26–30"
    return ">30"


def parse_batch_num(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def batch_has_plus(kategori: str) -> int:
    if pd.isna(kategori):
        return 0
    return 1 if "+" in str(kategori) else 0


def program_jobconnect_flag(kategori: str) -> int:
    if pd.isna(kategori):
        return 0
    s = str(kategori).lower()
    return 1 if "job" in s and "connect" in s else 0


def parse_provinsi_to_region(prov: str) -> str:
    if pd.isna(prov) or str(prov).strip() == "":
        return "Unknown"
    s = str(prov).strip().lower()

    # Very common groupings; (PDF-aligned spirit: simple region bucket)
    jawa = ["dki jakarta", "jakarta", "jawa barat", "jawa tengah", "jawa timur", "banten", "di yogyakarta", "yogyakarta"]
    sumatra = ["aceh", "sumatera utara", "sumatera barat", "riau", "kepulauan riau", "jambi", "sumatera selatan", "bengkulu", "lampung", "babel", "kep. bangka belitung", "bangka belitung"]
    kalimantan = ["kalimantan barat", "kalimantan tengah", "kalimantan selatan", "kalimantan timur", "kalimantan utara"]
    sulawesi = ["sulawesi utara", "sulawesi tengah", "sulawesi selatan", "sulawesi tenggara", "sulawesi barat", "gorontalo"]
    bali_nt = ["bali", "nusa tenggara barat", "ntb", "nusa tenggara timur", "ntt"]
    maluku_papua = ["maluku", "maluku utara", "papua", "papua barat", "papua selatan", "papua tengah", "papua pegunungan", "papua barat daya"]

    def hit(lst):
        return any(k in s for k in lst)

    if hit(jawa):
        return "Jawa"
    if hit(sumatra):
        return "Sumatra"
    if hit(kalimantan):
        return "Kalimantan"
    if hit(sulawesi):
        return "Sulawesi"
    if hit(bali_nt):
        return "Bali/NT"
    if hit(maluku_papua):
        return "Maluku/Papua"
    return "Lainnya"


def map_domain_pendidikan(jurusan: str) -> str:
    if pd.isna(jurusan) or str(jurusan).strip() == "":
        return "Unknown"
    s = str(jurusan).lower()

    if any(k in s for k in ["informatika", "computer", "teknik komputer", "sistem informasi", "ilmu komputer", "software", "data", "statistik"]):
        return "STEM/Tech"
    if any(k in s for k in ["ekonomi", "manajemen", "akuntansi", "bisnis", "finance", "keuangan", "marketing"]):
        return "Business"
    if any(k in s for k in ["psikologi", "sosiologi", "ilmu komunikasi", "hukum", "hubungan internasional", "politik"]):
        return "Social/Comm/Law"
    if any(k in s for k in ["kedokteran", "keperawatan", "farmasi", "kesehatan"]):
        return "Health"
    if any(k in s for k in ["pendidikan", "keguruan"]):
        return "Education"
    return "Other"


def map_domain_product(product: str) -> str:
    if pd.isna(product) or str(product).strip() == "":
        return "Unknown"
    s = str(product).lower()
    if any(k in s for k in ["data", "machine learning", "ai", "analytics", "science"]):
        return "Data/AI"
    if any(k in s for k in ["web", "frontend", "backend", "fullstack", "golang", "mobile", "android", "ios"]):
        return "Software Eng"
    if any(k in s for k in ["ui", "ux", "product", "pm", "design"]):
        return "Product/Design"
    if any(k in s for k in ["marketing", "digital marketing", "seo", "growth"]):
        return "Marketing"
    if any(k in s for k in ["finance", "accounting", "audit"]):
        return "Finance"
    if any(k in s for k in ["hr", "human resource"]):
        return "HR"
    return "Other"


def map_motivasi_cluster(text: str) -> str:
    # PDF: group motivasi into 3 clusters by keywords (approx alignment)
    if pd.isna(text) or str(text).strip() == "":
        return "Unknown"
    s = str(text).lower()

    # Cluster A: Career Growth / Skill upgrade
    if any(k in s for k in ["upgrade", "upskill", "skill", "karier", "career", "naik", "promosi", "improve", "belajar", "learn", "switch", "pindah"]):
        return "Career Growth / Upskill"

    # Cluster B: Job/Placement oriented
    if any(k in s for k in ["kerja", "job", "penyaluran", "placement", "diterima", "lolos", "interview", "rekrut", "rekrutmen"]):
        return "Job / Placement"

    # Cluster C: Curiosity / Exploration / Community
    if any(k in s for k in ["penasaran", "curious", "explore", "coba", "komunitas", "community", "network", "teman", "lingkungan"]):
        return "Exploration / Community"

    return "Other"


def compute_engagement_level(community_flag: int, event_flag: int) -> str:
    if community_flag == 1 and event_flag == 1:
        return "High"
    if community_flag == 1 or event_flag == 1:
        return "Medium"
    return "Low"


def flag_misalignment(row: pd.Series) -> int:
    """
    PDF snippet:
      - Motivasi_risk_flag = 1 kalau misalignment:
        (is_switcher=1 & motivasi=Upskill) OR (is_switcher=0 & motivasi=Job/Placement)
      - plus: engagement low -> risk++
    Kita implement versi yang aman (sama spiritnya).
    """
    motiv = str(row.get("Motivasi_cluster", "Unknown"))
    is_sw = int(row.get("is_switcher", 0) or 0)
    eng = str(row.get("Engagement_level", "Low"))

    risk = 0
    if is_sw == 1 and "Upskill" in motiv:
        risk = 1
    if is_sw == 0 and ("Job" in motiv or "Placement" in motiv):
        risk = 1
    if eng == "Low":
        risk = 1
    return risk


def persona_name_from_cluster(cluster_id: int) -> str:
    # PDF: fixed mapping
    persona_map = {
        0: "Fresh Graduate Explorer",
        1: "Working Professional Upskiller",
        2: "High Engagement Career Switcher",
    }
    return persona_map.get(int(cluster_id), f"Cluster {cluster_id}")


# ----------------------------
# Data loading + FE
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def build_feature_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Standardize some expected cols (if exist)
    col_email = _col(df, "Email")
    col_umur = _col(df, "Umur")
    col_batch = _col(df, "Batch")
    col_kategori = _col(df, "Kategori")
    col_product = _col(df, "Product")
    col_prov = _col(df, "Provinsi")
    col_community = _col(df, "Community")
    col_event = _col(df, "Pernah ikut acara dibimbing/tidak")
    col_jurusan = _col(df, "Jurusan pendidikan")
    col_motivasi = _col(df, "Motivasi")
    col_target = _col(df, "Penyaluran Kerja")

    # Minimal clean
    if col_email:
        df[col_email] = df[col_email].astype(str).str.strip()

    # Age
    if col_umur:
        df["Umur"] = df[col_umur].apply(parse_age_to_int)
    else:
        df["Umur"] = None
    df["Umur_bin"] = df["Umur"].apply(age_bin)

    # Region
    if col_prov:
        df["Region"] = df[col_prov].apply(parse_provinsi_to_region)
    else:
        df["Region"] = "Unknown"

    # Batch
    if col_batch:
        df["Batch_num"] = df[col_batch].apply(parse_batch_num)
    else:
        df["Batch_num"] = None

    # Flags from Kategori
    if col_kategori:
        df["Batch_has_plus"] = df[col_kategori].apply(batch_has_plus).astype(int)
        df["Program_jobconnect_flag"] = df[col_kategori].apply(program_jobconnect_flag).astype(int)
    else:
        df["Batch_has_plus"] = 0
        df["Program_jobconnect_flag"] = 0

    # Community/Event flags
    if col_community:
        df["Community_flag"] = df[col_community].fillna("").astype(str).str.lower().isin(["ya", "yes", "1", "true"]).astype(int)
    else:
        df["Community_flag"] = 0

    if col_event:
        df["Event_flag"] = df[col_event].fillna("").astype(str).str.lower().isin(["pernah", "ya", "yes", "1", "true"]).astype(int)
    else:
        df["Event_flag"] = 0

    df["Engagement_level"] = [
        compute_engagement_level(c, e) for c, e in zip(df["Community_flag"], df["Event_flag"])
    ]

    # Domain features
    if col_jurusan:
        df["Domain_pendidikan"] = df[col_jurusan].apply(map_domain_pendidikan)
    else:
        df["Domain_pendidikan"] = "Unknown"

    if col_product:
        df["Domain_product"] = df[col_product].apply(map_domain_product)
    else:
        df["Domain_product"] = "Unknown"

    # Motivasi cluster (from one column; if you have multiple motivasi cols, you can concat here)
    if col_motivasi:
        df["Motivasi_cluster"] = df[col_motivasi].apply(map_motivasi_cluster)
    else:
        df["Motivasi_cluster"] = "Unknown"

    # Switcher (PDF: mismatch Domain_pendidikan vs Domain_product)
    def _is_switcher(row):
        a = str(row.get("Domain_pendidikan", "Unknown"))
        b = str(row.get("Domain_product", "Unknown"))
        if a in ["Unknown"] or b in ["Unknown"]:
            return 0
        return 1 if a != b else 0

    df["is_switcher"] = df.apply(_is_switcher, axis=1).astype(int)

    # Segmen_karir (PDF: just based on switcher)
    df["Segmen_karir"] = df["is_switcher"].map(
        {0: "Working Professional Upskiller", 1: "High Engagement Career Switcher"}
    )

    # Risk flag
    df["Motivasi_risk_flag"] = df.apply(flag_misalignment, axis=1).astype(int)

    # Target (Placement) — PDF-aligned concept:
    # Penyaluran Kerja string -> Penyaluran_flag binary
    # IMPORTANT: adapt to your raw labels
    if col_target:
        def map_penyaluran_to_flag(x):
            if pd.isna(x):
                return 0
            s = str(x).strip().lower()
            # treat "1" / "ya" / "placed" / "tersalurkan" as positive
            if s in ["1", "ya", "yes", "placed", "tersalurkan", "tersalurkan kerja", "lolos", "diterima"]:
                return 1
            # if numeric-ish
            if re.fullmatch(r"\d+", s):
                return 1 if int(s) == 1 else 0
            # otherwise default negative
            return 0

        df["Penyaluran_flag"] = df[col_target].apply(map_penyaluran_to_flag).astype(int)
    else:
        df["Penyaluran_flag"] = 0

    # Add label
    df["Penyaluran_label"] = df["Penyaluran_flag"].map({0: "Tidak", 1: "Ya"})

    return df


# ----------------------------
# Filtering helpers
# ----------------------------
def apply_global_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for col, selected in filters.items():
        if not selected:
            continue
        if col not in out.columns:
            continue
        out = out[out[col].astype(str).isin([str(x) for x in selected])]
    return out


def summarize_target(df: pd.DataFrame, target_col: str) -> Tuple[int, int, float]:
    n = len(df)
    pos = int(df[target_col].sum()) if n else 0
    neg = int(n - pos)
    rate = (pos / n * 100.0) if n else 0.0
    return n, pos, rate


# ----------------------------
# Visual helpers (Altair)
# ----------------------------
def bar_target_rate(df_grouped: pd.DataFrame, group_col: str, rate_col: str, title: str) -> alt.Chart:
    base = alt.Chart(df_grouped).mark_bar().encode(
        y=alt.Y(f"{group_col}:N", sort="-x", title=None),
        x=alt.X(f"{rate_col}:Q", title="Target rate (%)"),
        tooltip=[group_col, "total", "positives", alt.Tooltip(rate_col, format=".2f")],
    ).properties(title=title, height=min(650, 24 * max(5, len(df_grouped))))

    text = alt.Chart(df_grouped).mark_text(align="left", dx=5).encode(
        y=alt.Y(f"{group_col}:N", sort="-x"),
        x=alt.X(f"{rate_col}:Q"),
        text=alt.Text(f"{rate_col}:Q", format=".1f"),
    )

    return (base + text)


def dist_target_chart(df: pd.DataFrame, target_col: str, title="Distribusi target") -> alt.Chart:
    tmp = df[target_col].value_counts(dropna=False).rename_axis("target").reset_index(name="count")
    tmp["target"] = tmp["target"].astype(str)
    return alt.Chart(tmp).mark_bar().encode(
        x=alt.X("target:N", title=target_col),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["target", "count"],
    ).properties(title=title, height=280)


def roc_curve_chart(y_true, y_proba) -> Optional[alt.Chart]:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        return alt.Chart(roc_df).mark_line().encode(
            x=alt.X("fpr:Q", title="False Positive Rate"),
            y=alt.Y("tpr:Q", title="True Positive Rate"),
            tooltip=["fpr", "tpr"],
        ).properties(title="ROC Curve", height=300)
    except Exception:
        return None


def pr_curve_chart(y_true, y_proba) -> Optional[alt.Chart]:
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        pr_df = pd.DataFrame({"precision": prec, "recall": rec})
        return alt.Chart(pr_df).mark_line().encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            tooltip=["recall", "precision"],
        ).properties(title="Precision-Recall Curve", height=300)
    except Exception:
        return None


# ----------------------------
# Clustering (PDF-aligned)
# ----------------------------
@dataclass
class ClusterArtifacts:
    kmeans: MiniBatchKMeans
    preprocess: ColumnTransformer
    svd2: TruncatedSVD
    df_out: pd.DataFrame
    svd2_points: pd.DataFrame


def run_clustering(df: pd.DataFrame) -> ClusterArtifacts:
    # PDF: X = fe_cols
    fe_cols = [
        "Umur", "Umur_bin", "Region", "Batch_num", "Batch_has_plus",
        "Community_flag", "Event_flag", "Engagement_level", "Program_jobconnect_flag",
        "Motivasi_cluster", "Motivasi_risk_flag", "Domain_pendidikan", "Domain_product",
        "is_switcher"
    ]
    use_cols = [c for c in fe_cols if c in df.columns]
    X = df[use_cols].copy()

    # Separate numeric/categorical like PDF approach
    num_cols = [c for c in ["Umur", "Batch_num", "Batch_has_plus", "Community_flag", "Event_flag", "Program_jobconnect_flag", "Motivasi_risk_flag", "is_switcher"] if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", _make_ohe())]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_enc = preprocess.fit_transform(X)

    # PDF: MiniBatchKMeans(k=3, random_state=42, batch_size=512, n_init=10)
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=512, n_init=10)
    cluster_id = kmeans.fit_predict(X_enc)

    # 2D viz: TruncatedSVD(2, random_state=42)
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X_svd2 = svd2.fit_transform(X_enc)

    df_out = df.copy()
    df_out["cluster_id"] = cluster_id
    df_out["Persona"] = df_out["cluster_id"].apply(persona_name_from_cluster)

    svd2_points = pd.DataFrame({"SVD1": X_svd2[:, 0], "SVD2": X_svd2[:, 1], "Persona": df_out["Persona"].values})
    return ClusterArtifacts(kmeans=kmeans, preprocess=preprocess, svd2=svd2, df_out=df_out, svd2_points=svd2_points)


# ----------------------------
# Supervised (PDF-aligned)
# ----------------------------
@dataclass
class SupervisedArtifacts:
    best_model: Pipeline
    best_params: dict
    pr_auc_cv: float
    metrics: dict
    scored_df: pd.DataFrame
    breakdown_df: pd.DataFrame


def eval_model_at_k(y_true: np.ndarray, y_proba: np.ndarray, k_list=(20, 50, 100, 200)) -> pd.DataFrame:
    df_score = pd.DataFrame({"y_true": y_true, "proba": y_proba}).sort_values("proba", ascending=False).reset_index(drop=True)
    base_rate = df_score["y_true"].mean() if len(df_score) else 0.0
    rows = []
    for k in k_list:
        k = int(min(k, len(df_score)))
        top = df_score.head(k)
        hit = top["y_true"].sum()
        precision_at_k = (hit / k) if k else 0.0
        recall_at_k = (hit / df_score["y_true"].sum()) if df_score["y_true"].sum() else 0.0
        lift = (precision_at_k / base_rate) if base_rate > 0 else np.nan
        rows.append({
            "K": k,
            "positives_captured": int(hit),
            "precision_at_k": float(precision_at_k),
            "recall_at_k": float(recall_at_k),
            "lift": float(lift) if np.isfinite(lift) else np.nan,
        })
    return pd.DataFrame(rows)


def run_supervised(df: pd.DataFrame, target_col: str, include_cluster_id: bool, test_size: float, k_capacity: int) -> SupervisedArtifacts:
    # Candidate supervised features (PDF UI list vibe)
    # We'll take engineered + common business cols if exist.
    cand = [
        "Umur", "Umur_bin", "Region", "Batch_num", "Batch_has_plus", "Community_flag", "Event_flag",
        "Engagement_level", "Program_jobconnect_flag", "Motivasi_cluster", "Motivasi_risk_flag",
        "Domain_pendidikan", "Domain_product", "is_switcher", "Segmen_karir",
        "Product", "Kategori", "Month", "Channel", "Level pendidikan", "Kategori Kesibukan",
        "Level Pekerjaan", "Kategori Pekerjaan", "Domisili", "Provinsi", "Negara",
    ]
    if include_cluster_id and "cluster_id" in df.columns:
        cand.append("cluster_id")

    feat_cols = [c for c in cand if c in df.columns]
    X = df[feat_cols].copy()
    y = df[target_col].astype(int).values

    # preprocess
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Fill NAs (avoid ValueError from transformers)
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", _make_ohe())]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Train/test split (safe for rare positives)
    # If too few positives after filters, fallback: train on full, limited eval.
    pos_count = int(np.sum(y))
    can_split = pos_count >= 5 and len(y) >= 50  # heuristic safe

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y  # eval on train (warn later)

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    # PDF: GridSearchCV over C, scoring=average_precision, StratifiedKFold(3, shuffle, rs=42)
    param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # If cannot split / too few positives, CV can break. Guard it.
    best_params = {}
    pr_auc_cv = float("nan")
    best_model = pipe

    try:
        if pos_count >= 10:
            gs = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="average_precision",
                cv=cv,
                n_jobs=-1
            )
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            pr_auc_cv = float(gs.best_score_)
        else:
            # Fit without tuning
            best_model.fit(X_train, y_train)
            best_params = {"note": "CV skipped (positives too few after filtering)"}
            pr_auc_cv = float("nan")
    except Exception as e:
        best_model.fit(X_train, y_train)
        best_params = {"note": f"CV skipped due to error: {type(e).__name__}"}
        pr_auc_cv = float("nan")

    # Evaluate
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # PR-AUC + ROC-AUC (report both; PR-AUC is main for imbalanced)
    pr_auc = average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")

    # Threshold metrics
    y_hat = (y_proba >= 0.5).astype(int)
    metrics = {
        "n_total_eval": int(len(y_test)),
        "pos_eval": int(np.sum(y_test)),
        "base_rate_eval": float(np.mean(y_test)) if len(y_test) else 0.0,
        "pr_auc": float(pr_auc) if np.isfinite(pr_auc) else np.nan,
        "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else np.nan,
        "accuracy@0.5": float(accuracy_score(y_test, y_hat)) if len(y_test) else np.nan,
        "precision@0.5": float(precision_score(y_test, y_hat, zero_division=0)) if len(y_test) else np.nan,
        "recall@0.5": float(recall_score(y_test, y_hat, zero_division=0)) if len(y_test) else np.nan,
        "f1@0.5": float(f1_score(y_test, y_hat, zero_division=0)) if len(y_test) else np.nan,
    }

    # Score full data for Top-K action list
    full_proba = best_model.predict_proba(X)[:, 1]
    scored_df = df.copy()
    scored_df["proba_penyaluran"] = full_proba
    scored_df = scored_df.sort_values("proba_penyaluran", ascending=False).reset_index(drop=True)

    # Breakdown Top-K vs baseline by Persona / Segmen_karir etc
    K = int(min(k_capacity, len(scored_df)))
    topk_df = scored_df.head(K).copy()

    def breakdown(group_col: str) -> pd.DataFrame:
        if group_col not in scored_df.columns:
            return pd.DataFrame()
        base = scored_df.groupby(group_col)[target_col].agg(["count", "sum"]).reset_index()
        base.columns = [group_col, "baseline_total", "baseline_pos"]
        base["baseline_rate_pct"] = (base["baseline_pos"] / base["baseline_total"] * 100.0).fillna(0.0)

        top = topk_df.groupby(group_col)[target_col].agg(["count", "sum"]).reset_index()
        top.columns = [group_col, "topk_total", "topk_pos"]
        top["topk_rate_pct"] = (top["topk_pos"] / top["topk_total"] * 100.0).fillna(0.0)

        out = base.merge(top, on=group_col, how="left").fillna(0)
        out["lift"] = np.where(out["baseline_rate_pct"] > 0, out["topk_rate_pct"] / out["baseline_rate_pct"], np.nan)
        out = out.sort_values("lift", ascending=False)
        return out

    breakdown_df = breakdown("Persona") if "Persona" in scored_df.columns else pd.DataFrame()

    return SupervisedArtifacts(
        best_model=best_model,
        best_params=best_params,
        pr_auc_cv=pr_auc_cv,
        metrics=metrics,
        scored_df=scored_df,
        breakdown_df=breakdown_df,
    )


# ----------------------------
# App UI
# ----------------------------
st.title("Persona Segmentation & Placement Prediction (PDF-aligned)")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload file (CSV)", type=["csv"], key="uploader_csv")

    st.divider()
    st.header("Global filters")
    st.caption("Pilih kolom untuk filter → pilih value-nya. (Ini mempengaruhi EDA/Clustering/Supervised)")

df_raw = None
if uploaded:
    df_raw = load_csv(uploaded)

if df_raw is None:
    st.info("Upload CSV dulu ya (contoh: raw_data.csv).")
    st.stop()

df_fe = build_feature_table(df_raw)

# Sidebar filter builder
with st.sidebar:
    # choose which columns are filterable (like your UI)
    default_filter_cols = [c for c in ["Product", "Kategori", "Month", "Segmen_karir", "Persona", "Region"] if c in df_fe.columns]
    filter_cols = st.multiselect(
        "Pilih kolom untuk filter",
        options=sorted(df_fe.columns.tolist()),
        default=default_filter_cols,
        key="filter_cols",
    )

    filters: Dict[str, List[str]] = {}
    for c in filter_cols:
        # limit unique values
        vals = sorted(df_fe[c].dropna().astype(str).unique().tolist())
        # avoid crazy list
        if len(vals) > 200:
            st.caption(f"⚠️ {c}: terlalu banyak unique ({len(vals)}). Tampilkan sebagian.")
            vals = vals[:200]
        sel = st.multiselect(f"{c}", options=vals, default=[], key=f"filter_{c}")
        if sel:
            filters[c] = sel

    st.divider()
    st.header("Target setup (match PDF)")
    target_col = "Penyaluran_flag"
    if target_col not in df_fe.columns:
        st.error("Kolom target Penyaluran_flag tidak terbentuk. Cek kolom 'Penyaluran Kerja' di CSV kamu.")
        st.stop()
    st.write("Target column: **Penyaluran Kerja (Penyaluran_flag)**")
    st.caption("Tujuan: prediksi peluang penyaluran kerja (placement), bukan sekadar tertarik/tidak.")

df = apply_global_filters(df_fe, filters)

# Quick KPIs
n, pos, rate = summarize_target(df, target_col)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows", f"{n:,}")
c2.metric("Positives (Penyaluran=1)", f"{pos:,}")
c3.metric("Negatives", f"{(n-pos):,}")
c4.metric("Positive rate", f"{rate:.2f}%")

tabs = st.tabs(["Overview", "EDA (Target-driven)", "Clustering (Persona)", "Supervised (Top-K Ranking)", "Dashboard Akhir (Bisnis)"])

# ----------------------------
# Overview
# ----------------------------
with tabs[0]:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Preview data (setelah feature engineering + filter)")
        st.dataframe(df.head(25), use_container_width=True)
    with right:
        st.subheader("Distribusi target")
        st.altair_chart(dist_target_chart(df, target_col), use_container_width=True)

        st.subheader("Catatan alignment ke tujuan proyek")
        st.markdown(
            """
- Fokus target: **Penyaluran kerja** (placement).  
- EDA dibuat **target-driven** (lihat pola yang meningkatkan/menurunkan peluang penyaluran).  
- Clustering: **3 persona** (PDF-aligned) untuk segmentasi yang “actionable”.  
- Supervised: tuning pakai **PR-AUC (Average Precision)** karena data timpang, **tetap report ROC-AUC**.
"""
        )

# ----------------------------
# EDA Target-driven
# ----------------------------
with tabs[1]:
    st.subheader("Target rate by group (Top-N) — PDF-aligned spirit")

    colA, colB, colC = st.columns([1.2, 0.8, 0.8])
    with colA:
        group_col = st.selectbox(
            "Group by",
            options=[c for c in ["Product", "Kategori", "Month", "Region", "Segmen_karir", "Engagement_level", "Motivasi_cluster", "Domain_product", "Domain_pendidikan"] if c in df.columns],
            index=0,
            key="eda_group_col",
        )
    with colB:
        min_n = st.number_input("Min group size", min_value=1, value=30, step=5, key="eda_min_n")
    with colC:
        top_n = st.number_input("Show top-N groups", min_value=5, value=40, step=5, key="eda_top_n")

    g = (
        df.groupby(group_col)[target_col]
        .agg(total="count", positives="sum")
        .reset_index()
    )
    g["target_rate_pct"] = (g["positives"] / g["total"] * 100.0).fillna(0.0)

    g = g[g["total"] >= int(min_n)].sort_values("target_rate_pct", ascending=False).head(int(top_n))
    g = g.rename(columns={group_col: "Group"})
    g_disp = g.rename(columns={"Group": group_col})

    st.altair_chart(bar_target_rate(g_disp, group_col, "target_rate_pct", f"Target rate by {group_col}"), use_container_width=True)
    st.dataframe(g_disp.reset_index(drop=True), use_container_width=True)

# ----------------------------
# Clustering
# ----------------------------
with tabs[2]:
    st.subheader("Persona clustering (3 cluster, PDF-aligned)")

    run_btn = st.button("Run clustering", type="primary", key="btn_run_cluster")

    if run_btn:
        with st.spinner("Running clustering..."):
            try:
                cl = run_clustering(df)
                st.session_state["cluster_artifacts"] = cl
                st.success("Clustering selesai (3 persona).")
            except Exception as e:
                st.error(f"Clustering gagal: {type(e).__name__}: {e}")

    cl: Optional[ClusterArtifacts] = st.session_state.get("cluster_artifacts")

    if cl is None:
        st.info("Klik **Run clustering** dulu.")
    else:
        df_c = cl.df_out
        st.session_state["df_clustered"] = df_c

        cL, cR = st.columns([1.4, 1])
        with cL:
            st.subheader("Cluster visualization (TruncatedSVD 2D) — Persona Named")
            chart = alt.Chart(cl.svd2_points).mark_circle(size=40, opacity=0.6).encode(
                x=alt.X("SVD1:Q"),
                y=alt.Y("SVD2:Q"),
                color=alt.Color("Persona:N"),
                tooltip=["Persona", "SVD1", "SVD2"],
            ).properties(height=520)
            st.altair_chart(chart, use_container_width=True)

        with cR:
            st.subheader("Cluster profiling (ringkas)")
            prof_cols = [c for c in ["Persona", "Segmen_karir", "Engagement_level", "Motivasi_cluster", "Domain_product", "Region"] if c in df_c.columns]
            if prof_cols:
                prof = df_c.groupby("Persona")[target_col].agg(total="count", positives="sum").reset_index()
                prof["target_rate_pct"] = (prof["positives"] / prof["total"] * 100.0).fillna(0.0)
                st.dataframe(prof.sort_values("target_rate_pct", ascending=False), use_container_width=True)

                show = st.selectbox("Lihat distribusi untuk kolom:", options=[c for c in prof_cols if c != "Persona"], key="profile_col")
                dist = df_c.groupby(["Persona", show])[target_col].agg(total="count", positives="sum").reset_index()
                dist["target_rate_pct"] = (dist["positives"] / dist["total"] * 100.0).fillna(0.0)
                st.dataframe(dist.sort_values(["Persona", "target_rate_pct"], ascending=[True, False]).head(50), use_container_width=True)
            else:
                st.warning("Tidak ada kolom profiling yang ditemukan di data setelah FE.")

# ----------------------------
# Supervised
# ----------------------------
with tabs[3]:
    st.subheader("Supervised ranking — Top-K (PDF-aligned)")

    # Need clustered df if include cluster_id toggle is on
    df_sup = st.session_state.get("df_clustered", df)

    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
    with c1:
        k_capacity = st.number_input("Business capacity K (berapa peserta diintervensi)", min_value=10, value=200, step=10, key="sup_k")
    with c2:
        test_size = st.slider("test_size (holdout)", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="sup_test_size")
    with c3:
        include_cluster_id = st.checkbox("Use cluster_id as feature (if available)", value=True, key="sup_use_cluster")
    with c4:
        run_sup = st.button("Run supervised ranking", type="primary", key="btn_run_sup")

    if run_sup:
        with st.spinner("Training supervised model..."):
            try:
                sup = run_supervised(
                    df=df_sup,
                    target_col=target_col,
                    include_cluster_id=include_cluster_id,
                    test_size=float(test_size),
                    k_capacity=int(k_capacity),
                )
                st.session_state["sup_artifacts"] = sup
                st.success("Supervised selesai.")
            except Exception as e:
                st.error(f"Supervised gagal: {type(e).__name__}: {e}")

    sup: Optional[SupervisedArtifacts] = st.session_state.get("sup_artifacts")

    if sup is None:
        st.info("Klik **Run supervised ranking** dulu.")
    else:
        m = sup.metrics

        # Warn if eval is train-on-train due to rare positives
        if m["pos_eval"] < 5:
            st.warning(
                "Positives di evaluasi sangat sedikit (atau tidak bisa split). "
                "Hasil metrik bisa tidak stabil. Coba longgarkan filter / pakai data lebih banyak."
            )

        st.subheader("Model summary (PR-AUC utama, ROC-AUC tetap dilaporkan)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("PR-AUC (eval)", f"{m['pr_auc']:.3f}" if np.isfinite(m["pr_auc"]) else "NA")
        mc2.metric("ROC-AUC (eval)", f"{m['roc_auc']:.3f}" if np.isfinite(m["roc_auc"]) else "NA")
        mc3.metric("Base rate (eval)", f"{m['base_rate_eval']*100:.2f}%")
        mc4.metric("Best CV PR-AUC", f"{sup.pr_auc_cv:.3f}" if np.isfinite(sup.pr_auc_cv) else "NA")

        st.caption(f"Best params: {sup.best_params}")

        # Curves
        # Build curves from eval set if possible
        # We reconstruct eval y_true/y_proba by scoring again on a split-like approximation:
        # For simplicity, we show curves from top of scored_df vs target if available.
        # If you need exact holdout curves, keep X_test/y_test objects (optional).
        # Here we do approximate curves using full scored_df.
        y_true_full = sup.scored_df[target_col].astype(int).values
        y_proba_full = sup.scored_df["proba_penyaluran"].values

        cc1, cc2 = st.columns(2)
        with cc1:
            prc = pr_curve_chart(y_true_full, y_proba_full)
            if prc is not None:
                st.altair_chart(prc, use_container_width=True)
        with cc2:
            rocc = roc_curve_chart(y_true_full, y_proba_full)
            if rocc is not None:
                st.altair_chart(rocc, use_container_width=True)

        st.subheader("Top-K capture & lift (actionable)")
        topk_eval = eval_model_at_k(y_true_full, y_proba_full, k_list=(20, 50, 100, int(k_capacity)))
        st.dataframe(topk_eval, use_container_width=True)

        st.subheader("Top-K list (untuk outreach/intervensi)")
        show_cols = [c for c in ["Email", "Nama", "Product", "Kategori", "Month", "Region", "Persona", "Segmen_karir", "Engagement_level", "Motivasi_cluster", "proba_penyaluran", target_col] if c in sup.scored_df.columns]
        st.dataframe(sup.scored_df.head(int(k_capacity))[show_cols], use_container_width=True)

        if not sup.breakdown_df.empty:
            st.subheader("Breakdown Top-K vs baseline by Persona")
            st.dataframe(sup.breakdown_df, use_container_width=True)

# ----------------------------
# Final Business Dashboard
# ----------------------------
with tabs[4]:
    st.subheader("Dashboard akhir (jawab tujuan proyek: segmentasi + prediksi + keputusan bisnis)")

    sup: Optional[SupervisedArtifacts] = st.session_state.get("sup_artifacts")
    df_dash = st.session_state.get("df_clustered", df)

    if sup is None:
        st.info("Jalankan dulu tab **Supervised (Top-K Ranking)** supaya dashboard akhir terisi.")
    else:
        # KPIs
        K = int(st.session_state.get("sup_k", 200))
        scored = sup.scored_df.copy()
        topk = scored.head(K).copy()

        base_rate = scored[target_col].mean() if len(scored) else 0.0
        topk_rate = topk[target_col].mean() if len(topk) else 0.0
        lift = (topk_rate / base_rate) if base_rate > 0 else float("nan")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Population base rate", f"{base_rate*100:.2f}%")
        d2.metric(f"Top-{K} rate", f"{topk_rate*100:.2f}%")
        d3.metric("Lift (Top-K vs base)", f"{lift:.2f}" if np.isfinite(lift) else "NA")
        d4.metric("PR-AUC (eval)", f"{sup.metrics['pr_auc']:.3f}" if np.isfinite(sup.metrics["pr_auc"]) else "NA")

        st.divider()

        # Persona insights
        st.subheader("Persona insights (segmentasi yang actionable)")
        if "Persona" in scored.columns:
            persona_tbl = scored.groupby("Persona")[target_col].agg(total="count", positives="sum").reset_index()
            persona_tbl["rate_pct"] = (persona_tbl["positives"] / persona_tbl["total"] * 100.0).fillna(0.0)

            top_persona = topk.groupby("Persona")[target_col].agg(total="count", positives="sum").reset_index()
            top_persona["topk_rate_pct"] = (top_persona["positives"] / top_persona["total"] * 100.0).fillna(0.0)

            merged = persona_tbl.merge(top_persona, on="Persona", how="left", suffixes=("_base", "_top")).fillna(0)
            merged["lift"] = np.where(merged["rate_pct"] > 0, merged["topk_rate_pct"] / merged["rate_pct"], np.nan)
            merged = merged.sort_values("lift", ascending=False)

            st.dataframe(merged, use_container_width=True)

            # Chart
            chart_df = merged[["Persona", "rate_pct", "topk_rate_pct"]].melt("Persona", var_name="type", value_name="rate")
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("rate:Q", title="Rate (%)"),
                y=alt.Y("Persona:N", sort="-x"),
                color="type:N",
                tooltip=["Persona", "type", alt.Tooltip("rate:Q", format=".2f")],
            ).properties(title="Baseline vs Top-K rate by Persona", height=320)
            st.altair_chart(chart, use_container_width=True)

        st.divider()

        st.subheader("Rekomendasi keputusan bisnis (berdasarkan Top-K & segment lift)")
        st.markdown(
            f"""
**Bagaimana pakai output ini untuk keputusan bisnis:**
1) **Akuisisi**: fokus channel/product/region yang punya **target rate tinggi** di EDA *dan* sering muncul di **Top-{K}**.  
2) **Desain program**: lihat persona mana yang paling “lifted” di Top-K → itu kandidat paling responsif untuk intervensi.  
3) **Intervensi peserta**: gunakan Top-K list sebagai prioritas outreach (CV review, mock interview, matching company), terutama yang:
   - **Engagement_level = Low** tapi proba tinggi → butuh dorongan aktivitas,
   - **Motivasi_risk_flag = 1** → butuh “re-alignment” (career goal / expectation setting).
"""
        )

        # Quick slice: top drivers
        driver_cols = [c for c in ["Product", "Kategori", "Region", "Segmen_karir", "Engagement_level", "Motivasi_cluster"] if c in scored.columns]
        pick = st.selectbox("Lihat lift Top-K vs baseline untuk dimensi:", options=driver_cols, index=0, key="dash_dim")
        base = scored.groupby(pick)[target_col].agg(total="count", positives="sum").reset_index()
        base["base_rate_pct"] = (base["positives"] / base["total"] * 100.0).fillna(0.0)

        top = topk.groupby(pick)[target_col].agg(total="count", positives="sum").reset_index()
        top["topk_rate_pct"] = (top["positives"] / top["total"] * 100.0).fillna(0.0)

        lift_tbl = base.merge(top, on=pick, how="left", suffixes=("_base", "_top")).fillna(0)
        lift_tbl["lift"] = np.where(lift_tbl["base_rate_pct"] > 0, lift_tbl["topk_rate_pct"] / lift_tbl["base_rate_pct"], np.nan)
        lift_tbl = lift_tbl.sort_values("lift", ascending=False)

        st.dataframe(lift_tbl, use_container_width=True)
