# app/project_persona_predict.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Persona Predict", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "raw_data"

TARGET_COL = "Penyaluran_flag"


# ----------------------------
# Utilities (data loading)
# ----------------------------
def list_repo_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])


@st.cache_data(show_spinner=False)
def read_csv_safely(path: Path) -> pd.DataFrame:
    # Try common encodings; adjust if your data uses something else
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # final fallback (let pandas decide)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def file_bytes(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1


# ----------------------------
# Feature engineering (minimal + robust)
# NOTE: PDF menunjukkan df_eda_final dipakai untuk model/clustering. :contentReference[oaicite:11]{index=11}
# Karena kita nggak punya notebook source .ipynb mentah di sini, kita lakukan FE yang:
# - tidak merusak kolom yang sudah ada
# - membuat kolom FE kalau belum ada (best effort)
# ----------------------------
def clean_text_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.replace("\n", " ", regex=False).str.strip()
    return out


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_bins_umur(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Umur" not in out.columns:
        # try infer from common naming
        for cand in ["Age", "umur", "UMUR"]:
            if cand in out.columns:
                out["Umur"] = safe_to_numeric(out[cand])
                break

    if "Umur" in out.columns and "Umur_bin" not in out.columns:
        umur = safe_to_numeric(out["Umur"])
        bins = [-np.inf, 20, 25, 30, 35, 40, np.inf]
        labels = ["<=20", "21-25", "26-30", "31-35", "36-40", "41+"]
        out["Umur_bin"] = pd.cut(umur, bins=bins, labels=labels)

    return out


def ensure_segmen_karir(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Segmen_karir" in out.columns:
        return out

    # best-effort inference
    # if there is is_switcher boolean/int
    if "is_switcher" in out.columns:
        out["Segmen_karir"] = np.where(safe_to_numeric(out["is_switcher"]).fillna(0).astype(int) == 1,
                                       "Career Switcher", "Upskiller")
        return out

    # fallbacks from common column
    for cand in ["Segmen Karir", "segmen_karir", "Status Karir", "status_karir"]:
        if cand in out.columns:
            out["Segmen_karir"] = out[cand]
            return out

    return out


def ensure_program_jobconnect_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Program_jobconnect_flag" in out.columns:
        return out

    # fallback from "Program" or "Job Connector" columns (best effort)
    for cand in ["Program", "program", "Program_flag", "jobconnect", "Job Connector"]:
        if cand in out.columns:
            s = out[cand].astype(str).str.lower()
            out["Program_jobconnect_flag"] = np.where(s.str.contains("jc") | s.str.contains("job"), 1, 0)
            return out

    return out


def ensure_motivasi_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Combine motivation text if those columns exist
    mot_cols = [c for c in ["Motivasi", "Motivasi_1", "Motivasi_2", "Motivasi_3",
                            "Motivasi_Utama", "Alasan", "Reason"] if c in out.columns]
    if mot_cols and "Motivasi_raw_all" not in out.columns:
        out["Motivasi_raw_all"] = (
            out[mot_cols].astype(str).agg(" | ".join, axis=1).str.replace("\n", " ", regex=False).str.strip()
        )

    # If Motivasi_cluster already exists from your preprocessing, keep it.
    # Otherwise, create a simple heuristic cluster so EDA pages can still render.
    if "Motivasi_cluster" not in out.columns:
        if "Motivasi_raw_all" in out.columns:
            s = out["Motivasi_raw_all"].astype(str).str.lower()
            out["Motivasi_cluster"] = np.select(
                [
                    s.str.contains("kerja") | s.str.contains("karir") | s.str.contains("switch"),
                    s.str.contains("skill") | s.str.contains("belajar") | s.str.contains("upskill"),
                    s.str.contains("sertif") | s.str.contains("portfolio") | s.str.contains("cv")
                ],
                ["Karir/Placement", "Upskilling", "Sertifikasi/Portfolio"],
                default="Lainnya"
            )
        else:
            out["Motivasi_cluster"] = "Lainnya"

    # Motivasi_risk_flag used in EDA per PDF :contentReference[oaicite:12]{index=12}
    if "Motivasi_risk_flag" not in out.columns:
        # best-effort: risk tinggi kalau motivasi cenderung "gak jelas/asal"
        if "Motivasi_raw_all" in out.columns:
            s = out["Motivasi_raw_all"].astype(str).str.lower()
            out["Motivasi_risk_flag"] = np.select(
                [
                    s.str.contains("coba") | s.str.contains("iseng") | s.str.contains("ngikut"),
                    s.str.contains("bingung") | s.str.contains("belum") | s.str.contains("tidak tahu"),
                ],
                ["High", "Medium"],
                default="Low"
            )
        else:
            out["Motivasi_risk_flag"] = "Low"

    return out


def build_df_eda_final(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Drop duplicates as in notebook logic :contentReference[oaicite:13]{index=13}
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove Batch_kelas if exists (not used) :contentReference[oaicite:14]{index=14}
    if "Batch_kelas" in df.columns:
        df = df.drop(columns=["Batch_kelas"])

    # Clean text columns that commonly contain \n :contentReference[oaicite:15]{index=15}
    df = clean_text_cols(df, ["Motivasi_raw_all", "Motivasi_cluster", "Motivasi_risk_flag"])

    # Ensure key FE columns used in EDA plots
    df = ensure_bins_umur(df)
    df = ensure_segmen_karir(df)
    df = ensure_program_jobconnect_flag(df)
    df = ensure_motivasi_features(df)

    # Target normalization
    if TARGET_COL in df.columns:
        df[TARGET_COL] = safe_to_numeric(df[TARGET_COL]).fillna(0).astype(int)

    return df


# ----------------------------
# EDA helpers (table + plot style like PDF)
# ----------------------------
def make_freq_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].fillna("Unknown").astype(str).value_counts(dropna=False)
    total = vc.sum()
    out = pd.DataFrame({
        col: vc.index,
        "N": vc.values,
        "Persentase (%)": (vc.values / total * 100.0)
    })
    return out


def plot_freq_with_table(stats: pd.DataFrame, title: str, x_label: str, y_label: str):
    # Align style with notebook: horizontal bar + embedded table :contentReference[oaicite:16]{index=16}
    fig, ax = plt.subplots(figsize=(8, 4))
    y = stats.iloc[:, 0].astype(str).values
    x = stats["Persentase (%)"].values

    ax.barh(y, x)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # value labels
    for i, v in enumerate(x):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

    # embedded table
    tbl = ax.table(
        cellText=[[r, f"{p:.1f}", int(n)] for r, p, n in zip(y, x, stats["N"].values)],
        colLabels=[stats.columns[0], "Persentase (%)", "N"],
        cellLoc="center",
        bbox=[0.62, 0.10, 0.36, 0.80],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    plt.tight_layout()
    st.pyplot(fig)


def plot_placement_rate_barh(df: pd.DataFrame, group_col: str, title: str, xlim_max: float):
    # Pattern appears repeatedly in PDF: groupby -> mean*100 -> barh + table + zoom xlim :contentReference[oaicite:17]{index=17}
    if group_col not in df.columns or TARGET_COL not in df.columns:
        st.warning(f"Kolom '{group_col}' atau target '{TARGET_COL}' tidak ditemukan.")
        return

    pct = (df.groupby(group_col)[TARGET_COL].mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(pct.index.astype(str), pct.values)
    ax.invert_yaxis()
    ax.set_xlim(0, xlim_max)
    ax.set_xlabel("Placement rate (%)")
    ax.set_title(title)

    for i, v in enumerate(pct.values):
        ax.text(v + (xlim_max * 0.02), i, f"{v:.2f}%", va="center", ha="left", fontsize=8)

    tbl = ax.table(
        cellText=[[k, f"{v:.4f}"] for k, v in zip(pct.index.astype(str), pct.values)],
        colLabels=[group_col, "Placement Rate (%)"],
        cellLoc="center",
        bbox=[0.55, 0.10, 0.42, 0.80],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)

    plt.tight_layout()
    st.pyplot(fig)


# ----------------------------
# Clustering (as per PDF pipeline idea)
# ----------------------------
def pick_feature_cols(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return [c for c in preferred if c in df.columns]


def run_clustering(df_eda_final: pd.DataFrame, k_min: int, k_max: int, seed: int) -> Tuple[pd.DataFrame, Dict]:
    # Feature list taken from PDF snippet :contentReference[oaicite:18]{index=18}
    preferred = [
        "Umur", "Level_pendidikan", "Jenjang_pendidikan",
        "Domisili", "Region", "Batch_num", "Batch_plus_flag",
        "Program_jobconnect_flag", "Engagement_score", "Engagement_bin",
        "Produk_utama_FE", "Kategori_utama_FE",
        "Skill_cluster", "Skill_risk_flag",
        "Motivasi_cluster", "Motivasi_risk_flag",
        "Segmen_karir", "is_switcher",
        "Pekerjaan_cluster", "Pekerjaan_risk_flag"
    ]
    fe_cols = pick_feature_cols(df_eda_final, preferred)

    if len(fe_cols) < 4:
        raise ValueError("Kolom FE terlalu sedikit. Pastikan raw_data kamu sudah punya kolom FE / atau sesuaikan mapping FE.")

    df_fe = df_eda_final[fe_cols].copy()

    num_cols = [c for c in ["Umur", "Batch_num", "Engagement_score", "is_switcher"] if c in df_fe.columns]
    cat_cols = [c for c in df_fe.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    k_range = list(range(int(k_min), int(k_max) + 1))
    sil_scores = []

    for k in k_range:
        pipe = Pipeline(steps=[
            ("prep", preprocess),
            ("kmeans", MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024))
        ])
        X = pipe.named_steps["prep"].fit_transform(df_fe)
        labels = pipe.named_steps["kmeans"].fit_predict(X)
        # silhouette on transformed space
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = np.nan
        sil_scores.append(score)

    best_idx = int(np.nanargmax(sil_scores))
    best_k = k_range[best_idx]
    best_sil = float(sil_scores[best_idx])

    final_pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("kmeans", MiniBatchKMeans(n_clusters=best_k, random_state=seed, batch_size=1024))
    ])
    X_all = final_pipe.named_steps["prep"].fit_transform(df_fe)
    cluster_id = final_pipe.named_steps["kmeans"].fit_predict(X_all)

    # 2D visualization (TruncatedSVD) matches what your Streamlit already shows
    svd = TruncatedSVD(n_components=2, random_state=seed)
    emb2 = svd.fit_transform(X_all)

    out = df_eda_final.copy()
    out["cluster_id"] = cluster_id
    out["_svd1"] = emb2[:, 0]
    out["_svd2"] = emb2[:, 1]

    meta = {
        "fe_cols": fe_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "k_range": k_range,
        "sil_scores": sil_scores,
        "best_k": best_k,
        "best_sil": best_sil,
    }
    return out, meta


def plot_silhouette_curve(meta: Dict):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(meta["k_range"], meta["sil_scores"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_title("Silhouette vs k")
    plt.tight_layout()
    st.pyplot(fig)


def plot_cluster_scatter(df: pd.DataFrame):
    if not {"_svd1", "_svd2", "cluster_id"}.issubset(df.columns):
        st.warning("Embedding/cluster belum tersedia.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for cid in sorted(df["cluster_id"].dropna().unique()):
        dd = df[df["cluster_id"] == cid]
        ax.scatter(dd["_svd1"], dd["_svd2"], s=10, alpha=0.7, label=str(cid))
    ax.set_title("Cluster scatter (TruncatedSVD 2D)")
    ax.set_xlabel("SVD1")
    ax.set_ylabel("SVD2")
    ax.legend(title="cluster_id")
    plt.tight_layout()
    st.pyplot(fig)


# ----------------------------
# Supervised ranking (LogReg) + Top-K narrative like PDF
# ----------------------------
def stratified_keep_all_positives(df: pd.DataFrame, target_col: str, sample_n: int, seed: int) -> pd.DataFrame:
    if sample_n <= 0 or sample_n >= len(df):
        return df
    if target_col not in df.columns:
        return df.sample(n=sample_n, random_state=seed)

    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    # Keep all positives, sample negatives to reach sample_n
    need_neg = max(sample_n - len(pos), 0)
    if need_neg <= 0:
        # if positives already exceed sample_n (rare), sample within positives
        return pos.sample(n=sample_n, random_state=seed)
    neg_s = neg.sample(n=min(need_neg, len(neg)), random_state=seed)
    out = pd.concat([pos, neg_s], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_supervised_pipeline(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Pipeline, List[str], List[str]]:
    X = df[feature_cols].copy()

    num_cols = [c for c in ["Umur", "Batch_num", "Engagement_score", "is_switcher"] if c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",   # important for extreme imbalance
        n_jobs=None
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    return pipe, num_cols, cat_cols


def plot_cdf_neg_with_pos(y_true: np.ndarray, y_proba: np.ndarray):
    # Replicate idea from PDF: CDF negative + vertical lines for positives :contentReference[oaicite:19]{index=19}
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    neg = y_proba[y_true == 0]
    pos = y_proba[y_true == 1]

    if len(neg) == 0 or len(pos) == 0:
        st.info("Tidak cukup data untuk plot CDF (butuh negatif & positif di holdout).")
        return

    neg_sorted = np.sort(neg)
    cdf = np.arange(1, len(neg_sorted) + 1) / len(neg_sorted)

    pos_percentiles = [100.0 * (np.searchsorted(neg_sorted, s, side="right") / len(neg_sorted)) for s in pos]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(neg_sorted, cdf, label="CDF skor negatif (y=0)")
    ax.set_xlabel("Predicted probability (y_proba)")
    ax.set_ylabel("CDF (proporsi negatif ≤ skor)")
    ax.set_title("CDF Negatif + Posisi skor positif (percentile among negatives)")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, linestyle=":", linewidth=2, label="threshold 0.5")

    for i, (s, pctl) in enumerate(zip(pos, pos_percentiles), start=1):
        ax.axvline(s, linestyle="--", linewidth=2, label=f"pos#{i} skor={s:.6f} | pct={pctl:.1f}%")

    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Ringkasan percentile skor positif di antara negatif:")
    for i, (s, pctl) in enumerate(zip(pos, pos_percentiles), start=1):
        st.write(f"- pos#{i}: skor={s:.6f} -> percentile among negatives = {pctl:.2f}%")


# ----------------------------
# UI
# ----------------------------
st.title("Persona Predict — EDA • Clustering • Supervised Ranking")

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Choose", ["Repo file", "Upload file (CSV)"], index=0)

    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

    use_sample = st.toggle("Use sample for training/plots?", value=False)
    sample_n = st.number_input("Sample size (keep all positives)", min_value=200, max_value=200000, value=2000, step=100)

    st.divider()

    df_raw = None
    selected_path = None

    if mode == "Repo file":
        files = list_repo_csv_files(DEFAULT_DATA_DIR)
        if not files:
            st.error(f"Tidak ada CSV di {DEFAULT_DATA_DIR}. Pastikan folder raw_data ada di repo.")
        else:
            labels = [str(p.relative_to(REPO_ROOT)) for p in files]
            choice = st.selectbox("Pilih file data di repo:", labels, index=0)
            selected_path = REPO_ROOT / choice
            df_raw = read_csv_safely(selected_path)

            st.caption("Debug file")
            st.write(f"Path: `{choice}`")
            st.write(f"Size (bytes): `{file_bytes(selected_path)}`")

    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df_raw = read_uploaded_csv(up)
            st.caption("Debug file")
            st.write(f"Filename: `{up.name}`")
            st.write(f"Size (bytes): `{up.size}`")

if df_raw is None:
    st.stop()

df_eda_final = build_df_eda_final(df_raw)

# Optional sample but keep all positives (prevents supervised from dying)
df_work = df_eda_final
if use_sample and TARGET_COL in df_eda_final.columns:
    df_work = stratified_keep_all_positives(df_eda_final, TARGET_COL, int(sample_n), int(seed))
elif use_sample:
    df_work = df_eda_final.sample(n=min(int(sample_n), len(df_eda_final)), random_state=int(seed)).reset_index(drop=True)

st.subheader("Preview")
c1, c2 = st.columns([1, 1])
with c1:
    st.write("Shape:", df_work.shape)
with c2:
    st.write(f"Target positives (if exists): {int(df_work[TARGET_COL].sum()) if TARGET_COL in df_work.columns else '—'}")

st.dataframe(df_work.head(20), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["1) EDA", "2) Persona clustering", "3) Supervised ranking (LogReg)"])

# ----------------------------
# EDA
# ----------------------------
with tab1:
    st.markdown("### EDA mengikuti pola di PDF vertopal (motivasi + placement-rate zoom)")

    st.markdown("#### Distribusi motivasi utama peserta")
    if "Motivasi_cluster" in df_work.columns:
        stats_mot = make_freq_table(df_work, "Motivasi_cluster")
        plot_freq_with_table(
            stats_mot,
            title="Distribusi motivasi utama peserta",
            x_label="Persentase peserta (%)",
            y_label="Kelompok motivasi"
        )
    else:
        st.warning("Kolom Motivasi_cluster tidak ada.")

    st.markdown("#### Placement rate per motivasi")
    plot_placement_rate_barh(df_work, "Motivasi_cluster", "Placement rate per motivasi", xlim_max=0.7)

    st.markdown("#### Placement rate per risk level")
    plot_placement_rate_barh(df_work, "Motivasi_risk_flag", "Placement rate per risk level", xlim_max=1.2)

    st.markdown("#### Placement rate per Segmen Karir / Umur / Region / JC (zoom)")
    cA, cB = st.columns(2)
    with cA:
        plot_placement_rate_barh(df_work, "Segmen_karir", "Placement Rate per Segmen Karir (Zoomed)", xlim_max=1.2)
        plot_placement_rate_barh(df_work, "Umur_bin", "Placement Rate per Kelompok Umur (Zoomed)", xlim_max=1.2)
    with cB:
        plot_placement_rate_barh(df_work, "Region", "Placement Rate per Region (Zoomed)", xlim_max=1.2)
        plot_placement_rate_barh(df_work, "Program_jobconnect_flag", "Placement Rate: JC vs Non-JC (Zoomed)", xlim_max=1.2)


# ----------------------------
# Clustering
# ----------------------------
with tab2:
    st.markdown("### Persona clustering (MiniBatchKMeans + TruncatedSVD 2D)")

    k_min = st.number_input("k_min", min_value=2, max_value=20, value=2, step=1)
    k_max = st.number_input("k_max", min_value=2, max_value=20, value=8, step=1)

    if st.button("Run clustering", type="primary"):
        try:
            df_clustered, meta = run_clustering(df_work, int(k_min), int(k_max), int(seed))

            st.success(f"Best k = {meta['best_k']} | silhouette = {meta['best_sil']:.4f}")
            plot_silhouette_curve(meta)

            st.markdown("#### Cluster scatter (TruncatedSVD 2D)")
            plot_cluster_scatter(df_clustered)

            st.markdown("#### Ringkasan ukuran cluster")
            st.dataframe(
                df_clustered["cluster_id"].value_counts().rename_axis("cluster_id").reset_index(name="N"),
                use_container_width=True
            )

            st.markdown("#### Contoh data per cluster")
            show_cols = [c for c in ["cluster_id", "Motivasi_cluster", "Motivasi_risk_flag", "Segmen_karir", "Umur", TARGET_COL] if c in df_clustered.columns]
            st.dataframe(df_clustered[show_cols].head(50), use_container_width=True)

        except Exception as e:
            st.error(f"Clustering gagal: {e}")


# ----------------------------
# Supervised ranking
# ----------------------------
with tab3:
    st.markdown("### Supervised ranking (Logistic Regression)")
    st.caption("Mengikuti narasi PDF: model dipakai untuk **ranking Top-K**, bukan threshold tunggal. "
               "Kalau positive sangat sedikit, probability absolut bisa kecil tapi ranking masih berguna.")

    if TARGET_COL not in df_work.columns:
        st.error(f"Target '{TARGET_COL}' tidak ditemukan di data.")
        st.stop()

    # Feature list for supervised should match df_eda_final modeling columns (exclude raw text/id columns)
    # PDF drops several raw columns before modeling :contentReference[oaicite:20]{index=20}
    drop_candidates = [
        "Penyaluran Kerja", "Status", "Tanggal Gabungan",
        "Motivasi_raw_all", "Skill_raw_all", "Pekerjaan_raw_all",
        "Motivasi_1", "Motivasi_2", "Motivasi_3", "Skill_1", "Skill_2", "Skill_3"
    ]
    df_model = df_work.drop(columns=[c for c in drop_candidates if c in df_work.columns], errors="ignore").copy()

    # Basic feature set: all columns except target + obvious non-features
    exclude = {TARGET_COL}
    feature_cols = [c for c in df_model.columns if c not in exclude and df_model[c].nunique(dropna=False) > 1]

    # Guardrails
    y_all = df_model[TARGET_COL].astype(int)
    pos_count = int(y_all.sum())
    neg_count = int((y_all == 0).sum())
    st.write(f"Total rows: {len(df_model)} | positives: {pos_count} | negatives: {neg_count}")

    if pos_count < 2 or neg_count < 2:
        st.error("Data tidak cukup untuk supervised (butuh minimal 2 positif & 2 negatif).")
        st.stop()

    # choose a test size that guarantees at least 1 positive in test & train
    base_test = 0.2
    min_test = 1.0 / pos_count  # at least 1 positive expected
    test_size = max(base_test, min_test)
    test_size = min(test_size, 0.5)  # keep train >= 50%

    pipe, num_cols, cat_cols = build_supervised_pipeline(df_model, feature_cols)

    if st.button("Run supervised ranking", type="primary"):
        X = df_model[feature_cols].copy()
        y = df_model[TARGET_COL].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(seed),
            stratify=y
        )

        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        st.success(f"Holdout PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | test_size={test_size:.3f}")
        st.info("Interpretasi yang dipakai di PDF: walaupun PR-AUC bisa rendah karena overlap & base rate kecil, "
                "ranking Top-K masih bisa menangkap positif.")

        # Top-K table
        top_k = st.slider("Top-K shortlist", min_value=10, max_value=min(200, len(X_test)), value=min(50, len(X_test)), step=10)
        res = X_test.copy()
        res["y_true"] = y_test.values
        res["y_proba"] = y_proba
        res = res.sort_values("y_proba", ascending=False).head(int(top_k))

        st.markdown("#### Top-K hasil ranking (holdout)")
        st.dataframe(res.reset_index(drop=True), use_container_width=True)

        st.markdown("#### CDF negatif + posisi skor positif (untuk narasi ranking)")
        plot_cdf_neg_with_pos(y_test.values, y_proba)
