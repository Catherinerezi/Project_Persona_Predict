# streamlit_app.py
# Persona Predict — Streamlit (Altair)
# - No Google Drive / requests
# - No LFS requirement
# - EDA + FE mengikuti pola di PDF Vertopal (Motivasi_cluster, risk flag, zoom <1%)
# - Clustering: MiniBatchKMeans + TruncatedSVD 2D (tanpa inertia chart)
# - Supervised: Logistic Regression + Top-K analysis (zoom)

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# App config
# =========================
st.set_page_config(page_title="Persona Predict", layout="wide")
alt.data_transformers.disable_max_rows()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)


# =========================
# Repo scanning (NO LFS logic, just real files)
# =========================
def _repo_root_from_file() -> Path:
    """
    Streamlit Cloud umumnya:
      /mount/src/<repo>/<subfolder>/<script>.py
    Kalau script di /app/, repo_root = parents[1]
    """
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def _is_data_file(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))


def sniff_file_head(path: Path, n_lines: int = 12) -> str:
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return "".join([next(f) for _ in range(n_lines)])
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join([next(f) for _ in range(n_lines)])
    except StopIteration:
        return ""
    except Exception as e:
        return f"[Gagal baca head: {e}]"


def _detect_delimiter_from_head(head: str) -> str:
    """
    PDF/Vertopal biasanya CSV normal.
    Auto delimiter tanpa engine='python' (biar gak kena error low_memory/python).
    """
    if not head.strip():
        return ","
    first_line = head.splitlines()[0]
    candidates = [",", ";", "\t", "|"]
    counts = {c: first_line.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    # kalau semuanya 0, fallback comma
    return best if counts[best] > 0 else ","


def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    # CSV / GZ
    if name.endswith((".csv", ".csv.gz", ".gz")):
        head = sniff_file_head(path, n_lines=5)
        sep = _detect_delimiter_from_head(head)

        # C-engine (stabil) + low_memory boleh
        try:
            return pd.read_csv(
                path,
                sep=sep,
                compression="gzip" if (name.endswith(".csv.gz") or name.endswith(".gz")) else None,
                encoding_errors="replace",
                on_bad_lines="skip",
                low_memory=False,
                engine="c",
            )
        except Exception:
            # fallback python engine TANPA low_memory (biar gak error)
            return pd.read_csv(
                path,
                sep=sep,
                compression="gzip" if (name.endswith(".csv.gz") or name.endswith(".gz")) else None,
                encoding_errors="replace",
                on_bad_lines="skip",
                engine="python",
            )

    raise ValueError(f"Format tidak didukung: {path.name}")


@st.cache_data(show_spinner=True)
def scan_repo_files() -> Tuple[str, str, List[str]]:
    script_path = Path(__file__).resolve()
    repo_root = _repo_root_from_file()

    rels: List[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if not _is_data_file(p):
            continue
        rel = str(p.relative_to(repo_root))

        # skip noise
        if any(part.startswith(".") for part in p.parts):
            continue
        if "venv" in rel or "__pycache__" in rel:
            continue

        rels.append(rel)

    rels = sorted(set(rels))
    return str(script_path), str(repo_root), rels


# =========================
# Column helpers (anti “kolom hilang”)
# =========================
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-_]+", "", s)  # remove spaces/_/-
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def resolve_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Cari kolom berdasarkan kandidat (robust terhadap spasi/underscore/case).
    """
    if df is None or df.empty:
        return None
    mapping: Dict[str, str] = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in mapping:
            return mapping[key]
    return None


# =========================
# Feature Engineering (mengikuti PDF)
# =========================
JABODETABEK = {
    "jakarta",
    "jakartabarat",
    "jakartapusat",
    "jakartaselatan",
    "jakartatimur",
    "jakartautara",
    "bogor",
    "depok",
    "tangerang",
    "tangerangselatan",
    "bekasi",
}


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    PDF: target dari 'Penyaluran Kerja' -> Penyaluran_flag
    """
    df = df.copy()

    if "Penyaluran_flag" in df.columns:
        return df

    col = resolve_col(df, ["Penyaluran Kerja", "penyaluran_kerja", "penyaluran"])
    if col is None:
        return df

    s = df[col].astype(str).fillna("")
    df["Penyaluran_flag"] = s.map(lambda x: 1 if YES_PATTERN.search(x) else 0).astype(int)
    df["Penyaluran_label"] = np.where(df["Penyaluran_flag"] == 1, "Tersalur kerja", "Belum tersalur")
    return df


def fe_core(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    FE inti mengikuti PDF (nama kolom dibuat sama):
    - Umur_bin: <=22, 23–25, 26–30, >30
    - Region: Jabodetabek vs Luar (berdasarkan Kota)
    - Batch_num, Batch_has_plus
    - Community_flag, Event_flag, Engagement_level
    - Program_jobconnect_flag
    - Domain_pendidikan, Domain_product, is_switcher, Segmen_karir
    - Level_pendidikan_FE
    - Motivasi_cluster, Motivasi_risk_flag
    """
    df = df_in.copy()

    # ---- Umur_bin
    umur_col = resolve_col(df, ["Umur", "age"])
    if umur_col and "Umur_bin" not in df.columns:
        umur = pd.to_numeric(df[umur_col], errors="coerce")
        bins = [-np.inf, 22, 25, 30, np.inf]
        labels = ["≤22", "23–25", "26–30", ">30"]
        df["Umur_bin"] = pd.cut(umur, bins=bins, labels=labels)

    # ---- Region dari Kota
    kota_col = resolve_col(df, ["Kota", "domisili", "city"])
    if kota_col and "Region" not in df.columns:
        def _to_region(x: str) -> str:
            t = _norm(str(x))
            return "Jabodetabek" if t in JABODETABEK else "Luar Jabodetabek"
        df["Region"] = df[kota_col].apply(_to_region)

    # ---- Batch
    batch_col = resolve_col(df, ["Batch", "batch_kelas", "Batch_kelas"])
    if batch_col:
        if "Batch_str" not in df.columns:
            df["Batch_str"] = df[batch_col].astype(str)
        if "Batch_num" not in df.columns:
            df["Batch_num"] = (
                df["Batch_str"]
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
                .fillna(0)
                .astype(int)
            )
        if "Batch_has_plus" not in df.columns:
            df["Batch_has_plus"] = df["Batch_str"].str.contains(r"\+", na=False).astype(int)

    # ---- Komunitas / Event -> flags + engagement level
    komunitas_col = resolve_col(df, ["Komunitas", "community", "ikut komunitas"])
    event_col = resolve_col(df, ["Event", "ikut event"])

    def _to_flag_series(colname: str) -> pd.Series:
        s = df[colname].astype(str).str.strip().str.lower()
        return s.isin(["1", "ya", "yes", "true", "ikut", "hadir", "y"]).astype(int)

    if komunitas_col and "Community_flag" not in df.columns:
        df["Community_flag"] = _to_flag_series(komunitas_col)

    if event_col and "Event_flag" not in df.columns:
        df["Event_flag"] = _to_flag_series(event_col)

    # IMPORTANT: robust against NaN / empty / weird values
    if "Engagement_level" not in df.columns:
        if "Community_flag" in df.columns:
            c = pd.to_numeric(df["Community_flag"], errors="coerce").fillna(0).astype(int)
        else:
            c = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        if "Event_flag" in df.columns:
            e = pd.to_numeric(df["Event_flag"], errors="coerce").fillna(0).astype(int)
        else:
            e = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

        df["Engagement_level"] = (c + e).astype(int)

    # ---- Program_jobconnect_flag
    prog_col = resolve_col(df, ["Program", "program", "Program yang diikuti"])
    if prog_col and "Program_jobconnect_flag" not in df.columns:
        df["Program_jobconnect_flag"] = df[prog_col].astype(str).str.lower().str.contains("jobconnect", na=False).astype(int)

    # ---- Domain pendidikan
    jurusan_col = resolve_col(df, ["Jurusan pendidikan", "Jurusan", "major", "jurusan_pendidikan"])
    def map_domain_pendidikan(x: str) -> str:
        t = str(x).lower()
        if any(k in t for k in ["informatika", "teknik komputer", "sistem informasi", "data", "statistik", "matematika"]):
            return "IT/Data"
        if any(k in t for k in ["manajemen", "bisnis", "akuntansi", "keuangan", "ekonomi", "administrasi"]):
            return "Bisnis/Manajemen"
        if any(k in t for k in ["komunikasi", "marketing", "periklanan", "public relation", "broadcast"]):
            return "Marketing/Komunikasi"
        if any(k in t for k in ["desain", "dkv", "arsitektur", "seni rupa", "creative"]):
            return "Design/Creative"
        return "Lainnya"

    if jurusan_col and "Domain_pendidikan" not in df.columns:
        df["Domain_pendidikan"] = df[jurusan_col].apply(map_domain_pendidikan)

    # ---- Domain product
    product_col = resolve_col(df, ["Product", "Produk", "product"])
    def map_domain_product(x: str) -> str:
        t = str(x).lower()
        if "data" in t or "machine learning" in t or "ai" in t:
            return "IT/Data"
        if "cyber" in t or "security" in t:
            return "IT/Data"
        if "ui" in t or "ux" in t or "product design" in t:
            return "Design/Creative"
        if "digital marketing" in t or "growth" in t or "seo" in t:
            return "Marketing/Komunikasi"
        if "product management" in t:
            return "Bisnis/Manajemen"
        return "Lainnya"

    if product_col and "Domain_product" not in df.columns:
        df["Domain_product"] = df[product_col].apply(map_domain_product)

    # ---- is_switcher + Segmen_karir
    if "Domain_pendidikan" in df.columns and "Domain_product" in df.columns:
        if "is_switcher" not in df.columns:
            df["is_switcher"] = (df["Domain_pendidikan"] != df["Domain_product"]).astype(int)
        if "Segmen_karir" not in df.columns:
            df["Segmen_karir"] = df["is_switcher"].map({0: "Upskiller", 1: "Career Switcher"}).fillna("Unknown")

    # ---- Level pendidikan FE
    pendidikan_col = resolve_col(df, ["Pendidikan", "Level pendidikan", "education"])
    def map_level_pendidikan(level: str) -> str:
        t = str(level).strip().lower()
        if t.startswith("low"):
            return "Low"
        if t.startswith("middle"):
            return "Middle"
        if t.startswith("high"):
            return "High"
        if "sma" in t or "smk" in t:
            return "Low"
        if "d3" in t or "d1" in t or "d2" in t:
            return "Middle"
        if "s1" in t or "sarjana" in t:
            return "High"
        if "s2" in t or "magister" in t:
            return "High"
        return "Unknown"

    if pendidikan_col and "Level_pendidikan_FE" not in df.columns:
        df["Level_pendidikan_FE"] = df[pendidikan_col].apply(map_level_pendidikan)

    # ---- Motivasi_raw_all -> Motivasi_cluster
    # PDF: gabung semua kolom yang mengandung "Motivasi"
    if "Motivasi_raw_all" not in df.columns:
        mot_cols = [c for c in df.columns if "motivasi" in _norm(c)]
        if mot_cols:
            df["Motivasi_raw_all"] = (
                df[mot_cols]
                .astype(str)
                .replace({"nan": "", "None": ""})
                .apply(lambda r: " | ".join([x for x in r.tolist() if str(x).strip() not in ["", "nan", "none"]]), axis=1)
            )
        else:
            df["Motivasi_raw_all"] = ""

    def map_motivasi_cluster(x: str) -> str:
        t = str(x).lower()
        # sesuai pola di PDF (kelompok besar)
        if any(k in t for k in ["career", "karir", "switch", "naik level", "promosi", "pekerjaan", "job", "placement"]):
            return "Karir/Placement"
        if any(k in t for k in ["skill", "belajar", "upskill", "reskill", "kompetensi", "portofolio"]):
            return "Upskill/Skill"
        if any(k in t for k in ["sertifikat", "certificate", "ijazah"]):
            return "Sertifikat"
        if any(k in t for k in ["gaji", "salary", "uang", "income"]):
            return "Kenaikan Gaji"
        if any(k in t for k in ["network", "relasi", "komunitas", "teman"]):
            return "Networking"
        return "Lainnya"

    if "Motivasi_cluster" not in df.columns:
        df["Motivasi_cluster"] = df["Motivasi_raw_all"].apply(map_motivasi_cluster)

    # ---- Motivasi_risk_flag (PDF: deteksi risk words)
    risk_words = [
        "gaji", "salary", "uang", "income",
        "sertifikat", "certificate",
        "placement", "job", "kerja", "pekerjaan",
        "cepat", "instan", "jamin", "garansi",
    ]
    if "Motivasi_risk_flag" not in df.columns:
        df["Motivasi_risk_flag"] = df["Motivasi_raw_all"].astype(str).str.lower().apply(
            lambda t: 1 if any(w in t for w in risk_words) else 0
        ).astype(int)

    # ---- Month dari Tanggal Gabungan (opsional, dipakai kalau ada)
    tg_col = resolve_col(df, ["Tanggal Gabungan", "tanggal_gabungan", "date"])
    if tg_col and "Month" not in df.columns:
        d = pd.to_datetime(df[tg_col], errors="coerce")
        df["Month"] = d.dt.to_period("M").astype(str)

    return df


# =========================
# Modeling helpers (sesuai PDF: imputer + scaler + OHE)
# =========================
def split_num_cat(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def make_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def drop_identifier_like(df: pd.DataFrame) -> pd.DataFrame:
    drop = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["email", "e-mail", "nama", "name", "phone", "telepon", "nohp", "no_hp", "hp"]):
            drop.append(c)
        if cl in ["id", "user_id", "id_user", "id_peserta"] or cl.endswith("_id") or cl.startswith("id_"):
            drop.append(c)
        # tanggal mentah dibuang, month boleh
        if ("tanggal" in cl or "date" in cl) and c != "Month":
            drop.append(c)
    return df.drop(columns=sorted(set(drop)), errors="ignore")


# =========================
# EDA charts (ALTair) — mengikuti PDF (zoom <1%)
# =========================
def pct_bar_h(df: pd.DataFrame, cat_col: str, title: str) -> alt.Chart:
    g = df[cat_col].fillna("Unknown").value_counts(dropna=False).reset_index()
    g.columns = [cat_col, "count"]
    g["pct"] = (g["count"] / g["count"].sum()) * 100.0

    return (
        alt.Chart(g)
        .mark_bar()
        .encode(
            y=alt.Y(f"{cat_col}:N", sort="-x", title=None),
            x=alt.X("pct:Q", title="Persentase peserta (%)"),
            tooltip=[alt.Tooltip(cat_col, type="nominal"), alt.Tooltip("count:Q"), alt.Tooltip("pct:Q", format=".2f")],
        )
        .properties(title=title, height=min(420, 28 * max(4, len(g))))
    )


def placement_rate_bar(
    df: pd.DataFrame,
    by_col: str,
    target_col: str = "Penyaluran_flag",
    title: str = "",
    zoom_max_pct: float = 1.2,  # sesuai PDF (bar bawah 1%)
) -> alt.Chart:
    g = df.copy()
    g[by_col] = g[by_col].fillna("Unknown").astype(str)

    agg = (
        g.groupby(by_col, dropna=False)[target_col]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"sum": "placed"})
    )
    agg["placement_rate_pct"] = np.where(agg["count"] > 0, (agg["placed"] / agg["count"]) * 100.0, np.nan)

    base = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            y=alt.Y(f"{by_col}:N", sort=alt.SortField("placement_rate_pct", order="descending"), title=None),
            x=alt.X(
                "placement_rate_pct:Q",
                title="Placement rate (%)",
                scale=alt.Scale(domain=[0, zoom_max_pct]),
            ),
            tooltip=[
                alt.Tooltip(by_col, type="nominal"),
                alt.Tooltip("count:Q", title="Total"),
                alt.Tooltip("placed:Q", title="Tersalur"),
                alt.Tooltip("placement_rate_pct:Q", title="Placement rate (%)", format=".3f"),
            ],
        )
        .properties(title=title or f"Placement rate per {by_col} (zoom)", height=min(420, 28 * max(4, len(agg))))
    )

    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("placement_rate_pct:Q", format=".2f"))
    return base + labels


# =========================
# Clustering (PDF style, no inertia chart)
# =========================
@dataclass
class ClusterOut:
    best_k: int
    labeled: pd.DataFrame
    svd2d: pd.DataFrame


@st.cache_data(show_spinner=False)
def fit_cluster_pdf_style(df_in: pd.DataFrame, feature_cols: List[str], k_min: int, k_max: int, seed: int) -> ClusterOut:
    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X = drop_identifier_like(X)

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    # pilih best_k by silhouette (tetap dihitung, tapi TIDAK divisualisasikan)
    from sklearn.metrics import silhouette_score

    best_k = k_min
    best_sil = -1.0

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
        pipe = Pipeline([("prep", prep), ("km", km)])
        pipe.fit(X)
        Xt = pipe.named_steps["prep"].transform(X)
        labels = pipe.named_steps["km"].labels_

        # silhouette butuh >=2 cluster + ada variasi
        try:
            sil = float(silhouette_score(Xt, labels))
        except Exception:
            sil = -1.0

        if sil > best_sil:
            best_sil = sil
            best_k = k

    km = MiniBatchKMeans(n_clusters=best_k, random_state=seed, batch_size=2048, n_init="auto")
    pipe = Pipeline([("prep", prep), ("km", km)])
    pipe.fit(X)
    labels = pipe.named_steps["km"].labels_

    Xt = pipe.named_steps["prep"].transform(X)
    svd = TruncatedSVD(n_components=2, random_state=seed)
    xy = svd.fit_transform(Xt)

    svd2d = pd.DataFrame({"SVD1": xy[:, 0], "SVD2": xy[:, 1], "cluster_id": labels})

    labeled = df.copy()
    labeled["cluster_id"] = labels
    return ClusterOut(best_k=int(best_k), labeled=labeled, svd2d=svd2d)


def chart_cluster_svd_altair(df2d: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df2d)
        .mark_circle(size=40, opacity=0.7)
        .encode(
            x=alt.X("SVD1:Q", title="SVD1"),
            y=alt.Y("SVD2:Q", title="SVD2"),
            color=alt.Color("cluster_id:N", title="cluster_id"),
            tooltip=["cluster_id:N", "SVD1:Q", "SVD2:Q"],
        )
        .properties(height=520, title="Cluster scatter (TruncatedSVD 2D) — sesuai PDF")
        .interactive()
    )


# =========================
# Supervised (PDF style: Logistic Regression + Top-K)
# =========================
@dataclass
class SupOut:
    summary: pd.DataFrame
    scored: pd.DataFrame
    pr_curve: pd.DataFrame
    best_model: Pipeline


def _safe_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """
    Biar gak error kalau positive sangat sedikit.
    - Kalau pos < 2: train full, no eval split.
    - Kalau pos cukup: stratify split.
    """
    y = y.astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos < 2 or neg < 2:
        return X, X, y, y, False  # no split

    # pastikan test punya >=1 positive dan train punya >=1 positive
    # kalau test_size bikin test_pos 0 -> adjust
    desired_test_pos = max(1, int(round(pos * test_size)))
    if desired_test_pos >= pos:
        desired_test_pos = pos - 1
    if desired_test_pos < 1:
        return X, X, y, y, False

    # adjust test_size (minimum) supaya stratify aman
    min_test_size = desired_test_pos / pos
    ts = max(test_size, min_test_size)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=ts, random_state=seed, stratify=y
    )
    # guard
    if ytr.nunique() < 2 or yte.nunique() < 2:
        return X, X, y, y, False
    return Xtr, Xte, ytr, yte, True


@st.cache_data(show_spinner=False)
def fit_supervised_pdf_style(df_in: pd.DataFrame, target: str, feature_cols: List[str], test_size: float, seed: int) -> SupOut:
    df = df_in.copy()
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    y = df[target].astype(int)

    X = drop_identifier_like(X)

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    Xtr, Xte, ytr, yte, has_eval = _safe_train_test_split(X, y, test_size=test_size, seed=seed)

    base = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear"))
    ])

    # tuning seperti PDF (grid C)
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed) if ytr.nunique() == 2 else None

    if cv is None:
        best = base.fit(Xtr, ytr)
    else:
        gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_

    # metrics (kalau ada eval split)
    rows = []
    if has_eval:
        p = best.predict_proba(Xte)[:, 1]
        rows.append({
            "Model": "Tuned Logistic Regression",
            "PR-AUC": float(average_precision_score(yte, p)),
            "ROC-AUC": float(roc_auc_score(yte, p)),
            "LogLoss": float(log_loss(yte, p, labels=[0, 1])),
            "Brier": float(brier_score_loss(yte, p)),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "n_test": int(len(yte)),
            "pos_test": int((yte == 1).sum()),
        })
    else:
        rows.append({
            "Model": "Tuned Logistic Regression",
            "PR-AUC": np.nan,
            "ROC-AUC": np.nan,
            "LogLoss": np.nan,
            "Brier": np.nan,
            "p_min": np.nan,
            "p_max": np.nan,
            "n_test": 0,
            "pos_test": 0,
        })

    summary = pd.DataFrame(rows)

    # score semua data
    pall = best.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["placement_score"] = pall
    scored = scored.sort_values("placement_score", ascending=False).reset_index(drop=True)

    # PR curve (kalau ada eval)
    if has_eval:
        precision, recall, thr = precision_recall_curve(yte, p)
        pr_df = pd.DataFrame({
            "precision": precision,
            "recall": recall,
        })
    else:
        pr_df = pd.DataFrame({"precision": [], "recall": []})

    return SupOut(summary=summary, scored=scored, pr_curve=pr_df, best_model=best)


def chart_pr_curve(pr_df: pd.DataFrame) -> alt.Chart:
    if pr_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().properties(title="PR Curve (tidak tersedia: positive terlalu sedikit)")
    return (
        alt.Chart(pr_df)
        .mark_line(point=False)
        .encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
        )
        .properties(height=300, title="Precision–Recall Curve (sesuai PDF)")
        .interactive()
    )


def chart_topk_capture(scored: pd.DataFrame, target: str) -> alt.Chart:
    df = scored.copy()
    df["rank"] = np.arange(1, len(df) + 1)
    df["y_true"] = df[target].astype(int)
    df["cum_hits"] = df["y_true"].cumsum()

    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("rank:Q", title="Rank (1=skor tertinggi)"),
            y=alt.Y("cum_hits:Q", title="Cumulative positives captured"),
            tooltip=["rank:Q", "cum_hits:Q", alt.Tooltip("placement_score:Q", format=".6f"), "y_true:Q"],
        )
        .properties(height=280, title="Top-K capture curve (sesuai PDF)")
        .interactive()
    )


def chart_precision_recall_at_k(scored: pd.DataFrame, target: str, k_max: int = 500) -> alt.Chart:
    df = scored.copy()
    df["y_true"] = df[target].astype(int)
    n = len(df)
    k_max = min(k_max, n)

    rows = []
    total_pos = int(df["y_true"].sum())
    cum = df["y_true"].cumsum().values
    for k in range(1, k_max + 1):
        tp = int(cum[k - 1])
        prec = tp / k
        rec = (tp / total_pos) if total_pos > 0 else np.nan
        rows.append({"k": k, "precision_at_k": prec, "recall_at_k": rec})
    met = pd.DataFrame(rows)

    base = alt.Chart(met).transform_fold(
        ["precision_at_k", "recall_at_k"],
        as_=["metric", "value"]
    ).mark_line().encode(
        x=alt.X("k:Q", title="K"),
        y=alt.Y("value:Q", title="Value", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", title=None),
        tooltip=["k:Q", "metric:N", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=300, title="Precision@K & Recall@K (sesuai PDF)").interactive()

    return base


# =========================
# Sidebar UI (marketing AB-ish: clean, single flow)
# =========================
with st.sidebar:
    st.header("Data source")

    source = st.radio("Choose", ["Repo file", "Upload file (CSV/XLSX/GZ)"], index=0)

    df_raw: Optional[pd.DataFrame] = None
    found_path: Optional[str] = None

    if source == "Repo file":
        script_path, repo_root, rels = scan_repo_files()

        st.caption("Debug (repo detection)")
        st.code(f"__file__: {script_path}\nrepo_root: {repo_root}")

        if not rels:
            st.error(
                "Tidak ada file data (.csv/.csv.gz/.gz/.xlsx) terdeteksi di repo.\n\n"
                "Solusi:\n"
                "- Pakai **Upload file**, atau\n"
                "- Pastikan file ada di repo"
            )
            st.stop()

        chosen = st.selectbox("Pilih file data di repo:", rels, index=0)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file (ini bukti file beneran kebaca)")
        try:
            st.write("Path:", chosen)
            st.write("Size (bytes):", full_path.stat().st_size)
        except Exception as e:
            st.error(f"Gagal akses file: {e}")
            st.stop()

        if full_path.name.lower().endswith((".csv", ".csv.gz", ".gz")):
            st.code(sniff_file_head(full_path, n_lines=12))

        try:
            df_raw = read_table(full_path)
            found_path = str(full_path)
            st.success(f"Loaded: {chosen}")
        except Exception as e:
            st.error(f"Gagal baca file: {chosen}\nError: {e}")
            st.stop()

    else:  # Upload
        f = st.file_uploader("Upload CSV/XLSX/GZ", type=["csv", "xlsx", "xls", "gz"])
        if f is not None:
            name = f.name.lower()
            try:
                if name.endswith(".csv"):
                    df_raw = pd.read_csv(f, encoding_errors="replace", on_bad_lines="skip")
                elif name.endswith((".xlsx", ".xls")):
                    df_raw = pd.read_excel(f)
                elif name.endswith(".gz") or name.endswith(".csv.gz"):
                    df_raw = pd.read_csv(f, compression="gzip", encoding_errors="replace", on_bad_lines="skip")
                else:
                    st.error("Upload CSV / XLSX / GZ.")
                    st.stop()
                found_path = "uploaded"
            except Exception as e:
                st.error(f"Gagal baca upload: {e}")
                st.stop()

    st.divider()
    st.header("Settings (PDF-style)")

    seed = st.number_input("random_state", value=42, step=1)

    use_sample = st.toggle("Use sample for training/plots?", value=True)
    sample_n = st.number_input("Sample size (keep all positives)", value=2000, min_value=500, step=500)

    test_size = st.slider("test_size", 0.05, 0.5, 0.2)

    st.divider()
    st.caption("Catatan: sampling akan *tetap menjaga semua baris positif* supaya supervised tidak mati.")


if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()


# =========================
# Main: Build dataset (target + FE mengikuti PDF)
# =========================
st.title("Persona Predict (PDF-aligned)")

st.caption(f"Loaded from: {found_path}")

df = make_target(df_raw)
df = fe_core(df)

# Sampling (keep all positives) — supaya supervised tidak “min_count=1”
df_work = df.copy()
if use_sample and len(df_work) > int(sample_n) and "Penyaluran_flag" in df_work.columns:
    pos = df_work[df_work["Penyaluran_flag"] == 1]
    neg = df_work[df_work["Penyaluran_flag"] == 0]
    take_neg = max(0, int(sample_n) - len(pos))
    neg_s = neg.sample(min(len(neg), take_neg), random_state=int(seed)) if take_neg > 0 else neg.head(0)
    df_work = pd.concat([pos, neg_s], axis=0).sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    st.warning(f"Using sample: {len(df_work):,} rows (kept all positives={len(pos):,}).")

# Overview
c1, c2 = st.columns([1.3, 1])
with c1:
    st.subheader("Preview")
    st.dataframe(df_work.head(25), use_container_width=True)
with c2:
    st.subheader("Stats")
    st.metric("Rows (work)", f"{len(df_work):,}")
    st.metric("Rows (full)", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")
    if "Penyaluran_flag" in df.columns:
        vc = df["Penyaluran_flag"].value_counts()
        st.write("Target counts (full)")
        st.write(vc)

st.divider()

# =========================
# Tabs: EDA / Clustering / Supervised
# =========================
tab_eda, tab_cluster, tab_sup = st.tabs(["EDA (PDF)", "Clustering (PDF)", "Supervised (PDF)"])


# -------------------------
# EDA — mengikuti PDF
# -------------------------
with tab_eda:
    st.subheader("EDA mengikuti pola di PDF Vertopal (motivasi + placement-rate zoom < 1%)")

    if "Motivasi_cluster" in df_work.columns:
        st.altair_chart(
            pct_bar_h(df_work, "Motivasi_cluster", "Distribusi motivasi utama peserta"),
            use_container_width=True,
        )
    else:
        st.warning("Kolom 'Motivasi_cluster' tidak terbentuk. Cek kolom motivasi di dataset (harus ada kata 'Motivasi' di header).")

    if "Penyaluran_flag" in df_work.columns:
        # placement rate per motivasi
        if "Motivasi_cluster" in df_work.columns:
            st.altair_chart(
                placement_rate_bar(
                    df_work, "Motivasi_cluster", "Penyaluran_flag",
                    title="Placement rate per motivasi (zoom < 1.2%)",
                    zoom_max_pct=1.2
                ),
                use_container_width=True,
            )
        else:
            st.info("Skip: Motivasi_cluster belum ada.")

        # placement rate per risk
        if "Motivasi_risk_flag" in df_work.columns:
            df_work["Motivasi_risk_label"] = df_work["Motivasi_risk_flag"].map({0: "Non-risk", 1: "Risk"})
            st.altair_chart(
                placement_rate_bar(
                    df_work, "Motivasi_risk_label", "Penyaluran_flag",
                    title="Placement rate per risk level (zoom < 1.2%)",
                    zoom_max_pct=1.2
                ),
                use_container_width=True,
            )
        else:
            st.info("Skip: Motivasi_risk_flag belum ada.")

        st.markdown("### Placement rate per Segmen Karir / Umur / Region / JobConnect (zoom)")

        g1, g2 = st.columns(2)
        with g1:
            if "Segmen_karir" in df_work.columns:
                st.altair_chart(
                    placement_rate_bar(df_work, "Segmen_karir", "Penyaluran_flag", zoom_max_pct=1.2),
                    use_container_width=True,
                )
            else:
                st.warning("Segmen_karir tidak ada.")

            if "Umur_bin" in df_work.columns:
                st.altair_chart(
                    placement_rate_bar(df_work, "Umur_bin", "Penyaluran_flag", zoom_max_pct=1.2),
                    use_container_width=True,
                )
            else:
                st.warning("Umur_bin tidak ada.")

        with g2:
            if "Region" in df_work.columns:
                st.altair_chart(
                    placement_rate_bar(df_work, "Region", "Penyaluran_flag", zoom_max_pct=1.2),
                    use_container_width=True,
                )
            else:
                st.warning("Region tidak ada.")

            if "Program_jobconnect_flag" in df_work.columns:
                df_work["JobConnect"] = df_work["Program_jobconnect_flag"].map({0: "Non-JobConnect", 1: "JobConnect"})
                st.altair_chart(
                    placement_rate_bar(df_work, "JobConnect", "Penyaluran_flag", zoom_max_pct=1.2),
                    use_container_width=True,
                )
            else:
                st.warning("Program_jobconnect_flag tidak ada.")
    else:
        st.warning("Target 'Penyaluran_flag' tidak ada (kolom 'Penyaluran Kerja' tidak ketemu).")


# -------------------------
# Clustering — mengikuti PDF (tanpa inertia chart)
# -------------------------
with tab_cluster:
    st.subheader("Persona clustering (MiniBatchKMeans + TruncatedSVD 2D) — sesuai PDF")

    # core_cols dari PDF (pakai yang ada saja)
    core_cols = [
        "Umur_bin", "Region",
        "Batch_num", "Batch_has_plus",
        "Community_flag", "Event_flag", "Engagement_level",
        "Kategori_Pekerjaan_FE", "Level_Pekerjaan_FE",
        "Domain_pendidikan", "Domain_product",
        "Segmen_karir", "Level_pendidikan_FE",
        "Motivasi_cluster", "Motivasi_risk_flag",
        "Program_jobconnect_flag",
    ]
    feat_cols = [c for c in core_cols if c in df_work.columns]

    if not feat_cols:
        st.error("Kolom fitur untuk clustering tidak terbentuk. Minimal: Umur_bin/Region/Motivasi_cluster.")
    else:
        a, b, c = st.columns([1, 1, 1])
        with a:
            kmin = st.number_input("k_min", value=2, min_value=2, step=1)
        with b:
            kmax = st.number_input("k_max", value=6, min_value=2, step=1)
        with c:
            run = st.button("Run clustering", use_container_width=True)

        if run:
            try:
                out = fit_cluster_pdf_style(df_work, feat_cols, int(kmin), int(kmax), int(seed))
                st.success(f"Best k (silhouette): {out.best_k}  — (silhouette dihitung, tapi inertia TIDAK ditampilkan)")
                st.altair_chart(chart_cluster_svd_altair(out.svd2d), use_container_width=True)

                st.markdown("### Snapshot cluster_id (head)")
                st.dataframe(out.labeled[feat_cols + (["Penyaluran_flag"] if "Penyaluran_flag" in out.labeled.columns else []) + ["cluster_id"]].head(25),
                             use_container_width=True)

            except Exception as e:
                st.error(f"Clustering failed: {e}")


# -------------------------
# Supervised — mengikuti PDF (Top-K + zoom)
# -------------------------
with tab_sup:
    st.subheader("Supervised ranking (Logistic Regression) — Top-K analysis sesuai PDF")

    if "Penyaluran_flag" not in df_work.columns:
        st.error("Target tidak ada. Pastikan ada kolom 'Penyaluran Kerja' di dataset.")
    else:
        # features supervised = core_cols + (cluster_id kalau user habis run clustering, tapi ini tab terpisah jadi opsional)
        core_cols = [
            "Umur_bin", "Region",
            "Batch_num", "Batch_has_plus",
            "Community_flag", "Event_flag", "Engagement_level",
            "Kategori_Pekerjaan_FE", "Level_Pekerjaan_FE",
            "Domain_pendidikan", "Domain_product",
            "Segmen_karir", "Level_pendidikan_FE",
            "Motivasi_cluster", "Motivasi_risk_flag",
            "Program_jobconnect_flag",
        ]
        feat_cols = [c for c in core_cols if c in df_work.columns]
        if not feat_cols:
            st.error("Fitur supervised belum terbentuk (minimal Motivasi_cluster + beberapa kolom lain).")
        else:
            run_sup = st.button("Run supervised ranking", use_container_width=True)

            if run_sup:
                try:
                    sup = fit_supervised_pdf_style(df_work, "Penyaluran_flag", feat_cols, float(test_size), int(seed))

                    st.markdown("### Model summary")
                    st.dataframe(sup.summary, use_container_width=True, hide_index=True)

                    st.markdown("### Precision–Recall curve")
                    st.altair_chart(chart_pr_curve(sup.pr_curve), use_container_width=True)

                    st.markdown("### Top-K capture curve")
                    st.altair_chart(chart_topk_capture(sup.scored, "Penyaluran_flag"), use_container_width=True)

                    st.markdown("### Precision@K & Recall@K")
                    kmax = st.slider("Max K for curve", 50, min(500, len(sup.scored)), 200)
                    st.altair_chart(chart_precision_recall_at_k(sup.scored, "Penyaluran_flag", k_max=int(kmax)), use_container_width=True)

                    st.markdown("### Top-N table (sesuai PDF: fokus Top-K)")
                    topn = st.slider("Top-N", 10, min(500, len(sup.scored)), 50)
                    st.dataframe(sup.scored.head(int(topn)), use_container_width=True)

                    st.download_button(
                        "Download Top-N (CSV)",
                        data=sup.scored.head(int(topn)).to_csv(index=False).encode("utf-8"),
                        file_name="persona_predict_topN.csv",
                        mime="text/csv",
                    )

                    # Explain Top-K choice (PDF logic: see capture & PR/precision@k)
                    st.info(
                        "Kenapa Top-K dipakai (sesuai PDF): karena target sangat imbalanced, "
                        "kita nilai model dari seberapa banyak positives yang ‘ketangkap’ di urutan atas "
                        "(capture curve) + precision@K/recall@K. Untuk rate <1%, chart sudah di-zoom."
                    )

                except Exception as e:
                    st.error(f"Supervised modelling failed: {e}")

st.caption("Output dibuat mengikuti pola PDF (EDA + clustering + supervised Top-K), tanpa Google Drive, tanpa LFS, semua chart Altair.")
