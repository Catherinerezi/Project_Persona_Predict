# streamlit_app.py
# Persona Predict — PDF-aligned (Interactive Streamlit + Altair)
# One-file app: EDA (filterable) + Clustering (persona named) + Supervised Top-K (dashboard)
# Works with repo scan OR upload. No Drive, no requests.

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Persona Predict (PDF-aligned)", layout="wide")
alt.data_transformers.disable_max_rows()

# ----------------------------
# Visual theme helper (dark-friendly)
# ----------------------------
def dark_cfg(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            labelColor="#EAEAEA",
            titleColor="#EAEAEA",
            gridColor="#2a2a2a",
            domainColor="#5a5a5a",
            tickColor="#5a5a5a",
        )
        .configure_legend(labelColor="#EAEAEA", titleColor="#EAEAEA")
        .configure_title(color="#EAEAEA")
        .configure_view(strokeOpacity=0)
    )

# ----------------------------
# Repo file scan + robust loader
# ----------------------------
def _repo_root_from_file() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[1]

def _is_data_file(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))

def sniff_file_head(path: Path, n_lines: int = 8) -> str:
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                out = []
                for _ in range(n_lines):
                    out.append(next(f))
                return "".join(out)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                out = []
                for _ in range(n_lines):
                    out.append(next(f))
                return "".join(out)
    except Exception as e:
        return f"[Gagal baca head: {e}]"

def _detect_delimiter(path: Path) -> str:
    candidates = [",", ";", "\t", "|"]
    sample = ""
    try:
        name = path.name.lower()
        if name.endswith(".csv.gz") or name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                sample = f.read(1024 * 64)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(1024 * 64)
    except Exception:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=candidates)
        return dialect.delimiter
    except Exception:
        counts = {c: sample.count(c) for c in candidates}
        return max(counts, key=counts.get) if counts else ","

def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    if name.endswith((".csv", ".csv.gz", ".gz")):
        sep = _detect_delimiter(path)
        return pd.read_csv(
            path,
            sep=sep,
            engine="c",
            compression="gzip" if (name.endswith(".csv.gz") or name.endswith(".gz")) else None,
            encoding_errors="replace",
            on_bad_lines="skip",
            low_memory=False,
        )

    raise ValueError(f"Format tidak didukung: {path.name}")

@st.cache_data(show_spinner=True)
def scan_repo_files() -> tuple[str, str, List[str]]:
    script_path = Path(__file__).resolve()
    repo_root = _repo_root_from_file()

    rels: List[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if not _is_data_file(p):
            continue
        rel = str(p.relative_to(repo_root))
        if any(part.startswith(".") for part in p.parts):
            continue
        if "venv" in rel or "__pycache__" in rel:
            continue
        rels.append(rel)
    rels = sorted(set(rels))
    return str(script_path), str(repo_root), rels

# ----------------------------
# Column helpers
# ----------------------------
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().strip())

def pick_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df.empty:
        return None
    norm_map = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        k = _norm_col(a)
        if k in norm_map:
            return norm_map[k]
    for c in df.columns:
        nc = _norm_col(c)
        for a in aliases:
            if _norm_col(a) in nc:
                return c
    return None

def ensure_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")

def to_int_safe(s: pd.Series, default: int = 0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(default).astype(int)

# ----------------------------
# TARGET creation
# raw_data kamu: "Penyaluran Kerja" = Tertarik / Tidak tertarik / "-"
# ----------------------------
NEG_PATTERN = re.compile(r"\b(tidak|tdk|no|not|belum|-\b|bukan)\b", re.I)
POS_INTEREST = re.compile(r"\b(tertarik)\b", re.I)
POS_PLACED = re.compile(r"\b(tersalur|placed|berhasil|hired)\b", re.I)

def make_target_interest(df_raw: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = df_raw.copy()
    s = ensure_str(df[col_name]).str.lower().str.strip()

    # negasi dulu
    neg = s.str.contains(NEG_PATTERN, regex=True)
    pos = s.str.contains(POS_INTEREST, regex=True)

    y = np.where(neg, 0, np.where(pos, 1, 0)).astype(int)
    df["Target_flag"] = y
    df["Target_label"] = np.where(df["Target_flag"] == 1, "Tertarik", "Tidak/Belum")
    df["Target_mode"] = f"Interest({col_name})"
    return df

def make_target_placed(df_raw: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = df_raw.copy()
    s = ensure_str(df[col_name]).str.lower().str.strip()
    neg = s.str.contains(NEG_PATTERN, regex=True)
    pos = s.str.contains(POS_PLACED, regex=True)
    y = np.where(neg, 0, np.where(pos, 1, 0)).astype(int)
    df["Target_flag"] = y
    df["Target_label"] = np.where(df["Target_flag"] == 1, "Placed/Tersalur", "Not placed")
    df["Target_mode"] = f"Placed({col_name})"
    return df

# ----------------------------
# Feature Engineering (PDF-ish)
# ----------------------------
def map_motivasi_cluster(val: str) -> str:
    v = str(val).lower()
    if "switch" in v or "pindah" in v:
        return "Switch career"
    if "freelance" in v:
        return "Freelance"
    if "upgrade" in v:
        return "Upgrade diri"
    if "skill" in v or "belajar" in v:
        return "Belajar skill"
    if "kerja" in v or "bekerja" in v:
        return "Dapat kerja"
    return "Lainnya"

def map_risk(cluster: str) -> str:
    # PDF kamu: high risk sering dikaitkan dengan “Dapat kerja” (pressure tinggi)
    return "High risk" if cluster == "Dapat kerja" else "Low risk"

def domain_product(p: str) -> str:
    v = str(p).lower()
    if any(k in v for k in ["data", "machine", "ai", "analyst", "engineer", "science"]):
        return "Data"
    if any(k in v for k in ["marketing", "growth", "digital marketing", "seo", "ads"]):
        return "Marketing"
    if any(k in v for k in ["ui", "ux", "ui/ux", "design"]):
        return "UI/UX"
    if any(k in v for k in ["web", "fullstack", "backend", "front end", "frontend", "golang", "mobile", "app"]):
        return "Software"
    if any(k in v for k in ["finance", "account", "audit", "investment", "bank"]):
        return "Finance"
    if any(k in v for k in ["hr", "human resource", "people"]):
        return "HR"
    if any(k in v for k in ["cyber", "security"]):
        return "Cyber"
    return "Other"

def domain_pendidikan(bg: str, jur: str) -> str:
    v = f"{bg} {jur}".lower()
    if any(k in v for k in ["informatika", "computer", "it", "teknik", "engineering", "system"]):
        return "STEM/IT"
    if any(k in v for k in ["ekonomi", "finance", "akunt", "account", "bisnis", "management"]):
        return "Business/Finance"
    if any(k in v for k in ["desain", "design", "dkv"]):
        return "Design"
    if any(k in v for k in ["komunikasi", "communication", "marketing"]):
        return "Communication/Marketing"
    return "Other"

def fe_core(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # strip object cols
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Umur + bin
    col_umur = pick_col(df, ["Umur", "Age"])
    if col_umur:
        df["Umur"] = pd.to_numeric(df[col_umur], errors="coerce")
        bins = [0, 22, 25, 30, 100]
        labels = ["<=22", "23-25", "26-30", "30+"]
        df["Umur_bin"] = pd.cut(df["Umur"], bins=bins, labels=labels, include_lowest=True)

    # Region (Jabodetabek)
    col_kota_j = pick_col(df, ["Kota(Jabodetabek)", "Kota (Jabodetabek)", "Domisili", "Kota"])
    if col_kota_j:
        s = ensure_str(df[col_kota_j]).str.lower()
        df["Region"] = np.where(s.str.contains("jabodetabek", na=False), "Jabodetabek", "Luar Jabodetabek")

    # Batch_num + plus
    col_batch = pick_col(df, ["Batch"])
    if col_batch:
        s = ensure_str(df[col_batch])
        df["Batch_num"] = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
        df["Batch_has_plus"] = s.str.contains(r"\+", regex=True).astype(int)

    # Jobconnect flag from Kategori contains "+"
    col_kat = pick_col(df, ["Kategori", "Category"])
    if col_kat:
        s = ensure_str(df[col_kat])
        df["Program_jobconnect_flag"] = s.str.contains(r"\+", regex=True).astype(int)

    # Community flag
    col_comm = pick_col(df, ["Community"])
    if col_comm:
        s = ensure_str(df[col_comm]).str.lower()
        df["Community_flag"] = (~s.str.contains("tidak", na=False)).astype(int)

    # Event flag
    col_event = pick_col(df, ["Pernah ikut acara dibimbing/tidak", "Pernah ikut acara", "Event"])
    if col_event:
        s = ensure_str(df[col_event]).str.lower()
        df["Event_flag"] = s.str.contains(r"(pernah|sudah|ya|ikut)", regex=True, na=False).astype(int)

    # Engagement level
    if all(c in df.columns for c in ["Community_flag", "Event_flag", "Program_jobconnect_flag"]):
        active = df[["Community_flag", "Event_flag", "Program_jobconnect_flag"]].sum(axis=1)
        df["Engagement_level"] = np.select(
            [active >= 2, active == 1, active == 0],
            ["High", "Medium", "Low"],
            default="Low",
        )

    # Motivasi_cluster + risk
    col_motiv = pick_col(df, ["Motivasi mengikuti bootcamp", "Motivasi utama", "Motivasi", "Motivasi_utama"])
    if col_motiv:
        mc = df[col_motiv].map(map_motivasi_cluster)
        df["Motivasi_cluster"] = mc
        df["Motivasi_risk_flag"] = mc.map(map_risk)

    # Domain product
    col_prod = pick_col(df, ["Product", "Program", "Produk"])
    if col_prod:
        df["Domain_product"] = df[col_prod].map(domain_product)

    # Domain pendidikan
    col_bg = pick_col(df, ["Background pendidikan", "Background pendidikan ", "Asal Sekolah"])
    col_jur = pick_col(df, ["Jurusan pendidikan", "Jurusan", "Major"])
    if col_bg or col_jur:
        bg = ensure_str(df[col_bg]) if col_bg else ""
        jr = ensure_str(df[col_jur]) if col_jur else ""
        if isinstance(bg, str):
            bg = pd.Series([bg] * len(df))
        if isinstance(jr, str):
            jr = pd.Series([jr] * len(df))
        df["Domain_pendidikan"] = [domain_pendidikan(b, j) for b, j in zip(bg, jr)]

    # Segmen_karir (dibentuk, karena kolom asli tidak ada)
    # Rule sederhana tapi stabil:
    # - Fresh Graduate Explorer: kesibukan/level pekerjaan mengandung mahasiswa/fresh graduate/intern
    # - Career Switcher: motivasi switch atau alasan category mengandung switch/pindah
    # - Working Professional Upskiller: sisanya
    col_kes = pick_col(df, ["Kesibukan sekarang apa", "Kategori Kesibukan", "Kesibukan"])
    col_lvl_job = pick_col(df, ["Level Pekerjaan", "Kategori Pekerjaan", "Pekerjaan"])
    col_alasan = pick_col(df, ["Alasan mengambil kategori Bootcamp", "Mengapa memilih Dibimbing"])

    kes = ensure_str(df[col_kes]).str.lower() if col_kes else pd.Series([""] * len(df))
    lvl = ensure_str(df[col_lvl_job]).str.lower() if col_lvl_job else pd.Series([""] * len(df))
    als = ensure_str(df[col_alasan]).str.lower() if col_alasan else pd.Series([""] * len(df))

    is_fresh = kes.str.contains(r"(mahasiswa|pelajar|fresh|intern)", regex=True, na=False) | lvl.str.contains(
        r"(fresh|intern|mahasiswa)", regex=True, na=False
    )
    is_switcher = (
        (df["Motivasi_cluster"].astype(str).str.lower().str.contains("switch", na=False) if "Motivasi_cluster" in df.columns else False)
        | als.str.contains(r"(switch|pindah|career switch|alih karir)", regex=True, na=False)
    )

    df["is_switcher"] = is_switcher.astype(int)
    df["Segmen_karir"] = np.select(
        [is_fresh, is_switcher],
        ["Fresh Graduate Explorer", "High Engagement Career Switcher"],
        default="Working Professional Upskiller",
    )

    return df

# ----------------------------
# Sampling (keep BOTH classes)
# ----------------------------
def safe_sample_binary(df: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)

    if target not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    df = df.copy()
    y = to_int_safe(df[target], 0).clip(0, 1)
    pos = df[y == 1]
    neg = df[y == 0]

    if len(pos) == 0 or len(neg) == 0:
        # kalau salah satu kelas kosong, jangan sampling agresif
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    # minimal neg biar supervised aman
    min_neg = min(len(neg), max(80, int(0.15 * n)))
    n_pos = n - min_neg
    n_pos = min(n_pos, len(pos))

    pos_s = pos.sample(n=n_pos, random_state=seed)
    neg_s = neg.sample(n=min_neg, random_state=seed)
    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out

# ----------------------------
# EDA helpers
# ----------------------------
def freq_table(s: pd.Series, top_n: int = 40) -> pd.DataFrame:
    x = ensure_str(s).fillna("Unknown")
    vc = x.value_counts(dropna=False)
    df = vc.rename_axis("Kategori").reset_index(name="Jumlah")
    if top_n and len(df) > top_n:
        head = df.iloc[:top_n].copy()
        tail_sum = int(df.iloc[top_n:]["Jumlah"].sum())
        tail = pd.DataFrame([{"Kategori": "Other (collapsed)", "Jumlah": tail_sum}])
        df = pd.concat([head, tail], axis=0, ignore_index=True)
    df["Persentase"] = (df["Jumlah"] / df["Jumlah"].sum()) * 100.0
    return df

def chart_dist_horizontal(df_freq: pd.DataFrame, title: str) -> alt.Chart:
    h = min(800, max(260, 26 * len(df_freq)))
    base = (
        alt.Chart(df_freq)
        .mark_bar()
        .encode(
            y=alt.Y("Kategori:N", sort="-x", title=None),
            x=alt.X("Persentase:Q", title="Persentase peserta (%)"),
            tooltip=["Kategori:N", "Jumlah:Q", alt.Tooltip("Persentase:Q", format=".2f")],
        )
        .properties(height=h, title=title)
    )
    text = base.mark_text(align="left", dx=4, color="#EAEAEA").encode(
        text=alt.Text("Persentase:Q", format=".1f")
    )
    return dark_cfg(base + text)

def target_rate_table(df: pd.DataFrame, by_col: str, target_col: str, min_count: int = 20) -> pd.DataFrame:
    g = df[[by_col, target_col]].copy()
    g[by_col] = ensure_str(g[by_col]).fillna("Unknown")
    g[target_col] = to_int_safe(g[target_col], 0).clip(0, 1)

    agg = g.groupby(by_col, dropna=False)[target_col].agg(["count", "sum"]).reset_index()
    agg = agg.rename(columns={by_col: "Group", "count": "Total", "sum": "Positives"})
    agg["Target_rate_pct"] = (agg["Positives"] / agg["Total"]) * 100.0
    agg = agg[agg["Total"] >= int(min_count)].sort_values("Target_rate_pct", ascending=False).reset_index(drop=True)
    return agg

def target_rate_bar(tab: pd.DataFrame, title: str, zoom_max_pct: float) -> alt.Chart:
    h = min(800, max(260, 26 * len(tab)))
    base = (
        alt.Chart(tab)
        .mark_bar()
        .encode(
            y=alt.Y("Group:N", sort="-x", title=None),
            x=alt.X(
                "Target_rate_pct:Q",
                title="Target rate (%)",
                scale=alt.Scale(domain=[0, float(zoom_max_pct)]),
            ),
            tooltip=[
                "Group:N",
                "Total:Q",
                "Positives:Q",
                alt.Tooltip("Target_rate_pct:Q", format=".3f"),
            ],
        )
        .properties(height=h, title=title)
    )
    text = base.mark_text(align="left", dx=4, color="#EAEAEA").encode(
        text=alt.Text("Target_rate_pct:Q", format=".2f")
    )
    return dark_cfg(base + text)

# ----------------------------
# ML preprocess helpers
# ----------------------------
def split_num_cat(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat

def make_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", ohe)]), cat_cols),
        ],
        remainder="drop",
    )

def candidate_feature_cols(df: pd.DataFrame) -> List[str]:
    # gabungan FE + beberapa raw penting (sesuai screenshot kamu)
    want = [
        # FE
        "Umur", "Umur_bin", "Region", "Batch_num", "Batch_has_plus",
        "Community_flag", "Event_flag", "Engagement_level", "Program_jobconnect_flag",
        "Motivasi_cluster", "Motivasi_risk_flag", "Domain_pendidikan", "Domain_product",
        "is_switcher", "Segmen_karir",
        # raw stable (kalau ada)
        "Product", "Kategori", "Month", "Channel", "Level pendidikan",
        "Kategori Kesibukan", "Level Pekerjaan", "Kategori Pekerjaan",
        "Domisili", "Provinsi", "Negara",
    ]
    return [c for c in want if c in df.columns]

# ----------------------------
# Clustering
# ----------------------------
@dataclass
class ClusterOut:
    labeled: pd.DataFrame
    svd2d: pd.DataFrame
    profile: pd.DataFrame

def persona_name_from_profile(row: pd.Series) -> str:
    # Heuristik penamaan 3 persona (PDF)
    # - Switcher dominan => High Engagement Career Switcher
    # - Umur muda / segmen fresh dominan => Fresh Graduate Explorer
    # - else => Working Professional Upskiller
    if row.get("Switcher_share", 0) >= 0.35:
        return "High Engagement Career Switcher"
    if row.get("Fresh_share", 0) >= 0.35 or row.get("Age_mean", 99) <= 23.5:
        return "Fresh Graduate Explorer"
    return "Working Professional Upskiller"

def cluster_profile(df: pd.DataFrame, cluster_col: str, target_col: str) -> pd.DataFrame:
    out = []
    for cid, g in df.groupby(cluster_col):
        y = to_int_safe(g[target_col], 0).clip(0, 1) if target_col in g.columns else pd.Series([0]*len(g))
        n = len(g)
        pos = int(y.sum())
        rate = (pos / n) * 100 if n else 0.0

        age_mean = float(pd.to_numeric(g["Umur"], errors="coerce").mean()) if "Umur" in g.columns else np.nan
        sw = float(pd.to_numeric(g["is_switcher"], errors="coerce").fillna(0).mean()) if "is_switcher" in g.columns else 0.0
        fresh = float((g["Segmen_karir"].astype(str) == "Fresh Graduate Explorer").mean()) if "Segmen_karir" in g.columns else 0.0
        high_eng = float((g["Engagement_level"].astype(str) == "High").mean()) if "Engagement_level" in g.columns else 0.0

        top_prod = g["Product"].astype(str).value_counts().head(1).index[0] if "Product" in g.columns and n else "-"
        top_chan = g["Channel"].astype(str).value_counts().head(1).index[0] if "Channel" in g.columns and n else "-"
        top_mot = g["Motivasi_cluster"].astype(str).value_counts().head(1).index[0] if "Motivasi_cluster" in g.columns and n else "-"

        out.append({
            "cluster_id": int(cid),
            "Size": n,
            "Target_rate_%": rate,
            "Age_mean": age_mean,
            "Switcher_share": sw,
            "Fresh_share": fresh,
            "HighEng_share": high_eng,
            "Top_Product": top_prod,
            "Top_Channel": top_chan,
            "Top_Motivasi": top_mot,
        })
    prof = pd.DataFrame(out).sort_values("cluster_id").reset_index(drop=True)
    prof["Persona_name"] = prof.apply(persona_name_from_profile, axis=1)
    return prof

@st.cache_data(show_spinner=False)
def fit_cluster(df_in: pd.DataFrame, feature_cols: List[str], k: int, seed: int, target_col: str) -> ClusterOut:
    df = df_in.copy()
    X = df[feature_cols].copy()

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    km = MiniBatchKMeans(n_clusters=int(k), random_state=int(seed), batch_size=2048, n_init="auto")
    pipe = Pipeline([("prep", prep), ("km", km)])
    pipe.fit(X)

    labels = pipe.named_steps["km"].labels_
    df["cluster_id"] = labels

    Xt = pipe.named_steps["prep"].transform(X)
    svd = TruncatedSVD(n_components=2, random_state=int(seed))
    xy = svd.fit_transform(Xt)
    svd2d = pd.DataFrame({"SVD1": xy[:, 0], "SVD2": xy[:, 1], "cluster_id": labels})

    prof = cluster_profile(df, "cluster_id", target_col)
    # attach persona name to points
    name_map = {int(r["cluster_id"]): r["Persona_name"] for _, r in prof.iterrows()}
    svd2d["Persona"] = svd2d["cluster_id"].map(lambda x: name_map.get(int(x), f"Cluster {x}"))

    return ClusterOut(labeled=df, svd2d=svd2d, profile=prof)

def chart_svd_persona(df2d: pd.DataFrame) -> alt.Chart:
    ch = (
        alt.Chart(df2d)
        .mark_circle(size=35, opacity=0.75)
        .encode(
            x=alt.X("SVD1:Q"),
            y=alt.Y("SVD2:Q"),
            color=alt.Color("Persona:N", title="Persona"),
            tooltip=["Persona:N", "cluster_id:N", alt.Tooltip("SVD1:Q", format=".3f"), alt.Tooltip("SVD2:Q", format=".3f")],
        )
        .properties(height=520, title="Cluster visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)")
        .interactive()
    )
    return dark_cfg(ch)

# ----------------------------
# Supervised Top-K
# ----------------------------
@dataclass
class SupOut:
    mode: str
    pr_auc: Optional[float]
    pr_curve: Optional[pd.DataFrame]
    score_holdout: Optional[pd.DataFrame]
    topk_table: Optional[pd.DataFrame]
    lift_table: Optional[pd.DataFrame]
    ranked_full: pd.DataFrame
    model: Pipeline

def topk_table_from_scores(y_true: np.ndarray, y_score: np.ndarray, K_list: List[int]) -> Tuple[pd.DataFrame, int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        raise ValueError("total_positive = 0. Tidak bisa hitung recall/PR.")
    rows = []
    for K in K_list:
        K = int(K)
        K_eff = min(K, len(y_sorted))
        captured = int(y_sorted[:K_eff].sum())
        prec_k = captured / K_eff if K_eff > 0 else 0.0
        rec_k = captured / total_pos
        rows.append({"K": K_eff, "Positives captured": captured, "Precision@K": prec_k, "Recall@K": rec_k})
    return pd.DataFrame(rows), total_pos

@st.cache_data(show_spinner=False)
def fit_supervised(
    df_in: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    test_size: float,
    seed: int,
    business_k: int,
    use_cluster_feature: bool,
) -> SupOut:
    df = df_in.copy()

    if target not in df.columns:
        raise ValueError(f"Target '{target}' tidak ada.")

    y = to_int_safe(df[target], 0).clip(0, 1)
    vc = y.value_counts()

    X = df[feature_cols].copy()
    if use_cluster_feature and "cluster_id" in df.columns and "cluster_id" not in X.columns:
        X["cluster_id"] = df["cluster_id"]

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    base = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))
    ])
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)

    # Always compute full ranking (business action)
    def fit_on_all_and_rank(model: Pipeline) -> Tuple[Pipeline, pd.DataFrame]:
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        ranked = df.copy()
        ranked["proba"] = proba
        ranked["row_id"] = np.arange(len(ranked))
        ranked = ranked.sort_values("proba", ascending=False).reset_index(drop=True)
        # attach business top-k marker
        k_eff = min(int(business_k), len(ranked))
        ranked["in_topK"] = 0
        if k_eff > 0:
            ranked.loc[:k_eff-1, "in_topK"] = 1
        return model, ranked

    # Case: one class only => train all, no eval
    if len(vc) < 2 or vc.min() < 2:
        model, ranked = fit_on_all_and_rank(gs)
        return SupOut(
            mode="FULL_ONLY (insufficient class for eval)",
            pr_auc=None,
            pr_curve=None,
            score_holdout=None,
            topk_table=None,
            lift_table=None,
            ranked_full=ranked,
            model=model,
        )

    # Try holdout
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=float(test_size), random_state=int(seed), stratify=y
        )
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_

        y_proba = best.predict_proba(Xte)[:, 1]
        pr_auc = float(average_precision_score(yte, y_proba))
        prec, rec, _ = precision_recall_curve(yte, y_proba)
        pr_curve = pd.DataFrame({"recall": rec, "precision": prec})

        score_df = pd.DataFrame({"y_true": np.asarray(yte).astype(int), "proba": y_proba}).sort_values(
            "proba", ascending=False
        ).reset_index(drop=True)

        N = len(score_df)
        K_list = sorted(set([
            10, 20, 50, 100, 200,
            int(0.005 * N), int(0.01 * N), int(0.02 * N), int(0.05 * N), int(0.1 * N),
            int(business_k),
        ]))
        K_list = [k for k in K_list if k > 0]
        topk_df, total_pos = topk_table_from_scores(score_df["y_true"].to_numpy(), score_df["proba"].to_numpy(), K_list)

        prevalence = total_pos / N
        lift = topk_df.copy()
        lift["Prevalence"] = prevalence
        lift["Lift@K"] = lift["Precision@K"] / prevalence
        lift["Random Lift"] = 1.0

        # Full ranking for business
        best_full, ranked = fit_on_all_and_rank(best)

        return SupOut(
            mode="HOLDOUT",
            pr_auc=pr_auc,
            pr_curve=pr_curve,
            score_holdout=score_df,
            topk_table=topk_df,
            lift_table=lift,
            ranked_full=ranked,
            model=best_full,
        )
    except Exception:
        # Fallback: train on all (still output ranking)
        model, ranked = fit_on_all_and_rank(gs)
        return SupOut(
            mode="FULL_ONLY (holdout failed)",
            pr_auc=None,
            pr_curve=None,
            score_holdout=None,
            topk_table=None,
            lift_table=None,
            ranked_full=ranked,
            model=model,
        )

def chart_pr_curve(pr_curve: pd.DataFrame, pr_auc: float) -> alt.Chart:
    ch = (
        alt.Chart(pr_curve)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
        )
        .properties(height=320, title=f"Precision–Recall Curve (Holdout) | PR-AUC={pr_auc:.3f}")
        .interactive()
    )
    return dark_cfg(ch)

def chart_precision_recall_at_k(topk: pd.DataFrame) -> alt.Chart:
    df = topk.copy()
    df_long = df.melt(id_vars=["K"], value_vars=["Recall@K", "Precision@K"], var_name="Metric", value_name="Rate")
    ch = (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("Rate:Q", title="Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title=None),
            tooltip=["K", "Metric", alt.Tooltip("Rate:Q", format=".3f")],
        )
        .properties(height=320, title="Top-K Curve (Precision@K & Recall@K) — PDF-aligned")
        .interactive()
    )
    return dark_cfg(ch)

def chart_lift(lift: pd.DataFrame) -> alt.Chart:
    base = (
        alt.Chart(lift)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("Lift@K:Q", title="Lift@K"),
            tooltip=["K", alt.Tooltip("Lift@K:Q", format=".2f"), alt.Tooltip("Recall@K:Q", format=".3f"), alt.Tooltip("Precision@K:Q", format=".3f")],
        )
        .properties(height=320, title="Lift@K (Gains) — PDF-aligned")
        .interactive()
    )
    baseline = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
    return dark_cfg(base + baseline)

# ----------------------------
# Sidebar: Data load + Controls
# ----------------------------
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
            st.error("Tidak ada file data terdeteksi. Pakai Upload atau taruh file di repo (mis: raw_data/).")
            st.stop()

        chosen = st.selectbox("Pilih file data di repo:", rels, index=0)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file")
        st.write("Path:", chosen)
        st.write("Size (bytes):", full_path.stat().st_size)
        if full_path.name.lower().endswith((".csv", ".csv.gz", ".gz")):
            st.code(sniff_file_head(full_path, n_lines=10))

        try:
            df_raw = read_table(full_path)
            found_path = str(full_path)
            st.success(f"Loaded: {chosen}")
        except Exception as e:
            st.error(f"Gagal baca file: {e}")
            st.stop()

    else:
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
    st.header("Global controls")
    seed = st.number_input("random_state", value=42, step=1)
    use_sample = st.toggle("Use sample (faster, keeps both classes)?", value=True)
    sample_n = st.number_input("Sample size", value=2500, min_value=500, step=250)

    st.divider()
    st.header("Target definition")
    # auto-suggest target column
    target_col_default = "Penyaluran Kerja" if (df_raw is not None and "Penyaluran Kerja" in df_raw.columns) else None
    target_col = st.selectbox(
        "Target column",
        options=list(df_raw.columns) if df_raw is not None else [],
        index=(list(df_raw.columns).index(target_col_default) if (df_raw is not None and target_col_default in df_raw.columns) else 0),
        disabled=(df_raw is None),
    )
    target_mode = st.radio("Target mode", ["Interest (Tertarik vs Tidak/Belum/-)", "Placed/Tersalur (tersalur/placed/berhasil)"], index=0)

    st.divider()
    st.header("Pipelines")
    run_cluster = st.toggle("Enable clustering tab", value=True)
    run_sup = st.toggle("Enable supervised tab", value=True)

# ----------------------------
# Main: build data
# ----------------------------
if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()

st.title("Persona Predict (PDF-aligned)")
st.caption(f"Loaded from: {found_path}")

# build target
if target_mode.startswith("Interest"):
    df0 = make_target_interest(df_raw, target_col)
else:
    df0 = make_target_placed(df_raw, target_col)

# build features
df = fe_core(df0)

# optional sampling (keep both classes)
df_work = df.copy()
if use_sample:
    df_work = safe_sample_binary(df_work, "Target_flag", int(sample_n), int(seed))
    st.warning(f"Using sample: {len(df_work):,} rows (full: {len(df):,}).")

# ----------------------------
# Quick sanity checks (fix 100% issue)
# ----------------------------
vc = df_work["Target_flag"].value_counts(dropna=False)
cA, cB, cC, cD = st.columns(4)
cA.metric("Rows (work)", f"{len(df_work):,}")
cB.metric("Target=1 (work)", f"{int(vc.get(1, 0)):,}")
cC.metric("Target=0 (work)", f"{int(vc.get(0, 0)):,}")
cD.metric("Target mode", str(df_work.get("Target_mode", ["-"])[0]) if "Target_mode" in df_work.columns else "-")

if int(vc.get(0, 0)) == 0 or int(vc.get(1, 0)) == 0:
    st.error(
        "⚠️ Target kamu masih cuma 1 kelas (semua 0 atau semua 1). "
        "Coba ganti Target mode/Target column di sidebar, atau matikan sampling dulu."
    )

st.divider()

# ----------------------------
# Filters (interactive EDA)
# ----------------------------
st.subheader("Filters (buat analisa & strategi bisnis)")

FILTER_CANDIDATES = [
    "Month", "Product", "Kategori", "Channel",
    "Domain_product", "Domain_pendidikan",
    "Segmen_karir", "Motivasi_cluster", "Motivasi_risk_flag",
    "Region", "Provinsi", "Negara",
    "Engagement_level",
]

fcols = st.columns(4)
filters: Dict[str, List[str]] = {}

i = 0
for col in FILTER_CANDIDATES:
    if col not in df_work.columns:
        continue
    vals = sorted(df_work[col].astype(str).fillna("Unknown").unique().tolist())
    with fcols[i % 4]:
        picked = st.multiselect(col, vals, default=[])
        if picked:
            filters[col] = picked
    i += 1

# numeric filter: umur
age_min, age_max = None, None
if "Umur" in df_work.columns:
    u = pd.to_numeric(df_work["Umur"], errors="coerce")
    if u.notna().any():
        mn, mx = float(u.min()), float(u.max())
        age_min, age_max = st.slider("Filter Umur (range)", min_value=int(mn), max_value=int(mx), value=(int(mn), int(mx)))

# apply filters
df_f = df_work.copy()
for col, picked in filters.items():
    df_f = df_f[df_f[col].astype(str).isin(picked)]
if age_min is not None and "Umur" in df_f.columns:
    u = pd.to_numeric(df_f["Umur"], errors="coerce")
    df_f = df_f[(u >= age_min) & (u <= age_max)]

st.caption(f"After filters: {len(df_f):,} rows")

st.divider()

# ----------------------------
# Tabs: Overview | EDA | Clustering | Supervised
# ----------------------------
tabs = ["📌 Overview", "🔎 EDA (Interactive)"]
if run_cluster:
    tabs.append("🧩 Clustering (Persona Named)")
if run_sup:
    tabs.append("🎯 Supervised (Top-K)")

tab_objs = st.tabs(tabs)

# ----------------------------
# Overview
# ----------------------------
with tab_objs[0]:
    st.subheader("Preview data (work)")
    st.dataframe(df_f.head(30), use_container_width=True)

    st.subheader("Missing (%)")
    miss = (df_f.isna().mean() * 100).sort_values(ascending=False).round(2).reset_index()
    miss.columns = ["column", "missing_pct"]
    st.dataframe(miss, use_container_width=True, hide_index=True)

# ----------------------------
# EDA Interactive
# ----------------------------
with tab_objs[1]:
    st.header("EDA — sesuai tujuan bisnis (segmentasi + strategi akuisisi + intervensi)")

    # pick breakdown for distribution
    cat_cols = [c for c in FILTER_CANDIDATES if c in df_f.columns]
    if not cat_cols:
        st.info("Belum ada kolom kategorikal yang bisa dianalisis.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            dist_col = st.selectbox("Distribusi: pilih kolom", cat_cols, index=0)
            top_n = st.slider("Top-N (high cardinality)", 10, 80, 40, 5)
        with c2:
            rate_col = st.selectbox("Target rate: pilih breakdown", cat_cols, index=min(1, len(cat_cols)-1))
            min_count = st.slider("Min count per group", 5, 200, 20, 5)
            zoom_max = st.number_input("Zoom max (%) untuk bar target-rate", value=20.0, min_value=0.1, step=1.0)

        # Distribution chart
        df_freq = freq_table(df_f[dist_col], top_n=int(top_n))
        st.altair_chart(chart_dist_horizontal(df_freq, f"Distribusi {dist_col} (Top-{top_n})"), use_container_width=True)
        st.dataframe(df_freq, use_container_width=True, hide_index=True)

        st.divider()

        # Target rate chart
        tab_rate = target_rate_table(df_f, rate_col, "Target_flag", min_count=int(min_count))
        if tab_rate.empty:
            st.info("Semua group terfilter oleh min_count / filter. Turunkan min_count atau longgarkan filter.")
        else:
            st.altair_chart(
                target_rate_bar(tab_rate.head(int(top_n)), f"Target rate by {rate_col} (Top-{top_n})", float(zoom_max)),
                use_container_width=True,
            )
            st.dataframe(tab_rate, use_container_width=True, hide_index=True)

# ----------------------------
# Clustering
# ----------------------------
if run_cluster:
    with tab_objs[2]:
        st.header("Clustering — PDF default 3 persona + profiling + penamaan")

        feat_candidates = candidate_feature_cols(df_f)
        if not feat_candidates:
            st.error("Fitur clustering tidak cukup. Pastikan FE sukses (Umur/Segmen/Engagement/Motivasi/etc).")
        else:
            cL, cR = st.columns([1.2, 1])
            with cL:
                st.subheader("Controls")
                cluster_mode = st.radio("Mode", ["PDF default (k=3)", "Auto (silhouette choose)"], index=0)
                k_min = st.number_input("k_min (auto)", value=2, min_value=2, step=1, disabled=(cluster_mode != "Auto (silhouette choose)"))
                k_max = st.number_input("k_max (auto)", value=8, min_value=2, step=1, disabled=(cluster_mode != "Auto (silhouette choose)"))
                k_fixed = st.number_input("k (PDF default)", value=3, min_value=2, step=1, disabled=(cluster_mode != "PDF default (k=3)"))
                feat_cols = st.multiselect("Feature cols", feat_candidates, default=feat_candidates)
                run_btn = st.button("Run clustering", use_container_width=True)

            with cR:
                st.subheader("Notes")
                st.write("- Default PDF: k=3 supaya persona match vertopal.")
                st.write("- Kalau mau eksperimen, pakai Auto silhouette.")

            if run_btn:
                if not feat_cols:
                    st.error("Pilih minimal 1 feature.")
                else:
                    k_use = int(k_fixed)
                    if cluster_mode == "Auto (silhouette choose)":
                        # quick silhouette scan
                        X = df_f[feat_cols].copy()
                        num_cols, cat_cols = split_num_cat(X)
                        prep = make_preprocess(num_cols, cat_cols)
                        Xt = prep.fit_transform(X)

                        rows = []
                        for k in range(int(k_min), int(k_max) + 1):
                            km = MiniBatchKMeans(n_clusters=k, random_state=int(seed), batch_size=2048, n_init="auto")
                            lab = km.fit_predict(Xt)
                            sil = float(silhouette_score(Xt, lab))
                            rows.append({"k": k, "silhouette": sil})
                        kdf = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
                        bestk = int(kdf.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
                        st.success(f"Auto best k (silhouette): {bestk}")

                        ch = alt.Chart(kdf).mark_line(point=True).encode(
                            x=alt.X("k:Q"), y=alt.Y("silhouette:Q"),
                            tooltip=["k", alt.Tooltip("silhouette:Q", format=".4f")]
                        ).properties(height=240, title="Silhouette by k")
                        st.altair_chart(dark_cfg(ch), use_container_width=True)
                        k_use = bestk

                    cl = fit_cluster(df_f, feat_cols, k=int(k_use), seed=int(seed), target_col="Target_flag")

                    # persist to session
                    st.session_state["cluster_out"] = cl

            # display if exists
            if "cluster_out" in st.session_state:
                cl: ClusterOut = st.session_state["cluster_out"]

                st.subheader("Cluster visualization (persona named)")
                st.altair_chart(chart_svd_persona(cl.svd2d), use_container_width=True)

                st.subheader("Cluster profiling (untuk penamaan persona & strategi intervensi)")
                prof = cl.profile.copy()
                prof["Target_rate_%"] = prof["Target_rate_%"].map(lambda x: f"{x:.2f}")
                prof["Switcher_share"] = prof["Switcher_share"].map(lambda x: f"{x*100:.1f}%")
                prof["Fresh_share"] = prof["Fresh_share"].map(lambda x: f"{x*100:.1f}%")
                prof["HighEng_share"] = prof["HighEng_share"].map(lambda x: f"{x*100:.1f}%")
                st.dataframe(prof, use_container_width=True, hide_index=True)

                # allow user to use cluster_id downstream
                st.caption("cluster_id sudah tersimpan di session_state untuk dipakai di Supervised (opsional).")

# ----------------------------
# Supervised Top-K
# ----------------------------
if run_sup:
    sup_tab_index = 3 if run_cluster else 2
    with tab_objs[sup_tab_index]:
        st.header("Supervised ranking — Top-K (PDF-aligned)")

        feat_candidates = candidate_feature_cols(df_f)
        if not feat_candidates:
            st.error("Fitur supervised tidak cukup. Pastikan FE sukses.")
        else:
            # bring cluster feature if available
            has_cluster = ("cluster_out" in st.session_state) and ("cluster_id" in st.session_state["cluster_out"].labeled.columns)

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.subheader("Controls")
                business_k = st.number_input("Business capacity K (berapa peserta yang bisa diintervensi)", value=200, min_value=10, step=10)
                test_size = st.slider("test_size (holdout)", 0.05, 0.5, 0.2, 0.05)
                use_cluster_feature = st.toggle("Use cluster_id as feature (if available)", value=True, disabled=(not has_cluster))
                feat_cols = st.multiselect("Feature cols", feat_candidates, default=feat_candidates)
                run_btn = st.button("Run supervised ranking", use_container_width=True)

            with c2:
                st.subheader("Kenapa Top-K?")
                st.write(
                    "Target biasanya imbalanced. Keputusan bisnis jarang intervensi semua orang.\n"
                    "Top-K membantu memilih K peserta dengan peluang target tertinggi untuk:\n"
                    "- prioritas outreach/mentoring\n"
                    "- strategi channel & produk\n"
                    "- desain program & intervensi"
                )

            if run_btn:
                if not feat_cols:
                    st.error("Pilih minimal 1 feature.")
                else:
                    # use labeled df if cluster exists, else df_f
                    df_model = df_f.copy()
                    if has_cluster:
                        # merge cluster_id from stored labeled by index alignment (safe if same df_f row order differs)
                        # simpler: recompute clustering label for df_f by joining on row_id we create here
                        # We'll do robust: if df_f is subset, re-run mapping using same order by resetting row_id
                        pass

                    sup = fit_supervised(
                        df_in=df_f,
                        target="Target_flag",
                        feature_cols=feat_cols,
                        test_size=float(test_size),
                        seed=int(seed),
                        business_k=int(business_k),
                        use_cluster_feature=bool(use_cluster_feature and ("cluster_id" in df_f.columns)),
                    )
                    st.session_state["sup_out"] = sup

            if "sup_out" in st.session_state:
                sup: SupOut = st.session_state["sup_out"]

                st.subheader("Status")
                st.write(f"Mode: **{sup.mode}**")
                if sup.mode != "HOLDOUT":
                    st.warning(
                        "Evaluasi holdout tidak tersedia (kelas minoritas kecil / split gagal). "
                        "Tapi ranking Top-K tetap keluar untuk kebutuhan bisnis."
                    )

                # If holdout metrics available
                if sup.mode == "HOLDOUT" and sup.pr_curve is not None and sup.pr_auc is not None:
                    st.altair_chart(chart_pr_curve(sup.pr_curve, sup.pr_auc), use_container_width=True)

                    st.subheader("Top-K metrics (holdout)")
                    topk = sup.topk_table.copy() if sup.topk_table is not None else None
                    if topk is not None:
                        show = topk.copy()
                        show["Precision@K"] = (show["Precision@K"] * 100).map(lambda x: f"{x:.2f}%")
                        show["Recall@K"] = (show["Recall@K"] * 100).map(lambda x: f"{x:.2f}%")
                        st.dataframe(show, use_container_width=True, hide_index=True)
                        st.altair_chart(chart_precision_recall_at_k(topk), use_container_width=True)

                    if sup.lift_table is not None:
                        st.altair_chart(chart_lift(sup.lift_table), use_container_width=True)

                st.divider()
                st.subheader("Business output: Ranked peserta (full data)")

                ranked = sup.ranked_full.copy()
                k_eff = min(int(st.session_state["sup_out"].ranked_full["in_topK"].sum()), len(ranked))
                st.write(f"Top-K flagged: **{k_eff}** baris teratas (in_topK=1)")

                # choose columns to show (safe)
                show_cols = [c for c in ["row_id", "proba", "in_topK", "Target_flag", "Target_label",
                                         "Product", "Kategori", "Channel", "Month",
                                         "Segmen_karir", "Motivasi_cluster", "Region", "Umur"] if c in ranked.columns]
                st.dataframe(ranked[show_cols].head(max(50, int(business_k))), use_container_width=True)

                st.caption("Tips: pakai filter di atas untuk melihat Top-K per Channel/Product/Segmen, agar actionable.")

st.divider()
st.caption("✅ Jika masih ingin 100% match ke vertopal (angka & bentuk), kirim screenshot bagian preprocessing/feature list di notebook (kolom yang dipakai) — nanti kita kunci fitur & k=3 persis.")
