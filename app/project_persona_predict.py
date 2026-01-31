# streamlit_app.py
# Persona Predict (PDF-aligned) — Streamlit + Altair (single-file)
# Fixes:
# - Target "Tertarik" vs "Tidak tertarik" (no substring bug)
# - EDA richer + multi-breakdown + global filters (selaras tujuan bisnis)
# - Clustering default k=3 + persona naming (PDF/notebook-like)
# - Supervised Top-K dashboard matang (kapasitas K, lift, recall/precision, PR curve)

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Page config + Altair theme for DARK Streamlit
# ----------------------------
st.set_page_config(page_title="Persona Predict (PDF-aligned)", layout="wide")
alt.data_transformers.disable_max_rows()


def _apply_altair_dark():
    # label default altair itu hitam → di dark theme jadi "ngilang"
    # + grid/domain juga perlu kontras
    alt.themes.enable("default")
    return (
        alt.Chart(pd.DataFrame({"_": [0]}))
        .mark_point(opacity=0)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="white",
            titleColor="white",
            gridColor="#333333",
            domainColor="#666666",
            tickColor="#666666",
        )
        .configure_legend(labelColor="white", titleColor="white")
        .configure_title(color="white")
    )


_ = _apply_altair_dark()

YES_PATTERN = re.compile(r"\b(ya|y|yes|sudah|tersalur|placed|berhasil)\b", re.I)

# Domain mapping untuk FE Segmen_karir (heuristik ringan)
DOMAIN_DATA = re.compile(r"(data|stat|matemat|informat|komputer|it|teknik|engineer|sistem|software)", re.I)
DOMAIN_DESIGN = re.compile(r"(ui|ux|design|dkv|visual|grafis|multimedia|product design)", re.I)
DOMAIN_BIZ = re.compile(r"(bisnis|manajem|akunt|finance|marketing|ekonomi|business|sales|hr)", re.I)


# ----------------------------
# Repo scan + robust loader
# ----------------------------
def _repo_root_from_file() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parents[1]


def _is_data_file(p: Path) -> bool:
    return p.name.lower().endswith((".csv", ".csv.gz", ".gz", ".xlsx", ".xls"))


def sniff_file_head(path: Path, n_lines: int = 12) -> str:
    try:
        name = path.name.lower()
        if name.endswith((".csv.gz", ".gz")):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                out = []
                for _ in range(n_lines):
                    try:
                        out.append(next(f))
                    except StopIteration:
                        break
                return "".join(out)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                out = []
                for _ in range(n_lines):
                    try:
                        out.append(next(f))
                    except StopIteration:
                        break
                return "".join(out)
    except Exception as e:
        return f"[Gagal baca head: {e}]"


def _detect_delimiter(path: Path) -> str:
    candidates = [",", ";", "\t", "|"]
    sample = ""
    try:
        name = path.name.lower()
        if name.endswith((".csv.gz", ".gz")):
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
            compression="gzip" if name.endswith((".csv.gz", ".gz")) else None,
            encoding_errors="replace",
            on_bad_lines="skip",
            low_memory=False,
        )

    raise ValueError(f"Format tidak didukung: {path.name}")


@st.cache_data(show_spinner=False)
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
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().strip())


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


def ensure_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")


def to_int_safe(s: pd.Series, default: int = 0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(default).astype(int)


# ----------------------------
# TARGET (FIX BUG: "Tidak tertarik" jangan dianggap "tertarik")
# ----------------------------
def _norm_answer(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def make_target(df_in: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df_in.copy()
    col = pick_col(df, ["Penyaluran Kerja", "Penyaluran_kerja", "Penyaluran", "Placement", "Placed"])
    if not col:
        return df

    s = ensure_str_series(df[col]).map(_norm_answer)

    if mode == "Interest (exact Tertarik vs Tidak/Belum)":
        # Positive ONLY kalau jawabannya memang "tertarik" (atau typo dekat)
        # dan bukan bentuk "tidak tertarik" / "belum tertarik"
        neg_tokens = ("tidak", "belum", "-", "masih dipertimbangkan", "pertimbangkan")
        is_neg = s.apply(lambda z: any(t in z for t in neg_tokens))
        is_pos = s.isin(["tertarik", "tetarik", "ttert", "ttert", "t ert"]) | s.str.fullmatch(r"tertarik+")
        df["Penyaluran_flag"] = ((is_pos) & (~is_neg)).astype(int)

    else:
        # Strict placed/tersalur logic (lebih jarang ada di CSV ini)
        # Positive: ya/sudah/placed/tersalur/berhasil
        # Negative: contains "tidak"/"belum" override
        is_yes = s.map(lambda x: 1 if YES_PATTERN.search(str(x)) else 0).astype(int)
        is_neg = s.str.contains(r"\b(tidak|belum)\b", regex=True, na=False).astype(int)
        df["Penyaluran_flag"] = ((is_yes == 1) & (is_neg == 0)).astype(int)

    df["Penyaluran_label"] = np.where(df["Penyaluran_flag"] == 1, "Positive", "Negative")
    return df


# ----------------------------
# FEATURE ENGINEERING (selaras PDF + cocok dengan raw_data.csv kamu)
# ----------------------------
def map_motivasi_cluster(val: str) -> str:
    v = str(val).lower()
    if "dapat kerja" in v or "bekerja" in v:
        return "Dapat kerja"
    if "belajar" in v and "skill" in v:
        return "Belajar skill"
    if "upgrade" in v:
        return "Upgrade diri"
    if "freelance" in v:
        return "Freelance"
    if "switch career" in v or "switch karir" in v:
        return "Switch career"
    return "Lainnya"


def map_motivasi_risk_flag(cluster: str) -> str:
    return "High risk" if cluster == "Dapat kerja" else "Low risk"


def map_domain(val: str) -> str:
    v = str(val)
    if DOMAIN_DATA.search(v):
        return "Data/IT"
    if DOMAIN_DESIGN.search(v):
        return "Design"
    if DOMAIN_BIZ.search(v):
        return "Business"
    return "Other"


def fe_core(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Umur + bin
    col_umur = pick_col(df, ["Umur", "Age"])
    if col_umur:
        df["Umur"] = pd.to_numeric(df[col_umur], errors="coerce")
        bins = [0, 22, 25, 30, 100]
        labels = ["<=22", "23-25", "26-30", "30+"]
        df["Umur_bin"] = pd.cut(df["Umur"], bins=bins, labels=labels, include_lowest=True)

    # Region (Kota(Jabodetabek) already exists)
    col_kota_j = pick_col(df, ["Kota(Jabodetabek)"])
    if col_kota_j:
        s = ensure_str_series(df[col_kota_j])
        df["Region"] = np.where(
            s.str.lower().str.contains("jabodetabek", na=False),
            "Jabodetabek",
            "Luar Jabodetabek",
        )

    # Batch features
    col_batch = pick_col(df, ["Batch"])
    if col_batch:
        s = ensure_str_series(df[col_batch])
        df["Batch_num"] = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
        df["Batch_has_plus"] = s.str.contains(r"\+", regex=True).astype(int)

    # Jobconnect flag (Kategori has "+")
    col_kategori = pick_col(df, ["Kategori"])
    if col_kategori:
        s = ensure_str_series(df[col_kategori])
        df["Program_jobconnect_flag"] = s.str.contains(r"\+", regex=True).astype(int)

    # Community_flag
    col_comm = pick_col(df, ["Community"])
    if col_comm:
        s = ensure_str_series(df[col_comm])
        df["Community_flag"] = (~s.str.lower().str.contains("tidak", na=False)).astype(int)

    # Event_flag
    col_event = pick_col(df, ["Pernah ikut acara dibimbing/tidak"])
    if col_event:
        s = ensure_str_series(df[col_event])
        df["Event_flag"] = s.str.lower().str.contains(r"(pernah|sudah|ya)", regex=True, na=False).astype(int)

    # Engagement level
    if all(c in df.columns for c in ["Community_flag", "Event_flag", "Program_jobconnect_flag"]):
        active_count = df[["Community_flag", "Event_flag", "Program_jobconnect_flag"]].sum(axis=1)
        passive_count = (1 - df["Community_flag"]) + (1 - df["Event_flag"]) + (1 - df["Program_jobconnect_flag"])
        df["Engagement_level"] = np.select(
            [active_count >= 2, (active_count == 1) & (passive_count <= 2), passive_count >= 2],
            ["High", "Medium", "Low"],
            default="Low",
        )

    # Motivasi
    col_motiv = pick_col(df, ["Motivasi mengikuti bootcamp", "Motivasi utama"])
    if col_motiv:
        mc = ensure_str_series(df[col_motiv]).map(map_motivasi_cluster)
        df["Motivasi_cluster"] = mc
        df["Motivasi_risk_flag"] = mc.map(map_motivasi_risk_flag)

    # Segmen_karir: compare Domain pendidikan vs Domain "product"
    col_jur = pick_col(df, ["Jurusan pendidikan", "Jurusan"])
    col_prod = pick_col(df, ["Product"])
    if col_jur:
        df["Domain_pendidikan"] = ensure_str_series(df[col_jur]).map(map_domain)
    if col_prod:
        df["Domain_product"] = ensure_str_series(df[col_prod]).map(map_domain)

    if ("Domain_pendidikan" in df.columns) and ("Domain_product" in df.columns):
        dp = df["Domain_pendidikan"].astype("string").fillna("Other")
        dpr = df["Domain_product"].astype("string").fillna("Other")
        df["is_switcher"] = ((dp != dpr) & (dp != "Other") & (dpr != "Other")).astype(int)
        df["Segmen_karir"] = df["is_switcher"].map({0: "Upskiller", 1: "Career Switcher"}).astype("string")

    return df


# ----------------------------
# Sampling (KEEP ALL POSITIVES)
# ----------------------------
def sample_keep_all_positives(df: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    if target not in df.columns:
        return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)

    y = to_int_safe(df[target], 0).clip(0, 1)
    pos = df[y == 1]
    neg = df[y == 0]

    if len(df) <= n:
        return df.reset_index(drop=True)

    remain = max(n - len(pos), 0)
    if remain <= 0:
        return pos.sample(min(n, len(pos)), random_state=seed).reset_index(drop=True)

    neg_s = neg.sample(min(remain, len(neg)), random_state=seed)
    out = pd.concat([pos, neg_s], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# ----------------------------
# EDA helpers
# ----------------------------
def freq_table(s: pd.Series) -> pd.DataFrame:
    x = s.astype("string").fillna("Unknown")
    vc = x.value_counts(dropna=False)
    out = vc.rename_axis("Kategori").reset_index(name="Jumlah")
    out["Persentase"] = (out["Jumlah"] / out["Jumlah"].sum()) * 100.0
    return out


def chart_dist_horizontal(df_freq: pd.DataFrame, title: str) -> alt.Chart:
    h = min(760, max(240, 28 * len(df_freq)))
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
    text = base.mark_text(align="left", dx=6, color="white").encode(text=alt.Text("Persentase:Q", format=".1f"))
    return base + text


def placement_rate_table(df: pd.DataFrame, by_col: str, target_col: str) -> pd.DataFrame:
    g = df[[by_col, target_col]].copy()
    g[by_col] = g[by_col].astype("string").fillna("Unknown")
    g[target_col] = to_int_safe(g[target_col], 0).clip(0, 1)

    agg = g.groupby(by_col, dropna=False)[target_col].agg(["count", "sum"]).reset_index()
    agg = agg.rename(columns={by_col: "Group", "count": "Total", "sum": "Positives"})
    agg["Rate_pct"] = (agg["Positives"] / agg["Total"]) * 100.0
    return agg.sort_values("Rate_pct", ascending=False).reset_index(drop=True)


def placement_rate_bar(
    tab: pd.DataFrame, title: str, zoom_max_pct: float, top_n: int, auto_zoom: bool = True
) -> alt.Chart:
    tab = tab.head(int(top_n)).copy()
    h = min(760, max(240, 28 * len(tab)))

    if auto_zoom:
        mx = float(tab["Rate_pct"].max()) if len(tab) else 1.0
        zoom_max_pct = max(0.1, min(100.0, mx * 1.15))

    base = (
        alt.Chart(tab)
        .mark_bar()
        .encode(
            y=alt.Y("Group:N", sort="-x", title=None),
            x=alt.X("Rate_pct:Q", title="Placement/Interest rate (%)", scale=alt.Scale(domain=[0, float(zoom_max_pct)])),
            tooltip=["Group:N", "Total:Q", "Positives:Q", alt.Tooltip("Rate_pct:Q", format=".3f")],
        )
        .properties(height=h, title=title)
    )
    text = base.mark_text(align="left", dx=6, color="white").encode(text=alt.Text("Rate_pct:Q", format=".3f"))
    return base + text


# ----------------------------
# Modeling helpers
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

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", ohe)])

    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")


def select_feature_cols(df: pd.DataFrame) -> List[str]:
    want = [
        "Umur",
        "Umur_bin",
        "Region",
        "Batch_num",
        "Batch_has_plus",
        "Community_flag",
        "Event_flag",
        "Engagement_level",
        "Program_jobconnect_flag",
        "Motivasi_cluster",
        "Motivasi_risk_flag",
        "Domain_pendidikan",
        "Domain_product",
        "is_switcher",
        "Segmen_karir",
        # raw signal columns (optional)
        "Product",
        "Kategori",
        "Month",
        "Channel",
        "Level pendidikan",
        "Kategori Kesibukan",
        "Level Pekerjaan",
        "Kategori Pekerjaan",
        "Domisili",
        "Provinsi",
        "Negara",
    ]
    return [c for c in want if c in df.columns]


@dataclass
class ClusterOut:
    k: int
    labeled: pd.DataFrame
    svd2d: pd.DataFrame
    profile: pd.DataFrame
    persona_map: Dict[int, str]
    silhouette: Optional[float] = None


def _persona_name_from_profile(row: pd.Series) -> str:
    # Heuristik mirip yang “nyambung” ke PDF/notebook kamu
    # 1) High Engagement Career Switcher
    if row.get("Engagement_High_share", 0) >= 0.45 and row.get("Switcher_share", 0) >= 0.45:
        return "High Engagement Career Switcher"
    # 2) Fresh Graduate Explorer (mahasiswa/fresh grad dominan atau usia muda)
    if row.get("Kesibukan_FG_Mhs_share", 0) >= 0.45 or row.get("Age_u25_share", 0) >= 0.45:
        return "Fresh Graduate Explorer"
    # 3) Default
    return "Working Professional Upskiller"


def profile_clusters(df: pd.DataFrame, cluster_col: str = "cluster_id") -> Tuple[pd.DataFrame, Dict[int, str]]:
    out_rows = []
    for cid, g in df.groupby(cluster_col):
        n = len(g)

        def share(col, val):
            if col not in g.columns:
                return 0.0
            s = g[col].astype("string").fillna("Unknown")
            return float((s == val).mean())

        # Engagement share
        eng_high = share("Engagement_level", "High")
        # Switcher share
        sw = float(g["is_switcher"].fillna(0).astype(int).mean()) if "is_switcher" in g.columns else 0.0
        # usia muda share
        if "Umur" in g.columns:
            age_u25 = float((pd.to_numeric(g["Umur"], errors="coerce") <= 25).mean())
        else:
            age_u25 = 0.0

        # kategori kesibukan FG/Mhs
        if "Kategori Kesibukan" in g.columns:
            kk = g["Kategori Kesibukan"].astype("string").fillna("Unknown").str.lower()
            fg_mhs = float(kk.isin(["mahasiswa", "fresh graduate"]).mean())
        else:
            fg_mhs = 0.0

        # top categories for explanation
        def top_cat(col):
            if col not in g.columns:
                return "NA"
            return g[col].astype("string").fillna("Unknown").value_counts().head(1).index[0]

        out_rows.append(
            {
                "cluster_id": int(cid),
                "n": int(n),
                "Engagement_High_share": eng_high,
                "Switcher_share": sw,
                "Age_u25_share": age_u25,
                "Kesibukan_FG_Mhs_share": fg_mhs,
                "Top_Product": top_cat("Product"),
                "Top_Kategori": top_cat("Kategori"),
                "Top_Motivasi": top_cat("Motivasi_cluster"),
                "Top_Segmen": top_cat("Segmen_karir"),
            }
        )

    prof = pd.DataFrame(out_rows).sort_values("cluster_id").reset_index(drop=True)
    persona_map = {int(r["cluster_id"]): _persona_name_from_profile(r) for _, r in prof.iterrows()}
    return prof, persona_map


@st.cache_data(show_spinner=False)
def fit_cluster_fixed_k(df_in: pd.DataFrame, feature_cols: List[str], k: int, seed: int, sil_sample: int) -> ClusterOut:
    X = df_in[feature_cols].copy()
    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    km = MiniBatchKMeans(n_clusters=int(k), random_state=seed, batch_size=2048, n_init="auto")
    pipe = Pipeline([("prep", prep), ("km", km)])
    pipe.fit(X)

    Xt = pipe.named_steps["prep"].transform(X)
    labels = pipe.named_steps["km"].labels_

    # silhouette (optional)
    ss = min(int(sil_sample), Xt.shape[0])
    sil = None
    try:
        if int(k) >= 2 and Xt.shape[0] >= 3:
            sil = float(silhouette_score(Xt, labels, sample_size=ss, random_state=seed))
    except Exception:
        sil = None

    svd = TruncatedSVD(n_components=2, random_state=seed)
    xy = svd.fit_transform(Xt)
    svd2d = pd.DataFrame({"SVD1": xy[:, 0], "SVD2": xy[:, 1], "cluster_id": labels})

    labeled = df_in.copy()
    labeled["cluster_id"] = labels

    prof, persona_map = profile_clusters(labeled, "cluster_id")
    return ClusterOut(k=int(k), labeled=labeled, svd2d=svd2d, profile=prof, persona_map=persona_map, silhouette=sil)


def chart_svd_with_persona(df2d: pd.DataFrame, persona_map: Dict[int, str]) -> alt.Chart:
    d = df2d.copy()
    d["persona"] = d["cluster_id"].map(lambda x: persona_map.get(int(x), f"Cluster {int(x)}"))
    return (
        alt.Chart(d)
        .mark_circle(size=35, opacity=0.75)
        .encode(
            x=alt.X("SVD1:Q"),
            y=alt.Y("SVD2:Q"),
            color=alt.Color("persona:N", title="Persona"),
            tooltip=[
                alt.Tooltip("persona:N"),
                alt.Tooltip("cluster_id:Q"),
                alt.Tooltip("SVD1:Q", format=".3f"),
                alt.Tooltip("SVD2:Q", format=".3f"),
            ],
        )
        .properties(height=520, title="Cluster Visualization (TruncatedSVD 2D) — Persona Named (PDF-aligned)")
        .interactive()
    )


@dataclass
class SupOut:
    pr_auc: float
    pr_curve: pd.DataFrame
    score_df: pd.DataFrame
    topk_table: pd.DataFrame
    lift_table: pd.DataFrame
    model: Pipeline
    prevalence: float
    total_pos: int


def topk_rates_table(y_true, y_score, K_list: List[int]) -> Tuple[pd.DataFrame, int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        raise ValueError("total_positive = 0 di y_true. Top-K tidak bisa dihitung.")
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
    use_grid: bool,
) -> SupOut:
    df = df_in.copy()
    y = to_int_safe(df[target], 0).clip(0, 1)
    X = df[feature_cols].copy()

    vc = y.value_counts()
    if len(vc) < 2 or vc.min() < 2:
        raise ValueError(f"Class minoritas terlalu kecil untuk split (min_count={int(vc.min()) if len(vc) else 0}).")

    num_cols, cat_cols = split_num_cat(X)
    prep = make_preprocess(num_cols, cat_cols)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    base = Pipeline(
        [
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        ]
    )

    if use_grid:
        grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
    else:
        best = base
        best.fit(Xtr, ytr)

    y_proba = best.predict_proba(Xte)[:, 1]
    pr_auc = float(average_precision_score(yte, y_proba))
    prec, rec, _ = precision_recall_curve(yte, y_proba)
    pr_curve = pd.DataFrame({"recall": rec, "precision": prec})

    score_df = pd.DataFrame({"y_true": np.asarray(yte).astype(int), "proba": y_proba}).sort_values(
        "proba", ascending=False
    ).reset_index(drop=True)

    N = len(score_df)
    # K list (PDF-ish)
    K_list = sorted(
        set(
            [
                10,
                20,
                50,
                100,
                200,
                int(0.005 * N),
                int(0.01 * N),
                int(0.02 * N),
                int(0.05 * N),
                int(0.1 * N),
                int(0.2 * N),
            ]
        )
    )
    K_list = [k for k in K_list if k > 0]
    topk_df, total_pos = topk_rates_table(score_df["y_true"], score_df["proba"], K_list)

    prevalence = total_pos / N
    lift = topk_df.copy()
    lift["Prevalence"] = prevalence
    lift["Lift@K"] = lift["Precision@K"] / prevalence
    lift["Random Precision"] = prevalence
    lift["Random Lift"] = 1.0

    return SupOut(
        pr_auc=pr_auc,
        pr_curve=pr_curve,
        score_df=score_df,
        topk_table=topk_df,
        lift_table=lift,
        model=best,
        prevalence=float(prevalence),
        total_pos=int(total_pos),
    )


def chart_pr_curve(pr_curve: pd.DataFrame, pr_auc: float) -> alt.Chart:
    return (
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


def chart_topk_capture(score_df: pd.DataFrame, K_list: List[int]) -> alt.Chart:
    df = score_df.copy()
    df["rank"] = np.arange(1, len(df) + 1)
    df["cum_hits"] = df["y_true"].cumsum()
    kset = set(K_list)
    df_k = df[df["rank"].isin(kset)].copy()

    line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("rank:Q", title="Top-K (jumlah peserta diprioritaskan)"),
            y=alt.Y("cum_hits:Q", title="Jumlah positif tertangkap (cumulative)"),
            tooltip=["rank", "cum_hits", "y_true", alt.Tooltip("proba:Q", format=".6f")],
        )
    )
    pts = alt.Chart(df_k).mark_circle(size=70).encode(x="rank:Q", y="cum_hits:Q")
    return (line + pts).properties(height=320, title="Top-K Capture Curve (Holdout) — PDF-aligned").interactive()


def chart_precision_recall_at_k(topk: pd.DataFrame) -> alt.Chart:
    df = topk.copy()
    df_long = df.melt(id_vars=["K"], value_vars=["Recall@K", "Precision@K"], var_name="Metric", value_name="Rate")
    return (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("Rate:Q", title="Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", title=None),
            tooltip=["K", "Metric", alt.Tooltip("Rate:Q", format=".3f")],
        )
        .properties(height=320, title="Precision@K & Recall@K — PDF-aligned")
        .interactive()
    )


def chart_lift(lift: pd.DataFrame) -> alt.Chart:
    df = lift.copy()
    base = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("K:Q", title="K (Top-K)"),
            y=alt.Y("Lift@K:Q", title="Lift@K"),
            tooltip=[
                "K",
                alt.Tooltip("Lift@K:Q", format=".2f"),
                alt.Tooltip("Recall@K:Q", format=".3f"),
                alt.Tooltip("Precision@K:Q", format=".3f"),
            ],
        )
        .properties(height=320, title="Lift@K (vs Random) — PDF-aligned")
        .interactive()
    )
    baseline = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
    return base + baseline


# ----------------------------
# Sidebar: Data load + global controls
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
            st.error("Tidak ada file data terdeteksi di repo. Pakai Upload.")
            st.stop()

        chosen = st.selectbox("Pilih file data di repo:", rels, index=0)
        full_path = Path(repo_root) / chosen

        st.caption("Debug file")
        try:
            st.write("Path:", chosen)
            st.write("Size (bytes):", full_path.stat().st_size)
        except Exception as e:
            st.error(f"Gagal akses file: {e}")
            st.stop()

        if full_path.name.lower().endswith((".csv", ".csv.gz", ".gz")):
            st.code(sniff_file_head(full_path, n_lines=10))

        try:
            df_raw = read_table(full_path)
            found_path = str(full_path)
            st.success(f"Loaded: {chosen}")
        except Exception as e:
            st.error(f"Gagal baca file: {chosen}\nError: {e}")
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
                elif name.endswith((".gz", ".csv.gz")):
                    df_raw = pd.read_csv(f, compression="gzip", encoding_errors="replace", on_bad_lines="skip")
                else:
                    st.error("Upload CSV / XLSX / GZ.")
                    st.stop()
                found_path = "uploaded"
            except Exception as e:
                st.error(f"Gagal baca upload: {e}")
                st.stop()

    st.divider()
    st.header("Core controls")
    seed = st.number_input("random_state", value=42, step=1)

    target_mode = st.selectbox(
        "Target definition",
        ["Interest (exact Tertarik vs Tidak/Belum)", "Strict placed/tersalur (regex)"],
        index=0,
    )

    use_sample = st.toggle("Use sample for training/plots?", value=True)
    sample_n = st.number_input("Sample size (keep all positives)", value=2500, min_value=500, step=250)

    st.divider()
    st.header("Pipelines")
    run_eda = st.toggle("EDA", value=True)
    run_cluster = st.toggle("Clustering", value=True)
    run_sup = st.toggle("Supervised Top-K", value=True)

    st.divider()
    st.header("Clustering options")
    k_fixed = st.number_input("k (fixed, PDF default=3)", value=3, min_value=2, step=1)
    sil_sample = st.number_input("Silhouette sample_size", value=2000, min_value=300, step=200)

    st.divider()
    st.header("Supervised options")
    test_size = st.slider("test_size", 0.05, 0.5, 0.2)
    use_grid = st.toggle("GridSearch C (slower)", value=False)


# ----------------------------
# Main
# ----------------------------
st.title("Persona Predict (PDF-aligned)")
if df_raw is None:
    st.info("Pilih Repo file / Upload dulu.")
    st.stop()

st.caption(f"Loaded from: {found_path}")

df0 = df_raw.copy()
df0.columns = [str(c).strip() for c in df0.columns]

df1 = make_target(df0, mode=target_mode)
df1 = fe_core(df1)

# Global Filters (lebih lengkap)
with st.sidebar:
    st.divider()
    st.header("Global filters (optional)")
    st.caption("Filter diterapkan sebelum EDA/Model. Kosong = no filter.")

    def _ms(col: str) -> Optional[List[str]]:
        if col in df1.columns:
            return sorted(df1[col].astype("string").fillna("Unknown").unique().tolist())
        return None

    filters = {}
    for col in ["Month", "Product", "Kategori", "Channel", "Batch", "Level pendidikan", "Kategori Kesibukan", "Level Pekerjaan"]:
        opts = _ms(col)
        if opts:
            filters[col] = st.multiselect(col, opts)

mask = pd.Series(True, index=df1.index)
for col, selected in filters.items():
    if selected:
        mask &= df1[col].astype("string").fillna("Unknown").isin(selected)

df = df1.loc[mask].reset_index(drop=True)

# Sampling for work df
df_work = df
if use_sample and len(df_work) > int(sample_n):
    if "Penyaluran_flag" in df_work.columns:
        df_work = sample_keep_all_positives(df_work, "Penyaluran_flag", int(sample_n), int(seed))
        st.warning(f"Using sample (keep all positives): {len(df_work):,} rows (full after filter: {len(df):,}).")
    else:
        df_work = df_work.sample(int(sample_n), random_state=int(seed)).reset_index(drop=True)
        st.warning(f"Using sample: {len(df_work):,} rows (full after filter: {len(df):,}).")

# Objective text (selaras tujuan)
with st.expander("🎯 Objective (selaras dokumen)", expanded=True):
    st.markdown(
        """
Proyek ini bertujuan untuk:
1) **Memahami pola peserta** (EDA & segment breakdown)
2) **Membangun segmentasi persona yang akurat** (clustering + profiling + persona naming)
3) **Mengembangkan model prediktif peluang penyaluran/interest** untuk mendukung keputusan bisnis berbasis data  
   (misal: strategi akuisisi, desain program, intervensi peserta) melalui **ranking Top-K**.
"""
    )

tabs = st.tabs(["📌 Overview", "🔎 EDA (Business-aligned)", "🧩 Clustering (Persona)", "🎯 Supervised (Top-K)"])

# ----------------------------
# Overview
# ----------------------------
with tabs[0]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Preview (work)")
        st.dataframe(df_work.head(30), use_container_width=True)
    with c2:
        st.subheader("Quick stats")
        st.metric("Rows (work)", f"{len(df_work):,}")
        st.metric("Rows (full after filter)", f"{len(df):,}")
        st.metric("Columns", f"{df.shape[1]:,}")

        if "Penyaluran_flag" in df_work.columns:
            vc = df_work["Penyaluran_flag"].value_counts(dropna=False)
            st.write("Target counts (work):", vc.to_dict())

        st.divider()
        st.write("FE columns status:")
        fe_cols = ["Motivasi_cluster", "Segmen_karir", "Engagement_level", "Program_jobconnect_flag", "Region", "Umur_bin"]
        st.write({c: (c in df_work.columns) for c in fe_cols})

        st.divider()
        # Show raw answer distribution for sanity (ini yang bikin target nggak lagi 100%)
        col_ans = pick_col(df_work, ["Penyaluran Kerja"])
        if col_ans:
            top = df_work[col_ans].astype("string").fillna("Unknown").value_counts().head(12)
            st.write("Top Penyaluran Kerja (sanity):")
            st.dataframe(top.rename_axis("value").reset_index(name="count"), use_container_width=True, hide_index=True)

# ----------------------------
# EDA (lebih matang, bisa multi breakdown)
# ----------------------------
with tabs[1]:
    if not run_eda:
        st.info("EDA dimatikan dari sidebar.")
    else:
        st.header("EDA — selaras tujuan bisnis (filterable + multi-breakdown)")

        target = "Penyaluran_flag"
        if target not in df_work.columns:
            st.warning("Target Penyaluran_flag tidak tersedia → rate charts akan terbatas.")
        else:
            prev = float(df_work[target].mean()) * 100
            st.metric("Overall target rate (work)", f"{prev:.2f}%")

        st.divider()
        st.subheader("1) Distribusi peserta (komposisi)")
        dist_cols = [c for c in [
            "Product", "Kategori", "Channel", "Month",
            "Kategori Kesibukan", "Level Pekerjaan", "Level pendidikan",
            "Motivasi_cluster", "Segmen_karir", "Region", "Umur_bin",
        ] if c in df_work.columns]

        pick_dist = st.multiselect("Pilih kolom distribusi (bisa lebih dari 1)", dist_cols, default=[c for c in ["Product","Kategori","Channel"] if c in dist_cols])
        topn_dist = st.slider("Top-N (distribution)", 5, 40, 15)

        for col in pick_dist:
            f = freq_table(df_work[col]).head(int(topn_dist))
            st.altair_chart(chart_dist_horizontal(f, f"Distribusi peserta: {col} (Top-{topn_dist})"), use_container_width=True)
            st.dataframe(f, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("2) Target/Placement/Interest rate per segmen (inti decision support)")

        if target in df_work.columns:
            rate_cols = [c for c in dist_cols if c != target]
            pick_rate = st.multiselect(
                "Pilih breakdown rate (bisa lebih dari 1)",
                rate_cols,
                default=[c for c in ["Motivasi_cluster", "Segmen_karir", "Region", "Product", "Kategori"] if c in rate_cols],
            )
            cA, cB, cC = st.columns([1, 1, 1])
            with cA:
                topn_rate = st.slider("Top-N (rate)", 5, 40, 20)
            with cB:
                auto_zoom = st.toggle("Auto-zoom", value=True)
            with cC:
                zoom_manual = st.number_input("Zoom max (%) manual", value=5.0, min_value=0.1, step=0.5)

            for col in pick_rate:
                tab = placement_rate_table(df_work, col, target)
                chart = placement_rate_bar(
                    tab, f"Target rate by {col} (Top-{topn_rate})",
                    zoom_max_pct=float(zoom_manual), top_n=int(topn_rate), auto_zoom=bool(auto_zoom),
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(tab.head(int(topn_rate)), use_container_width=True, hide_index=True)

# ----------------------------
# Clustering (k fixed=3 + persona naming)
# ----------------------------
with tabs[2]:
    if not run_cluster:
        st.info("Clustering dimatikan dari sidebar.")
    else:
        st.header("Clustering — Persona segmentation (PDF-aligned)")

        feat_cols = select_feature_cols(df_work)
        if not feat_cols:
            st.error("Fitur untuk clustering tidak cukup. Pastikan FE kebentuk.")
        else:
            st.caption(f"Fitur clustering dipakai: {feat_cols}")

            with st.form("cluster_form"):
                submitted = st.form_submit_button("Run clustering (fixed k)", use_container_width=True)

            if submitted:
                try:
                    cl = fit_cluster_fixed_k(df_work, feat_cols, int(k_fixed), int(seed), int(sil_sample))
                    st.session_state["cluster_out"] = cl
                    st.session_state["df_with_cluster"] = cl.labeled
                    st.success(f"Clustering done. k={cl.k} | silhouette={cl.silhouette if cl.silhouette is not None else 'NA'}")
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

            if "cluster_out" in st.session_state:
                cl: ClusterOut = st.session_state["cluster_out"]
                st.subheader("Cluster visualization (2D)")
                st.altair_chart(chart_svd_with_persona(cl.svd2d, cl.persona_map), use_container_width=True)

                st.subheader("Cluster profiling (untuk penamaan persona)")
                prof = cl.profile.copy()
                # prettier shares
                for c in ["Engagement_High_share", "Switcher_share", "Age_u25_share", "Kesibukan_FG_Mhs_share"]:
                    if c in prof.columns:
                        prof[c] = (prof[c] * 100).map(lambda x: f"{x:.1f}%")
                prof["Persona_name"] = prof["cluster_id"].map(lambda x: cl.persona_map.get(int(x), f"Cluster {int(x)}"))
                st.dataframe(prof, use_container_width=True, hide_index=True)

# ----------------------------
# Supervised Top-K (mature dashboard)
# ----------------------------
with tabs[3]:
    if not run_sup:
        st.info("Supervised dimatikan dari sidebar.")
    else:
        st.header("Supervised ranking — Top-K (PDF-aligned)")

        df_sup = st.session_state.get("df_with_cluster", df_work)
        target = "Penyaluran_flag"

        if target not in df_sup.columns:
            st.error("Target Penyaluran_flag tidak ada. Pastikan kolom 'Penyaluran Kerja' ter-detect.")
        else:
            vc = df_sup[target].value_counts(dropna=False)
            st.write("Target counts (work/sup):", vc.to_dict())

            feat_cols = select_feature_cols(df_sup)
            if "cluster_id" in df_sup.columns and "cluster_id" not in feat_cols:
                feat_cols = feat_cols + ["cluster_id"]

            if not feat_cols:
                st.error("Fitur supervised tidak cukup. Pastikan FE kebentuk.")
            else:
                st.caption(f"Fitur supervised dipakai: {feat_cols}")

                with st.form("sup_form"):
                    cap_k = st.number_input("Business capacity K (berapa peserta yang bisa diintervensi)", value=200, min_value=10, step=10)
                    submitted = st.form_submit_button("Run supervised ranking", use_container_width=True)

                if submitted:
                    try:
                        sup = fit_supervised(df_sup, target, feat_cols, float(test_size), int(seed), bool(use_grid))
                        st.session_state["sup_out"] = sup
                        st.session_state["cap_k"] = int(cap_k)
                        st.success("Supervised ranking done.")
                    except Exception as e:
                        st.error(str(e))

                if "sup_out" in st.session_state:
                    sup: SupOut = st.session_state["sup_out"]
                    cap_k = int(st.session_state.get("cap_k", 200))

                    st.subheader("Why Top-K? (Decision support)")
                    st.markdown(
                        f"""
- **Prevalence (positif di data holdout)** ≈ **{sup.prevalence*100:.2f}%**
- Jika tim hanya mampu follow-up **K={cap_k}** peserta, maka kita butuh **ranking** untuk memaksimalkan:
  - **Precision@K** (berapa % yang benar-benar positif di Top-K)
  - **Recall@K** (berapa % dari semua positif yang berhasil tertangkap)
  - **Lift@K** (seberapa lebih baik dibanding random)
"""
                    )

                    st.altair_chart(chart_pr_curve(sup.pr_curve, sup.pr_auc), use_container_width=True)

                    # TopK capture plot
                    K_list = sup.topk_table["K"].astype(int).tolist()
                    st.altair_chart(chart_topk_capture(sup.score_df, K_list), use_container_width=True)

                    st.subheader("Top-K table (PDF-style)")
                    show = sup.topk_table.copy()
                    show["Precision@K"] = (show["Precision@K"] * 100).map(lambda x: f"{x:.2f}%")
                    show["Recall@K"] = (show["Recall@K"] * 100).map(lambda x: f"{x:.2f}%")
                    st.dataframe(show[["K", "Positives captured", "Precision@K", "Recall@K"]], use_container_width=True, hide_index=True)

                    st.altair_chart(chart_precision_recall_at_k(sup.topk_table), use_container_width=True)
                    st.altair_chart(chart_lift(sup.lift_table), use_container_width=True)

                    # KPI for chosen capacity K
                    # find nearest K row
                    kk = int(cap_k)
                    idx = (sup.lift_table["K"] - kk).abs().idxmin()
                    row = sup.lift_table.loc[idx]
                    st.subheader("KPI untuk kapasitas K yang dipilih")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision@K", f"{row['Precision@K']*100:.2f}%")
                    c2.metric("Recall@K", f"{row['Recall@K']*100:.2f}%")
                    c3.metric("Lift@K", f"{row['Lift@K']:.2f}x")

                    # Optional: show top K rows (identifikasi) – pakai index + beberapa kolom penting
                    st.subheader("Preview Top-K candidates (untuk intervensi)")
                    topk = sup.score_df.head(int(row["K"])).copy()
                    topk["rank"] = np.arange(1, len(topk) + 1)
                    st.dataframe(topk[["rank", "proba", "y_true"]].head(50), use_container_width=True)

st.caption("Hasil clustering & supervised disimpan di session_state, jadi tidak hilang setelah rerun/klik lain.")
