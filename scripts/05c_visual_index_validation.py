import os
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import pandas as pd
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
TOPIC_COLS = [f"topic_{k}" for k in range(1, 9)]
FALLBACK_BETAS = {
    "topic_1": -0.0060,
    "topic_2": -0.0092,
    "topic_3": -0.0012,
    "topic_4": -0.0031,
    "topic_5": +0.0120,
    "topic_6": -0.0043,
    "topic_7": +0.0018,
    "topic_8": +0.0021,
}
GSV_ROOT = os.path.join("data", "images", "gsv")
PREFERRED_HEADINGS = [0, 90, 180]
COLOR_TOP = "#1a7a4a"
COLOR_BOT = "#8b1a1a"


# ── Step 0 — Load Ridge M1 NMF coefficients ───────────────────────────────────
def load_betas() -> dict:
    path = os.path.join("reports", "mejores_coefs.csv")
    try:
        df = pd.read_csv(path, encoding="utf-8")
        df.columns = df.columns.str.strip().str.lower()
        betas = {}
        # Format A: long table with 'variable'/'feature' + 'coef' columns
        name_col = next(
            (c for c in df.columns if c in ("variable", "feature", "name", "term")),
            None,
        )
        val_col = next(
            (c for c in df.columns if c in ("coef", "coefficient", "value", "beta")),
            None,
        )
        if name_col and val_col:
            for _, row in df.iterrows():
                feat = str(row[name_col]).strip().lower()
                if feat in TOPIC_COLS:
                    betas[feat] = float(row[val_col])
        # Format B: wide row where columns are topic names
        if len(betas) < 8 and set(TOPIC_COLS).issubset(set(df.columns)):
            row = df.iloc[0]
            betas = {t: float(row[t]) for t in TOPIC_COLS}
        if len(betas) == 8:
            print(f"[betas] loaded from {path}: {betas}")
            return betas
        print(f"[betas] could not parse {path} (found {len(betas)}/8) — using fallback")
    except Exception as e:
        print(f"[betas] error reading coefficients: {e} — using fallback values")
    return FALLBACK_BETAS


# ── Step 1 — Compute v_j ───────────────────────────────────────────────────────
def compute_vj(nmf: pd.DataFrame, betas: dict) -> pd.DataFrame:
    vj_series = sum(betas[t] * nmf[t] for t in TOPIC_COLS)
    out = nmf[["id_establecimiento"]].copy()
    out["vj"] = vj_series.values
    out = out.sort_values("vj", ascending=False).reset_index(drop=True)
    out["vj_rank"] = out["vj"].rank(ascending=False, method="first").astype(int)
    out["vj_quintile"] = pd.qcut(out["vj"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    print(
        f"[vj] n={len(out)}  range: {out['vj'].min():.4f} – {out['vj'].max():.4f}"
    )
    return out


# ── Step 2 — Pick top-5 / bottom-5 ────────────────────────────────────────────
def pick_extremes(vj_df: pd.DataFrame):
    top5 = vj_df[vj_df["vj_rank"] <= 5].sort_values("vj_rank")
    n = len(vj_df)
    bot5 = vj_df[vj_df["vj_rank"] > n - 5].sort_values("vj_rank", ascending=False)
    print("[top5] ids:", top5["id_establecimiento"].tolist())
    print("[bot5] ids:", bot5["id_establecimiento"].tolist())
    return top5.reset_index(drop=True), bot5.reset_index(drop=True)


# ── Step 3 — Find best image for a college ────────────────────────────────────
def find_image(estab_id, catalog: pd.DataFrame) -> str | None:
    subset = catalog[
        (catalog["id_establecimiento"] == estab_id)
        & (catalog["descargada"] == True)
    ]
    if subset.empty:
        return None
    for h in PREFERRED_HEADINGS:
        match = subset[subset["heading"] == h]
        if not match.empty:
            full = os.path.join(GSV_ROOT, match.iloc[0]["ruta_archivo"])
            if os.path.isfile(full):
                return full
    for _, row in subset.iterrows():
        full = os.path.join(GSV_ROOT, row["ruta_archivo"])
        if os.path.isfile(full):
            return full
    return None


def short_name(name: str, max_len: int = 20) -> str:
    name = re.sub(r"\bCOLEGIO\b", "", str(name), flags=re.IGNORECASE)
    name = re.sub(r"\(IED\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[: max_len - 1].rstrip() + "…"
    return name


# ── Step 4 — Build 2×5 validation figure ─────────────────────────────────────
def build_figure(top5, bot5, catalog, names_map) -> plt.Figure:
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#f9f9f9")

    main_title = (
        "Validacion del indice visual $v_j$ — "
        "Colegios con mayor y menor atractivo aparente"
    )
    subtitle = (
        "Modelo NMF K=8 + Ridge | "
        "$v_j$ = suma ponderada de topicos visuales"
    )
    fig.suptitle(
        f"{main_title}\n{subtitle}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
        color="#222222",
    )

    outer = gridspec.GridSpec(
        2, 1,
        figure=fig,
        hspace=0.50,
        top=0.93,
        bottom=0.04,
        left=0.05,
        right=0.98,
    )

    for row_idx, (group, color, row_label) in enumerate(
        [
            (top5,  COLOR_TOP, "Top 5 — Mayor atractivo visual"),
            (bot5,  COLOR_BOT, "Bottom 5 — Menor atractivo visual"),
        ]
    ):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[row_idx], wspace=0.10
        )

        # Row label written on the invisible parent axes
        ax_row = fig.add_subplot(outer[row_idx])
        ax_row.set_axis_off()
        ax_row.text(
            -0.012, 0.5, row_label,
            transform=ax_row.transAxes,
            fontsize=9, fontweight="bold", color=color,
            va="center", ha="right", rotation=90,
        )

        for col_idx in range(5):
            ax = fig.add_subplot(inner[col_idx])
            ax.set_axis_off()

            rec = group.iloc[col_idx]
            estab_id = rec["id_establecimiento"]
            vj_val = rec["vj"]
            display = short_name(names_map.get(estab_id, str(estab_id)))

            img_path = find_image(estab_id, catalog)
            tag = row_label.split("—")[0].strip()
            print(f"  [{tag}] {estab_id}  path={img_path}")

            if img_path:
                try:
                    img = imread(img_path)
                    ax.imshow(img, aspect="auto")
                except Exception as e:
                    print(f"    [warn] imread failed: {e}")
                    _placeholder(ax)
            else:
                _placeholder(ax)

            ax.set_title(
                f"{display}\n$v_j$ = {vj_val:.4f}",
                fontsize=7.5,
                color=color,
                pad=3,
                fontweight="semibold",
            )

    return fig


def _placeholder(ax):
    ax.set_facecolor("#cccccc")
    ax.text(
        0.5, 0.5, "No image",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=8, color="#555555",
    )


# ── Step 5 — Save outputs ─────────────────────────────────────────────────────
def save_outputs(fig: plt.Figure, vj_df: pd.DataFrame) -> None:
    fig_path = os.path.join("reports", "figures", "visual_index_validation.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[saved] figure  → {fig_path}")

    pq_path = os.path.join("data", "primary", "vj_scores.parquet")
    os.makedirs(os.path.dirname(pq_path), exist_ok=True)
    vj_df[["id_establecimiento", "vj", "vj_rank", "vj_quintile"]].to_parquet(
        pq_path, index=False
    )
    print(f"[saved] scores  → {pq_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== 05_visual_index_validation.py ===")

    nmf_path = os.path.join("data", "images", "embeddings", "gsv_nmf_K8.parquet")
    nmf = pd.read_parquet(nmf_path)
    print(f"[nmf] {nmf.shape[0]} rows, cols: {list(nmf.columns)}")

    catalog_path = os.path.join("data", "images", "gsv", "gsv_catalog.csv")
    catalog = pd.read_csv(catalog_path, encoding="utf-8")
    catalog["id_establecimiento"] = catalog["id_establecimiento"].astype(str)
    catalog["descargada"] = (
        catalog["descargada"]
        .astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "sí", "si"])
    )
    catalog["heading"] = pd.to_numeric(catalog["heading"], errors="coerce")
    print(
        f"[catalog] {len(catalog)} rows, {catalog['descargada'].sum()} downloaded"
    )

    betas = load_betas()
    vj_df = compute_vj(nmf, betas)
    top5, bot5 = pick_extremes(vj_df)

    names_map = (
        catalog.drop_duplicates("id_establecimiento")
        .set_index("id_establecimiento")["nombre_establecimiento"]
        .to_dict()
    )

    fig = build_figure(top5, bot5, catalog, names_map)
    save_outputs(fig, vj_df)
    plt.close(fig)
    print("=== done ===")


if __name__ == "__main__":
    main()
