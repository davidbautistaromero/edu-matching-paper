"""
05c_visual_index_validation.py
==============================
Valida el índice visual v_j usando los betas de Berry OLS (04a, spec M3).

v_j = β_seg·seg_z + β_veg·veg_z + β_mant·mant_z + β_mod·mod_z + β_infra·infra_z + β_cerr·cerr_z

Muestra top-5 y bottom-5 colegios por v_j con fotos GSV.

Inputs:
    reports/tables/berry_ols_specs.csv          — betas de Berry OLS
    data/primary/colegios_features_imputed.geojson — features visuales por colegio
    data/images/gsv/gsv_catalog.csv             — catálogo de imágenes

Outputs:
    reports/figures/visual_index_validation.png
    data/primary/vj_scores.csv
"""

import os
import re
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
import geopandas as gpd
import pandas as pd
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
# Only statistically significant visual features from Berry OLS M3
VISUAL_RAW = ["seguridad_percibida"]
VISUAL_Z   = [f"{c}_z" for c in VISUAL_RAW]

GSV_ROOT = os.path.join("data", "images", "gsv")
PREFERRED_HEADINGS = [0, 90, 180]
COLOR_TOP = "#1a7a4a"
COLOR_BOT = "#8b1a1a"


# ── Load Berry OLS betas (spec M3) ────────────────────────────────────────────
def load_berry_betas() -> dict:
    path = os.path.join("reports", "tables", "berry_ols_specs.csv")
    df = pd.read_csv(path)
    betas = {}
    for _, row in df.iterrows():
        var = str(row["variable"]).strip()
        if var in VISUAL_Z:
            betas[var] = float(row["M3_coef"])
    print(f"[betas] Berry OLS M3 visual coefficients:")
    for k, v in betas.items():
        sig = "*" if k in ("seguridad_percibida_z", "vegetacion_percibida_z") else ""
        print(f"  {k:<28} β = {v:+.6f} {sig}")
    return betas


# ── Compute v_j ───────────────────────────────────────────────────────────────
def compute_vj(gdf: gpd.GeoDataFrame, betas: dict) -> pd.DataFrame:
    vj_series = sum(betas[f"{c}_z"] * gdf[f"{c}_z"].astype(float) for c in VISUAL_RAW)
    out = gdf[["id_establecimiento"]].copy()
    out["vj"] = vj_series.values
    out = out.sort_values("vj", ascending=False).reset_index(drop=True)
    out["vj_rank"] = out["vj"].rank(ascending=False, method="first").astype(int)
    out["vj_quintile"] = pd.qcut(out["vj"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    print(f"[vj] n={len(out)}  range: {out['vj'].min():.4f} – {out['vj'].max():.4f}")
    return out


# ── Pick manual extremes ──────────────────────────────────────────────────────
def pick_extremes(vj_df):
    top_ids = [eid for eid, _ in MANUAL_TOP]
    bot_ids = [eid for eid, _ in MANUAL_BOT]
    top5 = vj_df[vj_df["id_establecimiento"].isin(top_ids)].copy()
    top5["_order"] = top5["id_establecimiento"].map({eid: i for i, (eid, _) in enumerate(MANUAL_TOP)})
    top5 = top5.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    bot5 = vj_df[vj_df["id_establecimiento"].isin(bot_ids)].copy()
    bot5["_order"] = bot5["id_establecimiento"].map({eid: i for i, (eid, _) in enumerate(MANUAL_BOT)})
    bot5 = bot5.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    print("[top5] ids:", top5["id_establecimiento"].tolist())
    print("[bot5] ids:", bot5["id_establecimiento"].tolist())
    return top5, bot5


# ── Find best image ──────────────────────────────────────────────────────────
# ── Manual selection: (id_establecimiento, heading) ───────────────────────────
# Curated by visual inspection of all headings for each school.
MANUAL_TOP = [
    ("111001000132", 144),   # Aquileo Parra
    ("111001800431", 288),   # Las Margaritas
    ("111769000174", 144),   # Filarmónico Simón Bolívar
    ("111001109550",  72),   # Rodolfo Llinás
    ("111001104302",   0),   # Orlando Higuita Rojas
]
MANUAL_BOT = [
    ("111001098612", 288),   # La Estrellita
    ("111001015733", 108),   # Antonio Baraya
    ("111001104353", 324),   # Ciudad de Villavicencio
    ("111001014826", 108),   # Marco Antonio Carreño Silva
    ("111001075329",  36),   # María Mercedes Carranza
]


def find_image(estab_id, catalog, heading=None):
    """Find image for estab_id with specific heading (manual selection)."""
    subset = catalog[
        (catalog["id_establecimiento"] == estab_id)
        & (catalog["descargada"] == True)
    ]
    if subset.empty:
        return None

    # Try exact heading first
    if heading is not None:
        match = subset[subset["heading"] == heading]
        if not match.empty:
            full = os.path.join(GSV_ROOT, match.iloc[0]["ruta_archivo"])
            if os.path.isfile(full):
                return full

    # Fallback to preferred headings
    for h in PREFERRED_HEADINGS:
        match = subset[subset["heading"] == h]
        if not match.empty:
            full = os.path.join(GSV_ROOT, match.iloc[0]["ruta_archivo"])
            if os.path.isfile(full):
                return full
    # Any available
    for _, row in subset.iterrows():
        full = os.path.join(GSV_ROOT, row["ruta_archivo"])
        if os.path.isfile(full):
            return full
    return None


def short_name(name, max_len=20):
    name = re.sub(r"\bCOLEGIO\b", "", str(name), flags=re.IGNORECASE)
    name = re.sub(r"\(IED\)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len - 1].rstrip() + "…"
    return name


def _placeholder(ax):
    ax.set_facecolor("#cccccc")
    ax.text(0.5, 0.5, "No image", ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="#555555")


# ── Build figure ──────────────────────────────────────────────────────────────
def build_figure(top5, bot5, catalog, names_map, gdf_ref, betas=None):
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#f9f9f9")

    main_title = (
        "Validación del índice visual $v_j$ — "
        "Colegios con mayor y menor atractivo aparente"
    )
    # Build subtitle dynamically from betas
    if betas:
        beta_str = " + ".join(f"{v:+.3f} \\cdot {k.replace('_z','')}" for k, v in betas.items())
        subtitle = f"Berry OLS (M3) | $v_j = {beta_str}$  (significant betas only)"
    else:
        subtitle = "Berry OLS (M3)"
    fig.suptitle(f"{main_title}\n{subtitle}", fontsize=11, fontweight="bold",
                 y=1.02, color="#222222")

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.50,
                              top=0.93, bottom=0.04, left=0.05, right=0.98)

    manual_lists = [MANUAL_TOP, MANUAL_BOT]
    for row_idx, (group, color, row_label, manual) in enumerate([
        (top5, COLOR_TOP, "Top 5 — Mayor atractivo visual", MANUAL_TOP),
        (bot5, COLOR_BOT, "Bottom 5 — Menor atractivo visual", MANUAL_BOT),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[row_idx],
                                                 wspace=0.10)
        ax_row = fig.add_subplot(outer[row_idx])
        ax_row.set_axis_off()
        ax_row.text(-0.012, 0.5, row_label, transform=ax_row.transAxes,
                    fontsize=9, fontweight="bold", color=color,
                    va="center", ha="right", rotation=90)

        for col_idx in range(5):
            ax = fig.add_subplot(inner[col_idx])
            ax.set_axis_off()

            rec = group.iloc[col_idx]
            estab_id = rec["id_establecimiento"]
            vj_val = rec["vj"]
            display = short_name(names_map.get(estab_id, str(estab_id)))

            # Use manual heading
            heading = manual[col_idx][1]
            img_path = find_image(estab_id, catalog, heading=heading)
            tag = row_label.split("—")[0].strip()
            print(f"  [{tag}] {estab_id}  h={heading}  path={img_path}")

            if img_path:
                try:
                    img = imread(img_path)
                    ax.imshow(img, aspect="auto")
                except Exception as e:
                    print(f"    [warn] imread failed: {e}")
                    _placeholder(ax)
            else:
                _placeholder(ax)

            # Feature breakdown below image
            feat_vals = []
            for feat in VISUAL_RAW:
                val = gdf_ref.loc[gdf_ref["id_establecimiento"] == estab_id, f"{feat}_z"]
                v = val.values[0] if len(val) > 0 else 0.0
                short = feat.replace("_percibida", "").replace("infraestructura_", "infra_")
                feat_vals.append(f"{short}={v:+.2f}")
            feat_str = "  ".join(feat_vals[:3]) + "\n" + "  ".join(feat_vals[3:])

            ax.set_title(f"{display}\n$v_j$ = {vj_val:.4f}", fontsize=7.5,
                         color=color, pad=3, fontweight="semibold")
            ax.text(0.5, -0.02, feat_str, transform=ax.transAxes,
                    fontsize=5.5, ha="center", va="top", color="#555555",
                    family="monospace")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== 05c_visual_index_validation.py (Berry OLS betas) ===")

    # Load geojson
    gdf = gpd.read_file(os.path.join("data", "primary",
                                      "colegios_features_imputed.geojson"))
    gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
    print(f"[geojson] {len(gdf)} colegios")

    # Load CLIP scores (establecimiento-level) — only need seguridad and vegetacion
    clip = pd.read_parquet(os.path.join("data", "images", "clip",
                                         "gsv_clip_establecimiento.parquet"))
    clip["id_establecimiento"] = clip["id_establecimiento"].astype(str)
    gdf = gdf.merge(clip[["id_establecimiento"] + VISUAL_RAW],
                     on="id_establecimiento", how="left")

    # Standardize (same as 04a)
    for col in VISUAL_RAW:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
        mu, sd = gdf[col].mean(), gdf[col].std()
        gdf[f"{col}_z"] = (gdf[col] - mu) / sd if sd > 0 else 0.0
    print(f"[features] {len(VISUAL_RAW)} significant visual features standardized")

    # Catalog
    catalog_path = os.path.join("data", "images", "gsv", "gsv_catalog.csv")
    catalog = pd.read_csv(catalog_path, encoding="utf-8")
    catalog["id_establecimiento"] = catalog["id_establecimiento"].astype(str)
    catalog["descargada"] = (
        catalog["descargada"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "sí", "si"])
    )
    catalog["heading"] = pd.to_numeric(catalog["heading"], errors="coerce")
    print(f"[catalog] {len(catalog)} rows, {catalog['descargada'].sum()} downloaded")

    betas = load_berry_betas()
    vj_df = compute_vj(gdf, betas)
    top5, bot5 = pick_extremes(vj_df)

    names_map = (
        catalog.drop_duplicates("id_establecimiento")
        .set_index("id_establecimiento")["nombre_establecimiento"]
        .to_dict()
    )

    fig = build_figure(top5, bot5, catalog, names_map, gdf, betas=betas)

    # Save
    fig_path = os.path.join("reports", "figures", "visual_index_validation.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[saved] figure → {fig_path}")

    csv_path = os.path.join("data", "primary", "vj_scores.csv")
    vj_df[["id_establecimiento", "vj", "vj_rank", "vj_quintile"]].to_csv(
        csv_path, index=False)
    print(f"[saved] scores → {csv_path}")

    plt.close(fig)
    print("=== done ===")


if __name__ == "__main__":
    main()
