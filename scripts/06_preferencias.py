# -*- coding: utf-8 -*-
"""
06_preferencias.py
==================
Calcula utilidades u_ij y rankings de preferencia para cada familia sobre colegios oficiales.

Modelo BLP (IV-BLP, especificacion preferida de 04b_blp.py):
  u_ij = δ_j + π₁·y_i·seg_z_j + λ₀·log1p(d_ij) + λ₁·y_i·log1p(d_ij) + ε_ij
  δ_j     : utilidad media BLP (mapping de contraccion), cargada de blp_delta_j.parquet
  π₁      : interaccion ingreso normalizado × seguridad percibida estandarizada
  λ₀      : coeficiente base de distancia
  λ₁      : interaccion ingreso normalizado × log-distancia
  y_i     : ingreso normalizado = N_ingpc / mean(N_ingpc) (media sobre toda la muestra)
  seg_z_j : seguridad_percibida estandarizada (z-score sobre todos los colegios)
  d_ij    : distancia en km (familia i -> colegio j)
  ε_ij    : ruido Gumbel(0,1) -- genera estocasticidad en rankings

Choice set: familias eligen solo dentro de su localidad.
Excepcion: La Candelaria (localidad 17) puede elegir en localidades 17, 3, 14, 15.

Inputs:
  data/processed/familias_expandidas.parquet        -- familias expandidas con FEX_C
  data/processed/distancias_expandidas.parquet      -- matriz de distancias expandida
  data/primary/blp_delta_j.parquet                  -- δ_j por colegio (IV-BLP preferido)
  data/images/clip/gsv_clip_establecimiento.parquet -- seguridad_percibida por colegio
  reports/tables/blp_results.csv                    -- parámetros estimados (pi1, lam0, lam1)

Outputs:
  data/primary/preferencias_familias.parquet     -- top-20 rankings por familia
  data/processed/utilidades_familias.parquet     -- matriz de utilidades (escrita en batches)
"""

import logging
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DELTA_BLP_PATH = ROOT / "data" / "primary" / "blp_delta_j.parquet"
CLIP_PATH      = ROOT / "data" / "images" / "clip" / "gsv_clip_establecimiento.parquet"
BLP_RESULTS    = ROOT / "reports" / "tables" / "blp_results.csv"
COLEGIOS       = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
FAM_PATH       = ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_PATH      = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
OUT_DIR        = ROOT / "data" / "processed"

# Candelaria (17) puede elegir en estas localidades tambien
CANDELARIA_EXTRA = {3, 14, 15}

# Mapping nombre_localidad -> codigo numerico (estandar Bogota)
LOC_NAME_TO_CODE = {
    "Usaquen": 1, "Chapinero": 2, "Santa Fe": 3, "San Cristobal": 4,
    "Usme": 5, "Tunjuelito": 6, "Bosa": 7, "Kennedy": 8,
    "Fontibon": 9, "Engativa": 10, "Suba": 11, "Barrios Unidos": 12,
    "Teusaquillo": 13, "Los Martires": 14, "Antonio Narino": 15,
    "Puente Aranda": 16, "La Candelaria": 17, "Rafael Uribe Uribe": 18,
    "Ciudad Bolivar": 19, "Sumapaz": 20,
}

TOP_K = 20
SEED  = 42

# ── Step 1: Cargar parametros BLP y datos de colegios ────────────────────────
log.info("Step 1 -- Cargando parámetros BLP y datos de colegios...")

blp_params = pd.read_csv(BLP_RESULTS)
iv_blp = blp_params[blp_params["spec"] == "iv_blp"].set_index("parametro")["estimacion"]
pi1  = float(iv_blp["pi1"])
lam0 = float(iv_blp["lam0"])
lam1 = float(iv_blp["lam1"])

log.info(f"  Parámetros IV-BLP cargados:")
log.info(f"    π₁  (y_i × seg_z_j)        = {pi1:+.6f}")
log.info(f"    λ₀  (log1p d_ij)            = {lam0:+.6f}")
log.info(f"    λ₁  (y_i × log1p d_ij)      = {lam1:+.6f}")

# delta_j por colegio (contraction mapping IV-BLP)
delta_df = pd.read_parquet(DELTA_BLP_PATH)
delta_df["id_establecimiento"] = delta_df["id_establecimiento"].astype(str).str.strip()

# seguridad_percibida -> z-score sobre todos los colegios
clip_df = pd.read_parquet(CLIP_PATH)
clip_df["id_establecimiento"] = clip_df["id_establecimiento"].astype(str).str.strip()
seg_mean = clip_df["seguridad_percibida"].mean()
seg_std  = clip_df["seguridad_percibida"].std()
clip_df["seg_z"] = (clip_df["seguridad_percibida"] - seg_mean) / seg_std
log.info(f"  seguridad_percibida: mean={seg_mean:.4f}  std={seg_std:.4f}")

# Merge delta_j + seg_z
schools = delta_df.merge(clip_df[["id_establecimiento", "seg_z"]], on="id_establecimiento", how="inner")
log.info(f"  Colegios con delta_j y seg_z: {len(schools)}")

# Codigo numerico de localidad para colegios
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower().strip()

loc_map = {strip_accents(name): code for name, code in LOC_NAME_TO_CODE.items()}

gdf = gpd.read_file(COLEGIOS)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str).str.strip()
gdf["cod_localidad"] = gdf["nombre_localidad"].apply(
    lambda x: loc_map.get(strip_accents(x)) if pd.notna(x) else None
)

schools = schools.merge(gdf[["id_establecimiento", "cod_localidad"]], on="id_establecimiento", how="inner")
unmapped = schools["cod_localidad"].isna().sum()
if unmapped > 0:
    log.info(f"  Eliminando {unmapped} colegios sin match de localidad")
    schools = schools[schools["cod_localidad"].notna()].reset_index(drop=True)
log.info(f"  Colegios restantes tras filtro de localidad: {len(schools)}")

school_ids = schools["id_establecimiento"].values
delta_j    = schools["delta_j_blp"].values
seg_z_j    = schools["seg_z"].values
school_loc = schools["cod_localidad"].values

log.info(f"  δ_j: min={delta_j.min():.4f}  max={delta_j.max():.4f}")

# ── Step 2: Cargar familias y distancias ──────────────────────────────────────
log.info("Step 2 -- Cargando familias y distancias...")
fam  = pd.read_parquet(FAM_PATH)
dist = pd.read_parquet(DIST_PATH)

# distancias_expandidas usa indice posicional -- alinear por posicion
n = min(len(fam), len(dist))
fam  = fam.iloc[:n].reset_index(drop=True)
dist = dist.iloc[:n].reset_index(drop=True)

common_schools = [sid for sid in school_ids if sid in dist.columns]
dist           = dist[common_schools]
idx_schools    = [list(school_ids).index(sid) for sid in common_schools]
delta_j_al     = delta_j[idx_schools]
seg_z_j_al     = seg_z_j[idx_schools]
school_ids_al  = np.array(common_schools)
school_loc_al  = school_loc[idx_schools]

dist_matrix = dist.values.astype(np.float32)
log.info(f"  Familias: {len(fam):,} | Colegios: {len(school_ids_al)}")

# ── Step 3: Ingreso normalizado y_i ──────────────────────────────────────────
log.info("Step 3 -- Calculando ingreso normalizado y_i...")
N_ingpc_arr = pd.to_numeric(fam["N_ingpc"], errors="coerce").values
mean_ingpc  = float(np.nanmean(N_ingpc_arr))
# Media de la muestra completa; imputar media donde falta o es <= 0
ingpc_safe  = np.where(np.isnan(N_ingpc_arr) | (N_ingpc_arr <= 0), mean_ingpc, N_ingpc_arr)
y_i         = ingpc_safe / mean_ingpc

log.info(f"  mean_ingpc={mean_ingpc:.0f} | y_i media={y_i.mean():.4f} | min={y_i.min():.4f} | max={y_i.max():.4f}")

# ── Step 4: Choice set mask por localidad ─────────────────────────────────────
log.info("Step 4 -- Construyendo choice sets por localidad...")
fam_loc = pd.to_numeric(fam["COD_LOCALIDAD"], errors="coerce").values

try:
    school_loc_num = pd.to_numeric(school_loc_al, errors="coerce")
except Exception:
    school_loc_num = school_loc_al

mask = np.zeros((len(fam), len(school_ids_al)), dtype=bool)
for i, loc_i in enumerate(fam_loc):
    if pd.isna(loc_i):
        mask[i, :] = True
        continue
    loc_i = int(loc_i)
    allowed = {loc_i}
    if loc_i == 17:
        allowed |= CANDELARIA_EXTRA
    mask[i, :] = np.isin(school_loc_num, list(allowed))

sizes = mask.sum(axis=1)
log.info(f"  Choice set promedio: {sizes.mean():.1f} | min={sizes.min()} | max={sizes.max()}")

# ── Step 5 & 6: Utilidades y rankings en batches ──────────────────────────────
log.info("Step 5 -- Calculando utilidades y rankings en batches...")
BATCH_SIZE = 50_000
rng = np.random.default_rng(SEED)

n_fam = len(fam)
all_rankings = []

out_util = OUT_DIR / "utilidades_familias.parquet"
util_writer = None

for batch_start in range(0, n_fam, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_fam)
    b_dist = dist_matrix[batch_start:batch_end]
    b_yi   = y_i[batch_start:batch_end]
    b_mask = mask[batch_start:batch_end]

    epsilon  = rng.gumbel(0, 1, size=(batch_end - batch_start, len(school_ids_al))).astype(np.float32)
    log_dist = np.log1p(b_dist)  # d_ij en km

    # u_ij = δ_j + π₁·y_i·seg_z_j + λ₀·log1p(d_ij) + λ₁·y_i·log1p(d_ij) + ε_ij
    u = (delta_j_al[None, :].astype(np.float32)
         + (pi1  * b_yi[:, None] * seg_z_j_al[None, :]).astype(np.float32)
         + (lam0 * log_dist).astype(np.float32)
         + (lam1 * b_yi[:, None] * log_dist).astype(np.float32)
         + epsilon)
    u[~b_mask] = -np.inf

    # Rankings
    rankings_idx = np.argsort(-u, axis=1)[:, :TOP_K]
    rankings_ids = school_ids_al[rankings_idx]
    all_rankings.append(rankings_ids)

    # Utilidades -- escribir batch a parquet incremental
    u_df = pd.DataFrame(u, columns=school_ids_al)
    u_df.insert(0, "DIRECTORIO", fam["DIRECTORIO"].values[batch_start:batch_end])
    table = pa.Table.from_pandas(u_df, preserve_index=False)
    if util_writer is None:
        util_writer = pq.ParquetWriter(str(out_util), table.schema, compression="snappy")
    util_writer.write_table(table)

    if (batch_start // BATCH_SIZE) % 5 == 0:
        log.info(f"  Procesados {batch_end:,} / {n_fam:,} familias...")

if util_writer is not None:
    util_writer.close()

log.info(f"  Completado: ({n_fam}, {len(school_ids_al)}) en batches de {BATCH_SIZE:,}")

rankings_arr = np.vstack(all_rankings)
rankings_df = pd.DataFrame(
    rankings_arr,
    index=fam["DIRECTORIO"].values,
    columns=[f"pref_{k+1}" for k in range(TOP_K)],
)
rankings_df.index.name = "DIRECTORIO"

# ── Step 7: Guardar ───────────────────────────────────────────────────────────
log.info("Step 7 -- Guardando outputs...")
out_rank = ROOT / "data" / "primary" / "preferencias_familias.parquet"

rankings_df.to_parquet(out_rank)
log.info(f"  preferencias_familias.parquet  ({out_rank.stat().st_size / 1e6:.1f} MB)")
log.info(f"  utilidades_familias.parquet    ({out_util.stat().st_size / 1e6:.1f} MB)")

# ── Step 8: Mapa distribución por cuartil de ingreso ─────────────────────────
log.info("Step 8 -- Generando mapa de distribución por cuartil de ingreso...")
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point

    CUARTIL_COLORS = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]  # Q1→Q4: rojo→azul
    CUARTIL_LABELS = ["Q1\n(ingreso más bajo)", "Q2", "Q3", "Q4\n(ingreso más alto)"]

    fam_plot = fam.copy()
    fam_plot["N_ingpc_num"] = pd.to_numeric(fam_plot["N_ingpc"], errors="coerce")
    fam_plot["cuartil_ing"] = pd.qcut(
        fam_plot["N_ingpc_num"], q=4, labels=[0, 1, 2, 3]
    ).cat.codes
    fam_plot = fam_plot.dropna(subset=["lat", "lon"])

    MAX_PLOT = 40_000
    if len(fam_plot) > MAX_PLOT:
        fam_plot = fam_plot.sample(MAX_PLOT, random_state=42)

    gdf_fam = gpd.GeoDataFrame(
        fam_plot,
        geometry=[Point(lon, lat) for lon, lat in zip(fam_plot["lon"], fam_plot["lat"])],
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    gdf_col = gpd.read_file(ROOT / "data" / "primary" / "colegios_features_imputed.geojson")
    gdf_col = gdf_col.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=(11, 12))

    for q in range(4):
        sub = gdf_fam[gdf_fam["cuartil_ing"] == q]
        sub.plot(ax=ax, color=CUARTIL_COLORS[q], markersize=8, alpha=0.65,
                 linewidth=0, label=CUARTIL_LABELS[q], zorder=2)

    gdf_col.plot(ax=ax, color="black", marker="^", markersize=15, zorder=5)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)

    familia_lat_min = fam_plot["lat"].min()
    pt_bottom = gpd.GeoDataFrame(
        geometry=[Point(0, familia_lat_min - 0.05)], crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    ax.set_ylim(bottom=pt_bottom.geometry[0].y)

    legend_patches = [
        mpatches.Patch(color=CUARTIL_COLORS[i], label=CUARTIL_LABELS[i]) for i in range(4)
    ] + [plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
                    markersize=7, label="Colegio oficial")]

    ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
              framealpha=0.85, title="Cuartil de ingreso per cápita", title_fontsize=9)
    ax.set_title("Distribución espacial de familias por cuartil de ingreso\n(Bogotá, EM2021)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_map = fig_dir / "mapa_ingreso_familias.png"
    fig.savefig(out_map, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  mapa_ingreso_familias.png  ({out_map.stat().st_size / 1e6:.1f} MB)")

except Exception as e:
    log.warning(f"  Mapa de ingreso no generado: {e}")

log.info("Done.")
