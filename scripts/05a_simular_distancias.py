"""
Simulate household locations within manzana polygons and compute Haversine
distances to every official school in Bogotá.

Each household is placed inside a random manzana that matches its estrato
within its UPZ, rather than anywhere inside the full UPZ polygon.

Inputs
------
- data/raw/em2021_familias_escolar.csv        : households (COD_UPZ_GRUPO, ESTRATO2021, …)
- data/raw/em2021_variables_adicionales.csv   : N_pobre_monetario, N_pobre_extremo, N_ingpc
- data/raw/upz/upz.shp                        : UPZ polygons (UPLCODIGO field)
- data/raw/manzana_estratificacion.geojson    : manzanas in ESRI JSON format (EPSG:3116)
- data/primary/colegios_features.geojson      : schools with Point geometry

Outputs
-------
- data/processed/familias_ubicadas.parquet    : households + simulated lat/lon
- data/processed/familias_distancias.parquet  : wide matrix (families × schools) in km
"""

import json
import random

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Point, Polygon, shape  # noqa: F401

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
FAMILIAS_CSV  = ROOT / "data" / "raw" / "em2021_familias_escolar.csv"
VARS_ADIC_CSV = ROOT / "data" / "raw" / "em2021_variables_adicionales.csv"
UPZ_SHP       = ROOT / "data" / "raw" / "upz" / "upz.shp"
MANZANAS_JSON = ROOT / "data" / "raw" / "manzana_estratificacion.geojson"
COLEGIOS_GEO  = ROOT / "data" / "primary" / "colegios_features.geojson"
OUT_DIR       = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)

# ── Step 1: Load families ──────────────────────────────────────────────────────
print("Step 1 - Loading families...")
fam = pd.read_csv(FAMILIAS_CSV)

n_total = len(fam)
fam = fam.dropna(subset=["COD_UPZ_GRUPO"])
n_dropped = n_total - len(fam)

fam["COD_UPZ_GRUPO"] = fam["COD_UPZ_GRUPO"].astype(float).astype(int)
fam["ESTRATO2021"]   = fam["ESTRATO2021"].astype(float).astype(int)
fam["estrato_real"]  = pd.to_numeric(fam["NVCBP11AA"], errors="coerce").replace(0, pd.NA).astype("Int64")

# Join con variables adicionales: pobreza e ingreso per cápita
vars_adic = pd.read_csv(
    VARS_ADIC_CSV,
    usecols=["directorio_hog", "N_pobre_monetario", "N_pobre_extremo", "N_ingpc"],
    dtype={"directorio_hog": str, "N_pobre_monetario": "Int8", "N_pobre_extremo": "Int8"},
)
vars_adic["N_ingpc"] = pd.to_numeric(vars_adic["N_ingpc"], errors="coerce")
# directorio_hog = DIRECTORIO + dígito de orden de hogar — truncar último dígito para hacer join
vars_adic["DIRECTORIO"] = vars_adic["directorio_hog"].str[:-1]
# Agregar a nivel hogar (tomar primera fila por DIRECTORIO — N_ingpc es a nivel hogar)
vars_adic = vars_adic.drop_duplicates(subset="DIRECTORIO")
fam["DIRECTORIO"] = fam["DIRECTORIO"].astype(str)
fam = fam.merge(vars_adic[["DIRECTORIO", "N_pobre_monetario", "N_pobre_extremo", "N_ingpc"]],
                on="DIRECTORIO", how="left")

n_matched = fam["N_ingpc"].notna().sum()
print(f"  Families kept   : {len(fam):,}")
print(f"  Families dropped: {n_dropped:,} (missing COD_UPZ_GRUPO)")
print(f"  Matched with vars adicionales: {n_matched:,} / {len(fam):,} (N_ingpc not null)")

# ── Step 2: Load UPZ polygons ──────────────────────────────────────────────────
print("\nStep 2 - Loading UPZ polygons...")
upz = gpd.read_file(UPZ_SHP)
upz["cod_upz"] = upz["UPLCODIGO"].str.extract(r"(\d+)")[0].astype(int)
upz = upz.to_crs("EPSG:4326")
print(f"  UPZ polygons loaded: {len(upz):,}")

# ── Step 3: Load manzanas (ESRI JSON format) ───────────────────────────────────
print("\nStep 3 - Loading manzanas...")
with open(MANZANAS_JSON, encoding="utf-8") as f:
    raw = json.load(f)

records = []
for feat in raw["features"]:
    attrs = feat["attributes"]
    rings = feat["geometry"]["rings"]
    shifted_ring = [(pt[0] + 900000, pt[1] + 900000) for pt in rings[0]]
    poly  = Polygon(shifted_ring)
    records.append(
        {
            "estrato":         attrs["ESTRATO"],
            "codigo_manzana":  attrs["CODIGO_MANZANA"],
            "geometry":        poly,
        }
    )

manzanas = gpd.GeoDataFrame(records, crs="EPSG:3116")

estratos_disponibles = sorted(manzanas["estrato"].dropna().unique().tolist())
print(f"  Total manzanas loaded : {len(manzanas):,}")
print(f"  Estratos available    : {estratos_disponibles}")

# ── Step 4: Spatial join - assign manzanas to UPZ, build lookup dicts ─────────
print("\nStep 4 - Joining manzanas to UPZ polygons...")
upz_proj = upz.to_crs("EPSG:3116")
manzanas_upz = gpd.sjoin(
    manzanas[["estrato", "codigo_manzana", "geometry"]],
    upz_proj[["cod_upz", "geometry"]],
    how="left",
    predicate="within",
)
manzanas_upz = manzanas_upz.dropna(subset=["cod_upz"])
manzanas_upz["cod_upz"] = manzanas_upz["cod_upz"].astype(int)
print(f"  Manzanas matched to UPZ: {len(manzanas_upz):,} (of {len(manzanas):,} total)")
manzanas_upz = manzanas_upz.to_crs("EPSG:4326")

# Build (cod_upz, estrato) → [geometry, …]
upz_estrato_manzanas: dict[tuple[int, int], list] = {}
for _, row in manzanas_upz.iterrows():
    key = (int(row["cod_upz"]), int(row["estrato"]))
    upz_estrato_manzanas.setdefault(key, []).append(row["geometry"])

# Build cod_upz → sorted list of available estratos
upz_estratos_available: dict[int, list[int]] = {}
for cod_upz, estrato in upz_estrato_manzanas:
    upz_estratos_available.setdefault(cod_upz, [])
    if estrato not in upz_estratos_available[cod_upz]:
        upz_estratos_available[cod_upz].append(estrato)
for cod_upz in upz_estratos_available:
    upz_estratos_available[cod_upz] = sorted(upz_estratos_available[cod_upz])

# ── Step 5: Simulate household locations ──────────────────────────────────────
print("\nStep 5 - Simulating household locations inside manzanas...")

lats: list[float] = []
lons: list[float] = []
skipped = 0

for i, (_, row) in enumerate(fam.iterrows()):
    cod_upz = int(row["COD_UPZ_GRUPO"])
    estrato  = int(row["estrato_real"]) if pd.notna(row["estrato_real"]) else 3

    if cod_upz not in upz_estratos_available:
        print(f"  WARNING: UPZ {cod_upz} not found in manzanas - skipping family.")
        lats.append(np.nan)
        lons.append(np.nan)
        skipped += 1
        continue

    key = (cod_upz, estrato)
    if key not in upz_estrato_manzanas:
        # Fallback: closest available estrato by absolute difference
        available = upz_estratos_available[cod_upz]
        estrato = min(available, key=lambda e: abs(e - estrato))
        key = (cod_upz, estrato)

    manzana_poly = random.choice(upz_estrato_manzanas[key])
    minx, miny, maxx, maxy = manzana_poly.bounds
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if manzana_poly.contains(Point(x, y)):
            break

    lons.append(x)
    lats.append(y)

    if (i + 1) % 3000 == 0:
        print(f"  {i + 1:,} / {len(fam):,} families processed...")

fam["lat"] = lats
fam["lon"] = lons

if skipped:
    print(f"  WARNING: {skipped:,} families skipped (UPZ not found in manzanas).")
fam = fam.dropna(subset=["lat", "lon"])
print(f"  Done. {len(fam):,} household locations simulated.")

# ── Step 6: Load schools ───────────────────────────────────────────────────────
print("\nStep 6 - Loading schools...")
schools = gpd.read_file(COLEGIOS_GEO)
schools = schools.to_crs("EPSG:4326")
schools["lat_school"] = schools.geometry.centroid.y
schools["lon_school"] = schools.geometry.centroid.x
schools_coords = schools[["id_establecimiento", "lat_school", "lon_school"]].copy()
schools_coords = schools_coords.dropna(subset=["lat_school", "lon_school"])
print(f"  Schools loaded: {len(schools_coords):,}")

# ── Step 7: Haversine distance matrix ─────────────────────────────────────────
print("\nStep 7 - Computing Haversine distance matrix...")


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Return (n_families, n_schools) distance matrix in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2[:, None] - lat1[None, :]  # (n_schools, n_families)
    dlon = lon2[:, None] - lon1[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1[None, :]) * np.cos(lat2[:, None]) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))  # (n_schools, n_families)


dist_matrix = haversine_matrix(
    fam["lat"].values,
    fam["lon"].values,
    schools_coords["lat_school"].values,
    schools_coords["lon_school"].values,
).T  # → (n_families, n_schools)

print(f"  Matrix shape  : {dist_matrix.shape}")
print(f"  Min distance  : {dist_matrix.min():.4f} km")
print(f"  Max distance  : {dist_matrix.max():.4f} km")

# ── Step 8: Save outputs ───────────────────────────────────────────────────────
print("\nStep 8 - Saving outputs...")

out_fam = OUT_DIR / "familias_ubicadas.parquet"
fam.to_parquet(out_fam, index=False)
print(f"  familias_ubicadas.parquet    saved ({out_fam.stat().st_size / 1e3:.1f} KB)")

dist_df = pd.DataFrame(
    dist_matrix,
    index=fam["DIRECTORIO"].values,
    columns=schools_coords["id_establecimiento"].values,
)
dist_df.index.name = "DIRECTORIO"

out_dist = OUT_DIR / "familias_distancias.parquet"
dist_df.to_parquet(out_dist)
print(f"  familias_distancias.parquet  saved ({out_dist.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")

# ── Step 9: Map of simulated households and schools ───────────────────────────
print("\nStep 9 - Generating map of simulated household locations...")

FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Convert families to GeoDataFrame in Web Mercator
fam_gdf = gpd.GeoDataFrame(
    fam,
    geometry=gpd.points_from_xy(fam["lon"], fam["lat"]),
    crs="EPSG:4326",
).to_crs("EPSG:3857")
fam_gdf["estrato_real"] = pd.to_numeric(fam_gdf["estrato_real"], errors="coerce").fillna(3).astype(int)

# Convert schools to Web Mercator
schools_gdf = schools[["geometry"]].copy().to_crs("EPSG:3857")

# Discrete colormap: 6 colors for estratos 1-6
ESTRATOS = [1, 2, 3, 4, 5, 6]
# Colores discretos y saturados por estrato (1=rojo vivo, 6=azul oscuro)
colors = ["#e41a1c", "#ff7f00", "#f7d800", "#4daf4a", "#377eb8", "#984ea3"]
estrato_color = dict(zip(ESTRATOS, colors))

fig, ax = plt.subplots(figsize=(10, 11))

for estrato, color in estrato_color.items():
    mask = fam_gdf["estrato_real"] == estrato
    fam_gdf[mask].plot(
        ax=ax,
        color=color,
        markersize=8,
        alpha=0.65,
        linewidth=0,
    )

schools_gdf.plot(
    ax=ax,
    color="black",
    marker="^",
    markersize=15,
    zorder=5,
)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=11)

# Legend: estratos
estrato_patches = [
    mpatches.Patch(color=estrato_color[e], label=f"Estrato {e}")
    for e in ESTRATOS
]
school_handle = plt.Line2D(
    [0], [0],
    marker="^",
    color="w",
    markerfacecolor="black",
    markersize=7,
    label="Colegio oficial",
)
legend1 = ax.legend(
    handles=estrato_patches,
    loc="lower left",
    fontsize=7,
    title="Estrato",
    title_fontsize=8,
    framealpha=0.8,
)
ax.add_artist(legend1)
ax.legend(handles=[school_handle], loc="lower right", fontsize=7, framealpha=0.8)

ax.set_title(
    "Familias simuladas por estrato y colegios oficiales - Bogotá 2021",
    fontsize=11,
    pad=10,
)
ax.set_axis_off()

out_map = FIGURES_DIR / "mapa_familias_simuladas.png"
fig.savefig(out_map, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Mapa guardado en: {out_map}")
