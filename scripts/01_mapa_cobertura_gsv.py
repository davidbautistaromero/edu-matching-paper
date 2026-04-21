#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_mapa_cobertura_gsv.py
========================
Genera mapa de cobertura de imágenes Google Street View para los
colegios oficiales de Bogotá.

SALIDAS
-------
  data/images/gsv/mapa_cobertura_gsv.png
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx

# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
RUTA_SEDES   = ROOT / "data" / "processed" / "colegios_dataset.geojson"
RUTA_CATALOGO = ROOT / "data" / "images" / "gsv" / "gsv_catalog.csv"
RUTA_SALIDA  = ROOT / "data" / "images" / "gsv" / "mapa_cobertura_gsv.png"

# ---------------------------------------------------------------------------
print("Cargando datos…")
gdf = gpd.read_file(RUTA_SEDES)
cat = pd.read_csv(RUTA_CATALOGO, dtype=str)
cat["descargada"] = cat["descargada"].str.lower() == "true"

# Imágenes OK por sede
ok_por_sede = (
    cat[cat["descargada"]]
    .groupby("id_sede")
    .size()
    .reset_index(name="n_imgs")
)

# Join al GeoDataFrame
gdf["id_sede"] = gdf["id_sede"].astype(str).str.strip()
gdf = gdf.merge(ok_por_sede, on="id_sede", how="left")
gdf["n_imgs"] = gdf["n_imgs"].fillna(0).astype(int)
gdf["con_imagen"] = gdf["n_imgs"] > 0

# Reproyectar a Web Mercator para contextily
gdf_wm = gdf.to_crs(epsg=3857)

con   = gdf_wm[gdf_wm["con_imagen"]]
sin   = gdf_wm[~gdf_wm["con_imagen"]]

n_con = len(con)
n_sin = len(sin)
n_total = len(gdf)

print(f"  Con cobertura GSV: {n_con} / {n_total} sedes ({n_con/n_total*100:.1f}%)")
print(f"  Sin cobertura:     {n_sin} sedes")

# ---------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 16))

# Sedes con imágenes — color por número de imágenes descargadas
sc = con.plot(
    ax=ax,
    color="#1a6faf",
    markersize=40,
    alpha=0.85,
    zorder=3,
)

# Sedes sin imágenes
if len(sin) > 0:
    sin.plot(
        ax=ax,
        color="crimson",
        marker="x",
        markersize=30,
        linewidths=1.5,
        alpha=0.9,
        zorder=4,
        label=f"Sin cobertura ({n_sin})",
    )

# Basemap
try:
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
except Exception:
    pass

# Leyenda
patch_con = mpatches.Patch(color="#2171b5", label=f"Con cobertura GSV ({n_con})")
patch_sin = mpatches.Patch(color="crimson", label=f"Sin cobertura ({n_sin})")
ax.legend(handles=[patch_con, patch_sin], loc="lower left", fontsize=10, framealpha=0.9)

ax.set_title(
    f"Cobertura Google Street View — Colegios Oficiales Bogotá\n"
    f"{n_con}/{n_total} sedes ({n_con/n_total*100:.1f}%) | 10 headings × sede | "
    f"Total: {cat['descargada'].sum()} imágenes",
    fontsize=12, pad=14,
)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(RUTA_SALIDA, dpi=150, bbox_inches="tight")
plt.close()
print(f"Mapa guardado en: {RUTA_SALIDA}")
