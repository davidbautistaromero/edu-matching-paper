#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_mapa_cobertura_gsv.py
========================
Genera mapa de cobertura de imágenes Google Street View para los
colegios oficiales de Bogotá.

Categorías:
  - Con cobertura real    : imágenes descargadas y con contenido válido
  - Imagen en blanco      : GSV devolvió placeholder gris ("no imagery")
  - Sin cobertura         : no se descargó ninguna imagen

SALIDAS
-------
  data/images/gsv/mapa_cobertura_gsv.png
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from PIL import Image

# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
RUTA_SEDES    = ROOT / "data" / "processed" / "colegios_dataset.geojson"
RUTA_CATALOGO = ROOT / "data" / "images" / "gsv" / "gsv_catalog.csv"
RUTA_GSV      = ROOT / "data" / "images" / "gsv"
RUTA_SALIDA   = ROOT / "data" / "images" / "gsv" / "mapa_cobertura_gsv.png"

BLANK_STD_THRESHOLD = 15.0


# ---------------------------------------------------------------------------
def sede_es_blanca(id_sede: str) -> bool:
    """True si todas las imágenes de la sede son el placeholder gris de GSV."""
    carpeta = RUTA_GSV / id_sede
    if not carpeta.exists():
        # Buscar en carpetas de establecimiento
        imgs = list(RUTA_GSV.rglob(f"{id_sede}_*.jpg"))
    else:
        imgs = list(carpeta.glob("*.jpg"))
    if not imgs:
        return False
    stds = []
    for img_path in imgs:
        try:
            arr = np.array(Image.open(img_path).convert('L'))
            stds.append(arr.std())
        except Exception:
            pass
    return bool(stds and np.mean(stds) < BLANK_STD_THRESHOLD)


# ---------------------------------------------------------------------------
print("Cargando datos…")
gdf = gpd.read_file(RUTA_SEDES)
cat = pd.read_csv(RUTA_CATALOGO, dtype=str)
cat["descargada"] = cat["descargada"].str.lower() == "true"

# Imágenes descargadas por sede
ok_por_sede = (
    cat[cat["descargada"]]
    .groupby("id_sede")
    .size()
    .reset_index(name="n_imgs")
)

gdf["id_sede"] = gdf["id_sede"].astype(str).str.strip()
gdf = gdf.merge(ok_por_sede, on="id_sede", how="left")
gdf["n_imgs"] = gdf["n_imgs"].fillna(0).astype(int)
gdf["con_imagen"] = gdf["n_imgs"] > 0

# Detectar sedes con imagen en blanco
print("Detectando imágenes en blanco…")
sedes_con_img = gdf.loc[gdf["con_imagen"], "id_sede"].tolist()
sedes_blancas = set(s for s in sedes_con_img if sede_es_blanca(s))
if sedes_blancas:
    print(f"  Sedes con imagen en blanco: {len(sedes_blancas)}")
    for s in sorted(sedes_blancas):
        nom = gdf.loc[gdf["id_sede"] == s, "nombre_establecimiento"].values
        print(f"    {s}  {nom[0] if len(nom) else ''}")

gdf["estado"] = "sin_cobertura"
gdf.loc[gdf["con_imagen"], "estado"] = "con_cobertura"
gdf.loc[gdf["id_sede"].isin(sedes_blancas), "estado"] = "imagen_blanca"

gdf_wm = gdf.to_crs(epsg=3857)

con    = gdf_wm[gdf_wm["estado"] == "con_cobertura"]
blanca = gdf_wm[gdf_wm["estado"] == "imagen_blanca"]
sin    = gdf_wm[gdf_wm["estado"] == "sin_cobertura"]

n_con    = len(con)
n_blanca = len(blanca)
n_sin    = len(sin)
n_total  = len(gdf)

print(f"  Con cobertura real:   {n_con} / {n_total} sedes ({n_con/n_total*100:.1f}%)")
print(f"  Imagen en blanco:     {n_blanca} sedes")
print(f"  Sin cobertura:        {n_sin} sedes")

# ---------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 16))

con.plot(ax=ax, color="#1a6faf", markersize=40, alpha=0.85, zorder=3)

if len(blanca) > 0:
    blanca.plot(
        ax=ax, color="#E68A2E", marker="^",
        markersize=55, alpha=0.95, zorder=5,
    )

if len(sin) > 0:
    sin.plot(
        ax=ax, color="crimson", marker="x",
        markersize=30, linewidths=1.5, alpha=0.9, zorder=4,
    )

try:
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
except Exception:
    pass

patch_con    = mpatches.Patch(color="#1a6faf", label=f"Con cobertura GSV ({n_con})")
patch_blanca = mpatches.Patch(color="#E68A2E", label=f"Imagen en blanco — sin cobertura real ({n_blanca})")
patch_sin    = mpatches.Patch(color="crimson", label=f"Sin cobertura ({n_sin})")

handles = [patch_con, patch_blanca, patch_sin] if n_blanca > 0 else [patch_con, patch_sin]
ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.9)

n_validas = n_con
ax.set_title(
    f"Cobertura Google Street View — Colegios Oficiales Bogotá\n"
    f"{n_validas}/{n_total} sedes con cobertura real ({n_validas/n_total*100:.1f}%) | "
    f"10 headings × sede | Total imágenes válidas: {n_validas * 10}",
    fontsize=12, pad=14,
)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(RUTA_SALIDA, dpi=150, bbox_inches="tight")
plt.close()
print(f"Mapa guardado en: {RUTA_SALIDA}")
