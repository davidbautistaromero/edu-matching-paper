"""
04c_build_capacidad.py
======================
Genera tabla minimalista de capacidad por colegio oficial.

Formula:
    capacidad = round(120_000 * matricula_j / sum(matricula_j))
    minimo 5 cupos por colegio

Inputs:
    data/primary/colegios_features_imputed.geojson

Output:
    data/primary/colegios_capacidad.csv
"""

import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
GEO_PATH = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
OUT_PATH = ROOT / "data" / "primary" / "colegios_capacidad.csv"

CUPOS_TOTALES = 120_000

LOC_NAME_TO_CODE = {
    "USAQUEN": 1, "CHAPINERO": 2, "SANTA FE": 3, "SAN CRISTOBAL": 4,
    "USME": 5, "TUNJUELITO": 6, "BOSA": 7, "KENNEDY": 8,
    "FONTIBON": 9, "ENGATIVA": 10, "SUBA": 11, "BARRIOS UNIDOS": 12,
    "TEUSAQUILLO": 13, "LOS MARTIRES": 14, "ANTONIO NARINO": 15,
    "PUENTE ARANDA": 16, "CANDELARIA": 17, "RAFAEL URIBE URIBE": 18,
    "CIUDAD BOLIVAR": 19,
}


def normalize(s):
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ── 1. Cargar geojson ─────────────────────────────────────────────────────────
print("Cargando colegios_features_imputed.geojson...")
gdf = gpd.read_file(GEO_PATH)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
gdf["matricula_total"] = gdf["matricula_total"].astype(float)
print(f"  {len(gdf):,} establecimientos")

# ── 2. Capacidad proporcional ─────────────────────────────────────────────────
print("Calculando capacidad proporcional...")
mat_total = gdf["matricula_total"].sum()
gdf["capacidad"] = (CUPOS_TOTALES * gdf["matricula_total"] / mat_total).round().clip(lower=5).astype(int)

# ── 3. Código de localidad ────────────────────────────────────────────────────
gdf["cod_localidad"] = gdf["nombre_localidad"].apply(
    lambda x: LOC_NAME_TO_CODE.get(normalize(x), np.nan)
)
unmapped = gdf["cod_localidad"].isna().sum()
if unmapped > 0:
    print(f"  WARNING: {unmapped} colegios sin localidad mapeada")

# ── 4. Guardar CSV minimalista ────────────────────────────────────────────────
out = gdf[["id_establecimiento", "capacidad", "cod_localidad"]].copy()
out["cod_localidad"] = out["cod_localidad"].astype("Int64")
out.to_csv(OUT_PATH, index=False)

cap_total = out["capacidad"].sum()
print(f"\nGuardado: {OUT_PATH}")
print(f"  {len(out):,} colegios")
print(f"  Capacidad total: {cap_total:,} cupos (target: {CUPOS_TOTALES:,})")
print(f"  Rango capacidad: {out['capacidad'].min()} - {out['capacidad'].max()}")
