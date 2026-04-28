"""
06b_capacidad_colegios.py
=========================
Calcula la capacidad estimada de cada colegio oficial y genera un parquet
con la informacion de colegios que entra al modelo de matching.

Formula:
    capacidad = max(round(matricula_total / 13), 5)
    (13 anios de escolaridad kinder-grado11; minimo 5 cupos)

Inputs:
    data/primary/colegios_features_imputed.geojson
    models/ridge_m1.joblib
    models/ridge_m1_meta.json
    data/images/embeddings/gsv_nmf_K8.parquet

Output:
    data/primary/colegios_capacidad.parquet
"""

import json
import unicodedata
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
GEO_PATH = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
NMF_PATH = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8.parquet"
MODEL_P  = ROOT / "models" / "ridge_m1.joblib"
META_P   = ROOT / "models" / "ridge_m1_meta.json"
OUT_PATH = ROOT / "data" / "primary" / "colegios_capacidad.parquet"

LOC_NAME_TO_CODE = {
    "USAQUEN": 1, "CHAPINERO": 2, "SANTA FE": 3, "SAN CRISTOBAL": 4,
    "USME": 5, "TUNJUELITO": 6, "BOSA": 7, "KENNEDY": 8,
    "FONTIBON": 9, "ENGATIVA": 10, "SUBA": 11, "BARRIOS UNIDOS": 12,
    "TEUSAQUILLO": 13, "LOS MARTIRES": 14, "ANTONIO NARINO": 15,
    "PUENTE ARANDA": 16, "CANDELARIA": 17, "RAFAEL URIBE URIBE": 18,
    "CIUDAD BOLIVAR": 19,
}

CUPOS_TOTALES = 120_000  # cupos disponibles en el sistema, distribuidos proporcionalmente


def normalize(s):
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ── 1. Cargar geojson ─────────────────────────────────────────────────────────
print("Cargando colegios_features_imputed.geojson...")
gdf = gpd.read_file(GEO_PATH)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
print(f"  {len(gdf):,} establecimientos")

# ── 2. Cargar NMF, merge e imputación espacial ────────────────────────────────
print("Cargando NMF K8...")
nmf = pd.read_parquet(NMF_PATH)
nmf["id_establecimiento"] = nmf["id_establecimiento"].astype(str)
topic_cols = [c for c in nmf.columns if c.startswith("topic_")]

df = gdf.merge(nmf[["id_establecimiento"] + topic_cols],
               on="id_establecimiento", how="left")
n_sin_nmf = df[topic_cols[0]].isna().sum()
print(f"  Tras merge NMF: {len(df):,} ({n_sin_nmf} sin topics — imputando espacialmente)")

if n_sin_nmf > 0:
    # Coordenadas de todos los colegios
    df["_lat"] = df.geometry.centroid.y
    df["_lon"] = df.geometry.centroid.x

    sin_idx = df[df[topic_cols[0]].isna()].index
    con_idx = df[df[topic_cols[0]].notna()].index

    lat_r  = np.radians(df.loc[con_idx, "_lat"].values)
    lon_r  = np.radians(df.loc[con_idx, "_lon"].values)
    topics_ref = df.loc[con_idx, topic_cols].values  # (n_con, K)

    RADIO_KM = 2.0
    imputed = 0
    for i in sin_idx:
        lat_i = np.radians(df.at[i, "_lat"])
        lon_i = np.radians(df.at[i, "_lon"])
        # Haversine vectorizado
        dlat = lat_r - lat_i
        dlon = lon_r - lon_i
        a    = np.sin(dlat/2)**2 + np.cos(lat_i) * np.cos(lat_r) * np.sin(dlon/2)**2
        dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        vecinos = dist <= RADIO_KM
        if vecinos.sum() > 0:
            df.loc[i, topic_cols] = topics_ref[vecinos].mean(axis=0)
            imputed += 1
        else:
            # Sin vecinos en 2km: usar media global
            df.loc[i, topic_cols] = df.loc[con_idx, topic_cols].mean().values
            imputed += 1

    df = df.drop(columns=["_lat", "_lon"])
    print(f"  Imputados: {imputed} / {n_sin_nmf} colegios sin NMF")

# ── 3. Construir features del modelo ──────────────────────────────────────────
print("Construyendo features del modelo Ridge M1...")

# puntaje_icfes_promedio
df["puntaje_icfes_promedio"] = df[["puntaje_2023", "punt_global_2022",
                                    "punt_global_2020"]].apply(
    pd.to_numeric, errors="coerce").mean(axis=1)

# estratos (pct_estrato_k → float)
for k in [2, 3, 4, 5, 6]:
    df[f"estrato_{k}"] = pd.to_numeric(df[f"pct_estrato_{k}"], errors="coerce")

# n_oficiales_localidad
df["matricula_total"] = pd.to_numeric(df["matricula_total"], errors="coerce")
cap_loc = df.groupby("nombre_localidad")["matricula_total"].transform("sum")
df["n_oficiales_localidad"] = (cap_loc - df["matricula_total"]).clip(lower=0)

# es_rural, es_tecnico
df["es_rural"] = df["zona"].isin(["RURAL", "EXPANSION"]).astype(float)
df["es_tecnico"] = np.where(
    df["caracter_media"].isin(["Sin informacion", "Sin información"]), 0.0,
    df["caracter_media"].isin(["Tecnico", "Academico - Tecnico"]).astype(float)
)

# ── 4. Predecir a_j ───────────────────────────────────────────────────────────
print("Prediciendo a_j con Ridge M1...")
with open(META_P) as f:
    meta = json.load(f)
features = meta["features"]

model = joblib.load(MODEL_P)

# Subset con features completas para prediccion
df_model = df[features].apply(pd.to_numeric, errors="coerce")
has_features = df_model.notna().all(axis=1)

df["a_j"] = np.nan
X = df_model[has_features].values
df.loc[has_features, "a_j"] = model.predict(X)
print(f"  Colegios con a_j predicho: {has_features.sum()} / {len(df)}")
print(f"  Colegios sin a_j (NaN): {(~has_features).sum()}")

# ── 5. Capacidad ──────────────────────────────────────────────────────────────
# Distribucion proporcional: capacidad_j = round(120K * matricula_j / sum(matricula_j))
mat_total_sistema = df["matricula_total"].sum()
df["capacidad"] = (CUPOS_TOTALES * df["matricula_total"] / mat_total_sistema).round().clip(lower=5).astype("Int64")

# ── 6. cod_localidad ──────────────────────────────────────────────────────────
df["cod_localidad"] = df["nombre_localidad"].apply(
    lambda x: LOC_NAME_TO_CODE.get(normalize(x), np.nan)
)

# ── 7. Coordenadas ────────────────────────────────────────────────────────────
if "lat" not in df.columns:
    df["lat"] = df.geometry.centroid.y
    df["lon"] = df.geometry.centroid.x

# ── 8. Guardar ────────────────────────────────────────────────────────────────
cols_out = [
    "id_establecimiento", "nombre_establecimiento", "nombre_localidad",
    "cod_localidad", "lat", "lon", "q_j", "sobre_demanda_j",
    "matricula_total", "capacidad", "a_j",
]
out = df[[c for c in cols_out if c in df.columns]].copy()
out.to_parquet(OUT_PATH, index=False)
print(f"\nGuardado: {OUT_PATH} ({len(out):,} colegios)")

# ── 9. Verificacion ───────────────────────────────────────────────────────────
cap_total = out["capacidad"].sum()
print(f"\n{'='*55}")
print(f"Total colegios:          {len(out):,}")
print(f"Capacidad total:         {cap_total:,.0f} cupos (debe ser ~{CUPOS_TOTALES:,})")
print(f"Matricula total sistema: {mat_total_sistema:,.0f}")
print(f"\nTop 5 colegios por capacidad:")
print(out.nlargest(5, "capacidad")[["nombre_establecimiento", "capacidad", "matricula_total"]].to_string(index=False))
print(f"\nCapacidad por localidad:")
print(out.groupby("nombre_localidad")["capacidad"].sum().sort_values(ascending=False).to_string())
print(f"\nColegios sin a_j: {out['a_j'].isna().sum()}")
print("="*55)
