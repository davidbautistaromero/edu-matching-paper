"""
04c_decompose_aj.py
===================
Descompone el índice de atractivo a_j en su componente visual y no visual
usando los coeficientes del Ridge M1 ya entrenado.

Motivación
----------
a_j = Ridge(X_visual + X_nonvisual) = a_j_visual + a_j_nonvisual + intercept

Como el Ridge es lineal, la descomposición es exacta:
  a_j_visual    = Σ β_topic_k · topic_k     (k=1..8, features NMF)
  a_j_nonvisual = Σ β_otros · X_otros       (calidad, seguridad, transporte, etc.)
  intercept     = β_0

Esta descomposición permite medir el sesgo visual del mecanismo de asignación:
  sesgo_visual = corr(estrato_familia, a_j_visual_colegio_asignado)

Inputs
------
  models/ridge_m1.joblib                         — pipeline Ridge M1 entrenado
  models/ridge_m1_meta.json                      — features y metadata
  data/primary/colegios_features_imputed.geojson — features no visuales por colegio
  data/images/embeddings/gsv_nmf_K8.parquet      — topics NMF por colegio
  data/primary/colegios_capacidad.parquet        — tabla maestra de colegios

Output
------
  data/primary/colegios_capacidad.parquet        — actualizado con:
      a_j            : atractivo compuesto (predicción completa del Ridge)
      a_j_visual     : componente visual (solo topics NMF)
      a_j_nonvisual  : componente no visual (calidad, seguridad, transporte, etc.)
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
MODEL_P    = ROOT / "models" / "ridge_m1.joblib"
META_P     = ROOT / "models" / "ridge_m1_meta.json"
GEO_P      = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
NMF_P      = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8.parquet"
CAP_P      = ROOT / "data" / "primary" / "colegios_capacidad.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar modelo y metadata
# ─────────────────────────────────────────────────────────────────────────────
log.info("Cargando Ridge M1...")
pipe = joblib.load(MODEL_P)
meta = json.loads(META_P.read_text(encoding="utf-8"))

all_features   = meta["features"]                                        # orden exacto del modelo
visual_features = [f for f in all_features if f.startswith("topic_")]
other_features  = [f for f in all_features if not f.startswith("topic_")]

log.info(f"  Features totales   : {len(all_features)}")
log.info(f"  Features visuales  : {visual_features}")
log.info(f"  Features no visual : {other_features}")

# Extraer el estimador Ridge del pipeline
# El pipeline puede tener scaler + ridge o solo ridge
regressor = pipe[-1]
coef      = regressor.coef_               # shape (n_features,)
intercept = float(regressor.intercept_)

# Índices de features visuales y no visuales dentro del vector de coefs
# Nota: si el pipeline tiene un scaler, los coefs ya están en espacio escalado.
# Para la descomposición usamos predict() con matrices parciales, no los coefs crudos.
# Esto evita tener que invertir el scaler manualmente.
log.info(f"  Intercepto: {intercept:.4f}")
log.info(f"  Coefs (top 5 por magnitud):")
for f, c in sorted(zip(all_features, coef), key=lambda x: abs(x[1]), reverse=True)[:5]:
    log.info(f"    {f:<35} {c:+.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cargar y unir features — replicando el feature engineering de 04_regresion.py
# ─────────────────────────────────────────────────────────────────────────────
log.info("Cargando features de colegios...")

gdf = gpd.read_file(GEO_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
df = pd.DataFrame(gdf.drop(columns="geometry"))

nmf = pd.read_parquet(NMF_P)
nmf["id_establecimiento"] = nmf["id_establecimiento"].astype(str)
df = df.merge(nmf, on="id_establecimiento", how="left")

log.info(f"  Colegios totales  : {len(df)}")
log.info(f"  Con topics NMF    : {df['topic_1'].notna().sum()}")

# ── Feature engineering (idéntico a 04_regresion.py) ─────────────────────────
df["puntaje_icfes_promedio"] = (
    df[["puntaje_2023", "punt_global_2022", "punt_global_2020"]]
    .apply(pd.to_numeric, errors="coerce")
    .mean(axis=1)
)

for k in [2, 3, 4, 5, 6]:
    df[f"estrato_{k}"] = pd.to_numeric(df[f"pct_estrato_{k}"], errors="coerce")

df["es_rural"]   = df["zona"].isin(["RURAL", "EXPANSION"]).astype(float)
df["es_tecnico"] = df["caracter_media"].isin(["Tecnico", "Academico - Tecnico"]).astype(float)
df.loc[df["caracter_media"].isin(["Sin informacion", "Sin información"]), "es_tecnico"] = np.nan

df["matricula_total"] = pd.to_numeric(df["matricula_total"], errors="coerce")
cap_loc = df.groupby("nombre_localidad")["matricula_total"].transform("sum")
df["n_oficiales_localidad"] = (cap_loc - df["matricula_total"]).clip(lower=0)

# Convertir controles a numérico
for col in other_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Construir X con todas las features en el orden exacto del modelo
X = df[all_features].copy()

# Imputar NaN con la media de la columna (mismo criterio que 04_regresion.py)
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].mean())

log.info(f"  NaN restantes en X: {X.isna().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Descomposición exacta de a_j
# ─────────────────────────────────────────────────────────────────────────────
log.info("Descomponiendo a_j en visual y no visual...")

# Predicción completa
a_j_full = pipe.predict(X)

# Para descomponer: construir X con las features no-visuales en cero → obtenemos
# solo la contribución visual. Luego a_j_nonvisual = a_j_full - a_j_visual_contrib.
#
# Nota: esto funciona porque predict() es lineal después del scaler.
# Con un scaler estándar: predict(X) = coef @ scaler.transform(X) + intercept
# No podemos simplemente poner ceros en X porque el scaler centraría las otras
# variables. En cambio, usamos el truco de la diferencia:
#
#   Contribución visual = predict(X) - predict(X con topics=0)
#   Contribución no visual = predict(X con topics=0) - intercept
#
# Esto es exacto independientemente del scaler.

X_no_visual = X.copy()
X_no_visual[visual_features] = 0.0

a_j_no_vis_plus_intercept = pipe.predict(X_no_visual)

a_j_visual    = a_j_full - a_j_no_vis_plus_intercept   # contribución pura de topics
a_j_nonvisual = a_j_no_vis_plus_intercept               # resto (incluye intercept)

# Verificación: a_j_visual + a_j_nonvisual = a_j_full (por construcción)
max_diff = np.max(np.abs(a_j_visual + a_j_nonvisual - a_j_full))
log.info(f"  Verificación descomposición (max diff): {max_diff:.2e}")

log.info(f"  a_j        — media: {a_j_full.mean():.4f}  std: {a_j_full.std():.4f}")
log.info(f"  a_j_visual — media: {a_j_visual.mean():.4f}  std: {a_j_visual.std():.4f}")
log.info(f"  a_j_nonvis — media: {a_j_nonvisual.mean():.4f}  std: {a_j_nonvisual.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Actualizar colegios_capacidad.parquet
# ─────────────────────────────────────────────────────────────────────────────
log.info("Actualizando colegios_capacidad.parquet...")

cap = pd.read_parquet(CAP_P)
cap["id_establecimiento"] = cap["id_establecimiento"].astype(str)

# Tabla de componentes para merge
decomp = pd.DataFrame({
    "id_establecimiento" : df["id_establecimiento"].values,
    "a_j"                : a_j_full,
    "a_j_visual"         : a_j_visual,
    "a_j_nonvisual"      : a_j_nonvisual,
})

# Quitar columnas viejas si existen y reemplazar
for col in ["a_j", "a_j_visual", "a_j_nonvisual"]:
    if col in cap.columns:
        cap = cap.drop(columns=[col])

cap = cap.merge(decomp, on="id_establecimiento", how="left")

n_with_aj = cap["a_j"].notna().sum()
log.info(f"  Colegios con a_j calculado: {n_with_aj} / {len(cap)}")
log.info(f"  Colegios sin NMF (a_j NaN): {cap['a_j'].isna().sum()}")

cap.to_parquet(CAP_P, index=False)
log.info(f"  Guardado: {CAP_P}")

log.info("=" * 60)
log.info("Columnas finales en colegios_capacidad.parquet:")
log.info(f"  {cap.columns.tolist()}")
log.info("Done.")
