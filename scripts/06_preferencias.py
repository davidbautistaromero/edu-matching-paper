"""
06_preferencias.py
==================
Calcula utilidades u_ij y rankings de preferencia para cada familia sobre colegios oficiales.

Modelo: u_ij = q_j - alpha_s * log1p(d_ij) + epsilon_ij
  q_j     : score de calidad predicho por ElasticNet M1 (NMF)
  alpha_s : penalización distancia heterogénea por estrato (Hastings et al. 2009)
             alpha_s = 0.30 / s^0.613  (power law calibrada: ratio estrato1/estrato6 = 3x)
  d_ij    : distancia en km (familia i → colegio j)
  epsilon : ruido Gumbel(0,1) — genera estocasticidad en rankings

Choice set: familias eligen solo dentro de su localidad.
Excepción: La Candelaria (localidad 17) puede elegir en localidades 17, 3, 14, 15.

Outputs:
  data/processed/preferencias_familias.parquet  — top-20 rankings por familia
  data/processed/utilidades_familias.parquet    — matriz de utilidades completa (float32)
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODEL_PATH  = ROOT / "models" / "elasticnet_M1.joblib"
META_PATH   = ROOT / "models" / "elasticnet_M1_meta.json"
COLEGIOS    = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
NMF_PATH    = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8.parquet"
FAM_PATH    = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
DIST_PATH   = ROOT / "data" / "processed" / "familias_distancias.parquet"
OUT_DIR     = ROOT / "data" / "processed"

# Candelaria (17) → puede elegir en estas localidades también
CANDELARIA_EXTRA = {3, 14, 15}

# Mapping nombre_localidad → código numérico (estándar Bogotá)
LOC_NAME_TO_CODE = {
    "Usaquén": 1, "Chapinero": 2, "Santa Fe": 3, "San Cristóbal": 4,
    "Usme": 5, "Tunjuelito": 6, "Bosa": 7, "Kennedy": 8,
    "Fontibón": 9, "Engativá": 10, "Suba": 11, "Barrios Unidos": 12,
    "Teusaquillo": 13, "Los Mártires": 14, "Antonio Nariño": 15,
    "Puente Aranda": 16, "La Candelaria": 17, "Rafael Uribe Uribe": 18,
    "Ciudad Bolívar": 19, "Sumapaz": 20,
}

ALPHA_0 = 0.30
GAMMA   = np.log(3) / np.log(6)   # ≈ 0.613
TOP_K   = 20
SEED    = 42

# ── Step 1: Load model and predict q_j ────────────────────────────────────────
log.info("Step 1 — Cargando modelo y prediciendo q_j...")
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
features = meta["features"]

gdf = gpd.read_file(COLEGIOS)
nmf = pd.read_parquet(NMF_PATH)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
nmf["id_establecimiento"] = nmf["id_establecimiento"].astype(str)
df  = gdf.merge(nmf[["id_establecimiento"] + [c for c in nmf.columns if c.startswith("topic_")]],
                on="id_establecimiento", how="inner")

df["puntaje_icfes_promedio"] = df[["puntaje_2023", "punt_global_2022", "punt_global_2020"]].mean(axis=1)
for k in range(2, 7):
    df[f"estrato_{k}"] = pd.to_numeric(df[f"pct_estrato_{k}"], errors="coerce")

# Capacidad pública por localidad (excluyendo el propio colegio) — replica 04_regresion.py
cap_loc = df.groupby("nombre_localidad")["matricula_total"].transform("sum")
df["n_oficiales_localidad"] = (cap_loc - df["matricula_total"]).clip(lower=0)

X_col = df[features]
q_j   = model.predict(X_col)
school_ids = df["id_establecimiento"].values

# Crear código numérico de localidad para colegios
# Normalizar encoding (el geojson puede tener tildes dañadas)
import unicodedata
def normalize_name(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    # Intentar normalizar unicode
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    return s

df["_nombre_loc_norm"] = df["nombre_localidad"].apply(normalize_name)
# Build mapping with normalized keys too
loc_map = {}
for name, code in LOC_NAME_TO_CODE.items():
    loc_map[name] = code
    loc_map[normalize_name(name)] = code

df["cod_localidad"] = df["_nombre_loc_norm"].map(loc_map)
# Fallback: fuzzy match for encoding issues
unmapped = df["cod_localidad"].isna().sum()
if unmapped > 0:
    log.warning(f"  {unmapped} colegios sin match de localidad — intentando match parcial")
    for idx in df[df["cod_localidad"].isna()].index:
        raw = str(df.loc[idx, "nombre_localidad"])
        for name, code in LOC_NAME_TO_CODE.items():
            # Compare first 4 chars (ignoring encoding)
            if raw[:4].lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u") == name[:4].lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u"):
                df.loc[idx, "cod_localidad"] = code
                break

school_loc = df["cod_localidad"].values
log.info(f"  Colegios con localidad asignada: {(~pd.isna(school_loc)).sum()} / {len(school_loc)}")

log.info(f"  Colegios puntuados: {len(q_j)} | q_j min={q_j.min():.4f} max={q_j.max():.4f}")

# ── Step 2: Load families and distances ───────────────────────────────────────
log.info("Step 2 — Cargando familias y distancias...")
fam  = pd.read_parquet(FAM_PATH)
dist = pd.read_parquet(DIST_PATH)

# Align: keep families present in both, schools present in both
common_fam     = fam["DIRECTORIO"].isin(dist.index)
fam            = fam[common_fam].reset_index(drop=True)
dist           = dist.loc[fam["DIRECTORIO"]]

common_schools = [sid for sid in school_ids if sid in dist.columns]
dist           = dist[common_schools]
idx_schools    = [list(school_ids).index(sid) for sid in common_schools]
q_j_aligned    = q_j[idx_schools]
school_ids_al  = np.array(common_schools)
school_loc_al  = school_loc[idx_schools]

dist_matrix    = dist.values.astype(np.float32)   # (n_fam, n_schools)
log.info(f"  Familias: {len(fam):,} | Colegios: {len(school_ids_al)}")

# ── Step 3: Compute alpha_s per family ────────────────────────────────────────
log.info("Step 3 — Calculando alpha_s por estrato...")
estrato = pd.to_numeric(fam["estrato_real"], errors="coerce").fillna(3).clip(1, 6).values
alpha_s = ALPHA_0 / estrato ** GAMMA   # (n_fam,)
for s in range(1, 7):
    n = (estrato == s).sum()
    log.info(f"  Estrato {s}: {n:,} familias | alpha={ALPHA_0 / s**GAMMA:.4f}")

# ── Step 4: Build choice set mask ─────────────────────────────────────────────
log.info("Step 4 — Construyendo choice sets por localidad...")
fam_loc = pd.to_numeric(fam["COD_LOCALIDAD"], errors="coerce").values   # (n_fam,)

# school_loc_al may be numeric or string — normalize to numeric if possible
try:
    school_loc_num = pd.to_numeric(school_loc_al, errors="coerce")
except Exception:
    school_loc_num = school_loc_al

mask = np.zeros((len(fam), len(school_ids_al)), dtype=bool)
for i, loc_i in enumerate(fam_loc):
    if pd.isna(loc_i):
        mask[i, :] = True   # no localidad info: allow all
        continue
    loc_i = int(loc_i)
    allowed = {loc_i}
    if loc_i == 17:
        allowed |= CANDELARIA_EXTRA
    mask[i, :] = np.isin(school_loc_num, list(allowed))

sizes = mask.sum(axis=1)
log.info(f"  Choice set promedio: {sizes.mean():.1f} | min={sizes.min()} | max={sizes.max()}")

# ── Step 5: Compute utility matrix ────────────────────────────────────────────
log.info("Step 5 — Calculando matriz de utilidad...")
rng     = np.random.default_rng(SEED)
epsilon = rng.gumbel(0, 1, size=(len(fam), len(school_ids_al))).astype(np.float32)

log_dist = np.log1p(dist_matrix)   # ln(1 + d_ij)
u = (q_j_aligned[None, :].astype(np.float32)
     - alpha_s[:, None].astype(np.float32) * log_dist
     + epsilon)

# Apply choice set: -inf outside
u[~mask] = -np.inf
log.info(f"  Utilidad matrix shape: {u.shape}")

# ── Step 6: Generate rankings ──────────────────────────────────────────────────
log.info("Step 6 — Generando rankings top-20...")
rankings_idx = np.argsort(-u, axis=1)[:, :TOP_K]   # (n_fam, TOP_K)
rankings_ids = school_ids_al[rankings_idx]           # replace indices with IDs

rankings_df = pd.DataFrame(
    rankings_ids,
    index=fam["DIRECTORIO"].values,
    columns=[f"pref_{k+1}" for k in range(TOP_K)],
)
rankings_df.index.name = "DIRECTORIO"

# ── Step 7: Save ──────────────────────────────────────────────────────────────
log.info("Step 7 — Guardando outputs...")
out_rank = ROOT / "data" / "primary" / "preferencias_familias.parquet"
out_util = OUT_DIR / "utilidades_familias.parquet"

rankings_df.to_parquet(out_rank)
log.info(f"  preferencias_familias.parquet  ({out_rank.stat().st_size / 1e6:.1f} MB)")

util_df = pd.DataFrame(u, index=fam["DIRECTORIO"].values, columns=school_ids_al)
util_df.index.name = "DIRECTORIO"
util_df.to_parquet(out_util)
log.info(f"  utilidades_familias.parquet    ({out_util.stat().st_size / 1e6:.1f} MB)")

log.info("Done.")
