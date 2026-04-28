# -*- coding: utf-8 -*-
"""
06_preferencias.py
==================
Calcula utilidades u_ij y rankings de preferencia para cada familia sobre colegios oficiales.

Modelo: u_ij = a_j - alpha_s * log1p(d_ij) + epsilon_ij
  a_j     : indice de atractivo del colegio j (predice log_sobredemanda con features observables)
  alpha_s : penalizacion distancia heterogenea por estrato (Hastings et al. 2009)
             alpha_s = 0.30 / s^0.613  (power law calibrada: ratio estrato1/estrato6 = 3x)
  d_ij    : distancia en km (familia i -> colegio j)
  epsilon : ruido Gumbel(0,1) -- genera estocasticidad en rankings

Choice set: familias eligen solo dentro de su localidad.
Excepcion: La Candelaria (localidad 17) puede elegir en localidades 17, 3, 14, 15.

Inputs:
  data/processed/familias_expandidas.parquet     -- familias expandidas con FEX_C
  data/processed/distancias_expandidas.parquet   -- matriz de distancias expandida

Outputs:
  data/primary/preferencias_familias.parquet     -- top-20 rankings por familia
  data/processed/utilidades_familias.parquet     -- matriz de utilidades (escrita en batches)
"""

import json
import logging
import unicodedata
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

def _find_best_model(models_dir):
    import glob
    metas = glob.glob(str(models_dir / "*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No *_meta.json found in {models_dir}")
    best_meta, best_rmse = None, float("inf")
    for mp in metas:
        with open(mp) as f:
            m = json.load(f)
        rmse = m.get("rmse_cv", float("inf"))
        if rmse < best_rmse:
            best_rmse, best_meta = rmse, mp
    meta_path = Path(best_meta)
    model_path = meta_path.parent / (meta_path.stem.replace("_meta", "") + ".joblib")
    return model_path, meta_path

MODEL_PATH, META_PATH = _find_best_model(ROOT / "models")
COLEGIOS  = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
NMF_PATH  = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8.parquet"
FAM_PATH  = ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_PATH = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
OUT_DIR   = ROOT / "data" / "processed"

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

ALPHA_0 = 0.30
GAMMA   = np.log(3) / np.log(6)   # ~0.613
TOP_K   = 20
SEED    = 42

# ── Step 1: Cargar modelo y predecir a_j ──────────────────────────────────────
log.info("Step 1 -- Cargando modelo y prediciendo a_j...")
model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)
features = meta["features"]

gdf = gpd.read_file(COLEGIOS)
nmf = pd.read_parquet(NMF_PATH)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
nmf["id_establecimiento"] = nmf["id_establecimiento"].astype(str)
df = gdf.merge(nmf[["id_establecimiento"] + [c for c in nmf.columns if c.startswith("topic_")]],
               on="id_establecimiento", how="inner")

df["puntaje_icfes_promedio"] = df[["puntaje_2023", "punt_global_2022", "punt_global_2020"]].mean(axis=1)
for k in range(2, 7):
    df[f"estrato_{k}"] = pd.to_numeric(df[f"pct_estrato_{k}"], errors="coerce")

cap_loc = df.groupby("nombre_localidad")["matricula_total"].transform("sum")
df["n_oficiales_localidad"] = (cap_loc - df["matricula_total"]).clip(lower=0)

X_col = df[features]
a_j   = model.predict(X_col)
school_ids = df["id_establecimiento"].values

# Codigo numerico de localidad para colegios (con fallback para encoding)
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower().strip()

loc_map = {}
for name, code in LOC_NAME_TO_CODE.items():
    loc_map[strip_accents(name)] = code

df["cod_localidad"] = df["nombre_localidad"].apply(
    lambda x: loc_map.get(strip_accents(x)) if pd.notna(x) else None
)

unmapped = df["cod_localidad"].isna().sum()
if unmapped > 0:
    log.warning(f"  {unmapped} colegios sin match de localidad -- fallback parcial")
    for idx in df[df["cod_localidad"].isna()].index:
        raw = strip_accents(str(df.loc[idx, "nombre_localidad"]))
        for name, code in LOC_NAME_TO_CODE.items():
            if raw[:5] == strip_accents(name)[:5]:
                df.loc[idx, "cod_localidad"] = code
                break

school_loc = df["cod_localidad"].values
log.info(f"  Colegios con localidad asignada: {(~pd.isna(school_loc)).sum()} / {len(school_loc)}")
log.info(f"  Colegios puntuados: {len(a_j)} | a_j min={a_j.min():.4f} max={a_j.max():.4f}")

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
a_j_aligned    = a_j[idx_schools]
school_ids_al  = np.array(common_schools)
school_loc_al  = school_loc[idx_schools]

dist_matrix = dist.values.astype(np.float32)
log.info(f"  Familias: {len(fam):,} | Colegios: {len(school_ids_al)}")

# ── Step 3: alpha_s por estrato ───────────────────────────────────────────────
log.info("Step 3 -- Calculando alpha_s por estrato...")
estrato = pd.to_numeric(fam["estrato_real"], errors="coerce").fillna(3).clip(1, 6).values
alpha_s = ALPHA_0 / estrato ** GAMMA
for s in range(1, 7):
    n_s = (estrato == s).sum()
    log.info(f"  Estrato {s}: {n_s:,} familias | alpha={ALPHA_0 / s**GAMMA:.4f}")

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
    b_dist  = dist_matrix[batch_start:batch_end]
    b_alpha = alpha_s[batch_start:batch_end]
    b_mask  = mask[batch_start:batch_end]

    epsilon  = rng.gumbel(0, 1, size=(batch_end - batch_start, len(school_ids_al))).astype(np.float32)
    log_dist = np.log1p(b_dist)
    u = (a_j_aligned[None, :].astype(np.float32)
         - b_alpha[:, None].astype(np.float32) * log_dist
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
log.info("Done.")
