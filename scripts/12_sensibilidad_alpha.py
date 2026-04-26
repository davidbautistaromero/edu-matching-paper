"""
12_sensibilidad_alpha.py
========================
Análisis de sensibilidad sobre la calibración de la penalización de distancia
en 06_preferencias.py.

La especificación actual usa alpha_s = alpha_0 / s^gamma con alpha_0=0.30 y
gamma=log(3)/log(6)≈0.613. Estos valores son una calibración ad-hoc inspirada
en Hastings et al. 2009 pero no toman números directamente del paper. Este
script verifica que los resultados de matching sean robustos a variaciones
plausibles de ambos parámetros.

Barrido: 3 valores de alpha_0 x 3 valores de gamma = 9 combinaciones.
Para cada combinación se reconstruyen las preferencias top-20 y se corre DA
sobre 13,568 familias x 303 colegios (igual que 07_matching_bm_da.py).

Outputs:
  reports/sensibilidad_alpha_comparison.csv
  data/results/matching_da_a{0}_g{1}.parquet  (9 archivos)
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from matching_utils import compute_metrics, deferred_acceptance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
MODEL_P  = ROOT / "models" / "elasticnet_M1.joblib"
COL_P    = ROOT / "data" / "primary"   / "colegios_features_imputed.geojson"
NMF_P    = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8.parquet"
FAM_P    = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
DIST_P   = ROOT / "data" / "processed" / "familias_distancias.parquet"
OUT_DIR  = ROOT / "data" / "results"
REP_DIR  = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_0_VALUES = [0.15, 0.30, 0.45]
GAMMA_VALUES   = [0.40, 0.613, 0.80]
TOP_K          = 20
SEED           = 42
COHORT_DIVISOR = 13
CAPACITY_MIN   = 5

CANDELARIA_EXTRA = {3, 14, 15}
LOC_NAME_TO_CODE = {
    "Usaquén": 1, "Chapinero": 2, "Santa Fe": 3, "San Cristóbal": 4,
    "Usme": 5, "Tunjuelito": 6, "Bosa": 7, "Kennedy": 8,
    "Fontibón": 9, "Engativá": 10, "Suba": 11, "Barrios Unidos": 12,
    "Teusaquillo": 13, "Los Mártires": 14, "Antonio Nariño": 15,
    "Puente Aranda": 16, "La Candelaria": 17, "Rafael Uribe Uribe": 18,
    "Ciudad Bolívar": 19, "Sumapaz": 20,
}

log.info("=" * 60)
log.info("Paso 1 — Cargando modelo y datos...")

model    = joblib.load(MODEL_P)
features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None

gdf = gpd.read_file(COL_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)
nmf = pd.read_parquet(NMF_P)
nmf["id_establecimiento"] = nmf["id_establecimiento"].astype(str)
df  = gdf.merge(nmf[["id_establecimiento"] + [c for c in nmf.columns if c.startswith("topic_")]],
                on="id_establecimiento", how="inner")

df["puntaje_icfes_promedio"] = df[["puntaje_2023", "punt_global_2022", "punt_global_2020"]].mean(axis=1)
for k in range(2, 7):
    df[f"estrato_{k}"] = pd.to_numeric(df[f"pct_estrato_{k}"], errors="coerce")
cap_loc = df.groupby("nombre_localidad")["matricula_total"].transform("sum")
df["n_oficiales_localidad"] = (cap_loc - df["matricula_total"]).clip(lower=0)

import json
with open(ROOT / "models" / "elasticnet_M1_meta.json") as f:
    meta = json.load(f)
features = meta["features"]

X_col = df[features]
q_j   = model.predict(X_col)
school_ids = df["id_establecimiento"].values

import unicodedata
def normalize_name(s):
    if pd.isna(s):
        return s
    return unicodedata.normalize("NFC", str(s).strip())

df["_loc_norm"]    = df["nombre_localidad"].apply(normalize_name)
loc_map            = {**LOC_NAME_TO_CODE, **{normalize_name(k): v for k, v in LOC_NAME_TO_CODE.items()}}
df["cod_localidad"] = df["_loc_norm"].map(loc_map)
school_loc          = df["cod_localidad"].values

fam  = pd.read_parquet(FAM_P)
dist = pd.read_parquet(DIST_P)

common_fam = fam["DIRECTORIO"].isin(dist.index)
fam        = fam[common_fam].reset_index(drop=True)
dist       = dist.loc[fam["DIRECTORIO"]]

common_schools = [sid for sid in school_ids if sid in dist.columns]
dist           = dist[common_schools]
idx_schools    = [list(school_ids).index(sid) for sid in common_schools]
q_j_aligned    = q_j[idx_schools]
school_ids_al  = np.array(common_schools)
school_loc_al  = pd.to_numeric(school_loc[idx_schools], errors="coerce")

dist_matrix = dist.values.astype(np.float32)
log_dist    = np.log1p(dist_matrix)

estrato_arr = pd.to_numeric(fam["estrato_real"], errors="coerce").fillna(3).clip(1, 6).values
fam_loc     = pd.to_numeric(fam["COD_LOCALIDAD"], errors="coerce").values

log.info(f"  Familias: {len(fam):,} | Colegios: {len(school_ids_al)}")

log.info("Paso 2 — Construyendo choice sets...")
mask = np.zeros((len(fam), len(school_ids_al)), dtype=bool)
for i, loc_i in enumerate(fam_loc):
    if pd.isna(loc_i):
        mask[i, :] = True
        continue
    loc_i = int(loc_i)
    allowed = {loc_i}
    if loc_i == 17:
        allowed |= CANDELARIA_EXTRA
    mask[i, :] = np.isin(school_loc_al, list(allowed))

log.info("Paso 3 — Capacidades escolares...")
gdf["capacidad"] = (gdf["matricula_total"] / COHORT_DIVISOR).round().clip(lower=CAPACITY_MIN).astype(int)
school_info = gdf.set_index("id_establecimiento")[["capacidad", "q_j", "sobre_demanda_j", "nombre_localidad"]].copy()
school_cap  = school_info.loc[school_ids_al, "capacidad"].to_dict()

dist_matrix_dict = {sid: dist_matrix[:, k] for k, sid in enumerate(school_ids_al)}

def make_priority_fn():
    school_col_idx = {sid: k for k, sid in enumerate(school_ids_al)}
    def priority_fn(student_idx: int, school_id: str) -> float:
        col = school_col_idx.get(school_id)
        if col is None:
            return np.inf
        return float(dist_matrix[student_idx, col])
    return priority_fn

priority_fn = make_priority_fn()

rng = np.random.default_rng(SEED)
epsilon_base = rng.gumbel(0, 1, size=(len(fam), len(school_ids_al))).astype(np.float32)

log.info("=" * 60)
log.info("Paso 4 — Barrido sobre 9 combinaciones (alpha_0, gamma)...")

results     = []
assignments = {}

for a0 in ALPHA_0_VALUES:
    for g in GAMMA_VALUES:
        label   = f"a{int(a0*100):03d}_g{int(g*100):03d}"
        alpha_s = (a0 / estrato_arr ** g).astype(np.float32)

        u = (q_j_aligned[None, :].astype(np.float32)
             - alpha_s[:, None] * log_dist
             + epsilon_base)
        u[~mask] = -np.inf

        rank_idx = np.argsort(-u, axis=1)[:, :TOP_K]
        rank_ids = school_ids_al[rank_idx]

        valid_schools = set(school_ids_al)
        pref_lists = [
            [s for s in row if s in valid_schools]
            for row in rank_ids
        ]

        log.info(f"  {label}: alpha_0={a0} gamma={g} | corriendo DA...")
        t0 = time.time()
        asgn = deferred_acceptance(pref_lists, school_cap, priority_fn)
        dt = time.time() - t0

        met = compute_metrics(
            asgn, pref_lists, school_cap, priority_fn,
            school_info, quality_col="q_j", visual_col="sobre_demanda_j",
            estrato_arr=estrato_arr, label=label,
        )
        met["alpha_0"] = a0
        met["gamma"]   = g
        met["t_da_s"]  = round(dt, 2)
        results.append(met)
        assignments[label] = asgn

log.info("=" * 60)
log.info("Paso 5 — Guardando resultados...")

for label, asgn in assignments.items():
    df_out = pd.DataFrame({
        "DIRECTORIO"         : fam["DIRECTORIO"].values,
        "estrato_real"       : estrato_arr,
        "id_establecimiento" : asgn,
    }).merge(
        school_info[["q_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
        on="id_establecimiento", how="left",
    )
    df_out.to_parquet(OUT_DIR / f"matching_da_{label}.parquet", index=False)

comp = pd.DataFrame(results)
comp.to_csv(REP_DIR / "sensibilidad_alpha_comparison.csv", index=False)

log.info("=" * 60)
log.info("RESUMEN — Sensibilidad alpha_0 x gamma")
cols = ["condicion", "alpha_0", "gamma", "n_asignados", "eficiencia_q",
        "equidad_corr", "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comp[cols].to_string(index=False))
log.info("Done.")
