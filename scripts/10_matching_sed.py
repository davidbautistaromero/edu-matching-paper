"""
10_matching_sed.py
==================
Aplica el mecanismo de la Secretaría de Educación del Distrito (SED Bogotá)
sobre los datos reales de familias bogotanas con hijos en colegio oficial.

Mecanismo SED estilizado
------------------------
Resolución 1587 de 2025: la SED asigna cupos mediante un índice de priorización
que combina variables socioeconómicas y de ubicación del estudiante, sobre
disponibilidad de cupos. Operacionalizamos esto como Deferred Acceptance con
prioridad compuesta:

    priority_sed(i, j) = w_estrato * estrato_norm(i) + (1 - w_estrato) * dist_norm(i, j)

donde:
  estrato_norm(i)  = (estrato_i - 1) / 5            ∈ [0, 1]
  dist_norm(i, j)  = min(d_ij, D_max) / D_max       ∈ [0, 1], D_max = p95(d)

Caso base: w_estrato = 0.3 (distancia dominante, consistente con Gallego &
Hernando 2009 sobre la primacía geográfica en school choice latinoamericano).

Sensibilidad: w_estrato ∈ {0.0, 0.3, 0.5, 0.7, 1.0}. Los extremos sirven como
validación: w=0.0 reproduce DA puro distancia (sanity check vs 07_matching_bm_da.py),
w=1.0 representa priorización pura por estrato.
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from matching_utils import compute_metrics, deferred_acceptance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
PREF_P  = ROOT / "data" / "primary"   / "preferencias_familias.parquet"
FAM_P   = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
DIST_P  = ROOT / "data" / "processed" / "familias_distancias.parquet"
COL_P   = ROOT / "data" / "primary"   / "colegios_features_imputed.geojson"
OUT_DIR = ROOT / "data" / "results"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)

COHORT_DIVISOR = 13
CAPACITY_MIN   = 5

W_ESTRATO_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
W_BASELINE       = 0.3

log.info("=" * 60)
log.info("Paso 1 — Cargando datos...")

pref_df = pd.read_parquet(PREF_P)
fam_df  = pd.read_parquet(FAM_P)
dist_df = pd.read_parquet(DIST_P)

gdf = gpd.read_file(COL_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)

gdf["capacidad"] = (
    gdf["matricula_total"] / COHORT_DIVISOR
).round().clip(lower=CAPACITY_MIN).astype(int)

school_info = gdf.set_index("id_establecimiento")[[
    "capacidad", "q_j", "sobre_demanda_j", "nombre_localidad"
]].copy()

log.info(f"  Familias : {len(fam_df):,}")
log.info(f"  Colegios : {len(school_info):,}")

log.info("Paso 2 — Alineando índices...")
common_fam = (
    fam_df["DIRECTORIO"].isin(dist_df.index)
    & fam_df["DIRECTORIO"].isin(pref_df.index)
)
fam_df  = fam_df[common_fam].reset_index(drop=True)
dist_df = dist_df.loc[fam_df["DIRECTORIO"]]
pref_df = pref_df.loc[fam_df["DIRECTORIO"]]

schools_in_prefs = set(pref_df.values.flatten())
schools_in_info  = set(school_info.index)
valid_schools    = schools_in_prefs & schools_in_info

dist_cols = [c for c in dist_df.columns if c in valid_schools]
dist_df   = dist_df[dist_cols]

student_ids    = fam_df["DIRECTORIO"].values
estrato_arr    = (
    pd.to_numeric(fam_df["estrato_real"], errors="coerce")
    .fillna(3).clip(1, 6).values
)
dist_matrix    = dist_df.values.astype(np.float32)
school_col_idx = {sid: k for k, sid in enumerate(dist_df.columns)}

pref_cols = [f"pref_{k}" for k in range(1, 21)]
pref_lists = [
    [s for s in row.values if isinstance(s, str) and s in valid_schools]
    for _, row in pref_df[pref_cols].iterrows()
]

D_MAX = float(np.percentile(dist_matrix[dist_matrix < np.inf], 95))
estrato_norm = (estrato_arr - 1) / 5.0

log.info(f"  D_max (p95): {D_MAX:.2f} km")
log.info(f"  Caso base : w_estrato={W_BASELINE} | w_dist={1-W_BASELINE}")

school_cap = school_info["capacidad"].to_dict()

def make_priority_fn(w_estrato: float):
    w_dist = 1.0 - w_estrato
    def priority_fn(student_idx: int, school_id: str) -> float:
        col = school_col_idx.get(school_id)
        if col is None:
            return np.inf
        d = float(dist_matrix[student_idx, col])
        d_norm = min(d, D_MAX) / D_MAX
        return w_estrato * estrato_norm[student_idx] + w_dist * d_norm
    return priority_fn

results = []
assignments = {}

log.info("=" * 60)
log.info("Paso 3 — Ejecutando SED para diferentes pesos de estrato...")

for w in W_ESTRATO_VALUES:
    label = f"SED-w{int(w*100):03d}"
    log.info(f"  {label}: w_estrato={w} | w_dist={1-w}")
    pfn = make_priority_fn(w)
    t0 = time.time()
    asgn = deferred_acceptance(pref_lists, school_cap, pfn)
    log.info(f"    DA completado en {time.time()-t0:.1f}s")

    met = compute_metrics(
        asgn, pref_lists, school_cap, pfn,
        school_info, quality_col="q_j", visual_col="sobre_demanda_j",
        estrato_arr=estrato_arr, label=label,
    )
    met["w_estrato"] = w
    met["w_dist"]    = 1 - w
    results.append(met)
    assignments[label] = asgn

log.info("=" * 60)
log.info("Paso 4 — Guardando resultados...")

for label, asgn in assignments.items():
    df = pd.DataFrame({
        "DIRECTORIO"         : student_ids,
        "estrato_real"       : estrato_arr,
        "id_establecimiento" : asgn,
    }).merge(
        school_info[["q_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
        on="id_establecimiento", how="left",
    )
    fname = f"matching_{label.lower().replace('-', '_')}.parquet"
    df.to_parquet(OUT_DIR / fname, index=False)
    log.info(f"  {fname}")

comp = pd.DataFrame(results)
comp.to_csv(REP_DIR / "matching_sed_comparison.csv", index=False)
log.info("  matching_sed_comparison.csv")

log.info("=" * 60)
log.info("RESUMEN — Mecanismo SED")
cols_show = ["condicion", "w_estrato", "n_asignados", "eficiencia_q",
             "equidad_corr", "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comp[cols_show].to_string(index=False))
log.info("Done.")
