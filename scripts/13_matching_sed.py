"""
13_matching_sed.py
==================
SED Bogotá: DA con prioridad lexicográfica de la Resolución 1587 de 2025.
Compara contra DA puro distancia (escenario "sin priorización SED").

Modelo de prioridad
-------------------
Resolución 1587 de 2025 (Art. 19-20): asignación por orden lexicográfico entre
categorías priorizadas (numerales 2.1 a 2.8) antes que población general (2.9),
con distancia residencial como criterio de desempate dentro de cada categoría.

EM2021 sólo permite identificar de forma limpia las categorías 2.8 (SISBEN A-B,
aproximada por estratos 1-2) y 2.9 (general, estratos 3-6). Las otras 7 categorías
(discapacidad, preescolar, étnicos, víctimas, hermanos, gestantes, SRPA) no son
identificables sin inventar datos. Modelamos las dos categorías limpias.

Codificación lexicográfica
--------------------------
priority_fn(i, j) = cat(i) * 1e6 + dist(i, j)

donde cat = 0 para SISBEN_AB y cat = 1 para general. El multiplicador 1e6 garantiza
que el orden de categorías domine sobre cualquier distancia plausible (max ~30 km).

Inputs
------
  data/processed/familias_expandidas.parquet
  data/processed/distancias_expandidas.parquet
  data/primary/colegios_capacidad.parquet
  data/primary/preferencias_familias.parquet

Outputs
-------
  data/results/matching_sed_lex.parquet     — asignación con prioridad SED
  data/results/matching_sed_dist.parquet    — asignación con prioridad por distancia (control)
  reports/matching_sed_comparison.csv
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from matching_utils import compute_metrics, deferred_acceptance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FAM_P  = ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_P = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
COL_P  = ROOT / "data" / "primary"   / "colegios_capacidad.parquet"
PREF_P = ROOT / "data" / "primary"   / "preferencias_familias.parquet"
OUT_DIR = ROOT / "data" / "results"
REP_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_OFFSET = 1_000_000

log.info("Cargando datos...")
fam  = pd.read_parquet(FAM_P)
col  = pd.read_parquet(COL_P)
pref = pd.read_parquet(PREF_P)
dist = pd.read_parquet(DIST_P)

log.info(f"  Familias: {len(fam):,}")
log.info(f"  Colegios: {len(col)}")
log.info(f"  Capacidad total: {col['capacidad'].sum():,}")
log.info(f"  Ratio cupos/familia: {col['capacidad'].sum() / len(fam):.3f}")

col["id_establecimiento"] = col["id_establecimiento"].astype(str)
school_info = col.set_index("id_establecimiento")
school_cap  = school_info["capacidad"].to_dict()

# Alinear familias y preferencias
fam  = fam.reset_index(drop=True)
pref = pref.reset_index(drop=True)   # alineacion posicional con fam
dist = dist.reset_index(drop=True)   # alineacion posicional con fam

# filtrar familias sin estrato (drop posicional simultaneo)
mask = fam["estrato_real"].notna().values
n_drop = (~mask).sum()
if n_drop > 0:
    log.warning(f"  Drop {n_drop:,} familias sin estrato_real")
    fam  = fam.loc[mask].reset_index(drop=True)
    pref = pref.loc[mask].reset_index(drop=True)
    dist = dist.loc[mask].reset_index(drop=True)

estrato_arr = fam["estrato_real"].astype(int).values
categoria   = (estrato_arr > 2).astype(int)  # 0 = SISBEN_AB (estrato 1-2), 1 = general

log.info(f"  SISBEN_AB (estrato 1-2): {(categoria == 0).sum():,} ({(categoria == 0).mean()*100:.1f}%)")
log.info(f"  General (estrato 3-6):   {(categoria == 1).sum():,} ({(categoria == 1).mean()*100:.1f}%)")

valid_schools = set(school_info.index)
dist_cols = [c for c in dist.columns if c in valid_schools]
dist_arr  = dist[dist_cols].values.astype(np.float32)
school_col_idx = {sid: k for k, sid in enumerate(dist_cols)}

pref_cols = [c for c in pref.columns if c.startswith("pref_")]
pref_lists = [
    [s for s in row if isinstance(s, str) and s in valid_schools]
    for row in pref[pref_cols].astype(str).values
]
log.info(f"  Longitud media lista de prefs: {np.mean([len(p) for p in pref_lists]):.1f}")

def make_priority_lex(categoria_arr, dist_arr, school_col_idx):
    def priority_fn(i, sid):
        col_idx = school_col_idx.get(sid)
        if col_idx is None:
            return np.inf
        return float(categoria_arr[i]) * CAT_OFFSET + float(dist_arr[i, col_idx])
    return priority_fn

def make_priority_dist(dist_arr, school_col_idx):
    def priority_fn(i, sid):
        col_idx = school_col_idx.get(sid)
        if col_idx is None:
            return np.inf
        return float(dist_arr[i, col_idx])
    return priority_fn

results = {}
assignments = {}

log.info("=" * 60)
log.info("Corriendo DA con prioridad SED lexicográfica...")
t0 = time.time()
pfn_lex = make_priority_lex(categoria, dist_arr, school_col_idx)
asgn_lex = deferred_acceptance(pref_lists, school_cap, pfn_lex)
log.info(f"  Completado en {time.time()-t0:.1f}s")

met_lex = compute_metrics(
    asgn_lex, pref_lists, school_cap, pfn_lex,
    school_info, quality_col="q_j", visual_col="sobre_demanda_j",
    estrato_arr=estrato_arr, label="SED-lex",
)
results["SED-lex"] = met_lex
assignments["sed_lex"] = asgn_lex

log.info("=" * 60)
log.info("Corriendo DA con prioridad por distancia (control)...")
t0 = time.time()
pfn_dist = make_priority_dist(dist_arr, school_col_idx)
asgn_dist = deferred_acceptance(pref_lists, school_cap, pfn_dist)
log.info(f"  Completado en {time.time()-t0:.1f}s")

met_dist = compute_metrics(
    asgn_dist, pref_lists, school_cap, pfn_dist,
    school_info, quality_col="q_j", visual_col="sobre_demanda_j",
    estrato_arr=estrato_arr, label="DA-dist",
)
results["DA-dist"] = met_dist
assignments["sed_dist"] = asgn_dist

log.info("=" * 60)
log.info("Guardando resultados...")
for label, asgn in assignments.items():
    df = pd.DataFrame({
        "DIRECTORIO": fam["DIRECTORIO"].values,
        "estrato_real": estrato_arr,
        "categoria": categoria,
        "id_establecimiento": asgn,
    }).merge(
        school_info[["q_j", "sobre_demanda_j", "a_j", "nombre_localidad"]].reset_index(),
        on="id_establecimiento", how="left",
    )
    df.to_parquet(OUT_DIR / f"matching_{label}.parquet", index=False)
    log.info(f"  matching_{label}.parquet")

comp = pd.DataFrame(list(results.values()))
comp.to_csv(REP_DIR / "matching_sed_comparison.csv", index=False)
log.info("  matching_sed_comparison.csv")

log.info("=" * 60)
log.info("RESUMEN")
cols = ["condicion", "n_asignados", "eficiencia_q", "equidad_corr",
        "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comp[cols].to_string(index=False))
log.info("Done.")
