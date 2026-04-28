# -*- coding: utf-8 -*-
"""
07_matching_bm_da.py
====================
Aplica Boston Mechanism (BM) y Deferred Acceptance (DA) sobre los datos
de familias bogotanas expandidas con factor de expansion FEX_C.

Pipeline de entrada
-------------------
  05_simular_distancias.py  ->  familias_distancias.parquet
  05b_expandir_familias.py  ->  familias_expandidas.parquet, distancias_expandidas.parquet
  06_preferencias.py        ->  preferencias_familias.parquet (rankings top-20)
  04_regresion.py           ->  colegios_features_imputed.geojson

Diseno del mecanismo
--------------------
  Capacidad escolar:
    cupos_j = round((matricula_j / matricula_total) * n_familias)
    minimo CAPACITY_MIN cupos por colegio.
    Justificacion: distribuye los cupos disponibles proporcionalmente al
    tamano real de cada colegio. La suma total de cupos iguala n_familias,
    garantizando que todos puedan ser asignados.

  Prioridad escolar:
    Distancia Haversine en km (priority_fn).
    Menor distancia -> mayor prioridad -> replica criterio SED Bogota.

  Choice set:
    Heredado de 06_preferencias.py: familias eligen solo entre colegios
    de su localidad (excepcion: Candelaria puede elegir en 3, 14, 15).

Inputs
------
  data/primary/preferencias_familias.parquet       -- rankings top-20 (537K x 20)
  data/processed/familias_expandidas.parquet       -- familias expandidas con FEX_C
  data/processed/distancias_expandidas.parquet     -- distancias expandidas (537K x 303)
  data/primary/colegios_features_imputed.geojson   -- features de cada colegio

Outputs
-------
  data/results/matching_bm.parquet      -- asignacion BM
  data/results/matching_da.parquet      -- asignacion DA
  reports/matching_comparison.csv       -- tabla comparativa de metricas
  reports/figures/matching/             -- figuras
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matching_utils import (
    boston_mechanism,
    compute_metrics,
    deferred_acceptance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PREF_P  = ROOT / "data" / "primary"   / "preferencias_familias.parquet"
FAM_P   = ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_P  = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
COL_P   = ROOT / "data" / "primary"   / "colegios_features_imputed.geojson"
CAP_P   = ROOT / "data" / "primary"   / "capacidades_colegios.parquet"
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_MIN = 5   # minimo de cupos por colegio

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar datos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 -- Cargando datos...")

pref_df = pd.read_parquet(PREF_P)
fam_df  = pd.read_parquet(FAM_P)
dist_df = pd.read_parquet(DIST_P)

gdf = gpd.read_file(COL_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)

# Alinear por posicion (los tres datasets expandidos tienen el mismo orden)
n = min(len(fam_df), len(dist_df), len(pref_df))
fam_df  = fam_df.iloc[:n].reset_index(drop=True)
dist_df = dist_df.iloc[:n].reset_index(drop=True)
pref_df = pref_df.iloc[:n].reset_index(drop=True)

log.info(f"  Familias : {len(fam_df):,}")
log.info(f"  Colegios : {len(gdf):,}")
log.info(f"  Distancias shape: {dist_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Cargar capacidades escolares (pre-calculadas en 01b_build_capacidades.py)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 -- Cargando capacidades escolares...")

cap_df = pd.read_parquet(CAP_P)
cap_df["id_establecimiento"] = cap_df["id_establecimiento"].astype(str)

# Merge capacidades en gdf
gdf = gdf.merge(cap_df[["id_establecimiento", "capacidad"]], on="id_establecimiento", how="left")
gdf["capacidad"] = gdf["capacidad"].fillna(CAPACITY_MIN).astype(int)

school_info = gdf.set_index("id_establecimiento")[[
    "capacidad", "a_j", "sobre_demanda_j", "nombre_localidad"
]].copy()

log.info(f"  Capacidad media por colegio : {school_info['capacidad'].mean():.0f}")
log.info(f"  Capacidad total             : {school_info['capacidad'].sum():,}")
log.info(f"  Familias a asignar          : {len(fam_df):,}")
log.info(f"  Ratio cupos/familias        : {school_info['capacidad'].sum()/len(fam_df):.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Filtrar colegios validos
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 3 -- Filtrando colegios validos...")

schools_in_prefs = set(pref_df.values.flatten()) - {None, np.nan}
schools_in_info  = set(school_info.index)
valid_schools    = schools_in_prefs & schools_in_info

dist_cols = [c for c in dist_df.columns if c in valid_schools]
dist_df   = dist_df[dist_cols]

log.info(f"  Colegios validos: {len(valid_schools)} | en distancias: {len(dist_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Preparar arrays de trabajo
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 4 -- Preparando arrays de trabajo...")

student_ids = fam_df["DIRECTORIO"].values

estrato_arr = (
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

log.info(f"  Longitud media lista de prefs: {np.mean([len(p) for p in pref_lists]):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Funcion de prioridad (distancia Haversine)
# ─────────────────────────────────────────────────────────────────────────────
def priority_fn(student_idx: int, school_id: str) -> float:
    """Distancia en km. Menor valor = mayor prioridad."""
    col = school_col_idx.get(school_id)
    if col is None:
        return np.inf
    return float(dist_matrix[student_idx, col])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Ejecutar mecanismos
# ─────────────────────────────────────────────────────────────────────────────
school_cap = school_info["capacidad"].to_dict()

log.info("=" * 60)
log.info("Paso 5 -- Ejecutando Boston Mechanism...")
t0 = time.time()
bm_assignment = boston_mechanism(pref_lists, school_cap, priority_fn)
log.info(f"  Boston completado en {time.time()-t0:.1f}s | "
         f"asignados: {sum(a is not None for a in bm_assignment):,}")

log.info("Paso 6 -- Ejecutando Deferred Acceptance (Gale-Shapley)...")
t0 = time.time()
da_assignment = deferred_acceptance(pref_lists, school_cap, priority_fn)
log.info(f"  DA completado en {time.time()-t0:.1f}s | "
         f"asignados: {sum(a is not None for a in da_assignment):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Metricas
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 7 -- Calculando metricas...")

bm_metrics = compute_metrics(
    bm_assignment, pref_lists, school_cap, priority_fn,
    school_info, quality_col="a_j", visual_col="sobre_demanda_j",
    estrato_arr=estrato_arr, label="BM",
)
da_metrics = compute_metrics(
    da_assignment, pref_lists, school_cap, priority_fn,
    school_info, quality_col="a_j", visual_col="sobre_demanda_j",
    estrato_arr=estrato_arr, label="DA",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 8 -- Guardando resultados...")

def build_result_df(assignment, student_ids, estrato_arr, school_info):
    df = pd.DataFrame({
        "DIRECTORIO"         : student_ids,
        "estrato_real"       : estrato_arr,
        "id_establecimiento" : assignment,
    })
    return df.merge(
        school_info[["a_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
        on="id_establecimiento",
        how="left",
    )

bm_df = build_result_df(bm_assignment, student_ids, estrato_arr, school_info)
da_df = build_result_df(da_assignment, student_ids, estrato_arr, school_info)

bm_df.to_parquet(OUT_DIR / "matching_bm.parquet", index=False)
da_df.to_parquet(OUT_DIR / "matching_da.parquet", index=False)
log.info(f"  matching_bm.parquet -- {len(bm_df):,} filas")
log.info(f"  matching_da.parquet -- {len(da_df):,} filas")

comparison = pd.DataFrame([bm_metrics, da_metrics])
comparison.to_csv(REP_DIR / "matching_comparison.csv", index=False)
log.info("  matching_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Figuras
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 9 -- Generando figuras...")

estratos = list(range(1, 7))
w        = 0.35

# Figura 1: calidad asignada por estrato + metricas resumen
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

q_bm = [bm_metrics.get(f"q_estrato_{s}", np.nan) for s in estratos]
q_da = [da_metrics.get(f"q_estrato_{s}", np.nan) for s in estratos]
x    = np.arange(len(estratos))

axes[0].bar(x - w/2, q_bm, w, label="Boston (BM)", color="#2196F3", alpha=0.85)
axes[0].bar(x + w/2, q_da, w, label="DA (Gale-Shapley)", color="#FF9800", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Estrato {s}" for s in estratos], rotation=20)
axes[0].set_ylabel("Atractivo medio del colegio asignado (a_j)")
axes[0].set_title("Atractivo asignado por estrato")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

metric_labels = ["Eficiencia (a_j)", "|corr(estrato,a_j)|\nEquidad", "|corr(estrato,SD)|\nSesgo visual"]
bm_vals = [bm_metrics["eficiencia_q"], abs(bm_metrics["equidad_corr"]), abs(bm_metrics["sesgo_visual"])]
da_vals = [da_metrics["eficiencia_q"], abs(da_metrics["equidad_corr"]), abs(da_metrics["sesgo_visual"])]

x2 = np.arange(len(metric_labels))
axes[1].bar(x2 - w/2, bm_vals, w, label="Boston (BM)", color="#2196F3", alpha=0.85)
axes[1].bar(x2 + w/2, da_vals, w, label="DA", color="#FF9800", alpha=0.85)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(metric_labels)
axes[1].set_title("Comparacion de metricas resumen")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Boston Mechanism vs Deferred Acceptance -- Bogota (poblacion expandida)", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "bm_vs_da_metricas.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figura 2: distribucion de atractivo asignado
fig, ax = plt.subplots(figsize=(9, 5))
a_bm_vals = bm_df["a_j"].dropna().values
a_da_vals = da_df["a_j"].dropna().values
ax.hist(a_bm_vals, bins=40, alpha=0.6, label=f"Boston (BM)  mu={a_bm_vals.mean():.3f}", color="#2196F3")
ax.hist(a_da_vals, bins=40, alpha=0.6, label=f"DA           mu={a_da_vals.mean():.3f}", color="#FF9800")
ax.axvline(a_bm_vals.mean(), color="#1565C0", linestyle="--", linewidth=1.5)
ax.axvline(a_da_vals.mean(), color="#E65100", linestyle="--", linewidth=1.5)
ax.set_xlabel("Atractivo del colegio asignado (a_j = log_sobredemanda predicho)")
ax.set_ylabel("Familias")
ax.set_title("Distribucion de atractivo asignado -- BM vs DA")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "distribucion_atractivo_asignado.png", dpi=150)
plt.close(fig)

# Figura 3: sesgo visual (sobre_demanda_j) por estrato
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (mech_df, mech_name, color) in zip(
    axes,
    [(bm_df, "Boston BM", "#4CAF50"), (da_df, "DA", "#9C27B0")]
):
    means = (
        mech_df.dropna(subset=["sobre_demanda_j"])
        .groupby("estrato_real")["sobre_demanda_j"]
        .mean()
    )
    ax.bar(means.index, means.values, color=color, alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="SD=1 (sin sobredemanda)")
    ax.set_xlabel("Estrato del hogar")
    ax.set_ylabel("Sobredemanda media (demanda / matricula)")
    ax.set_title(f"Sesgo visual en asignacion -- {mech_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Estratos altos reciben colegios mas sobre-demandados?", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "sesgo_visual_por_estrato.png", dpi=150, bbox_inches="tight")
plt.close(fig)

log.info("  bm_vs_da_metricas.png")
log.info("  distribucion_atractivo_asignado.png")
log.info("  sesgo_visual_por_estrato.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN COMPARATIVO -- Poblacion expandida")
log.info("=" * 60)
cols_show = ["condicion", "n_asignados", "eficiencia_q", "equidad_corr",
             "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comparison[cols_show].to_string(index=False))
log.info("Done.")
