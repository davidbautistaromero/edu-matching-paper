"""
07_matching_bm_da.py
====================
Aplica Boston Mechanism (BM) y Deferred Acceptance (DA) sobre los datos REALES
de familias bogotanas con hijos en colegio oficial.

Pipeline de entrada (scripts previos requeridos)
-------------------------------------------------
  05_simular_distancias.py  →  familias_distancias.parquet   (matriz Haversine)
  06_preferencias.py        →  preferencias_familias.parquet  (rankings top-20)
  04_regresion.py           →  colegios_features_imputed.geojson (q_j, matricula_total)

Inputs
------
  data/primary/preferencias_familias.parquet       — rankings top-20 por familia (13,568 × 20)
  data/processed/familias_ubicadas.parquet         — estrato y localidad de cada familia
  data/processed/familias_distancias.parquet       — dist. Haversine en km (13,568 × 303)
  data/primary/colegios_features_imputed.geojson   — features de cada colegio

Diseño del mecanismo
--------------------
  Capacidad escolar:
    matricula_total / 13 redondeado, mínimo 5.
    Justificación: la matrícula total refleja todos los grados activos (~13 años de
    escolaridad). Dividir por 13 aproxima el cupo anual por colegio. Con total ≈ 45k
    cupos y 13.5k familias hay holgura suficiente para que la competencia sea real
    dentro de cada localidad sin dejar familias sin opción.

  Prioridad escolar:
    Distancia Haversine en km (función priority_fn).
    Justificación: el criterio de asignación del SED Bogotá es proximidad geográfica.
    Menor distancia → mayor prioridad → replica el sistema vigente.

  Choice set:
    Heredado de 06_preferencias.py: familias eligen sólo entre colegios de su
    localidad (excepción: La Candelaria puede elegir en localidades 3, 14, 15).

Outputs
-------
  data/results/matching_bm.parquet      — asignación BM: DIRECTORIO → colegio + q_j + v_j
  data/results/matching_da.parquet      — asignación DA: DIRECTORIO → colegio + q_j + v_j
  reports/matching_comparison.csv       — tabla comparativa de métricas
  reports/figures/matching/             — figuras de distribución y equidad
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unicodedata

# Importar algoritmos y métricas del módulo compartido
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
FAM_P   = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
DIST_P  = ROOT / "data" / "processed" / "familias_distancias.parquet"
COL_P   = ROOT / "data" / "primary"   / "colegios_features_imputed.geojson"
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Capacidad = round(matricula_total / COHORT_DIVISOR), mínimo CAPACITY_MIN
COHORT_DIVISOR = 13   # años de escolaridad (kinder a grado 11)
CAPACITY_MIN   = 5    # garantía de al menos 5 cupos por sede (colegios muy pequeños)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar datos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 — Cargando datos...")

pref_df = pd.read_parquet(PREF_P)   # (13,568 × 20): una fila por familia, columnas pref_1..pref_20
fam_df  = pd.read_parquet(FAM_P)    # (13,568 × 10): DIRECTORIO, estrato_real, COD_LOCALIDAD, lat, lon
dist_df = pd.read_parquet(DIST_P)   # (13,568 × 306): distancias Haversine en km

gdf = gpd.read_file(COL_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)   # asegurar tipo string

log.info(f"  Familias : {len(fam_df):,}")
log.info(f"  Colegios : {len(gdf):,}")
log.info(f"  Distancias shape: {dist_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Calcular capacidades escolares
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 — Calculando capacidades escolares...")

# matricula_total = alumnos actualmente matriculados en todos los grados.
# Dividido por 13 ≈ cupos anuales disponibles por colegio.
gdf["capacidad"] = (
    gdf["matricula_total"] / COHORT_DIVISOR
).round().clip(lower=CAPACITY_MIN).astype(int)

# school_info: DataFrame indexado por id_establecimiento con los atributos
# que necesitan las métricas. Se construye una vez y se pasa a compute_metrics.
school_info = gdf.set_index("id_establecimiento")[[
    "capacidad", "q_j", "sobre_demanda_j", "nombre_localidad"
]].copy()

log.info(f"  Capacidad media por colegio : {school_info['capacidad'].mean():.0f}")
log.info(f"  Capacidad total             : {school_info['capacidad'].sum():,}")
log.info(f"  (Familias a asignar: {len(fam_df):,} — ratio cupos/familias: "
         f"{school_info['capacidad'].sum()/len(fam_df):.1f}x)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Alinear índices entre los tres DataFrames
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 3 — Alineando índices familia ↔ distancias ↔ preferencias...")

# Conservar sólo familias presentes en los tres archivos.
# (Algunas pueden haberse perdido en alguno de los pasos de la pipeline.)
common_fam = (
    fam_df["DIRECTORIO"].isin(dist_df.index)
    & fam_df["DIRECTORIO"].isin(pref_df.index)
)
fam_df = fam_df[common_fam].reset_index(drop=True)

# Reindexar distancias y preferencias para que el orden de filas coincida con fam_df
dist_df = dist_df.loc[fam_df["DIRECTORIO"]]
pref_df = pref_df.loc[fam_df["DIRECTORIO"]]

# Filtrar colegios: sólo los que aparecen tanto en las listas de preferencias
# como en el GeoJSON con features. Algunos colegios en el directorio SED no
# tienen embeddings NMF (sin imágenes GSV) y fueron excluidos de 06_preferencias.py.
schools_in_prefs = set(pref_df.values.flatten())
schools_in_info  = set(school_info.index)
valid_schools    = schools_in_prefs & schools_in_info

# Filtrar columnas de distancias a los colegios válidos
dist_cols = [c for c in dist_df.columns if c in valid_schools]
dist_df   = dist_df[dist_cols]

log.info(f"  Familias alineadas: {len(fam_df):,} | Colegios en modelo: {len(dist_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Preparar arrays de trabajo
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 4 — Preparando arrays de trabajo...")

# IDs de familias (para reconstruir el DataFrame de resultados)
student_ids = fam_df["DIRECTORIO"].values   # (N,) array de identificadores

# Estrato real del hogar (1–6). Valores inválidos → 3 (estrato medio por defecto)
estrato_arr = (
    pd.to_numeric(fam_df["estrato_real"], errors="coerce")
    .fillna(3)
    .clip(1, 6)
    .values
)

# Matriz de distancias como NumPy array float32 para operaciones vectorizadas.
# dist_matrix[i, k] = distancia en km de la familia i al colegio en columna k.
dist_matrix    = dist_df.values.astype(np.float32)   # (N, M)
school_col_idx = {sid: k for k, sid in enumerate(dist_df.columns)}   # id → índice columna

# Listas de preferencias: convertir el DataFrame de rankings en lista de listas.
# Filtrar elementos que no sean colegios válidos (NaN, colegios sin features).
pref_cols = [f"pref_{k}" for k in range(1, 21)]
pref_lists = [
    [s for s in row.values if isinstance(s, str) and s in valid_schools]
    for _, row in pref_df[pref_cols].iterrows()
]

log.info(f"  Longitud media lista de prefs: {np.mean([len(p) for p in pref_lists]):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Definir función de prioridad (distancia)
# ─────────────────────────────────────────────────────────────────────────────
# La función de prioridad encapsula la lógica de "quién tiene precedencia".
# Devuelve la distancia en km; menor distancia = mayor prioridad del estudiante
# sobre el colegio. Esto replica el criterio del SED Bogotá.
#
# Se pasa como argumento a boston_mechanism y deferred_acceptance para que
# ambos algoritmos usen exactamente la misma lógica sin código duplicado.

def priority_fn(student_idx: int, school_id: str) -> float:
    """
    Distancia Haversine en km del estudiante al colegio.
    Menor valor = mayor prioridad (el colegio prefiere al estudiante más cercano).
    Devuelve inf si el colegio no tiene columna en dist_matrix (caso de respaldo).
    """
    col = school_col_idx.get(school_id)
    if col is None:
        return np.inf   # colegio sin datos de distancia: no puede ser priorizado
    return float(dist_matrix[student_idx, col])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Ejecutar mecanismos
# ─────────────────────────────────────────────────────────────────────────────
school_cap = school_info["capacidad"].to_dict()   # {id_establecimiento: cupos}

log.info("=" * 60)
log.info("Paso 5 — Ejecutando Boston Mechanism...")
t0 = time.time()
bm_assignment = boston_mechanism(pref_lists, school_cap, priority_fn)
log.info(f"  Boston completado en {time.time()-t0:.1f}s | "
         f"asignados: {sum(a is not None for a in bm_assignment):,}")

log.info("Paso 6 — Ejecutando Deferred Acceptance (Gale-Shapley)...")
t0 = time.time()
da_assignment = deferred_acceptance(pref_lists, school_cap, priority_fn)
log.info(f"  DA completado en {time.time()-t0:.1f}s | "
         f"asignados: {sum(a is not None for a in da_assignment):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Calcular métricas
# ─────────────────────────────────────────────────────────────────────────────
# quality_col  = "q_j"            : puntaje ICFES promedio del colegio
# visual_col   = "sobre_demanda_j": proxy de señal visual (demanda/matrícula)
#   sobre_demanda_j > 1 indica que el colegio recibe más solicitudes de cupo
#   de las que su matrícula justifica. Esto puede reflejar atractivo visual
#   más allá de la calidad académica (hipótesis central del paper).
log.info("=" * 60)
log.info("Paso 7 — Calculando métricas...")

bm_metrics = compute_metrics(
    bm_assignment, pref_lists, school_cap, priority_fn,
    school_info, quality_col="q_j", visual_col="sobre_demanda_j",
    estrato_arr=estrato_arr, label="BM",
)
da_metrics = compute_metrics(
    da_assignment, pref_lists, school_cap, priority_fn,
    school_info, quality_col="q_j", visual_col="sobre_demanda_j",
    estrato_arr=estrato_arr, label="DA",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 8 — Guardando resultados...")

def build_result_df(
    assignment: list,
    student_ids: np.ndarray,
    estrato_arr: np.ndarray,
    school_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el DataFrame de resultados enriquecido con atributos del colegio asignado.
    Merge por left para que las familias no asignadas queden con NaN en los atributos.
    """
    df = pd.DataFrame({
        "DIRECTORIO"         : student_ids,
        "estrato_real"       : estrato_arr,
        "id_establecimiento" : assignment,
    })
    return df.merge(
        school_info[["q_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
        on="id_establecimiento",
        how="left",
    )

bm_df = build_result_df(bm_assignment, student_ids, estrato_arr, school_info)
da_df = build_result_df(da_assignment, student_ids, estrato_arr, school_info)

bm_df.to_parquet(OUT_DIR / "matching_bm.parquet", index=False)
da_df.to_parquet(OUT_DIR / "matching_da.parquet", index=False)
log.info(f"  matching_bm.parquet — {len(bm_df):,} filas")
log.info(f"  matching_da.parquet — {len(da_df):,} filas")

comparison = pd.DataFrame([bm_metrics, da_metrics])
comparison.to_csv(REP_DIR / "matching_comparison.csv", index=False)
log.info("  matching_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Figuras
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 9 — Generando figuras...")

estratos = list(range(1, 7))
w        = 0.35

# ── Figura 1: métricas resumen + calidad por estrato ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: calidad media asignada por estrato
q_bm = [bm_metrics.get(f"q_estrato_{s}", np.nan) for s in estratos]
q_da = [da_metrics.get(f"q_estrato_{s}", np.nan) for s in estratos]
x    = np.arange(len(estratos))

axes[0].bar(x - w/2, q_bm, w, label="Boston (BM)", color="#2196F3", alpha=0.85)
axes[0].bar(x + w/2, q_da, w, label="DA (Gale-Shapley)", color="#FF9800", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Estrato {s}" for s in estratos], rotation=20)
axes[0].set_ylabel("Calidad ICFES media (q_j)")
axes[0].set_title("Calidad asignada por estrato — datos reales")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Panel derecho: resumen de métricas clave
metric_labels = ["Eficiencia (q)", "|corr(estrato,q)|\nEquidad", "|corr(estrato,SD)|\nSesgo visual"]
bm_vals = [bm_metrics["eficiencia_q"], abs(bm_metrics["equidad_corr"]), abs(bm_metrics["sesgo_visual"])]
da_vals = [da_metrics["eficiencia_q"], abs(da_metrics["equidad_corr"]), abs(da_metrics["sesgo_visual"])]

x2 = np.arange(len(metric_labels))
axes[1].bar(x2 - w/2, bm_vals, w, label="Boston (BM)", color="#2196F3", alpha=0.85)
axes[1].bar(x2 + w/2, da_vals, w, label="DA", color="#FF9800", alpha=0.85)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(metric_labels)
axes[1].set_title("Comparación de métricas resumen")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Boston Mechanism vs Deferred Acceptance — Bogotá (datos reales)", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "bm_vs_da_metricas.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figura 2: distribución de calidad asignada ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
q_bm_vals = bm_df["q_j"].dropna().values
q_da_vals = da_df["q_j"].dropna().values
ax.hist(q_bm_vals, bins=40, alpha=0.6, label=f"Boston (BM)  μ={q_bm_vals.mean():.2f}", color="#2196F3")
ax.hist(q_da_vals, bins=40, alpha=0.6, label=f"DA          μ={q_da_vals.mean():.2f}", color="#FF9800")
ax.axvline(q_bm_vals.mean(), color="#1565C0", linestyle="--", linewidth=1.5)
ax.axvline(q_da_vals.mean(), color="#E65100", linestyle="--", linewidth=1.5)
ax.set_xlabel("Calidad ICFES del colegio asignado (q_j, puntos Saber 11)")
ax.set_ylabel("Familias")
ax.set_title("Distribución de calidad asignada — BM vs DA (datos reales)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "distribucion_calidad_asignada.png", dpi=150)
plt.close(fig)

# ── Figura 3: sesgo visual (sobre_demanda_j) por estrato ──────────────────
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
    # Línea de referencia: ratio = 1.0 significa demanda = matrícula (sin sobredemanda)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               label="SD=1 (sin sobredemanda)")
    ax.set_xlabel("Estrato del hogar")
    ax.set_ylabel("Sobredemanda media (demanda / matrícula)")
    ax.set_title(f"Sesgo visual en asignación — {mech_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("¿Estratos altos reciben colegios más sobre-demandados?", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "sesgo_visual_por_estrato.png", dpi=150, bbox_inches="tight")
plt.close(fig)

log.info("  bm_vs_da_metricas.png")
log.info("  distribucion_calidad_asignada.png")
log.info("  sesgo_visual_por_estrato.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN COMPARATIVO — Datos reales")
log.info("=" * 60)
cols_show = ["condicion", "n_asignados", "eficiencia_q", "equidad_corr",
             "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comparison[cols_show].to_string(index=False))
log.info("Done.")
