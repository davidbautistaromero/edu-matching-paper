"""
07_boston_mechanism.py
======================
Aplica el Boston Mechanism (BM) sobre la población expandida de Bogotá.

Algoritmo
---------
El BM corre en rondas. En cada ronda k:
  1. Cada familia no asignada propone al k-ésimo colegio de su lista
     (saltando colegios sin cupos disponibles).
  2. Cada colegio acepta DEFINITIVAMENTE a los mejores candidatos de la
     ronda hasta su cupo (criterio: menor distancia Haversine = mayor prioridad).
     Los rechazados pasan a la ronda k+1.
  3. El proceso termina cuando no quedan propuestas pendientes.

Propiedad clave: la aceptación es irrevocable. Un colegio no puede rechazar
en rondas posteriores a quien aceptó antes, aunque llegue alguien con mayor
prioridad. Por eso el BM NO es strategy-proof.

Inputs
------
  data/primary/preferencias_familias.parquet     — rankings top-20 (537K × 20)
  data/processed/familias_expandidas.parquet     — familias con estrato_real
  data/processed/distancias_expandidas.parquet   — distancias Haversine (537K × 416)
  data/primary/colegios_capacidad.parquet        — capacidad y atributos por colegio

Outputs
-------
  data/results/matching_bm.parquet               — asignación final
  reports/matching_bm_summary.csv                — métricas por estrato
  reports/figures/matching/bm_calidad_estrato.png
  reports/figures/matching/bm_distribucion_aj.png
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matching_utils import boston_mechanism, compute_metrics

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
CAP_P   = ROOT / "data" / "primary"   / "colegios_capacidad.parquet"
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_MIN = 5   # cupos mínimos garantizados por colegio

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar datos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 — Cargando datos...")

pref_df = pd.read_parquet(PREF_P)
fam_df  = pd.read_parquet(FAM_P)
dist_df = pd.read_parquet(DIST_P)
cap_df  = pd.read_parquet(CAP_P)

cap_df["id_establecimiento"] = cap_df["id_establecimiento"].astype(str)
dist_df.columns = dist_df.columns.astype(str)

# Los tres datasets expandidos están alineados por posición (mismo orden, mismo índice)
n = min(len(fam_df), len(pref_df), len(dist_df))
fam_df  = fam_df.iloc[:n].reset_index(drop=True)
pref_df = pref_df.iloc[:n].reset_index(drop=True)
dist_df = dist_df.iloc[:n].reset_index(drop=True)

log.info(f"  Familias : {n:,}")
log.info(f"  Colegios en distancias: {dist_df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Construir school_info
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 — Construyendo school_info desde colegios_capacidad...")

school_info = cap_df.set_index("id_establecimiento")[[
    "capacidad", "a_j", "q_j", "sobre_demanda_j", "nombre_localidad"
]].copy()
school_info["capacidad"] = school_info["capacidad"].clip(lower=CAPACITY_MIN).astype(int)

school_cap = school_info["capacidad"].to_dict()

log.info(f"  Colegios con capacidad: {len(school_info):,}")
log.info(f"  Capacidad total       : {school_info['capacidad'].sum():,}")
log.info(f"  Ratio cupos/familias  : {school_info['capacidad'].sum() / n:.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Filtrar colegios válidos
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 3 — Filtrando colegios válidos...")

pref_cols = [f"pref_{k}" for k in range(1, 21)]
schools_in_prefs = {s for col in pref_cols for s in pref_df[col].dropna().astype(str)}
schools_in_dist  = set(dist_df.columns.astype(str))
schools_in_info  = set(school_info.index.astype(str))
valid_schools    = schools_in_prefs & schools_in_dist & schools_in_info

dist_df = dist_df[[c for c in dist_df.columns if c in valid_schools]]
dist_matrix    = dist_df.values.astype(np.float32)   # (N, M)
school_col_idx = {sid: k for k, sid in enumerate(dist_df.columns)}

log.info(f"  Colegios válidos: {len(valid_schools)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Preparar arrays de trabajo
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 4 — Preparando arrays...")

estrato_arr = (
    pd.to_numeric(fam_df["estrato_real"], errors="coerce")
    .fillna(3).clip(1, 6).astype(int).values
)

pref_lists = [
    [s for s in row.values if isinstance(s, str) and s in valid_schools]
    for _, row in pref_df[pref_cols].iterrows()
]

log.info(f"  Longitud media lista de prefs: {np.mean([len(p) for p in pref_lists]):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Función de prioridad — distancia Haversine
# ─────────────────────────────────────────────────────────────────────────────
def priority_fn(student_idx: int, school_id: str) -> float:
    """Distancia en km. Menor = mayor prioridad (replica criterio SED Bogotá)."""
    col = school_col_idx.get(school_id)
    if col is None:
        return np.inf
    return float(dist_matrix[student_idx, col])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Ejecutar Boston Mechanism
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 5 — Ejecutando Boston Mechanism...")

t0 = time.time()
bm_assignment = boston_mechanism(pref_lists, school_cap, priority_fn)
elapsed = time.time() - t0

n_assigned = sum(a is not None for a in bm_assignment)
log.info(f"  Completado en {elapsed:.1f}s")
log.info(f"  Asignados   : {n_assigned:,} / {n:,} ({100*n_assigned/n:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Métricas
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 6 — Calculando métricas...")

bm_metrics = compute_metrics(
    assignment   = bm_assignment,
    pref_lists   = pref_lists,
    school_cap   = school_cap,
    priority_fn  = priority_fn,
    school_info  = school_info,
    quality_col  = "a_j",
    visual_col   = "sobre_demanda_j",
    estrato_arr  = estrato_arr,
    label        = "BM",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 7 — Guardando resultados...")

bm_df = pd.DataFrame({
    "DIRECTORIO"         : fam_df["DIRECTORIO"].values,
    "estrato_real"       : estrato_arr,
    "id_establecimiento" : bm_assignment,
}).merge(
    school_info[["a_j", "q_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
    on="id_establecimiento",
    how="left",
)

bm_df.to_parquet(OUT_DIR / "matching_bm.parquet", index=False)
log.info(f"  matching_bm.parquet — {len(bm_df):,} filas")

# Métricas por estrato
summary_rows = [{"estrato": "TOTAL", **{k: v for k, v in bm_metrics.items()}}]
for s in range(1, 7):
    mask = estrato_arr == s
    q_s  = [school_info.loc[a, "a_j"] for a, m in zip(bm_assignment, mask)
             if m and a is not None and a in school_info.index]
    summary_rows.append({
        "estrato"      : s,
        "n_familias"   : int(mask.sum()),
        "n_asignados"  : sum(1 for a, m in zip(bm_assignment, mask) if m and a is not None),
        "a_j_media"    : round(float(np.mean(q_s)), 4) if q_s else np.nan,
    })

pd.DataFrame(summary_rows).to_csv(REP_DIR / "matching_bm_summary.csv", index=False)
log.info("  matching_bm_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Figuras
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 8 — Generando figuras...")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "figure.dpi": 150,
})

# Fig 1: a_j media asignada por estrato
estratos = list(range(1, 7))
q_by_s   = [bm_metrics.get(f"q_estrato_{s}", np.nan) for s in estratos]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(estratos, q_by_s, color="#4292c6", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Estrato del hogar")
ax.set_ylabel("Atractivo medio del colegio asignado ($a_j$)")
ax.set_xticks(estratos)
ax.set_xticklabels([f"E{s}" for s in estratos])
ax.axhline(bm_metrics["eficiencia_q"], color="#08519c", linestyle="--",
           linewidth=1.2, label=f"Media global = {bm_metrics['eficiencia_q']:.3f}")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "bm_calidad_estrato.png", bbox_inches="tight")
plt.close(fig)

# Fig 2: distribución de a_j asignado
a_vals = bm_df["a_j"].dropna().values
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(a_vals, bins=40, color="#4292c6", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(a_vals.mean(), color="#08519c", linestyle="--", linewidth=1.2,
           label=f"Media = {a_vals.mean():.3f}")
ax.set_xlabel("Atractivo del colegio asignado ($a_j$)")
ax.set_ylabel("Familias")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "bm_distribucion_aj.png", bbox_inches="tight")
plt.close(fig)

log.info("  bm_calidad_estrato.png")
log.info("  bm_distribucion_aj.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN — Boston Mechanism (población expandida)")
log.info("=" * 60)
for k, v in bm_metrics.items():
    log.info(f"  {k:<20} {v}")
log.info("Done.")
