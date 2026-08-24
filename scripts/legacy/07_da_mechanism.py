"""
08_da_mechanism.py
==================
Aplica Deferred Acceptance / Gale-Shapley (DA) sobre la población expandida
de Bogotá. Solo familias de primer ingreso (n_hijos_ingreso > 0).

Algoritmo
---------
DA opera en rondas con aceptaciones PROVISIONALES:
  1. Cada familia no asignada propone al siguiente colegio de su lista.
  2. Cada colegio acepta provisionalmente a los mejores candidatos hasta
     su cupo (criterio: menor distancia Haversine = mayor prioridad).
  3. Si llega un candidato con mejor prioridad que el peor provisional,
     lo desplaza — el expulsado vuelve al mercado.
  4. El proceso termina cuando nadie es rechazado en una ronda.

Propiedades:
  - Strategy-proof: declarar preferencias verdaderas es estrategia dominante.
  - Estable: el matching no tiene blocking pairs (DA debería dar BP = 0).
  - Student-optimal: mejor matching estable posible para los estudiantes.

Inputs  (idénticos a 07_boston_mechanism.py)
------
  data/primary/preferencias_familias.parquet
  data/processed/familias_expandidas.parquet
  data/processed/distancias_expandidas.parquet
  data/primary/colegios_capacidad.parquet

Outputs
-------
  data/results/matching_da.parquet
  reports/matching_da_summary.csv
  reports/figures/matching/da_equidad_estrato.png
  reports/figures/matching/da_distribucion_qj.png
"""

# Este script fue movido a un subdirectorio; ROOT y matching_utils
# se resuelven relativos a scripts/ para que siga siendo ejecutable.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matching_utils import deferred_acceptance, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
PREF_P  = ROOT / "data" / "primary"   / "preferencias_familias.parquet"
FAM_P   = ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_P  = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
CAP_P   = ROOT / "data" / "primary"   / "colegios_capacidad.parquet"
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_MIN = 5

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

n = min(len(fam_df), len(pref_df), len(dist_df))
fam_df  = fam_df.iloc[:n].reset_index(drop=True)
pref_df = pref_df.iloc[:n].reset_index(drop=True)
dist_df = dist_df.iloc[:n].reset_index(drop=True)

# Solo familias de primer ingreso
mask    = (fam_df["n_hijos_ingreso"] > 0).values
fam_df  = fam_df[mask].reset_index(drop=True)
pref_df = pref_df[mask].reset_index(drop=True)
dist_df = dist_df[mask].reset_index(drop=True)

log.info(f"  Familias totales        : {n:,}")
log.info(f"  Familias primer ingreso : {len(fam_df):,}")
log.info(f"  Colegios en distancias  : {dist_df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Construir school_info
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 — Construyendo school_info desde colegios_capacidad...")

school_info = cap_df.set_index("id_establecimiento")[[
    "capacidad", "a_j", "a_j_visual", "a_j_nonvisual", "q_j", "sobre_demanda_j", "nombre_localidad"
]].copy()
school_info["capacidad"] = school_info["capacidad"].clip(lower=CAPACITY_MIN).astype(int)

school_cap = school_info["capacidad"].to_dict()

log.info(f"  Colegios con capacidad  : {len(school_info):,}")
log.info(f"  Capacidad total         : {school_info['capacidad'].sum():,}")
log.info(f"  Ratio cupos/familias    : {school_info['capacidad'].sum() / len(fam_df):.2f}x")

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
dist_matrix    = dist_df.values.astype(np.float32)
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
# 6. Ejecutar DA
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 5 — Ejecutando Deferred Acceptance (Gale-Shapley)...")

t0 = time.time()
da_assignment = deferred_acceptance(pref_lists, school_cap, priority_fn)
elapsed = time.time() - t0

n_assigned = sum(a is not None for a in da_assignment)
log.info(f"  Completado en {elapsed:.1f}s")
log.info(f"  Asignados   : {n_assigned:,} / {len(fam_df):,} ({100*n_assigned/len(fam_df):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Métricas
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 6 — Calculando métricas...")

da_metrics = compute_metrics(
    assignment   = da_assignment,
    pref_lists   = pref_lists,
    school_cap   = school_cap,
    priority_fn  = priority_fn,
    school_info  = school_info,
    quality_col  = "q_j",
    visual_col   = "a_j_visual",
    estrato_arr  = estrato_arr,
    label        = "DA",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 7 — Guardando resultados...")

da_df = pd.DataFrame({
    "DIRECTORIO"         : fam_df["DIRECTORIO"].values,
    "estrato_real"       : estrato_arr,
    "sisben_cat"         : pd.to_numeric(fam_df["sisben_cat"], errors="coerce").fillna(3).astype(int).values if "sisben_cat" in fam_df.columns else np.full(len(fam_df), 3, dtype=int),
    "id_establecimiento" : da_assignment,
}).merge(
    school_info[["a_j", "q_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
    on="id_establecimiento",
    how="left",
)

da_df.to_parquet(OUT_DIR / "matching_da.parquet", index=False)
log.info(f"  matching_da.parquet — {len(da_df):,} filas")

summary_rows = [{"estrato": "TOTAL", **{k: v for k, v in da_metrics.items()}}]
for s in range(1, 7):
    mask_s = estrato_arr == s
    q_s    = [school_info.loc[a, "a_j"] for a, m in zip(da_assignment, mask_s)
              if m and a is not None and a in school_info.index]
    summary_rows.append({
        "estrato"     : s,
        "n_familias"  : int(mask_s.sum()),
        "n_asignados" : sum(1 for a, m in zip(da_assignment, mask_s) if m and a is not None),
        "a_j_media"   : round(float(np.mean(q_s)), 4) if q_s else np.nan,
    })

pd.DataFrame(summary_rows).to_csv(REP_DIR / "matching_da_summary.csv", index=False)
log.info("  matching_da_summary.csv")

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

estratos  = list(range(1, 7))
labels_e  = [f"E{s}" for s in estratos]
aj_by_s   = [da_metrics.get(f"aj_estrato_{s}", np.nan) for s in estratos]
qj_by_s   = [da_metrics.get(f"qj_estrato_{s}", np.nan) for s in estratos]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.bar(estratos, aj_by_s, color="#4292c6", edgecolor="white", linewidth=0.5)
ax.set_ylim(min(v for v in aj_by_s if not np.isnan(v)) * 0.95,
            max(v for v in aj_by_s if not np.isnan(v)) * 1.05)
ax.set_xlabel("Estrato del hogar")
ax.set_ylabel("Atractivo medio del colegio asignado ($a_j$)")
ax.set_xticks(estratos); ax.set_xticklabels(labels_e)
ax.axhline(da_metrics["equidad_aj"], color="#08519c", linestyle="--",
           linewidth=1.2, label=f"Media = {da_metrics['equidad_aj']:.3f}")
ax.legend(fontsize=9)

ax = axes[1]
ax.bar(estratos, qj_by_s, color="#74c476", edgecolor="white", linewidth=0.5)
ax.set_ylim(min(v for v in qj_by_s if not np.isnan(v)) * 0.995,
            max(v for v in qj_by_s if not np.isnan(v)) * 1.005)
ax.set_xlabel("Estrato del hogar")
ax.set_ylabel("Puntaje Saber 11 medio del colegio asignado ($q_j$)")
ax.set_xticks(estratos); ax.set_xticklabels(labels_e)
ax.axhline(da_metrics["qj_medio"], color="#238b45", linestyle="--",
           linewidth=1.2, label=f"Media = {da_metrics['qj_medio']:.1f}")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "da_equidad_estrato.png", bbox_inches="tight")
plt.close(fig)

q_vals = da_df["q_j"].dropna().values
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(q_vals, bins=40, color="#74c476", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(q_vals.mean(), color="#238b45", linestyle="--", linewidth=1.2,
           label=f"Media = {q_vals.mean():.1f}")
ax.set_xlabel("Puntaje Saber 11 del colegio asignado ($q_j$)")
ax.set_ylabel("Familias")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "da_distribucion_qj.png", bbox_inches="tight")
plt.close(fig)

log.info("  da_equidad_estrato.png")
log.info("  da_distribucion_qj.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN — Deferred Acceptance (población expandida)")
log.info("=" * 60)
for k, v in da_metrics.items():
    log.info(f"  {k:<25} {v}")
log.info("Done.")
