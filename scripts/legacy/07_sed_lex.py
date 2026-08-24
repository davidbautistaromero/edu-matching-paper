"""
09_sed_lex.py
=============
Aplica el mecanismo de la Secretaría de Educación del Distrito (SED Bogotá)
con prioridad lexicográfica según la Resolución 1587 de 2025.
Solo familias de primer ingreso (n_hijos_ingreso > 0).

Modelo de prioridad
-------------------
La Resolución 1587/2025 (Art. 19-20) establece un orden lexicográfico de
categorías priorizadas antes que la población general. Con EM2021 se usan
los umbrales DANE 2024 sobre N_ingpc (ingreso per cápita mensual):
  - SISBEN A → N_ingpc < 227,220             (cat = 0, prioridad máxima)
  - SISBEN B → N_ingpc < 460,198             (cat = 1, prioridad alta)
  - SISBEN C → N_ingpc < 897,987             (cat = 2, prioridad media)
  - SISBEN D → N_ingpc ≥ 897,987             (cat = 3, prioridad baja)
  Fallback: si N_ingpc es nulo → E1→A, E2→B, E3→C, E4-6→D.

Codificación:
    priority(i, j) = cat(i) × 1e6 + dist_haversine(i, j)

El multiplicador 1e6 garantiza que el orden de categorías domine sobre
cualquier distancia plausible (máx ~30 km en Bogotá). Dentro de cada
categoría, desempata por proximidad geográfica — idéntico al criterio SED.

Comparación
-----------
También corre DA puro por distancia como control para aislar el efecto
de la priorización lexicográfica sobre las métricas de equidad y eficiencia.

Inputs
------
  data/primary/preferencias_familias.parquet
  data/processed/familias_expandidas.parquet
  data/processed/distancias_expandidas.parquet
  data/primary/colegios_capacidad.parquet

Outputs
-------
  data/results/matching_sed_lex.parquet      — asignación SED lexicográfico
  data/results/matching_sed_dist.parquet     — asignación DA puro distancia (control)
  reports/matching_sed_comparison.csv        — métricas comparativas
  reports/figures/matching/sed_equidad.png
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

from matching_utils import compute_metrics, deferred_acceptance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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
CAT_OFFSET   = 1_000_000   # garantiza que categoría domine sobre distancia

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

# Filtrar: solo familias de primer ingreso
mask    = (fam_df["n_hijos_ingreso"] > 0).values
fam_df  = fam_df[mask].reset_index(drop=True)
pref_df = pref_df[mask].reset_index(drop=True)
dist_df = dist_df[mask].reset_index(drop=True)

log.info(f"  Familias totales        : {n:,}")
log.info(f"  Familias primer ingreso : {len(fam_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Construir school_info y capacidades
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 — Construyendo school_info...")

school_info = cap_df.set_index("id_establecimiento")[[
    "capacidad", "a_j", "a_j_visual", "a_j_nonvisual", "q_j", "sobre_demanda_j", "nombre_localidad"
]].copy()
school_info["capacidad"] = school_info["capacidad"].clip(lower=CAPACITY_MIN).astype(int)
school_cap = school_info["capacidad"].to_dict()

log.info(f"  Capacidad total      : {school_info['capacidad'].sum():,}")
log.info(f"  Ratio cupos/familias : {school_info['capacidad'].sum() / len(fam_df):.2f}x")

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
# 4. Preparar arrays
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 4 — Preparando arrays...")

estrato_arr = (
    pd.to_numeric(fam_df["estrato_real"], errors="coerce")
    .fillna(3).clip(1, 6).astype(int).values
)

# Categoría de prioridad SED — leer sisben_cat precalculado en familias_expandidas.parquet
# Fallback cat=3 (D, no priorizado) si la columna aún no existe (pipeline no recorrido)
if "sisben_cat" in fam_df.columns:
    categoria = pd.to_numeric(fam_df["sisben_cat"], errors="coerce").fillna(3).astype(int).values
else:
    log.warning("  sisben_cat no encontrado — fallback por estrato (correr 05b primero)")
    categoria = np.where(estrato_arr <= 1, 0,
                np.where(estrato_arr <= 2, 1,
                np.where(estrato_arr <= 3, 2, 3))).astype(int)

SISBEN_LABELS = ["A (pobreza extrema)", "B (pobreza moderada)", "C (vulnerable)", "D (no priorizado)"]
for cat, label in enumerate(SISBEN_LABELS):
    n = (categoria == cat).sum()
    log.info(f"  SISBEN {label}: {n:,} ({n/len(categoria)*100:.1f}%)")

pref_lists = [
    [s for s in row.values if isinstance(s, str) and s in valid_schools]
    for _, row in pref_df[pref_cols].iterrows()
]

log.info(f"  Longitud media lista de prefs: {np.mean([len(p) for p in pref_lists]):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Funciones de prioridad
# ─────────────────────────────────────────────────────────────────────────────
def priority_lex(student_idx: int, school_id: str) -> float:
    """Lexicográfico: categoría primero, distancia como desempate."""
    col = school_col_idx.get(school_id)
    if col is None:
        return np.inf
    return float(categoria[student_idx]) * CAT_OFFSET + float(dist_matrix[student_idx, col])

def priority_dist(student_idx: int, school_id: str) -> float:
    """Solo distancia Haversine — DA puro (control)."""
    col = school_col_idx.get(school_id)
    if col is None:
        return np.inf
    return float(dist_matrix[student_idx, col])

# ─────────────────────────────────────────────────────────────────────────────
# 6. Ejecutar mecanismos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 5 — Ejecutando DA con prioridad SED lexicográfica...")
t0 = time.time()
asgn_lex = deferred_acceptance(pref_lists, school_cap, priority_lex)
log.info(f"  Completado en {time.time()-t0:.1f}s | asignados: {sum(a is not None for a in asgn_lex):,}")

log.info("Paso 6 — Ejecutando DA puro distancia (control)...")
t0 = time.time()
asgn_dist = deferred_acceptance(pref_lists, school_cap, priority_dist)
log.info(f"  Completado en {time.time()-t0:.1f}s | asignados: {sum(a is not None for a in asgn_dist):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Métricas
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 7 — Calculando métricas...")

met_lex = compute_metrics(
    assignment  = asgn_lex,
    pref_lists  = pref_lists,
    school_cap  = school_cap,
    priority_fn = priority_lex,
    school_info = school_info,
    quality_col = "q_j",
    visual_col  = "a_j_visual",
    estrato_arr = estrato_arr,
    label       = "SED-lex",
)

met_dist = compute_metrics(
    assignment  = asgn_dist,
    pref_lists  = pref_lists,
    school_cap  = school_cap,
    priority_fn = priority_dist,
    school_info = school_info,
    quality_col = "q_j",
    visual_col  = "a_j_visual",
    estrato_arr = estrato_arr,
    label       = "DA-dist",
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 8 — Guardando resultados...")

def build_df(asgn, label):
    return pd.DataFrame({
        "DIRECTORIO"         : fam_df["DIRECTORIO"].values,
        "estrato_real"       : estrato_arr,
        "sisben_cat"         : categoria,
        "id_establecimiento" : asgn,
    }).merge(
        school_info[["a_j", "q_j", "sobre_demanda_j", "nombre_localidad"]].reset_index(),
        on="id_establecimiento", how="left",
    )

build_df(asgn_lex,  "SED-lex").to_parquet(OUT_DIR / "matching_sed_lex.parquet",  index=False)
build_df(asgn_dist, "DA-dist").to_parquet(OUT_DIR / "matching_sed_dist.parquet", index=False)
log.info("  matching_sed_lex.parquet")
log.info("  matching_sed_dist.parquet")

comp = pd.DataFrame([met_lex, met_dist])
comp.to_csv(REP_DIR / "matching_sed_comparison.csv", index=False)
log.info("  matching_sed_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Figuras: equidad por estrato y por categoría SISBEN
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 9 — Generando figuras...")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "figure.dpi": 150,
})

w = 0.35
COLOR_LEX  = "#4292c6"
COLOR_DIST = "#969696"

# ── Panel A: por estrato ──────────────────────────────────────────────────────
estratos = list(range(1, 7))
labels_e = [f"E{s}" for s in estratos]
qj_lex_e  = [met_lex.get(f"qj_estrato_{s}",  np.nan) for s in estratos]
qj_dist_e = [met_dist.get(f"qj_estrato_{s}", np.nan) for s in estratos]

# ── Panel B: por categoría SISBEN (calculado directamente desde los matchings) ─
SISBEN_LABELS_SHORT = ["A\n(ext.)", "B\n(mod.)", "C\n(vuln.)", "D\n(no prior.)"]
result_lex  = build_df(asgn_lex,  "SED-lex")
result_dist = build_df(asgn_dist, "DA-dist")

qj_lex_s, qj_dist_s = [], []
for cat_i in range(4):
    mask_cat = result_lex["sisben_cat"] == cat_i
    qj_lex_s.append(pd.to_numeric(result_lex.loc[mask_cat, "q_j"], errors="coerce").mean())
    mask_cat = result_dist["sisben_cat"] == cat_i
    qj_dist_s.append(pd.to_numeric(result_dist.loc[mask_cat, "q_j"], errors="coerce").mean())

# ── Figura combinada 1×2 ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, labels, qj_lex_v, qj_dist_v, xlabel, title in [
    (axes[0], labels_e,           qj_lex_e, qj_dist_e,
     "Estrato del hogar",         "(a) Calidad asignada por estrato"),
    (axes[1], SISBEN_LABELS_SHORT, qj_lex_s, qj_dist_s,
     "Categoría SISBEN",          "(b) Calidad asignada por categoría SISBEN"),
]:
    x = np.arange(len(labels))
    ax.bar(x - w/2, qj_lex_v,  w, label="SED lexicográfico", color=COLOR_LEX,  edgecolor="white")
    ax.bar(x + w/2, qj_dist_v, w, label="DA puro distancia", color=COLOR_DIST, edgecolor="white")
    vals = [v for v in qj_lex_v + qj_dist_v if v is not None and not np.isnan(v)]
    if vals:
        ax.set_ylim(min(vals) * 0.995, max(vals) * 1.005)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("$q_j$ medio (Saber 11 estandarizado)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=8, frameon=False)

fig.suptitle("Equidad en calidad académica asignada — SED lexicográfico vs DA distancia",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "sed_equidad.png", bbox_inches="tight")
plt.close(fig)
log.info("  sed_equidad.png (panel estrato + panel SISBEN)")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("COMPARATIVA SED-lex vs DA-dist")
log.info("=" * 60)
cols = ["condicion", "n_asignados", "rank_medio", "blocking_pairs",
        "equidad_aj", "sesgo_visual", "qj_medio", "rechazo_total"]
print(comp[cols].to_string(index=False))
log.info("Done.")
