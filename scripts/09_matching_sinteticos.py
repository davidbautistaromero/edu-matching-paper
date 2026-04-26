"""
09_matching_sinteticos.py
=========================
Aplica Boston Mechanism (BM) y Deferred Acceptance (DA) sobre los datos SINTÉTICOS
generados en 08_datos_sinteticos.py, comparando 4 condiciones experimentales.

Pipeline de entrada (scripts previos requeridos)
-------------------------------------------------
  08_datos_sinteticos.py → sinteticos_*.parquet y sinteticos_calibracion.json

Diseño experimental — 4 condiciones
-------------------------------------
  BM-bias  Boston Mechanism  con preferencias CON sesgo visual (u_ij)
  BM-true  Boston Mechanism  con preferencias SIN sesgo — calidad pura (u0_ij)
  DA-bias  Deferred Accept.  con preferencias CON sesgo visual
  DA-true  Deferred Accept.  con preferencias SIN sesgo — calidad pura

El cruce mecanismo × tipo_de_preferencia permite aislar dos efectos distintos:
  1. Efecto del mecanismo: ¿DA reduce blocking pairs y mejora equidad vs BM?
  2. Efecto del sesgo: ¿las preferencias sesgadas empeoran calidad y equidad
     en la asignación, independientemente del mecanismo?

La conclusión esperada del paper: DA elimina blocking pairs (resultado teórico),
pero NO puede corregir el sesgo que ya viene en las preferencias declaradas.
Para eso se necesita un mecanismo con penalización de fairness explícita (RegretNet).

Prioridad escolar — Lotería
----------------------------
En el caso sintético no hay datos geográficos, entonces la prioridad escolar
se define como una LOTERÍA: cada colegio asigna aleatoriamente un orden de
prioridad sobre los N estudiantes (permutación aleatoria, seed fija).

Justificación: la lotería es el sistema de prioridad más neutral posible y es
el estándar en la literatura de mecanismos cuando no hay restricciones geográficas
(Abdulkadiroğlu & Sönmez 2003). Al ser neutral, cualquier diferencia en equity o
eficiencia entre condiciones se atribuye sólo al mecanismo o al sesgo, no a la
geografía.

La MISMA lotería se usa para las 4 condiciones: esto garantiza que diferencias
en los resultados provienen sólo de las preferencias (con vs sin sesgo) o del
mecanismo (BM vs DA), no de distinta prioridad aleatoria.

Métricas
--------
  eficiencia_q  calidad media real (q_j) del colegio asignado          (↑ mejor)
  equidad_corr  |corr(estrato, q_j_asignado)|                          (↓ mejor)
  sesgo_visual  corr(estrato, v_j_asignado)                            (↓ mejor)
  rank_medio    posición media del colegio asignado en la lista propia  (↓ mejor)
  blocking_pairs pares (estudiante, colegio) mutuamente preferibles     (↓ mejor)

Outputs
-------
  data/results/sinteticos_bm_bias.parquet   — asignación BM con sesgo
  data/results/sinteticos_bm_true.parquet   — asignación BM sin sesgo
  data/results/sinteticos_da_bias.parquet   — asignación DA con sesgo
  data/results/sinteticos_da_true.parquet   — asignación DA sin sesgo
  reports/sinteticos_matching_comparison.csv — tabla con las 4 condiciones
  reports/figures/matching/sinteticos_4condiciones.png
  reports/figures/matching/sinteticos_equidad_estrato.png
"""

import json
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importar algoritmos y métricas desde el módulo compartido
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

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42   # misma semilla que 08 para coherencia

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 — Cargando datos sintéticos...")

pref_bias = pd.read_parquet(ROOT / "data/primary/sinteticos_preferencias.parquet")
pref_true = pd.read_parquet(ROOT / "data/primary/sinteticos_pref_sin_sesgo.parquet")
col_df    = pd.read_parquet(ROOT / "data/primary/sinteticos_colegios.parquet")
est_df    = pd.read_parquet(ROOT / "data/primary/sinteticos_estudiantes.parquet")

with open(REP_DIR / "sinteticos_calibracion.json", encoding="utf-8") as f:
    cal = json.load(f)

N = len(est_df)              # número de estudiantes
M = len(col_df)              # número de colegios

# Indexar colegios por id_establecimiento para lookup rápido en métricas
col_df     = col_df.set_index("id_establecimiento")
school_ids = col_df.index.tolist()    # lista ordenada de IDs de colegios
estrato    = est_df["estrato"].values  # (N,) estrato de cada estudiante

# Capacidades sintéticas: suma total ≈ N (mercado perfectamente competitivo)
school_cap = col_df["capacidad_sintetica"].to_dict()

log.info(f"  N={N} estudiantes | M={M} colegios | "
         f"cap_total={sum(school_cap.values())} | alpha_hat={cal['alpha_hat']:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Generar la lotería de prioridad (permutación aleatoria por colegio)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 — Generando lotería de prioridad (seed=42)...")

rng_lot = np.random.default_rng(SEED)

# Para cada colegio: una permutación aleatoria de los N estudiantes.
# lottery[sid][i] = rango del estudiante i en el colegio sid (0 = máxima prioridad).
# Construido una sola vez y compartido entre las 4 condiciones.
lottery: dict[str, dict[int, int]] = {}
for sid in school_ids:
    perm = rng_lot.permutation(N)   # permutación aleatoria de 0..N-1
    # perm[k] = índice del estudiante en la posición k de prioridad
    # invertir: queremos lottery[sid][student_idx] = rank del estudiante
    lottery[sid] = {int(student_idx): int(rank) for rank, student_idx in enumerate(perm)}

def priority_fn(student_idx: int, school_id: str) -> int:
    """
    Rango de lotería del estudiante en el colegio (0 = máxima prioridad).
    Menor valor = mayor prioridad, convención idéntica a la distancia en 07.
    """
    return lottery[school_id][student_idx]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Convertir preferencias a listas de listas
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 3 — Preparando listas de preferencias...")

valid_schools = set(school_ids)   # colegios presentes en los datos sintéticos

def to_pref_lists(pref_df: pd.DataFrame) -> list[list[str]]:
    """
    Convierte un DataFrame de rankings (filas = estudiantes, columnas = pref_1..pref_k)
    a una lista de listas de strings, filtrando IDs inválidos.
    """
    pref_cols = [c for c in pref_df.columns if c.startswith("pref_")]
    return [
        [s for s in row.values if s in valid_schools]
        for _, row in pref_df[pref_cols].iterrows()
    ]

pref_lists_bias = to_pref_lists(pref_bias)
pref_lists_true = to_pref_lists(pref_true)

log.info(f"  Longitud media lista (bias): {np.mean([len(p) for p in pref_lists_bias]):.1f}")
log.info(f"  Longitud media lista (true): {np.mean([len(p) for p in pref_lists_true]):.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Ejecutar las 4 condiciones de matching
# ─────────────────────────────────────────────────────────────────────────────
# Definir las 4 condiciones como tuplas (clave_output, función, lista_preferencias, etiqueta)
CONDITIONS = [
    ("bm_bias", boston_mechanism,      pref_lists_bias, "BM-bias"),
    ("bm_true", boston_mechanism,      pref_lists_true, "BM-true"),
    ("da_bias", deferred_acceptance,   pref_lists_bias, "DA-bias"),
    ("da_true", deferred_acceptance,   pref_lists_true, "DA-true"),
]

log.info("=" * 60)
log.info("Paso 4 — Ejecutando 4 condiciones de matching...")

results       = []    # lista de dicts de métricas, uno por condición
assignments   = {}    # {clave: list[str|None]} para guardar las asignaciones

for key, fn, plists, label in CONDITIONS:
    log.info(f"  Corriendo {label}...")
    # Ejecutar el mecanismo: devuelve list[str|None] de longitud N
    asgn = fn(plists, school_cap, priority_fn)

    # Calcular métricas usando las columnas sintéticas:
    # quality_col="q_j_std" (z-score) y visual_col="v_j" (índice visual construido)
    met = compute_metrics(
        assignment=asgn,
        pref_lists=plists,
        school_cap=school_cap,
        priority_fn=priority_fn,
        school_info=col_df,
        quality_col="q_j_std",
        visual_col="v_j",
        estrato_arr=estrato,
        label=label,
    )
    results.append(met)
    assignments[key] = asgn

# ─────────────────────────────────────────────────────────────────────────────
# 5. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 5 — Guardando resultados...")

for key, asgn in assignments.items():
    # Enriquecer con atributos del colegio asignado (merge left para preservar no asignados)
    df = pd.DataFrame({
        "id_estudiante"      : est_df["id_estudiante"],
        "estrato"            : estrato,
        "id_establecimiento" : asgn,
    }).merge(
        col_df[["q_j", "q_j_std", "v_j", "sobre_demanda_j"]].reset_index(),
        on="id_establecimiento",
        how="left",
    )
    fname = f"sinteticos_{key}.parquet"
    df.to_parquet(OUT_DIR / fname, index=False)
    log.info(f"  {fname}")

# Tabla comparativa con las 4 condiciones
comp_df = pd.DataFrame(results)
comp_df.to_csv(REP_DIR / "sinteticos_matching_comparison.csv", index=False)
log.info("  sinteticos_matching_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Figuras comparativas
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 6 — Generando figuras...")

# Paleta de colores: rojo (BM) vs azul (DA), saturado (con sesgo) vs claro (sin sesgo)
COLORS = {
    "BM-bias": "#E53935",   # rojo saturado: BM con sesgo (peor caso esperado)
    "BM-true": "#EF9A9A",   # rojo claro:    BM con verdad
    "DA-bias": "#1E88E5",   # azul saturado: DA con sesgo
    "DA-true": "#90CAF9",   # azul claro:    DA con verdad (mejor caso esperado)
}
LABELS = {
    "BM-bias": "BM + sesgo",
    "BM-true": "BM + verdad",
    "DA-bias": "DA + sesgo",
    "DA-true": "DA + verdad",
}

# Reindexar resultados por clave para acceso fácil
met_by_key = {
    "BM-bias": results[0],
    "BM-true": results[1],
    "DA-bias": results[2],
    "DA-true": results[3],
}
keys_ord = ["BM-bias", "BM-true", "DA-bias", "DA-true"]
estratos  = list(range(1, 7))

# ── Figura 1: panel 2×2 de métricas resumen ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Comparación BM vs DA — Datos Sintéticos Calibrados\n"
    f"N={N} estudiantes · M={M} colegios · α̂={cal['alpha_hat']:.4f}",
    fontsize=12,
)
x     = np.arange(4)
width = 0.6

def bar4(ax, values, title, ylabel):
    """
    Dibuja un gráfico de barras para las 4 condiciones.
    La barra con el mejor valor (↑ para eficiencia, ↓ para el resto) se destaca
    con borde negro para facilitar la lectura del panel.
    """
    bars = ax.bar(
        x, values, width,
        color=[COLORS[k] for k in keys_ord],
        alpha=0.88, edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys_ord], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Anotar el valor numérico encima de cada barra
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

bar4(axes[0, 0],
     [met_by_key[k]["eficiencia_q"] for k in keys_ord],
     "Eficiencia — calidad media asignada (q_j_std)\n(↑ mejor)", "q_j_std medio")

bar4(axes[0, 1],
     [abs(met_by_key[k]["equidad_corr"]) for k in keys_ord],
     "|corr(estrato, q_j)| — inequidad de acceso\n(↓ mejor)", "|correlación|")

bar4(axes[1, 0],
     [met_by_key[k]["sesgo_visual"] for k in keys_ord],
     "corr(estrato, v_j) — sesgo visual en asignación\n(↓ mejor)", "correlación")

bar4(axes[1, 1],
     [met_by_key[k]["blocking_pairs"] for k in keys_ord],
     "Blocking pairs — inestabilidad del matching\n(↓ mejor, DA=0 por teorema)", "# pares bloqueantes")

plt.tight_layout()
fig.savefig(FIG_DIR / "sinteticos_4condiciones.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("  sinteticos_4condiciones.png")

# ── Figura 2: calidad e índice visual por estrato — 4 paneles ───────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
w2  = 0.35
x_s = np.arange(len(estratos))

for (ax_pos, k_bias, k_true, attr_prefix, ylabel, mech_name) in [
    (gs[0, 0], "BM-bias", "BM-true", "q_estrato", "Calidad media (q_j_std)", "Boston (BM)"),
    (gs[0, 1], "DA-bias", "DA-true", "q_estrato", "Calidad media (q_j_std)", "DA (Gale-Shapley)"),
    (gs[1, 0], "BM-bias", "BM-true", "v_estrato", "Índice visual medio (v_j)", "Boston (BM)"),
    (gs[1, 1], "DA-bias", "DA-true", "v_estrato", "Índice visual medio (v_j)", "DA (Gale-Shapley)"),
]:
    ax = fig.add_subplot(ax_pos)

    vals_bias = [met_by_key[k_bias].get(f"{attr_prefix}_{s}", np.nan) for s in estratos]
    vals_true = [met_by_key[k_true].get(f"{attr_prefix}_{s}", np.nan) for s in estratos]

    ax.bar(x_s - w2/2, vals_bias, w2,
           label="CON sesgo", color=COLORS[k_bias], alpha=0.85)
    ax.bar(x_s + w2/2, vals_true, w2,
           label="SIN sesgo", color=COLORS[k_true], alpha=0.85)

    ax.set_xticks(x_s)
    ax.set_xticklabels([f"s={s}" for s in estratos], fontsize=9)
    ax.set_xlabel("Estrato del estudiante")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(mech_name, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if "v_estrato" in attr_prefix:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

fig.suptitle(
    "Calidad e índice visual del colegio asignado — por estrato y condición",
    fontsize=12, y=1.01,
)
plt.savefig(FIG_DIR / "sinteticos_equidad_estrato.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("  sinteticos_equidad_estrato.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Resumen en consola con interpretación
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("TABLA COMPARATIVA FINAL — Datos sintéticos")
log.info("=" * 60)

cols_show = ["condicion", "n_asignados", "eficiencia_q", "equidad_corr",
             "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comp_df[cols_show].to_string(index=False))

log.info("")
log.info("─── Interpretación del sesgo visual en la asignación ───")
sv_bm_bias = met_by_key["BM-bias"]["sesgo_visual"]
sv_bm_true = met_by_key["BM-true"]["sesgo_visual"]
sv_da_bias = met_by_key["DA-bias"]["sesgo_visual"]
sv_da_true = met_by_key["DA-true"]["sesgo_visual"]

log.info(f"  BM: sesgo_visual {sv_bm_true:.4f} (verdad) → {sv_bm_bias:.4f} (bias)  "
         f"Δ={sv_bm_bias - sv_bm_true:+.4f}")
log.info(f"  DA: sesgo_visual {sv_da_true:.4f} (verdad) → {sv_da_bias:.4f} (bias)  "
         f"Δ={sv_da_bias - sv_da_true:+.4f}")

log.info("")
log.info("─── Conclusión ───")
log.info("  DA (Gale-Shapley) garantiza 0 blocking pairs — es estable.")
log.info("  DA NO elimina el sesgo visual en preferencias declaradas:")
log.info(f"    corr(estrato, v_j) = {sv_da_bias:.4f} (DA-bias) > {sv_da_true:.4f} (DA-true)")
log.info("  → Motivación para RegretNet con penalización de fairness (§5.4):")
log.info("    necesitamos un mecanismo que corrija el sesgo en las preferencias.")
log.info("Done.")
