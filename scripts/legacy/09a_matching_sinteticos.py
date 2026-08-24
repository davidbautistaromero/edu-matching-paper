"""
09b_matching_sinteticos.py
==========================
Aplica tres mecanismos de asignación sobre los datos sintéticos generados
por 08b_datos_sinteticos.py. Familias y colegios distribuidos uniformemente
en [0,1]² — aísla el efecto puro del sesgo visual.

Mecanismos comparados
---------------------
1. BM   — Boston Mechanism (ineficiente, inestable, no strategy-proof)
2. DA   — Deferred Acceptance puro (estable, strategy-proof)
3. SED  — DA con prioridad lexicográfica: estratos 1-2 primero, lotería como
           desempate dentro de cada categoría

Para cada mecanismo se corre con dos sets de preferencias:
  - bias : preferencias que incluyen sesgo visual (U_bias)
  - true : preferencias verdaderas sin sesgo visual (U_true)

Esto produce 6 condiciones: BM-bias, BM-true, DA-bias, DA-true, SED-bias, SED-true.

La comparación bias vs true dentro de cada mecanismo aísla el efecto causal
del sesgo visual sobre la equidad de la asignación.
La comparación entre mecanismos (con mismas preferencias) muestra qué mecanismo
mitiga mejor el daño del sesgo visual.

Prioridad
---------
BM, DA y SED usan lotería como prioridad base (equidad ex-ante).
SED añade la capa lexicográfica: estratos 1-2 tienen prioridad sobre 3-6;
dentro de cada categoría desempata por lotería.

Nota metodológica: el uso de lotería (vs distancia) en el sintético es
deliberado — permite aislar el efecto del sesgo visual en las preferencias
sin mezclar el canal geográfico. El SED real usa distancia como desempate,
lo que suprime el sesgo visual; con lotería se muestra que esa propiedad
viene del criterio de desempate, no de la prioridad por estrato.

Inputs
------
  data/primary/sinteticos_b_colegios.parquet
  data/primary/sinteticos_b_estudiantes.parquet
  data/primary/sinteticos_b_preferencias_bias.parquet
  data/primary/sinteticos_b_preferencias_true.parquet

Outputs
-------
  data/results/sinteticos_b_resultados.parquet
  reports/sinteticos_b_comparativa.csv
  reports/figures/matching/sinteticos_b_comparativa.png
"""

# Este script fue movido a un subdirectorio; ROOT y matching_utils
# se resuelven relativos a scripts/ para que siga siendo ejecutable.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matching_utils import boston_mechanism, deferred_acceptance, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
IN_DIR  = ROOT / "data" / "primary"
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
CAT_OFFSET = 1_000_000   # garantiza que categoría lexicográfica domine sobre lotería

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 — Cargando datos sintéticos...")

col_df = pd.read_parquet(IN_DIR / "sinteticos_b_colegios.parquet")
col_df = col_df.set_index("id_establecimiento")
school_ids = col_df.index.tolist()
school_cap = col_df["capacidad_sintetica"].to_dict()
M = len(col_df)

est_df    = pd.read_parquet(IN_DIR / "sinteticos_b_estudiantes.parquet")
pref_bias = pd.read_parquet(IN_DIR / "sinteticos_b_preferencias_bias.parquet")
pref_true = pd.read_parquet(IN_DIR / "sinteticos_b_preferencias_true.parquet")

N = len(est_df)
log.info(f"  N={N} | M={M} | Cupos={sum(school_cap.values())} | Ratio={sum(school_cap.values())/N:.3f}x")

# "estrato" en este parquet contiene sisben_cat (0=A extrema, 1=B moderada, 2=C, 3=D)
sisben_arr  = est_df["estrato"].values.astype(int)
# ingreso per cápita real — se usa como variable continua para corr(ingreso, v_j)
ingpc_arr   = pd.to_numeric(est_df["N_ingpc"], errors="coerce").values
estrato_arr = ingpc_arr   # compute_metrics usa estrato_arr para las correlaciones

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preparar listas de preferencias
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 2 — Preparando listas de preferencias...")

valid_schools = set(school_ids)
pref_cols_all = [f"pref_{k+1}" for k in range(M)]

def to_pref_lists(df):
    cols = [c for c in pref_cols_all if c in df.columns]
    return [
        [s for s in row.values if s in valid_schools]
        for _, row in df[cols].iterrows()
    ]

lista_bias = to_pref_lists(pref_bias)
lista_true = to_pref_lists(pref_true)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Lotería de prioridad (equidad ex-ante)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 3 — Generando lotería de prioridad (seed=42)...")

rng = np.random.default_rng(SEED)
lottery = {}
for sid in school_ids:
    perm = rng.permutation(N)
    lottery[sid] = {int(idx): int(rank) for rank, idx in enumerate(perm)}

def priority_lottery(i, sid):
    return lottery[sid][i]

def priority_sed(i, sid, _sisben=sisben_arr):
    # Cat 0 (A, extrema) = máxima prioridad; cat 3 (D) = mínima
    # sisben_cat ya es 0-3: menor valor = mayor prioridad
    return int(_sisben[i]) * CAT_OFFSET + lottery[sid][i]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Ejecutar 6 condiciones
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 4 — Ejecutando 6 condiciones...")

CONDITIONS = [
    ("BM-bias",  boston_mechanism,    lista_bias, priority_lottery),
    ("BM-true",  boston_mechanism,    lista_true, priority_lottery),
    ("DA-bias",  deferred_acceptance, lista_bias, priority_lottery),
    ("DA-true",  deferred_acceptance, lista_true, priority_lottery),
    ("SED-bias", deferred_acceptance, lista_bias, priority_sed),
    ("SED-true", deferred_acceptance, lista_true, priority_sed),
]

results  = []
all_rows = []

for cond_label, fn, plists, pfn in CONDITIONS:
    log.info(f"  {cond_label}...")
    asgn = fn(plists, school_cap, pfn)
    met  = compute_metrics(
        assignment  = asgn,
        pref_lists  = plists,
        school_cap  = school_cap,
        priority_fn = pfn,
        school_info = col_df,
        quality_col = "q_j_std",
        visual_col  = "v_j",
        estrato_arr = estrato_arr,
        label       = cond_label,
    )
    met["equidad_qj"] = met.pop("equidad_aj")
    # limpiar métricas por estrato (no aplican con sisben_cat)
    for s in range(1, 7):
        met.pop(f"aj_estrato_{s}", None)
        met.pop(f"rechazo_estrato_{s}", None)
    results.append(met)

    # Rechazo por categoría SISBEN
    SISBEN_LABELS = {0: "A_extrema", 1: "B_moderada", 2: "C_vulnerable", 3: "D_no_prio"}
    for cat, label in SISBEN_LABELS.items():
        mask_cat = sisben_arr == cat
        n_cat    = mask_cat.sum()
        n_sin_asgn = sum(1 for i, a in enumerate(asgn) if mask_cat[i] and a is None)
        met[f"rechazo_sisben_{label}"] = round(n_sin_asgn / n_cat, 4) if n_cat > 0 else np.nan

    for i, a in enumerate(asgn):
        all_rows.append({
            "condicion"          : cond_label,
            "id_estudiante"      : est_df.iloc[i]["id_estudiante"],
            "sisben_cat"         : int(sisben_arr[i]),
            "id_establecimiento" : a,
        })

comp_df = pd.DataFrame(results)

log.info("\n  Efecto sesgo visual:")
for mec in ["BM", "DA", "SED"]:
    b = comp_df[comp_df["condicion"] == f"{mec}-bias"].iloc[0]
    t = comp_df[comp_df["condicion"] == f"{mec}-true"].iloc[0]
    log.info(f"    {mec}: Δequidad_qj={b['equidad_qj']-t['equidad_qj']:+.4f}  "
             f"Δsesgo_visual={b['sesgo_visual']-t['sesgo_visual']:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 5 — Guardando resultados...")

res_df = pd.DataFrame(all_rows).merge(
    col_df[["q_j", "q_j_std", "v_j"]].reset_index(),
    on="id_establecimiento", how="left"
)
res_df.to_parquet(OUT_DIR / "sinteticos_b_resultados.parquet", index=False)
comp_df.to_csv(REP_DIR / "sinteticos_b_comparativa.csv", index=False)
log.info("  sinteticos_b_resultados.parquet")
log.info("  sinteticos_b_comparativa.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Figura comparativa
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 6 — Generando figura...")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "figure.dpi": 150,
})

mecanismos = ["BM", "DA", "SED"]
color_bias = "#4292c6"
color_true = "#bdd7e7"
x = np.arange(len(mecanismos))
w = 0.35

METRICAS = [
    ("rank_medio",   "Rank medio obtenido",        "Eficiencia (Pareto)\nmenor = mejor",                ".2f"),
    ("equidad_qj",   r"corr(ingreso, $q_j$)",      "Equidad — calidad académica", ".3f"),
    ("sesgo_visual", r"corr(ingreso, $v_j^{vis}$)","Sesgo visual\n(positivo = ricos en mejores instalaciones)", ".3f"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

def get(mec, col, suffix):
    r = comp_df[comp_df["condicion"] == f"{mec}-{suffix}"].iloc[0]
    return float(r[col])

for col_j, (metric, ylabel, title, fmt) in enumerate(METRICAS):
    ax = axes[col_j]
    vals_b = [get(m, metric, "bias") for m in mecanismos]
    vals_t = [get(m, metric, "true") for m in mecanismos]
    b1 = ax.bar(x - w/2, vals_b, w, color=color_bias, edgecolor="white",
                label="Con sesgo visual")
    b2 = ax.bar(x + w/2, vals_t, w, color=color_true, edgecolor="white",
                label="Sin sesgo visual")
    ax.set_xticks(x); ax.set_xticklabels(mecanismos)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    all_v  = vals_b + vals_t
    margin = (max(all_v) - min(all_v)) * 0.22
    ax.set_ylim(min(all_v) - margin, max(all_v) + margin)
    if min(all_v) < 0 < max(all_v):
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    for bar in list(b1) + list(b2):
        v   = bar.get_height()
        off = margin * 0.15 if v >= 0 else -margin * 0.40
        ax.text(bar.get_x() + bar.get_width()/2, v + off,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
    if col_j == 2:
        ax.legend(fontsize=8, loc="lower left")

plt.suptitle(
    "Experimento sintético — Comparativa de mecanismos\n"
    "N=10,000 estudiantes | M=100 colegios | ratio=1.16x | "
    "Supuesto contrafactual: corr($v_j$, $q_j$) = 0",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
fig.savefig(FIG_DIR / "sinteticos_b_comparativa.png", bbox_inches="tight")
plt.close(fig)
log.info("  sinteticos_b_comparativa.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Resumen consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN COMPARATIVO")
log.info("=" * 60)

SISBEN_LABELS = {0: "A_extrema", 1: "B_moderada", 2: "C_vulnerable", 3: "D_no_prio"}
cols_main = ["condicion", "n_asignados", "rank_medio", "blocking_pairs",
             "equidad_qj", "sesgo_visual", "rechazo_total"]
cols_rec  = ["condicion"] + [f"rechazo_sisben_{l}" for l in SISBEN_LABELS.values()]

print(comp_df[cols_main].to_string(index=False))
print()
print(comp_df[cols_rec].to_string(index=False))

log.info("Done.")
