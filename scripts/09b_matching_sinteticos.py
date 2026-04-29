"""
09b_matching_sinteticos.py
==========================
Aplica tres mecanismos de asignación sobre los datos sintéticos del escenario A
(sin segregación espacial) generados por 08b_datos_sinteticos.py.

Mecanismos comparados
---------------------
1. BM   — Boston Mechanism (ineficiente, inestable, no strategy-proof)
2. DA   — Deferred Acceptance puro distancia (estable, strategy-proof)
3. SED  — DA con prioridad lexicográfica: estratos 1-2 primero, distancia como
           desempate (Resolución 1587/2025 Bogotá estilizada)

Para cada mecanismo se corre con dos sets de preferencias:
  - bias : preferencias que incluyen sesgo visual (U_bias)
  - true : preferencias verdaderas sin sesgo visual (U_true)

Esto produce 6 condiciones: BM-bias, BM-true, DA-bias, DA-true, SED-bias, SED-true.

La comparación bias vs true dentro de cada mecanismo aísla el efecto causal
del sesgo visual sobre la equidad de la asignación.
La comparación entre mecanismos (con mismas preferencias) muestra qué mecanismo
mitiga mejor el daño del sesgo visual.

Prioridad en los mecanismos
---------------------------
DA y SED usan lotería como prioridad base (equidad ex-ante).
SED añade la capa lexicográfica: estratos 1-2 tienen prioridad sobre 3-6,
dentro de cada categoría desempata por lotería.

Inputs
------
  data/primary/sinteticos_b_colegios.parquet
  data/primary/sinteticos_b_estudiantes.parquet
  data/primary/sinteticos_b_preferencias_bias.parquet
  data/primary/sinteticos_b_preferencias_true.parquet

Outputs
-------
  data/results/sinteticos_b_resultados.parquet   — asignaciones de las 6 condiciones
  reports/sinteticos_b_comparativa.csv           — métricas comparativas
  reports/figures/matching/sinteticos_b_comparativa.png
"""

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

ROOT    = Path(__file__).resolve().parent.parent
IN_DIR  = ROOT / "data" / "primary"
OUT_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
CAT_OFFSET  = 1_000_000   # garantiza que categoría lexicográfica domine sobre lotería

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

# Cargar los dos escenarios
ESCENARIOS = {}
for lbl in ["A", "B"]:
    est  = pd.read_parquet(IN_DIR / f"sinteticos_b_estudiantes_{lbl}.parquet")
    pb   = pd.read_parquet(IN_DIR / f"sinteticos_b_preferencias_bias_{lbl}.parquet")
    pt   = pd.read_parquet(IN_DIR / f"sinteticos_b_preferencias_true_{lbl}.parquet")
    ESCENARIOS[lbl] = {"est_df": est, "pref_bias": pb, "pref_true": pt}

N = len(ESCENARIOS["A"]["est_df"])
log.info(f"  N={N} | M={M} | Cupos={sum(school_cap.values())} | Ratio={sum(school_cap.values())/N:.3f}x")
log.info(f"  Escenarios: A (sin segregación) | B (clusters sigma=0.15)")

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

# ─────────────────────────────────────────────────────────────────────────────
# 3. Lotería de prioridad (equidad ex-ante, compartida entre escenarios)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 3 — Generando lotería de prioridad (seed=42)...")

rng = np.random.default_rng(SEED)
lottery = {}
for sid in school_ids:
    perm = rng.permutation(N)
    lottery[sid] = {int(idx): int(rank) for rank, idx in enumerate(perm)}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Correr 6 condiciones × 2 escenarios
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 4 — Ejecutando 6 condiciones × 2 escenarios...")

all_results = {}   # lbl → DataFrame de métricas
all_rows    = []   # para parquet de resultados

for lbl, esc_data in ESCENARIOS.items():
    log.info(f"\n--- Escenario {lbl} ---")
    est_df_lbl  = esc_data["est_df"]
    estrato_arr = est_df_lbl["estrato"].values
    lista_bias  = to_pref_lists(esc_data["pref_bias"])
    lista_true  = to_pref_lists(esc_data["pref_true"])

    # cat para SED: 0=E1-E2 (prioridad alta), 1=E3-E6
    categoria = (estrato_arr > 2).astype(int)

    def priority_lottery(i, sid, _cat=None):
        return lottery[sid][i]

    def priority_sed(i, sid, _cat=categoria):
        return int(_cat[i]) * CAT_OFFSET + lottery[sid][i]

    CONDITIONS = [
        ("BM-bias",  boston_mechanism,    lista_bias, priority_lottery),
        ("BM-true",  boston_mechanism,    lista_true, priority_lottery),
        ("DA-bias",  deferred_acceptance, lista_bias, priority_lottery),
        ("DA-true",  deferred_acceptance, lista_true, priority_lottery),
        ("SED-bias", deferred_acceptance, lista_bias, priority_sed),
        ("SED-true", deferred_acceptance, lista_true, priority_sed),
    ]

    results_lbl = []
    for cond_label, fn, plists, pfn in CONDITIONS:
        log.info(f"  [{lbl}] {cond_label}...")
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
        met["escenario"]  = lbl
        for s in range(1, 7):
            met.pop(f"aj_estrato_{s}", None)
        results_lbl.append(met)

        for i, a in enumerate(asgn):
            all_rows.append({
                "escenario": lbl, "condicion": cond_label,
                "id_estudiante": est_df_lbl.iloc[i]["id_estudiante"],
                "estrato": int(estrato_arr[i]),
                "id_establecimiento": a,
            })

    comp_lbl = pd.DataFrame(results_lbl)
    all_results[lbl] = comp_lbl

    log.info(f"\n  Efecto sesgo visual — Escenario {lbl}:")
    for mec in ["BM", "DA", "SED"]:
        b = comp_lbl[comp_lbl["condicion"] == f"{mec}-bias"].iloc[0]
        t = comp_lbl[comp_lbl["condicion"] == f"{mec}-true"].iloc[0]
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

all_comp = pd.concat(all_results.values(), ignore_index=True)
all_comp.to_csv(REP_DIR / "sinteticos_b_comparativa.csv", index=False)
log.info("  sinteticos_b_resultados.parquet")
log.info("  sinteticos_b_comparativa.csv")

comp_df = all_comp  # alias para la figura

# ─────────────────────────────────────────────────────────────────────────────
# 7. Figura comparativa
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 6 — Generando figura comparativa A vs B...")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "figure.dpi": 150,
})

mecanismos  = ["BM", "DA", "SED"]
color_bias  = "#4292c6"
color_true  = "#bdd7e7"
x = np.arange(len(mecanismos))
w = 0.35

TITULOS_ESC = {
    "A": "Escenario A — Sin segregación espacial",
    "B": "Escenario B — Con segregación espacial (σ=0.10)",
}
METRICAS = [
    ("rank_medio",   "Rank medio obtenido",        "Eficiencia (Pareto)\nmenor = mejor",              ".2f"),
    ("equidad_qj",   "corr(estrato, $q_j$)",        "Equidad — calidad académica\nmenor = mejor",      ".3f"),
    ("sesgo_visual", "corr(estrato, $v_j$)",         "Sesgo visual\nmenor = menos sesgo",               ".3f"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 10))

for row_i, lbl in enumerate(["A", "B"]):
    df_esc = all_results[lbl]

    def get(mec, col, suffix):
        r = df_esc[df_esc["condicion"] == f"{mec}-{suffix}"].iloc[0]
        return float(r[col])

    for col_j, (metric, ylabel, title, fmt) in enumerate(METRICAS):
        ax = axes[row_i, col_j]
        vals_b = [get(m, metric, "bias") for m in mecanismos]
        vals_t = [get(m, metric, "true") for m in mecanismos]
        b1 = ax.bar(x - w/2, vals_b, w, color=color_bias, edgecolor="white",
                    label="Con sesgo visual")
        b2 = ax.bar(x + w/2, vals_t, w, color=color_true, edgecolor="white",
                    label="Sin sesgo visual")
        ax.set_xticks(x); ax.set_xticklabels(mecanismos)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        all_v = vals_b + vals_t
        margin = (max(all_v) - min(all_v)) * 0.22
        ax.set_ylim(min(all_v) - margin, max(all_v) + margin)
        if min(all_v) < 0 < max(all_v):
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        for bar in list(b1) + list(b2):
            v = bar.get_height()
            off = margin * 0.15 if v >= 0 else -margin * 0.40
            ax.text(bar.get_x() + bar.get_width()/2, v + off,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)
        if col_j == 2:
            ax.legend(fontsize=8, loc="lower left")

    # Etiqueta de fila
    axes[row_i, 0].annotate(
        TITULOS_ESC[lbl], xy=(0, 0.5), xytext=(-60, 0),
        xycoords="axes fraction", textcoords="offset points",
        fontsize=9, fontweight="bold", rotation=90, va="center",
    )

plt.suptitle("Experimento sintético — Comparativa de mecanismos\n"
             "N=1,000 estudiantes | M=50 colegios | ratio=1.16x | "
             "Supuesto contrafactual: corr($v_j$, $q_j$) = 0",
             fontsize=11, fontweight="bold")
plt.tight_layout(rect=[0.04, 0, 1, 0.96])
fig.savefig(FIG_DIR / "sinteticos_b_comparativa.png", bbox_inches="tight")
plt.close(fig)
log.info("  sinteticos_b_comparativa.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Resumen consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN COMPARATIVO")
log.info("=" * 60)

cols_main = ["escenario", "condicion", "n_asignados", "rank_medio", "blocking_pairs",
             "equidad_qj", "sesgo_visual", "rechazo_total"]
cols_rec  = ["escenario", "condicion"] + [f"rechazo_estrato_{s}" for s in range(1, 7)]

for lbl in ["A", "B"]:
    print(f"\n=== Escenario {lbl} ===")
    df = all_results[lbl]
    print(df[cols_main].to_string(index=False))
    print()
    print(df[cols_rec].to_string(index=False))

log.info("Done.")
