"""
compare_mechanisms.py
=====================
Genera figura y tabla comparativa de los 3 mecanismos de asignación:
  - BM     : Boston Mechanism
  - DA     : Deferred Acceptance puro distancia
  - SED-lex: DA con prioridad lexicográfica Resolución 1587/2025

Inputs
------
  reports/matching_bm_summary.csv
  reports/matching_da_summary.csv
  reports/matching_sed_comparison.csv

Outputs
-------
  reports/figures/matching/comparativa_mecanismos.png
  reports/comparativa_mecanismos.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = Path(__file__).resolve().parent.parent
REP_DIR = ROOT / "reports"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar métricas de cada mecanismo
# ─────────────────────────────────────────────────────────────────────────────
bm_row  = pd.read_csv(REP_DIR / "matching_bm_summary.csv").query("estrato == 'TOTAL'").iloc[0]
da_row  = pd.read_csv(REP_DIR / "matching_da_summary.csv").query("estrato == 'TOTAL'").iloc[0]
sed_df  = pd.read_csv(REP_DIR / "matching_sed_comparison.csv")
sed_row = sed_df[sed_df["condicion"] == "SED-lex"].iloc[0]

mecanismos = ["BM", "DA", "SED-lex"]

def g(row, col):
    return float(row[col]) if col in row.index else np.nan

data = {
    "rank_medio"      : [g(bm_row, "rank_medio"),      g(da_row, "rank_medio"),      g(sed_row, "rank_medio")],
    "blocking_pairs"  : [g(bm_row, "blocking_pairs"),  g(da_row, "blocking_pairs"),  g(sed_row, "blocking_pairs")],
    "equidad_aj"      : [g(bm_row, "equidad_aj"),       g(da_row, "equidad_aj"),       g(sed_row, "equidad_aj")],
    "sesgo_visual"    : [g(bm_row, "sesgo_visual"),     g(da_row, "sesgo_visual"),     g(sed_row, "sesgo_visual")],
    "rechazo_e1"      : [g(bm_row, "rechazo_estrato_1"),g(da_row, "rechazo_estrato_1"),g(sed_row, "rechazo_estrato_1")],
    "rechazo_e2"      : [g(bm_row, "rechazo_estrato_2"),g(da_row, "rechazo_estrato_2"),g(sed_row, "rechazo_estrato_2")],
    "rechazo_total"   : [g(bm_row, "rechazo_total"),    g(da_row, "rechazo_total"),    g(sed_row, "rechazo_total")],
    "n_asignados"     : [g(bm_row, "n_asignados"),      g(da_row, "n_asignados"),      g(sed_row, "n_asignados")],
}

# Rechazo promedio E1+E2 (estratos vulnerables)
data["rechazo_vulnerable"] = [
    np.nanmean([data["rechazo_e1"][i], data["rechazo_e2"][i]])
    for i in range(3)
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Tabla resumen
# ─────────────────────────────────────────────────────────────────────────────
tabla = pd.DataFrame({
    "Mecanismo"              : mecanismos,
    "Asignados"              : [int(v) for v in data["n_asignados"]],
    "Rank medio"             : [round(v, 3) for v in data["rank_medio"]],
    "Blocking pairs"         : [int(v) for v in data["blocking_pairs"]],
    "Equidad corr(E, a_j)"   : [round(v, 4) for v in data["equidad_aj"]],
    "Sesgo visual corr"      : [round(v, 4) for v in data["sesgo_visual"]],
    "Rechazo E1-2 (%)"       : [f"{v:.1%}" for v in data["rechazo_vulnerable"]],
    "Rechazo total (%)"      : [f"{v:.1%}" for v in data["rechazo_total"]],
})

tabla.to_csv(REP_DIR / "comparativa_mecanismos.csv", index=False)
print("\nTabla comparativa:")
print(tabla.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Figura — 4 paneles
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "figure.dpi": 150,
})

colors = ["#4292c6", "#969696", "#fd8d3c"]
x = np.arange(len(mecanismos))
w = 0.55

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Panel 1: rank medio — eficiencia Pareto (menor es mejor)
ax = axes[0]
bars = ax.bar(x, data["rank_medio"], width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel("Rank medio obtenido")
ax.set_title("Eficiencia (Pareto)\nmenor = mejor")
ax.set_ylim(0, max(data["rank_medio"]) * 1.25)
for bar, v in zip(bars, data["rank_medio"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{v:.2f}", ha="center", va="bottom", fontsize=9)

# Panel 2: equidad — corr(estrato, a_j) (menor es mejor)
ax = axes[1]
bars = ax.bar(x, data["equidad_aj"], width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel("corr(estrato, $a_j$)")
ax.set_title("Equidad — atractivo asignado\nmenor = más equitativo")
ymin = min(data["equidad_aj"]) * 0.90
ymax = max(data["equidad_aj"]) * 1.10
ax.set_ylim(ymin, ymax)
for bar, v in zip(bars, data["equidad_aj"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ymax - ymin) * 0.01,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)

# Panel 3: sesgo visual — corr(estrato, a_j_visual) (menor es mejor)
ax = axes[2]
bars = ax.bar(x, data["sesgo_visual"], width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel("corr(estrato, $a_j^{\\mathrm{visual}}$)")
ax.set_title("Sesgo visual\nmenor = menos sesgo")
ymin = min(data["sesgo_visual"]) * 0.90
ymax = max(data["sesgo_visual"]) * 1.10
ax.set_ylim(ymin, ymax)
for bar, v in zip(bars, data["sesgo_visual"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ymax - ymin) * 0.01,
            f"{v:.4f}", ha="center", va="bottom", fontsize=9)

# Panel 4: tasa de rechazo estratos vulnerables E1-E2 (menor es mejor)
ax = axes[3]
bars = ax.bar(x, [v * 100 for v in data["rechazo_vulnerable"]],
              width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel("Tasa de rechazo (%)")
ax.set_title("Rechazo estratos vulnerables (E1-E2)\nmenor = más inclusivo")
ymax = max(v * 100 for v in data["rechazo_vulnerable"]) * 1.3
ax.set_ylim(0, ymax)
for bar, v in zip(bars, data["rechazo_vulnerable"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.01,
            f"{v:.1%}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Comparativa de mecanismos de asignación escolar — Bogotá\n"
             "Población de primer ingreso (99,890 familias expandidas)",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "comparativa_mecanismos.png", bbox_inches="tight")
plt.close(fig)

print(f"\nFigura: reports/figures/matching/comparativa_mecanismos.png")
print(f"Tabla:  reports/comparativa_mecanismos.csv")
