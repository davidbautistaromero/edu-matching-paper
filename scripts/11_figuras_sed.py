"""
11_figuras_sed.py
=================
Genera las 3 figuras comparativas del mecanismo SED Bogotá:
  1. sed_tradeoff.png            curvas de eficiencia y equidad vs w_estrato
  2. todos_mecanismos.png        comparación BM, DA y SED en sus 5 pesos
  3. sed_calidad_por_estrato.png distribución de calidad por estrato (extremos)

Inputs:
  reports/matching_sed_comparison.csv
  reports/matching_comparison.csv
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
REP_DIR = ROOT / "reports"
FIG_DIR = REP_DIR / "figures" / "matching"
FIG_DIR.mkdir(parents=True, exist_ok=True)

comp = pd.read_csv(REP_DIR / "matching_sed_comparison.csv")
mech = pd.read_csv(REP_DIR / "matching_comparison.csv")

log.info("Figura 1: trade-off SED")

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(comp["w_estrato"], comp["eficiencia_q"], "o-",
             color="#1565C0", linewidth=2, markersize=9)
axes[0].set_ylabel("Calidad media asignada (q)")
axes[0].set_title("Eficiencia")
axes[0].grid(alpha=0.3)
axes[0].ticklabel_format(useOffset=False, axis="y")

axes[1].plot(comp["w_estrato"], comp["equidad_corr"], "s-",
             color="#E53935", linewidth=2, markersize=9)
axes[1].set_xlabel("Peso del estrato en la prioridad (w_estrato)")
axes[1].set_ylabel("corr(estrato, q)")
axes[1].set_title("Inequidad de acceso")
axes[1].grid(alpha=0.3)
axes[1].ticklabel_format(useOffset=False, axis="y")

fig.suptitle("Mecanismo SED, sensibilidad sobre w_estrato", y=1.00)
fig.tight_layout()
fig.savefig(FIG_DIR / "sed_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close(fig)

log.info("Figura 2: comparación de mecanismos")

all_mech = pd.concat([
    mech[["condicion", "eficiencia_q", "equidad_corr", "sesgo_visual", "blocking_pairs"]],
    comp[["condicion", "eficiencia_q", "equidad_corr", "sesgo_visual", "blocking_pairs"]],
]).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.arange(len(all_mech))
colors = ["#FF9800", "#FF9800"] + ["#1976D2"] * 5

axes[0].bar(x, all_mech["eficiencia_q"], color=colors, alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(all_mech["condicion"], rotation=35, ha="right")
axes[0].set_ylabel("q media asignada")
axes[0].set_title("Eficiencia")
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(258.65, 258.80)

axes[1].bar(x, all_mech["equidad_corr"].abs(), color=colors, alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(all_mech["condicion"], rotation=35, ha="right")
axes[1].set_ylabel("|corr(estrato, q)|")
axes[1].set_title("Inequidad de acceso (mejor si baja)")
axes[1].grid(axis="y", alpha=0.3)

axes[2].bar(x, all_mech["blocking_pairs"], color=colors, alpha=0.85)
axes[2].set_xticks(x)
axes[2].set_xticklabels(all_mech["condicion"], rotation=35, ha="right")
axes[2].set_ylabel("Blocking pairs")
axes[2].set_title("Inestabilidad (mejor si baja)")
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("Bogotá, comparación de mecanismos sobre datos reales", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "todos_mecanismos.png", dpi=150, bbox_inches="tight")
plt.close(fig)

log.info("Figura 3: calidad por estrato, extremos")

estratos = list(range(1, 7))
q_w000 = [comp[comp["condicion"] == "SED-w000"][f"q_estrato_{s}"].iloc[0] for s in estratos]
q_w100 = [comp[comp["condicion"] == "SED-w100"][f"q_estrato_{s}"].iloc[0] for s in estratos]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(estratos))
w_bar = 0.4

ax.bar(x - w_bar/2, q_w000, w_bar, label="w_estrato=0 (puro distancia)",
       color="#1976D2", alpha=0.85)
ax.bar(x + w_bar/2, q_w100, w_bar, label="w_estrato=1 (puro estrato)",
       color="#7B1FA2", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"Estrato {s}" for s in estratos])
ax.set_ylabel("q media del colegio asignado")
ax.set_title("Distribución de calidad por estrato, extremos del peso SED")
ax.set_ylim(min(min(q_w000), min(q_w100)) - 1, max(max(q_w000), max(q_w100)) + 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "sed_calidad_por_estrato.png", dpi=150, bbox_inches="tight")
plt.close(fig)

log.info("Done.")
