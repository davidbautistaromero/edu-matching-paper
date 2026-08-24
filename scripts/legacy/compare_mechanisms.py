"""
compare_mechanisms.py
=====================
Genera figura y tabla comparativa de los 3 mecanismos de asignación:
  - BM     : Boston Mechanism
  - DA     : Deferred Acceptance puro distancia
  - SED-lex: DA con prioridad lexicográfica Resolución 1587/2025

Grupos vulnerables definidos como grupos A-B (hogares de menores ingresos).
  La tasa de rechazo vulnerable = fracción de hogares grupos A-B sin asignación.

Inputs
------
  data/results/matching_bm.parquet
  data/results/matching_da.parquet
  data/results/matching_sed_lex.parquet
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

ROOT    = Path(__file__).resolve().parents[2]
REP_DIR = ROOT / "reports"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar métricas agregadas
# ─────────────────────────────────────────────────────────────────────────────
bm_row  = pd.read_csv(REP_DIR / "matching_bm_summary.csv").query("estrato == 'TOTAL'").iloc[0]
da_row  = pd.read_csv(REP_DIR / "matching_da_summary.csv").query("estrato == 'TOTAL'").iloc[0]
sed_df  = pd.read_csv(REP_DIR / "matching_sed_comparison.csv")
sed_row = sed_df[sed_df["condicion"] == "SED-lex"].iloc[0]

mecanismos = ["BM", "DA", "SED-lex"]

def g(row, col):
    return float(row[col]) if col in row.index else np.nan

data = {
    "rank_medio"    : [g(bm_row, "rank_medio"),     g(da_row, "rank_medio"),     g(sed_row, "rank_medio")],
    "blocking_pairs": [g(bm_row, "blocking_pairs"),  g(da_row, "blocking_pairs"),  g(sed_row, "blocking_pairs")],
    "equidad_aj"    : [g(bm_row, "equidad_aj"),      g(da_row, "equidad_aj"),      g(sed_row, "equidad_aj")],
    "sesgo_visual"  : [g(bm_row, "sesgo_visual"),    g(da_row, "sesgo_visual"),    g(sed_row, "sesgo_visual")],
    "rechazo_total" : [g(bm_row, "rechazo_total"),   g(da_row, "rechazo_total"),   g(sed_row, "rechazo_total")],
    "n_asignados"   : [g(bm_row, "n_asignados"),     g(da_row, "n_asignados"),     g(sed_row, "n_asignados")],
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Rechazo hogares vulnerables — calculado desde los parquets de matching
# ─────────────────────────────────────────────────────────────────────────────
def rechazo_vulnerable(parquet_path: Path) -> float:
    """Fracción de hogares SISBEN A+B (sisben_cat 0 o 1) sin asignación."""
    if not parquet_path.exists():
        return np.nan
    df = pd.read_parquet(parquet_path)
    if "sisben_cat" not in df.columns:
        return np.nan
    vulnerable = pd.to_numeric(df["sisben_cat"], errors="coerce") <= 1
    sub = df[vulnerable]
    if len(sub) == 0:
        return np.nan
    return sub["id_establecimiento"].isna().mean()

def equidad_qj(parquet_path: Path) -> float:
    """Correlación Pearson entre sisben_cat y q_j del colegio asignado.
    sisben_cat: 0=A (más pobre) … 3=D (menos pobre), así que
    corr > 0 significa que hogares más acomodados obtienen mayor q_j."""
    if not parquet_path.exists():
        return np.nan
    df = pd.read_parquet(parquet_path)
    if "q_j" not in df.columns:
        return np.nan
    # Solo asignados
    df = df[df["id_establecimiento"].notna()].copy()
    if len(df) < 10:
        return np.nan
    # Usar sisben_cat si existe, si no estrato_real
    if "sisben_cat" in df.columns:
        proxy = pd.to_numeric(df["sisben_cat"], errors="coerce")
    elif "estrato_real" in df.columns:
        proxy = pd.to_numeric(df["estrato_real"], errors="coerce")
    else:
        return np.nan
    qj    = pd.to_numeric(df["q_j"], errors="coerce")
    valid = proxy.notna() & qj.notna()
    if valid.sum() < 10:
        return np.nan
    return float(np.corrcoef(proxy[valid], qj[valid])[0, 1])

OUT_DIR = ROOT / "data" / "results"
rv = [
    rechazo_vulnerable(OUT_DIR / "matching_bm.parquet"),
    rechazo_vulnerable(OUT_DIR / "matching_da.parquet"),
    rechazo_vulnerable(OUT_DIR / "matching_sed_lex.parquet"),
]
data["rechazo_vulnerable"] = rv

data["equidad_qj"] = [
    equidad_qj(OUT_DIR / "matching_bm.parquet"),
    equidad_qj(OUT_DIR / "matching_da.parquet"),
    equidad_qj(OUT_DIR / "matching_sed_lex.parquet"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Tabla resumen
# ─────────────────────────────────────────────────────────────────────────────
tabla = pd.DataFrame({
    "Mecanismo"                      : mecanismos,
    "Asignados"                      : [int(v) for v in data["n_asignados"]],
    "Rank medio"                     : [round(v, 3) for v in data["rank_medio"]],
    "Blocking pairs"                 : [int(v) for v in data["blocking_pairs"]],
    "Equidad corr(ingreso, q_j)"     : [round(v, 4) if not np.isnan(v) else np.nan for v in data["equidad_qj"]],
    "Sesgo visual corr(ingreso, v_j)": [round(v, 2) for v in data["sesgo_visual"]],
    "Rechazo grupos A-B (%)"          : [f"{v:.1%}" if not np.isnan(v) else "—" for v in data["rechazo_vulnerable"]],
    "Rechazo total (%)"              : [f"{v:.1%}" for v in data["rechazo_total"]],
})

tabla.to_csv(REP_DIR / "comparativa_mecanismos.csv", index=False)
print("\nTabla comparativa:")
print(tabla.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Figura — 4 paneles
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

# Panel (a): rank medio — eficiencia
ax = axes[0]
bars = ax.bar(x, data["rank_medio"], width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel("Rank medio obtenido")
ax.set_title("(a) Eficiencia (Pareto)\nmenor = mejor", loc="left")
ax.set_ylim(0, max(data["rank_medio"]) * 1.25)
for bar, v in zip(bars, data["rank_medio"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{v:.2f}", ha="center", va="bottom", fontsize=9)

# Panel (b): equidad — corr(ingreso, q_j)
ax = axes[1]
vals_b = [v if not np.isnan(v) else 0 for v in data["equidad_qj"]]
bars = ax.bar(x, vals_b, width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel(r"corr(ingreso, $q_j$)")
ax.set_title("(b) Equidad — calidad académica asignada\nmayor = más inequitativo", loc="left")
ymin = min(vals_b) * 1.10 if min(vals_b) < 0 else min(vals_b) * 0.90
ymax = max(vals_b) * 1.10
ax.set_ylim(ymin, ymax)
ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, v in zip(bars, data["equidad_qj"]):
    label = f"{v:.3f}" if not np.isnan(v) else "—"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (ymax - ymin) * 0.01,
            label, ha="center", va="bottom", fontsize=9)

# Panel (c): sesgo visual — corr(ingreso, v_j)
ax = axes[2]
bars = ax.bar(x, data["sesgo_visual"], width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel(r"corr(ingreso, $v_j$)")
ax.set_title("(c) Sesgo visual\nmenor = menos sesgo", loc="left")
ymin = min(data["sesgo_visual"]) * 0.5
ymax = max(data["sesgo_visual"]) * 1.10
ax.set_ylim(ymin, ymax)
for bar, v in zip(bars, data["sesgo_visual"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ymax - ymin) * 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# Panel (d): tasa de rechazo hogares vulnerables
ax = axes[3]
rv_pct = [v * 100 if not np.isnan(v) else 0 for v in data["rechazo_vulnerable"]]
bars = ax.bar(x, rv_pct, width=w, color=colors, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(mecanismos)
ax.set_ylabel("Tasa de rechazo (%)")
ax.set_title("(d) Rechazo grupos A–B\n(proxy vulnerabilidad)", loc="left")
ymax = max(rv_pct) * 1.3 if max(rv_pct) > 0 else 5
ax.set_ylim(0, ymax)
for bar, v, rv_raw in zip(bars, rv_pct, data["rechazo_vulnerable"]):
    label = f"{rv_raw:.1%}" if not np.isnan(rv_raw) else "—"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.01,
            label, ha="center", va="bottom", fontsize=9)

plt.suptitle(
    "Comparativa de mecanismos de asignación escolar — Bogotá\n"
    "Grupos vulnerables: grupos A–B (hogares de menores ingresos)",
    fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
fig.savefig(FIG_DIR / "comparativa_mecanismos.png", bbox_inches="tight")
plt.close(fig)

print(f"\nFigura: reports/figures/matching/comparativa_mecanismos.png")
print(f"Tabla:  reports/comparativa_mecanismos.csv")
