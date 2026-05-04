"""
08b_datos_sinteticos.py
=======================
Experimento completamente sintético de sesgo visual en elección escolar.
Familias y colegios distribuidos uniformemente en [0,1]² — aísla el efecto
puro del sesgo visual sin confundir con segregación residencial.

Solo dos inputs de datos reales:
  1. mu_q, std_q del puntaje Saber 11 (colegios_features_imputed.geojson, col q_j)
  2. Distribución de estratos 1-6 (familias_ubicadas.parquet, col estrato_real)

Modelo de utilidad
------------------
    U_bias(i,j) = q_j_std − alpha_i · log(1 + d_ij) + gamma_i · v_j + ε_ij
    U_true(i,j) = q_j_std − alpha_i · log(1 + d_ij)                  + ε_ij

Donde:
    alpha_s  = ALPHA_0 / s^GAMMA_POW   (Gallego & Hernando 2009)
    gamma_s  = 0.26 / s                (calibrado desde datos reales)
    v_j      generado con corr(v_j, q_j_std) = RHO controlada
             RHO=0 → escenario contrafactual: infraestructura y calidad académica
             son señales independientes. En los datos reales corr(v_j, q_j)≈0.30;
             el supuesto RHO=0 simula un contexto donde la inversión en
             infraestructura no se traduce en mejores resultados académicos —
             situación documentada en municipios de América Latina post-planes
             de infraestructura escolar (Neilson 2021, Duarte et al. 2012).
    ε_ij     ~ Gumbel(0, SIGMA)

Outputs
-------
  data/primary/sinteticos_b_colegios.parquet
  data/primary/sinteticos_b_estudiantes.parquet
  data/primary/sinteticos_b_preferencias_bias.parquet
  data/primary/sinteticos_b_preferencias_true.parquet
  reports/sinteticos_b_calibracion.json
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
COL_P   = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
FAM_P   = ROOT / "data" / "processed" / "familias_expandidas.parquet"   # tiene sisben_cat
OUT_P   = ROOT / "data" / "primary"
REP_DIR = ROOT / "reports"

# ─── Parámetros globales ──────────────────────────────────────────────────────
N_STUDENTS = 10_000
M_SCHOOLS  = 100
RATIO_D_O  = 1.16          # calibrado San Cristóbal
RHO        = 0.00          # contrafactual: v_j ⊥ q_j (datos reales = 0.30)
# Nota: RHO=0 simula el escenario de riesgo donde infraestructura y calidad
# académica están desacopladas. Justificación en docstring del módulo.
SIGMA      = 1.0
SEED       = 42
ALPHA_0    = 0.30
GAMMA_POW  = np.log(3) / np.log(6)   # ≈ 0.613

CAPACIDAD_TOTAL = round(N_STUDENTS / RATIO_D_O)   # ≈ 862

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar datos reales: mu_q, std_q  +  distribución de estratos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 — Cargando inputs reales...")

gdf   = gpd.read_file(COL_P)
q_all = gdf["q_j"].dropna().values.astype(float)
mu_q  = float(q_all.mean())
std_q = float(q_all.std())
log.info(f"  mu_q={mu_q:.4f}  std_q={std_q:.4f}  (n={len(q_all)} colegios reales)")

fam_df = pd.read_parquet(FAM_P)
fam_df["sisben_cat"] = pd.to_numeric(fam_df["sisben_cat"], errors="coerce").astype("Int64")
fam_df["N_ingpc"]    = pd.to_numeric(fam_df["N_ingpc"],    errors="coerce")

# Distribución por categoría SISBEN (0=A extrema, 1=B moderada, 2=C vulnerable, 3=D no priorizado)
vc     = fam_df["sisben_cat"].dropna().astype(int).value_counts()
sisben_counts = {int(c): int(n) for c, n in vc.items() if int(c) in range(4)}
total_fam     = sum(sisben_counts.values())
sisben_dist   = {c: sisben_counts[c] / total_fam for c in sorted(sisben_counts)}
SISBEN_LABELS = {0: "A (extrema)", 1: "B (moderada)", 2: "C (vulnerable)", 3: "D (no priorizado)"}

# Pool de ingresos reales por categoría SISBEN — para asignar N_ingpc a estudiantes sintéticos
ingpc_pool = {
    c: fam_df.loc[fam_df["sisben_cat"] == c, "N_ingpc"].dropna().values
    for c in range(4)
}

log.info(f"  Distribución SISBEN:")
for c, p in sisben_dist.items():
    log.info(f"    Cat {c} — {SISBEN_LABELS[c]}: {p:.2%}  | pool N_ingpc: {len(ingpc_pool[c]):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Generar M=50 colegios sintéticos
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Paso 2 — Generando M={M_SCHOOLS} colegios sintéticos...")

q_j     = rng.normal(mu_q, std_q, M_SCHOOLS)
q_j_std = (q_j - mu_q) / std_q

# v_j con correlación controlada RHO respecto a q_j_std
eta = rng.normal(0, 1, M_SCHOOLS)
v_j = RHO * q_j_std + np.sqrt(1 - RHO**2) * eta

# Capacidad base aleatoria, escalada para que sumen CAPACIDAD_TOTAL
cap_base = rng.integers(15, 40, M_SCHOOLS).astype(float)
cap_base = cap_base * CAPACIDAD_TOTAL / cap_base.sum()
cap_base = np.clip(np.round(cap_base).astype(int), 5, None)
# Ajuste fino del total por redondeo
diff = CAPACIDAD_TOTAL - cap_base.sum()
if diff != 0:
    idx = np.argmax(cap_base)
    cap_base[idx] += diff

coord_colegios = rng.uniform(size=(M_SCHOOLS, 2))

school_ids = [f"COL_{j:03d}" for j in range(M_SCHOOLS)]

colegios = pd.DataFrame({
    "id_establecimiento": school_ids,
    "q_j":               q_j,
    "q_j_std":           q_j_std,
    "v_j":               v_j,
    "capacidad_sintetica": cap_base,
    "coord_x":           coord_colegios[:, 0],
    "coord_y":           coord_colegios[:, 1],
})

corr_vq = float(np.corrcoef(v_j, q_j_std)[0, 1])
log.info(f"  corr(v_j, q_j_std) = {corr_vq:+.4f}  (objetivo RHO={RHO})")
log.info(f"  Capacidad total: {colegios['capacidad_sintetica'].sum()} cupos  (objetivo {CAPACIDAD_TOTAL})")
log.info(f"  Ratio cupos/estudiantes: {colegios['capacidad_sintetica'].sum() / N_STUDENTS:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros de utilidad (constantes, no calibrados con OLS)
# ─────────────────────────────────────────────────────────────────────────────
# alpha y gamma indexados por SISBEN cat (0-3), más vulnerable = mayor penalización distancia
# Mapeo SISBEN → proxy estrato: A≈E1, B≈E2, C≈E3, D≈E5
# alpha: Gallego & Hernando 2009 (escala por vulnerabilidad relativa)
# gamma: calibrado desde datos reales por estrato
_estrato_proxy = {0: 1, 1: 2, 2: 3, 3: 5}  # cat SISBEN → estrato proxy
alpha_s = {c: ALPHA_0 / (_estrato_proxy[c] ** GAMMA_POW) for c in range(4)}
gamma_s = {c: 1.0 / _estrato_proxy[c] for c in range(4)}   # γ₀=1: relación 1:1 v_j vs q_j para cat A

# ─────────────────────────────────────────────────────────────────────────────
# Función principal: generar datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
def generar_escenario(
    rng_local: np.random.Generator,
    estrato_arr: np.ndarray,
    colegios: pd.DataFrame,
) -> dict:
    """
    Genera las matrices de utilidad y rankings de preferencia.
    Familias y colegios distribuidos uniformemente en [0,1]².

    Parameters
    ----------
    rng_local : np.random.Generator
    estrato_arr : np.ndarray shape (N,)
    colegios : pd.DataFrame

    Returns
    -------
    dict con: estrato_arr, pref_bias_df, pref_true_df, dist_matrix,
              U_bias, U_true, coord_estudiantes, corr_v_bias, corr_v_true, delta
    """
    N = len(estrato_arr)
    M = len(colegios)
    school_arr     = colegios["id_establecimiento"].to_numpy(dtype=str)
    coord_colegios = colegios[["coord_x", "coord_y"]].values
    q_j_std = colegios["q_j_std"].values
    v_j     = colegios["v_j"].values

    coord_est   = rng_local.uniform(size=(N, 2))
    dist_matrix = cdist(coord_est, coord_colegios).astype(np.float32)
    log.info(f"  dist media={dist_matrix.mean():.4f}  max={dist_matrix.max():.4f}")

    alpha_i  = np.array([alpha_s[int(s)] for s in estrato_arr])
    gamma_i  = np.array([gamma_s[int(s)] for s in estrato_arr])
    eps      = rng_local.gumbel(0, SIGMA, size=(N, M))
    dist_pen = alpha_i[:, None] * np.log1p(dist_matrix)

    U_bias = q_j_std[None, :] - dist_pen + gamma_i[:, None] * v_j[None, :] + eps
    U_true = q_j_std[None, :] - dist_pen + eps

    pref_cols     = [f"pref_{k+1}" for k in range(M)]
    rank_bias_idx = np.argsort(-U_bias, axis=1)
    rank_true_idx = np.argsort(-U_true, axis=1)

    ids = pd.Index([f"S{i:04d}" for i in range(N)], name="id_estudiante")
    pref_bias_df = pd.DataFrame(school_arr[rank_bias_idx], index=ids, columns=pref_cols)
    pref_true_df = pd.DataFrame(school_arr[rank_true_idx], index=ids, columns=pref_cols)

    v_by_id = colegios.set_index("id_establecimiento")["v_j"]
    top1_b  = school_arr[np.argmax(U_bias, axis=1)]
    top1_t  = school_arr[np.argmax(U_true, axis=1)]
    corr_b  = float(np.corrcoef(estrato_arr, [v_by_id[s] for s in top1_b])[0, 1])
    corr_t  = float(np.corrcoef(estrato_arr, [v_by_id[s] for s in top1_t])[0, 1])
    delta   = corr_b - corr_t

    log.info(f"  corr(estrato,v_top1) bias={corr_b:+.4f}  true={corr_t:+.4f}  Δ={delta:+.4f}")

    return {
        "estrato_arr"       : estrato_arr,
        "pref_bias_df"      : pref_bias_df,
        "pref_true_df"      : pref_true_df,
        "dist_matrix"       : dist_matrix,
        "U_bias"            : U_bias,
        "U_true"            : U_true,
        "coord_estudiantes" : coord_est,
        "corr_v_bias"       : corr_b,
        "corr_v_true"       : corr_t,
        "delta"             : delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generar estudiantes (compartidos entre escenarios — mismos estratos)
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Paso 3 — Generando N={N_STUDENTS} estudiantes (grupos SISBEN + ingreso real)...")

cats  = sorted(sisben_dist.keys())
probs = np.array([sisben_dist[c] for c in cats])
probs = probs / probs.sum()
sisben_arr  = rng.choice(cats, size=N_STUDENTS, p=probs)  # 0,1,2,3
estrato_arr = sisben_arr  # alias para compatibilidad

# Asignar ingreso per cápita real sorteando del pool empírico por categoría
ingpc_arr = np.zeros(N_STUDENTS, dtype=float)
for c in cats:
    mask = sisben_arr == c
    pool = ingpc_pool[c]
    if len(pool) > 0:
        ingpc_arr[mask] = rng.choice(pool, size=mask.sum(), replace=True)
    else:
        ingpc_arr[mask] = np.nan

for c, count in zip(*np.unique(sisben_arr, return_counts=True)):
    log.info(f"  Cat {c} ({SISBEN_LABELS[c]}): {count} ({count/N_STUDENTS:.1%}) | "
             f"ingreso medio = ${ingpc_arr[sisben_arr==c].mean():,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Generar datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 4 — Generando datos sintéticos...")

rng_main = np.random.default_rng(SEED + 1)
esc = generar_escenario(rng_main, estrato_arr, colegios)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Guardar outputs
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 5 — Guardando outputs...")

colegios.to_parquet(OUT_P / "sinteticos_b_colegios.parquet", index=False)

est_df = pd.DataFrame({
    "id_estudiante": [f"S{i:04d}" for i in range(N_STUDENTS)],
    "estrato":       sisben_arr,      # sisben_cat 0-3
    "N_ingpc":       ingpc_arr,       # ingreso per cápita real sorteado del pool empírico
    "coord_x":       esc["coord_estudiantes"][:, 0],
    "coord_y":       esc["coord_estudiantes"][:, 1],
})
est_df.to_parquet(OUT_P / "sinteticos_b_estudiantes.parquet", index=False)
esc["pref_bias_df"].to_parquet(OUT_P / "sinteticos_b_preferencias_bias.parquet")
esc["pref_true_df"].to_parquet(OUT_P / "sinteticos_b_preferencias_true.parquet")

cal = {
    "N_estudiantes"   : N_STUDENTS,
    "M_colegios"      : M_SCHOOLS,
    "ratio_d_o"       : RATIO_D_O,
    "capacidad_total" : int(colegios["capacidad_sintetica"].sum()),
    "rho"             : RHO,
    "sigma"           : SIGMA,
    "seed"            : SEED,
    "alpha_0"         : ALPHA_0,
    "gamma_pow"       : round(GAMMA_POW, 6),
    "mu_q"            : round(mu_q, 6),
    "std_q"           : round(std_q, 6),
    "sisben_dist"     : {str(c): round(p, 5) for c, p in sisben_dist.items()},
    "alpha_s"         : {str(s): round(a, 6) for s, a in alpha_s.items()},
    "gamma_s"         : {str(s): round(g, 6) for s, g in gamma_s.items()},
    "corr_vj_qj"      : round(float(np.corrcoef(colegios["v_j"], colegios["q_j_std"])[0,1]), 5),
    "utilidad"        : "U_bias = q_j_std - alpha_s*log1p(d) + gamma_s*v_j + eps | gamma_s=1.0/proxy_estrato (γ₀=1: relación 1:1 v_j/q_j para cat A)",
    "corr_v_bias"     : round(esc["corr_v_bias"], 5),
    "corr_v_true"     : round(esc["corr_v_true"], 5),
    "delta"           : round(esc["delta"], 5),
}

with open(REP_DIR / "sinteticos_b_calibracion.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=2, ensure_ascii=False)

log.info("  sinteticos_b_colegios.parquet")
log.info("  sinteticos_b_estudiantes.parquet")
log.info("  sinteticos_b_preferencias_bias.parquet")
log.info("  sinteticos_b_preferencias_true.parquet")
log.info("  sinteticos_b_calibracion.json")
log.info("Done.")
