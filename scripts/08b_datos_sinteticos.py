"""
08b_datos_sinteticos.py
=======================
Experimento completamente sintético de sesgo visual en elección escolar.
Genera dos escenarios de segregación espacial para el análisis de sensibilidad
del efecto del sesgo visual.

Solo dos inputs de datos reales:
  1. mu_q, std_q del puntaje Saber 11 (colegios_features_imputed.geojson, col q_j)
  2. Distribución de estratos 1-6 (familias_ubicadas.parquet, col estrato_real)

Modelo de utilidad
------------------
    U_bias(i,j) = q_j_std − alpha_i · log(1 + d_ij) + gamma_i · v_j + ε_ij
    U_true(i,j) = q_j_std − alpha_i · log(1 + d_ij)                  + ε_ij

Donde:
    alpha_s  = ALPHA_0 / s^GAMMA_POW   (Gallego & Hernando 2009)
    gamma_s  = 0.30 / s                (Neilson 2021)
    v_j      generado con corr(v_j, q_j_std) = RHO controlada
             RHO=0 → escenario contrafactual: infraestructura y calidad académica
             son señales independientes. En los datos reales corr(v_j, q_j)≈0.30;
             el supuesto RHO=0 simula un contexto donde la inversión en
             infraestructura no se traduce en mejores resultados académicos —
             situación documentada en municipios de América Latina post-planes
             de infraestructura escolar (Neilson 2021, Duarte et al. 2012).
    ε_ij     ~ Gumbel(0, SIGMA)

Escenarios de segregación espacial
-----------------------------------
A — Sin segregación (sigma_cluster=None):
    Familias y colegios en posiciones aleatorias uniformes [0,1]².
    Aísla el efecto puro del sesgo visual sin confundir con segregación residencial.

B — Con segregación (sigma_cluster=0.15):
    Familias agrupadas en clusters por estrato (E1-E2 en zona sur, E5-E6 en zona
    norte). Colegios distribuidos aleatoriamente — sin correlación con estratos.
    Muestra cómo el sesgo visual se amplifica con segregación espacial preexistente.

Función pública
---------------
    generar_escenario(sigma_cluster, rng, ...) -> dict
    Usada por 09b_matching_sinteticos.py para generar ambos escenarios.

Outputs (cuando se corre directamente)
-------
  data/primary/sinteticos_b_colegios.parquet
  data/primary/sinteticos_b_estudiantes_A.parquet
  data/primary/sinteticos_b_estudiantes_B.parquet
  data/primary/sinteticos_b_preferencias_bias_A.parquet
  data/primary/sinteticos_b_preferencias_true_A.parquet
  data/primary/sinteticos_b_preferencias_bias_B.parquet
  data/primary/sinteticos_b_preferencias_true_B.parquet
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
FAM_P   = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
OUT_P   = ROOT / "data" / "primary"
REP_DIR = ROOT / "reports"

# ─── Parámetros globales ──────────────────────────────────────────────────────
N_STUDENTS = 1_000
M_SCHOOLS  = 50
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
vc     = fam_df["estrato_real"].value_counts()
strato_counts = {int(s): int(n) for s, n in vc.items() if int(s) in range(1, 7)}
total_fam     = sum(strato_counts.values())
strato_dist   = {s: strato_counts[s] / total_fam for s in sorted(strato_counts)}
s_bar         = sum(s * p for s, p in strato_dist.items())

log.info(f"  s_bar = {s_bar:.3f}")
log.info(f"  Distribución estratos: { {s: f'{p:.2%}' for s, p in strato_dist.items()} }")

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
alpha_s = {s: ALPHA_0 / (s ** GAMMA_POW) for s in range(1, 7)}   # Gallego & Hernando 2009
gamma_s = {s: 0.30 / s for s in range(1, 7)}                      # Neilson 2021

# Centros de cluster por estrato — para escenario con segregación espacial
# Inspirado en la geografía de Bogotá: sur-occidente (E1-E2) → centro (E3) → norte (E5-E6)
# Separación mínima entre estratos adyacentes ~0.16 (~1.6 sigmas con sigma=0.10)
# Separación E1-E6 ~0.99 (~10 sigmas) → sin solapamiento entre extremos
CENTROS_CLUSTER = {
    1: [0.15, 0.15],   # sur profundo (Ciudad Bolívar, Usme)
    2: [0.30, 0.20],   # sur (San Cristóbal, Rafael Uribe)
    3: [0.45, 0.35],   # centro-sur (Santa Fe, Los Mártires)
    4: [0.55, 0.60],   # centro-norte (Teusaquillo, Barrios Unidos)
    5: [0.70, 0.75],   # norte (Suba, Usaquén)
    6: [0.85, 0.85],   # norte alto (Chapinero norte, Usaquén alto)
}


# ─────────────────────────────────────────────────────────────────────────────
# Función principal: generar un escenario dado el grado de segregación
# ─────────────────────────────────────────────────────────────────────────────
def generar_escenario(
    sigma_cluster: float | None,
    rng_local: np.random.Generator,
    estrato_arr: np.ndarray,
    colegios: pd.DataFrame,
    label: str = "A",
) -> dict:
    """
    Genera las matrices de utilidad y rankings de preferencia para un escenario.

    Parameters
    ----------
    sigma_cluster : float or None
        None  → escenario A: coordenadas uniformes aleatorias (sin segregación)
        float → escenario B: clusters gaussianos por estrato (sigma controla dispersión)
    rng_local : np.random.Generator
        RNG con seed propio para reproducibilidad.
    estrato_arr : np.ndarray shape (N,)
        Estrato de cada estudiante.
    colegios : pd.DataFrame
        DataFrame con columnas q_j_std, v_j, capacidad_sintetica, coord_x, coord_y.
    label : str
        Etiqueta del escenario para logging.

    Returns
    -------
    dict con: estrato_arr, pref_bias_df, pref_true_df, dist_matrix, U_bias, U_true,
              coord_estudiantes, corr_v_bias, corr_v_true, delta
    """
    N = len(estrato_arr)
    M = len(colegios)
    school_arr    = colegios["id_establecimiento"].to_numpy(dtype=str)
    coord_colegios = colegios[["coord_x", "coord_y"]].values
    q_j_std = colegios["q_j_std"].values
    v_j     = colegios["v_j"].values

    # Coordenadas de estudiantes
    if sigma_cluster is None:
        coord_est = rng_local.uniform(size=(N, 2))
        log.info(f"  [{label}] Coordenadas uniformes (sin segregación)")
    else:
        coord_est = np.zeros((N, 2))
        for i, s in enumerate(estrato_arr):
            centro = CENTROS_CLUSTER[int(s)]
            coord_est[i] = np.clip(
                rng_local.normal(centro, sigma_cluster, size=2), 0, 1
            )
        log.info(f"  [{label}] Clusters gaussianos sigma={sigma_cluster}")

    dist_matrix = cdist(coord_est, coord_colegios).astype(np.float32)
    log.info(f"  [{label}] dist media={dist_matrix.mean():.4f}  max={dist_matrix.max():.4f}")

    alpha_i = np.array([alpha_s[int(s)] for s in estrato_arr])
    gamma_i = np.array([gamma_s[int(s)] for s in estrato_arr])
    eps     = rng_local.gumbel(0, SIGMA, size=(N, M))
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

    log.info(f"  [{label}] corr(estrato,v_top1) bias={corr_b:+.4f}  true={corr_t:+.4f}  Δ={delta:+.4f}")

    return {
        "label"              : label,
        "estrato_arr"        : estrato_arr,
        "pref_bias_df"       : pref_bias_df,
        "pref_true_df"       : pref_true_df,
        "dist_matrix"        : dist_matrix,
        "U_bias"             : U_bias,
        "U_true"             : U_true,
        "coord_estudiantes"  : coord_est,
        "corr_v_bias"        : corr_b,
        "corr_v_true"        : corr_t,
        "delta"              : delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generar estudiantes (compartidos entre escenarios — mismos estratos)
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"Paso 3 — Generando N={N_STUDENTS} estudiantes...")

strategos   = sorted(strato_dist.keys())
probs       = np.array([strato_dist[s] for s in strategos])
probs       = probs / probs.sum()
estrato_arr = rng.choice(strategos, size=N_STUDENTS, p=probs)

for s, count in zip(*np.unique(estrato_arr, return_counts=True)):
    log.info(f"  E{s}: {count} ({count/N_STUDENTS:.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Generar los dos escenarios
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 4 — Generando escenarios A y B...")

rng_A = np.random.default_rng(SEED + 1)
rng_B = np.random.default_rng(SEED + 2)

esc_A = generar_escenario(None,  rng_A, estrato_arr, colegios, label="A")
esc_B = generar_escenario(0.10,  rng_B, estrato_arr, colegios, label="B")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Guardar outputs
# ─────────────────────────────────────────────────────────────────────────────
log.info("Paso 5 — Guardando outputs...")

colegios.to_parquet(OUT_P / "sinteticos_b_colegios.parquet", index=False)

for esc in [esc_A, esc_B]:
    lbl = esc["label"]
    est_df = pd.DataFrame({
        "id_estudiante": [f"S{i:04d}" for i in range(N_STUDENTS)],
        "estrato":       estrato_arr,
        "coord_x":       esc["coord_estudiantes"][:, 0],
        "coord_y":       esc["coord_estudiantes"][:, 1],
    })
    est_df.to_parquet(OUT_P / f"sinteticos_b_estudiantes_{lbl}.parquet", index=False)
    esc["pref_bias_df"].to_parquet(OUT_P / f"sinteticos_b_preferencias_bias_{lbl}.parquet")
    esc["pref_true_df"].to_parquet(OUT_P / f"sinteticos_b_preferencias_true_{lbl}.parquet")
    log.info(f"  sinteticos_b_estudiantes_{lbl}.parquet")
    log.info(f"  sinteticos_b_preferencias_bias_{lbl}.parquet")
    log.info(f"  sinteticos_b_preferencias_true_{lbl}.parquet")

# Actualizar preferencias escenario A para compatibilidad con versión anterior
esc_A["pref_bias_df"].to_parquet(OUT_P / "sinteticos_b_preferencias_bias.parquet")
esc_A["pref_true_df"].to_parquet(OUT_P / "sinteticos_b_preferencias_true.parquet")

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
    "s_bar"           : round(s_bar, 4),
    "strato_dist"     : {str(s): round(p, 5) for s, p in strato_dist.items()},
    "alpha_s"         : {str(s): round(a, 6) for s, a in alpha_s.items()},
    "gamma_s"         : {str(s): round(g, 6) for s, g in gamma_s.items()},
    "corr_vj_qj"      : round(float(np.corrcoef(colegios["v_j"], colegios["q_j_std"])[0,1]), 5),
    "utilidad"        : "U_bias = q_j_std - alpha_s*log1p(d) + gamma_s*v_j + eps",
    "escenario_A"     : {"sigma_cluster": None,  **{k: round(esc_A[k], 5) for k in ["corr_v_bias","corr_v_true","delta"]}},
    "escenario_B"     : {"sigma_cluster": 0.15,  **{k: round(esc_B[k], 5) for k in ["corr_v_bias","corr_v_true","delta"]}},
}

with open(REP_DIR / "sinteticos_b_calibracion.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=2, ensure_ascii=False)

log.info("  sinteticos_b_colegios.parquet")
log.info("  sinteticos_b_calibracion.json")
log.info("Done.")
