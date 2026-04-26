"""
08_datos_sinteticos.py
======================
Genera la población sintética calibrada descrita en la sección 5.3 del diseño metodológico.

Propósito
---------
En los datos reales (07_matching_bm_da.py) las preferencias se forman por distancia
geográfica. Este script construye una población donde las preferencias se forman
explícitamente por CALIDAD + SESGO VISUAL, según el modelo teórico del paper.
Eso permite aislar el efecto del sesgo visual sobre la asignación, sin el ruido
de la localización geográfica.

Modelo de utilidad (sección 5.3)
---------------------------------
Con sesgo:
    u_ij  = q_j_std + (alpha_hat + gamma_s_i) * v_j + eps_ij

Sin sesgo (preferencias verdaderas — sólo calidad):
    u0_ij = q_j_std + eps_ij

Donde:
  q_j_std       calidad ICFES estandarizada (z-score sobre los M colegios muestreados)
  v_j           índice visual del entorno escolar (definido abajo)
  alpha_hat     peso global del sesgo visual (estimado por OLS de los datos reales)
  gamma_s_i     componente del sesgo que varía con el estrato del estudiante i:
                hogares de mayor estrato son más sensibles a señales visuales
                (Hastings et al. 2009; Gallego & Hernando 2009)
  eps_ij        ruido idiosincrático ~ Normal(0, sigma²)

Nota: se usa el MISMO eps en u_ij y u0_ij para hacer la comparación justa:
las diferencias entre ranking con y sin sesgo se deben sólo a alpha_hat * v_j,
no a ruido diferente.

Construcción de v_j (índice visual)
-------------------------------------
Los archivos NMF de tópicos visuales (gsv_nmf_K8.parquet) tienen un problema
de formato que impide su lectura con pyarrow. Como alternativa, se construye
v_j directamente desde sobre_demanda_j:

  Paso 1 — OLS: log(sobre_demanda_j) = beta0 + beta1 * q_j_std + e_j
  Paso 2 — v_j_raw = e_j   (residuo = señal de demanda no explicada por calidad)
  Paso 3 — v_j = z-score(v_j_raw)

Esto garantiza que v_j ⊥ q_j por construcción algebraica (propiedades de los
residuos OLS), que es el supuesto del modelo teórico.

Estimación de alpha_hat
-----------------------
  OLS: log(sobre_demanda_j) ~ v_j
  alpha_hat = pendiente de esta regresión.
  Interpretación: un aumento de +1 SD en v_j está asociado con +alpha_hat * 100%
  más sobredemanda, controlando por calidad académica.

Calibración de gamma_s
-----------------------
  gamma_s = alpha_hat * (s / s_bar)
  donde s_bar = media ponderada del estrato en Bogotá (≈ 2.24).
  Justificación: los hogares de mayor estrato tienen más opciones de mercado
  (colegios privados, cambio de barrio) y son más sensibles a señales visuales
  (Gallego & Hernando 2009). El ratio gamma_6 / gamma_1 = 6/1 es el caso límite;
  en la práctica la sensibilidad crece gradualmente.

Outputs
-------
  data/primary/sinteticos_colegios.parquet       — M=50 colegios con q_j, v_j, capacidades
  data/primary/sinteticos_estudiantes.parquet    — N=1000 estudiantes con estrato
  data/primary/sinteticos_preferencias.parquet   — rankings CON sesgo (u_ij)
  data/primary/sinteticos_pref_sin_sesgo.parquet — rankings SIN sesgo (u0_ij)
  data/primary/sinteticos_utilidades.parquet     — matriz U completa (N × M)
  reports/sinteticos_calibracion.json            — parámetros y métricas de validación
  reports/figures/matching/sinteticos_*.png      — figuras de validación del sesgo
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
COL_P   = ROOT / "data" / "primary"   / "colegios_features_imputed.geojson"
FAM_P   = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
OUT_P   = ROOT / "data" / "primary"
FIG_DIR = ROOT / "reports" / "figures" / "matching"
REP_DIR = ROOT / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Parámetros configurables ───────────────────────────────────────────────────
N_STUDENTS = 1000    # número de estudiantes sintéticos
M_SCHOOLS  = 50      # número de colegios a muestrear del universo real
SIGMA      = 1.0     # desviación estándar del ruido idiosincrático (Normal)
SEED       = 42      # semilla para reproducibilidad de todos los pasos aleatorios

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Paso 1 — Cargar colegios y construir el índice visual v_j
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 1 — Construyendo índice visual v_j...")

gdf = gpd.read_file(COL_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)

# ── 1a. Estandarizar q_j (calidad ICFES) ─────────────────────────────────────
# Z-score: (x - media) / std. Lleva q_j a una escala con media 0 y std 1,
# comparable con v_j que también estará en z-score.
q_raw          = gdf["q_j"].values.astype(float)
q_mean, q_std  = q_raw.mean(), q_raw.std()
gdf["q_j_std"] = (q_raw - q_mean) / q_std

# ── 1b. Regresión OLS para separar calidad de señal visual ───────────────────
# El modelo de demanda es: log(SD_j) = beta0 + beta1 * q_j_std + e_j
# donde SD_j = sobre_demanda_j = demanda / matrícula.
# Los residuos e_j capturan la demanda no explicada por calidad académica.
# Son la señal visual v_j_raw (antes de estandarizar).
log_sd     = np.log(gdf["sobre_demanda_j"].values.astype(float))
q_std_vals = gdf["q_j_std"].values

# Construir la matriz de diseño X = [1, q_j_std] para OLS con intercepto
X       = np.column_stack([np.ones(len(q_std_vals)), q_std_vals])
# lstsq resuelve min||Xb - y||² en sentido de mínimos cuadrados (sin invertir X'X)
beta_ols, *_ = np.linalg.lstsq(X, log_sd, rcond=None)
fitted        = X @ beta_ols       # valores ajustados: beta0 + beta1 * q_j_std
v_j_raw       = log_sd - fitted    # residuos: demanda no explicada por calidad

# ── 1c. Estandarizar v_j ─────────────────────────────────────────────────────
# Z-score de los residuos: v_j tiene media 0, std ≈ 1.
# Propiedad algebraica: residuos OLS son ortogonales a los regresores,
# entonces corr(v_j, q_j_std) = 0 exactamente.
v_mean, v_sd   = v_j_raw.mean(), v_j_raw.std()
gdf["v_j"]     = (v_j_raw - v_mean) / v_sd

# Verificación
corr_vq = np.corrcoef(gdf["v_j"], gdf["q_j_std"])[0, 1]

log.info(f"  OLS log(SD) ~ q_j_std: beta0={beta_ols[0]:.4f}  beta_q={beta_ols[1]:.4f}")
log.info(f"  v_j: media={gdf['v_j'].mean():.4f}  std={gdf['v_j'].std():.4f}  "
         f"min={gdf['v_j'].min():.3f}  max={gdf['v_j'].max():.3f}")
log.info(f"  corr(v_j, q_j_std) = {corr_vq:.4f}  [debe ser ≈0 — garantía OLS]")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 2 — Estimar alpha_hat (peso global del sesgo visual)
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 2 — Estimando alpha_hat...")

# Como v_j ya es el residuo de log(SD) ~ q_j_std, la regresión de log(SD) ~ v_j
# recupera exactamente la varianza no explicada por q_j:
#   log(SD_j) = alpha0 + alpha_hat * v_j + r_j
# donde r_j es el componente explicado por q_j (que v_j no captura porque son ortogonales).
# La pendiente alpha_hat es el alpha_hat del modelo de la sección 5.2.
v_j_vals = gdf["v_j"].values
slope, intercept, r_val, p_val, se = stats.linregress(v_j_vals, log_sd)
alpha_hat = slope

log.info(f"  alpha_hat = {alpha_hat:.5f}  (R²={r_val**2:.4f}, p={p_val:.4f})")
log.info(f"  Interpretación: +1 SD en v_j → +{alpha_hat*100:.2f}% sobredemanda (neta de calidad)")
log.info(f"  [Referencia LASSO M1-NMF: topic_1=+0.0087, topic_2=-0.0064]")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 — Calibrar gamma_s (heterogeneidad del sesgo por estrato)
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 3 — Calibrando gamma_s por estrato...")

# Distribución de estrato de los hogares con hijos en colegio oficial (EM2021)
fam_df       = pd.read_parquet(FAM_P)
vc           = fam_df["estrato_real"].value_counts()
strato_counts = {int(s): int(n) for s, n in vc.items() if int(s) in range(1, 7)}
total_fam    = sum(strato_counts.values())
strato_dist  = {s: n / total_fam for s, n in strato_counts.items()}

# Estrato medio ponderado: promedio del estrato en la distribución real de Bogotá
s_bar = sum(s * p for s, p in strato_dist.items())

# gamma_s = alpha_hat * (s / s_bar):
# - En estrato s_bar (el promedio), gamma_s = alpha_hat → efecto igual al global
# - En estrato s > s_bar, gamma_s > alpha_hat → mayor sensibilidad visual
# - En estrato s < s_bar, gamma_s < alpha_hat → menor sensibilidad visual
# El total del peso visual del estudiante i sobre el colegio j es:
#   (alpha_hat + gamma_s_i) = alpha_hat * (1 + s_i / s_bar)
gamma_s = {s: alpha_hat * (s / s_bar) for s in range(1, 7)}

log.info(f"  s_bar (estrato medio ponderado) = {s_bar:.2f}")
for s, g in gamma_s.items():
    log.info(f"  gamma_s={s}: {g:.5f}  ({strato_dist.get(s, 0)*100:.1f}% familias)")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 4 — Muestrear M colegios estratificados por nivel de calidad
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"Paso 4 — Muestreando M={M_SCHOOLS} colegios...")

# Estratificar por cuartil de q_j antes de muestrear garantiza que la muestra
# cubra todo el rango de calidad (no sólo colegios promedio), lo que hace el
# matching más informativo: habrá colegios buenos y malos compitiendo.
gdf["q_cuartil"] = pd.qcut(gdf["q_j"], q=4, labels=False)

# Muestrear ~M/4 colegios de cada cuartil (sample estratificado)
colegios_sample = (
    gdf.groupby("q_cuartil", group_keys=False)
    .apply(
        lambda g: g.sample(min(len(g), M_SCHOOLS // 4 + 2), random_state=SEED),
        include_groups=False,   # evitar warning de pandas sobre groupby keys
    )
    .head(M_SCHOOLS)           # truncar exactamente a M_SCHOOLS
    .reset_index(drop=True)
)

log.info(f"  Colegios muestreados: {len(colegios_sample)}")
log.info(f"  q_j_std: [{colegios_sample['q_j_std'].min():.2f}, {colegios_sample['q_j_std'].max():.2f}]")
log.info(f"  v_j:     [{colegios_sample['v_j'].min():.2f}, {colegios_sample['v_j'].max():.2f}]")

# ── Capacidad escolar para la simulación ─────────────────────────────────────
# Misma lógica que 07: matricula_total / 13 ≈ cupo anual
colegios_sample["capacidad"] = (
    colegios_sample["matricula_total"] / 13
).round().clip(lower=5).astype(int)

# Escalar la capacidad para que la suma total ≈ N_STUDENTS.
# Esto crea un mercado perfectamente competitivo: cada cupo es disputado en promedio
# por exactamente 1 estudiante. Sin este escalado, todos obtendrían su 1ª preferencia
# y BM y DA serían indistinguibles.
cap_scale = N_STUDENTS / colegios_sample["capacidad"].sum()
colegios_sample["capacidad_sintetica"] = (
    colegios_sample["capacidad"] * cap_scale
).round().clip(lower=1).astype(int)

log.info(f"  Capacidad sintética total: {colegios_sample['capacidad_sintetica'].sum():,} "
         f"(≈ N={N_STUDENTS})")

# Seleccionar columnas de salida: sólo las necesarias para el matching y las métricas
colegios_out = colegios_sample[[
    "id_establecimiento", "nombre_establecimiento", "nombre_localidad",
    "q_j", "q_j_std", "v_j", "sobre_demanda_j",
    "matricula_total", "capacidad", "capacidad_sintetica",
]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Paso 5 — Generar N estudiantes sintéticos
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"Paso 5 — Generando N={N_STUDENTS} estudiantes sintéticos...")

# Ordenar estratos y construir vector de probabilidades según distribución EM2021
strategos = sorted(strato_dist.keys())
probs     = np.array([strato_dist[s] for s in strategos])
probs     = probs / probs.sum()   # renormalizar por si hay redondeo

# Muestrear estrato con reemplazo según distribución real
estrato_sintetico = rng.choice(strategos, size=N_STUDENTS, p=probs)

log.info("  Distribución de estrato generada:")
for s in strategos:
    n = (estrato_sintetico == s).sum()
    log.info(f"    Estrato {s}: {n:4d} ({n/N_STUDENTS*100:.1f}%)  [real: {strato_dist.get(s,0)*100:.1f}%]")

# Un id único por estudiante para tracking en los outputs
estudiantes_out = pd.DataFrame({
    "id_estudiante": [f"S{i:04d}" for i in range(N_STUDENTS)],
    "estrato":       estrato_sintetico,
})

# ─────────────────────────────────────────────────────────────────────────────
# Paso 6 — Calcular matrices de utilidad (con y sin sesgo visual)
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 6 — Calculando matrices de utilidad U_bias y U_true...")

q_j = colegios_out["q_j_std"].values   # (M,) calidad estandarizada de cada colegio
v_j = colegios_out["v_j"].values       # (M,) señal visual de cada colegio
s_i = estrato_sintetico                 # (N,) estrato de cada estudiante

# Ruido idiosincrático: misma realización para CON y SIN sesgo.
# Usar el mismo eps garantiza que las diferencias entre U_bias y U_true reflejen
# sólo el efecto de (alpha_hat + gamma_i) * v_j, no varianza aleatoria diferente.
eps = rng.normal(0, SIGMA, size=(N_STUDENTS, M_SCHOOLS))   # (N, M)

# Peso visual total por estudiante: alpha_hat (global) + gamma_s_i (específico del estrato)
# Es el coeficiente total con el que el estudiante i valora v_j del colegio j
gamma_i      = np.array([gamma_s[int(s)] for s in s_i])   # (N,)
visual_weight = alpha_hat + gamma_i                         # (N,) total weight on v_j

# U_bias (N × M): utilidad cuando las preferencias incluyen sesgo visual
# Broadcasting: q_j[None,:] de (1,M) → (N,M); visual_weight[:,None] de (N,1) → (N,M)
U_bias = (
    q_j[None, :]                              # componente de calidad (igual para todos)
    + visual_weight[:, None] * v_j[None, :]   # sesgo visual (heterogéneo por estrato)
    + eps                                      # ruido idiosincrático
)

# U_true (N × M): preferencias verdaderas sin sesgo (sólo calidad + ruido)
U_true = q_j[None, :] + eps

log.info(f"  U_bias: [{U_bias.min():.2f}, {U_bias.max():.2f}]  media={U_bias.mean():.2f}")
log.info(f"  U_true: [{U_true.min():.2f}, {U_true.max():.2f}]  media={U_true.mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 7 — Generar rankings de preferencia (ordenar U descendente)
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 7 — Generando rankings...")

school_ids = colegios_out["id_establecimiento"].values   # (M,) IDs de colegios
pref_cols  = [f"pref_{k+1}" for k in range(M_SCHOOLS)]  # nombres de columnas

# argsort con negativo: ordena de mayor a menor utilidad (primer preferencia = más alta)
# rank_bias_idx[i, 0] = índice del colegio más preferido por estudiante i bajo sesgo
rank_bias_idx = np.argsort(-U_bias, axis=1)   # (N, M) índices en orden descendente
rank_bias_ids = school_ids[rank_bias_idx]      # (N, M) IDs de colegios en ese orden

rank_true_idx = np.argsort(-U_true, axis=1)
rank_true_ids = school_ids[rank_true_idx]

# Construir DataFrames con id_estudiante como índice y pref_1..pref_50 como columnas
pref_bias_df = pd.DataFrame(
    rank_bias_ids,
    index=estudiantes_out["id_estudiante"],
    columns=pref_cols,
)
pref_bias_df.index.name = "id_estudiante"

pref_true_df = pd.DataFrame(
    rank_true_ids,
    index=estudiantes_out["id_estudiante"],
    columns=pref_cols,
)
pref_true_df.index.name = "id_estudiante"

# ─────────────────────────────────────────────────────────────────────────────
# Paso 8 — Validar que el sesgo distorsiona las preferencias en la dirección esperada
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 8 — Validando sesgo en preferencias...")

# Atributos del colegio para cada primera preferencia
school_attrs  = colegios_out.set_index("id_establecimiento")[["v_j", "q_j_std"]]
top1_v_bias   = pref_bias_df["pref_1"].map(school_attrs["v_j"])    # v_j del top-1 con sesgo
top1_v_true   = pref_true_df["pref_1"].map(school_attrs["v_j"])    # v_j del top-1 sin sesgo
top1_q_bias   = pref_bias_df["pref_1"].map(school_attrs["q_j_std"])
top1_q_true   = pref_true_df["pref_1"].map(school_attrs["q_j_std"])

# Correlación de Pearson entre estrato del estudiante y atributo de su colegio preferido.
# Si el sesgo funciona correctamente: corr(estrato, v_j_top1) debe ser MAYOR con sesgo.
corr_v_bias = np.corrcoef(s_i, top1_v_bias.values)[0, 1]
corr_v_true = np.corrcoef(s_i, top1_v_true.values)[0, 1]
corr_q_bias = np.corrcoef(s_i, top1_q_bias.values)[0, 1]
corr_q_true = np.corrcoef(s_i, top1_q_true.values)[0, 1]

log.info(f"  corr(estrato, v_j_top1) CON sesgo  = {corr_v_bias:.4f}")
log.info(f"  corr(estrato, v_j_top1) SIN sesgo  = {corr_v_true:.4f}")
log.info(f"  corr(estrato, q_j_top1) CON sesgo  = {corr_q_bias:.4f}")
log.info(f"  corr(estrato, q_j_top1) SIN sesgo  = {corr_q_true:.4f}")
log.info(f"  → El sesgo amplifica corr(estrato, v_j): {corr_v_true:.4f} → {corr_v_bias:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 9 — Guardar outputs
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 9 — Guardando outputs...")

colegios_out.to_parquet(OUT_P / "sinteticos_colegios.parquet", index=False)
estudiantes_out.to_parquet(OUT_P / "sinteticos_estudiantes.parquet", index=False)
pref_bias_df.to_parquet(OUT_P / "sinteticos_preferencias.parquet")
pref_true_df.to_parquet(OUT_P / "sinteticos_pref_sin_sesgo.parquet")

# Guardar también la matriz U_bias completa (puede ser útil para RegretNet en el futuro)
util_df = pd.DataFrame(U_bias, index=estudiantes_out["id_estudiante"], columns=school_ids)
util_df.index.name = "id_estudiante"
util_df.to_parquet(OUT_P / "sinteticos_utilidades.parquet")

# JSON con todos los parámetros de calibración (trazabilidad y reproducibilidad)
calibracion = {
    "N_estudiantes"     : N_STUDENTS,
    "M_colegios"        : M_SCHOOLS,
    "sigma"             : SIGMA,
    "seed"              : SEED,
    "alpha_hat"         : round(float(alpha_hat), 6),
    "alpha_hat_pval"    : round(float(p_val), 6),
    "s_bar"             : round(s_bar, 4),
    "gamma_s"           : {str(s): round(g, 6) for s, g in gamma_s.items()},
    "v_j_construccion"  : "residuo de OLS(log_sobredemanda ~ q_j_std), z-score",
    "corr_v_j_q_j"      : round(corr_vq, 6),
    "validacion_sesgo"  : {
        "corr_estrato_vj_bias" : round(corr_v_bias, 5),
        "corr_estrato_vj_true" : round(corr_v_true, 5),
        "corr_estrato_qj_bias" : round(corr_q_bias, 5),
        "corr_estrato_qj_true" : round(corr_q_true, 5),
    },
}
with open(REP_DIR / "sinteticos_calibracion.json", "w", encoding="utf-8") as f:
    json.dump(calibracion, f, indent=2, ensure_ascii=False)

log.info(f"  sinteticos_colegios.parquet         — {len(colegios_out)} colegios")
log.info(f"  sinteticos_estudiantes.parquet      — {len(estudiantes_out)} estudiantes")
log.info(f"  sinteticos_preferencias.parquet     — {pref_bias_df.shape} (con sesgo)")
log.info(f"  sinteticos_pref_sin_sesgo.parquet   — {pref_true_df.shape} (sin sesgo)")
log.info(f"  sinteticos_utilidades.parquet       — {util_df.shape}")
log.info("  sinteticos_calibracion.json")

# ─────────────────────────────────────────────────────────────────────────────
# Paso 10 — Figuras de validación
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Paso 10 — Generando figuras...")

# ── Figura 1: validación del sesgo en preferencias ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x_s = np.arange(1, 7)
w   = 0.35

# Panel izquierdo: v_j vs q_j en los M colegios (deben ser ortogonales)
ax = axes[0]
sc = ax.scatter(
    colegios_out["q_j_std"], colegios_out["v_j"],
    c=colegios_out["sobre_demanda_j"], cmap="RdYlGn_r", s=60, alpha=0.8,
)
plt.colorbar(sc, ax=ax, label="Sobredemanda observada")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Calidad ICFES (q_j, z-score)")
ax.set_ylabel("Índice visual (v_j, z-score de residuo)")
ax.set_title(f"M={M_SCHOOLS} colegios: v_j ⊥ q_j")
ax.text(0.05, 0.95, f"corr(v,q)={corr_vq:.3f}", transform=ax.transAxes, va="top", fontsize=9)

# Panel central: v_j del colegio top-1 por estrato (verificar amplificación del sesgo)
ax = axes[1]
mean_v_bias = [top1_v_bias[s_i == s].mean() for s in range(1, 7)]
mean_v_true = [top1_v_true[s_i == s].mean() for s in range(1, 7)]
ax.bar(x_s - w/2, mean_v_bias, w, label="CON sesgo",   color="#E53935", alpha=0.85)
ax.bar(x_s + w/2, mean_v_true, w, label="SIN sesgo",   color="#1E88E5", alpha=0.85)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Estrato")
ax.set_ylabel("v_j medio del colegio top-1")
ax.set_title("Sesgo visual en 1ª preferencia\n"
             f"Δcorr(estrato,v): {corr_v_true:.3f} → {corr_v_bias:.3f}")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# Panel derecho: q_j del colegio top-1 por estrato (calidad en la primera preferencia)
ax = axes[2]
mean_q_bias = [top1_q_bias[s_i == s].mean() for s in range(1, 7)]
mean_q_true = [top1_q_true[s_i == s].mean() for s in range(1, 7)]
ax.bar(x_s - w/2, mean_q_bias, w, label="CON sesgo",   color="#E53935", alpha=0.85)
ax.bar(x_s + w/2, mean_q_true, w, label="SIN sesgo",   color="#1E88E5", alpha=0.85)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Estrato")
ax.set_ylabel("q_j_std medio del colegio top-1")
ax.set_title("Calidad en 1ª preferencia\n"
             f"Δcorr(estrato,q): {corr_q_true:.3f} → {corr_q_bias:.3f}")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Validación del sesgo visual en preferencias — Datos sintéticos (§5.3)",
             y=1.01, fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "sinteticos_validacion.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figura 2: distribución de v_j y estimación de alpha_hat ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
ax.hist(gdf["v_j"], bins=30, color="#7B1FA2", alpha=0.8, edgecolor="white")
ax.axvline(0, color="black", linewidth=1.2)
ax.set_xlabel("Índice visual v_j (z-score, todos los colegios)")
ax.set_ylabel("Colegios")
ax.set_title(f"Distribución de v_j — 303 colegios reales\nα̂={alpha_hat:.4f}  p={p_val:.4f}")

ax = axes[1]
ax.scatter(gdf["v_j"], log_sd, s=20, alpha=0.5, color="#7B1FA2")
xr = np.linspace(gdf["v_j"].min(), gdf["v_j"].max(), 100)
ax.plot(xr, intercept + slope * xr, color="black", linewidth=1.5,
        label=f"OLS: pendiente = {slope:.4f}")
ax.set_xlabel("Índice visual v_j (z-score)")
ax.set_ylabel("log(sobredemanda_j)")
ax.set_title("Estimación de α̂: OLS log(SD) ~ v_j")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "sinteticos_v_j_construccion.png", dpi=150)
plt.close(fig)

log.info("  sinteticos_validacion.png")
log.info("  sinteticos_v_j_construccion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("RESUMEN DE CALIBRACIÓN")
log.info("=" * 60)
log.info(f"  N={N_STUDENTS} estudiantes | M={M_SCHOOLS} colegios | sigma={SIGMA}")
log.info(f"  alpha_hat = {alpha_hat:.5f}  (p={p_val:.4f})")
log.info(f"  gamma_s   = " + "  ".join(f"s{s}={g:.4f}" for s, g in gamma_s.items()))
log.info(f"  v_j ⊥ q_j: corr={corr_vq:.4f}")
log.info(f"  Amplificación del sesgo: corr(estrato,v) {corr_v_true:.4f} → {corr_v_bias:.4f}"
         f"  (Δ={corr_v_bias-corr_v_true:+.4f})")
log.info("Done.")
