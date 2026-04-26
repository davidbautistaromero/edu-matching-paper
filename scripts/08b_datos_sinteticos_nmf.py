"""
08b_datos_sinteticos_nmf.py
============================
Versión alternativa de 08_datos_sinteticos.py que construye el índice visual v_j
a partir de los tópicos NMF reales seleccionados por el Lasso M1, en lugar de
usar el residuo de log(SD) ~ q_j_std.

Justificación
-------------
La construcción original de v_j en 08_datos_sinteticos.py define v_j como el
residuo z-scoreado de la regresión OLS de log(sobre_demanda) sobre q_j_std.
Esto produce un v_j que es por construcción ortogonal a q_j_std, pero pierde
la interpretación visual: v_j termina siendo "demanda no explicada por calidad",
que mezcla apariencia con cualquier otro factor omitido (entorno, capital social,
percepción de seguridad).

Este script reemplaza esa construcción con la predicción visual del Lasso M1:

    v_j = sum_{k in S} beta_k * topic_k_std(j)

donde S = {topic_1, topic_2, topic_6} son los tópicos seleccionados por el Lasso
(coeficientes no cero), beta_k sus coeficientes, y topic_k_std la versión
estandarizada con el StandardScaler entrenado por el modelo. La combinación se
estandariza después para que v_j tenga media 0 y std 1, comparable con q_j_std.

Esto es post-Lasso (Belloni & Chernozhukov 2013): una vez que el regularizador
selecciona las variables relevantes, se trabaja exclusivamente con ellas.

Outputs (paralelos a 08_datos_sinteticos.py, prefijo "nmf_"):
  data/primary/sinteticos_nmf_colegios.parquet
  data/primary/sinteticos_nmf_estudiantes.parquet
  data/primary/sinteticos_nmf_preferencias.parquet
  data/primary/sinteticos_nmf_pref_sin_sesgo.parquet
  data/primary/sinteticos_nmf_utilidades.parquet
  reports/sinteticos_nmf_calibracion.json
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
COL_P      = ROOT / "data" / "primary"   / "colegios_features_imputed.geojson"
NMF_P      = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8.parquet"
FAM_P      = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
MODEL_P    = ROOT / "models" / "elasticnet_M1.joblib"
META_P     = ROOT / "models" / "elasticnet_M1_meta.json"
OUT_P      = ROOT / "data" / "primary"
REP_DIR    = ROOT / "reports"

N_STUDENTS = 1000
M_SCHOOLS  = 50
SIGMA      = 1.0
SEED       = 42

rng = np.random.default_rng(SEED)

log.info("=" * 60)
log.info("Paso 1 — Cargando modelo Lasso M1 y extrayendo coeficientes visuales...")

model = joblib.load(MODEL_P)
with open(META_P) as f:
    meta = json.load(f)

scaler = model.named_steps["scaler"]
lasso  = model.named_steps["lasso"]
features = meta["features"]

topic_names = [f"topic_{i}" for i in range(1, 9)]
topic_idx   = [features.index(t) for t in topic_names]

beta_topics = np.array([lasso.coef_[i] for i in topic_idx])
mu_topics   = np.array([scaler.mean_[i]  for i in topic_idx])
sd_topics   = np.array([scaler.scale_[i] for i in topic_idx])

active = np.abs(beta_topics) > 1e-9
log.info("  Tópicos activos en el Lasso (post-selección):")
for k, name in enumerate(topic_names):
    if active[k]:
        log.info(f"    {name}: beta={beta_topics[k]:+.5f}  mean={mu_topics[k]:.4f}  sd={sd_topics[k]:.4f}")

log.info("=" * 60)
log.info("Paso 2 — Cargando colegios y construyendo v_j desde los tópicos NMF...")

gdf = gpd.read_file(COL_P)
gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str)

nmf = pd.read_parquet(NMF_P)
nmf["id_establecimiento"] = nmf["id_establecimiento"].astype(str)

gdf = gdf.merge(nmf[["id_establecimiento"] + topic_names],
                on="id_establecimiento", how="inner")
log.info(f"  Colegios con tópicos NMF: {len(gdf)}")

T_raw = gdf[topic_names].values.astype(float)
T_std = (T_raw - mu_topics) / sd_topics

v_raw = T_std @ beta_topics
v_mean, v_sd = v_raw.mean(), v_raw.std()
gdf["v_j"] = (v_raw - v_mean) / v_sd

q_raw = gdf["q_j"].values.astype(float)
q_mean, q_std = q_raw.mean(), q_raw.std()
gdf["q_j_std"] = (q_raw - q_mean) / q_std

corr_vq = np.corrcoef(gdf["v_j"], gdf["q_j_std"])[0, 1]
v_mean_log = gdf["v_j"].mean()
v_std_log  = gdf["v_j"].std()
log.info(f"  v_j: media={v_mean_log:+.4f}  std={v_std_log:.4f}")
log.info(f"  corr(v_j, q_j_std) = {corr_vq:+.4f}")
log.info("  [No es 0 por construcción: refleja correlación real entre apariencia y calidad]")

log.info("=" * 60)
log.info("Paso 3 — Estimando alpha_hat con OLS de log(SD) ~ v_j + q_j_std...")

log_sd = np.log(gdf["sobre_demanda_j"].values.astype(float))
X_ols  = np.column_stack([np.ones(len(gdf)), gdf["v_j"].values, gdf["q_j_std"].values])
beta_ols, residuals, rank, sv = np.linalg.lstsq(X_ols, log_sd, rcond=None)

y_pred = X_ols @ beta_ols
ss_res = ((log_sd - y_pred) ** 2).sum()
ss_tot = ((log_sd - log_sd.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

n, p = X_ols.shape
se_beta1 = np.sqrt(ss_res / (n - p) * np.linalg.inv(X_ols.T @ X_ols)[1, 1])
t_beta1  = beta_ols[1] / se_beta1
p_beta1  = 2 * (1 - stats.t.cdf(abs(t_beta1), df=n - p))

alpha_hat = beta_ols[1]
log.info(f"  intercepto    = {beta_ols[0]:+.5f}")
log.info(f"  alpha_hat (v) = {beta_ols[1]:+.5f}  (se={se_beta1:.5f}, t={t_beta1:+.2f}, p={p_beta1:.4f})")
log.info(f"  beta_q        = {beta_ols[2]:+.5f}")
log.info(f"  R² del modelo = {r2:.4f}")
log.info(f"  Interpretación: +1 SD en v_j produce {alpha_hat*100:+.2f}% sobredemanda, controlando por calidad")

log.info("=" * 60)
log.info("Paso 4 — Calibrando gamma_s por estrato...")

fam_df       = pd.read_parquet(FAM_P)
vc           = fam_df["estrato_real"].value_counts()
strato_counts = {int(s): int(n) for s, n in vc.items() if int(s) in range(1, 7)}
total_fam    = sum(strato_counts.values())
strato_dist  = {s: n / total_fam for s, n in strato_counts.items()}
s_bar        = sum(s * p for s, p in strato_dist.items())
gamma_s      = {s: alpha_hat * (s / s_bar) for s in range(1, 7)}

log.info(f"  s_bar = {s_bar:.3f}")
for s, g in gamma_s.items():
    log.info(f"  gamma_s={s}: {g:+.5f}")

log.info("=" * 60)
log.info(f"Paso 5 — Muestreando M={M_SCHOOLS} colegios estratificados por q...")

gdf["q_cuartil"] = pd.qcut(gdf["q_j"], q=4, labels=False)
colegios_sample = (
    gdf.groupby("q_cuartil", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), M_SCHOOLS // 4 + 2), random_state=SEED),
           include_groups=False)
    .head(M_SCHOOLS)
    .reset_index(drop=True)
)

colegios_sample["capacidad"] = (
    colegios_sample["matricula_total"] / 13
).round().clip(lower=5).astype(int)
cap_scale = N_STUDENTS / colegios_sample["capacidad"].sum()
colegios_sample["capacidad_sintetica"] = (
    colegios_sample["capacidad"] * cap_scale
).round().clip(lower=1).astype(int)

colegios_out = colegios_sample[[
    "id_establecimiento", "nombre_establecimiento", "nombre_localidad",
    "q_j", "q_j_std", "v_j", "sobre_demanda_j",
    "matricula_total", "capacidad", "capacidad_sintetica",
]].copy()

v_min = colegios_out["v_j"].min()
v_max = colegios_out["v_j"].max()
log.info(f"  Colegios muestreados: {len(colegios_out)}")
log.info(f"  v_j en muestra: [{v_min:+.2f}, {v_max:+.2f}]")

log.info("=" * 60)
log.info(f"Paso 6 — Generando N={N_STUDENTS} estudiantes y matrices de utilidad...")

strategos = sorted(strato_dist.keys())
probs     = np.array([strato_dist[s] for s in strategos])
probs     = probs / probs.sum()
estrato_sintetico = rng.choice(strategos, size=N_STUDENTS, p=probs)

estudiantes_out = pd.DataFrame({
    "id_estudiante": [f"S{i:04d}" for i in range(N_STUDENTS)],
    "estrato":       estrato_sintetico,
})

q_j  = colegios_out["q_j_std"].values
v_j  = colegios_out["v_j"].values
s_i  = estrato_sintetico

eps = rng.normal(0, SIGMA, size=(N_STUDENTS, M_SCHOOLS))
gamma_i       = np.array([gamma_s[int(s)] for s in s_i])
visual_weight = alpha_hat + gamma_i

U_bias = q_j[None, :] + visual_weight[:, None] * v_j[None, :] + eps
U_true = q_j[None, :] + eps

log.info("=" * 60)
log.info("Paso 7 — Generando rankings...")

school_ids = colegios_out["id_establecimiento"].values
pref_cols  = [f"pref_{k+1}" for k in range(M_SCHOOLS)]

rank_bias_idx = np.argsort(-U_bias, axis=1)
rank_true_idx = np.argsort(-U_true, axis=1)

pref_bias_df = pd.DataFrame(
    school_ids[rank_bias_idx],
    index=estudiantes_out["id_estudiante"], columns=pref_cols,
)
pref_bias_df.index.name = "id_estudiante"

pref_true_df = pd.DataFrame(
    school_ids[rank_true_idx],
    index=estudiantes_out["id_estudiante"], columns=pref_cols,
)
pref_true_df.index.name = "id_estudiante"

school_attrs  = colegios_out.set_index("id_establecimiento")[["v_j", "q_j_std"]]
top1_v_bias   = pref_bias_df["pref_1"].map(school_attrs["v_j"])
top1_v_true   = pref_true_df["pref_1"].map(school_attrs["v_j"])

corr_v_bias = np.corrcoef(s_i, top1_v_bias.values)[0, 1]
corr_v_true = np.corrcoef(s_i, top1_v_true.values)[0, 1]

log.info(f"  corr(estrato, v_top1) CON sesgo = {corr_v_bias:+.4f}")
log.info(f"  corr(estrato, v_top1) SIN sesgo = {corr_v_true:+.4f}")
log.info(f"  Δ = {corr_v_bias - corr_v_true:+.4f}")

log.info("=" * 60)
log.info("Paso 8 — Guardando outputs...")

colegios_out.to_parquet(OUT_P / "sinteticos_nmf_colegios.parquet", index=False)
estudiantes_out.to_parquet(OUT_P / "sinteticos_nmf_estudiantes.parquet", index=False)
pref_bias_df.to_parquet(OUT_P / "sinteticos_nmf_preferencias.parquet")
pref_true_df.to_parquet(OUT_P / "sinteticos_nmf_pref_sin_sesgo.parquet")

util_df = pd.DataFrame(U_bias, index=estudiantes_out["id_estudiante"], columns=school_ids)
util_df.index.name = "id_estudiante"
util_df.to_parquet(OUT_P / "sinteticos_nmf_utilidades.parquet")

calibracion = {
    "metodo"            : "post-Lasso: v_j = combinación lineal de tópicos NMF seleccionados",
    "topicos_activos"   : [topic_names[k] for k in range(8) if active[k]],
    "betas_lasso"       : {topic_names[k]: round(float(beta_topics[k]), 6)
                           for k in range(8) if active[k]},
    "N_estudiantes"     : N_STUDENTS,
    "M_colegios"        : M_SCHOOLS,
    "sigma"             : SIGMA,
    "seed"              : SEED,
    "alpha_hat"         : round(float(alpha_hat), 6),
    "alpha_hat_se"      : round(float(se_beta1), 6),
    "alpha_hat_pval"    : round(float(p_beta1), 6),
    "beta_q_control"    : round(float(beta_ols[2]), 6),
    "r2_modelo"         : round(float(r2), 6),
    "s_bar"             : round(s_bar, 4),
    "gamma_s"           : {str(s): round(g, 6) for s, g in gamma_s.items()},
    "corr_v_j_q_j"      : round(corr_vq, 6),
    "validacion_sesgo"  : {
        "corr_estrato_vj_bias" : round(corr_v_bias, 5),
        "corr_estrato_vj_true" : round(corr_v_true, 5),
    },
}
with open(REP_DIR / "sinteticos_nmf_calibracion.json", "w", encoding="utf-8") as f:
    json.dump(calibracion, f, indent=2, ensure_ascii=False)

log.info("  sinteticos_nmf_*.parquet (5 archivos)")
log.info("  sinteticos_nmf_calibracion.json")
log.info("Done.")
