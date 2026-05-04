#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_regresion.py
===============
Comparacion de 4 estimadores (OLS, Ridge, Lasso, ElasticNet) x 5 modelos
(M0-M4) para predecir sobre-demanda escolar en Bogota.

Ejecutar desde cualquier directorio:
  python /ruta/a/scripts/04_regresion.py
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================

import datetime
import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURACION
# =============================================================================

COLEGIOS_PATH = r'C:\paper-AI\data\primary\colegios_features_imputed.geojson'
NMF_PATH      = r'C:\paper-AI\data\images\embeddings\gsv_nmf_K8.parquet'
CS_PATH       = r'C:\paper-AI\data\images\segmentation\gsv_cs_establecimiento.parquet'
CLIP_PATH     = r'C:\paper-AI\data\images\clip\gsv_clip_establecimiento.parquet'

OUT_DIR        = r'C:\paper-AI\reports'
OUT_TABLE_PATH = r'C:\paper-AI\reports\comparativa_estimadores.csv'
OUT_COEFS_PATH = r'C:\paper-AI\reports\mejores_coefs.csv'
MODELS_DIR     = r'C:\paper-AI\models'

DEPVAR_RAW = 'sobre_demanda_j'
DEPVAR     = 'log_sobredemanda'

CONTROLES = [
    'puntaje_icfes_promedio',
    'dist_sitp_m',
    'pct_no_oficial',
    'hurto_personas',
    'homicidios',
    'n_oficiales_localidad',
    'estrato_2',
    'estrato_3',
    'estrato_4',
    'estrato_5',
    'estrato_6',
    'es_rural',
    'es_tecnico',
]

NMF_FEATURES  = [f'topic_{i}' for i in range(1, 9)]
CS_FEATURES   = [
    'infraestructura_vial', 'edificacion', 'cerramiento',
    'vegetacion', 'vehiculos', 'mobiliario_urbano',
]
CLIP_FEATURES = [
    'mantenimiento', 'vegetacion_percibida',
    'accesibilidad', 'seguridad_percibida',
]

K_FOLDS      = 5
RANDOM_STATE = 42
L1_RATIOS    = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]

# Nombre del step dentro de cada pipeline (para extraer atributos post-fit)
_STEP_KEY = {
    'OLS':        'ols',
    'Ridge':      'ridge',
    'Lasso':      'lasso',
    'ElasticNet': 'en',
}

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# =============================================================================
# CARGA Y FUSION
# =============================================================================

def cargar_y_fusionar() -> pd.DataFrame:
    """
    Lee los cuatro archivos y hace inner join por id_establecimiento.
    Informa cuantos establecimientos se pierden en cada paso.
    """
    log.info(f'Leyendo colegios: {COLEGIOS_PATH}')
    gdf = gpd.read_file(COLEGIOS_PATH)
    log.info(f'  Establecimientos base: {len(gdf):,}')

    df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    df['id_establecimiento'] = df['id_establecimiento'].astype(str).str.strip()
    n_base = len(df)

    log.info(f'Leyendo NMF: {NMF_PATH}')
    nmf = pd.read_parquet(NMF_PATH)
    nmf['id_establecimiento'] = nmf['id_establecimiento'].astype(str).str.strip()
    df = df.merge(nmf[['id_establecimiento'] + NMF_FEATURES],
                  on='id_establecimiento', how='inner')
    log.info(f'  Tras merge NMF: {len(df):,}  (perdidos: {n_base - len(df):,})')

    log.info(f'Leyendo Cityscapes: {CS_PATH}')
    cs = pd.read_parquet(CS_PATH)
    cs['id_establecimiento'] = cs['id_establecimiento'].astype(str).str.strip()
    n_antes = len(df)
    df = df.merge(cs[['id_establecimiento'] + CS_FEATURES],
                  on='id_establecimiento', how='inner')
    log.info(f'  Tras merge Cityscapes: {len(df):,}  (perdidos: {n_antes - len(df):,})')

    log.info(f'Leyendo CLIP: {CLIP_PATH}')
    clip = pd.read_parquet(CLIP_PATH)
    clip['id_establecimiento'] = clip['id_establecimiento'].astype(str).str.strip()
    n_antes = len(df)
    df = df.merge(clip[['id_establecimiento'] + CLIP_FEATURES],
                  on='id_establecimiento', how='inner')
    log.info(f'  Tras merge CLIP: {len(df):,}  (perdidos: {n_antes - len(df):,})')

    log.info(f'  Perdida total: {n_base - len(df):,} ({(n_base - len(df)) / n_base:.1%})')
    log.info(f'  Dataset de analisis: {len(df):,} establecimientos')
    return df


# =============================================================================
# CONSTRUCCION DE PIPELINES
# =============================================================================

def construir_pipelines() -> dict:
    """
    Devuelve pipelines frescos (sin ajustar) para cada estimador.
    StandardScaler se aplica dentro del pipeline para evitar data leakage en CV.
    """
    return {
        'OLS': Pipeline([
            ('scaler', StandardScaler()),
            ('ols',    LinearRegression()),
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge',  RidgeCV(cv=K_FOLDS)),
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('lasso',  LassoCV(cv=K_FOLDS, max_iter=10_000, random_state=RANDOM_STATE)),
        ]),
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('en',     ElasticNetCV(cv=K_FOLDS, max_iter=10_000,
                                    l1_ratio=L1_RATIOS, random_state=RANDOM_STATE)),
        ]),
    }


# =============================================================================
# ESTIMACION DE UN (modelo, estimador)
# =============================================================================

def estimar(nombre_est: str, pipeline: Pipeline,
            X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Calcula metricas para un pipeline sobre un conjunto de features.

    RMSE_cv: CV externo de 5 pliegues (cross_val_score) sobre el pipeline
    completo, garantizando comparabilidad entre estimadores.  Cuando el
    estimador interno es LassoCV/RidgeCV/ElasticNetCV, cada pliegue corre
    su propia seleccion de hiperparametros — CV anidado correcto.

    R2_adj: calculado en muestra completa (in-sample) tras ajustar el
    pipeline sobre todos los datos.
    """
    p = X.shape[1]
    n = len(y)

    # CV externo para RMSE comparable entre estimadores
    scores  = cross_val_score(pipeline, X, y, cv=K_FOLDS,
                              scoring='neg_mean_squared_error')
    rmse_cv = float(np.sqrt(-scores.mean()))

    # Ajuste en muestra completa para R2_adj y coeficientes definitivos
    # (cross_val_score clona el pipeline internamente, no lo modifica)
    pipeline.fit(X, y)
    y_pred  = pipeline.predict(X)

    ss_res = float(np.sum((y.values - y_pred) ** 2))
    ss_tot = float(np.sum((y.values - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

    step     = _STEP_KEY[nombre_est]
    est      = pipeline.named_steps[step]
    coefs    = est.coef_

    if nombre_est == 'OLS':
        activas    = p
        hiperparam = '-'
    elif nombre_est == 'Ridge':
        activas    = p
        hiperparam = f'alpha={est.alpha_:.4g}'
    elif nombre_est == 'Lasso':
        activas    = int(np.sum(coefs != 0))
        hiperparam = f'alpha={est.alpha_:.6g}'
    else:  # ElasticNet
        activas    = int(np.sum(coefs != 0))
        hiperparam = f'alpha={est.alpha_:.6g} l1={est.l1_ratio_:.2f}'

    spearman = float(spearmanr(y.values, y_pred).statistic)

    return {
        'coefs':      coefs,
        'activas':    activas,
        'r2_adj':     r2_adj,
        'rmse_cv':    rmse_cv,
        'spearman':   spearman,
        'hiperparam': hiperparam,
        'features':   X.columns.tolist(),
        'p_in':       p,
    }


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    log.info('=' * 70)
    log.info('REGRESION COMPARATIVA -- FEATURES VISUALES Y DEMANDA ESCOLAR')
    log.info('=' * 70)

    df = cargar_y_fusionar()

    # sobre_demanda_j > 1 siempre; log directo sin log1p
    df[DEPVAR] = np.log(df[DEPVAR_RAW])

    # Promedio ICFES: suaviza variaciones anuales usando los tres anos disponibles
    df['puntaje_icfes_promedio'] = (
        df[['puntaje_2023', 'punt_global_2022', 'punt_global_2020']].mean(axis=1)
    )

    # Proporciones de estrato como dummies continuas; base = estrato 1
    for k in [2, 3, 4, 5, 6]:
        df[f'estrato_{k}'] = pd.to_numeric(df[f'pct_estrato_{k}'], errors='coerce')

    # Dummies de zona y caracter (base: URBANA, Academico)
    df['es_rural']   = df['zona'].isin(['RURAL', 'EXPANSION']).astype(float)
    df['es_tecnico'] = df['caracter_media'].isin(['Tecnico', 'Academico - Tecnico']).astype(float)
    # "Sin informacion" en caracter_media -> NaN para que caiga en dropna
    df.loc[df['caracter_media'].isin(['Sin informacion', 'Sin información']), 'es_tecnico'] = np.nan

    # Capacidad publica en la misma localidad excluyendo el establecimiento propio
    df['matricula_total'] = pd.to_numeric(df['matricula_total'], errors='coerce')
    cap_loc = df.groupby('nombre_localidad')['matricula_total'].transform('sum')
    df['n_oficiales_localidad'] = (cap_loc - df['matricula_total']).clip(lower=0)

    for col in CONTROLES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_antes = len(df)
    df = df.dropna(subset=[DEPVAR] + CONTROLES).reset_index(drop=True)
    log.info(f'Filas eliminadas por NaN en controles/depvar: {n_antes - len(df)}')
    log.info(f'Muestra de analisis: n={len(df):,}')
    log.info(f'Depvar -- media={df[DEPVAR].mean():.3f}  '
             f'std={df[DEPVAR].std():.3f}  '
             f'min={df[DEPVAR].min():.3f}  '
             f'max={df[DEPVAR].max():.3f}')

    y = df[DEPVAR]

    especificaciones = [
        ('M0', 'Controles',  []),           # benchmark silencioso
        ('M1', 'NMF',        NMF_FEATURES),
        ('M2', 'Cityscapes', CS_FEATURES),
        ('M3', 'CLIP',       CLIP_FEATURES),
        ('M4', 'CS+CLIP',    CS_FEATURES + CLIP_FEATURES),
    ]

    filas      = []
    todos_res  = {}   # {(modelo_id, nombre_est): res}
    todos_pipes = {}  # {(modelo_id, nombre_est): fitted pipeline}

    for modelo_id, modelo_desc, features_vis in especificaciones:
        all_features = CONTROLES + features_vis
        X = df[all_features].copy()

        log.info('-' * 70)
        log.info(f'Estimando {modelo_id} ({modelo_desc}) -- p={len(all_features)}')

        for nombre_est, pipe in construir_pipelines().items():
            res = estimar(nombre_est, pipe, X, y)
            todos_res[(modelo_id, nombre_est)]   = res
            todos_pipes[(modelo_id, nombre_est)] = pipe

            log.info(f'  [{nombre_est:<11}]  R2_adj={res["r2_adj"]:.4f}  '
                     f'Spearman={res["spearman"]:+.4f}  '
                     f'RMSE_cv={res["rmse_cv"]:.4f}  '
                     f'activas={res["activas"]}/{len(all_features)}  '
                     f'hp={res["hiperparam"]}')

            filas.append({
                'modelo':         f'{modelo_id} - {modelo_desc}',
                'estimador':      nombre_est,
                'p_in':           res['p_in'],
                'activas':        res['activas'],
                'R2_adj':         round(res['r2_adj'], 4),
                'Spearman':       round(res['spearman'], 4),
                'RMSE_cv':        round(res['rmse_cv'], 4),
                'hiperparametro': res['hiperparam'],
            })

    # -------------------------------------------------------------------------
    # Tabla comparativa: agregar ΔR²_adj vs M0 y ordenar por Spearman desc
    # -------------------------------------------------------------------------
    tabla = pd.DataFrame(filas)

    # ΔR²_adj: ganancia incremental de features visuales vs M0 (mismo estimador)
    r2_m0 = tabla[tabla['modelo'].str.startswith('M0')].set_index('estimador')['R2_adj']
    tabla['delta_R2_adj'] = tabla.apply(
        lambda r: round(r['R2_adj'] - r2_m0.get(r['estimador'], float('nan')), 4),
        axis=1
    )

    # Ordenar por Spearman descendente; M0 queda como referencia
    tabla = tabla.sort_values('Spearman', ascending=False).reset_index(drop=True)

    log.info('=' * 70)
    log.info('TABLA COMPARATIVA (ordenada por Spearman descendente)')
    log.info('=' * 70)

    header = (f"{'Modelo':<28} {'Estimador':<12} {'activas':>8} "
              f"{'R2_adj':>8} {'ΔR2_adj':>9} {'Spearman':>10} {'RMSE_cv':>9}  Hiperparametro")
    log.info(header)
    log.info('-' * (len(header) + 5))
    for _, row in tabla.iterrows():
        log.info(
            f"{row['modelo']:<28} {row['estimador']:<12} {row['activas']:>8} "
            f"{row['R2_adj']:>8.4f} {row['delta_R2_adj']:>+9.4f} "
            f"{row['Spearman']:>10.4f} {row['RMSE_cv']:>9.4f}  "
            f"{row['hiperparametro']}"
        )

    tabla.to_csv(OUT_TABLE_PATH, index=False, encoding='utf-8-sig')
    log.info(f'Tabla comparativa guardada: {OUT_TABLE_PATH}')

    # -------------------------------------------------------------------------
    # Coeficientes del mejor modelo: mayor Spearman excluyendo M0
    # -------------------------------------------------------------------------
    tabla_sin_m0 = tabla[~tabla['modelo'].str.startswith('M0')]
    mejor_fila      = tabla_sin_m0.iloc[0]
    mejor_modelo_id = mejor_fila['modelo'].split(' - ')[0]
    mejor_est       = mejor_fila['estimador']
    mejor_res       = todos_res[(mejor_modelo_id, mejor_est)]

    coefs_df = pd.DataFrame({
        'variable': mejor_res['features'],
        'coef':     mejor_res['coefs'],
        'activa':   mejor_res['coefs'] != 0,
    }).sort_values('coef', key=abs, ascending=False)

    coefs_df.to_csv(OUT_COEFS_PATH, index=False, encoding='utf-8-sig')
    log.info(f'Coeficientes del mejor modelo guardados: {OUT_COEFS_PATH}')

    activas_list = coefs_df[coefs_df['activa']]['variable'].tolist()

    log.info('=' * 70)
    log.info('MEJOR COMBINACION (Spearman maximo, excl. M0)')
    log.info(f'  Modelo:      {mejor_fila["modelo"]}')
    log.info(f'  Estimador:   {mejor_est}')
    log.info(f'  Spearman:    {mejor_fila["Spearman"]:.4f}')
    log.info(f'  ΔR2_adj:     {mejor_fila["delta_R2_adj"]:+.4f}  (vs M0)')
    log.info(f'  R2_adj:      {mejor_fila["R2_adj"]:.4f}')
    log.info(f'  RMSE_cv:     {mejor_fila["RMSE_cv"]:.4f}')
    log.info(f'  Hiperpar.:   {mejor_fila["hiperparametro"]}')
    log.info(f'  Features activas ({len(activas_list)}): {", ".join(activas_list)}')
    log.info(f'  Outputs en:  {OUT_DIR}')
    log.info('=' * 70)

    # -------------------------------------------------------------------------
    # Guardar todos los modelos
    # -------------------------------------------------------------------------
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    for (modelo_id, nombre_est), pipe in todos_pipes.items():
        slug = f'{nombre_est.lower()}_{modelo_id.lower()}'
        joblib.dump(pipe, models_dir / f'{slug}.joblib')
        res  = todos_res[(modelo_id, nombre_est)]
        meta = {
            'model':        f'{nombre_est} {modelo_id}',
            'features':     res['features'],
            'target':       'log_sobredemanda',
            'n_obs':        len(df),
            'rmse_cv':      round(float(res['rmse_cv']), 6),
            'r2_adj':       round(float(res['r2_adj']), 6),
            'spearman':     round(float(res['spearman']), 6),
            'fit_date':     datetime.date.today().isoformat(),
        }
        with open(models_dir / f'{slug}_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f'  {len(todos_pipes)} modelos guardados en: {MODELS_DIR}')


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == '__main__':
    main()
