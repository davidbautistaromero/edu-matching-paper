#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_regresion.py
===============
Comparacion de metodos de features visuales para predecir sobre-demanda
escolar en Bogota usando LASSO con validacion cruzada.

Modelos estimados:
  M0 - Linea base:  log_sobredemanda ~ controles
  M1 - NMF:         M0 + topicos NMF (K=8)
  M2 - Cityscapes:  M0 + proporciones de segmentacion semantica
  M3 - CLIP:        M0 + indices perceptuales CLIP
  M4 - Combinado:   M0 + Cityscapes + CLIP

Ejecutar desde cualquier directorio:
  python /ruta/a/scripts/04_regresion.py
"""

# =============================================================================
# BLOQUE 1: IMPORTACIONES
# =============================================================================

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# BLOQUE 2: CONFIGURACION
# =============================================================================

# -- Rutas de entrada ----------------------------------------------------------
COLEGIOS_PATH = r'C:\paper-AI\data\primary\colegios_features_imputed.geojson'
NMF_PATH      = r'C:\paper-AI\data\images\embeddings\gsv_nmf_K8.parquet'
CS_PATH       = r'C:\paper-AI\data\images\segmentation\gsv_cs_establecimiento.parquet'
CLIP_PATH     = r'C:\paper-AI\data\images\clip\gsv_clip_establecimiento.parquet'

# -- Rutas de salida -----------------------------------------------------------
OUT_DIR        = r'C:\paper-AI\reports'
OUT_TABLE_PATH = r'C:\paper-AI\reports\lasso_comparativa.csv'
OUT_COEFS_PATH = r'C:\paper-AI\reports\lasso_M4_coefs.csv'

# -- Variable dependiente ------------------------------------------------------
DEPVAR_RAW = 'sobre_demanda_j'   # columna original en colegios_features
DEPVAR     = 'log_sobredemanda'  # version transformada log(x); valores > 1

# -- Variables de control (linea base M0) --------------------------------------
CONTROLES = [
    'puntaje_icfes_promedio',      # promedio 2020/2022/2023 -- calculado en main()
    'tasa_pobreza_monetaria',
    'ingreso_percapita_promedio',
    'dist_sitp_m',                 # accesibilidad SITP -- mas relevante que TM para colegios
    'pct_no_oficial',              # competencia privada en la localidad
    'hurto_personas',              # seguridad del entorno
    'homicidios',
]

# -- Caracteristicas visuales por modelo ---------------------------------------
NMF_FEATURES = [f'topic_{i}' for i in range(1, 9)]  # topic_1 ... topic_8

CS_FEATURES = [
    'infraestructura_vial',
    'edificacion',
    'cerramiento',
    'vegetacion',
    'vehiculos',
    'mobiliario_urbano',
]

CLIP_FEATURES = [
    'mantenimiento',
    'vegetacion_percibida',
    'accesibilidad',
    'seguridad_percibida',
]

# -- Validacion cruzada --------------------------------------------------------
K_FOLDS      = 5
RANDOM_STATE = 42

# -- Crear carpeta de outputs si no existe -------------------------------------
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# -- Sistema de logging --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# =============================================================================
# BLOQUE 3: FUNCION -- CARGAR Y FUSIONAR DATASETS
# =============================================================================

def cargar_y_fusionar() -> pd.DataFrame:
    """
    Lee los cuatro archivos de datos, los fusiona por id_establecimiento y
    reporta cuantos establecimientos se pierden en cada join.

    Estrategia: inner join en cada paso para garantizar informacion completa
    en las cuatro fuentes. Se registra la perdida acumulada.
    """
    # -- Colegios (GeoJSON) ----------------------------------------------------
    log.info(f'Leyendo colegios: {COLEGIOS_PATH}')
    gdf = gpd.read_file(COLEGIOS_PATH)
    log.info(f'  Establecimientos en colegios_features: {len(gdf):,}')

    log.info('  Columnas disponibles en colegios_features.geojson:')
    for col in gdf.columns:
        log.info(f'    {col}')

    df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    df['id_establecimiento'] = df['id_establecimiento'].astype(str).str.strip()
    n_base = len(df)

    # -- NMF -------------------------------------------------------------------
    log.info(f'Leyendo NMF: {NMF_PATH}')
    nmf = pd.read_parquet(NMF_PATH)
    nmf['id_establecimiento'] = nmf['id_establecimiento'].astype(str).str.strip()
    log.info(f'  Establecimientos en NMF: {nmf["id_establecimiento"].nunique():,}')

    df = df.merge(nmf[['id_establecimiento'] + NMF_FEATURES],
                  on='id_establecimiento', how='inner')
    log.info(f'  Tras merge NMF: {len(df):,}  (perdidos: {n_base - len(df):,})')

    # -- Cityscapes ------------------------------------------------------------
    log.info(f'Leyendo Cityscapes: {CS_PATH}')
    cs = pd.read_parquet(CS_PATH)
    cs['id_establecimiento'] = cs['id_establecimiento'].astype(str).str.strip()
    log.info(f'  Establecimientos en Cityscapes: {cs["id_establecimiento"].nunique():,}')

    n_antes = len(df)
    df = df.merge(cs[['id_establecimiento'] + CS_FEATURES],
                  on='id_establecimiento', how='inner')
    log.info(f'  Tras merge Cityscapes: {len(df):,}  '
             f'(perdidos en este paso: {n_antes - len(df):,})')

    # -- CLIP ------------------------------------------------------------------
    log.info(f'Leyendo CLIP: {CLIP_PATH}')
    clip = pd.read_parquet(CLIP_PATH)
    clip['id_establecimiento'] = clip['id_establecimiento'].astype(str).str.strip()
    log.info(f'  Establecimientos en CLIP: {clip["id_establecimiento"].nunique():,}')

    n_antes = len(df)
    df = df.merge(clip[['id_establecimiento'] + CLIP_FEATURES],
                  on='id_establecimiento', how='inner')
    log.info(f'  Tras merge CLIP: {len(df):,}  '
             f'(perdidos en este paso: {n_antes - len(df):,})')

    log.info(f'  Perdida total respecto al base: {n_base - len(df):,} '
             f'({(n_base - len(df)) / n_base:.1%})')
    log.info(f'  Dataset de analisis: {len(df):,} establecimientos, '
             f'{df.shape[1]} columnas')

    return df


# =============================================================================
# BLOQUE 4: FUNCION -- ESTIMAR LASSO CON CV
# =============================================================================

def estimar_lasso(y: pd.Series, X: pd.DataFrame) -> dict:
    """
    Ajusta LassoCV(cv=5) dentro de un Pipeline con StandardScaler.

    StandardScaler se ajusta solo sobre el fold de entrenamiento en cada
    iteracion de CV, evitando data leakage entre pliegues.

    El RMSE_cv se extrae de mse_path_ en el indice de lambda*, usando los
    mismos pliegues que seleccionaron ese lambda.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso',  LassoCV(cv=K_FOLDS, random_state=RANDOM_STATE, max_iter=10_000)),
    ])
    pipeline.fit(X, y)

    lasso  = pipeline.named_steps['lasso']
    y_pred = pipeline.predict(X)
    n, p   = X.shape

    ss_res = np.sum((y.values - y_pred) ** 2)
    ss_tot = np.sum((y.values - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

    # RMSE_cv en lambda*: media de MSE sobre los K pliegues internos de LassoCV
    alpha_idx = np.argmin(np.abs(lasso.alphas_ - lasso.alpha_))
    rmse_cv   = np.sqrt(lasso.mse_path_[alpha_idx].mean())

    return {
        'pipeline':    pipeline,
        'coefs':       lasso.coef_,
        'intercept':   lasso.intercept_,
        'lambda_star': lasso.alpha_,
        'r2_adj':      r2_adj,
        'rmse_cv':     rmse_cv,
        'n_nonzero':   int(np.sum(lasso.coef_ != 0)),
        'n_obs':       n,
    }


# =============================================================================
# BLOQUE 5: FUNCION PRINCIPAL
# =============================================================================

def main():
    log.info('=' * 65)
    log.info('REGRESION COMPARATIVA -- CARACTERISTICAS VISUALES Y DEMANDA ESCOLAR')
    log.info('=' * 65)

    # -- Paso 1: Cargar y fusionar datos ---------------------------------------
    df = cargar_y_fusionar()

    # -- Paso 2: Transformacion logaritmica de la variable dependiente ---------
    # sobre_demanda_j > 1 siempre, se usa log natural sin log1p
    df[DEPVAR] = np.log(df[DEPVAR_RAW])

    # Promedio ICFES: suaviza variaciones anuales usando los tres anos disponibles
    # Se usa la media de los valores no-nulos para cada fila
    df['puntaje_icfes_promedio'] = df[['puntaje_2023', 'punt_global_2022', 'punt_global_2020']].mean(axis=1)

    for col in CONTROLES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_antes = len(df)
    df = df.dropna(subset=[DEPVAR] + CONTROLES)
    if len(df) < n_antes:
        log.info(f'  Filas eliminadas por NaN en controles: {n_antes - len(df)}')

    log.info(f'Variable dependiente: log({DEPVAR_RAW}) -> "{DEPVAR}"')
    log.info(f'  Media: {df[DEPVAR].mean():.3f}  '
             f'Std: {df[DEPVAR].std():.3f}  '
             f'Min: {df[DEPVAR].min():.3f}  '
             f'Max: {df[DEPVAR].max():.3f}')

    # -- Paso 3: Definir especificaciones --------------------------------------
    especificaciones = [
        ('M0 - Linea base',  []),
        ('M1 - NMF',         NMF_FEATURES),
        ('M2 - Cityscapes',  CS_FEATURES),
        ('M3 - CLIP',        CLIP_FEATURES),
        ('M4 - Combinado',   CS_FEATURES + CLIP_FEATURES),
    ]

    # -- Paso 4: Estimar modelos y recopilar metricas --------------------------
    y          = df[DEPVAR]
    resultados = []
    modelos    = {}   # guarda res + features por modelo para extraer coefs despues

    for nombre, features_vis in especificaciones:
        log.info('-' * 65)
        log.info(f'Estimando {nombre}')

        all_features = CONTROLES + features_vis

        # M0 sin features visuales: LASSO solo sobre controles
        if not all_features:
            X = pd.DataFrame({'_const': np.ones(len(y))}, index=df.index)
        else:
            X = df[all_features].copy()

        res = estimar_lasso(y, X)
        modelos[nombre] = {'res': res, 'features': all_features}

        log.info(f'  lambda* = {res["lambda_star"]:.6f}  |  '
                 f'activas = {res["n_nonzero"]} / {len(all_features)}')
        log.info(f'  R2_adj = {res["r2_adj"]:.4f}  |  '
                 f'RMSE_cv = {res["rmse_cv"]:.4f}  |  n = {res["n_obs"]:,}')

        resultados.append({
            'modelo':      nombre,
            'p_in':        len(all_features),
            'activas':     res['n_nonzero'],
            'lambda_star': round(res['lambda_star'], 6),
            'R2_adj':      round(res['r2_adj'],      4),
            'RMSE_cv':     round(res['rmse_cv'],      4),
        })

    # -- Paso 5: Tabla comparativa ---------------------------------------------
    log.info('=' * 65)
    log.info('TABLA COMPARATIVA DE MODELOS (LASSO)')
    log.info('=' * 65)

    tabla = pd.DataFrame(resultados)
    header = (f"{'Modelo':<28} {'p_in':>5} {'activas':>8} "
              f"{'lambda*':>10} {'R2_adj':>8} {'RMSE_cv':>9}")
    log.info(header)
    log.info('-' * len(header))
    for _, row in tabla.iterrows():
        log.info(f"{row['modelo']:<28} {row['p_in']:>5} "
                 f"{row['activas']:>8} {row['lambda_star']:>10.6f} "
                 f"{row['R2_adj']:>8.4f} {row['RMSE_cv']:>9.4f}")

    tabla.to_csv(OUT_TABLE_PATH, index=False, encoding='utf-8-sig')
    log.info(f'Tabla comparativa guardada: {OUT_TABLE_PATH}')

    # -- Paso 6: Coeficientes del mejor modelo (mayor R2_adj) ------------------
    mejor_nombre = tabla.loc[tabla['R2_adj'].idxmax(), 'modelo']
    mejor        = modelos[mejor_nombre]
    log.info('-' * 65)
    log.info(f'Extrayendo coeficientes del mejor modelo: {mejor_nombre}')

    coefs = pd.DataFrame({
        'variable': mejor['features'],
        'coef':     mejor['res']['coefs'],
        'activa':   mejor['res']['coefs'] != 0,
    }).sort_values('coef', key=abs, ascending=False)

    coefs.to_csv(OUT_COEFS_PATH, index=False, encoding='utf-8-sig')
    log.info(f'Coeficientes guardados: {OUT_COEFS_PATH}')

    activas = coefs[coefs['activa']]['variable'].tolist()
    log.info(f'  Features seleccionadas ({len(activas)}): {", ".join(activas)}')

    # -- Resumen final ---------------------------------------------------------
    log.info('=' * 65)
    log.info('RESUMEN FINAL')
    log.info(f'  Mejor R2_adj:  {tabla.loc[tabla["R2_adj"].idxmax(), "modelo"]}')
    log.info(f'  Menor RMSE_cv: {tabla.loc[tabla["RMSE_cv"].idxmin(), "modelo"]}')
    log.info(f'  Outputs en:    {OUT_DIR}')
    log.info('=' * 65)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == '__main__':
    main()
