#!/usr/bin/env python3
"""
04_visual_index.py
Construye el índice visual v_j por establecimiento usando PCA sobre los tópicos LDA.

Metodología:
  - PCA sobre las proporciones de K tópicos LDA
  - PC1 = v_j (primera componente principal = índice escalar de calidad visual)
  - v_j captura la mayor varianza entre las distribuciones de tópicos visuales
  - Unir v_j y proporciones de tópicos al dataset maestro de colegios

Inputs:
  data/images/embeddings/gsv_lda_K{K}.parquet
  data/primary/colegios_features.geojson

Outputs:
  data/primary/colegios_visual.geojson    (NUEVO archivo, no sobreescribe colegios_features)
  data/primary/colegios_visual.parquet    (para análisis tabular)
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
K = 8   # tópicos LDA a usar (debe existir gsv_lda_K{K}.parquet)

LDA_PATH       = f'data/images/embeddings/gsv_lda_K{K}.parquet'
GEOJSON_IN     = 'data/primary/colegios_features.geojson'
GEOJSON_OUT    = 'data/primary/colegios_visual.geojson'
PARQUET_OUT    = 'data/primary/colegios_visual.parquet'

# =============================================================================
# IMPORTACIONES
# =============================================================================
import logging
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# =============================================================================
# 1. CARGAR PROPORCIONES DE TÓPICOS LDA
# =============================================================================
log.info('=' * 60)
log.info(f'CONSTRUCCIÓN DEL ÍNDICE VISUAL v_j  |  K={K}')
log.info('=' * 60)

log.info(f'Cargando proporciones de tópicos: {LDA_PATH}')
df_lda = pd.read_parquet(LDA_PATH)
log.info(f'  Shape: {df_lda.shape}  |  Establecimientos: {len(df_lda)}')

topic_cols = [c for c in df_lda.columns if c.startswith('topic_')]
log.info(f'  Tópicos: {topic_cols}')

# Matriz de proporciones [N, K]
X_topics = df_lda[topic_cols].values

# =============================================================================
# 2. PCA SOBRE PROPORCIONES DE TÓPICOS → v_j = PC1
# =============================================================================
log.info('Ajustando PCA sobre proporciones de tópicos...')
pca = PCA(n_components=K, random_state=42)
pca.fit(X_topics)

# Varianza explicada
var_exp = pca.explained_variance_ratio_
log.info(f'  Varianza explicada por componente:')
for i, v in enumerate(var_exp, 1):
    log.info(f'    PC{i}: {v*100:.2f}%')
log.info(f'  PC1 explica {var_exp[0]*100:.2f}% de la varianza total')

# Scores = coordenadas de cada establecimiento en el espacio PCA
scores = pca.transform(X_topics)   # [N, K]
v_j_raw = scores[:, 0]            # PC1 = índice visual

# Cargas de PC1 (cómo cada tópico contribuye al índice)
pc1_loadings = pca.components_[0]   # [K]

print('\n' + '─' * 50)
print(f'CARGAS DE PC1 (v_j) sobre los {K} tópicos:')
for col, loading in zip(topic_cols, pc1_loadings):
    bar = '█' * int(abs(loading) * 30)
    sign = '+' if loading >= 0 else '−'
    print(f'  {col}: {sign}{abs(loading):.4f}  {bar}')
print('─' * 50)

# =============================================================================
# 3. NORMALIZACIÓN DE v_j A ESCALA [0, 1]
# =============================================================================
v_min, v_max = v_j_raw.min(), v_j_raw.max()
v_j_norm = (v_j_raw - v_min) / (v_max - v_min)

log.info(f'v_j raw:   min={v_min:.4f}  max={v_max:.4f}  mean={v_j_raw.mean():.4f}')
log.info(f'v_j norm:  min=0.0000  max=1.0000  mean={v_j_norm.mean():.4f}')

# =============================================================================
# 4. CORRELACIÓN ENTRE v_j Y CADA TÓPICO (INTERPRETACIÓN)
# =============================================================================
print('\n' + '─' * 50)
print('CORRELACIÓN v_j con cada tópico (ayuda a interpretar PC1):')
for col in topic_cols:
    corr = np.corrcoef(v_j_raw, df_lda[col].values)[0, 1]
    bar = '█' * int(abs(corr) * 20)
    sign = '+' if corr >= 0 else '−'
    print(f'  {col}: {sign}{abs(corr):.4f}  {bar}')
print('─' * 50)

# =============================================================================
# 5. CONSTRUIR DATAFRAME CON RESULTADOS
# =============================================================================
df_visual = df_lda[['id_establecimiento'] + topic_cols].copy()
df_visual['v_j']            = v_j_raw
df_visual['v_j_normalized'] = v_j_norm
df_visual['pc1_loading']    = pc1_loadings.mean()   # escalar resumen de las cargas

# Agregar scores de PC1 como columna de referencia (redundante con v_j, útil para debug)
# No necesario en el output final

# =============================================================================
# 6. UNIR CON DATASET MAESTRO colegios_features.geojson
# =============================================================================
log.info(f'Cargando dataset maestro: {GEOJSON_IN}')
gdf_colegios = gpd.read_file(GEOJSON_IN)
log.info(f'  Establecimientos en geojson: {len(gdf_colegios)}')

# Verificar columna de join
if 'id_establecimiento' in gdf_colegios.columns:
    join_key = 'id_establecimiento'
elif 'DANE12_EST' in gdf_colegios.columns:
    join_key = 'DANE12_EST'
    # Homologar nombre en df_visual
    df_visual = df_visual.rename(columns={'id_establecimiento': 'DANE12_EST'})
else:
    raise KeyError('No se encontró id_establecimiento ni DANE12_EST en colegios_features.geojson')

log.info(f'  Columna de join: {join_key}')

# Join: colegios (izquierda) + visual (derecha)
# Usamos left join para conservar todos los establecimientos del geojson
cols_to_add = [join_key, 'v_j', 'v_j_normalized'] + topic_cols

# Si la columna v_j ya existe en colegios, eliminarla para evitar duplicados
drop_cols = [c for c in ['v_j', 'v_j_normalized'] + topic_cols if c in gdf_colegios.columns]
if drop_cols:
    log.info(f'  Eliminando columnas previas del geojson: {drop_cols}')
    gdf_colegios = gdf_colegios.drop(columns=drop_cols)

gdf_visual = gdf_colegios.merge(
    df_visual[cols_to_add],
    on=join_key,
    how='left',
)

n_con_vj    = gdf_visual['v_j'].notna().sum()
n_sin_vj    = gdf_visual['v_j'].isna().sum()
log.info(f'  Establecimientos con v_j:    {n_con_vj}')
log.info(f'  Establecimientos sin v_j:    {n_sin_vj}  (no tienen imágenes GSV)')

# =============================================================================
# 7. GUARDAR OUTPUTS
# =============================================================================
log.info(f'Guardando: {GEOJSON_OUT}')
gdf_visual.to_file(GEOJSON_OUT, driver='GeoJSON')
log.info(f'  GeoJSON guardado  ({gdf_visual.shape})')

log.info(f'Guardando: {PARQUET_OUT}')
# Para el parquet, eliminar la geometría (no serializable directamente)
df_out = pd.DataFrame(gdf_visual.drop(columns='geometry'))
df_out.to_parquet(PARQUET_OUT, index=False)
log.info(f'  Parquet guardado  ({df_out.shape})')

# =============================================================================
# 8. ESTADÍSTICAS DESCRIPTIVAS DE v_j
# =============================================================================
vj_vals = gdf_visual['v_j'].dropna()

print('\n' + '=' * 60)
print('ESTADÍSTICAS DE v_j (ÍNDICE VISUAL)')
print('=' * 60)
print(f'  N establecimientos con v_j:  {len(vj_vals)}')
print(f'  Media:                       {vj_vals.mean():.4f}')
print(f'  Mediana:                     {vj_vals.median():.4f}')
print(f'  Desv. estándar:              {vj_vals.std():.4f}')
print(f'  Mínimo:                      {vj_vals.min():.4f}')
print(f'  Máximo:                      {vj_vals.max():.4f}')
print(f'  % sobre la mediana:          {(vj_vals > vj_vals.median()).mean()*100:.1f}%')

# Correlación con q_j (calidad académica) si existe
if 'q_j' in gdf_visual.columns:
    df_corr = gdf_visual[['v_j', 'q_j']].dropna()
    corr_qj = df_corr['v_j'].corr(df_corr['q_j'])
    print(f'\n  Correlación v_j ↔ q_j (calidad académica): {corr_qj:.4f}')
    print(f'  (N para correlación: {len(df_corr)})')
else:
    print('\n  [q_j no encontrado en el dataset — sin correlación con calidad académica]')

print('\n  Distribución de v_j normalizado [0,1]:')
vj_norm_vals = gdf_visual['v_j_normalized'].dropna()
cuartiles = vj_norm_vals.quantile([0.25, 0.50, 0.75])
print(f'    P25: {cuartiles[0.25]:.4f}  |  P50: {cuartiles[0.50]:.4f}  |  P75: {cuartiles[0.75]:.4f}')

print('\n' + '=' * 60)
print('ARCHIVOS CREADOS')
print('=' * 60)
print(f'  {GEOJSON_OUT}')
print(f'  {PARQUET_OUT}')
print(f'\nColumnas añadidas: v_j, v_j_normalized, {", ".join(topic_cols)}')
print('=' * 60)
