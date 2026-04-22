#!/usr/bin/env python3
"""
03_lda_topics.py
Aprende tópicos visuales con LDA sobre los embeddings VGG19 de establecimientos.

Justificación para usar LDA:
  Las activaciones de block5_pool en VGG19 son no-negativas (después de ReLU),
  lo que cumple el requisito de LDA. Cada tópico representa una combinación de
  patrones visuales latentes (texturas, colores, estructuras arquitectónicas).

Metodología:
  - Normalización L1 de los vectores de 512d (transforma a distribuciones)
  - LDA con K ∈ {6, 8, 10} tópicos
  - K=8 se toma como el modelo principal

Inputs:
  data/images/embeddings/gsv_vgg19_establecimiento.parquet
  data/images/gsv/gsv_catalog.csv  (para nombres de establecimientos)

Outputs (por cada K):
  data/images/embeddings/gsv_lda_K{k}.parquet         (proporciones de tópicos)
  data/images/embeddings/gsv_lda_K{k}_topics.json     (top-10 features por tópico)
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
K_VALUES    = [6, 8, 10]          # valores de K a evaluar
K_DEFAULT   = 8                   # K principal para pasos siguientes
MAX_ITER    = 100                 # iteraciones máximas de LDA
RANDOM_SEED = 42                  # semilla para reproducibilidad
TOP_N_FEAT  = 10                  # top features por tópico a guardar
TOP_N_EST   = 3                   # top establecimientos por tópico a imprimir

EMBEDDINGS_PATH = 'data/images/embeddings/gsv_vgg19_establecimiento.parquet'
CATALOG_PATH    = 'data/images/gsv/gsv_catalog.csv'
EMBEDDINGS_DIR  = 'data/images/embeddings'

# =============================================================================
# IMPORTACIONES
# =============================================================================
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_embeddings() -> tuple[pd.DataFrame, np.ndarray, list]:
    """
    Carga los embeddings por establecimiento.
    Devuelve: (dataframe completo, matriz de features [N, 512], columnas de features)
    """
    log.info(f'Cargando embeddings: {EMBEDDINGS_PATH}')
    df = pd.read_parquet(EMBEDDINGS_PATH)
    log.info(f'  Shape: {df.shape}  |  Establecimientos: {df["id_establecimiento"].nunique()}')

    # Separar columna de ID y columnas de features
    feat_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feat_cols].values.astype(np.float64)

    log.info(f'  Features: {len(feat_cols)}  |  Min: {X.min():.4f}  Max: {X.max():.4f}')
    return df, X, feat_cols


def load_nombres() -> dict:
    """Carga mapeo id_establecimiento → nombre_establecimiento desde el catálogo."""
    try:
        catalog = pd.read_csv(CATALOG_PATH, usecols=['id_establecimiento', 'nombre_establecimiento'])
        catalog = catalog.drop_duplicates('id_establecimiento')
        return dict(zip(catalog['id_establecimiento'], catalog['nombre_establecimiento']))
    except Exception as e:
        log.warning(f'No se pudo cargar el catálogo para nombres: {e}')
        return {}


# =============================================================================
# LDA PARA UN VALOR DE K
# =============================================================================
def run_lda(X_norm: np.ndarray, ids: pd.Series, nombres: dict, K: int) -> tuple:
    """
    Ajusta LDA con K tópicos sobre los embeddings normalizados (L1).

    Devuelve:
      df_topics  : DataFrame [N, K] con proporciones de tópicos + id_establecimiento
      topics_json: dict con top-N feature indices por tópico
      lda        : modelo ajustado
    """
    log.info(f'  Ajustando LDA con K={K}...')

    lda = LatentDirichletAllocation(
        n_components=K,
        max_iter=MAX_ITER,
        learning_method='batch',        # batch es más estable para corpus pequeños
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    topic_proportions = lda.fit_transform(X_norm)  # [N, K]

    log.info(f'  Log-verosimilitud final: {lda.bound_:.2f}')
    log.info(f'  Perplejidad:             {lda.perplexity(X_norm):.2f}')

    # Proporciones de tópicos como DataFrame
    topic_cols = [f'topic_{k+1}' for k in range(K)]
    df_topics = pd.DataFrame(topic_proportions, columns=topic_cols)
    df_topics.insert(0, 'id_establecimiento', ids.values)

    # Top-N features por tópico (índices en el espacio de 512 dimensiones)
    topics_json = {}
    for k_idx in range(K):
        top_feats = np.argsort(lda.components_[k_idx])[::-1][:TOP_N_FEAT].tolist()
        topics_json[f'topic_{k_idx + 1}'] = top_feats

    return df_topics, topics_json, lda


# =============================================================================
# ESTADÍSTICAS Y DIAGNÓSTICO POR TÓPICO
# =============================================================================
def print_topic_stats(df_topics: pd.DataFrame, nombres: dict, K: int):
    """Imprime estadísticas de distribución de tópicos y top establecimientos."""
    topic_cols = [f'topic_{k+1}' for k in range(K)]

    print(f'\n{"="*60}')
    print(f'ESTADÍSTICAS DE TÓPICOS (K={K})')
    print(f'{"="*60}')

    for col in topic_cols:
        vals = df_topics[col]
        print(f'\n{col}:')
        print(f'  Media: {vals.mean():.4f}  |  Std: {vals.std():.4f}  '
              f'|  Max: {vals.max():.4f}')

        # Top-N establecimientos con mayor proporción en este tópico
        top_idx = df_topics.nlargest(TOP_N_EST, col)['id_establecimiento'].tolist()
        top_nombres = [
            nombres.get(eid, str(eid)) for eid in top_idx
        ]
        print(f'  Top {TOP_N_EST} establecimientos:')
        for i, (eid, nom) in enumerate(zip(top_idx, top_nombres), 1):
            prop = df_topics.loc[
                df_topics['id_establecimiento'] == eid, col
            ].values[0]
            print(f'    {i}. {nom[:50]:<50} ({prop:.4f})')


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    log.info('=' * 60)
    log.info('APRENDIZAJE DE TÓPICOS VISUALES CON LDA')
    log.info('=' * 60)

    # -------------------------------------------------------------------------
    # 1. Cargar embeddings y nombres
    # -------------------------------------------------------------------------
    df_emb, X, feat_cols = load_embeddings()
    nombres = load_nombres()
    ids = df_emb['id_establecimiento']

    # -------------------------------------------------------------------------
    # 2. Normalización L1 (transforma vectores a distribuciones de probabilidad)
    # -------------------------------------------------------------------------
    log.info('Normalizando vectores (L1)...')
    X_norm = normalize(X, norm='l1')
    log.info(f'  Suma de filas después de L1: min={X_norm.sum(axis=1).min():.4f}  '
             f'max={X_norm.sum(axis=1).max():.4f}')

    # -------------------------------------------------------------------------
    # 3. LDA para cada valor de K
    # -------------------------------------------------------------------------
    for K in K_VALUES:
        log.info(f'\n{"─"*60}')
        log.info(f'K = {K}')
        log.info(f'{"─"*60}')

        df_topics, topics_json, lda = run_lda(X_norm, ids, nombres, K)

        # Guardar proporciones de tópicos
        topics_path = os.path.join(EMBEDDINGS_DIR, f'gsv_lda_K{K}.parquet')
        df_topics.to_parquet(topics_path, index=False)
        log.info(f'  Proporciones guardadas: {topics_path}  ({df_topics.shape})')

        # Guardar top features por tópico
        json_path = os.path.join(EMBEDDINGS_DIR, f'gsv_lda_K{K}_topics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(topics_json, f, indent=2)
        log.info(f'  Top features guardados: {json_path}')

        # Imprimir estadísticas
        print_topic_stats(df_topics, nombres, K)

        # Marcar el modelo principal
        if K == K_DEFAULT:
            log.info(f'\n  *** K={K} es el modelo principal (K_DEFAULT) ***')

    # -------------------------------------------------------------------------
    # 4. Resumen final
    # -------------------------------------------------------------------------
    log.info('\n' + '=' * 60)
    log.info('RESUMEN')
    for K in K_VALUES:
        tag = ' ← PRINCIPAL' if K == K_DEFAULT else ''
        log.info(f'  K={K}: gsv_lda_K{K}.parquet + gsv_lda_K{K}_topics.json{tag}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
