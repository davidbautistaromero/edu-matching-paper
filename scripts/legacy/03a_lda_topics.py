#!/usr/bin/env python3
"""
03_lda_topics.py
Aprende tópicos visuales con LDA sobre los embeddings VGG19.

Metodología:
  1. Carga embeddings crudos por imagen (gsv_vgg19_raw.parquet)
  2. PCA blanqueado (whiten=True): 512d → PCA_N_COMPONENTS
     - Whitening iguala la varianza de todas las PCs antes de L1-norm,
       evitando que PC1 domine y LDA colapse en 1-2 tópicos efectivos
     - Shift a no-negativo por columna (requerido por LDA)
  3. Normalización L1 (transforma vectores a distribuciones)
  4. LDA con K ∈ K_VALUES sobre imágenes individuales
  5. Agrega proporciones de tópicos por establecimiento (media)

Inputs:
  data/images/embeddings/gsv_vgg19_raw.parquet
  data/images/gsv/gsv_catalog.csv

Outputs (por cada K):
  data/images/embeddings/gsv_lda_K{k}_images.parquet   (por imagen)
  data/images/embeddings/gsv_lda_K{k}.parquet          (por establecimiento)
  data/images/embeddings/gsv_lda_K{k}_topics.json      (top-10 PCs por tópico)
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
K_VALUES         = [6, 8, 10]
K_DEFAULT        = 8
PCA_N_COMPONENTS = 70     # 80% varianza explicada — ver 02a_diagnose_embeddings.py
MAX_ITER         = 200
RANDOM_SEED      = 42
TOP_N_FEAT       = 10
TOP_N_EST        = 3

RAW_EMBEDDINGS_PATH = 'data/images/embeddings/gsv_vgg19_raw.parquet'
CATALOG_PATH        = 'data/images/gsv/gsv_catalog.csv'
EMBEDDINGS_DIR      = 'data/images/embeddings'

# =============================================================================
# IMPORTACIONES
# =============================================================================
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# CARGA
# =============================================================================
def load_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    log.info(f'Cargando embeddings crudos: {RAW_EMBEDDINGS_PATH}')
    df = pd.read_parquet(RAW_EMBEDDINGS_PATH)

    feat_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feat_cols].values.astype(np.float64)

    n_img = len(df)
    n_est = df['id_establecimiento'].nunique()
    log.info(f'  Imágenes: {n_img:,}  |  Establecimientos: {n_est}  |  Features: {len(feat_cols)}')
    log.info(f'  Rango valores: min={X.min():.4f}  max={X.max():.4f}')
    return df, X


def load_nombres() -> dict:
    try:
        catalog = pd.read_csv(
            CATALOG_PATH,
            usecols=['id_establecimiento', 'nombre_establecimiento'],
        )
        catalog = catalog.drop_duplicates('id_establecimiento')
        return dict(zip(catalog['id_establecimiento'], catalog['nombre_establecimiento']))
    except Exception as e:
        log.warning(f'No se pudo cargar catálogo para nombres: {e}')
        return {}


# =============================================================================
# PCA BLANQUEADO + SHIFT A NO-NEGATIVO
# =============================================================================
def apply_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    """
    Reduce dimensión con PCA blanqueado y shiftea a no-negativo.
    whiten=True iguala la varianza de todas las PCs, evitando que las primeras
    componentes dominen después de L1-norm y colapsen LDA.
    """
    log.info(f'Aplicando PCA blanqueado: 512d → {n_components}d ...')
    pca = PCA(n_components=n_components, whiten=True, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    var_acum = pca.explained_variance_ratio_.cumsum()
    log.info(f'  Varianza explicada: {100*var_acum[-1]:.1f}%')
    log.info(f'  PC1: {100*pca.explained_variance_ratio_[0]:.1f}%  '
             f'PC2: {100*pca.explained_variance_ratio_[1]:.1f}%')

    X_pca -= X_pca.min(axis=0)
    log.info(f'  Rango después de shift: min={X_pca.min():.4f}  max={X_pca.max():.4f}')
    return X_pca, pca


# =============================================================================
# LDA
# =============================================================================
def run_lda(
    X_norm: np.ndarray,
    df_meta: pd.DataFrame,
    nombres: dict,
    K: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Ajusta LDA con K tópicos sobre imágenes individuales.
    Devuelve proporciones por imagen y por establecimiento (media).
    """
    log.info(f'  Ajustando LDA con K={K} sobre {len(X_norm):,} imágenes...')

    lda = LatentDirichletAllocation(
        n_components=K,
        max_iter=MAX_ITER,
        learning_method='batch',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    topic_props = lda.fit_transform(X_norm)  # [N_images, K]

    log.info(f'  Log-verosimilitud: {lda.bound_:.2f}')
    log.info(f'  Perplejidad:       {lda.perplexity(X_norm):.2f}')

    topic_cols = [f'topic_{k+1}' for k in range(K)]

    df_img = pd.DataFrame(topic_props, columns=topic_cols)
    for col in ['id_establecimiento', 'id_sede', 'heading']:
        df_img.insert(list(df_img.columns).index(topic_cols[0]), col, df_meta[col].values)

    df_est = (
        df_img
        .groupby('id_establecimiento')[topic_cols]
        .mean()
        .reset_index()
    )

    topics_json = {
        f'topic_{k+1}': {
            'top_features': np.argsort(lda.components_[k])[::-1][:TOP_N_FEAT].tolist(),
            'feature_space': f'pca_{PCA_N_COMPONENTS}d_whitened',
        }
        for k in range(K)
    }

    return df_img, df_est, topics_json, lda


# =============================================================================
# ESTADÍSTICAS POR TÓPICO
# =============================================================================
def print_topic_stats(df_est: pd.DataFrame, nombres: dict, K: int):
    topic_cols = [f'topic_{k+1}' for k in range(K)]

    print(f'\n{"="*60}')
    print(f'ESTADÍSTICAS DE TÓPICOS (K={K}) — por establecimiento')
    print(f'{"="*60}')

    for col in topic_cols:
        vals = df_est[col]
        print(f'\n{col}:')
        print(f'  Media: {vals.mean():.4f}  |  Std: {vals.std():.4f}  |  Max: {vals.max():.4f}')

        top_rows = df_est.nlargest(TOP_N_EST, col)
        print(f'  Top {TOP_N_EST} establecimientos:')
        for i, row in enumerate(top_rows.itertuples(), 1):
            nom = nombres.get(row.id_establecimiento, str(row.id_establecimiento))
            prop = getattr(row, col)
            print(f'    {i}. {nom[:55]:<55} ({prop:.4f})')


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info('=' * 60)
    log.info('APRENDIZAJE DE TÓPICOS VISUALES CON LDA + PCA')
    log.info('=' * 60)

    df_raw, X = load_embeddings()
    nombres = load_nombres()

    X_pca, _ = apply_pca(X, PCA_N_COMPONENTS)

    log.info('Normalizando vectores (L1)...')
    X_norm = normalize(X_pca, norm='l1')
    log.info(f'  Suma de filas: min={X_norm.sum(axis=1).min():.4f}  '
             f'max={X_norm.sum(axis=1).max():.4f}')

    for K in K_VALUES:
        log.info(f'\n{"─"*60}')
        log.info(f'K = {K}')
        log.info(f'{"─"*60}')

        df_img, df_est, topics_json, _ = run_lda(X_norm, df_raw, nombres, K)

        img_path = os.path.join(EMBEDDINGS_DIR, f'gsv_lda_K{K}_images.parquet')
        df_img.to_parquet(img_path, index=False)
        log.info(f'  Por imagen guardado:          {img_path}  {df_img.shape}')

        est_path = os.path.join(EMBEDDINGS_DIR, f'gsv_lda_K{K}.parquet')
        df_est.to_parquet(est_path, index=False)
        log.info(f'  Por establecimiento guardado: {est_path}  {df_est.shape}')

        json_path = os.path.join(EMBEDDINGS_DIR, f'gsv_lda_K{K}_topics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(topics_json, f, indent=2)
        log.info(f'  Top features guardados:       {json_path}')

        print_topic_stats(df_est, nombres, K)

        if K == K_DEFAULT:
            log.info(f'\n  *** K={K} es el modelo principal (K_DEFAULT) ***')

    log.info('\n' + '=' * 60)
    log.info('RESUMEN')
    log.info(f'  Preproceso: PCA blanqueado 512d → {PCA_N_COMPONENTS}d + L1')
    for K in K_VALUES:
        tag = ' ← PRINCIPAL' if K == K_DEFAULT else ''
        log.info(f'  K={K}: gsv_lda_K{K}.parquet + gsv_lda_K{K}_images.parquet{tag}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
