#!/usr/bin/env python3
"""
03_nmf_topics.py
Aprende tópicos visuales con NMF sobre los embeddings VGG19.

Por qué NMF y no LDA:
  LDA colapsa en este dataset porque los embeddings VGG19 de fachadas escolares
  en Bogotá son suficientemente homogéneos para que el prior de Dirichlet domine
  y todos los documentos converjan a distribuciones casi uniformes. NMF no asume
  distribuciones probabilísticas, trabaja directamente sobre los features
  no-negativos (garantizados por ReLU en VGG19) y produce representaciones
  parts-based más interpretables para datos visuales densos.

Metodología:
  1. Carga embeddings crudos por imagen (gsv_vgg19_raw.parquet)
  2. Normalización L2 por fila (estabiliza la escala entre imágenes)
  3. NMF con K ∈ K_VALUES sobre imágenes individuales
  4. Normalización L1 de proporciones de salida (W → suma por fila = 1)
  5. Agrega proporciones de tópicos por establecimiento (media)

Inputs:
  data/images/embeddings/gsv_vgg19_raw.parquet
  data/images/gsv/gsv_catalog.csv

Outputs (por cada K):
  data/images/embeddings/gsv_nmf_K{k}_images.parquet   (por imagen)
  data/images/embeddings/gsv_nmf_K{k}.parquet          (por establecimiento)
  data/images/embeddings/gsv_nmf_K{k}_topics.json      (top-10 features por tópico)
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
K_VALUES    = [6, 8, 10]
K_DEFAULT   = 8
MAX_ITER    = 500
RANDOM_SEED = 42
TOP_N_FEAT  = 10
TOP_N_EST   = 3

RAW_EMBEDDINGS_PATH = 'data/images/embeddings/gsv_vgg19_raw.parquet'
CATALOG_PATH        = 'data/images/gsv/gsv_catalog.csv'
EXCLUSION_PATH      = 'data/raw/excluded_schools.csv'
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
from sklearn.decomposition import NMF
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

    excluded = pd.read_csv(EXCLUSION_PATH, dtype={'id_establecimiento': str})
    excluded_ids = set(excluded['id_establecimiento'].str.strip())
    df['id_establecimiento'] = df['id_establecimiento'].astype(str).str.strip()
    mask = df['id_establecimiento'].isin(excluded_ids)
    n_excluded_imgs = mask.sum()
    n_excluded_schools = df.loc[mask, 'id_establecimiento'].nunique()
    df = df[~mask].copy()
    log.info(f'Excluded {n_excluded_imgs} images from {n_excluded_schools} rural schools, {len(df)} images remaining')

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
# NMF
# =============================================================================
def run_nmf(
    X_norm: np.ndarray,
    df_meta: pd.DataFrame,
    nombres: dict,
    K: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Ajusta NMF con K tópicos sobre imágenes individuales.
    Devuelve proporciones (L1-normalizadas) por imagen y por establecimiento.
    """
    log.info(f'  Ajustando NMF con K={K} sobre {len(X_norm):,} imágenes...')

    nmf = NMF(
        n_components=K,
        init='nndsvda',      # inicialización determinista basada en SVD
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
        l1_ratio=0.0,        # regularización L2 pura (más estable)
    )
    W = nmf.fit_transform(X_norm)   # [N_images, K] — proporciones no normalizadas

    log.info(f'  Error de reconstrucción: {nmf.reconstruction_err_:.4f}')
    log.info(f'  Iteraciones:             {nmf.n_iter_}')

    # Normalizar W por fila (L1) → proporciones que suman 1
    W_norm = normalize(W, norm='l1')

    topic_cols = [f'topic_{k+1}' for k in range(K)]

    # --- Por imagen ---
    df_img = pd.DataFrame(W_norm, columns=topic_cols)
    for col in ['id_establecimiento', 'id_sede', 'heading']:
        df_img.insert(list(df_img.columns).index(topic_cols[0]), col, df_meta[col].values)

    # --- Por establecimiento (media de proporciones) ---
    df_est = (
        df_img
        .groupby('id_establecimiento')[topic_cols]
        .mean()
        .reset_index()
    )

    # Top-N features por tópico (índices en el espacio 512d original)
    topics_json = {
        f'topic_{k+1}': {
            'top_features': np.argsort(nmf.components_[k])[::-1][:TOP_N_FEAT].tolist(),
            'feature_space': 'raw_512d',
        }
        for k in range(K)
    }

    return df_img, df_est, topics_json, nmf


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
    log.info('APRENDIZAJE DE TÓPICOS VISUALES CON NMF')
    log.info('=' * 60)

    df_raw, X = load_embeddings()
    nombres = load_nombres()

    # Normalización L2 por fila — estabiliza escala sin destruir no-negatividad
    log.info('Normalizando vectores (L2)...')
    X_norm = normalize(X, norm='l2')
    log.info(f'  Norma L2 media: {np.linalg.norm(X_norm, axis=1).mean():.4f}')

    for K in K_VALUES:
        log.info(f'\n{"─"*60}')
        log.info(f'K = {K}')
        log.info(f'{"─"*60}')

        df_img, df_est, topics_json, nmf = run_nmf(X_norm, df_raw, nombres, K)

        img_path = os.path.join(EMBEDDINGS_DIR, f'gsv_nmf_K{K}_images.parquet')
        df_img['id_establecimiento'] = df_img['id_establecimiento'].astype(str)
        df_img.to_parquet(img_path, index=False)
        log.info(f'  Por imagen guardado:          {img_path}  {df_img.shape}')

        est_path = os.path.join(EMBEDDINGS_DIR, f'gsv_nmf_K{K}.parquet')
        df_est['id_establecimiento'] = df_est['id_establecimiento'].astype(str)
        df_est.to_parquet(est_path, index=False)
        log.info(f'  Por establecimiento guardado: {est_path}  {df_est.shape}')

        json_path = os.path.join(EMBEDDINGS_DIR, f'gsv_nmf_K{K}_topics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(topics_json, f, indent=2)
        log.info(f'  Top features guardados:       {json_path}')

        print_topic_stats(df_est, nombres, K)

        if K == K_DEFAULT:
            log.info(f'\n  *** K={K} es el modelo principal (K_DEFAULT) ***')

    log.info('\n' + '=' * 60)
    log.info('RESUMEN')
    for K in K_VALUES:
        tag = ' ← PRINCIPAL' if K == K_DEFAULT else ''
        log.info(f'  K={K}: gsv_nmf_K{K}.parquet + gsv_nmf_K{K}_images.parquet{tag}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
