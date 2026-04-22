#!/usr/bin/env python3
"""
02_extract_embeddings.py
Extrae embeddings visuales de imágenes GSV usando VGG19 preentrenado en ImageNet.

Metodología:
  - VGG19 sin cabeza de clasificación (hasta block5_pool + global average pool)
  - Vector de 512 dimensiones por imagen (no-negativo por ReLU — válido para LDA)
  - Promedio de N headings por sede → 1 vector por sede
  - Promedio de sedes por establecimiento → 1 vector por establecimiento (558 total)

Inputs:
  data/images/gsv/gsv_catalog.csv
  data/images/gsv/{id_sede}/{id_sede}_{heading:03d}.jpg

Outputs:
  data/images/embeddings/gsv_vgg19_raw.parquet            (por imagen)
  data/images/embeddings/gsv_vgg19_establecimiento.parquet (por establecimiento, 558 filas × 512 cols)
"""

# =============================================================================
# CONFIGURACIÓN — modificar aquí antes de correr
# =============================================================================
MODE = 'full'   # 'sample' → solo 10 establecimientos | 'full' → todos
BATCH_SIZE = 32   # imágenes por lote (reducir si hay problemas de memoria)

CATALOG_PATH   = 'data/images/gsv/gsv_catalog.csv'
GSV_DIR        = 'data/images/gsv'
EMBEDDINGS_DIR = 'data/images/embeddings'

# =============================================================================
# IMPORTACIONES
# =============================================================================
import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

# Suprimir advertencias de deprecación de pesos
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# =============================================================================
# DIRECTORIO DE SALIDA
# =============================================================================
Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# DISPOSITIVO (GPU si está disponible, CPU en caso contrario)
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Dispositivo: {device}')

# =============================================================================
# MODELO VGG19 — extraer hasta block5_pool + global average pooling → 512d
# =============================================================================
def build_extractor():
    """Carga VGG19 y construye extractor de características (512d por imagen)."""
    try:
        # API moderna de torchvision
        from torchvision.models import vgg19, VGG19_Weights
        base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        log.info('VGG19 cargado con VGG19_Weights.IMAGENET1K_V1')
    except (ImportError, AttributeError):
        # Fallback para versiones antiguas de torchvision
        from torchvision.models import vgg19
        base = vgg19(pretrained=True)
        log.info('VGG19 cargado con pretrained=True (API legacy)')

    # Extractor: features (hasta block5_pool) → global avg pool → flatten → 512d
    extractor = nn.Sequential(
        base.features,                    # [B, 512, 7, 7] para entrada 224×224
        nn.AdaptiveAvgPool2d((1, 1)),     # [B, 512, 1, 1]
        nn.Flatten(),                     # [B, 512]
    ).eval().to(device)

    # Congelar parámetros (solo inferencia)
    for param in extractor.parameters():
        param.requires_grad = False

    return extractor


# =============================================================================
# TRANSFORMACIONES ESTÁNDAR DE IMAGENET
# =============================================================================
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# CARGA DEL CATÁLOGO GSV
# =============================================================================
def load_catalog(mode: str) -> pd.DataFrame:
    """Carga y filtra el catálogo GSV según el modo de ejecución."""
    log.info(f'Cargando catálogo: {CATALOG_PATH}')
    catalog = pd.read_csv(CATALOG_PATH)
    log.info(f'  Total filas en catálogo: {len(catalog):,}')

    # Filtrar solo imágenes descargadas
    catalog = catalog[catalog['descargada'].astype(str).str.strip() == 'True'].copy()
    log.info(f'  Imágenes descargadas: {len(catalog):,}')

    # Construir ruta completa
    catalog['filepath'] = catalog['ruta_archivo'].apply(
        lambda r: os.path.join(GSV_DIR, r)
    )

    if mode == 'sample':
        # Solo los primeros 10 establecimientos únicos
        sample_ids = catalog['id_establecimiento'].unique()[:10]
        catalog = catalog[catalog['id_establecimiento'].isin(sample_ids)].copy()
        log.info(f'  MODO SAMPLE: {len(sample_ids)} establecimientos, {len(catalog)} imágenes')
    else:
        n_est = catalog['id_establecimiento'].nunique()
        log.info(f'  MODO FULL: {n_est} establecimientos, {len(catalog)} imágenes')

    return catalog


# =============================================================================
# PROCESAMIENTO POR LOTES
# =============================================================================
def process_batch(filepaths: list, extractor: nn.Module) -> np.ndarray:
    """
    Carga un lote de imágenes, aplica transformación y extrae embeddings.
    Devuelve array [N, 512] o [N_válidas, 512] si algunas imágenes fallan.
    """
    tensors = []
    valid_indices = []

    for i, fp in enumerate(filepaths):
        try:
            img = Image.open(fp).convert('RGB')
            tensors.append(imagenet_transform(img))
            valid_indices.append(i)
        except Exception as e:
            log.warning(f'No se pudo cargar imagen: {fp} — {e}')

    if not tensors:
        return np.empty((0, 512)), valid_indices

    batch_tensor = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = extractor(batch_tensor)

    return embeddings.cpu().numpy(), valid_indices


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    log.info('=' * 60)
    log.info(f'EXTRACCIÓN DE EMBEDDINGS VGG19  |  MODE={MODE}')
    log.info('=' * 60)

    # -------------------------------------------------------------------------
    # 1. Cargar catálogo
    # -------------------------------------------------------------------------
    catalog = load_catalog(MODE)

    # -------------------------------------------------------------------------
    # 2. Construir extractor VGG19
    # -------------------------------------------------------------------------
    log.info('Construyendo extractor VGG19...')
    extractor = build_extractor()
    log.info('  Extractor listo (block5_pool → global avg pool → 512d)')

    # -------------------------------------------------------------------------
    # 3. Procesar imágenes en lotes y guardar embeddings por imagen
    # -------------------------------------------------------------------------
    log.info('Extrayendo embeddings por imagen...')

    all_embeddings = []
    all_meta       = []

    filepaths  = catalog['filepath'].tolist()
    ids_est    = catalog['id_establecimiento'].tolist()
    ids_sede   = catalog['id_sede'].tolist()
    headings   = catalog['heading'].tolist()

    n_total   = len(filepaths)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    n_errors  = 0

    for b in tqdm(range(n_batches), desc='Lotes', unit='lote'):
        start = b * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n_total)

        batch_fps  = filepaths[start:end]
        batch_embs, valid_idx = process_batch(batch_fps, extractor)

        n_invalid = (end - start) - len(valid_idx)
        n_errors += n_invalid

        for local_i, emb in zip(valid_idx, batch_embs):
            global_i = start + local_i
            all_embeddings.append(emb)
            all_meta.append({
                'id_establecimiento': ids_est[global_i],
                'id_sede':            ids_sede[global_i],
                'heading':            headings[global_i],
                'filepath':           filepaths[global_i],
            })

    log.info(f'  Imágenes procesadas: {len(all_embeddings):,}')
    log.info(f'  Imágenes con error:  {n_errors:,}')

    # -------------------------------------------------------------------------
    # 4. Guardar embeddings raw (por imagen)
    # -------------------------------------------------------------------------
    emb_array = np.array(all_embeddings)   # [N, 512]
    emb_cols  = [f'f_{i:03d}' for i in range(512)]

    df_raw = pd.DataFrame(emb_array, columns=emb_cols)
    df_raw = pd.concat([
        pd.DataFrame(all_meta).reset_index(drop=True),
        df_raw,
    ], axis=1)

    raw_path = os.path.join(EMBEDDINGS_DIR, 'gsv_vgg19_raw.parquet')
    df_raw.to_parquet(raw_path, index=False)
    log.info(f'Embeddings por imagen guardados: {raw_path}  ({df_raw.shape})')

    # -------------------------------------------------------------------------
    # 5. Promedio por sede
    # -------------------------------------------------------------------------
    log.info('Promediando embeddings por sede...')
    feature_cols = emb_cols

    df_sede = (
        df_raw
        .groupby(['id_establecimiento', 'id_sede'])[feature_cols]
        .mean()
        .reset_index()
    )
    log.info(f'  Sedes con embedding: {len(df_sede):,}')

    # -------------------------------------------------------------------------
    # 6. Promedio por establecimiento
    # -------------------------------------------------------------------------
    log.info('Promediando embeddings por establecimiento...')
    df_est = (
        df_sede
        .groupby('id_establecimiento')[feature_cols]
        .mean()
        .reset_index()
    )
    log.info(f'  Establecimientos con embedding: {len(df_est):,}')

    est_path = os.path.join(EMBEDDINGS_DIR, 'gsv_vgg19_establecimiento.parquet')
    df_est.to_parquet(est_path, index=False)
    log.info(f'Embeddings por establecimiento guardados: {est_path}  ({df_est.shape})')

    # -------------------------------------------------------------------------
    # 7. Resumen final
    # -------------------------------------------------------------------------
    log.info('=' * 60)
    log.info('RESUMEN')
    log.info(f'  Imágenes procesadas:        {len(all_embeddings):,}')
    log.info(f'  Imágenes con error:         {n_errors:,}')
    log.info(f'  Sedes con embedding:        {len(df_sede):,}')
    log.info(f'  Establecimientos con emb.:  {len(df_est):,}')
    log.info(f'  Dimensión del vector:       512')
    log.info('Archivos creados:')
    log.info(f'  {raw_path}')
    log.info(f'  {est_path}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
