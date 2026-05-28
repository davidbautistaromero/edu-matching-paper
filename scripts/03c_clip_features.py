#!/usr/bin/env python3
"""
03c_clip_features.py
====================
Extracción de características perceptuales de imágenes GSV usando CLIP (ViT-B/32).

¿Qué hace este script?
  1. Lee las imágenes .jpg de establecimientos educativos de Bogotá
  2. Para cada imagen, calcula un score de similitud semántica contra
     4 dimensiones perceptuales definidas como pares positivo/negativo
  3. Score = cosine_similarity(imagen, frase_positiva)
             − cosine_similarity(imagen, frase_negativa)
     → > 0 : la imagen se parece más a la descripción positiva
     → < 0 : la imagen se parece más a la descripción negativa
  4. Toma el máximo de los scores de todas las fotos (headings) del mismo establecimiento
  5. Guarda los resultados como .parquet listo para regresión

Las 4 dimensiones capturan: mantenimiento físico, vegetación percibida,
accesibilidad del ingreso y seguridad percibida del entorno inmediato.

Referencias:
  Radford et al. (2021) — "Learning Transferable Visual Models From Natural
    Language Supervision". ICML. CLIP: zero-shot visual-semantic embeddings
    mediante aprendizaje contrastivo en 400M pares imagen-texto de internet.
  Naik et al. (2017) — "Computer vision uncovers predictors of physical urban
    change". PNAS. Primer uso sistemático de percepción visual de calle para
    predecir cambio urbano y bienestar en ciudades.
  Dubey et al. (2016) — "Deep Learning the City: Quantifying Urban Perception
    At A Global Scale". ECCV. Place Pulse: percepción pairwise de seguridad,
    riqueza y animación usando street-view imagery a escala global.

Instalación requerida:
  pip install git+https://github.com/openai/CLIP.git

Formato esperado del nombre de archivo:
  {id_establecimiento}_{heading:03d}.jpg
  ejemplo: 111001010031_000.jpg

Ejecutar desde cualquier directorio:
  python /ruta/a/scripts/02c_clip_features.py
"""

# =============================================================================
# BLOQUE 1: IMPORTACIONES
# Cargamos todas las librerías que el script necesita.
# =============================================================================

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import clip  # pip install git+https://github.com/openai/CLIP.git


# =============================================================================
# BLOQUE 2: CONFIGURACIÓN
# Las únicas líneas que deberías necesitar cambiar para adaptar el script.
# =============================================================================

# Carpeta raíz con subdirectorios por establecimiento, cada uno con .jpg
IMAGES_DIR = r'C:\paper-AI\data\images\gsv'

# Carpeta donde se guardarán los outputs (se crea automáticamente si no existe)
OUT_DIR = r'C:\paper-AI\data\images\clip'

# CSV con IDs de escuelas rurales a excluir del análisis
EXCLUSION_PATH = 'data/raw/excluded_schools.csv'

# Modo de ejecución:
#   'sample' → procesa solo los primeros 10 establecimientos (para probar)
#   'full'   → procesa todos los establecimientos de la carpeta
MODE = 'full'

# Variante de CLIP. ViT-B/32 es el modelo estándar de la literatura urbana:
# buen balance entre velocidad de inferencia y calidad del embedding.
CLIP_MODEL = 'ViT-B/32'

# Número de imágenes procesadas en paralelo por el encoder de imagen.
# 32 es seguro en CPU y en GPUs con ≥ 4 GB VRAM.
BATCH_SIZE = 32

# Crear carpeta de output si no existe
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Detectar GPU automáticamente; en CPU la inferencia es más lenta pero correcta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurar el sistema de logging para imprimir mensajes con hora
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# =============================================================================
# BLOQUE 3: PARES DE FRASES CLIP
#
# Cada dimensión perceptual se define mediante un par de frases en inglés:
#   (frase_positiva, frase_negativa)
#
# IMPORTANTE: estas frases son fijas para garantizar replicabilidad del paper.
# Cualquier modificación debe quedar explícitamente documentada en el texto.
#
# La elección de frases en inglés es deliberada: CLIP fue preentrenado
# principalmente en texto en inglés, por lo que sus embeddings de texto
# tienen mayor resolución semántica en ese idioma.
# =============================================================================

PARES = {
    'mantenimiento': (
        "a school building with a clean and well-maintained facade",
        "a school building with a deteriorated and neglected facade",
    ),
    'vegetacion_percibida': (
        "a school surrounded by trees and green areas",
        "a school with no vegetation or green spaces around it",
    ),
    'modernidad': (
        "a modern and recently built school building",
        "an old and outdated school building",
    ),
    'seguridad_percibida': (
        "a school in a safe and calm street environment",
        "a school in a dangerous and chaotic street environment",
    ),
}

# Lista ordenada de nombres de dimensiones (serán las columnas del output)
DIMENSIONES = list(PARES.keys())


# =============================================================================
# BLOQUE 4: FUNCIÓN — CONSTRUIR CATÁLOGO DESDE NOMBRES DE ARCHIVO
#
# En lugar de leer un CSV externo, construimos la tabla de imágenes
# directamente desde los nombres de archivo .jpg en la carpeta.
# Misma lógica que 02b_seg_cityscapes.py para consistencia del pipeline.
# =============================================================================

def build_catalog(images_dir: str, mode: str) -> pd.DataFrame:
    """
    Recorre subdirectorios de images_dir y extrae id_establecimiento
    del nombre del archivo.

    El nombre sigue el formato:
      {id_establecimiento}_{heading:03d}.jpg
      ejemplo: 111001010031_000.jpg

    Las imágenes están en:
      images_dir/{id_establecimiento}/{id_establecimiento}_{heading:03d}.jpg

    Parámetros:
      images_dir : ruta raíz que contiene subdirectorios por establecimiento
      mode       : 'sample' (10 establecimientos) o 'full' (todos)

    Retorna:
      DataFrame con columnas: id_establecimiento, filename, filepath
    """
    registros = []

    for root, dirs, files in os.walk(images_dir):
        for fname in sorted(files):

            # Ignorar archivos que no sean .jpg
            if not fname.lower().endswith('.jpg'):
                continue

            # Separar el nombre por '_' para extraer el id del establecimiento
            # '111001010031_000.jpg' → partes = ['111001010031', '000']
            partes = fname.replace('.jpg', '').split('_')

            if len(partes) < 2:
                log.warning(f'Nombre de archivo inesperado, se omite: {fname}')
                continue

            id_est = partes[0]

            registros.append({
                'id_establecimiento': id_est,
                'filename':           fname,
                'filepath':           os.path.join(root, fname),
            })

    df = pd.DataFrame(registros)
    log.info(f'Imágenes encontradas:       {len(df):,}')
    log.info(f'Establecimientos únicos:    {df["id_establecimiento"].nunique():,}')

    # En modo sample, quedarse solo con los primeros 10 establecimientos
    if mode == 'sample':
        ids = df['id_establecimiento'].unique()[:10]
        df  = df[df['id_establecimiento'].isin(ids)].copy()
        log.info(f'MODO SAMPLE: {len(ids)} establecimientos, {len(df)} imágenes')

    return df


# =============================================================================
# BLOQUE 5: FUNCIÓN — CARGAR MODELO CLIP
#
# CLIP (Contrastive Language–Image Pre-Training, Radford et al. 2021) mapea
# imágenes y texto al mismo espacio de embedding de 512 dimensiones.
# La similitud coseno entre embeddings mide la afinidad semántica directamente,
# sin necesidad de fine-tuning ni ejemplos etiquetados — zero-shot.
# =============================================================================

def load_clip_model():
    """
    Carga el modelo CLIP ViT-B/32 y su función de preprocesamiento.

    clip.load() descarga los pesos la primera vez (~340 MB) y los cachea
    localmente en ~/.cache/clip para ejecuciones posteriores.

    Retorna:
      (model, preprocess) — modelo en modo eval y transform de imagen CLIP
    """
    log.info(f'Cargando modelo CLIP: {CLIP_MODEL}  |  device={device}')
    model, preprocess = clip.load(CLIP_MODEL, device=device)

    # Modo evaluación: desactiva dropout y batch normalization de entrenamiento
    model.eval()

    # Congelar parámetros: solo hacemos inferencia, no fine-tuning
    for p in model.parameters():
        p.requires_grad = False

    log.info('Modelo CLIP listo')
    return model, preprocess


# =============================================================================
# BLOQUE 6: FUNCIÓN — PRE-CODIFICAR FRASES DE TEXTO
#
# Codificamos los textos UNA sola vez antes del loop de imágenes.
# Reutilizar estos embeddings evita tokenizar y codificar las mismas
# 8 frases 5.580 veces, lo que reduciría la velocidad sin ganancia.
#
# Los embeddings se normalizan a norma unitaria: con vectores unitarios,
# coseno(a, b) = a · bᵀ (producto punto), lo que simplifica el cálculo.
# =============================================================================

def encode_text_phrases(model) -> dict:
    """
    Tokeniza y codifica las frases positiva y negativa de cada dimensión.

    Retorna:
      dict { dimensión: (emb_positivo, emb_negativo) }
      donde cada embedding tiene shape [1, 512], norma unitaria, en `device`
    """
    text_embeddings = {}

    with torch.no_grad():
        for dim, (pos_phrase, neg_phrase) in PARES.items():

            # Tokenizar: convierte texto a índices de vocabulario CLIP (BPE)
            tokens = clip.tokenize([pos_phrase, neg_phrase]).to(device)

            # Codificar con el text encoder de CLIP → shape [2, 512]
            emb = model.encode_text(tokens)

            # Normalizar: transforma a norma unitaria → producto punto = coseno
            emb = emb / emb.norm(dim=-1, keepdim=True)

            text_embeddings[dim] = (
                emb[0].unsqueeze(0),  # shape [1, 512] — frase positiva
                emb[1].unsqueeze(0),  # shape [1, 512] — frase negativa
            )

    log.info(f'Frases codificadas: {len(text_embeddings)} dimensiones × 2 frases')
    return text_embeddings


# =============================================================================
# BLOQUE 7: FUNCIÓN — PUNTUAR UN LOTE DE IMÁGENES
#
# Cargamos BATCH_SIZE imágenes en paralelo y las codificamos en una sola
# pasada por el encoder de imagen de CLIP, lo que es mucho más eficiente
# que procesar imagen por imagen.
#
# Para cada imagen calculamos:
#   score_dim = coseno(imagen, frase_pos) − coseno(imagen, frase_neg)
#
# El resultado está centrado en 0: positivo indica mayor afinidad con la
# descripción deseable, negativo con la descripción indeseable.
# =============================================================================

def score_batch(filepaths: list, model, preprocess, text_embeddings: dict) -> list:
    """
    Codifica un lote de imágenes y calcula los scores CLIP por dimensión.

    Parámetros:
      filepaths      : lista de rutas de imagen (hasta BATCH_SIZE elementos)
      model          : modelo CLIP cargado
      preprocess     : transform de CLIP (resize, crop, normalizar ImageNet)
      text_embeddings: output de encode_text_phrases()

    Retorna:
      lista de tuplas (filepath, scores_dict) para las imágenes procesadas
      con éxito; las imágenes que no se puedan abrir se omiten silenciosamente
    """
    tensors   = []
    valid_fps = []

    for fp in filepaths:
        try:
            img = Image.open(fp).convert('RGB')
            tensors.append(preprocess(img))
            valid_fps.append(fp)
        except Exception as e:
            log.warning(f'Error abriendo imagen: {fp} — {e}')

    if not tensors:
        return []

    # Apilar en batch: [B, C, H, W] donde B = número de imágenes válidas
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        # Codificar con el image encoder de CLIP → shape [B, 512]
        img_emb = model.encode_image(batch)
        # Normalizar: producto punto = coseno con vectores de texto unitarios
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    results = []
    for i, fp in enumerate(valid_fps):
        vec    = img_emb[i].unsqueeze(0)  # [1, 512]
        scores = {}
        for dim, (emb_pos, emb_neg) in text_embeddings.items():
            sim_pos    = (vec @ emb_pos.T).item()
            sim_neg    = (vec @ emb_neg.T).item()
            scores[dim] = sim_pos - sim_neg
        results.append((fp, scores))

    return results


# =============================================================================
# BLOQUE 8: FUNCIÓN PRINCIPAL — LOOP SOBRE TODAS LAS IMÁGENES
# =============================================================================

def main():
    log.info('=' * 60)
    log.info(f'CLIP FEATURES  |  MODE={MODE}  |  model={CLIP_MODEL}  |  device={device}')
    log.info('=' * 60)

    # ── Paso 1: Construir el catálogo de imágenes ─────────────────────────
    catalog = build_catalog(IMAGES_DIR, MODE)

    if catalog.empty:
        log.warning('No se encontraron imágenes. Verificar IMAGES_DIR.')
        return

    # ── Excluir escuelas rurales ───────────────────────────────────────────
    excluded = pd.read_csv(EXCLUSION_PATH, dtype={'id_establecimiento': str})
    excluded_ids = set(excluded['id_establecimiento'].str.strip())
    catalog['id_establecimiento'] = catalog['id_establecimiento'].astype(str).str.strip()
    mask = catalog['id_establecimiento'].isin(excluded_ids)
    n_excluded_imgs = mask.sum()
    n_excluded_schools = catalog.loc[mask, 'id_establecimiento'].nunique()
    catalog = catalog[~mask].copy()
    log.info(f'Excluded {n_excluded_imgs} images from {n_excluded_schools} rural schools, {len(catalog)} images remaining')

    # ── Paso 2: Cargar el modelo CLIP ─────────────────────────────────────
    model, preprocess = load_clip_model()

    # ── Paso 3: Pre-codificar frases de texto (una sola vez) ──────────────
    # Esto garantiza que los 8 embeddings de texto se computen exactamente
    # una vez y se reutilicen para las 5.580 imágenes.
    text_embeddings = encode_text_phrases(model)

    # ── Paso 4: Tabla de lookup rápida filepath → metadatos ───────────────
    # Evita hacer catalog[catalog['filepath'] == fp] en cada imagen (O(n))
    meta_lookup = {
        row['filepath']: (row['id_establecimiento'], row['filename'])
        for _, row in catalog.iterrows()
    }

    # ── Paso 5: Procesar imágenes en lotes de BATCH_SIZE ─────────────────
    records   = []
    filepaths = catalog['filepath'].tolist()
    n_batches = (len(filepaths) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in tqdm(range(0, len(filepaths), BATCH_SIZE),
                      total=n_batches, desc='Codificando imágenes'):
        batch_fps = filepaths[start:start + BATCH_SIZE]
        for fp, scores in score_batch(batch_fps, model, preprocess, text_embeddings):
            id_est, fname = meta_lookup[fp]
            record = {
                'id_establecimiento': id_est,
                'filename':           fname,
            }
            record.update(scores)
            records.append(record)

    # ── Paso 6: Armar DataFrame con todos los resultados ──────────────────
    df = pd.DataFrame(records)
    # Cada fila = una imagen, con sus 4 scores dimensionales

    # ── Paso 7: Agregar por establecimiento (máximo sobre todos los headings)
    # Cada establecimiento tiene hasta 10 fotos desde ángulos distintos
    # (headings 0°, 36°, …, 324°). El máximo preserva la señal perceptual
    # más fuerte: si la fachada aparece en un solo heading, su score no
    # se diluye con los ángulos que muestran el entorno deteriorado.
    df_est = (
        df
        .groupby('id_establecimiento')[DIMENSIONES]
        .max()
        .reset_index()
    )

    # ── Paso 8: Guardar outputs ───────────────────────────────────────────

    # Raw: una fila por imagen (útil para análisis de varianza intra-escuela)
    raw_path = f'{OUT_DIR}/gsv_clip_raw.parquet'
    df.to_parquet(raw_path, index=False)
    log.info(f'Raw guardado:      {raw_path}')

    # Agregado: una fila por establecimiento (el que usarás en regresión)
    est_path = f'{OUT_DIR}/gsv_clip_establecimiento.parquet'
    df_est.to_parquet(est_path, index=False)
    log.info(f'Agregado guardado: {est_path}')

    # ── Paso 9: Resumen final ─────────────────────────────────────────────
    log.info('=' * 60)
    log.info('RESUMEN')
    log.info(f'  Imágenes procesadas:        {len(df):,}')
    log.info(f'  Establecimientos con datos: {len(df_est):,}')
    log.info('  Score medio por dimensión (> 0 = afinidad positiva):')
    for dim in DIMENSIONES:
        log.info(f'    {dim:<25} {df_est[dim].mean():.4f}')
    log.info(f'  Archivos guardados en: {OUT_DIR}')
    log.info('=' * 60)


# =============================================================================
# PUNTO DE ENTRADA
# Esta línea hace que main() solo se ejecute cuando corres el script
# directamente (python 02c_clip_features.py), no cuando lo importas.
# =============================================================================

if __name__ == '__main__':
    main()
