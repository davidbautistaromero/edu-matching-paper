#!/usr/bin/env python3
"""
02b_seg_cityscapes.py
=====================
Segmentación semántica de imágenes de calle usando DeepLabV3+ entrenado
en Cityscapes (19 clases urbanas).

¿Qué hace este script?
  1. Lee las imágenes .jpg de una carpeta
  2. Extrae el id del establecimiento desde el nombre del archivo
  3. Pasa cada imagen por DeepLabV3+ → cada píxel recibe una etiqueta
     (road, building, vegetation, sky, person, car, etc.)
  4. Cuenta qué proporción de píxeles pertenece a cada categoría
  5. Toma el máximo de las proporciones de todas las fotos del mismo establecimiento
  6. Guarda los resultados como .parquet listo para regresión

Formato esperado del nombre de archivo:
  {id_establecimiento}_{fecha}_{id_foto}.jpg
  ejemplo: 111001010031_2022-06-20_138365378811821.jpg

Ejecutar desde cualquier directorio:
  python /ruta/a/scripts/02b_seg_cityscapes.py
"""

# =============================================================================
# BLOQUE 1: IMPORTACIONES
# Cargamos todas las librerías que el script necesita.
# =============================================================================

import os       # para manejar rutas de archivos y carpetas
import sys      # para modificar el path de Python (necesario para importar 'deeplabv3')
import logging  # para imprimir mensajes con hora y nivel (INFO, WARNING, etc.)
from pathlib import Path  # para crear carpetas si no existen

import numpy as np          # operaciones numéricas sobre arrays (los mapas de píxeles)
import pandas as pd         # para construir y guardar tablas de resultados
from PIL import Image       # para abrir imágenes .jpg
from tqdm import tqdm       # barra de progreso en el loop principal
import torch                # framework de deep learning
from torchvision import transforms  # transformaciones estándar de imagen (resize, normalizar)

# El módulo 'deeplabv3' vive en la misma carpeta que este script.
# Usamos __file__ para que la ruta sea absoluta y funcione sin importar
# desde qué directorio se ejecute el script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deeplabv3 import modeling  # aquí está la definición de DeepLabV3+

# Raiz del repositorio, derivada de la ubicacion de este script.
# Evita rutas absolutas y hace el pipeline reproducible en cualquier maquina.
_ROOT = Path(__file__).resolve().parent.parent



# =============================================================================
# BLOQUE 2: CONFIGURACIÓN
# Las únicas líneas que deberías necesitar cambiar para adaptar el script.
# =============================================================================

# Carpeta con las imágenes .jpg a segmentar
IMAGES_DIR = str(_ROOT / 'data' / 'images' / 'gsv')

# Carpeta donde se guardarán los outputs (se crea automáticamente si no existe)
OUT_DIR = str(_ROOT / 'data' / 'images' / 'segmentation')

# CSV con IDs de escuelas rurales a excluir del análisis
EXCLUSION_PATH = str(_ROOT / 'data' / 'raw' / 'excluded_schools.csv')

# Ruta al checkpoint del modelo preentrenado en Cityscapes (449 MB, no se mueve)
CKPT_PATH = str(_ROOT / 'checkpoints' / 'best_deeplabv3plus_resnet101_cityscapes_os16.pth')

# Modo de ejecución:
#   'sample' → procesa solo los primeros 10 establecimientos (para probar)
#   'full'   → procesa todos los establecimientos de la carpeta
MODE = 'full'

# Número de clases de Cityscapes (siempre 19, no cambiar)
NUM_CLASSES = 19

# Crear carpeta de output si no existe
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Detectar si hay GPU disponible; si no, usar CPU
# En tu Mac sin GPU dedicada esto será siempre 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurar el sistema de logging para imprimir mensajes con hora
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# =============================================================================
# BLOQUE 3: DEFINICIÓN DE CLASES Y AGRUPACIÓN TEMÁTICA
#
# Cityscapes tiene 19 clases urbanas (índices 0 a 18):
#   0:road  1:sidewalk  2:building  3:wall  4:fence  5:pole
#   6:traffic light  7:traffic sign  8:vegetation  9:terrain
#   10:sky  11:person  12:rider  13:car  14:truck  15:bus
#   16:train  17:motorcycle  18:bicycle
#
# Para el análisis econométrico, agrupamos estas 19 clases en
# 7 categorías temáticas interpretables. Esta es la decisión
# metodológica más importante del script.
# =============================================================================

GRUPOS = {
    # Infraestructura vial: road (0) + sidewalk (1)
    # → proxy de pavimentación y accesibilidad peatonal al colegio
    'infraestructura_vial': [0, 1],

    # Edificación: solo building (2)
    # → presencia de estructuras construidas en el entorno inmediato
    # Separado de cerramiento para distinguir edificios visibles de muros
    'edificacion': [2],

    # Cerramiento: wall (3) + fence (4)
    # → proxy de aislamiento físico del colegio respecto al entorno
    # En Bogotá los muros perimetrales son muy frecuentes y tienen
    # una señal distinta a la edificación abierta
    'cerramiento': [3, 4],

    # Vegetación: vegetation (8) + terrain (9)
    # → cobertura verde; variable clave en literatura de bienestar urbano
    'vegetacion': [8, 9],

    # Vehículos: car(13) + truck(14) + bus(15) + train(16) + moto(17) + bici(18)
    # → tráfico; proxy de ruido, contaminación y peligrosidad vial
    'vehiculos': [13, 14, 15, 16, 17, 18],

    # Mobiliario urbano: pole(5) + traffic light(6) + traffic sign(7)
    # → formalidad urbana y mantenimiento del espacio público
    'mobiliario_urbano': [5, 6, 7],

    # Categoría de referencia (excluir en regresión para evitar multicolinealidad)
    # Agrupa: sky (10) + person (11) + rider (12)
    # sky: apertura espacial — poco interpretable como variable independiente
    # person/rider: varianza casi cero en fotos de fachadas escolares (~0.2%)
    'referencia': [10, 11, 12],
}

# Construimos un diccionario inverso: dado un índice de clase (0-18),
# devuelve el nombre del grupo temático al que pertenece.
# Ejemplo: IDX_TO_GRUPO[8] → 'vegetacion'
IDX_TO_GRUPO = {
    idx: grupo
    for grupo, indices in GRUPOS.items()
    for idx in indices
}

# Lista ordenada de nombres de grupos (serán las columnas del output)
GRUPO_NAMES = list(GRUPOS.keys())


# =============================================================================
# BLOQUE 4: PALETA DE COLORES PARA DIAGNÓSTICO VISUAL
#
# Cityscapes define un color RGB oficial por clase. Lo usamos para
# generar una imagen de diagnóstico que nos permite verificar visualmente
# que el modelo está segmentando correctamente.
#
# Orden: índice 0 (road) → índice 18 (bicycle)
# =============================================================================

PALETTE = [
    (128, 64, 128),   # 0:  road          → morado oscuro
    (244, 35, 232),   # 1:  sidewalk      → fucsia
    (70,  70,  70),   # 2:  building      → gris oscuro
    (102, 102, 156),  # 3:  wall          → gris azulado
    (190, 153, 153),  # 4:  fence         → rosado grisáceo
    (153, 153, 153),  # 5:  pole          → gris medio
    (250, 170,  30),  # 6:  traffic light → naranja
    (220, 220,   0),  # 7:  traffic sign  → amarillo
    (107, 142,  35),  # 8:  vegetation    → verde oliva
    (152, 251, 152),  # 9:  terrain       → verde claro
    (70,  130, 180),  # 10: sky           → azul acero
    (220,  20,  60),  # 11: person        → rojo carmesí
    (255,   0,   0),  # 12: rider         → rojo puro
    (0,    0,  142),  # 13: car           → azul marino
    (0,    0,   70),  # 14: truck         → azul muy oscuro
    (0,   60,  100),  # 15: bus           → azul petróleo
    (0,   80,  100),  # 16: train         → azul verdoso
    (0,    0,  230),  # 17: motorcycle    → azul brillante
    (119,  11,  32),  # 18: bicycle       → granate
]


# =============================================================================
# BLOQUE 5: FUNCIÓN — CONSTRUIR CATÁLOGO DESDE NOMBRES DE ARCHIVO
#
# En lugar de leer un CSV externo, construimos la tabla de imágenes
# directamente desde los nombres de archivo .jpg en la carpeta.
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
            # '111001010031_000.jpg'
            #  → partes = ['111001010031', '000']
            partes = fname.replace('.jpg', '').split('_')

            # Si el nombre no tiene al menos 2 partes, el formato es inesperado
            if len(partes) < 2:
                log.warning(f'Nombre de archivo inesperado, se omite: {fname}')
                continue

            # La primera parte siempre es el id del establecimiento
            id_est = partes[0]

            registros.append({
                'id_establecimiento': id_est,
                'filename':           fname,
                'filepath':           os.path.join(root, fname),
            })

    df = pd.DataFrame(registros)
    log.info(f'Imágenes encontradas:       {len(df):,}')
    log.info(f'Establecimientos únicos:    {df["id_establecimiento"].nunique():,}')

    # Excluir escuelas rurales
    if os.path.exists(EXCLUSION_PATH):
        excluded = pd.read_csv(EXCLUSION_PATH, dtype={'id_establecimiento': str})
        excluded_ids = set(excluded['id_establecimiento'].str.strip())
        mask = df['id_establecimiento'].isin(excluded_ids)
        n_excluded_imgs = mask.sum()
        n_excluded_schools = df.loc[mask, 'id_establecimiento'].nunique()
        df = df[~mask].copy()
        log.info(f'Excluded {n_excluded_imgs} images from {n_excluded_schools} rural schools, {len(df)} images remaining')

    # En modo sample, quedarse solo con los primeros 10 establecimientos
    if mode == 'sample':
        ids = df['id_establecimiento'].unique()[:10]
        df  = df[df['id_establecimiento'].isin(ids)].copy()
        log.info(f'MODO SAMPLE: {len(ids)} establecimientos, {len(df)} imágenes')

    return df


# =============================================================================
# BLOQUE 6: FUNCIÓN — CARGAR MODELO DEEPLABV3+
#
# Carga la arquitectura del modelo y le aplica los pesos preentrenados
# en Cityscapes desde el archivo .pth descargado.
# =============================================================================

def build_model():
    """
    Carga DeepLabV3+ con backbone ResNet-101 y pesos de Cityscapes.

    ¿Qué hace internamente?
      1. Crea la arquitectura vacía (sin pesos)
      2. Carga el archivo .pth que contiene los pesos aprendidos
      3. Inyecta esos pesos en la arquitectura
      4. Pone el modelo en modo evaluación (no entrena, solo predice)
      5. Congela los parámetros (ahorra memoria, acelera la inferencia)
    """
    log.info(f'Cargando checkpoint: {CKPT_PATH}')

    # Crear la arquitectura vacía de DeepLabV3+ con ResNet-101
    # num_classes=19 porque Cityscapes tiene 19 clases
    # output_stride=16 es la configuración estándar (relación entre
    # tamaño de entrada y tamaño del mapa de features interno)
    model = modeling.__dict__['deeplabv3plus_resnet101'](
        num_classes=NUM_CLASSES,
        output_stride=16,
    )

    # Cargar el archivo .pth en memoria (map_location='cpu' porque
    # no tenemos GPU; evita errores de dispositivo)
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)

    # El checkpoint guarda los pesos bajo la clave 'model_state'.
    # Si por alguna razón no existe esa clave, usamos el dict completo.
    state = ckpt.get('model_state', ckpt)

    # Inyectar los pesos en la arquitectura
    model.load_state_dict(state)

    # Modo evaluación: desactiva dropout y batch normalization de entrenamiento
    model.eval()

    # Mover el modelo al dispositivo (cpu en tu caso)
    model.to(device)

    # Congelar parámetros: le decimos a PyTorch que no necesita
    # calcular gradientes, lo que ahorra memoria y acelera la inferencia
    for p in model.parameters():
        p.requires_grad = False

    log.info('Modelo listo (DeepLabV3+ ResNet-101, 19 clases Cityscapes)')
    return model


# =============================================================================
# BLOQUE 7: TRANSFORMACIÓN DE IMAGEN
#
# Antes de pasarle una imagen al modelo, hay que prepararla:
#   1. Redimensionar a 512×1024 (resolución nativa de Cityscapes)
#   2. Convertir a tensor (array numérico que PyTorch entiende)
#   3. Normalizar con la media y desviación estándar de ImageNet
#      (los mismos valores que se usaron al preentrenar la red)
#
# La normalización es crítica: sin ella el modelo "ve" colores
# completamente distintos a los que aprendió durante el entrenamiento.
# =============================================================================

transform = transforms.Compose([
    # Redimensionar a alto=512, ancho=1024
    # Las imágenes GSV son cuadradas, así que se estiran un poco
    # horizontalmente — aceptable para contar proporciones de área
    transforms.Resize((512, 1024)),

    # Convertir PIL Image → tensor de PyTorch [C, H, W] con valores en [0, 1]
    transforms.ToTensor(),

    # Normalizar: (píxel - media) / desviación_estándar por canal RGB
    # Estos valores son estándar para redes preentrenadas en ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    ),
])


# =============================================================================
# BLOQUE 8: FUNCIÓN — SEGMENTAR UNA IMAGEN
#
# Toma la ruta de una imagen, la pasa por el modelo, y devuelve:
#   - props     : dict {grupo: proporción} donde la suma ≈ 1.0
#   - class_map : array [H, W] con el índice de clase de cada píxel
# =============================================================================

def segmentar(filepath: str, model):
    """
    Segmenta una imagen y extrae proporciones por grupo temático.

    Parámetros:
      filepath : ruta completa al archivo .jpg
      model    : modelo DeepLabV3+ ya cargado

    Retorna:
      (props, class_map) o None si la imagen no se puede abrir
    """

    # Intentar abrir la imagen; si falla (archivo corrupto, etc.) retornar None
    try:
        img = Image.open(filepath).convert('RGB')
    except Exception as e:
        log.warning(f'Error abriendo imagen: {filepath} — {e}')
        return None

    # Aplicar la transformación definida arriba
    # unsqueeze(0) agrega una dimensión de batch: [C,H,W] → [1,C,H,W]
    # porque el modelo espera un lote (batch) de imágenes, aunque sea de 1
    tensor = transform(img).unsqueeze(0).to(device)

    # Inferencia: pasar el tensor por el modelo
    # torch.no_grad() le dice a PyTorch que no guarde gradientes
    # (no estamos entrenando, solo prediciendo) → ahorra memoria
    with torch.no_grad():
        output = model(tensor)   # shape: [1, 19, 512, 1024]
        # Para cada píxel tenemos 19 números (uno por clase).
        # Cada número es la "confianza" del modelo en esa clase.

    # argmax(dim=1): para cada píxel, elegir la clase con mayor confianza
    # squeeze(): eliminar la dimensión de batch → [512, 1024]
    # .cpu().numpy(): convertir tensor de PyTorch a array de NumPy
    class_map = output.argmax(dim=1).squeeze().cpu().numpy()
    # Ahora class_map es una matriz de 512×1024 donde cada valor
    # es un entero 0-18 que indica la clase ganadora de ese píxel

    # Contar píxeles por grupo y normalizar a proporciones
    total = class_map.size   # total de píxeles = 512 × 1024 = 524.288

    props = {g: 0.0 for g in GRUPO_NAMES}  # inicializar en 0

    for idx in range(NUM_CLASSES):
        # Buscar a qué grupo pertenece este índice de clase
        grupo = IDX_TO_GRUPO.get(idx)
        if grupo:
            # Contar cuántos píxeles tienen este índice y acumular en el grupo
            # (class_map == idx) crea una máscara booleana True/False
            # .sum() cuenta los True → número de píxeles de esa clase
            props[grupo] += float((class_map == idx).sum()) / total

    return props, class_map


# =============================================================================
# BLOQUE 9: FUNCIÓN — GUARDAR MAPA DE COLOR PARA DIAGNÓSTICO
#
# Convierte el mapa de clases (números 0-18) en una imagen RGB
# usando los colores oficiales de Cityscapes. Sirve para verificar
# visualmente que el modelo está funcionando correctamente.
# =============================================================================

def guardar_mapa_color(class_map, ruta_out):
    """
    Genera imagen de diagnóstico coloreada según la paleta de Cityscapes.
    road=morado, vegetation=verde, sky=azul, building=gris, etc.
    """
    h, w   = class_map.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)  # imagen en negro

    # Para cada clase, pintar sus píxeles con el color correspondiente
    for idx, color in enumerate(PALETTE):
        canvas[class_map == idx] = color

    Image.fromarray(canvas).save(ruta_out)
    log.info(f'Diagnóstico visual guardado: {ruta_out}')


# =============================================================================
# BLOQUE 10: FUNCIÓN PRINCIPAL — LOOP SOBRE TODAS LAS IMÁGENES
# =============================================================================

def main():
    log.info('=' * 60)
    log.info(f'SEGMENTACIÓN CITYSCAPES  |  MODE={MODE}  |  device={device}')
    log.info('=' * 60)

    # ── Paso 1: Construir el catálogo de imágenes ─────────────────────────
    catalog = build_catalog(IMAGES_DIR, MODE)

    # ── Paso 2: Cargar el modelo ──────────────────────────────────────────
    model = build_model()

    # ── Paso 3: Procesar cada imagen ──────────────────────────────────────
    records          = []    # aquí acumulamos los resultados fila por fila
    primera_guardada = False  # bandera para guardar solo 1 imagen diagnóstico

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc='Segmentando'):

        resultado = segmentar(row['filepath'], model)

        # Si la imagen falló, saltar a la siguiente
        if resultado is None:
            continue

        props, class_map = resultado

        # Guardar imagen de diagnóstico solo para la primera foto procesada
        # Ábrela después para verificar que los colores tienen sentido
        if not primera_guardada:
            diag_path = os.path.join(OUT_DIR, 'diagnostico_primera_foto.png')
            guardar_mapa_color(class_map, diag_path)
            primera_guardada = True

        # Construir la fila de resultados para esta imagen
        record = {
            'id_establecimiento': row['id_establecimiento'],
            'filename':           row['filename'],
        }
        record.update(props)  # agregar las 7 proporciones
        records.append(record)

    # ── Paso 4: Armar DataFrame con todos los resultados ──────────────────
    df = pd.DataFrame(records)
    # Cada fila = una imagen, con sus 7 proporciones de categorías

    # ── Paso 5: Agregar por establecimiento (máximo sobre todas sus fotos) ──
    # Un establecimiento puede tener varias fotos desde distintos ángulos.
    # Tomamos el máximo por categoría: si una sola foto muestra fachada moderna
    # o alta cobertura verde, esa señal se preserva en lugar de diluirse con
    # el promedio de fotos del entorno.
    # Nota: las proporciones ya no suman 1 entre grupos (cada una viene de la
    # foto que maximiza esa categoría), pero son válidas como features separadas.
    df_est = (
        df
        .groupby('id_establecimiento')[GRUPO_NAMES]
        .max()
        .reset_index()
    )

    # ── Paso 6: Guardar outputs ───────────────────────────────────────────

    # Raw: una fila por imagen (útil para análisis detallado)
    raw_path = f'{OUT_DIR}/gsv_cs_raw.parquet'
    df.to_parquet(raw_path, index=False)

    # Agregado: una fila por establecimiento (el que usarás en regresión)
    est_path = f'{OUT_DIR}/gsv_cs_establecimiento.parquet'
    df_est.to_parquet(est_path, index=False)

    # ── Paso 7: Resumen final ─────────────────────────────────────────────
    log.info('=' * 60)
    log.info('RESUMEN')
    log.info(f'  Imágenes procesadas:        {len(df):,}')
    log.info(f'  Establecimientos con datos: {len(df_est):,}')
    log.info('  Máximos medios por grupo (promedio del max entre establecimientos):')
    for g in GRUPO_NAMES:
        log.info(f'    {g:<25} {df_est[g].mean():.3f}')
    log.info(f'  Archivos guardados en: {OUT_DIR}')
    log.info('=' * 60)


# =============================================================================
# PUNTO DE ENTRADA
# Esta línea hace que main() solo se ejecute cuando corres el script
# directamente (python 02b_seg_cityscapes.py), no cuando lo importas
# desde otro script.
# =============================================================================

if __name__ == '__main__':
    main()