"""
gsv_config.py
=============
Parámetros de configuración para la descarga de Google Street View.
Fuente de verdad única — cambiar aquí afecta 00_download_gsv_colegios.py.

MODO_MUESTRA      : True para pruebas con N_MUESTRA sedes aleatorias.
N_MUESTRA         : número de sedes en modo prueba.
N_HEADINGS        : imágenes por sede (headings uniformes 360°/N_HEADINGS).
IMG_SIZE          : resolución de cada imagen (máx 640x640 en tier gratuito).
FOV               : campo visual en grados (90 = perspectiva natural).
PITCH             : ángulo vertical (0 = horizonte, positivo = arriba).
FORZAR_REDESCARGA : True para re-descargar aunque el archivo ya exista.
MAX_CONCURRENT    : máximo de requests HTTP simultáneos.
"""

# ---------------------------------------------------------------------------
# Modo de prueba — pon MODO_MUESTRA = True para probar con pocas sedes
# ---------------------------------------------------------------------------
MODO_MUESTRA = False
N_MUESTRA    = 5

# ---------------------------------------------------------------------------
# Parámetros de imagen
# ---------------------------------------------------------------------------
N_HEADINGS = 10          # headings: 0, 36, 72, 108, 144, 180, 216, 252, 288, 324
IMG_SIZE   = "640x640"
FOV        = 90
PITCH      = 0

# ---------------------------------------------------------------------------
# Control de descarga
# ---------------------------------------------------------------------------
FORZAR_REDESCARGA = False   # False = salta imágenes ya en disco
MAX_CONCURRENT    = 10      # semáforo asyncio
MAX_REINTENTOS    = 3       # reintentos ante 429 / error de red
ESPERA_429_S      = 5       # segundos de espera tras un 429
MIN_SIZE_BYTES    = 5_000   # imágenes < 5 KB = "no imagery available"
