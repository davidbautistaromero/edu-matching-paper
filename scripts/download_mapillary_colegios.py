#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_mapillary_colegios.py
==============================
Descarga las imágenes de Mapillary seleccionadas según los criterios
definidos en los parámetros. Lee el catálogo de metadatos generado por
analyze_mapillary_colegios.py y descarga solo las imágenes que cumplen
los filtros y aún no existen en disco.

PREREQUISITO
  Ejecutar primero analyze_mapillary_colegios.py para construir el catálogo:
    python scripts/analyze_mapillary_colegios.py

CRITERIOS DE SELECCIÓN (sincronizados con analyze_mapillary_colegios.py)
  · FECHA_DESDE     : solo imágenes capturadas a partir de esta fecha.
  · EXCLUIR_PANORAMICAS : excluye imágenes 360° incompatibles con VGG19.
  · DEDUP_POR_SECUENCIA : 1 imagen (la más cercana) por (colegio, secuencia).

REANUDABILIDAD
  El script comprueba si cada archivo .jpg ya existe en disco antes de
  descargar. Puede interrumpirse con Ctrl+C y reanudarse sin duplicados.
  Al terminar actualiza la columna 'descargada' en mapillary_catalog.csv.

SALIDA
  data/images/mapillary/
  └── {DANE12_EST}_{YYYY-MM-DD}_{image_id}.jpg   (una línea por imagen)

  El nombre de archivo identifica el colegio (DANE), la fecha de captura
  y el ID único de Mapillary, permitiendo cruzar imágenes con cualquier
  otro dataset del proyecto y citar la fuente exacta en el paper.

REQUISITOS
  pip install aiohttp pandas
"""

# ---------------------------------------------------------------------------
# Biblioteca estándar
# ---------------------------------------------------------------------------
import asyncio
import signal
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Terceros
# ---------------------------------------------------------------------------
import aiohttp
import pandas as pd


# =============================================================================
# PARÁMETROS — deben coincidir con analyze_mapillary_colegios.py
# =============================================================================

# Número máximo de peticiones HTTP simultáneas.
# Mapillary sirve imágenes desde CDN; 30 conexiones concurrentes logran
# ~20–40 MB/s en una red doméstica sin superar límites de la plataforma.
# Bajar a 15 si aparecen errores de conexión o timeouts frecuentes.
MAX_CONCURRENT = 30

# Resolución de los thumbnails a descargar.
# "thumb_2048_url" → 2048 px (recomendado para VGG19; ~400 KB/imagen).
# "thumb_1024_url" → 1024 px (más rápido; ~120 KB/imagen).
IMG_FIELD = "thumb_2048_url"

# --- Criterios de selección — MANTENER SINCRONIZADOS con analyze_mapillary_colegios.py ---

# Fecha mínima de captura. "" o None para no filtrar.
FECHA_DESDE = "2021-01-01"

# True → excluir panorámicas (is_pano = True).
# VGG19 fue entrenado en imágenes perspectiva estándar; las panorámicas
# tienen distorsión equirectangular que degrada los embeddings resultantes.
EXCLUIR_PANORAMICAS = True

# True → 1 imagen por (colegio, secuencia), la más cercana al colegio.
# Elimina pseudorreplicación de fotogramas consecutivos (~47 imgs/seq).
DEDUP_POR_SECUENCIA = True

# Rutas del proyecto.
_ROOT         = Path(__file__).resolve().parents[1]
RUTA_CATALOGO = _ROOT / "data" / "images" / "mapillary" / "mapillary_catalog.csv"
RUTA_SALIDA   = _ROOT / "data" / "images" / "mapillary"

# =============================================================================


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los criterios de selección al catálogo y devuelve el subconjunto
    de imágenes que deben descargarse.

    is_pano se normaliza de string CSV ("True"/"False") a booleano real
    antes de filtrar; bool("False") == True en Python, por lo que no basta
    con astype(bool) directamente sobre la columna leída desde CSV.
    """
    sel = df.copy()

    sel["is_pano"] = sel["is_pano"].astype(str).str.strip().str.lower() == "true"

    if FECHA_DESDE:
        sel = sel[sel["fecha"] >= FECHA_DESDE]

    if EXCLUIR_PANORAMICAS:
        sel = sel[~sel["is_pano"]]

    if DEDUP_POR_SECUENCIA:
        sel = (
            sel.sort_values("distancia_m", na_position="last")
               .drop_duplicates(subset=["dane_est", "sequence"], keep="first")
        )

    return sel.reset_index(drop=True)


async def descargar_imagenes() -> None:
    """
    Lee mapillary_catalog.csv, aplica filtros y descarga en paralelo las
    imágenes pendientes.

    Diseño
    ------
    - La columna 'descargada' se recalcula desde el disco al inicio (no se
      confía en el valor almacenado), lo que hace la reanudación idempotente.
    - El semáforo MAX_CONCURRENT limita las conexiones simultáneas al CDN.
    - Ctrl+C (SIGINT) activa asyncio.Event que detiene nuevas descargas;
      las peticiones en vuelo terminan antes de que el proceso salga.
    - Al finalizar, 'descargada' se actualiza en el CSV con el estado real.
    """
    if not RUTA_CATALOGO.exists():
        print("[ERROR] No se encontró mapillary_catalog.csv.")
        print(f"  → Ejecuta primero: python scripts/analyze_mapillary_colegios.py")
        sys.exit(1)

    df = pd.read_csv(RUTA_CATALOGO, dtype={"dane_est": str})

    # Estado real desde disco (robusto ante interrupciones previas)
    df["descargada"] = df["nombre_archivo"].apply(
        lambda n: (RUTA_SALIDA / str(n)).exists()
    )

    sel       = aplicar_filtros(df)
    pendientes = sel[
        sel["descargada"].eq(False) & sel["url_descarga"].notna()
    ].copy()

    n_pano_excluidas = (
        df[df["is_pano"].astype(str).str.lower() == "true"].__len__()
        if EXCLUIR_PANORAMICAS else 0
    )

    print(f"Imágenes en catálogo      : {len(df):,}")
    if FECHA_DESDE:
        print(f"Filtro de fecha           : >= {FECHA_DESDE}")
    if EXCLUIR_PANORAMICAS:
        print(f"Excluir panorámicas       : sí")
    if DEDUP_POR_SECUENCIA:
        print(f"Dedup por secuencia       : sí  (1 imagen/secuencia/colegio)")
    print(f"Imágenes seleccionadas    : {len(sel):,}  ({sel['dane_est'].nunique()} colegios)")
    print(f"Ya en disco               : {int(sel['descargada'].sum()):,}")
    print(f"Pendientes de descarga    : {len(pendientes):,}")
    print(f"Resolución                : {IMG_FIELD}")
    print(f"Concurrencia              : {MAX_CONCURRENT} peticiones simultáneas")

    if len(pendientes) == 0:
        print("\nTodas las imágenes seleccionadas ya están en disco.")
        return

    est_gb = len(pendientes) * 400 / 1024 / 1024
    print(f"Tamaño estimado           : ~{est_gb:.1f} GB")
    print(f"\nPresiona Ctrl+C para detener y guardar el progreso.\n")

    parar     = asyncio.Event()
    semaforo  = asyncio.Semaphore(MAX_CONCURRENT)
    conector  = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGINT,
        lambda: (
            sys.stdout.write(
                "\n\n[AVISO] Deteniendo. Las imágenes en disco se conservan.\n"
                "        Vuelve a ejecutar el script para continuar.\n"
            ),
            sys.stdout.flush(),
            parar.set(),
        ),
    )

    descargadas_sesion: list = []
    procesadas = 0

    async def descargar_una(row: pd.Series) -> None:
        nonlocal procesadas
        if parar.is_set():
            return

        ruta_local = RUTA_SALIDA / row["nombre_archivo"]
        if ruta_local.exists():
            procesadas += 1
            return

        async with semaforo:
            for intento in range(3):
                try:
                    async with session.get(
                        row["url_descarga"],
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as r:
                        if r.status == 429:
                            await asyncio.sleep(2 ** intento * 5)
                            continue
                        r.raise_for_status()
                        contenido = await r.read()
                    ruta_local.write_bytes(contenido)
                    descargadas_sesion.append(row["nombre_archivo"])
                    break
                except asyncio.CancelledError:
                    return
                except Exception:
                    if intento == 2:
                        return
                    await asyncio.sleep(2 ** intento)

        procesadas += 1
        sys.stdout.write(
            f"\r  Procesadas: {procesadas:,}/{len(pendientes):,}"
            f"  |  Descargadas esta sesión: {len(descargadas_sesion):,}"
        )
        sys.stdout.flush()

    async with aiohttp.ClientSession(connector=conector) as session:
        await asyncio.gather(*[descargar_una(row) for _, row in pendientes.iterrows()])

    # Actualizar catálogo con estado real del disco
    df["descargada"] = df["nombre_archivo"].apply(
        lambda n: (RUTA_SALIDA / str(n)).exists()
    )
    df.to_csv(RUTA_CATALOGO, index=False, encoding="utf-8")

    print(f"\n\nDescargadas esta sesión   : {len(descargadas_sesion):,}")
    print(f"Total en disco            : {int(df['descargada'].sum()):,}")
    print(f"Catálogo actualizado en   : {RUTA_CATALOGO}")


def main() -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    asyncio.run(descargar_imagenes())


if __name__ == "__main__":
    main()
