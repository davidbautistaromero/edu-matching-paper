#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_download_mapillary_colegios.py
=================================
Descarga las imágenes de Mapillary seleccionadas según los criterios
definidos en los parámetros. Lee el catálogo de metadatos generado por
00_analyze_mapillary_colegios.py y descarga solo las imágenes que cumplen
los filtros y aún no existen en disco.

PREREQUISITO
  Ejecutar primero 00_analyze_mapillary_colegios.py para construir el catálogo:
    python scripts/00_analyze_mapillary_colegios.py

CRITERIOS DE SELECCIÓN
  Definidos en mapillary_filtros.py (fuente de verdad compartida con
  00_analyze_mapillary_colegios.py). Ver ese módulo para cambiarlos.

REANUDABILIDAD
  El script comprueba si cada archivo .jpg ya existe en disco antes de
  descargar. Puede interrumpirse con Ctrl+C y reanudarse sin duplicados.
  Al terminar actualiza la columna 'descargada' en mapillary_catalog.csv.

SALIDA
  data/images/mapillary/
  └── {DANE12_EST}_{YYYY-MM-DD}_{image_id}.jpg   (una línea por imagen)

CAMBIOS RESPECTO AL ORIGINAL
  · aplicar_filtros y sus parámetros viven en mapillary_filtros.py
    (fuente de verdad única compartida con 00_analyze_mapillary_colegios.py).
  · descargar_una: el semáforo se adquiere solo durante la petición HTTP
    activa; los sleeps de backoff (429 y errores de red) ocurren fuera del
    semáforo, evitando que un slot bloqueado paralice la concurrencia completa.

REQUISITOS
  pip install aiohttp pandas
"""

# ---------------------------------------------------------------------------
# Biblioteca estándar
# ---------------------------------------------------------------------------
import asyncio
import random
import signal
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Módulo compartido — criterios de selección y aplicar_filtros
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from mapillary_filtros import (
    FECHA_DESDE,
    FECHA_HASTA,
    EXCLUIR_PANORAMICAS,
    DEDUP_POR_SECUENCIA,
    MODO_MUESTRA,
    aplicar_filtros,
)

# ---------------------------------------------------------------------------
# Terceros
# ---------------------------------------------------------------------------
import aiohttp
import pandas as pd


# =============================================================================
# PARÁMETROS
# Los criterios de selección (FECHA_DESDE, EXCLUIR_PANORAMICAS,
# DEDUP_POR_SECUENCIA) se importan de mapillary_filtros.py.
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

# Rutas del proyecto.
_ROOT         = Path(__file__).resolve().parents[1]
_sfx          = "_muestra" if MODO_MUESTRA else ""
RUTA_SALIDA   = _ROOT / "data" / "images" / "mapillary"
RUTA_CATALOGO = RUTA_SALIDA / f"mapillary_catalog{_sfx}.csv"

# =============================================================================


async def descargar_imagenes() -> None:
    """
    Lee mapillary_catalog.csv, aplica filtros y descarga en paralelo las
    imágenes pendientes.

    Diseño
    ------
    - La columna 'descargada' se recalcula desde el disco al inicio (no se
      confía en el valor almacenado), lo que hace la reanudación idempotente.
    - El semáforo MAX_CONCURRENT se adquiere solo durante la petición HTTP
      activa; los sleeps de backoff ocurren fuera del semáforo para no
      bloquear slots de concurrencia mientras se espera.
    - Ctrl+C (SIGINT) activa asyncio.Event que detiene nuevas descargas;
      las peticiones en vuelo terminan antes de que el proceso salga.
    - Al finalizar, 'descargada' se actualiza en el CSV con el estado real.
    """
    if not RUTA_CATALOGO.exists():
        print("[ERROR] No se encontró mapillary_catalog.csv.")
        print(f"  → Ejecuta primero: python scripts/00_analyze_mapillary_colegios.py")
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

    print(f"Imágenes en catálogo      : {len(df):,}")
    if FECHA_DESDE or FECHA_HASTA:
        rango = f"{FECHA_DESDE or '*'} – {FECHA_HASTA or '*'}"
        print(f"Filtro de fecha           : {rango}")
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
    def _on_sigint():
        sys.stdout.write(
            "\n\n[AVISO] Deteniendo. Las imágenes en disco se conservan.\n"
            "        Vuelve a ejecutar el script para continuar.\n"
        )
        sys.stdout.flush()
        parar.set()

    try:
        # Unix: add_signal_handler se integra nativamente con el event loop
        # de asyncio y puede activar el evento parar desde dentro del loop.
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except NotImplementedError:
        # Windows: ProactorEventLoop no implementa add_signal_handler y lanza
        # NotImplementedError. Como fallback se usa signal.signal(), que en
        # Windows ejecuta el handler en el hilo principal (fuera del loop).
        # call_soon_threadsafe() es necesario para cruzar de ese hilo al loop
        # de asyncio de forma segura sin condiciones de carrera.
        signal.signal(
            signal.SIGINT,
            lambda sig, frame: loop.call_soon_threadsafe(_on_sigint),
        )

    descargadas_sesion: list = []
    procesadas = 0
    timeout = aiohttp.ClientTimeout(total=60)

    async def descargar_una(row: dict) -> None:
        nonlocal procesadas
        if parar.is_set():
            return

        ruta_local = RUTA_SALIDA / row["nombre_archivo"]
        if ruta_local.exists():
            procesadas += 1
            return

        for intento in range(3):
            rate_limited = False
            contenido = None
            try:
                async with semaforo:
                    async with session.get(row["url_descarga"], timeout=timeout) as r:
                        if r.status == 429:
                            rate_limited = True
                        else:
                            r.raise_for_status()
                            contenido = await r.read()
            except asyncio.CancelledError:
                return
            except Exception:
                if intento == 2:
                    return
                await asyncio.sleep(2 ** intento + random.uniform(0, 1))
                continue

            if rate_limited:
                await asyncio.sleep(2 ** intento * 5 + random.uniform(0, 2))
                continue

            ruta_local.write_bytes(contenido)
            descargadas_sesion.append(row["nombre_archivo"])
            break

        procesadas += 1
        sys.stdout.write(
            f"\r  Procesadas: {procesadas:,}/{len(pendientes):,}"
            f"  |  Descargadas esta sesión: {len(descargadas_sesion):,}"
        )
        sys.stdout.flush()

    registros = pendientes.to_dict("records")
    async with aiohttp.ClientSession(connector=conector) as session:
        await asyncio.gather(*[descargar_una(row) for row in registros])

    # Actualizar catálogo: solo marcar las filas descargadas esta sesión.
    # df["descargada"] ya reflejaba el estado real del disco al inicio;
    # basta con activar las nuevas sin releer el disco para cada fila.
    if descargadas_sesion:
        df.loc[df["nombre_archivo"].isin(set(descargadas_sesion)), "descargada"] = True
    df.to_csv(RUTA_CATALOGO, index=False, encoding="utf-8")

    print(f"\n\nDescargadas esta sesión   : {len(descargadas_sesion):,}")
    print(f"Total en disco            : {int(df['descargada'].sum()):,}")
    print(f"Catálogo actualizado en   : {RUTA_CATALOGO}")


def main() -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    asyncio.run(descargar_imagenes())


if __name__ == "__main__":
    main()
