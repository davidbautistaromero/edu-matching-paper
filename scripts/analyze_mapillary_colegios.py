#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_mapillary_colegios.py
=============================
Construye el catálogo de metadatos Mapillary (Fase 1) y genera mapas
de cobertura estáticos para explorar qué imágenes están disponibles
antes de descargar.

SALIDAS
-------
  data/images/mapillary/
  ├── mapillary_catalog.csv     metadatos de todas las imágenes encontradas
  ├── resumen_fechas.csv        resumen por colegio (n_imgs, fechas, distancias)
  ├── mapa_cobertura_total.png  catálogo completo sin filtros
  └── mapa_cobertura_filtrada.png imágenes seleccionadas tras aplicar filtros

MAPAS
-----
  mapa_cobertura_total.png
    Todos los colegios y todas las fotos disponibles en el catálogo (sin
    ningún filtro). Permite evaluar la cobertura bruta de Mapillary en
    Bogotá antes de aplicar criterios de selección.

  mapa_cobertura_filtrada.png
    Resultado después de aplicar todos los filtros (fecha, exclusión de
    panorámicas, deduplicación por secuencia). Muestra la ubicación exacta
    de cada foto seleccionada y el estado de cobertura de cada colegio.
    · Puntos azules: ubicación exacta de cada foto seleccionada.
    · Círculos verdes: colegios con imágenes regulares disponibles.
    · Triángulos naranjas: colegios con solo panorámicas (excluidas).
    · Cruces rojas: colegios sin cobertura alguna dentro del radio.

REANUDABILIDAD
  Si mapillary_catalog.csv ya existe y FORZAR_RECATALOGO = False, se
  salta la consulta a la API y se usan los metadatos existentes. Solo
  regenera el resumen y los mapas.

REQUISITOS
  pip install geopandas aiohttp pandas pyproj matplotlib
"""

# ---------------------------------------------------------------------------
# Biblioteca estándar
# ---------------------------------------------------------------------------
import asyncio
import csv
import math
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Terceros
# ---------------------------------------------------------------------------
import aiohttp
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PARÁMETROS — edita solo esta sección
# IMPORTANTE: mantener sincronizados con download_mapillary_colegios.py
# =============================================================================

# Token de acceso a la Graph API v4 de Mapillary.
# Ver instrucciones en fetch_mapillary_colegios.py o en el README.
MAPILLARY_TOKEN = "MLY|26901297819500775|94f2d09dcf6652dae352c160b97a7d3c"

# Radio de búsqueda alrededor del punto de cada colegio, en metros.
# Se implementa como bounding box (ver bbox_desde_punto); la API de Mapillary
# limita el parámetro closeto+radius a 50 m máximo.
RADIO_M = 100

# True  → reconstruye el catálogo desde la API aunque ya exista en disco.
# False → carga el catálogo existente y solo regenera resumen y mapas.
FORZAR_RECATALOGO = False

# Número máximo de peticiones HTTP simultáneas al construir el catálogo.
MAX_CONCURRENT = 20

# Resultados por página (máximo permitido por la API: 2000).
LIMIT_POR_PAGINA = 2000

# --- Criterios de selección — deben coincidir con download_mapillary_colegios.py ---

# Fecha mínima de captura incluida en la selección (formato YYYY-MM-DD).
# "" o None para no filtrar por fecha.
FECHA_DESDE = "2021-01-01"

# True → excluir imágenes panorámicas (is_pano = True) de la selección.
# Justificación: VGG19 fue entrenado en imágenes perspectiva estándar;
# las panorámicas tienen distorsión equirectangular que degrada los embeddings.
EXCLUIR_PANORAMICAS = True

# True → por cada par (colegio, secuencia), conservar solo la imagen más
# cercana al colegio. Elimina pseudorreplicación de fotogramas consecutivos
# de un mismo recorrido de captura (~47 imgs/secuencia en este dataset).
DEDUP_POR_SECUENCIA = True

# Rutas del proyecto (relativas a este script para portabilidad).
_ROOT         = Path(__file__).resolve().parents[1]
RUTA_GEOJSON  = _ROOT / "data" / "raw" / "demandacupos04_2024.geojson"
RUTA_SALIDA   = _ROOT / "data" / "images" / "mapillary"
RUTA_CATALOGO = RUTA_SALIDA / "mapillary_catalog.csv"
RUTA_RESUMEN  = RUTA_SALIDA / "resumen_fechas.csv"

# Rutas de salida de mapas.
RUTA_MAPA_TOTAL_PNG     = RUTA_SALIDA / "mapa_cobertura_total.png"
RUTA_MAPA_FILTRADA_PNG  = RUTA_SALIDA / "mapa_cobertura_filtrada.png"

# Columnas del catálogo (orden fijo para compatibilidad con pipeline posterior).
COLUMNAS_CSV = [
    "dane_est", "nombre_est", "image_id",
    "lon_img", "lat_img", "lon_colegio", "lat_colegio",
    "distancia_m", "fecha", "compass_angle", "is_pano",
    "sequence", "url_descarga", "nombre_archivo", "descargada",
]

# =============================================================================


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def bbox_desde_punto(lon: float, lat: float, radio_m: float) -> tuple:
    """
    Bounding box cuadrado (west, south, east, north) en grados WGS84 para
    un punto y radio en metros. Reemplaza closeto+radius de la API, que
    tiene un límite de 50 m y devuelve HTTP 500 si se supera.
    """
    dlat = radio_m / 111_320
    dlon = radio_m / (111_320 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en metros entre dos puntos WGS84."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2)
        * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def ms_a_fecha(ms: int) -> str:
    """Unix timestamp en milisegundos → YYYY-MM-DD (UTC)."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def nombre_archivo(dane: str, fecha: str, image_id: str) -> str:
    """Nombre de archivo local: {DANE12_EST}_{YYYY-MM-DD}_{image_id}.jpg"""
    return f"{dane}_{fecha}_{image_id}.jpg"


def construir_fila(imagen: dict, dane: str, nombre_est: str,
                   lon_col: float, lat_col: float) -> dict:
    """Dict de imagen de la API → fila del catálogo CSV."""
    geom   = imagen.get("computed_geometry") or imagen.get("geometry") or {}
    coords = geom.get("coordinates", [None, None])
    lon_img, lat_img = coords[0], coords[1]

    ms    = imagen.get("captured_at") or 0
    fecha = ms_a_fecha(ms) if ms else "fecha_desconocida"
    iid   = imagen["id"]

    dist = None
    if lon_img is not None and lat_img is not None:
        dist = round(haversine_m(lat_col, lon_col, lat_img, lon_img), 1)

    return {
        "dane_est":       dane,
        "nombre_est":     nombre_est,
        "image_id":       iid,
        "lon_img":        lon_img,
        "lat_img":        lat_img,
        "lon_colegio":    lon_col,
        "lat_colegio":    lat_col,
        "distancia_m":    dist,
        "fecha":          fecha,
        "compass_angle":  imagen.get("compass_angle"),
        "is_pano":        imagen.get("is_pano", False),
        "sequence":       imagen.get("sequence"),
        "url_descarga":   imagen.get("thumb_2048_url"),
        "nombre_archivo": nombre_archivo(dane, fecha, iid),
        "descargada":     False,
    }


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los criterios de selección definidos en los parámetros y devuelve
    el subconjunto de imágenes para descarga.

    Los filtros se aplican en este orden:
      1. Fecha mínima (FECHA_DESDE).
      2. Exclusión de panorámicas (EXCLUIR_PANORAMICAS).
      3. Deduplicación por secuencia (DEDUP_POR_SECUENCIA): por cada par
         (colegio, secuencia) se conserva la imagen más cercana al colegio.

    is_pano se lee desde CSV como string ("True"/"False"); se normaliza
    antes de filtrar para evitar que bool("False") == True en Python.
    """
    sel = df.copy()

    # Normalizar is_pano a booleano real (CSV lo almacena como string)
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


# ---------------------------------------------------------------------------
# Consulta paginada a la API de Mapillary
# ---------------------------------------------------------------------------

async def paginar_imagenes(session: aiohttp.ClientSession,
                           semaforo: asyncio.Semaphore,
                           lon: float, lat: float) -> list:
    """
    Recupera metadatos de TODAS las imágenes dentro del radio RADIO_M
    alrededor de (lon, lat), siguiendo la paginación de la API.

    El semáforo se adquiere por página para que MAX_CONCURRENT refleje
    exactamente el número de peticiones HTTP en vuelo en cada instante.
    """
    campos = ",".join([
        "id", "geometry", "computed_geometry",
        "captured_at", "thumb_2048_url",
        "compass_angle", "is_pano", "sequence",
    ])
    w, s, e, n = bbox_desde_punto(lon, lat, RADIO_M)
    url_base = (
        f"https://graph.mapillary.com/images"
        f"?fields={campos}"
        f"&bbox={w},{s},{e},{n}"
        f"&limit={LIMIT_POR_PAGINA}"
        f"&access_token={MAPILLARY_TOKEN}"
    )

    imagenes = []
    url_sig  = url_base

    while url_sig:
        async with semaforo:
            for intento in range(3):
                try:
                    async with session.get(url_sig, timeout=aiohttp.ClientTimeout(total=30)) as r:
                        if r.status == 401:
                            print("\n[ERROR] Token de Mapillary inválido. Verifica MAPILLARY_TOKEN.")
                            sys.exit(1)
                        if r.status == 429:
                            await asyncio.sleep(2 ** intento * 5)
                            continue
                        r.raise_for_status()
                        datos = await r.json()
                    imagenes.extend(datos.get("data", []))
                    url_sig = datos.get("paging", {}).get("next")
                    break
                except asyncio.CancelledError:
                    return imagenes
                except Exception:
                    if intento == 2:
                        url_sig = None
                        break
                    await asyncio.sleep(2 ** intento)

    return imagenes


# =============================================================================
# FASE 1: CATÁLOGO DE METADATOS
# =============================================================================

async def construir_catalogo(colegios: gpd.GeoDataFrame) -> None:
    """
    Consulta Mapillary para cada colegio y almacena metadatos en
    mapillary_catalog.csv sin descargar imágenes.

    Reanudabilidad: si el catálogo existe y no se fuerza recatalogación,
    solo procesa los colegios cuyo DANE no figure en el CSV.
    Escritura concurrente: asyncio.Lock serializa las escrituras al CSV.
    """
    danes_procesados: set = set()
    if RUTA_CATALOGO.exists() and not FORZAR_RECATALOGO:
        df_prev = pd.read_csv(RUTA_CATALOGO, dtype={"dane_est": str})
        danes_procesados = set(df_prev["dane_est"].unique())
        print(f"Catálogo existente: {len(danes_procesados)} colegios ya procesados.")

    pendientes = colegios[~colegios["DANE12_EST"].isin(danes_procesados)].copy()
    print(f"Colegios pendientes: {len(pendientes)} / {len(colegios)}")

    if pendientes.empty:
        print("Catálogo completo. Omitiendo consulta a la API.")
        return

    modo     = "a" if RUTA_CATALOGO.exists() and not FORZAR_RECATALOGO else "w"
    lock_csv = asyncio.Lock()
    semaforo = asyncio.Semaphore(MAX_CONCURRENT)
    conector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    procesados = 0
    total_imgs = 0

    with open(RUTA_CATALOGO, modo, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        if modo == "w":
            writer.writeheader()

        async with aiohttp.ClientSession(connector=conector) as session:

            async def procesar(dane, nombre, lon, lat):
                nonlocal procesados, total_imgs
                imgs  = await paginar_imagenes(session, semaforo, lon, lat)
                filas = [construir_fila(img, dane, nombre, lon, lat) for img in imgs]
                async with lock_csv:
                    for fila in filas:
                        writer.writerow(fila)
                    f.flush()
                procesados += 1
                total_imgs += len(filas)
                sys.stdout.write(
                    f"\r  Procesados: {procesados}/{len(pendientes)}"
                    f"  |  Imágenes encontradas: {total_imgs:,}"
                )
                sys.stdout.flush()

            await asyncio.gather(*[
                procesar(r["DANE12_EST"], r["NOMBRE_EST"],
                         r.geometry.x, r.geometry.y)
                for _, r in pendientes.iterrows()
            ])

    print(f"\n\nCatálogo guardado en: {RUTA_CATALOGO}")


def generar_resumen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce resumen_fechas.csv con estadísticas por colegio del catálogo
    completo (sin filtros): n_imgs, rango de fechas, fechas únicas,
    distancias y porcentaje de panorámicas.
    """
    df["is_pano"] = df["is_pano"].astype(str).str.strip().str.lower() == "true"

    resumen = (
        df.groupby(["dane_est", "nombre_est"])
        .agg(
            n_imagenes      =("image_id",    "count"),
            fecha_min       =("fecha",       "min"),
            fecha_max       =("fecha",       "max"),
            n_fechas_unicas =("fecha",       "nunique"),
            fechas_unicas   =("fecha",       lambda x: "|".join(sorted(x.dropna().unique()))),
            dist_min_m      =("distancia_m", "min"),
            dist_max_m      =("distancia_m", "max"),
            pct_panoramicas =("is_pano",     lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .sort_values("n_imagenes", ascending=False)
    )
    resumen.to_csv(RUTA_RESUMEN, index=False, encoding="utf-8")

    print(f"\n{'─'*72}")
    print(f"  RESUMEN DEL CATÁLOGO  |  radio={RADIO_M} m  |  {len(resumen)} colegios con imágenes")
    print(f"{'─'*72}")
    print(f"{'DANE':<15} {'Imgs':>7} {'F.únicas':>9} {'Fecha min':>12} {'Fecha max':>12} {'%pano':>6}")
    print(f"{'─'*72}")
    for _, r in resumen.head(15).iterrows():
        print(f"{r['dane_est']:<15}{r['n_imagenes']:>7,}{r['n_fechas_unicas']:>9}"
              f"{str(r['fecha_min']):>12}{str(r['fecha_max']):>12}{r['pct_panoramicas']:>5.1f}%")
    if len(resumen) > 15:
        print(f"  ... y {len(resumen)-15} colegios más.")
    print(f"{'─'*72}")
    print(f"  Total imágenes: {int(resumen['n_imagenes'].sum()):,}")
    print(f"  Sin cobertura : {407 - len(resumen)} colegios")
    print(f"\nresumen_fechas.csv guardado en: {RUTA_RESUMEN}")
    return resumen


# =============================================================================
# MAPAS ESTÁTICOS (matplotlib)
# =============================================================================

def _scatter_base(ax, df_fotos, col_fotos, df_colegios_ok,
                  col_ok, df_colegios_pano, df_colegios_sin,
                  titulo, mostrar_leyenda_colegios=True):
    """Dibuja una figura base de scatter reutilizable para ambos mapas PNG."""
    ax.set_facecolor("#f5f5f5")
    ax.set_aspect("equal")

    # Fotos
    if len(df_fotos) > 0:
        ax.scatter(df_fotos["lon_img"], df_fotos["lat_img"],
                   s=2, color=col_fotos, alpha=0.4, zorder=2,
                   label=f"Fotos ({len(df_fotos):,})")

    # Colegios con imágenes regulares
    if len(df_colegios_ok) > 0:
        ax.scatter(df_colegios_ok["lon_colegio"], df_colegios_ok["lat_colegio"],
                   s=35, color="#27ae60", marker="o", zorder=4,
                   label=f"Colegio con imágenes ({len(df_colegios_ok)})")

    # Colegios solo panorámicas
    if len(df_colegios_pano) > 0:
        ax.scatter(df_colegios_pano["lon_colegio"], df_colegios_pano["lat_colegio"],
                   s=35, color="#f39c12", marker="^", zorder=4,
                   label=f"Solo panorámicas ({len(df_colegios_pano)})")

    # Colegios sin cobertura
    if len(df_colegios_sin) > 0:
        ax.scatter(df_colegios_sin["lon"], df_colegios_sin["lat"],
                   s=35, color="#e74c3c", marker="x", zorder=4,
                   label=f"Sin cobertura ({len(df_colegios_sin)})")

    ax.set_title(titulo, fontsize=12, pad=10)
    ax.set_xlabel("Longitud (WGS84)")
    ax.set_ylabel("Latitud (WGS84)")
    ax.legend(loc="lower left", fontsize=7, markerscale=2)
    ax.grid(True, linewidth=0.3, alpha=0.5)


def generar_mapas_png(df_total: pd.DataFrame, df_filtrada: pd.DataFrame,
                      colegios: gpd.GeoDataFrame) -> None:
    """
    Genera dos mapas estáticos PNG:
      mapa_cobertura_total.png    — todas las fotos del catálogo, sin filtros.
      mapa_cobertura_filtrada.png — fotos seleccionadas tras aplicar filtros.

    En ambos mapas:
      · Puntos azules: ubicación exacta de cada foto.
      · Puntos verdes (círculo): colegios con imágenes regulares disponibles.
      · Triángulos naranjas: colegios cuyas únicas fotos son panorámicas.
      · Cruces rojas: colegios sin ninguna imagen dentro del radio.
    """
    df_total    = df_total.dropna(subset=["lon_img", "lat_img"])
    df_filtrada = df_filtrada.dropna(subset=["lon_img", "lat_img"])

    # Clasificar colegios según estado post-filtro
    danes_ok   = set(df_filtrada["dane_est"].unique())
    danes_cat  = set(df_total["dane_est"].unique())
    danes_pano = danes_cat - danes_ok   # tienen imgs en catálogo pero solo panorámicas
    danes_sin  = set(colegios["DANE12_EST"]) - danes_cat  # sin ninguna imagen

    col_ok   = df_total[df_total["dane_est"].isin(danes_ok)].drop_duplicates("dane_est")
    col_pano = df_total[df_total["dane_est"].isin(danes_pano)].drop_duplicates("dane_est")
    col_sin  = colegios[colegios["DANE12_EST"].isin(danes_sin)].copy()
    col_sin["lon"] = col_sin.geometry.x
    col_sin["lat"] = col_sin.geometry.y

    # --- Mapa 1: catálogo total ---
    fig, ax = plt.subplots(figsize=(10, 12))
    _scatter_base(
        ax,
        df_fotos=df_total,
        col_fotos="#3498db",
        df_colegios_ok=col_ok,
        col_ok="#27ae60",
        df_colegios_pano=col_pano,
        df_colegios_sin=col_sin,
        titulo=f"Cobertura Mapillary — catálogo completo\n"
               f"radio={RADIO_M} m | {len(df_total):,} fotos | {len(danes_cat)} colegios",
    )
    plt.tight_layout()
    plt.savefig(RUTA_MAPA_TOTAL_PNG, dpi=150)
    plt.close()
    print(f"  PNG total     → {RUTA_MAPA_TOTAL_PNG.name}")

    # --- Mapa 2: selección filtrada ---
    fig, ax = plt.subplots(figsize=(10, 12))
    _scatter_base(
        ax,
        df_fotos=df_filtrada,
        col_fotos="#3498db",
        df_colegios_ok=col_ok,
        col_ok="#27ae60",
        df_colegios_pano=col_pano,
        df_colegios_sin=col_sin,
        titulo=f"Cobertura Mapillary — imágenes seleccionadas\n"
               f"fecha≥{FECHA_DESDE} | sin panorámicas | dedup por secuencia\n"
               f"{len(df_filtrada):,} fotos | {len(danes_ok)} colegios cubiertos",
    )
    plt.tight_layout()
    plt.savefig(RUTA_MAPA_FILTRADA_PNG, dpi=150)
    plt.close()
    print(f"  PNG filtrado  → {RUTA_MAPA_FILTRADA_PNG.name}")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main() -> None:
    if MAPILLARY_TOKEN == "TU_TOKEN_AQUI":
        print("[ERROR] Configura MAPILLARY_TOKEN en el script.")
        sys.exit(1)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    # Cargar y reproyectar colegios
    print(f"Cargando {RUTA_GEOJSON.name} ...")
    colegios = gpd.read_file(RUTA_GEOJSON).to_crs("EPSG:4326")
    print(f"  {len(colegios)} colegios | CRS: {colegios.crs.to_string()}")
    print(f"  Radio de búsqueda: {RADIO_M} m\n")

    # Fase 1: construir catálogo
    print("=== FASE 1: CATÁLOGO DE METADATOS ===\n")
    asyncio.run(construir_catalogo(colegios))

    # Cargar catálogo completo
    df = pd.read_csv(RUTA_CATALOGO, dtype={"dane_est": str})
    print(f"\nCatálogo cargado: {len(df):,} imágenes en {df['dane_est'].nunique()} colegios")

    # Resumen
    print("\n=== RESUMEN DEL CATÁLOGO ===")
    resumen = generar_resumen(df)

    # Aplicar filtros
    df_filtrada = aplicar_filtros(df)
    n_sin_regular = df["dane_est"].nunique() - df_filtrada["dane_est"].nunique()
    print(f"\n=== SELECCIÓN TRAS FILTROS ===")
    print(f"  fecha >= {FECHA_DESDE}  |  sin panorámicas  |  dedup por secuencia")
    print(f"  {len(df_filtrada):,} imágenes  |  {df_filtrada['dane_est'].nunique()} colegios cubiertos")
    print(f"  {n_sin_regular} colegios excluidos (solo tienen panorámicas)")
    print(f"  {407 - df['dane_est'].nunique()} colegios sin cobertura Mapillary en {RADIO_M} m")

    # Mapas
    print("\n=== GENERANDO MAPAS ===")
    generar_mapas_png(df, df_filtrada, colegios)

    print(f"\nTodos los archivos guardados en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
