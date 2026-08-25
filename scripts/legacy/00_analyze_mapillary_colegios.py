#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_analyze_mapillary_colegios.py
================================
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

CAMBIOS RESPECTO AL ORIGINAL
  · aplicar_filtros y sus parámetros viven en mapillary_filtros.py
    (fuente de verdad única compartida con 00_download_mapillary_colegios.py).
  · paginar_imagenes: el semáforo se adquiere solo durante la petición HTTP;
    los sleeps de backoff (429 y errores de red) ocurren fuera del semáforo,
    evitando que un slot bloqueado paralice la concurrencia completa.

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
import random
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Cargar variables del archivo .env si existe (sin dependencias externas)
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

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
    N_MUESTRA,
    aplicar_filtros,
)

# ---------------------------------------------------------------------------
# Terceros
# ---------------------------------------------------------------------------
import aiohttp
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PARÁMETROS — edita solo esta sección
# Los criterios de selección (FECHA_DESDE, EXCLUIR_PANORAMICAS,
# DEDUP_POR_SECUENCIA) se importan de mapillary_filtros.py.
# =============================================================================

# Token de acceso a la Graph API v4 de Mapillary.
# Se lee de la variable de entorno para no almacenar credenciales en el código.
#   export MAPILLARY_TOKEN='MLY|...'   (bash)
#   $env:MAPILLARY_TOKEN='MLY|...'     (PowerShell)
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN", "")

# Radio de búsqueda alrededor del punto de cada colegio, en metros.
RADIO_M = 100

# True  → reconstruye el catálogo desde la API aunque ya exista en disco.
# False → carga el catálogo existente y solo regenera resumen y mapas.
FORZAR_RECATALOGO = False

# Número máximo de peticiones HTTP simultáneas al construir el catálogo.
MAX_CONCURRENT = 20

# Resultados por página (máximo permitido por la API: 2000).
LIMIT_POR_PAGINA = 2000

# Rutas del proyecto (relativas a este script para portabilidad).
_ROOT         = Path(__file__).resolve().parents[2]
RUTA_GEOJSON  = _ROOT / "data" / "processed" / "colegios_dataset.geojson"
RUTA_SALIDA   = _ROOT / "data" / "images" / "mapillary"
_sfx          = "_muestra" if MODO_MUESTRA else ""
RUTA_CATALOGO = RUTA_SALIDA / f"mapillary_catalog{_sfx}.csv"
RUTA_RESUMEN  = RUTA_SALIDA / f"resumen_fechas{_sfx}.csv"

# Rutas de salida de mapas.
RUTA_MAPA_TOTAL_PNG     = RUTA_SALIDA / "mapa_cobertura_total.png"
RUTA_MAPA_FILTRADA_PNG  = RUTA_SALIDA / "mapa_cobertura_filtrada.png"

# Columnas del catálogo (orden fijo para compatibilidad con pipeline posterior).
COLUMNAS_CSV = [
    "dane_est", "nombre_est", "image_id",
    "lon_img", "lat_img", "lon_colegio", "lat_colegio",
    "distancia_m", "fecha", "compass_angle",
    "bearing_al_colegio", "diff_angulo",
    "is_pano", "sequence", "url_descarga", "nombre_archivo", "descargada",
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


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Rumbo en grados (0–360, N=0, E=90) desde (lat1,lon1) hacia (lat2,lon2)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl   = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


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

    dist        = None
    bearing     = None
    diff_angulo = None
    if lon_img is not None and lat_img is not None:
        dist    = round(haversine_m(lat_col, lon_col, lat_img, lon_img), 1)
        bearing = round(bearing_deg(lat_img, lon_img, lat_col, lon_col), 1)
        ca = imagen.get("compass_angle")
        if ca is not None:
            d = abs(bearing - ca) % 360
            diff_angulo = round(min(d, 360 - d), 1)

    return {
        "dane_est":           dane,
        "nombre_est":         nombre_est,
        "image_id":           iid,
        "lon_img":            lon_img,
        "lat_img":            lat_img,
        "lon_colegio":        lon_col,
        "lat_colegio":        lat_col,
        "distancia_m":        dist,
        "fecha":              fecha,
        "compass_angle":      imagen.get("compass_angle"),
        "bearing_al_colegio": bearing,
        "diff_angulo":        diff_angulo,
        "is_pano":            imagen.get("is_pano", False),
        "sequence":           imagen.get("sequence"),
        "url_descarga":       imagen.get("thumb_2048_url"),
        "nombre_archivo":     nombre_archivo(dane, fecha, iid),
        "descargada":         False,
    }


# ---------------------------------------------------------------------------
# Consulta paginada a la API de Mapillary
# ---------------------------------------------------------------------------

async def paginar_imagenes(session: aiohttp.ClientSession,
                           semaforo: asyncio.Semaphore,
                           lon: float, lat: float) -> list:
    """
    Recupera metadatos de TODAS las imágenes dentro del radio RADIO_M
    alrededor de (lon, lat), siguiendo la paginación de la API.

    El semáforo se adquiere únicamente durante la petición HTTP activa.
    Los sleeps de backoff (429 y errores de red) ocurren fuera del
    semáforo para no bloquear slots de concurrencia mientras se espera.
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
    timeout  = aiohttp.ClientTimeout(total=30)

    while url_sig:
        for intento in range(3):
            rate_limited = False
            datos = None
            try:
                async with semaforo:
                    async with session.get(url_sig, timeout=timeout) as r:
                        if r.status == 401:
                            print("\n[ERROR] Token de Mapillary inválido. Verifica MAPILLARY_TOKEN.")
                            sys.exit(1)
                        if r.status == 429:
                            rate_limited = True
                        else:
                            r.raise_for_status()
                            datos = await r.json()
            except asyncio.CancelledError:
                return imagenes
            except Exception:
                if intento == 2:
                    url_sig = None
                    break
                await asyncio.sleep(2 ** intento + random.uniform(0, 1))
                continue

            if rate_limited:
                await asyncio.sleep(2 ** intento * 5 + random.uniform(0, 2))
                continue

            imagenes.extend(datos.get("data", []))
            url_sig = datos.get("paging", {}).get("next")
            break

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

            tareas = [
                procesar(r.DANE12_EST, r.NOMBRE_EST, r.geometry.x, r.geometry.y)
                for r in pendientes.itertuples(index=False)
            ]
            await asyncio.gather(*tareas)

    print(f"\n\nCatálogo guardado en: {RUTA_CATALOGO}")


def generar_resumen(df: pd.DataFrame, n_total: int = 0) -> pd.DataFrame:
    """
    Produce resumen_fechas.csv con estadísticas por colegio del catálogo
    completo (sin filtros): n_imgs, rango de fechas, fechas únicas,
    distancias y porcentaje de panorámicas.
    """
    df["is_pano"] = df["is_pano"].astype(str).str.strip().str.lower() == "true"

    # Agregaciones nativas de pandas (compiladas en C, sin lambdas Python).
    resumen = (
        df.groupby(["dane_est", "nombre_est"])
        .agg(
            n_imagenes      =("image_id",    "count"),
            fecha_min       =("fecha",       "min"),
            fecha_max       =("fecha",       "max"),
            n_fechas_unicas =("fecha",       "nunique"),
            dist_min_m      =("distancia_m", "min"),
            dist_max_m      =("distancia_m", "max"),
            pct_panoramicas =("is_pano",     "mean"),
        )
        .reset_index()
    )
    resumen["pct_panoramicas"] = (resumen["pct_panoramicas"] * 100).round(1)

    # fechas_unicas se computa aparte: dedup y sort vectorizados + join builtin,
    # más rápido que una lambda Python dentro del agg principal.
    fechas_unicas = (
        df[df["fecha"].notna()]
        .drop_duplicates(["dane_est", "fecha"])
        .sort_values(["dane_est", "fecha"])
        .groupby("dane_est")["fecha"]
        .agg("|".join)
        .rename("fechas_unicas")
        .reset_index()
    )
    resumen = (
        resumen
        .merge(fechas_unicas, on="dane_est", how="left")
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
    if n_total:
        print(f"  Sin cobertura : {n_total - len(resumen)} colegios")
    print(f"\nresumen_fechas.csv guardado en: {RUTA_RESUMEN}")
    return resumen


# =============================================================================
# MAPAS ESTÁTICOS (matplotlib)
# =============================================================================

def _a_gdf(df: pd.DataFrame, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    """Convierte un DataFrame con columnas lon/lat a GeoDataFrame en EPSG:3857."""
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")


def _scatter_base(ax, df_fotos, col_fotos, df_colegios_ok,
                  col_ok, df_colegios_pano, df_colegios_sin,
                  titulo, mostrar_leyenda_colegios=True):
    """
    Dibuja scatter sobre mapa base CartoDB.Positron (igual que el notebook EDA).
    Datos en WGS84; se reproyectan a EPSG:3857 internamente para contextily.
    """
    if len(df_fotos) > 0:
        _a_gdf(df_fotos, "lon_img", "lat_img").plot(
            ax=ax, markersize=2, color=col_fotos, alpha=0.4, zorder=2,
            label=f"Fotos ({len(df_fotos):,})",
        )
    if len(df_colegios_ok) > 0:
        _a_gdf(df_colegios_ok, "lon_colegio", "lat_colegio").plot(
            ax=ax, markersize=35, color="#27ae60", marker="o", zorder=4,
            label=f"Colegio con imágenes ({len(df_colegios_ok)})",
        )
    if len(df_colegios_pano) > 0:
        _a_gdf(df_colegios_pano, "lon_colegio", "lat_colegio").plot(
            ax=ax, markersize=35, color="#f39c12", marker="^", zorder=4,
            label=f"Solo panorámicas ({len(df_colegios_pano)})",
        )
    if len(df_colegios_sin) > 0:
        _a_gdf(df_colegios_sin, "lon", "lat").plot(
            ax=ax, markersize=35, color="#e74c3c", marker="x", zorder=4,
            label=f"Sin cobertura ({len(df_colegios_sin)})",
        )

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom=12,
        zorder=1,
    )
    ax.set_title(titulo, fontsize=12, pad=10)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, markerscale=2)


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
    danes_pano = danes_cat - danes_ok   # tienen imgs en catálogo pero ninguna pasa los filtros
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
               f"fecha≥{FECHA_DESDE}{' | sin panorámicas' if EXCLUIR_PANORAMICAS else ''} | dedup por secuencia\n"
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
    if not MAPILLARY_TOKEN:
        print("[ERROR] Define la variable de entorno MAPILLARY_TOKEN antes de ejecutar.")
        print("  export MAPILLARY_TOKEN='MLY|...'")
        sys.exit(1)

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    # Cargar y reproyectar colegios
    print(f"Cargando {RUTA_GEOJSON.name} ...")
    colegios = gpd.read_file(RUTA_GEOJSON).to_crs("EPSG:4326")

    # Filtrar sector oficial (el GeoJSON procesado ya lo garantiza; se aplica
    # explícitamente como salvaguarda ante regeneraciones parciales del archivo)
    colegios = colegios[colegios["sector"] == "Oficial"].copy()

    # El GeoJSON de colegios tiene una fila por sede. Se agrega a nivel de
    # establecimiento tomando la sede PRINCIPAL; si no existe, la de menor
    # orden_sede. Esto evita buscar Mapillary dos veces para el mismo colegio.
    colegios["_orden"] = colegios["orden_sede"].astype(str).str.strip().str.upper()
    colegios["_es_principal"] = (colegios["_orden"] == "PRINCIPAL").astype(int)
    colegios = (
        colegios
        .sort_values(["_es_principal", "_orden"], ascending=[False, True])
        .drop_duplicates(subset="id_establecimiento", keep="first")
        .drop(columns=["_orden", "_es_principal"])
        .reset_index(drop=True)
    )

    # Normalizar nombres de columna para compatibilidad con el resto del script
    colegios["DANE12_EST"] = (
        colegios["id_establecimiento"].astype(str).str.strip().str.zfill(12)
    )
    colegios["NOMBRE_EST"] = colegios["nombre_establecimiento"]

    print(f"  {len(colegios)} establecimientos oficiales | CRS: {colegios.crs.to_string()}")
    print(f"  Radio de búsqueda: {RADIO_M} m\n")

    if MODO_MUESTRA:
        colegios = colegios.sample(
            min(N_MUESTRA, len(colegios)), random_state=42
        ).reset_index(drop=True)
        print(f"  [MODO MUESTRA] {len(colegios)} colegios seleccionados aleatoriamente.")
        print(f"  Catálogo de salida: {RUTA_CATALOGO.name}\n")
        if RUTA_CATALOGO.exists():
            RUTA_CATALOGO.unlink()

    # Fase 1: construir catálogo
    print("=== FASE 1: CATÁLOGO DE METADATOS ===\n")
    asyncio.run(construir_catalogo(colegios))

    # Cargar catálogo completo
    df = pd.read_csv(RUTA_CATALOGO, dtype={"dane_est": str})
    print(f"\nCatálogo cargado: {len(df):,} imágenes en {df['dane_est'].nunique()} colegios")

    # Resumen
    print("\n=== RESUMEN DEL CATÁLOGO ===")
    resumen = generar_resumen(df, n_total=len(colegios))

    # Aplicar filtros
    df_filtrada = aplicar_filtros(df)
    n_sin_regular = df["dane_est"].nunique() - df_filtrada["dane_est"].nunique()
    print(f"\n=== SELECCIÓN TRAS FILTROS ===")
    rango = f"{FECHA_DESDE} – {FECHA_HASTA}" if FECHA_HASTA else f">= {FECHA_DESDE}"
    pano_label = "sin panorámicas" if EXCLUIR_PANORAMICAS else "con panorámicas"
    print(f"  fecha: {rango}  |  {pano_label}  |  dedup por secuencia")
    print(f"  {len(df_filtrada):,} imágenes  |  {df_filtrada['dane_est'].nunique()} colegios cubiertos")
    print(f"  {n_sin_regular} colegios con imágenes pero sin selección tras filtros")
    print(f"  {len(colegios) - df['dane_est'].nunique()} colegios sin cobertura Mapillary en {RADIO_M} m")

    # Mapas
    print("\n=== GENERANDO MAPAS ===")
    generar_mapas_png(df, df_filtrada, colegios)

    print(f"\nTodos los archivos guardados en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
