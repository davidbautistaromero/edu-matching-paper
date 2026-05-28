#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_download_gsv_colegios.py
===========================
Descarga imágenes de Google Street View Static API para las 558 sedes
de colegios oficiales de Bogotá.

SALIDAS
-------
  data/images/gsv/
  ├── {id_establecimiento}/
  │   └── {id_sede}_{heading:03d}.jpg   (N_HEADINGS imágenes por sede)
  └── gsv_catalog.csv                   metadatos de todas las imágenes

REANUDABILIDAD
  Si gsv_catalog.csv existe y FORZAR_REDESCARGA=False, salta las filas
  con descargada=True y continúa desde donde quedó.

MODO PRUEBA
  Pon MODO_MUESTRA = True en gsv_config.py para probar con N_MUESTRA sedes
  antes del run completo. El catálogo de prueba se guarda como gsv_catalog_muestra.csv.

REQUISITOS
  pip install geopandas aiohttp pandas
"""

import asyncio
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Leer .env (sin python-dotenv)
# ---------------------------------------------------------------------------
_ROOT    = Path(__file__).resolve().parents[1]
_env_path = _ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Importar config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from gsv_config import (
    MODO_MUESTRA,
    N_MUESTRA,
    N_HEADINGS,
    IMG_SIZE,
    FOV,
    PITCH,
    FORZAR_REDESCARGA,
    MAX_CONCURRENT,
    MAX_REINTENTOS,
    ESPERA_429_S,
    MIN_SIZE_BYTES,
)

import aiohttp
import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
API_KEY        = os.getenv("GSV_API_TOKEN", "")
BASE_URL       = "https://maps.googleapis.com/maps/api/streetview"
RUTA_SEDES     = _ROOT / "data" / "processed" / "colegios_dataset.geojson"
RUTA_GEOCODED  = _ROOT / "data" / "raw" / "colegios_coordenadas_google.csv"
RUTA_SALIDA    = _ROOT / "data" / "images" / "gsv"
_sfx         = "_muestra" if MODO_MUESTRA else ""
RUTA_CATALOGO = RUTA_SALIDA / f"gsv_catalog{_sfx}.csv"

COLUMNAS_CSV = [
    "id_establecimiento", "id_sede",
    "nombre_establecimiento", "nombre_sede",
    "lat", "lon", "heading",
    "ruta_archivo", "descargada", "fecha_descarga",
]

HEADINGS = [int(360 / N_HEADINGS * i) for i in range(N_HEADINGS)]

# ---------------------------------------------------------------------------
# Reemplazo de coordenadas con Google Geocoding
# ---------------------------------------------------------------------------

def aplicar_coordenadas_google(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Reemplaza la geometría de cada sede con las coordenadas de Google
    cuando están disponibles en RUTA_GEOCODED (status == 'OK').
    Las sedes sin match conservan su geometría original.
    """
    from shapely.geometry import Point

    if not RUTA_GEOCODED.exists():
        print("  [coords] colegios_coordenadas_google.csv no encontrado — usando coordenadas originales")
        return gdf

    geo_df = pd.read_csv(RUTA_GEOCODED, dtype={"id_establecimiento": str})
    ok = geo_df[geo_df["status"] == "OK"].set_index("id_establecimiento")
    lookup = {idx: (row["lat_google"], row["lon_google"]) for idx, row in ok.iterrows()}

    gdf = gdf.copy()
    reemplazados = 0
    for i, row in gdf.iterrows():
        id_est = str(row["id_establecimiento"]).strip()
        if id_est in lookup:
            lat, lon = lookup[id_est]
            gdf.at[i, "geometry"] = Point(lon, lat)
            reemplazados += 1

    print(f"  [coords] {reemplazados}/{len(gdf)} sedes con coordenadas Google "
          f"({len(gdf) - reemplazados} conservan coordenadas originales)")
    return gdf


# ---------------------------------------------------------------------------
# Catálogo (carga / inicializa)
# ---------------------------------------------------------------------------

UMBRAL_COORD = 0.001  # ~100 m; si la diferencia supera esto se re-descarga


def invalidar_coords_cambiadas(gdf: "gpd.GeoDataFrame", catalogo: dict) -> tuple[dict, int]:
    """
    Elimina del catálogo las entradas cuyas coordenadas difieren de las
    actuales en más de UMBRAL_COORD grados. Esas imágenes se re-descargarán
    con la posición corregida por Google.
    Devuelve el catálogo limpio y el número de entradas invalidadas.
    """
    # Índice id_sede → (lat, lon) nuevos
    coord_nuevas: dict[str, tuple[float, float]] = {}
    for _, row in gdf.iterrows():
        id_sede = str(row["id_sede"]).strip()
        coord_nuevas[id_sede] = (row.geometry.y, row.geometry.x)

    invalidadas = 0
    claves_a_borrar = []
    for clave, entrada in catalogo.items():
        id_sede = str(entrada.get("id_sede", "")).strip()
        if id_sede not in coord_nuevas:
            continue
        try:
            lat_cat = float(entrada["lat"])
            lon_cat = float(entrada["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        lat_new, lon_new = coord_nuevas[id_sede]
        if abs(lat_cat - lat_new) > UMBRAL_COORD or abs(lon_cat - lon_new) > UMBRAL_COORD:
            claves_a_borrar.append(clave)
            invalidadas += 1

    for clave in claves_a_borrar:
        del catalogo[clave]

    return catalogo, invalidadas


def cargar_catalogo() -> dict:
    """Devuelve dict {ruta_archivo: row_dict} de imágenes ya descargadas."""
    if RUTA_CATALOGO.exists() and not FORZAR_REDESCARGA:
        df = pd.read_csv(RUTA_CATALOGO, dtype=str)
        df["descargada"] = df["descargada"].str.lower() == "true"
        return {r["ruta_archivo"]: r.to_dict() for _, r in df.iterrows()}
    return {}


def guardar_catalogo(filas: list[dict]) -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    with open(RUTA_CATALOGO, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        w.writeheader()
        w.writerows(filas)

# ---------------------------------------------------------------------------
# Descarga de una imagen
# ---------------------------------------------------------------------------

async def descargar_imagen(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    params: dict,
    ruta: Path,
) -> bool:
    """Descarga una imagen y la guarda en ruta. Devuelve True si OK."""
    for intento in range(1, MAX_REINTENTOS + 1):
        async with sem:
            try:
                async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        print(f"    [429] esperando {ESPERA_429_S}s…")
                    elif resp.status != 200:
                        print(f"    [HTTP {resp.status}] {ruta.name}")
                        return False
                    else:
                        data = await resp.read()
                        if len(data) < MIN_SIZE_BYTES:
                            # Imagen de "no imagery available"
                            return False
                        ruta.parent.mkdir(parents=True, exist_ok=True)
                        ruta.write_bytes(data)
                        return True
            except Exception as e:
                print(f"    [error intento {intento}] {ruta.name}: {e}")

        if intento < MAX_REINTENTOS:
            await asyncio.sleep(ESPERA_429_S)

    return False

# ---------------------------------------------------------------------------
# Construcción del plan de descarga
# ---------------------------------------------------------------------------

def construir_plan(gdf: gpd.GeoDataFrame, catalogo: dict) -> list[dict]:
    """
    Devuelve lista de tareas pendientes (imágenes no descargadas aún).
    Cada tarea es un dict con toda la info necesaria.
    """
    tareas = []
    for _, row in gdf.iterrows():
        id_est  = str(row["id_establecimiento"]).strip()
        id_sede = str(row["id_sede"]).strip()
        nombre_est  = str(row.get("nombre_establecimiento", "")).strip()
        nombre_sede = str(row.get("nombre_sede", "")).strip()
        lon = row.geometry.x
        lat = row.geometry.y

        for heading in HEADINGS:
            fname    = f"{id_sede}_{heading:03d}.jpg"
            ruta_rel = f"{id_est}/{fname}"
            ruta_abs = RUTA_SALIDA / id_est / fname

            entrada = catalogo.get(ruta_rel)
            if entrada and str(entrada.get("descargada", "")).lower() == "true" and not FORZAR_REDESCARGA:
                continue  # ya descargada

            tareas.append({
                "id_establecimiento": id_est,
                "id_sede":            id_sede,
                "nombre_establecimiento": nombre_est,
                "nombre_sede":        nombre_sede,
                "lat":   lat,
                "lon":   lon,
                "heading": heading,
                "ruta_rel": ruta_rel,
                "ruta_abs": ruta_abs,
            })
    return tareas

# ---------------------------------------------------------------------------
# Loop asíncrono principal
# ---------------------------------------------------------------------------

async def descargar_todas(tareas: list[dict], catalogo: dict) -> list[dict]:
    sem     = asyncio.Semaphore(MAX_CONCURRENT)
    filas   = list(catalogo.values())   # filas ya existentes en el catálogo
    ok = err = 0
    total = len(tareas)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = []
        for t in tareas:
            params = {
                "size":     IMG_SIZE,
                "location": f"{t['lat']},{t['lon']}",
                "heading":  t["heading"],
                "fov":      FOV,
                "pitch":    PITCH,
                "key":      API_KEY,
            }
            coros.append((t, descargar_imagen(session, sem, params, t["ruta_abs"])))

        for i, (t, coro) in enumerate(coros, 1):
            exito = await coro
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if exito:
                ok += 1
            else:
                err += 1

            filas.append({
                "id_establecimiento":     t["id_establecimiento"],
                "id_sede":                t["id_sede"],
                "nombre_establecimiento": t["nombre_establecimiento"],
                "nombre_sede":            t["nombre_sede"],
                "lat":            t["lat"],
                "lon":            t["lon"],
                "heading":        t["heading"],
                "ruta_archivo":   t["ruta_rel"],
                "descargada":     exito,
                "fecha_descarga": ts if exito else "",
            })

            if i % 50 == 0 or i == total:
                print(f"  {i}/{total} imágenes | OK: {ok} | Errores: {err}")
                guardar_catalogo(filas)  # checkpoint

    return filas

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: GSV_API_TOKEN no encontrado en .env")
        sys.exit(1)

    print("=" * 70)
    print("DESCARGA GOOGLE STREET VIEW — Colegios Bogotá")
    print("=" * 70)

    # Cargar sedes
    gdf = gpd.read_file(RUTA_SEDES)
    print(f"  {len(gdf)} sedes cargadas | CRS: {gdf.crs.to_string()}")
    gdf = aplicar_coordenadas_google(gdf)

    if MODO_MUESTRA:
        gdf = gdf.sample(min(N_MUESTRA, len(gdf)), random_state=42).reset_index(drop=True)
        print(f"  [MODO MUESTRA] {len(gdf)} sedes seleccionadas\n")

    # Catálogo existente
    catalogo = cargar_catalogo()
    ya_descargadas = sum(1 for r in catalogo.values() if str(r.get("descargada", "")).lower() == "true")
    print(f"  Catálogo previo: {ya_descargadas} imágenes ya descargadas")

    catalogo, invalidadas = invalidar_coords_cambiadas(gdf, catalogo)
    if invalidadas:
        print(f"  [coords] {invalidadas} entradas invalidadas por cambio de coordenadas (>{UMBRAL_COORD}°)")

    # Plan
    tareas = construir_plan(gdf, catalogo)
    print(f"  Imágenes pendientes: {len(tareas)} ({len(gdf)} sedes × {N_HEADINGS} headings)")
    print(f"  Costo estimado: ~${len(tareas) * 7 / 1000:.2f} USD\n")

    if not tareas:
        print("Nada que descargar. ¡Todo listo!")
        return

    # Descarga
    filas = asyncio.run(descargar_todas(tareas, catalogo))

    # Resumen final
    ok  = sum(1 for r in filas if str(r.get("descargada", "")).lower() == "true")
    err = sum(1 for r in filas if str(r.get("descargada", "")).lower() == "false")
    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"  Sedes procesadas:     {len(gdf)}")
    print(f"  Imágenes descargadas: {ok}")
    print(f"  Sin cobertura/error:  {err}")
    print(f"  Catálogo guardado en: {RUTA_CATALOGO}")
    print(f"  Imágenes en:          {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
