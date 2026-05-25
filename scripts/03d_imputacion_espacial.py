#!/usr/bin/env python3
"""
03b_imputacion_espacial.py
==========================
Imputa valores faltantes en colegios_features.geojson usando un radio de
vecindad espacial (haversine via sklearn BallTree), con fallback a la mediana
global de la columna.

Fundamento metodológico:
  Las variables socioeconómicas y de infraestructura exhiben autocorrelación
  espacial fuerte en Bogotá (escuelas cercanas comparten UPZ, condiciones de
  calle, densidad de transporte). Imputar con vecinos geográficos preserva esa
  estructura espacial mejor que la mediana global, reduciendo el sesgo de
  atenuación en las regresiones de la sección empírica.
"""
import os

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# =============================================================================
# CONFIGURACIÓN
# Las únicas líneas que deberías cambiar si se mueven los archivos o si
# quieres explorar un radio diferente de vecindad.
# =============================================================================

RADIO_KM    = 2.0
INPUT_PATH  = r'C:\paper-AI\data\primary\colegios_features.geojson'
OUTPUT_PATH = r'C:\paper-AI\data\primary\colegios_features_imputed.geojson'

EARTH_RADIUS_KM = 6371.0

# Columnas que no deben imputarse: identificadores, coordenadas y categóricas.
# Todo lo demás se detecta automáticamente como numérico.
EXCLUIR = {
    'id_establecimiento', 'DANE12_EST', 'codigo_upz',
    'lon', 'lat', 'geometry',
    'nombre_establecimiento', 'nombre_localidad', 'nombre_upz',
    'zona', 'caracter_media',
}

# =============================================================================
# LOGGING  (mismo estilo que 04_regresion.py y 02_seg_cityscapes.py)
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# =============================================================================
# UTILIDADES
# =============================================================================

def nan_report(df: pd.DataFrame, cols: list[str], label: str) -> None:
    """Imprime conteo de NaN por columna para auditoría del pipeline."""
    log.info(f'── NaN por columna {label} ──')
    any_nan = False
    for col in cols:
        n = df[col].isna().sum()
        if n > 0:
            log.info(f'  {col}: {n}')
            any_nan = True
    if not any_nan:
        log.info('  (ningún valor faltante)')


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info('=' * 65)
    log.info('IMPUTACIÓN ESPACIAL — colegios_features.geojson')
    log.info('=' * 65)

    # ── Paso 1: Cargar GeoJSON, reportar NaN antes de imputar ────────────────
    log.info(f'Leyendo: {INPUT_PATH}')
    gdf = gpd.read_file(INPUT_PATH)
    log.info(f'  {len(gdf):,} establecimientos, {gdf.shape[1]} columnas')

    # ── Filtrar colegios rurales excluidos ────────────────────────────────────
    excl_path = os.path.join(os.path.dirname(INPUT_PATH), '..', 'raw', 'excluded_schools.csv')
    if os.path.exists(excl_path):
        excl_df = pd.read_csv(excl_path, dtype={'id_establecimiento': str})
        excl_ids = set(excl_df['id_establecimiento'].str.strip())
        gdf['id_establecimiento'] = gdf['id_establecimiento'].astype(str).str.strip()
        mask_rural = gdf['id_establecimiento'].isin(excl_ids)
        n_rural = int(mask_rural.sum())
        gdf = gdf[~mask_rural].reset_index(drop=True)
        log.info(f'  Excluidos {n_rural} colegios rurales, quedan {len(gdf):,}')

    # Trabajar sobre una vista plana (sin geometría) para detectar tipos y NaN.
    # Mantenemos gdf como contenedor principal para el GeoJSON de salida.
    df_attrs = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))

    # Detectar automáticamente columnas numéricas candidatas a imputación.
    # Excluimos identificadores y categóricas definidos en EXCLUIR para no
    # contaminar el modelo con variables que no deben interpolarse.
    num_cols = [
        c for c in df_attrs.select_dtypes(include=[np.number]).columns
        if c not in EXCLUIR
    ]
    log.info(f'  Columnas numéricas detectadas: {len(num_cols)}')

    cols_con_nan = [c for c in num_cols if df_attrs[c].isna().any()]
    log.info(f'  Columnas con valores faltantes: {len(cols_con_nan)}')
    nan_report(df_attrs, cols_con_nan, 'ANTES de imputación')

    if not cols_con_nan:
        log.info('No hay valores faltantes en columnas numéricas. Nada que hacer.')
        return

    # ── Paso 2: Extraer coordenadas en radianes para BallTree haversine ───────
    # BallTree con métrica haversine espera (lat, lon) en radianes.
    # Preferimos columnas explícitas lon/lat si existen; si no, usamos el
    # centroide de la geometría (cubre el caso de polígonos).
    if 'lat' in df_attrs.columns and 'lon' in df_attrs.columns:
        lats = df_attrs['lat'].values.astype(float)
        lons = df_attrs['lon'].values.astype(float)
    else:
        centroids = gdf.geometry.centroid
        lons = centroids.x.values.astype(float)
        lats = centroids.y.values.astype(float)

    # Escuelas sin coordenadas válidas no pueden participar como fuentes de
    # vecindad ni ser imputadas espacialmente; reciben la mediana global.
    mask_valid_coords = ~(np.isnan(lats) | np.isnan(lons))
    n_sin_coords = int((~mask_valid_coords).sum())
    if n_sin_coords > 0:
        log.info(f'  AVISO: {n_sin_coords} escuelas sin coordenadas → usarán mediana global')

    coords_rad = np.deg2rad(np.column_stack([lats, lons]))

    # ── Paso 3: Construir BallTree sobre escuelas con coordenadas válidas ─────
    # El árbol se construye una sola vez y se reutiliza para todas las columnas.
    # Guardamos valid_indices para mapear los índices del árbol (0..M-1) de
    # vuelta a índices globales del DataFrame (0..N-1).
    valid_indices = np.where(mask_valid_coords)[0]
    tree = BallTree(coords_rad[mask_valid_coords], metric='haversine')

    # Convertir radio en km a radianes (unidad interna de BallTree haversine)
    radio_rad = RADIO_KM / EARTH_RADIUS_KM
    log.info(f'  BallTree construido con {len(valid_indices):,} escuelas '
             f'(radio = {RADIO_KM} km)')

    # ── Paso 4: Imputar columna por columna ───────────────────────────────────
    reporte = []  # acumula (col, n_espaciales, n_fallback) para el resumen final

    for col in cols_con_nan:
        valores = df_attrs[col].values.copy().astype(float)
        missing_idx = np.where(np.isnan(valores))[0]

        # Mediana global calculada antes de tocar valores — sirve de ancla para
        # el fallback y no se contamina con los valores que vamos imputando.
        mediana_global = float(np.nanmedian(valores))

        n_espaciales = 0
        n_fallback   = 0

        # Pre-computar medianas por UPZ y por localidad para fallback jerárquico
        upz_col = 'codigo_upz' if 'codigo_upz' in df_attrs.columns else None
        loc_col = 'nombre_localidad' if 'nombre_localidad' in df_attrs.columns else None

        mediana_upz = {}
        mediana_loc = {}
        if upz_col:
            for upz, group in df_attrs.groupby(upz_col):
                med = group[col].median()
                if not np.isnan(med):
                    mediana_upz[upz] = med
        if loc_col:
            for loc, group in df_attrs.groupby(loc_col):
                med = group[col].median()
                if not np.isnan(med):
                    mediana_loc[loc] = med

        n_upz_fb = 0
        n_loc_fb = 0

        for i in missing_idx:
            # Escuelas sin coordenadas: fallback jerárquico directo
            if not mask_valid_coords[i]:
                fb = None
                if upz_col and df_attrs[upz_col].iloc[i] in mediana_upz:
                    fb = mediana_upz[df_attrs[upz_col].iloc[i]]
                    n_upz_fb += 1
                elif loc_col and df_attrs[loc_col].iloc[i] in mediana_loc:
                    fb = mediana_loc[df_attrs[loc_col].iloc[i]]
                    n_loc_fb += 1
                else:
                    fb = mediana_global
                    n_fallback += 1
                valores[i] = fb
                continue

            # query_radius devuelve índices relativos al sub-árbol (valid_indices)
            neighbors_rel = tree.query_radius(
                coords_rad[i].reshape(1, -1),
                r=radio_rad,
            )[0]

            # Convertir índices del árbol a índices globales del DataFrame
            neighbors_global = valid_indices[neighbors_rel]

            # Excluir la propia escuela (no puede imputarse con su propio NaN)
            neighbors_global = neighbors_global[neighbors_global != i]

            # Filtrar vecinos que sí tienen dato en esta columna
            valid_neighbors = neighbors_global[~np.isnan(valores[neighbors_global])]

            if len(valid_neighbors) > 0:
                # Imputación espacial: media de vecinos con dato en el radio
                valores[i] = float(np.mean(valores[valid_neighbors]))
                n_espaciales += 1
            else:
                # Fallback jerárquico: UPZ → localidad → mediana global
                fb = None
                if upz_col and df_attrs[upz_col].iloc[i] in mediana_upz:
                    fb = mediana_upz[df_attrs[upz_col].iloc[i]]
                    n_upz_fb += 1
                elif loc_col and df_attrs[loc_col].iloc[i] in mediana_loc:
                    fb = mediana_loc[df_attrs[loc_col].iloc[i]]
                    n_loc_fb += 1
                else:
                    fb = mediana_global
                    n_fallback += 1
                valores[i] = fb

        # Escribir los valores imputados de vuelta al GeoDataFrame
        gdf[col] = valores
        reporte.append((col, n_espaciales, n_upz_fb, n_loc_fb, n_fallback))
        log.info(f'  {col}: {n_espaciales} espacial + {n_upz_fb} UPZ + {n_loc_fb} localidad + {n_fallback} global')

    # ── Paso 5: Verificar que no queden NaN en columnas numéricas ─────────────
    df_post = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
    nan_report(df_post, cols_con_nan, 'DESPUÉS de imputación')

    remaining = sum(df_post[c].isna().sum() for c in cols_con_nan)
    if remaining > 0:
        log.warning(f'  Aún quedan {remaining} NaN — revisar columnas sin mediana válida')

    # ── Paso 6: Excluir colegios con missing en q_j Y en controles EM2021 ──────
    mask_excluir = gdf['q_j'].isna() & gdf['tasa_pobreza_monetaria'].isna()
    n_excluidos = int(mask_excluir.sum())
    gdf = gdf[~mask_excluir].reset_index(drop=True)
    log.info(f'  Excluidos {n_excluidos} colegios con missing en q_j y EM2021 simultaneamente')

    # ── Paso 7: Guardar GeoJSON con el mismo CRS que el input ─────────────────
    gdf.to_file(OUTPUT_PATH, driver='GeoJSON')
    size_mb = Path(OUTPUT_PATH).stat().st_size / 1e6
    log.info(f'Guardado: {OUTPUT_PATH}  ({len(gdf):,} features | {size_mb:.1f} MB)')

    # ── Paso 7: Reporte resumen ────────────────────────────────────────────────
    log.info('=' * 65)
    log.info('RESUMEN DE IMPUTACIÓN')
    log.info(f"  {'Columna':<32} {'Espacial':>9} {'UPZ':>9} {'Localidad':>9} {'Global':>9}")
    log.info('  ' + '-' * 72)
    for col, n_esp, n_upz, n_loc, n_fb in reporte:
        log.info(f'  {col:<32} {n_esp:>9,} {n_upz:>9,} {n_loc:>9,} {n_fb:>9,}')
    log.info('  ' + '-' * 72)
    total_esp = sum(r[1] for r in reporte)
    total_upz = sum(r[2] for r in reporte)
    total_loc = sum(r[3] for r in reporte)
    total_fb  = sum(r[4] for r in reporte)
    log.info(f"  {'TOTAL':<32} {total_esp:>9,} {total_upz:>9,} {total_loc:>9,} {total_fb:>9,}")
    log.info('=' * 65)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == '__main__':
    main()
