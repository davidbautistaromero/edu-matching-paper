"""
mapillary_filtros.py
====================
Parámetros de selección y función aplicar_filtros compartidos entre
00_analyze_mapillary_colegios.py y 00_download_mapillary_colegios.py.

Fuente de verdad única: cambiar aquí afecta ambos scripts simultáneamente.

PARÁMETROS
----------
FECHA_DESDE            : descarta imágenes anteriores a esta fecha.
FECHA_HASTA            : descarta imágenes posteriores a esta fecha.
EXCLUIR_PANORAMICAS    : elimina imágenes 360° incompatibles con VGG19.
DEDUP_POR_SECUENCIA    : 1 imagen (la más cercana) por (colegio, secuencia),
                         evita pseudorreplicación de fotogramas consecutivos.
ANGULO_MAX_DEG         : máxima diferencia entre compass_angle y el rumbo
                         imagen→colegio. Selecciona imágenes donde la cámara
                         apunta hacia el edificio, no a lo largo de la calle.
                         None para desactivar.
DIST_DEDUP_ESPACIAL_M  : distancia mínima en metros entre imágenes seleccionadas
                         del mismo colegio. Elimina tomas desde el mismo punto
                         físico aunque vengan de secuencias distintas.
                         0 para desactivar.
N_MAX_POR_COLEGIO      : máximo de imágenes por establecimiento tras todos los
                         filtros anteriores, ordenadas por distancia. 0 = sin límite.
MODO_MUESTRA / N_MUESTRA : prueba con N colegios aleatorios antes del run completo.
"""

import math
import pandas as pd

# Rango de captura (YYYY-MM-DD). "" o None para no filtrar cada extremo.
FECHA_DESDE = "2020-01-01"
FECHA_HASTA = "2024-12-31"

# True → excluir panorámicas incompatibles con VGG19 (distorsión equirectangular).
EXCLUIR_PANORAMICAS = False

# True → 1 imagen por (colegio, secuencia), la más cercana al colegio.
DEDUP_POR_SECUENCIA = True

# Máxima diferencia angular (grados) entre la dirección de la cámara y el
# rumbo imagen→colegio. 30° es más estricto que 45° — la cámara debe apuntar
# casi directamente al edificio. None desactiva el filtro.
ANGULO_MAX_DEG = 90

# Distancia mínima en metros entre dos imágenes seleccionadas del mismo colegio.
# Elimina tomas capturadas desde el mismo punto físico por distintas secuencias
# (p.ej. varios recorridos que pasan por la misma esquina). 0 para desactivar.
DIST_DEDUP_ESPACIAL_M = 10

# Máximo de imágenes por establecimiento tras todos los filtros.
# 0 = sin límite.
N_MAX_POR_COLEGIO = 10

# Modo de prueba: consulta y descarga solo N_MUESTRA colegios aleatorios.
# Útil para validar ángulos y encuadres antes de procesar el catálogo completo.
# False = procesar todos los colegios.
MODO_MUESTRA = False
N_MUESTRA    = 10


# ---------------------------------------------------------------------------
# Funciones auxiliares (privadas)
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en metros entre dos puntos WGS84."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2)
        * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def _dedup_espacial_grupo(grupo: pd.DataFrame, min_dist_m: float) -> pd.DataFrame:
    """
    Filtra imágenes demasiado cercanas entre sí dentro de un colegio.

    Algoritmo greedy: ordena por distancia_m al colegio y va seleccionando
    imágenes siempre que estén al menos min_dist_m de todas las ya elegidas.
    Así se garantiza diversidad espacial sin importar cuántas secuencias
    distintas pasaron por el mismo punto.
    """
    grupo = grupo.reset_index(drop=True)
    indices_sel: list[int] = []
    coords_sel:  list[tuple] = []

    for row in grupo.itertuples(index=True):
        lat = getattr(row, "lat_img", None)
        lon = getattr(row, "lon_img", None)
        if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)):
            indices_sel.append(row.Index)
            continue
        demasiado_cerca = any(
            _haversine_m(lat, lon, lat2, lon2) < min_dist_m
            for lat2, lon2 in coords_sel
        )
        if not demasiado_cerca:
            indices_sel.append(row.Index)
            coords_sel.append((lat, lon))

    return grupo.loc[indices_sel]


# ---------------------------------------------------------------------------
# Filtro principal
# ---------------------------------------------------------------------------

def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los criterios de selección definidos en los parámetros y devuelve
    el subconjunto de imágenes para descarga.

    Orden de aplicación:
      1. Rango de fechas (FECHA_DESDE / FECHA_HASTA).
      2. Exclusión de panorámicas (EXCLUIR_PANORAMICAS).
      3. Deduplicación por secuencia (DEDUP_POR_SECUENCIA).
      4. Filtro de ángulo (ANGULO_MAX_DEG).
      5. Deduplicación espacial (DIST_DEDUP_ESPACIAL_M): elimina imágenes
         capturadas desde el mismo punto físico en distintas secuencias.
      6. Límite por colegio (N_MAX_POR_COLEGIO).

    is_pano se normaliza de string CSV a booleano antes de filtrar.
    """
    sel = df.copy()
    sel["is_pano"] = sel["is_pano"].astype(str).str.strip().str.lower() == "true"

    if FECHA_DESDE:
        sel = sel[sel["fecha"] >= FECHA_DESDE]

    if FECHA_HASTA:
        sel = sel[sel["fecha"] <= FECHA_HASTA]

    if EXCLUIR_PANORAMICAS:
        sel = sel[~sel["is_pano"]]

    if DEDUP_POR_SECUENCIA:
        sel = (
            sel.sort_values("distancia_m", na_position="last")
               .drop_duplicates(subset=["dane_est", "sequence"], keep="first")
        )

    if ANGULO_MAX_DEG is not None and "diff_angulo" in sel.columns:
        # Las panorámicas 360° capturan todas las direcciones; su compass_angle
        # refleja el rumbo del vehículo, no la dirección de la cámara, así que
        # el filtro de ángulo no aplica para ellas.
        sin_dato = sel["diff_angulo"].isna()
        sel = sel[sel["is_pano"] | sin_dato | (sel["diff_angulo"] <= ANGULO_MAX_DEG)]

    if DIST_DEDUP_ESPACIAL_M and not sel.empty:
        # Loop explícito en lugar de groupby().apply() para evitar que pandas 2.x
        # mueva dane_est al índice al reconstruir el resultado del apply.
        grupos = [
            _dedup_espacial_grupo(g, DIST_DEDUP_ESPACIAL_M)
            for _, g in sel.sort_values("distancia_m", na_position="last")
                           .groupby("dane_est", sort=False)
        ]
        sel = pd.concat(grupos, ignore_index=True) if grupos else sel.iloc[0:0]

    if N_MAX_POR_COLEGIO:
        sel = (
            sel.sort_values("distancia_m", na_position="last")
               .groupby("dane_est", group_keys=False)
               .head(N_MAX_POR_COLEGIO)
        )

    return sel.reset_index(drop=True)
