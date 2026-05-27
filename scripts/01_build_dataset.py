"""
Construye el dataset maestro de colegios oficiales de Bogota.

Unidad de analisis: establecimiento educativo (no sede).
Llave de cruce: DANE12_EST (codigo DANE de 12 digitos del establecimiento).

Pasos:
  1. Colegios: agrega sedes -> establecimiento (sede principal como punto geografico)
  2. Saber 11: construye q_j = promedio de puntajes 2020/2022/2023
  3. Matricula: agrega sedes -> establecimiento (suma TMATRIC_GE)
  4. Demanda: ya esta a nivel de establecimiento (DTotal)
  5. sobre_demanda_j = DTotal / TMATRIC_GE
  6. Controles EM2021: join por UPZ
  7. Controles geograficos: distancia al TM, SITP, parques mas cercanos
  8. Delitos: join por localidad

Inputs (data/processed/):
  colegios_dataset.geojson
  saber_bogota_merged.geojson
  demanda_clean.geojson
  matriculas_clean.geojson
  em2021_por_upz.csv

Inputs (data/raw/):
  estaciones_transmilenio.geojson
  paraderos_sitp.geojson
  parques_bogota.geojson
  delitos_alto_impacto.geojson

Output (data/processed/):
  colegios_features.geojson
"""

import json
import math
import unicodedata
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

warnings.filterwarnings("ignore")

PROC    = Path(__file__).parent.parent / "data" / "processed"
RAW     = Path(__file__).parent.parent / "data" / "raw"
PRIMARY = Path(__file__).parent.parent / "data" / "primary"
OUT     = PRIMARY / "colegios_features.geojson"


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_geojson(path: Path) -> tuple[pd.DataFrame, dict]:
    """Carga un GeoJSON y retorna (DataFrame de propiedades, dict feature por indice)."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    rows = []
    geoms = {}
    for i, feat in enumerate(gj["features"]):
        row = dict(feat["properties"])
        row["_geom"] = feat["geometry"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, gj


def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia en metros entre dos puntos (lat/lon en grados)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_distance(point_lon, point_lat, targets_lon, targets_lat):
    """Distancia en metros al punto mas cercano en targets."""
    min_d = float("inf")
    for tlon, tlat in zip(targets_lon, targets_lat):
        d = haversine_m(point_lat, point_lon, tlat, tlon)
        if d < min_d:
            min_d = d
    return min_d if min_d < float("inf") else np.nan


def extract_coords(geom):
    """Extrae (lon, lat) de una geometria Point."""
    if geom and geom.get("type") == "Point":
        coords = geom.get("coordinates", [None, None])
        return coords[0], coords[1]
    return np.nan, np.nan


def load_points_from_geojson(path: Path) -> tuple[list, list]:
    """
    Extrae listas de (lon, lat) de un GeoJSON.
    Soporta geometrias estandar (Point, Polygon, etc.) y el formato ESRI
    donde geometry = {"x": lon, "y": lat} sin campo "type".
    """
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lons, lats = [], []
    for feat in gj["features"]:
        g = feat.get("geometry") or {}
        gtype = g.get("type")

        # Formato ESRI: {"x": lon, "y": lat}
        if not gtype and "x" in g and "y" in g:
            x, y = g["x"], g["y"]
            if x is not None and y is not None:
                lons.append(float(x))
                lats.append(float(y))
            continue

        if not gtype:
            continue

        if gtype == "Point":
            lons.append(g["coordinates"][0])
            lats.append(g["coordinates"][1])
        elif gtype == "MultiPoint":
            for c in g["coordinates"]:
                lons.append(c[0])
                lats.append(c[1])
        elif gtype == "Polygon":
            ring = g["coordinates"][0]
            lons.append(sum(p[0] for p in ring) / len(ring))
            lats.append(sum(p[1] for p in ring) / len(ring))
        elif gtype == "MultiPolygon":
            for poly in g["coordinates"]:
                ring = poly[0]
                lons.append(sum(p[0] for p in ring) / len(ring))
                lats.append(sum(p[1] for p in ring) / len(ring))
    return lons, lats


# ── Paso 1: Colegios sedes -> establecimiento ─────────────────────────────────

def build_colegios(path: Path) -> pd.DataFrame:
    print("Paso 1 - Agregando sedes a nivel de establecimiento...")
    df, _ = load_geojson(path)

    # Extraer coordenadas de la geometria
    df[["lon", "lat"]] = df["_geom"].apply(
        lambda g: pd.Series(extract_coords(g))
    )

    # Normalizar orden_sede para identificar la sede principal
    df["orden_sede"] = df["orden_sede"].astype(str).str.strip().str.upper()

    # Sede principal: "PRINCIPAL" si existe, si no la de menor orden_sede
    def get_principal(grp):
        principal = grp[grp["orden_sede"] == "PRINCIPAL"]
        if len(principal) > 0:
            return principal.iloc[0]
        return grp.sort_values("orden_sede").iloc[0]

    result = (
        df.groupby("id_establecimiento", group_keys=False)
        .apply(get_principal, include_groups=False)
        .reset_index()  # id_establecimiento queda como columna
    )

    # Construir DANE12: los primeros 12 chars del id_sede o id_establecimiento
    # El directorio no trae DANE12_EST directamente -- lo derivamos del id
    # id_establecimiento suele ser el codigo DANE de 12 digitos
    result["DANE12_EST"] = result["id_establecimiento"].astype(str).str.strip().str.zfill(12)

    cols = [
        "DANE12_EST", "id_establecimiento", "nombre_establecimiento",
        "nombre_localidad", "codigo_upz", "nombre_upz",
        "estrato", "zona", "caracter_media",
        "lon", "lat", "_geom",
    ]
    result = result[[c for c in cols if c in result.columns]]

    # Asignar localidad via sjoin con poligonos oficiales (reemplaza string del CSV)
    result = asignar_localidad_sjoin(result)

    print(f"  {len(result):,} establecimientos")
    return result


def asignar_localidad_sjoin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza nombre_localidad del CSV por el nombre oficial del poligono
    de localidades (Datos Abiertos Bogota). Usa sjoin nearest como fallback
    para colegios que caigan fuera de los poligonos (rurales en borde).
    """
    loc_path = RAW / "localidades_bogota.geojson"
    if not loc_path.exists():
        print("  AVISO: localidades_bogota.geojson no encontrado, se mantiene nombre del CSV")
        return df

    print("  Asignando localidad via sjoin con poligonos oficiales...")
    localidades = gpd.read_file(loc_path)[["LocNombre", "LocCodigo", "geometry"]].copy()
    localidades["LocNombre"] = localidades["LocNombre"].str.title()   # USME -> Usme

    # Crear GeoDataFrame de colegios
    mask_valid = df["lon"].notna() & df["lat"].notna()
    gdf_col = gpd.GeoDataFrame(
        df[mask_valid].copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df[mask_valid]["lon"], df[mask_valid]["lat"])],
        crs="EPSG:4326"
    )

    # sjoin within: asigna localidad si el punto cae dentro del poligono
    joined = gpd.sjoin(gdf_col, localidades, how="left", predicate="within")

    # Fallback nearest para puntos fuera de poligonos (borde rural)
    sin_loc = joined["LocNombre"].isna()
    if sin_loc.any():
        nearest = gpd.sjoin_nearest(gdf_col[sin_loc], localidades, how="left")
        joined.loc[sin_loc, "LocNombre"] = nearest["LocNombre"].values
        joined.loc[sin_loc, "LocCodigo"] = nearest["LocCodigo"].values
        print(f"    {sin_loc.sum()} colegios asignados por nearest (borde)")

    df = df.copy()
    df.loc[mask_valid, "nombre_localidad"] = joined["LocNombre"].values
    df.loc[mask_valid, "codigo_localidad"]  = joined["LocCodigo"].values
    n_ok = df["nombre_localidad"].notna().sum()
    print(f"    {n_ok} / {len(df)} colegios con localidad asignada via poligono")
    return df


# ── Paso 2: Saber 11 -> q_j ──────────────────────────────────────────────────

def build_qj(path: Path) -> pd.DataFrame:
    print("Paso 2 - Construyendo q_j desde Saber 11...")
    df, _ = load_geojson(path)

    df["DANE12_EST"] = df["DANE12_EST"].astype(str).str.strip().str.zfill(12)

    puntaje_cols = [c for c in ["puntaje_2023", "punt_global_2020", "punt_global_2022"] if c in df.columns]
    for c in puntaje_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["q_j"] = df[puntaje_cols].mean(axis=1, skipna=True)

    result = df[["DANE12_EST", "q_j"] + puntaje_cols].copy()
    result = result.drop_duplicates("DANE12_EST")
    print(f"  {len(result):,} establecimientos con Saber 11 | q_j NaN: {result['q_j'].isna().sum()}")
    return result


# ── Paso 3: Matricula sedes -> establecimiento ────────────────────────────────

def build_matricula(path: Path) -> pd.DataFrame:
    print("Paso 3 - Agregando matricula por establecimiento...")
    df, _ = load_geojson(path)

    df["DANE12_EST"] = df["DANE12_EST"].astype(str).str.strip().str.zfill(12)
    df["TMATRIC_GE"] = pd.to_numeric(df["TMATRIC_GE"], errors="coerce").fillna(0)

    result = (
        df.groupby("DANE12_EST", as_index=False)
        .agg(matricula_total=("TMATRIC_GE", "sum"))
    )
    print(f"  {len(result):,} establecimientos con matricula")
    return result


# ── Paso 4: Demanda ───────────────────────────────────────────────────────────

def build_demanda(path: Path) -> pd.DataFrame:
    print("Paso 4 - Cargando demanda de cupos...")
    df, _ = load_geojson(path)

    df["DANE12_EST"] = df["DANE12_EST"].astype(str).str.strip().str.zfill(12)
    df["DTotal"] = pd.to_numeric(df["DTotal"], errors="coerce")

    # Si hay duplicados (poco probable), sumar
    result = df.groupby("DANE12_EST", as_index=False).agg(demanda_total=("DTotal", "sum"))
    print(f"  {len(result):,} establecimientos con demanda")
    return result


# ── Paso 5: sobre_demanda_j ───────────────────────────────────────────────────

def build_sobre_demanda(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula sobre_demanda_j = demanda_total / matricula_total."""
    mask_valid = (df["matricula_total"] > 0) & df["demanda_total"].notna()
    df["sobre_demanda_j"] = np.where(
        mask_valid,
        df["demanda_total"] / df["matricula_total"],
        np.nan,
    )
    n_valid = mask_valid.sum()
    print(f"Paso 5 - sobre_demanda_j: {n_valid} validos | {(~mask_valid).sum()} NaN")
    return df


# ── Paso 6: Controles EM2021 por UPZ ─────────────────────────────────────────

def join_em2021(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    print("Paso 6 - Uniendo controles EM2021 por UPZ...")
    em = pd.read_csv(path, dtype=str)
    em["COD_UPZ_GRUPO"] = em["COD_UPZ_GRUPO"].str.strip().str.zfill(3)

    # Llave de join: codigo_upz del directorio -> COD_UPZ_GRUPO de EM2021
    df["_upz_key"] = df["codigo_upz"].astype(str).str.strip().str.zfill(3)

    # Convertir columnas numericas de EM2021
    em_num_cols = [c for c in em.columns if c != "COD_UPZ_GRUPO"]
    for c in em_num_cols:
        em[c] = pd.to_numeric(em[c], errors="coerce")

    result = df.merge(
        em.rename(columns={"COD_UPZ_GRUPO": "_upz_key"}),
        on="_upz_key",
        how="left",
    ).drop(columns=["_upz_key"])

    matched = result["tasa_pobreza_monetaria"].notna().sum()
    print(f"  {matched} / {len(result)} colegios con match en EM2021")
    return result


# ── Paso 6b: s₀ por localidad (participacion del sector no oficial) ──────────

def join_s0_localidad(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    print("Paso 6b - Calculando s0 por localidad desde EM2021...")
    em = pd.read_csv(path, dtype=str)

    em["FEX_C"]  = pd.to_numeric(em["FEX_C"],  errors="coerce")
    em["NPCEP4"] = pd.to_numeric(em["NPCEP4"], errors="coerce")
    em["COD_LOCALIDAD"] = em["COD_LOCALIDAD"].astype(str).str.strip()

    edad_escolar = em[em["NPCEP4"].between(5, 17)].copy()

    n_escolar = (
        edad_escolar
        .groupby("COD_LOCALIDAD")["FEX_C"]
        .sum()
        .rename("N_escolar")
    )
    n_oficial = (
        edad_escolar[
            (edad_escolar["NPCHP2"].str.strip() == "1") &
            (edad_escolar["NPCHP12"].str.strip() == "1")
        ]
        .groupby("COD_LOCALIDAD")["FEX_C"]
        .sum()
        .rename("N_oficial")
    )

    s0_df = pd.concat([n_escolar, n_oficial], axis=1).reset_index()
    s0_df["s0_localidad"] = 1 - (s0_df["N_oficial"] / s0_df["N_escolar"])

    print(f"  s0 por localidad — min: {s0_df['s0_localidad'].min():.4f}  "
          f"mediana: {s0_df['s0_localidad'].median():.4f}  "
          f"max: {s0_df['s0_localidad'].max():.4f}")

    # Normalizar a int para evitar mismatch "01" vs "1"
    s0_df["_loc_key"] = pd.to_numeric(s0_df["COD_LOCALIDAD"], errors="coerce").astype("Int64")
    df["_loc_key"] = pd.to_numeric(df["codigo_localidad"], errors="coerce").astype("Int64")
    result = df.merge(
        s0_df[["_loc_key", "s0_localidad"]],
        on="_loc_key",
        how="left",
    ).drop(columns=["_loc_key"])

    matched = result["s0_localidad"].notna().sum()
    print(f"  {matched} / {len(result)} colegios con s0_localidad")
    return result


# ── Paso 7: Controles geograficos ────────────────────────────────────────────

def join_geo_controls(df: pd.DataFrame) -> pd.DataFrame:
    print("Paso 7 - Calculando distancias a equipamientos...")

    sources = {
        "dist_transmilenio_m": RAW  / "estaciones_transmilenio.geojson",
        "dist_sitp_m":         PROC / "sitp_clean.geojson",
        "dist_parque_m":       PROC / "parques_clean.geojson",   # pendiente re-descarga
    }

    for col, path in sources.items():
        if not path.exists():
            print(f"  AVISO: {path.name} no encontrado -- {col} = NaN")
            df[col] = np.nan
            continue
        try:
            lons, lats = load_points_from_geojson(path)
        except Exception as e:
            print(f"  AVISO: error leyendo {path.name} ({e}) -- {col} = NaN")
            df[col] = np.nan
            continue
        distances = []
        for _, row in df.iterrows():
            if pd.isna(row["lon"]) or pd.isna(row["lat"]):
                distances.append(np.nan)
            else:
                distances.append(nearest_distance(row["lon"], row["lat"], lons, lats))
        df[col] = distances
        median_d = pd.Series(distances).median()
        print(f"  {col}: mediana = {median_d:.0f} m")

    return df


# ── Paso 8: Delitos por localidad ─────────────────────────────────────────────

def join_delitos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Une controles de delitos desde data/processed/delitos_por_localidad.csv
    (generado por scripts/clean_delitos.py).
    """
    import unicodedata

    # Mapeo de nombres del poligono oficial -> nombre en el CSV de delitos
    LOCALIDAD_ALIAS = {
        "CANDELARIA": "LA CANDELARIA",
        "MARTIRES":   "LOS MARTIRES",
    }

    def normalize_str(s):
        s = str(s).strip().upper()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return LOCALIDAD_ALIAS.get(s, s)

    print("Paso 8 - Uniendo delitos por localidad...")
    path = PROC / "delitos_por_localidad.csv"
    if not path.exists():
        print(f"  AVISO: {path.name} no encontrado -- corre clean_delitos.py primero")
        return df

    del_df = pd.read_csv(path, encoding="utf-8")
    # localidad_norm ya esta normalizada en el processed
    del_df = del_df.rename(columns={"localidad_norm": "_loc_key"})

    df["_loc_key"] = df["nombre_localidad"].apply(normalize_str)
    df = df.merge(del_df, on="_loc_key", how="left").drop(columns=["_loc_key"])

    matched = df["homicidios"].notna().sum()
    print(f"  {matched} / {len(df)} colegios con match en delitos")
    print(f"  Columnas de delitos agregadas: {list(del_df.columns[1:])}")
    return df


# ── Paso 9: Competencia sector privado ───────────────────────────────────────

def join_competencia_privada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Une el porcentaje de sedes no oficiales por localidad desde
    data/processed/competencia_privada_localidad.csv
    (generado por scripts/00_build_competencia_privada.py).
    """
    import unicodedata

    def normalize_str(s):
        s = str(s).strip().upper()
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    # Mapeo de nombres del poligono oficial -> nombre en el CSV de competencia
    LOCALIDAD_ALIAS = {
        "CANDELARIA": "LA CANDELARIA",
        "MARTIRES":   "LOS MARTIRES",
    }

    def normalize_str(s):
        s = str(s).strip().upper()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return LOCALIDAD_ALIAS.get(s, s)

    print("Paso 9 - Uniendo competencia sector privado por localidad...")
    path = PROC / "competencia_privada_localidad.csv"
    if not path.exists():
        print(f"  AVISO: {path.name} no encontrado -- corre 00_build_competencia_privada.py primero")
        return df

    comp = pd.read_csv(path, encoding="utf-8")
    comp = comp.rename(columns={"localidad_norm": "_loc_key"})

    df["_loc_key"] = df["nombre_localidad"].apply(normalize_str)
    df = df.merge(
        comp[["_loc_key", "pct_no_oficial", "n_sedes_total", "n_sedes_no_oficial"]],
        on="_loc_key", how="left"
    ).drop(columns=["_loc_key"])

    matched = df["pct_no_oficial"].notna().sum()
    print(f"  {matched} / {len(df)} colegios con dato de competencia privada")
    return df


# ── Paso 10: Output GeoJSON ───────────────────────────────────────────────────

def save_geojson(df: pd.DataFrame, path: Path) -> None:
    features = []
    for _, row in df.iterrows():
        geom = row.get("_geom")
        props = {
            k: (None if (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in row.items()
            if k != "_geom"
        }
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    gj = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)
    size_mb = path.stat().st_size / 1e6
    print(f"\nGuardado: {path}  ({len(features):,} features | {size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PROC.mkdir(parents=True, exist_ok=True)
    PRIMARY.mkdir(parents=True, exist_ok=True)

    # Paso 1-4: cargar y preparar cada fuente
    df      = build_colegios(PROC / "colegios_dataset.geojson")
    df_qj   = build_qj(PROC / "saber_bogota_merged.geojson")
    df_mat  = build_matricula(PROC / "matriculas_clean.geojson")
    df_dem  = build_demanda(PROC / "demanda_clean.geojson")

    # Joins secuenciales
    df = df.merge(df_qj,  on="DANE12_EST", how="left")
    df = df.merge(df_mat, on="DANE12_EST", how="left")
    df = df.merge(df_dem, on="DANE12_EST", how="left")

    # Paso 5: construir variable dependiente
    df = build_sobre_demanda(df)

    # Paso 6: controles socioeconomicos
    df = join_em2021(df, PROC / "em2021_por_upz.csv")

    # Paso 6b: s0 por localidad
    df = join_s0_localidad(df, RAW / "em2021_encuesta_principal.csv")

    # Paso 7: distancias geograficas
    df = join_geo_controls(df)

    # Excluir solo Sumapaz (zona rural sin cobertura EM2021 ni familias encuestadas)
    n_antes = len(df)
    df = df[~df["nombre_localidad"].isin(["Sumapaz"])].reset_index(drop=True)
    print(f"  Sumapaz excluida: {n_antes - len(df)} establecimientos removidos")

    # Paso 8: delitos
    df = join_delitos(df)

    # Paso 9: competencia del sector privado por localidad
    df = join_competencia_privada(df)

    # Resumen
    print("\n=== Resumen del dataset ===")
    print(f"  Total establecimientos:   {len(df):,}")
    print(f"  Con q_j (Saber 11):       {df['q_j'].notna().sum():,}")
    print(f"  Con sobre_demanda_j:      {df['sobre_demanda_j'].notna().sum():,}")
    print(f"  Con controles EM2021:     {df['tasa_pobreza_monetaria'].notna().sum():,}")
    print(f"  Con dist_transmilenio_m:  {df['dist_transmilenio_m'].notna().sum():,}")
    print(f"\n  Columnas totales: {len(df.columns)}")

    save_geojson(df, OUT)


if __name__ == "__main__":
    main()
