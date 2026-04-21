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
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

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
    print(f"  {len(result):,} establecimientos")
    return result


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

    def normalize_str(s):
        s = str(s).strip().upper()
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

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

    # Paso 7: distancias geograficas
    df = join_geo_controls(df)

    # Paso 8: delitos
    df = join_delitos(df)

    # Paso 9: competencia del sector privado por localidad
    df = join_competencia_privada(df)

    # Placeholder para indice visual (se completa en 02_visual_index.py)
    df["v_j"] = np.nan

    # Resumen
    print("\n=== Resumen del dataset ===")
    print(f"  Total establecimientos:   {len(df):,}")
    print(f"  Con q_j (Saber 11):       {df['q_j'].notna().sum():,}")
    print(f"  Con sobre_demanda_j:      {df['sobre_demanda_j'].notna().sum():,}")
    print(f"  Con controles EM2021:     {df['tasa_pobreza_monetaria'].notna().sum():,}")
    print(f"  Con dist_transmilenio_m:  {df['dist_transmilenio_m'].notna().sum():,}")
    print(f"  Con v_j:                  {df['v_j'].notna().sum():,}  (pendiente embeddings)")
    print(f"\n  Columnas totales: {len(df.columns)}")

    save_geojson(df, OUT)


if __name__ == "__main__":
    main()
