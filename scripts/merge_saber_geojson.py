"""
Une todas las fuentes de resultados Saber 11 en un único GeoJSON.

Parte 1: agrega el CSV (2020-2022) por (anio, cole_caracter,
          cole_cod_dane_establecimiento) y promedia punt_global.

Parte 2: toma el GeoJSON de 2023 como base (DANE12_EST, NOMBRE_EST,
          ORDEN_DE_S, P_Puntaje_ → puntaje_2023 + geometría) y le une
          cole_caracter y punt_global_YYYY (una columna por año del CSV).
"""

import json
import pandas as pd
from pathlib import Path

CSV_PATH      = Path(__file__).parent.parent / "data" / "raw" / "saber11_bogota_2020_2022.csv"
SABER23_PATH  = Path(__file__).parent.parent / "data" / "raw" / "pruebassaber2023.geojson"
OUT_PATH      = Path(__file__).parent.parent / "data" / "processed" / "saber_bogota_merged.geojson"

GROUP_COLS = ["anio", "cole_caracter", "cole_cod_dane_establecimiento"]


# ── Parte 1 ──────────────────────────────────────────────────────────────────

def aggregate_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype={"cole_cod_dane_establecimiento": str})
    df["punt_global"] = pd.to_numeric(df["punt_global"], errors="coerce")
    df["anio"] = df["periodo"].astype(str).str[:4].astype(int)

    df = df[df["anio"].isin([2020, 2022])]

    agg = (
        df.groupby(GROUP_COLS, as_index=False)
        .agg(punt_global_promedio=("punt_global", "mean"))
    )
    agg["punt_global_promedio"] = agg["punt_global_promedio"].round(2)
    return agg


# ── Parte 2 ──────────────────────────────────────────────────────────────────

def load_saber2023() -> tuple[pd.DataFrame, dict]:
    """Devuelve el DataFrame de atributos y un dict DANE12_EST → feature completo."""
    with SABER23_PATH.open(encoding="utf-8") as f:
        gj = json.load(f)

    rows, features_by_dane = [], {}
    for feat in gj["features"]:
        p = feat["properties"]
        dane = p["DANE12_EST"]
        rows.append({
            "DANE12_EST":  dane,
            "NOMBRE_EST":  p.get("NOMBRE_EST"),
            "ORDEN_DE_S":  p.get("ORDEN_DE_S"),
            "puntaje_2023": p.get("P_Puntaje_"),
        })
        # Si hay duplicados por establecimiento, conserva el de ORDEN_DE_S menor
        if dane not in features_by_dane:
            features_by_dane[dane] = feat
        else:
            existing_orden = features_by_dane[dane]["properties"].get("ORDEN_DE_S") or ""
            if (p.get("ORDEN_DE_S") or "") < existing_orden:
                features_by_dane[dane] = feat

    df = pd.DataFrame(rows)

    # Si hay duplicados: promedia puntaje, toma ORDEN_DE_S menor
    df_agg = (
        df.sort_values("ORDEN_DE_S")
        .groupby("DANE12_EST", as_index=False)
        .agg(
            NOMBRE_EST=("NOMBRE_EST", "first"),
            ORDEN_DE_S=("ORDEN_DE_S", "first"),
            puntaje_2023=("puntaje_2023", "mean"),
        )
    )
    df_agg["puntaje_2023"] = df_agg["puntaje_2023"].round(2)
    return df_agg, features_by_dane


def pivot_csv(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Obtiene cole_caracter por escuela y pivota punt_global por año."""
    # cole_caracter: valor más frecuente por establecimiento
    caracter = (
        df_agg.groupby("cole_cod_dane_establecimiento")["cole_caracter"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    # Pivot: una columna punt_global_YYYY por año
    pivot = (
        df_agg.pivot_table(
            index="cole_cod_dane_establecimiento",
            columns="anio",
            values="punt_global_promedio",
            aggfunc="mean",
        )
        .round(2)
        .reset_index()
    )
    pivot.columns = (
        ["cole_cod_dane_establecimiento"]
        + [f"punt_global_{int(y)}" for y in pivot.columns[1:]]
    )

    return caracter.merge(pivot, on="cole_cod_dane_establecimiento")


def build_geojson(df_2023: pd.DataFrame, features_by_dane: dict, df_csv: pd.DataFrame) -> dict:
    merged = df_2023.merge(
        df_csv,
        left_on="DANE12_EST",
        right_on="cole_cod_dane_establecimiento",
        how="left",
    ).drop(columns=["cole_cod_dane_establecimiento"])

    feat_list = []
    for _, row in merged.iterrows():
        dane = row["DANE12_EST"]
        geometry = features_by_dane[dane]["geometry"] if dane in features_by_dane else None
        properties = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        feat_list.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": properties,
        })

    return {"type": "FeatureCollection", "features": feat_list}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Parte 1 – Agregando CSV Saber 11 2020-2022...")
    df_agg = aggregate_csv()
    df_csv = pivot_csv(df_agg)
    print(f"  Establecimientos en CSV: {len(df_csv):,}")

    print("Parte 2 – Cargando GeoJSON Saber 2023...")
    df_2023, features_by_dane = load_saber2023()
    print(f"  Establecimientos en 2023: {len(df_2023):,}")

    print("Uniendo y generando GeoJSON...")
    geojson = build_geojson(df_2023, features_by_dane, df_csv)
    OUT_PATH.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Guardado en: {OUT_PATH}")
    print(f"  Total features: {len(geojson['features']):,}")


if __name__ == "__main__":
    main()
