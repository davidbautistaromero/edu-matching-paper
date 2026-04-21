"""
Limpia y agrega el GeoJSON de delitos de alto impacto por localidad.

El archivo raw tiene una fila por localidad x mes, con columnas por tipo de delito.
Este script agrega a nivel de localidad (suma todos los meses disponibles)
y estandariza el nombre de localidad para el cruce con el directorio de colegios.

Ajuste especifico:
  "Candelaria" en delitos -> "La Candelaria" en el directorio de colegios.

Input:  data/raw/delitos_alto_impacto.geojson
Output: data/processed/delitos_por_localidad.csv
"""

import json
import unicodedata
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw" / "delitos_alto_impacto.geojson"
OUT = Path(__file__).parent.parent / "data" / "processed" / "delitos_por_localidad.csv"


# Mapa de nombres para alinear con el directorio de colegios SED
NOMBRE_MAP = {
    "CANDELARIA": "LA CANDELARIA",
}


def normalize(s: str) -> str:
    """Elimina tildes y convierte a mayusculas."""
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    for enc in ("utf-8", "latin-1"):
        try:
            with open(RAW, encoding=enc) as f:
                gj = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    rows = [(feat.get("properties") or feat) for feat in gj["features"]]
    df = pd.DataFrame(rows)
    print(f"Filas raw: {len(df):,} | Columnas: {len(df.columns)}")

    # Columnas de conteo por tipo de delito (patron: CM**TOTAL o CM**CONT)
    delito_cols = {
        "homicidios":          "CMHTOTAL",
        "lesiones_personales": "CMLPTOTAL",
        "hurto_personas":      "CMHPTOTAL",
        "hurto_residencias":   "CMHRTOTAL",
        "hurto_automotores":   "CMHATOTAL",
        "hurto_bicicletas":    "CMHBTOTAL",
        "hurto_comercio":      "CMHCTOTAL",
        "hurto_entidades":     "CMHCETOTAL",
        "violencia_intrafam":  "CMVITOTAL",
        "delitos_sexuales":    "CMDSTOTAL",
    }

    # Normalizar nombre de localidad
    df["localidad_norm"] = df["CMNOMLOCAL"].apply(normalize)
    df["localidad_norm"] = df["localidad_norm"].replace(NOMBRE_MAP)

    # Convertir columnas numericas
    for col in delito_cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Agregar por localidad (suma de todos los meses)
    agg_dict = {
        out_col: (src_col, "sum")
        for out_col, src_col in delito_cols.items()
        if src_col in df.columns
    }
    result = df.groupby("localidad_norm").agg(**agg_dict).reset_index()
    result = result[result["localidad_norm"] != "SIN LOCALIZACION"]
    result = result.sort_values("localidad_norm").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Guardado: {OUT}")
    print(f"  {len(result)} localidades | columnas: {list(result.columns)}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
