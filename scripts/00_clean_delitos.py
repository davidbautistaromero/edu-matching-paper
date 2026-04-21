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

    # El GeoJSON tiene una fila por localidad con columnas CM**TOTAL que son
    # totales de Bogota entera (no por localidad). Los valores correctos por
    # localidad estan en las columnas CM**[18-26]CONT (periodos anuales).
    # Prefijos por tipo de delito:
    #   CMH    = homicidios
    #   CMLP   = lesiones personales
    #   CMHP   = hurto a personas
    #   CMHR   = hurto a residencias
    #   CMHA   = hurto de automotores
    #   CMHB   = hurto de bicicletas
    #   CMHC   = hurto a comercio
    #   CMHCE  = hurto a entidades
    #   CMVI   = violencia intrafamiliar
    #   CMDS   = delitos sexuales

    PREFIXES = {
        "homicidios":          "CMH",
        "lesiones_personales": "CMLP",
        "hurto_personas":      "CMHP",
        "hurto_residencias":   "CMHR",
        "hurto_automotores":   "CMHA",
        "hurto_bicicletas":    "CMHB",
        "hurto_comercio":      "CMHC",
        "hurto_entidades":     "CMHCE",
        "violencia_intrafam":  "CMVI",
        "delitos_sexuales":    "CMDS",
    }

    # Normalizar nombre de localidad
    df["localidad_norm"] = df["CMNOMLOCAL"].apply(normalize)
    df["localidad_norm"] = df["localidad_norm"].replace(NOMBRE_MAP)
    df = df[df["localidad_norm"] != "SIN LOCALIZACION"].copy()

    # Para cada tipo de delito, sumar todas las columnas de periodo (CM**[0-9]+CONT)
    import re
    result_rows = []
    for loc, grp in df.groupby("localidad_norm"):
        row = {"localidad_norm": loc}
        for out_col, prefix in PREFIXES.items():
            # Columnas del tipo CM<PREFIX>[YY]CONT (exactamente ese prefijo)
            pat = re.compile(rf"^{re.escape(prefix)}\d+CONT?$", re.IGNORECASE)
            cols = [c for c in grp.columns if pat.match(c)]
            if cols:
                vals = pd.to_numeric(grp[cols].iloc[0], errors="coerce")
                row[out_col] = vals.sum()
            else:
                row[out_col] = None
        result_rows.append(row)

    result = pd.DataFrame(result_rows).sort_values("localidad_norm").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8")

    print(f"Guardado: {OUT}")
    print(f"  {len(result)} localidades | columnas: {list(result.columns)}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
