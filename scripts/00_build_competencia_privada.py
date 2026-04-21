"""
Construye el indice de competencia del sector privado por localidad.

Calcula el porcentaje de sedes no oficiales sobre el total de sedes activas
por localidad. Esta variable captura la intensidad competitiva del mercado
educativo local: en localidades con alta oferta privada, los hogares tienen
mas alternativas y pueden ser mas sensibles a senales no academicas.

Unidad de analisis: localidad (se une al dataset maestro por nombre_localidad).

Input:  data/raw/colegios_dataset.csv
Output: data/processed/competencia_privada_localidad.csv

Columnas output:
  localidad_norm     : nombre de la localidad normalizado (sin tildes, mayusculas)
  n_sedes_total      : total de sedes activas en la localidad
  n_sedes_no_oficial : sedes del sector No Oficial
  n_sedes_oficial    : sedes del sector Oficial
  pct_no_oficial     : porcentaje de sedes no oficiales (0-100)
"""

import unicodedata
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw" / "colegios_dataset.csv"
OUT = Path(__file__).parent.parent / "data" / "processed" / "competencia_privada_localidad.csv"


def normalize(s: str) -> str:
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    # Cargar directorio completo (oficiales + no oficiales)
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(RAW, encoding=enc, low_memory=False)
            break
        except Exception:
            continue

    # Normalizar nombres de columnas
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("\u00e1", "a").str.replace("\u00e9", "e")
        .str.replace("\u00ed", "i").str.replace("\u00f3", "o")
        .str.replace("\u00fa", "u").str.replace("\u00f1", "n")
        .str.replace("\u00e3", "a").str.replace("\u00e2", "a")
    )

    # Columnas clave post-normalizacion
    loc_col    = next(c for c in df.columns if "localidad" in c)
    sector_col = next(c for c in df.columns if c == "sector")
    estado_col = next((c for c in df.columns if "estado" in c), None)

    print(f"N sedes raw: {len(df)}")
    print(f"Columna localidad: {loc_col}")
    print(f"Valores sector: {df[sector_col].value_counts().to_dict()}")

    # Filtrar solo sedes activas (valores: "Antiguo Activo", "Nuevo Activo")
    if estado_col:
        df = df[df[estado_col].str.strip().str.upper().str.contains("ACTIVO")]
        print(f"N sedes activas: {len(df)}")

    df["localidad_norm"] = df[loc_col].apply(normalize)
    df["es_no_oficial"]  = (df[sector_col].str.strip() == "No Oficial").astype(int)

    agg = (
        df.groupby("localidad_norm")
        .agg(
            n_sedes_total     =("es_no_oficial", "count"),
            n_sedes_no_oficial=("es_no_oficial", "sum"),
        )
        .reset_index()
    )
    agg["n_sedes_oficial"] = agg["n_sedes_total"] - agg["n_sedes_no_oficial"]
    agg["pct_no_oficial"]  = (agg["n_sedes_no_oficial"] / agg["n_sedes_total"] * 100).round(2)
    agg = agg.sort_values("pct_no_oficial", ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT, index=False, encoding="utf-8")

    print(f"\nGuardado: {OUT}")
    print(f"Localidades: {len(agg)}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
