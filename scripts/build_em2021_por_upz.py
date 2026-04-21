"""
Une la encuesta principal y las variables adicionales de la EM2021,
y agrega por UPZ de residencia usando el factor de expansión muestral.

Inputs (data/raw/):
  em2021_encuesta_principal.csv     → hogares con UPZ y factor de expansión
  em2021_variables_adicionales.csv  → índices de pobreza, déficit, ingreso

Join: DIRECTORIO (encuesta principal) ↔ directorio_hog (variables adicionales)
      Left join desde encuesta principal — conserva todos los hogares con UPZ

Agregación por COD_UPZ_GRUPO (ponderada por FEX_C):
  Proporciones  → tasa_pobreza_monetaria, tasa_pobreza_extrema, tasa_ipm,
                  tasa_deficit_cuantitativo, tasa_deficit_cualitativo,
                  tasa_deficit_habitacional, pct_estrato_1 … pct_estrato_6
  Medias        → ingreso_percapita_promedio, capacidad_pago_promedio,
                  tamano_hogar_promedio, gasto_educ_promedio
  Auxiliares    → n_hogares_muestra, poblacion_expandida

Output (data/processed/):
  em2021_por_upz.csv   ← una fila por UPZ, lista para cruzar con colegios
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_PATH = PROC_DIR / "em2021_por_upz.csv"

# ── Carga ─────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Cargando encuesta principal...")
    enc = pd.read_csv(
        RAW_DIR / "em2021_encuesta_principal.csv",
        dtype=str,
        encoding="utf-8",
    )
    enc.columns = enc.columns.str.strip().str.upper()
    print(f"  {len(enc):,} hogares | columnas: {list(enc.columns)}")

    print("Cargando variables adicionales...")
    var = pd.read_csv(
        RAW_DIR / "em2021_variables_adicionales.csv",
        dtype=str,
        encoding="utf-8",
    )
    var.columns = var.columns.str.strip()
    # Renombrar llave para el join
    var = var.rename(columns={"directorio_hog": "DIRECTORIO"})
    var["DIRECTORIO"] = var["DIRECTORIO"].astype(str).str.strip()
    print(f"  {len(var):,} registros | columnas: {list(var.columns)}")

    print("Uniendo tablas (left join por DIRECTORIO)...")
    # directorio_hog tiene un dígito extra al final respecto a DIRECTORIO
    # (ej: encuesta="166238", variables="1662381" → hogar 1 de la vivienda 166238)
    # La llave correcta es truncar directorio_hog quitando el último dígito
    enc["DIRECTORIO"] = enc["DIRECTORIO"].astype(str).str.strip()
    var["DIRECTORIO"] = var["DIRECTORIO"].astype(str).str.strip().str[:-1]
    # Si hay varios hogares por vivienda, quedamos con el primero (variables son del hogar, no la vivienda)
    var = var.drop_duplicates(subset="DIRECTORIO", keep="first")
    df = enc.merge(var, on="DIRECTORIO", how="left")
    print(f"  Resultado: {len(df):,} filas | {df['COD_UPZ_GRUPO'].isna().sum():,} sin UPZ")
    return df


# ── Limpieza de tipos ─────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "FEX_C",
    "NVCBP11AA",              # estrato real
    "N_pobre_monetario",
    "N_pobre_extremo",
    "N_pobre_ipm",
    "N_ingpc",
    "N_sin_cp",
    "N_nper",
    "N_gm_educ_hog",
    "N_deficit_cuantitativo",
    "N_deficit_cualitativo",
    "N_deficit_habitacional",
]

def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Factor de expansión: si falta, peso uniforme = 1
    df["FEX_C"] = df["FEX_C"].fillna(1.0)
    return df


# ── Agregación por UPZ ────────────────────────────────────────────────────────

def wavg(group: pd.DataFrame, col: str) -> float:
    """Media ponderada por FEX_C, ignorando NaN en la variable."""
    mask = group[col].notna()
    w = group.loc[mask, "FEX_C"]
    v = group.loc[mask, col]
    return (v * w).sum() / w.sum() if w.sum() > 0 else np.nan


def aggregate_by_upz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["COD_UPZ_GRUPO"])
    df["COD_UPZ_GRUPO"] = df["COD_UPZ_GRUPO"].astype(str).str.strip().str.zfill(3)

    print(f"Agregando {len(df):,} hogares en {df['COD_UPZ_GRUPO'].nunique()} UPZs...")

    rows = []
    for upz, grp in df.groupby("COD_UPZ_GRUPO"):
        row: dict = {"COD_UPZ_GRUPO": upz}

        # Auxiliares
        row["n_hogares_muestra"]  = len(grp)
        row["poblacion_expandida"] = grp["FEX_C"].sum()

        # Medias ponderadas — continuas y binarias (media de binaria = proporción)
        continuas = {
            "tasa_pobreza_monetaria":    "N_pobre_monetario",
            "tasa_pobreza_extrema":      "N_pobre_extremo",
            "tasa_ipm":                  "N_pobre_ipm",
            "ingreso_percapita_promedio": "N_ingpc",
            "capacidad_pago_promedio":   "N_sin_cp",
            "tamano_hogar_promedio":     "N_nper",
            "gasto_educ_promedio":       "N_gm_educ_hog",
            "tasa_deficit_cuantitativo": "N_deficit_cuantitativo",
            "tasa_deficit_cualitativo":  "N_deficit_cualitativo",
            "tasa_deficit_habitacional": "N_deficit_habitacional",
        }
        for out_col, src_col in continuas.items():
            row[out_col] = wavg(grp, src_col) if src_col in grp.columns else np.nan

        # Distribución de estrato (proporción ponderada por estrato)
        if "NVCBP11AA" in grp.columns:
            total_w = grp["FEX_C"].sum()
            for est in range(1, 7):
                mask = grp["NVCBP11AA"] == est
                row[f"pct_estrato_{est}"] = (
                    grp.loc[mask, "FEX_C"].sum() / total_w if total_w > 0 else np.nan
                )

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("COD_UPZ_GRUPO").reset_index(drop=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = cast_numeric(df)
    df_upz = aggregate_by_upz(df)

    df_upz.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nGuardado: {OUT_PATH}")
    print(f"  {len(df_upz)} UPZs | {len(df_upz.columns)} columnas")
    print(f"\nMuestra:")
    print(df_upz.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
