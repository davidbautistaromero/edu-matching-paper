"""
00_fetch_poblacion_upz.py
=========================
Limpia el Excel de proyecciones de población por UPZ (DANE/SDP, 2018-2024)
y genera una tabla procesada con la población menor de 18 años por UPZ para 2024.

Fuente:
  data/raw/poblacion-localidad-upz-bogota-2018-2024.xlsx
  Hoja: 'UPZ Bogota 2018_2024'

Estructura del Excel:
  - Filas 0-6 : encabezados / texto libre (se saltan)
  - Fila 7    : nombres de columnas
  - Fila 8+   : datos por UPZ y año (2018-2024)

Columnas de interés:
  AREA GEOGRÁFICA  : nombre de la UPZ
  AÑO              : año de proyección
  UPZ              : código UPZ (3 dígitos)
  LOC              : código localidad (2 dígitos)
  Hombres_0-4  ... Hombres_15-19  : hombres por quinquenio
  Mujeres_0-4  ... Mujeres_15-19  : mujeres por quinquenio

Nota metodológica:
  El quinquenio 15-19 incluye personas de 15 a 19 años. Se incluye como
  proxy de menores de 18 ya que el dato desagregado por año simple no está
  disponible. Se documenta esta decisión en el paper.

Output:
  data/processed/poblacion_upz_2024.parquet
  Columnas: upz_codigo, upz_nombre, loc_codigo, pob_menor18
"""

from pathlib import Path
import pandas as pd

# ── rutas ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
INPUT_FILE = ROOT / "data" / "raw" / "poblacion-localidad-upz-bogota-2018-2024.xlsx"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "poblacion_upz_2024.parquet"

SHEET      = "UPZ Bogota 2018_2024"
SKIP_ROWS  = 7   # filas 0-6 vacías/título; fila 7 = headers

# Quinquenios que cubren < 18 años (incluyendo 15-19 como proxy)
QUINQUENIOS = ["0-4", "5-9", "10-14", "15-19"]
COLS_MENOR18 = (
    [f"Hombres_{q}" for q in QUINQUENIOS] +
    [f"Mujeres_{q}" for q in QUINQUENIOS]
)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo: {INPUT_FILE.name}  |  hoja: {SHEET}")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET, skiprows=SKIP_ROWS, engine="openpyxl")

    print(f"  Filas leídas:   {len(df):,}")
    print(f"  Columnas:       {df.shape[1]}")

    # Normalizar nombres de columnas (quitar espacios extra)
    df.columns = df.columns.str.strip()

    # Filtrar 2024
    df = df[df["AÑO"] == 2024].copy()
    print(f"  Filas 2024:     {len(df):,}")

    # Las columnas Total_* contienen fórmulas Excel como strings — calculamos desde crudas
    # Convertir a numérico (por si alguna celda quedó como string)
    for col in COLS_MENOR18:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Suma de menores de 18
    df["pob_menor18"] = df[COLS_MENOR18].sum(axis=1)

    # Seleccionar y renombrar columnas de salida
    result = df[["AREA GEOGRÁFICA", "AÑO", "UPZ", "LOC", "pob_menor18"]].copy()
    result.columns = ["upz_nombre", "anio", "upz_codigo", "loc_codigo", "pob_menor18"]

    # Limpiar: asegurar tipos correctos
    result["upz_codigo"] = result["upz_codigo"].astype(float).astype(int).astype(str).str.zfill(3)
    result["loc_codigo"] = result["loc_codigo"].astype(float).astype(int).astype(str).str.zfill(2)
    result = result.drop(columns="anio")
    result = result.reset_index(drop=True)

    print(f"\n  UPZs procesadas: {len(result)}")
    print(f"  pob_menor18 total Bogotá: {result['pob_menor18'].sum():,.0f}")
    print(f"\n  Muestra:")
    print(result.head(5).to_string(index=False))

    result.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nGuardado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
