"""
Descarga las dos tablas de la Encuesta Multipropósito 2021 (EM2021)
y las deja en data/raw/ con solo las columnas necesarias para el análisis.

Tabla 1 — Encuesta principal (resource b3fd892e):
  Descarga: CSV completo vía URL directa (no tiene datastore activo)
  Columnas seleccionadas:
    DIRECTORIO          → llave de cruce con variables adicionales
    COD_UPZ_GRUPO       → UPZ de residencia (identificador geográfico)
    COD_LOCALIDAD       → localidad de residencia
    ESTRATO2021         → estrato de muestreo
    NVCBP11AA           → estrato para tarifa (estrato real del hogar)
    FEX_C               → factor de expansión (para promedios ponderados)
    NPCHP4              → nivel educativo más alto del jefe del hogar
    NPCJP9AI            → satisfacción con su educación (0-10)

Tabla 2 — Variables adicionales (resource ba44798f):
  Descarga: vía API Datastore CKAN (tiene datastore activo → solo columnas necesarias)
  Columnas seleccionadas:
    directorio_hog      → llave de cruce con encuesta principal
    N_pobre_monetario   → pobreza monetaria (0/1)
    N_pobre_extremo     → pobreza extrema (0/1)
    N_pobre_ipm         → Índice de Pobreza Multidimensional (0/1)
    N_ingpc             → ingreso per cápita del hogar
    N_sin_cp            → índice de capacidad de pago
    N_nper              → número de personas en el hogar
    N_gm_educ_hog       → gasto mensual en educación
    N_deficit_cuantitativo  → déficit cuantitativo de vivienda (0/1)
    N_deficit_cualitativo   → déficit cualitativo de vivienda (0/1)
    N_deficit_habitacional  → déficit habitacional total (0/1)

Tabla 3 — Personas en edad escolar (nivel persona, mismo CSV que Tabla 1):
  Descarga: mismo CSV completo vía URL directa (streaming)
  Columnas seleccionadas (demografía DBF_MTP_258_1 + educación DBF_MTP_258_4):
    DIRECTORIO  → llave de vivienda / cruce con encuesta principal
    ORDEN       → orden de la persona dentro del hogar
    NPCEP4      → edad en años (demografía)
    NPCHP2      → ¿estudia actualmente? 1=Sí 2=No
    NPCHP6      → nivel en el que está matriculado
    NPCHP12     → tipo de institución: 1=oficial, 2=no oficial
  Filtros:
    NPCHP2 == '1'   (estudia actualmente)
    NPCHP12 == '1'  (institución oficial)
    NPCEP4 numérico entre 5 y 17 inclusive
  Cruce con Tabla 1 para obtener COD_UPZ_GRUPO, COD_LOCALIDAD, ESTRATO2021, FEX_C
  Agregación a nivel hogar (DIRECTORIO):
    DIRECTORIO                  → identificador del hogar
    COD_UPZ_GRUPO               → UPZ (primer valor del hogar)
    COD_LOCALIDAD               → localidad (primer valor del hogar)
    ESTRATO2021                 → estrato (primer valor del hogar)
    FEX_C                       → factor de expansión (primer valor del hogar)
    n_hijos_oficial             → conteo de hijos en institución oficial

Outputs:
  data/raw/em2021_encuesta_principal.csv
  data/raw/em2021_variables_adicionales.csv
  data/raw/em2021_familias_escolar.csv

Fuente: https://datosabiertos.bogota.gov.co/dataset/encuesta-multiproposito-2021-sdp
"""

import io
import requests
import pandas as pd
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Tabla 1 — encuesta principal (sin datastore → descarga CSV completo)
ENCUESTA_URL = (
    "https://datosabiertos.bogota.gov.co/dataset/"
    "8ac12a95-1415-4812-b343-f07f90608014/resource/"
    "b3fd892e-b9f9-4f34-ac22-6cc50612eac9/download/"
    "20240430_em_2021.csv"
)
ENCUESTA_COLS = [
    "DIRECTORIO",     # llave de cruce
    "COD_UPZ_GRUPO",  # UPZ de residencia
    "COD_LOCALIDAD",  # localidad de residencia
    "ESTRATO2021",    # estrato de muestreo
    "NVCBP11AA",      # estrato para tarifa (real)
    "FEX_C",          # factor de expansión
    "NPCHP4",         # nivel educativo jefe del hogar
    "NPCJP9AI",       # satisfacción con su educación (0-10)
]
ENCUESTA_OUT = RAW_DIR / "em2021_encuesta_principal.csv"

# Tabla 2 — variables adicionales (con datastore → solo columnas necesarias)
VARS_RESOURCE_ID = "ba44798f-5257-4c5c-8a57-0966da1bb4fa"
DATASTORE_URL = "https://datosabiertos.bogota.gov.co/api/3/action/datastore_search"
VARS_COLS = [
    "directorio_hog",          # llave de cruce
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
VARS_OUT = RAW_DIR / "em2021_variables_adicionales.csv"

PAGE_SIZE = 10_000

# Tabla 3 — personas en edad escolar (mismo CSV que Tabla 1)
PERSONAS_COLS = [
    "DIRECTORIO",  # llave de vivienda
    "ORDEN",       # orden de la persona en el hogar
    "NPCEP4",      # edad en años (demografía)
    "NPCHP2",      # ¿estudia actualmente? 1=Sí 2=No
    "NPCHP6",      # nivel matriculado
    "NPCHP12",     # tipo institución: 1=oficial, 2=no oficial
]
FAMILIAS_OUT = RAW_DIR / "em2021_familias_escolar.csv"


# ── Tabla 1: encuesta principal ───────────────────────────────────────────────

def fetch_encuesta_principal() -> None:
    """
    Descarga el CSV en streaming y filtra columnas en memoria por chunks.
    El archivo completo (~1.2GB) nunca se escribe a disco — solo el resultado filtrado.
    """
    print("Descargando encuesta principal (streaming + filtro en memoria)...")
    resp = requests.get(ENCUESTA_URL, timeout=300, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    if total:
        print(f"  Tamaño total: {total/1e6:.1f} MB (solo se guardan las columnas necesarias)")

    # Stream → buffer en memoria → leer con pandas en chunks
    # Acumulamos el stream completo pero filtramos de inmediato por chunk
    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"  Descargados {downloaded/1e6:.1f} / {total/1e6:.1f} MB...", end="\r")
    print(f"\n  Descarga completa: {downloaded/1e6:.1f} MB")

    buf.seek(0)
    # Separador: coma (confirmado inspeccionando el header)
    sep = ","

    chunks_df = []
    cols_to_keep = None
    missing = []
    total_rows = 0

    for chunk in pd.read_csv(
        buf,
        sep=sep,
        encoding="latin-1",
        dtype=str,
        low_memory=False,
        chunksize=50_000,
    ):
        if cols_to_keep is None:
            print(f"  Columnas en el archivo: {len(chunk.columns)}")
            col_map = {c.strip('"').upper(): c for c in chunk.columns}
            cols_to_keep = []
            for want in ENCUESTA_COLS:
                match = col_map.get(want.upper())
                if match:
                    cols_to_keep.append(match)
                else:
                    missing.append(want)
            if missing:
                print(f"  ⚠ Columnas no encontradas: {missing}")
            if not cols_to_keep:
                print("  ✗ Sin columnas válidas. Verifica los nombres en el diccionario.")
                return

        filtered = chunk[cols_to_keep].copy()
        filtered.columns = [c.strip('"').upper() for c in filtered.columns]
        chunks_df.append(filtered)
        total_rows += len(filtered)
        print(f"  Procesadas {total_rows:,} filas...", end="\r")

    print()
    df_clean = pd.concat(chunks_df, ignore_index=True)
    df_clean.to_csv(ENCUESTA_OUT, index=False, encoding="utf-8")
    size_mb = ENCUESTA_OUT.stat().st_size / 1e6
    print(f"  Guardado: {ENCUESTA_OUT}")
    print(f"  {len(df_clean):,} filas | {len(df_clean.columns)} columnas | {size_mb:.1f} MB")


# ── Tabla 2: variables adicionales ───────────────────────────────────────────

def get_available_fields() -> list[str]:
    resp = requests.get(
        DATASTORE_URL,
        params={"resource_id": VARS_RESOURCE_ID, "limit": 0},
        timeout=30,
    )
    resp.raise_for_status()
    return [f["id"] for f in resp.json()["result"]["fields"]]


def fetch_variables_adicionales() -> None:
    print("\nDescargando variables adicionales (API Datastore)...")

    available = get_available_fields()
    fields_to_fetch = [c for c in VARS_COLS if c in available]
    missing = [c for c in VARS_COLS if c not in available]
    if missing:
        print(f"  ⚠ Columnas no encontradas (se omiten): {missing}")
    print(f"  Columnas a descargar: {fields_to_fetch}")

    all_records = []
    offset = 0
    fields_str = ",".join(fields_to_fetch)

    while True:
        resp = requests.get(
            DATASTORE_URL,
            params={
                "resource_id": VARS_RESOURCE_ID,
                "fields": fields_str,
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        records = data["result"]["records"]
        if not records:
            break

        all_records.extend(records)
        total = data["result"]["total"]
        offset += PAGE_SIZE
        print(f"  {min(offset, total):,} / {total:,} registros...", end="\r")

        if offset >= total:
            break

    print()
    df = pd.DataFrame(all_records).drop(columns=["_id"], errors="ignore")
    df.to_csv(VARS_OUT, index=False, encoding="utf-8")
    size_mb = VARS_OUT.stat().st_size / 1e6
    print(f"  Guardado: {VARS_OUT}  ({len(df):,} filas, {len(df.columns)} columnas, {size_mb:.1f} MB)")


# ── Tabla 3: personas en edad escolar en institución oficial ──────────────────

def fetch_personas_escolar() -> None:
    """
    Reutiliza el mismo CSV de la encuesta principal (streaming) para extraer
    columnas de demografía y educación a nivel persona.  Filtra niños 5-17 años
    que estudian actualmente en institución oficial, cruza con
    em2021_encuesta_principal.csv y agrega a nivel hogar (DIRECTORIO).
    """
    print("\nDescargando familias con hijos en edad escolar (streaming + filtro en memoria)...")
    resp = requests.get(ENCUESTA_URL, timeout=300, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    if total:
        print(f"  Tamaño total: {total/1e6:.1f} MB")

    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            print(f"  Descargados {downloaded/1e6:.1f} / {total/1e6:.1f} MB...", end="\r")
    print(f"\n  Descarga completa: {downloaded/1e6:.1f} MB")

    buf.seek(0)

    chunks_df = []
    cols_to_keep = None
    missing = []
    total_rows = 0

    for chunk in pd.read_csv(
        buf,
        sep=",",
        encoding="latin-1",
        dtype=str,
        low_memory=False,
        chunksize=50_000,
    ):
        if cols_to_keep is None:
            print(f"  Columnas en el archivo: {len(chunk.columns)}")
            col_map = {c.strip('"').upper(): c for c in chunk.columns}
            cols_to_keep = []
            for want in PERSONAS_COLS:
                match = col_map.get(want.upper())
                if match:
                    cols_to_keep.append(match)
                else:
                    missing.append(want)
            if missing:
                print(f"  ⚠ Columnas no encontradas: {missing}")
            if not cols_to_keep:
                print("  ✗ Sin columnas válidas. Verifica los nombres en el diccionario.")
                return

        filtered = chunk[cols_to_keep].copy()
        filtered.columns = [c.strip('"').upper() for c in filtered.columns]
        chunks_df.append(filtered)
        total_rows += len(filtered)
        print(f"  Procesadas {total_rows:,} filas...", end="\r")

    print()
    df = pd.concat(chunks_df, ignore_index=True)

    # ── Filtros ───────────────────────────────────────────────────────────────
    df = df[df["NPCHP2"].str.strip() == "1"]    # estudia actualmente
    df = df[df["NPCHP12"].str.strip() == "1"]   # institución oficial
    df["NPCEP4_num"] = pd.to_numeric(df["NPCEP4"], errors="coerce")
    df = df[df["NPCEP4_num"].between(5, 17)]
    print(f"  Filas tras filtros (5-17 años, estudia, oficial): {len(df):,}")

    # ── Cruce con encuesta principal ──────────────────────────────────────────
    enc = pd.read_csv(
        ENCUESTA_OUT,
        dtype=str,
        usecols=["DIRECTORIO", "COD_UPZ_GRUPO", "COD_LOCALIDAD", "ESTRATO2021", "NVCBP11AA", "FEX_C"],
    )
    df = df.merge(enc, on="DIRECTORIO", how="left")

    fex_missing = df["FEX_C"].isna().sum()
    if fex_missing:
        print(f"  ⚠ {fex_missing:,} filas sin FEX_C tras el cruce")

    df["FEX_C"] = pd.to_numeric(df["FEX_C"], errors="coerce")

    # ── Marcar niños en edad de ingreso (5-6 años) ───────────────────────────
    df["es_ingreso"] = df["NPCEP4_num"].between(5, 6).astype(int)

    # ── Agregación a nivel hogar ──────────────────────────────────────────────
    agg = df.groupby("DIRECTORIO", dropna=False).agg(
        COD_UPZ_GRUPO=("COD_UPZ_GRUPO", "first"),
        COD_LOCALIDAD=("COD_LOCALIDAD", "first"),
        ESTRATO2021=("ESTRATO2021", "first"),
        NVCBP11AA=("NVCBP11AA", "first"),
        FEX_C=("FEX_C", "first"),
        n_hijos_oficial=("DIRECTORIO", "count"),
        n_hijos_ingreso=("es_ingreso", "sum"),
    ).reset_index()

    agg.to_csv(FAMILIAS_OUT, index=False, encoding="utf-8")
    size_mb = FAMILIAS_OUT.stat().st_size / 1e6
    print(f"  Guardado: {FAMILIAS_OUT}")
    print(f"  {len(agg):,} hogares con hijos en institución oficial | {size_mb:.2f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_encuesta_principal()
    fetch_variables_adicionales()
    fetch_personas_escolar()
    print("\n✓ Listo. Archivos en data/raw/")


if __name__ == "__main__":
    main()

