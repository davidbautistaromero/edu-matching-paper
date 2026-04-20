"""
Descarga el dataset Resultados únicos Saber 11 (2010-2022) desde la API Socrata de datos.gov.co.
Fuente: https://www.datos.gov.co/resource/kgxf-xxbe/about_data
Total registros: ~7,109,704
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import time
from pathlib import Path

# Resultados únicos Saber 11 (2010-2022), excluye periodo 20204
BASE_URL_MAIN = "https://www.datos.gov.co/resource/kgxf-xxbe.json"
FILTER_MAIN = "cole_cod_depto_ubicacion='11' AND periodo >= '20201'"

# Saber 11° 2020-2 (periodo 20204, ausente en el dataset principal)
BASE_URL_2020 = "https://www.datos.gov.co/resource/rnvb-vnyh.json"
FILTER_2020 = "cole_cod_depto_ubicacion='11'"

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "saber11_bogota.csv"
PAGE_SIZE = 10_000

SELECT = ",".join([
    # Colegio
    "periodo",
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_calendario",
    "cole_caracter",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion",
    "cole_cod_mcpio_ubicacion",
    "cole_codigo_icfes",
    "cole_depto_ubicacion",
    "cole_genero",
    "cole_jornada",
    "cole_mcpio_ubicacion",
    "cole_naturaleza",
    "cole_nombre_establecimiento",
    "cole_nombre_sede",
    "cole_sede_principal",
    # Estudiante
    "estu_consecutivo",
    # Puntajes
    "desemp_ingles",
    "punt_c_naturales",
    "punt_global",
    "punt_ingles",
    "punt_lectura_critica",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
])


def fetch_page(url: str, where: str, offset: int, session: requests.Session) -> list[dict]:
    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": ":id",
        "$where": where,
        "$select": SELECT,
    }
    response = session.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def fetch_total_count(url: str, where: str, session: requests.Session) -> int:
    params = {"$select": "count(*)", "$limit": 1, "$where": where}
    response = session.get(url, params=params, timeout=120)
    response.raise_for_status()
    return int(response.json()[0]["count"])


def fetch_all(url: str, where: str, session: requests.Session) -> list[dict]:
    total = fetch_total_count(url, where, session)
    print(f"  Total registros: {total:,}")
    records = []
    offset = 0
    while offset < total:
        print(f"    Descargando {offset:,} – {min(offset + PAGE_SIZE, total):,} ...")
        page = fetch_page(url, where, offset, session)
        if not page:
            break
        records.extend(page)
        offset += len(page)
        time.sleep(0.3)
    return records


def make_session() -> requests.Session:
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with make_session() as session:
        print("--- Fuente principal (2020-2022) ---")
        records_main = fetch_all(BASE_URL_MAIN, FILTER_MAIN, session)

        print("--- Saber 11° 2020-2 (periodo 20204) ---")
        records_2020 = fetch_all(BASE_URL_2020, FILTER_2020, session)

    all_records = records_main + records_2020
    print(f"\nTotal combinado: {len(all_records):,} registros")
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Guardado en: {OUTPUT_PATH}")
    print(f"Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
