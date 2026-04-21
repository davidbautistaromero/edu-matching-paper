"""
Descarga los GeoJSON de colegios de Bogotá (Datos Abiertos Bogotá, abril 2024)
y los guarda en data/raw/ como archivos .geojson.

Fuentes:
  - demandacupos04_2024.geojson  : demanda de cupos por colegio y nivel educativo
  - matriculatotal04_2024.geojson: matrícula total con desagregación por discapacidad y etnia
  - pruebassaber2023.geojson     : resultados Saber 11° 2023 por colegio (puntajes y categorías)
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path

SOURCES = {
    "demandacupos04_2024.geojson": (
        "https://datosabiertos.bogota.gov.co/dataset/dffe1a92-12db-4571-8808-96a8e4964815"
        "/resource/e889a95b-22a4-42bb-8c61-e9b5b36d13f6/download/demandacupos04_2024.geojson"
    ),
    "matriculatotal04_2024.geojson": (
        "https://datosabiertos.bogota.gov.co/dataset/02e88fbc-5081-443f-ac5e-4071191c8703"
        "/resource/5146c1e2-caf6-4caa-a17b-f36dbbf5655d/download/matriculatotal04_2024.geojson"
    ),
    "pruebassaber2023.geojson": (
        "https://datosabiertos.bogota.gov.co/dataset/9f0b67b8-f38a-4824-9458-fff7152e13ea"
        "/resource/52ed105f-e99c-43c2-a08c-7f6cb7a6ed4d/download/pruebassaber2023.geojson"
    ),
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def make_session() -> requests.Session:
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with make_session() as session:
        for filename, url in SOURCES.items():
            output_path = OUTPUT_DIR / filename
            print(f"Descargando {filename} ...")
            response = session.get(url, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            print(f"  Guardado en: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
