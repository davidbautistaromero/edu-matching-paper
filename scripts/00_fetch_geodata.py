"""
Descarga los GeoJSON de Bogotá (Datos Abiertos Bogotá) y los guarda en data/raw/.

Fuentes:
  - demandacupos04_2024.geojson  : demanda de cupos por colegio y nivel educativo
  - matriculatotal04_2024.geojson: matrícula total con desagregación por discapacidad y etnia
  - pruebassaber2023.geojson     : resultados Saber 11° 2023 por colegio (puntajes y categorías)
  - parques_bogota.geojson       : ubicación de parques en Bogotá (formato ESRI JSON, requiere 00_clean_parques.py)
"""

import io
import zipfile

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
    "parques_bogota.geojson": (
        "https://datosabiertos.bogota.gov.co/dataset/1ca41514-3671-41d6-8c3b-a970dc8c24a7"
        "/resource/16288e7f-0345-4680-84aa-40e987706ea8/download/parque.json"
    ),
    "manzana_estratificacion.geojson": (
        "https://datosabiertos.bogota.gov.co/dataset/55467552-0af4-4524-a390-a2956035744e"
        "/resource/29f2d770-bd5d-4450-9e95-8737167ba12f/download/manzanaestratificacion.json"
    ),
    "localidades_bogota.geojson": (
        "https://datosabiertos.bogota.gov.co/dataset/856cb657-8ca3-4ee8-857f-37211173b1f8"
        "/resource/497b8756-0927-4aee-8da9-ca4e32ca3a8a/download/loca.json"
    ),
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"

SOURCES_ZIP = {
    "upz": (
        "https://datosabiertos.bogota.gov.co/dataset/30105648-f24f-41b3-8e88-50004c3bf972/resource/50a34bd9-ad86-405d-b0a8-8f98feb8ce3f/download/pensionadosupz_042023.zip",
        OUTPUT_DIR / "upz"
    ),
}


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

        for name, (url, target_dir) in SOURCES_ZIP.items():
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Descargando {name}.zip ...")
            response = session.get(url, timeout=120)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(target_dir)
            # Aplanar subcarpetas y renombrar todos los archivos a upz.*
            all_files = [f for f in target_dir.rglob("*") if f.is_file()]
            for f in all_files:
                new_name = target_dir / f"upz{f.suffix}"
                f.rename(new_name)
            # Eliminar subdirectorios vacíos que hayan quedado
            for d in sorted(target_dir.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
            print(f"  Extraído en: {target_dir} (archivos renombrados a upz.*)")


if __name__ == "__main__":
    main()
