"""
Geocodifica los colegios oficiales urbanos usando Google Geocoding API.

Por cada colegio (sede principal, Orden A) construye la query:
    "{Nombre del establecimiento educativo}, {Localidad}, Bogotá, Colombia"

Resultado: data/raw/colegios_coordenadas_google.csv
  - id_establecimiento  : identificador primario del colegio
  - nombre              : nombre oficial del establecimiento
  - localidad           : nombre de la localidad
  - direccion           : dirección registrada en el dataset
  - lat_orig / lon_orig : coordenadas originales del CSV (con coma decimal → float)
  - lat_google          : latitud devuelta por Google
  - lon_google          : longitud devuelta por Google
  - location_type       : precisión (ROOFTOP, RANGE_INTERPOLATED, …)
  - formatted_address   : dirección formateada por Google
  - status              : OK | ZERO_RESULTS | REQUEST_DENIED | …
"""

import os
import time
import logging

import pandas as pd
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
API_KEY = os.getenv("GSV_API_TOKEN")
DELAY_S = 0.05  # 50 ms entre llamadas → ~20 req/s, bien bajo el límite

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
INPUT_CSV = RAW_DIR / "colegios_dataset.csv"
OUTPUT_CSV = RAW_DIR / "colegios_coordenadas_google.csv"


def _coord_str_to_float(val: str) -> float:
    """Convierte '−74,03570999983405' (coma decimal, locale ES) a float."""
    return float(str(val).replace(",", "."))


def geocodificar(nombre: str, localidad: str) -> dict:
    query = f"{nombre}, {localidad}, Bogotá, Colombia"
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"address": query, "key": API_KEY, "language": "es"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning("Error HTTP para '%s': %s", query, e)
        return {"lat_google": None, "lon_google": None, "location_type": None,
                "formatted_address": None, "status": "REQUEST_ERROR"}

    status = data.get("status", "UNKNOWN")
    if status != "OK" or not data.get("results"):
        return {"lat_google": None, "lon_google": None, "location_type": None,
                "formatted_address": None, "status": status}

    result = data["results"][0]
    loc = result["geometry"]["location"]
    return {
        "lat_google": loc["lat"],
        "lon_google": loc["lng"],
        "location_type": result["geometry"].get("location_type"),
        "formatted_address": result.get("formatted_address"),
        "status": "OK",
    }


def main():
    if not API_KEY:
        raise EnvironmentError("GSV_API_TOKEN no encontrado en .env")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")

    # Colegios oficiales urbanos, sede principal
    mask = (
        (df["Sector"] == "Oficial")
        & (df["Zona"] == "URBANA")
        & (df["Orden de la sede"] == "A")
    )
    colegios = df[mask].copy()
    log.info("Colegios a geocodificar: %d", len(colegios))

    registros = []
    for i, (_, row) in enumerate(colegios.iterrows(), 1):
        nombre = row["Nombre del establecimiento educativo"]
        localidad = row["Nombre de la Localidad"]
        id_est = row["Id del establecimiento educativo"]

        geo = geocodificar(nombre, localidad)

        registros.append({
            "id_establecimiento": id_est,
            "nombre": nombre,
            "localidad": localidad,
            "direccion": row["Dirección"],
            "lat_orig": _coord_str_to_float(row["coord_y"]),
            "lon_orig": _coord_str_to_float(row["coord_x"]),
            **geo,
        })

        if i % 50 == 0 or i == len(colegios):
            ok_count = sum(1 for r in registros if r["status"] == "OK")
            log.info("[%d/%d] OK: %d  sin resultado: %d",
                     i, len(colegios), ok_count, i - ok_count)

        time.sleep(DELAY_S)

    result_df = pd.DataFrame(registros)

    # Diagnóstico rápido de discrepancias
    ok = result_df[result_df["status"] == "OK"].copy()
    ok["delta_lat"] = (ok["lat_google"] - ok["lat_orig"]).abs()
    ok["delta_lon"] = (ok["lon_google"] - ok["lon_orig"]).abs()
    grandes = ok[ok["delta_lat"] > 0.005]  # ~500 m de diferencia
    log.info("Colegios con discrepancia > 500 m en latitud: %d", len(grandes))
    if not grandes.empty:
        log.info("\n%s", grandes[["nombre", "lat_orig", "lat_google", "delta_lat"]].to_string())

    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Guardado: %s (%d filas)", OUTPUT_CSV, len(result_df))


if __name__ == "__main__":
    main()
