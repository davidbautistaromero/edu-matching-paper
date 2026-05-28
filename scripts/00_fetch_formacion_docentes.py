"""
00_fetch_formacion_docentes.py
==============================
Descarga el GeoJSON de formación de docentes por localidad (SED Bogotá)
y extrae una tabla limpia con el % de docentes con posgrado por localidad.

Fuente:
  Secretaría de Educación del Distrito — Datos Abiertos Bogotá
  "Formación de docentes de clase distrital por localidad. Bogotá D.C."
  Corte: mayo 2025

Output:
  data/raw/formacion_docentes_localidad.csv      (tabla limpia)
"""

import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://datosabiertos.bogota.gov.co/dataset/"
    "a6177062-84ad-42a2-80e2-44c4066e384e/resource/"
    "8bc1a0ba-e340-4cfc-b0cc-0b61fddf7ed0/download/"
    "formacion_de_docentes_por_localidad.geojson"
)

CSV_PATH = RAW_DIR / "formacion_docentes_localidad.csv"


def download():
    print("Descargando formación de docentes por localidad...")
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    r = session.get(URL, timeout=30)
    r.raise_for_status()
    print(f"  Descargado ({len(r.content):,} bytes)")
    return json.loads(r.content)


def extract_table(geojson: dict):
    """Extrae tabla limpia de properties del GeoJSON."""
    rows = []
    for feat in geojson["features"]:
        p = feat["properties"]
        cod_loc = str(p.get("Nombre_de_la_Localidad", "")).strip().zfill(2)

        total    = p.get("Total_Docentes", 0) or 0
        m_posg   = p.get("Mujeres_con_formación_en_Posgrado", 0) or 0
        h_posg   = p.get("Hombres_con_formación_en_Posgrado", 0) or 0
        m_prof   = p.get("Mujeres_con_formación_Profesional", 0) or 0
        h_prof   = p.get("Hombres_con_formación_Profesional", 0) or 0
        m_norm   = p.get("Mujeres_con_formación_Normalista_Superior", 0) or 0
        h_norm   = p.get("Hombres_con_formación_Normalista_Superior", 0) or 0
        m_tec    = p.get("Mujeres_con_formación_Técnico_o_Tecnólogo", 0) or 0
        h_tec    = p.get("Hombres_con_formación_Técnico_o_Tecnólogo", 0) or 0
        m_bach   = p.get("Mujeres_con_formación_Bachiller", 0) or 0
        h_bach   = p.get("Hombres_con_formación_Bachiller", 0) or 0
        m_otra   = p.get("Mujeres_con_otra_formación", 0) or 0
        h_otra   = p.get("Hombres_con_otra_formación", 0) or 0

        posgrado    = int(m_posg + h_posg)
        profesional = int(m_prof + h_prof)
        total       = int(total)

        pct_postgrado = posgrado / total if total > 0 else None

        rows.append({
            "codigo_localidad": cod_loc,
            "total_docentes": total,
            "docentes_postgrado": posgrado,
            "docentes_profesional": profesional,
            "docentes_normalista": int(m_norm + h_norm),
            "docentes_tecnico": int(m_tec + h_tec),
            "docentes_bachiller": int(m_bach + h_bach),
            "docentes_otro": int(m_otra + h_otra),
            "pct_docentes_postgrado": round(pct_postgrado, 4) if pct_postgrado is not None else None,
        })

    # Escribir CSV
    import csv
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Guardado: {CSV_PATH} ({len(rows)} localidades)")
    print(f"\n  {'Localidad':>10}  {'Docentes':>8}  {'Postgrado':>10}  {'%Postgrado':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")
    for r in sorted(rows, key=lambda x: x["codigo_localidad"]):
        pct = f"{r['pct_docentes_postgrado']:.1%}" if r['pct_docentes_postgrado'] is not None else "N/A"
        print(f"  {r['codigo_localidad']:>10}  {r['total_docentes']:>8,}  {r['docentes_postgrado']:>10,}  {pct:>10}")


if __name__ == "__main__":
    gj = download()
    extract_table(gj)
    print("\nDone.")
