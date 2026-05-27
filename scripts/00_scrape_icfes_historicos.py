"""
Scraper: históricos ICFES Saber 11 por colegio y año — Bogotá
Fuente: https://www.gip.com.co/gipdata/icfes/ranking_icfes_paginas/icfes.departamento/BOGOTA

Produce: data/raw/icfes_historicos_bogota.csv
Columnas: codigo_icfes, nombre_institucion, municipio, año, eval,
          lec, mat, soc, cie, ing, d_st, prom, global, pos
"""

import csv
import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = "https://www.gip.com.co"
LISTING_BASE = f"{BASE_URL}/gipdata/icfes/ranking_icfes_paginas/icfes.departamento/BOGOTA"
DETAIL_BASE = f"{BASE_URL}/gipdata/icfes/resultados_institucion/codigo_icfes"
OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "icfes_historicos_bogota.csv"
DELAY = 0.6        # segundos entre requests
TIMEOUT = 20
PAGE_SIZE = 40     # registros por página en el ranking

HEADERS = {"User-Agent": "academic-research-scraper/1.0 (ds.bautista@platzi.com)"}

FIELDNAMES = [
    "codigo_icfes", "nombre_institucion", "municipio",
    "año", "eval", "lec", "mat", "soc", "cie", "ing",
    "d_st", "prom", "global", "pos",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def get_soup(session: requests.Session, url: str, params=None) -> BeautifulSoup:
    resp = session.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


# ---------------------------------------------------------------------------
# Paso 1: recopilar colegios del ranking (todas las páginas)
# ---------------------------------------------------------------------------
def _parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    schools = []
    for row in soup.select("table tbody tr"):
        link = row.find("a", href=True)
        if not link or "resultados_institucion" not in link["href"]:
            continue
        href = link["href"]
        parsed = urlparse(href)
        path_parts = parsed.path.rstrip("/").split("/")
        try:
            idx = path_parts.index("codigo_icfes")
            codigo = path_parts[idx + 1]
        except (ValueError, IndexError):
            continue

        # Nombre y municipio viven en los query params ins= y mun= del href
        qs = parse_qs(parsed.query)
        nombre = qs.get("ins", [""])[0].replace("+", " ").strip()
        municipio = qs.get("mun", [""])[0].replace("+", " ").strip()

        # Fallback: buscar en celdas si los params no están
        if not nombre or not municipio:
            cells = row.find_all("td")
            if not nombre:
                nombre = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            if not municipio:
                municipio = cells[2].get_text(strip=True) if len(cells) > 2 else ""

        schools.append({
            "codigo_icfes": codigo,
            "nombre_institucion": nombre,
            "municipio": municipio,
            "detail_url": href if href.startswith("http") else urljoin(BASE_URL, href),
        })
    return schools


def _detect_total_pages(soup: BeautifulSoup) -> int:
    import re
    for tag in soup.find_all(string=True):
        text = tag.strip()
        # Busca patrón "38 de 38" (último número = total de páginas)
        m = re.search(r"\b(\d+)\s+de\s+\1\b", text)
        if m:
            return int(m.group(1))
    # Fallback desde total de registros
    for tag in soup.find_all(string=True):
        m = re.search(r"de\s+(1[0-9]{3})", tag)
        if m:
            total_records = int(m.group(1))
            return -(-total_records // PAGE_SIZE)
    return 38  # valor conocido para Bogotá


def collect_schools(session: requests.Session) -> list[dict]:
    all_schools: list[dict] = []
    seen: set[str] = set()

    log.info("Recopilando lista de colegios de Bogotá…")

    # Paginación via GET: limit_start=<página 1..N>, limit_count=40
    BASE_PARAMS = {"orderby": "cie", "ordertype": "DESC", "limit_count": PAGE_SIZE}

    soup = get_soup(session, LISTING_BASE, params={**BASE_PARAMS, "limit_start": 1})
    total_pages = _detect_total_pages(soup)
    log.info(f"Total páginas detectadas: {total_pages}")

    def add_schools(schools):
        for s in schools:
            if s["codigo_icfes"] not in seen:
                all_schools.append(s)
                seen.add(s["codigo_icfes"])

    add_schools(_parse_listing_page(soup))
    time.sleep(DELAY)

    for page in range(2, total_pages + 1):
        log.info(f"  Página {page}/{total_pages} ({len(all_schools)} colegios hasta ahora)")
        try:
            soup = get_soup(session, LISTING_BASE, params={**BASE_PARAMS, "limit_start": page})
            schools = _parse_listing_page(soup)
            if not schools:
                log.warning(f"  Sin resultados en página {page}, deteniendo.")
                break
            add_schools(schools)
        except requests.RequestException as e:
            log.warning(f"  Error en página {page}: {e}")
        time.sleep(DELAY)

    log.info(f"Total colegios únicos: {len(all_schools)}")
    return all_schools


# ---------------------------------------------------------------------------
# Paso 2: extraer histórico anual de cada colegio
# ---------------------------------------------------------------------------
def _safe_float(val: str):
    try:
        return float(val.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def _safe_int(val: str):
    try:
        return int(val)
    except (ValueError, AttributeError):
        return None


def parse_school_history(soup: BeautifulSoup, codigo: str, nombre: str, municipio: str) -> list[dict]:
    """
    Estructura esperada de la tabla de detalle (12 columnas):
    Codigo | Periodo | Eval | Lec | Mat | Soc | Cie | Ing | D.St | Prom | Global | Pos
       0       1        2     3     4     5     6     7     8      9      10      11
    """
    rows = []
    for tr in soup.select("table tbody tr"):
        cells = tr.find_all("td")
        texts = [c.get_text(strip=True) for c in cells]
        if len(texts) < 4:
            continue

        # Localizar columna de año (número entre 2000–2030)
        año = None
        año_idx = -1
        for i, t in enumerate(texts):
            if t.isdigit() and 2000 <= int(t) <= 2030:
                año = int(t)
                año_idx = i
                break
        if año is None:
            continue

        # Los valores numéricos vienen después del año
        vals = texts[año_idx + 1:]

        row = {
            "codigo_icfes": codigo,
            "nombre_institucion": nombre,
            "municipio": municipio,
            "año": año,
            "eval":   _safe_int(vals[0])   if len(vals) > 0 else None,
            "lec":    _safe_float(vals[1]) if len(vals) > 1 else None,
            "mat":    _safe_float(vals[2]) if len(vals) > 2 else None,
            "soc":    _safe_float(vals[3]) if len(vals) > 3 else None,
            "cie":    _safe_float(vals[4]) if len(vals) > 4 else None,
            "ing":    _safe_float(vals[5]) if len(vals) > 5 else None,
            "d_st":   _safe_float(vals[6]) if len(vals) > 6 else None,
            "prom":   _safe_float(vals[7]) if len(vals) > 7 else None,
            "global": _safe_float(vals[8]) if len(vals) > 8 else None,
            "pos":    _safe_int(vals[9])   if len(vals) > 9 else None,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Paso 1: lista de colegios
    schools = collect_schools(session)

    # Reanudar si el CSV ya existe parcialmente
    already_done: set[str] = set()
    if OUTPUT.exists():
        with open(OUTPUT, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add(row["codigo_icfes"])
        log.info(f"Reanudando: {len(already_done)} colegios ya procesados.")

    total = len(schools)
    pending = [s for s in schools if s["codigo_icfes"] not in already_done]
    log.info(f"Colegios por procesar: {len(pending)} de {total}")

    mode = "a" if OUTPUT.exists() else "w"
    with open(OUTPUT, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for i, school in enumerate(pending, 1):
            codigo = school["codigo_icfes"]
            nombre = school["nombre_institucion"]
            municipio = school["municipio"]
            url = f"{DETAIL_BASE}/{codigo}"
            log.info(f"[{i}/{len(pending)}] {codigo} — {nombre}")
            try:
                soup = get_soup(session, url)
                rows = parse_school_history(soup, codigo, nombre, municipio)
                if rows:
                    writer.writerows(rows)
                    f.flush()
                    log.info(f"  → {len(rows)} registros anuales")
                else:
                    log.warning(f"  Sin datos históricos para {codigo}")
            except requests.RequestException as e:
                log.warning(f"  Error HTTP para {codigo}: {e}")
            except Exception as e:
                log.warning(f"  Error inesperado para {codigo}: {e}")
            time.sleep(DELAY)

    log.info(f"\nListo. CSV guardado en: {OUTPUT}")


if __name__ == "__main__":
    main()
