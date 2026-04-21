"""
Genera versiones limpias de los GeoJSON de demanda y matrícula,
conservando únicamente los campos relevantes para el análisis.

Inputs  (data/raw/):
  - demandacupos04_2024.geojson
  - matriculatotal04_2024.geojson

Outputs (data/processed/):
  - demanda_clean.geojson
  - matriculas_clean.geojson
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "processed"

DEMANDA_FIELDS   = {"DANE12_EST", "NOMBRE_EST", "DTotal"}
MATRICULA_FIELDS = {"DANE12_EST", "NOMBRE_EST", "NOMBRE_SED", "ORDEN_DE_S", "TMATRIC_GE"}


def filter_geojson(input_path: Path, keep_fields: set, output_path: Path) -> None:
    with input_path.open(encoding="utf-8") as f:
        gj = json.load(f)

    clean_features = []
    for feature in gj["features"]:
        props = {k: v for k, v in feature["properties"].items() if k in keep_fields}
        clean_features.append({
            "type": "Feature",
            "geometry": feature["geometry"],
            "properties": props,
        })

    output = {
        "type": "FeatureCollection",
        "crs": gj.get("crs"),
        "features": clean_features,
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {output_path.name}: {len(clean_features):,} features, campos: {sorted(keep_fields)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Procesando demanda de cupos...")
    filter_geojson(
        RAW_DIR / "demandacupos04_2024.geojson",
        DEMANDA_FIELDS,
        OUT_DIR / "demanda_clean.geojson",
    )

    print("Procesando matrícula total...")
    filter_geojson(
        RAW_DIR / "matriculatotal04_2024.geojson",
        MATRICULA_FIELDS,
        OUT_DIR / "matriculas_clean.geojson",
    )


if __name__ == "__main__":
    main()
