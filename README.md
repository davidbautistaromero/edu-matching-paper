# Matching Escolar – Bogotá

Código de reproducción del paper *"Señales Visuales y Mecanismos de Asignación Escolar en Bogotá"*. Estima el índice visual de colegios oficiales a partir de imágenes GSV, lo incorpora en un modelo de utilidad de familias y compara tres mecanismos de asignación (BM, DA, SED-lex) sobre datos reales y una población sintética calibrada.

---

## Setup

```powershell
cd C:\ruta\al\repo
.\setup.ps1
```

`setup.ps1` crea el entorno virtual `.venv`, detecta GPU NVIDIA e instala PyTorch con CUDA 12.4 (o CPU si no hay GPU), instala `requirements.txt` y descarga el checkpoint DeepLabV3+ ResNet-101 (~449 MB) en `checkpoints/`.

> Si la descarga del checkpoint falla, bajarlo manualmente desde:
> https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb
> y guardarlo en `checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth`

Activar el entorno antes de correr cualquier script:

```powershell
.venv\Scripts\activate
```

---

## Reproducir resultados

El pipeline completo sigue el orden numérico de los scripts. Correr cada uno desde la raíz del repositorio:

```
# 1. Descargar y limpiar datos
python scripts/00_fetch_geodata.py
python scripts/00_fetch_saber11_bogota.py
python scripts/00_fetch_em2021_variables.py
python scripts/00_fetch_poblacion_upz.py
python scripts/00_colegios_csv_to_geojson.py
python scripts/00_merge_saber_geojson.py
python scripts/00_demand_capacity_colegios.py
python scripts/00_build_em2021_por_upz.py
python scripts/00_clean_sitp.py
python scripts/00_clean_parques.py
python scripts/00_clean_delitos.py
python scripts/00_build_competencia_privada.py

# 2. Imágenes GSV (requiere API key en gsv_config.py)
python scripts/00_download_gsv_colegios.py

# 3. Dataset maestro de colegios
python scripts/01_build_dataset.py

# 4. Features visuales
python scripts/02a_extract_embeddings.py
python scripts/02b_diagnose_embeddings.py
python scripts/02c_seg_cityscapes.py      # requiere GPU recomendado
python scripts/03a_lda_topics.py
python scripts/03b_nmf_topics.py
python scripts/03c_clip_features.py
python scripts/03d_imputacion_espacial.py

# 5. Regresión y atractivo
python scripts/04a_regresion.py
python scripts/04b_build_capacidad.py
python scripts/04c_decompose_aj.py
python scripts/05c_visual_index_validation.py

# 6. Simulación de familias
python scripts/05a_simular_distancias.py
python scripts/05b_expandir_familias.py
python scripts/06_preferencias.py

# 7. Matching sobre datos reales
python scripts/07_boston_mechanism.py
python scripts/07_da_mechanism.py
python scripts/07_sed_lex.py
python scripts/compare_mechanisms.py

# 8. Experimento sintético
python scripts/08_datos_sinteticos.py
python scripts/09a_matching_sinteticos.py
python scripts/09b_robustez_gamma.py
```

---

## Datos externos

Los siguientes archivos deben descargarse manualmente y colocarse en `data/raw/` antes de correr el pipeline:

| Archivo | Fuente |
|---|---|
| `estaciones_transmilenio.geojson` | [Datos Abiertos Bogotá](https://datosabiertos.bogota.gov.co) – Transmilenio S.A. |
| `paraderos_sitp.geojson` | Datos Abiertos Bogotá – Transmilenio S.A. |
| `parques_bogota.geojson` | Datos Abiertos Bogotá – Secretaría Distrital de Planeación |
| `delitos_alto_impacto.geojson` | Datos Abiertos Bogotá – Secretaría Distrital de Seguridad |
| `localidades_bogota.geojson` | Datos Abiertos Bogotá – IDECA |
| `manzana_estratificacion.geojson` | Datos Abiertos Bogotá – UAESP |
| `poblacion-localidad-upz-bogota-2018-2024.xlsx` | DANE – Proyecciones de población |
| `upz/upz.shp` (+ archivos `.dbf`, `.prj`, `.shx`) | Datos Abiertos Bogotá – IDECA |

Las imágenes GSV y Mapillary no se incluyen en el repositorio. Ver `scripts/gsv_config.py` y `scripts/mapillary_filtros.py` para configurar las descargas.

---

## Estructura del proyecto

```
paper-AI/
├── data/
│   ├── raw/                        <- Datos crudos, tal como vienen de la fuente (no modificar)
│   │   ├── colegios_dataset.csv
│   │   ├── saber11_bogota_2020_2022.csv
│   │   ├── pruebassaber2023.geojson
│   │   ├── demandacupos04_2024.geojson
│   │   ├── matriculatotal04_2024.geojson
│   │   ├── em2021_encuesta_principal.csv
│   │   ├── em2021_variables_adicionales.csv
│   │   ├── em2021_familias_escolar.csv
│   │   ├── estaciones_transmilenio.geojson
│   │   ├── paraderos_sitp.geojson
│   │   ├── parques_bogota.geojson
│   │   ├── delitos_alto_impacto.geojson
│   │   ├── localidades_bogota.geojson
│   │   ├── manzana_estratificacion.geojson
│   │   ├── poblacion-localidad-upz-bogota-2018-2024.xlsx
│   │   └── upz/
│   ├── processed/                  <- Intermedios limpios
│   │   ├── colegios_dataset.geojson
│   │   ├── saber_bogota_merged.geojson
│   │   ├── demanda_clean.geojson
│   │   ├── matriculas_clean.geojson
│   │   ├── em2021_por_upz.csv
│   │   ├── sitp_clean.geojson
│   │   ├── parques_clean.geojson
│   │   ├── delitos_por_localidad.csv
│   │   ├── competencia_privada_localidad.csv
│   │   ├── poblacion_upz_2024.parquet
│   │   ├── familias_ubicadas.parquet
│   │   ├── familias_distancias.parquet
│   │   ├── familias_expandidas.parquet
│   │   ├── distancias_expandidas.parquet
│   │   └── utilidades_familias.parquet
│   ├── primary/                    <- Datos de análisis principal
│   │   ├── colegios_features.geojson
│   │   ├── colegios_features_imputed.geojson
│   │   ├── colegios_capacidad.parquet
│   │   ├── vj_scores.parquet
│   │   ├── preferencias_familias.parquet
│   │   ├── sinteticos_b_colegios.parquet
│   │   ├── sinteticos_b_estudiantes.parquet
│   │   ├── sinteticos_b_preferencias_bias.parquet
│   │   └── sinteticos_b_preferencias_true.parquet
│   ├── results/                    <- Asignaciones finales de cada mecanismo
│   │   ├── matching_bm.parquet
│   │   ├── matching_da.parquet
│   │   ├── matching_sed_lex.parquet
│   │   ├── matching_sed_dist.parquet
│   │   ├── sinteticos_b_resultados.parquet
│   │   ├── sinteticos_bm_bias.parquet
│   │   ├── sinteticos_bm_true.parquet
│   │   ├── sinteticos_da_bias.parquet
│   │   └── sinteticos_da_true.parquet
│   └── images/
│       ├── gsv/                    <- Imágenes GSV (ignoradas en git)
│       │   ├── {id_establecimiento}/{id_sede}_{heading:03d}.jpg
│       │   ├── gsv_catalog.csv
│       │   └── mapa_cobertura_gsv.png
│       ├── mapillary/              <- Catálogo Mapillary (imágenes ignoradas en git)
│       │   ├── mapillary_catalog.csv
│       │   ├── resumen_fechas.csv
│       │   ├── mapa_cobertura_total.png
│       │   └── mapa_cobertura_filtrada.png
│       └── embeddings/
│           ├── gsv_vgg19_raw.parquet
│           ├── gsv_nmf_K{k}.parquet
│           ├── gsv_nmf_K{k}_images.parquet
│           ├── gsv_nmf_K{k}_topics.json
│           ├── gsv_lda_K{k}.parquet
│           └── diagnostico_embeddings.json
├── checkpoints/
│   └── best_deeplabv3plus_resnet101_cityscapes_os16.pth
├── docs/
│   ├── em2021_diccionario.ods
│   └── em2021_variablesadicionales_diccionario.ods
├── models/
│   ├── ridge_m1.joblib
│   ├── ridge_m1_meta.json
│   └── {ols,ridge,lasso,elasticnet}_m{0-4}.joblib + _meta.json
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── comparativa_estimadores.csv
│   ├── comparativa_mecanismos.csv
│   ├── matching_bm_summary.csv
│   ├── matching_da_summary.csv
│   ├── matching_sed_comparison.csv
│   ├── mejores_coefs.csv
│   ├── tabla_ridge_m1.tex
│   ├── sinteticos_b_calibracion.json
│   ├── sinteticos_b_comparativa.csv
│   ├── robustez_gamma.csv
│   └── figures/
│       ├── eda/
│       ├── matching/
│       │   ├── bm_equidad_estrato.png
│       │   ├── bm_distribucion_qj.png
│       │   ├── da_equidad_estrato.png
│       │   ├── da_distribucion_qj.png
│       │   ├── sed_equidad.png
│       │   ├── comparativa_mecanismos.png
│       │   └── sinteticos_b_comparativa.png
│       ├── visual_index_validation.png
│       ├── robustez_gamma.png
│       ├── topic_1_top8.png ... topic_8_top8.png
│       ├── mapa_familias_simuladas.png
│       └── pca_component_selection.png
│   └── paper/
├── scripts/
│   ├── — Descarga y limpieza —
│   ├── 00_fetch_geodata.py
│   ├── 00_fetch_saber11_bogota.py
│   ├── 00_fetch_em2021_variables.py
│   ├── 00_fetch_poblacion_upz.py
│   ├── 00_colegios_csv_to_geojson.py
│   ├── 00_merge_saber_geojson.py
│   ├── 00_demand_capacity_colegios.py
│   ├── 00_build_em2021_por_upz.py
│   ├── 00_clean_sitp.py
│   ├── 00_clean_parques.py
│   ├── 00_clean_delitos.py
│   ├── 00_build_competencia_privada.py
│   ├── 00_analyze_mapillary_colegios.py
│   ├── 00_download_mapillary_colegios.py
│   ├── 00_download_gsv_colegios.py
│   ├── — Features visuales —
│   ├── 01_build_dataset.py
│   ├── 02a_extract_embeddings.py
│   ├── 02b_diagnose_embeddings.py
│   ├── 02c_seg_cityscapes.py
│   ├── 03a_lda_topics.py
│   ├── 03b_nmf_topics.py
│   ├── 03c_clip_features.py
│   ├── 03d_imputacion_espacial.py
│   ├── — Regresión y validación —
│   ├── 04a_regresion.py
│   ├── 04b_build_capacidad.py
│   ├── 04c_decompose_aj.py
│   ├── 05c_visual_index_validation.py
│   ├── — Simulación de familias —
│   ├── 05a_simular_distancias.py
│   ├── 05b_expandir_familias.py
│   ├── 06_preferencias.py
│   ├── — Matching —
│   ├── 07_boston_mechanism.py
│   ├── 07_da_mechanism.py
│   ├── 07_sed_lex.py
│   ├── — Experimento sintético —
│   ├── 08_datos_sinteticos.py
│   ├── 09a_matching_sinteticos.py
│   ├── 09b_robustez_gamma.py
│   ├── — Utilidades —
│   ├── matching_utils.py
│   ├── compare_mechanisms.py
│   ├── compare_mapillary_filters.py
│   ├── topic_viz.py
│   ├── mapa_cobertura_gsv.py
│   ├── gsv_config.py
│   ├── mapillary_filtros.py
│   └── deeplabv3/
│       ├── modeling.py
│       ├── _deeplab.py
│       ├── utils.py
│       └── backbone/
├── requirements.txt
└── README.md
```

---

## Scripts

### Descarga y limpieza

| Script | Descripción | Output |
|---|---|---|
| `00_fetch_geodata.py` | Descarga directorio SED, demanda y matrícula desde datos abiertos Bogotá | `colegios_dataset.csv`, `demandacupos04_2024.geojson`, `matriculatotal04_2024.geojson` |
| `00_fetch_saber11_bogota.py` | Descarga resultados ICFES Saber 11 | `saber11_bogota_2020_2022.csv` |
| `00_fetch_em2021_variables.py` | Descarga EM2021 (encuesta principal + variables adicionales + familias escolares) | `em2021_*.csv` |
| `00_fetch_poblacion_upz.py` | Limpia Excel DANE de población por UPZ | `poblacion_upz_2024.parquet` |
| `00_colegios_csv_to_geojson.py` | Geocodifica directorio SED, filtra sector Oficial | `data/processed/colegios_dataset.geojson` |
| `00_merge_saber_geojson.py` | Unifica Saber 11 multi-año por establecimiento | `saber_bogota_merged.geojson` |
| `00_demand_capacity_colegios.py` | Limpia demanda y matrícula | `demanda_clean.geojson`, `matriculas_clean.geojson` |
| `00_build_em2021_por_upz.py` | Agrega EM2021 por UPZ con ponderación `FEX_C` | `em2021_por_upz.csv` |
| `00_clean_sitp.py` | Normaliza GeoJSON paraderos SITP | `sitp_clean.geojson` |
| `00_clean_parques.py` | Convierte parques de MAGNA-SIRGAS a WGS84 | `parques_clean.geojson` |
| `00_clean_delitos.py` | Agrega delitos por localidad | `delitos_por_localidad.csv` |
| `00_build_competencia_privada.py` | Calcula % sedes no oficiales por localidad | `competencia_privada_localidad.csv` |
| `00_download_gsv_colegios.py` | Descarga imágenes GSV (requiere API key en `gsv_config.py`) | `data/images/gsv/` |
| `00_analyze_mapillary_colegios.py` | Construye catálogo Mapillary y mapas de cobertura | `mapillary_catalog.csv`, mapas |
| `00_download_mapillary_colegios.py` | Descarga imágenes Mapillary filtradas | `data/images/mapillary/` |

### Features visuales

| Script | Descripción | Output |
|---|---|---|
| `01_build_dataset.py` | Integra todas las fuentes en dataset maestro de colegios | `colegios_features.geojson` |
| `02a_extract_embeddings.py` | Extrae embeddings VGG19 (512d) por imagen GSV | `gsv_vgg19_raw.parquet` |
| `02b_diagnose_embeddings.py` | Diagnóstico de calidad y selección de d PCA | `diagnostico_embeddings.json`, `pca_component_selection.png` |
| `02c_seg_cityscapes.py` | Segmentación semántica DeepLabV3+ (19 clases Cityscapes) | `gsv_cs_raw.parquet`, `gsv_cs_establecimiento.parquet` |
| `03a_lda_topics.py` | PCA(68d) + LDA sobre imágenes individuales | `gsv_lda_K{k}*.parquet` |
| `03b_nmf_topics.py` | PCA(68d) + NMF sobre imágenes → tópicos por establecimiento (modelo principal) | `gsv_nmf_K{k}*.parquet` |
| `03c_clip_features.py` | Scores perceptuales CLIP (mantenimiento, vegetación, accesibilidad, seguridad) | `gsv_clip_establecimiento.parquet` |
| `03d_imputacion_espacial.py` | Imputa NaN por media de vecinos en radio 2 km | `colegios_features_imputed.geojson` |

### Regresión y validación

| Script | Descripción | Output |
|---|---|---|
| `04a_regresion.py` | Comparativa OLS/Ridge/Lasso/ElasticNet sobre `log(sobredemanda)` | `models/`, `comparativa_estimadores.csv`, `mejores_coefs.csv` |
| `04b_build_capacidad.py` | Capacidad estimada (`matrícula / 13`) y atractivo `a_j` por colegio | `colegios_capacidad.parquet` |
| `04c_decompose_aj.py` | Descompone `a_j` en componente visual y no-visual | — |
| `05c_visual_index_validation.py` | Construye índice visual `v_j`, rank y quintil; genera figura de validación | `vj_scores.parquet`, `visual_index_validation.png` |

### Simulación de familias

| Script | Descripción | Output |
|---|---|---|
| `05a_simular_distancias.py` | Ubica familias en manzana por estrato + distancias Haversine a colegios | `familias_ubicadas.parquet`, `familias_distancias.parquet` |
| `05b_expandir_familias.py` | Expande muestra con factor `FEX_C` y asigna grupo SISBEN `s` por ingreso | `familias_expandidas.parquet`, `distancias_expandidas.parquet` |
| `06_preferencias.py` | Modelo de utilidad RUM + rankings top-20 por familia | `preferencias_familias.parquet`, `utilidades_familias.parquet` |

### Matching sobre datos reales

| Script | Descripción | Output |
|---|---|---|
| `07_boston_mechanism.py` | Boston Mechanism con prioridad por distancia | `matching_bm.parquet`, `matching_bm_summary.csv` |
| `07_da_mechanism.py` | Deferred Acceptance (Gale-Shapley) con prioridad por distancia | `matching_da.parquet`, `matching_da_summary.csv` |
| `07_sed_lex.py` | SED-lex (prioridad SISBEN + distancia) y SED-dist (control) | `matching_sed_lex.parquet`, `matching_sed_dist.parquet`, `matching_sed_comparison.csv` |
| `compare_mechanisms.py` | Tabla y figura comparativa de los tres mecanismos | `comparativa_mecanismos.csv`, `comparativa_mecanismos.png` |

### Experimento sintético

| Script | Descripción | Output |
|---|---|---|
| `08_datos_sinteticos.py` | Genera población sintética N=10,000 / M=100 con sesgo visual calibrado (`γ₀=1`) | `sinteticos_b_*.parquet`, `sinteticos_b_calibracion.json` |
| `09a_matching_sinteticos.py` | Matching en 6 condiciones (BM/DA/SED × sesgo/verdad), seed=42 | `sinteticos_b_resultados.parquet`, `sinteticos_b_comparativa.csv`, `sinteticos_b_comparativa.png` |
| `09b_robustez_gamma.py` | Análisis de robustez a γ₀ ∈ {0.25, 0.50, 0.75, 1.00, 1.25, 1.50} | `robustez_gamma.csv`, `robustez_gamma.png` |

### Módulo compartido

`matching_utils.py` implementa `boston_mechanism`, `deferred_acceptance`, `count_blocking_pairs` y `compute_metrics` como funciones puras que reciben una `priority_fn` configurable; compartido por todos los scripts de matching.
