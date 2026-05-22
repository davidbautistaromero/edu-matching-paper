# Matching Escolar – Bogotá

Código de reproducción del paper *"Señales Visuales y Mecanismos de Asignación Escolar en Bogotá"*. Estima un índice visual de ~558 colegios oficiales a partir de imágenes de Google Street View, lo incorpora en un modelo de utilidad de hogares y compara tres mecanismos de asignación (Boston, Deferred Acceptance, SED-lex) sobre datos reales y una población sintética calibrada.

---

## Setup

```powershell
.\setup.ps1
```

Crea `.venv`, detecta GPU NVIDIA e instala PyTorch con CUDA 12.4 (o CPU), instala `requirements.txt`, descarga el checkpoint DeepLabV3+ ResNet-101 (~449 MB) en `checkpoints/`.

> Si la descarga del checkpoint falla, bajarlo manualmente desde:  
> https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb  
> y guardarlo como `checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth`

Activar el entorno antes de correr cualquier script:

```powershell
.venv\Scripts\Activate.ps1
```

Las credenciales de API van en `.env` (ya en `.gitignore`):

```
GSV_API_TOKEN=...
MAPILLARY_TOKEN=...
```

---

## Datos externos

Estos archivos deben descargarse manualmente y colocarse en `data/raw/` antes de correr el pipeline:

| Archivo | Fuente |
|---|---|
| `estaciones_transmilenio.geojson` | [Datos Abiertos Bogotá](https://datosabiertos.bogota.gov.co) – Transmilenio S.A. |
| `paraderos_sitp.geojson` | Datos Abiertos Bogotá – Transmilenio S.A. |
| `parques_bogota.geojson` | Datos Abiertos Bogotá – Secretaría Distrital de Planeación |
| `delitos_alto_impacto.geojson` | Datos Abiertos Bogotá – Secretaría Distrital de Seguridad |
| `localidades_bogota.geojson` | Datos Abiertos Bogotá – IDECA |
| `manzana_estratificacion.geojson` | Datos Abiertos Bogotá – UAESP |
| `poblacion-localidad-upz-bogota-2018-2024.xlsx` | DANE – Proyecciones de población |
| `upz/upz.shp` (+ `.dbf`, `.prj`, `.shx`) | Datos Abiertos Bogotá – IDECA |

Las imágenes GSV y Mapillary no se incluyen en el repositorio. Ver `scripts/gsv_config.py` y `scripts/mapillary_filtros.py` para configurar las descargas.

---

## Reproducir resultados

Los scripts están numerados y deben correrse en orden desde la raíz del repositorio:

```powershell
# Fase 0 — Descarga y limpieza (independientes entre sí, excepto 00_download_gsv_colegios.py)
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
python scripts/00_download_gsv_colegios.py      # requiere GSV_API_TOKEN

# Fase 1 — Dataset maestro
python scripts/01_build_dataset.py

# Fase 2-3 — Features visuales (GPU recomendado para 02c y 03c)
python scripts/02a_extract_embeddings.py
python scripts/02b_diagnose_embeddings.py
python scripts/02c_seg_cityscapes.py
python scripts/03b_nmf_topics.py                # modelo principal (K=8)
python scripts/03c_clip_features.py
python scripts/03d_imputacion_espacial.py

# Fase 4-5 — Regresión, atractivo e índice visual
python scripts/04a_regresion.py
python scripts/04b_build_capacidad.py
python scripts/04c_decompose_aj.py
python scripts/05c_visual_index_validation.py

# Fase 5-6 — Simulación de familias y preferencias
python scripts/05a_simular_distancias.py
python scripts/05b_expandir_familias.py
python scripts/06_preferencias.py

# Fase 7 — Matching sobre datos reales
python scripts/07_boston_mechanism.py
python scripts/07_da_mechanism.py
python scripts/07_sed_lex.py
python scripts/compare_mechanisms.py

# Fase 8 — Experimento sintético y robustez
python scripts/08_datos_sinteticos.py
python scripts/09a_matching_sinteticos.py
python scripts/09b_robustez_gamma.py
```

> `03a_lda_topics.py` (LDA) está superado por NMF en la práctica; no es necesario para reproducir los resultados del paper.

---

## Arquitectura

```
data/raw/        →  00_* scripts  →  data/processed/
data/processed/  →  01_build_dataset  →  data/primary/colegios_features.geojson
GSV images       →  02-03 scripts →  embeddings, segmentación, scores CLIP
colegios_features →  04a_regresion →  models/  (Ridge seleccionado)
                  →  04b           →  colegios_capacidad.parquet  (índice a_j)
families + a_j   →  05-06         →  utilidades_familias.parquet
                  →  07_*          →  data/results/matching_*.parquet
```

**Módulos compartidos:**
- `matching_utils.py` — algoritmos `boston_mechanism` y `deferred_acceptance` como funciones puras con `priority_fn` configurable; también `compute_metrics` para evaluar eficiencia, equidad y sesgo visual.
- `gsv_config.py` — parámetros de descarga GSV (N_HEADINGS, IMG_SIZE, FOV). Modificar aquí, no en scripts individuales.
- `scripts/deeplabv3/` — implementación local DeepLabV3+ (backbone ResNet-101). No modificar salvo actualización del modelo de segmentación.

**Variables clave:**
- `a_j` — atractivo del colegio (variable dependiente en regresión, construida como log(sobredemanda))
- `v_j` — sub-índice visual (componente NMF + CLIP de `a_j`)
- `FEX_C` — factor de expansión EM2021; siempre ponderar agregaciones con esta columna
- `id_establecimiento` — identificador primario de colegio en todos los datasets
- `γ (gamma)` — parámetro de sesgo visual en experimento sintético; rango {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}

Resultados, tablas y figuras se guardan en `reports/`. Las decisiones metodológicas están documentadas en `reports/paper/notas_metodologicas.md`.
