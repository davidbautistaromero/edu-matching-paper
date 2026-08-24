# Matching Escolar – Bogotá

Código de reproducción del paper *"Señales Visuales y Mecanismos de Asignación Escolar en Bogotá"*. Construye un índice visual de 382 colegios oficiales urbanos a partir de imágenes de Google Street View, lo incorpora en un modelo estructural de demanda (BLP con micro-momentos), y compara mecanismos de asignación escolar sobre datos reales y población sintética calibrada.

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

Las imagenes GSV no se incluyen en el repositorio. Ver `scripts/gsv_config.py` para configurar la descarga. La rama Mapillary quedo descartada (el paper usa solo GSV) y esta archivada en `scripts/legacy/`; `MAPILLARY_TOKEN` solo hace falta para ejecutarla.

---

## Reproducir resultados

Los scripts están numerados y deben correrse en orden desde la raíz del repositorio:

```powershell
# ── Fase 0 — Descarga y limpieza ──────────────────────────────────────────
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

# ── Fase 1 — Dataset maestro ──────────────────────────────────────────────
python scripts/01_build_dataset.py

# ── Fase 2-3 — Features visuales (GPU recomendado) ────────────────────
python scripts/02c_seg_cityscapes.py         # DeepLabV3+ Cityscapes -> proporciones semanticas
python scripts/03c_clip_features.py          # CLIP ViT-B/32 -> scores de prompts calibrados
python scripts/03d_imputacion_espacial.py    # imputacion espacial -> features_imputed

# ── Fase 4 — Estimación de demanda ────────────────────────────────────────
python scripts/04a_berry_ols.py              # inversion de Berry + OLS/2SLS (M0-M3)
python scripts/04b_blp.py                    # BLP-GMM: baseline + IV-BLP
python scripts/04c_build_capacidad.py        # capacidad escolar

# ── Fase 5 — Validación visual + simulación de familias ───────────────────
python scripts/05a_simular_distancias.py
python scripts/05b_expandir_familias.py      # expansión FEX_C + ruido en distancias
python scripts/05c_visual_index_validation.py  # figura top/bottom 5 (fotos curadas)

# ── Fase 6 — Preferencias (utilidad BLP) ──────────────────────────────────
python scripts/06_preferencias.py            # u_ij = δ_j + π₁·y_i·seg_z + λ₀·log1p(d) + λ₁·y_i·log1p(d) + ε

# ── Fase 7 — WP-Rule: calibración de θ* y aprendizaje de W ───────────
python scripts/07_WP_rule.py                 # -> reports/wp_calibracion.json

# ── Fase 8 — Mecanismos en mercados sintéticos calibrados ────────────
python scripts/08_simulacion_mecanismos.py   # -> wp_rule_results.csv, wp_rho_sweep.csv

# ── Fase 9 — Mecanismos en datos reales ─────────────────────────────
python scripts/09_mecanismos_reales.py       # -> mecanismos_reales_results.csv
```

### Correspondencia con las tablas del paper

`reports/paper/short-paper.tex` tiene sus tablas escritas a mano. Cada una proviene de:

| Tabla | Archivo de resultados | Script |
|---|---|---|
| T1 · Berry IV (M0–M3) | `reports/tables/berry_iv_specs.csv` | `04a_berry_ols.py` |
| T1 · columna IV-BLP | `reports/tables/blp_results.csv` (fila `iv_blp`) | `04b_blp.py` |
| T2 · mecanismos sintéticos | `reports/tables/wp_rule_results.csv` | `08_simulacion_mecanismos.py` |
| T3 · mecanismos reales | `reports/tables/mecanismos_reales_results.csv` | `09_mecanismos_reales.py` |

Correr las fases 0–9 en orden reproduce estos cuatro archivos. `07_WP_rule.py` es
obligatorio antes de las fases 8 y 9: ambas leen `reports/wp_calibracion.json`.

### `scripts/legacy/` y `scripts/diagnostics/`

- **`legacy/`** — ramas superadas o abandonadas, conservadas por trazabilidad y
  porque reproducen figuras de `ppt.tex`: mecanismos v1 (`07_boston_mechanism`,
  `07_da_mechanism`, `07_sed_lex`, `compare_mechanisms`), sintéticos v1
  (`08_datos_sinteticos`, `09a_matching_sinteticos`, `09b_robustez_gamma`,
  `09d_uniqueness_check`), BLP v1 (`10_blp_utility_estimation`), rama VGG19/NMF
  (`02a`, `02b`, `03a_lda_topics`, `03b_nmf_topics`, `topic_viz`) y rama Mapillary.
  **Ninguno alimenta el short paper.**
- **`diagnostics/`** — chequeos que sólo escriben a consola y respaldan
  `notas_metodologicas.md` (unicidad del retículo estable, redistribución, acceso
  SISBEN, escala del retículo). No producen artefactos del pipeline.

---

## Arquitectura

```
data/raw/          →  00_* scripts     →  data/processed/
data/processed/    →  01_build_dataset →  data/primary/colegios_features.geojson
GSV images         →  02c, 03c         →  segmentación Cityscapes, scores CLIP
colegios_features  →  04a_berry_ols    →  berry_delta_j.parquet (inversión Berry)
                   →  04b_blp          →  blp_delta_j.parquet (δ_j BLP, preferido)
                   →  04c              →  colegios_capacidad.parquet
familias + δ_j BLP →  05-06            →  preferencias_familias.parquet
                   →  07_WP_rule       →  reports/wp_calibracion.json (θ*, W)
                   →  08_simulacion    →  wp_rule_results.csv (sintéticos)
                   →  09_mecanismos    →  matching_real_*.parquet + tabla real
```

### Módulos compartidos

- **`matching_utils.py`** — Algoritmos `boston_mechanism` y `deferred_acceptance` como funciones puras con `priority_fn` configurable. También `compute_metrics` para evaluar eficiencia, equidad y sesgo visual.
- **`gsv_config.py`** — Parámetros de descarga GSV (N_HEADINGS=10, IMG_SIZE=640×640, FOV=90°).
- **`scripts/deeplabv3/`** — Implementación local DeepLabV3+ (backbone ResNet-101).

### Modelo estructural

**Berry IV** (`04a`): inversion de Berry d_j = log(s_j) - log(s_0), luego OLS y 2SLS con cuatro especificaciones (M0-M3) que introducen las senales visuales de forma incremental. `q_j` se instrumenta con la calidad media de rivales ponderada por 1/d dentro de la localidad. La Tabla 1 del paper usa las columnas IV.

**BLP con micro-momentos** (`04b`):
```
u_ij = δ_j + π₁·yᵢ·seg_z_j + λ₀·log(1+d_ij) + λ₁·yᵢ·log(1+d_ij) + ε_ij
```
- θ = (π₁, λ₀, λ₁) estimados por GMM
- yᵢ = ingreso normalizado (N_ingpc / media)
- seg_z_j = seguridad percibida estandarizada (score CLIP)
- Dos specs: Baseline (Z=X) e IV-BLP (instrumentos BLP para q_j endógeno)
- First-stage F ~ 18.8 en IV-BLP y 43-47 en Berry IV (instrumento excluido); SE por bootstrap
- **Resultado clave:** seguridad percibida (+0.12) significativa; calidad académica q_j (−0.18) NO atrae demanda

**Preferencias** (`06`): usa directamente d_j y theta del BLP. Genera rankings top-20 para las 537.031 familias expandidas por FEX_C. En `09_mecanismos_reales.py` el universo efectivo se restringe a las 99.890 familias que buscan cupo de primer ingreso, sobre 380 colegios con senal visual observada y 113.857 cupos: son las cifras que reporta el paper.

### Variables clave

| Variable | Descripción |
|---|---|
| `δ_j` | Utilidad media BLP (contraction mapping) |
| `v_j` | Sub-índice visual = β_seg · seg_z (único beta visual significativo en Berry M3) |
| `ξ_j` | Calidad no observada (δ_j − X·β) |
| `FEX_C` | Factor de expansión EM2021; siempre ponderar con esta columna |
| `id_establecimiento` | Identificador primario de colegio (código DANE) |
| `N_ingpc` | Ingreso per cápita del hogar (continuo) |
| `γ` | Parámetro de sesgo visual en experimento sintético; rango {0.25, ..., 1.5} |

### WP-Rule (Weighted Polytope Rule) - implementada

Diseno de mecanismos via StructSVM (Narasimhan, Agarwal & Parkes, 2016). Aprende
pesos lambda_ij sobre el politopo de matchings estables para aproximar un target
de equidad.

- `07_WP_rule.py` calibra la prioridad theta* (ingreso-distancia-visual) y aprende
  W = (a, b, d_v). Escribe `reports/wp_calibracion.json`, insumo obligatorio de las
  fases 8 y 9.
- `08_simulacion_mecanismos.py` evalua BM / DA / SED-lex / WP_learned en mercados
  sinteticos calibrados a Bogota, generados internamente (semilla 123).
- `09_mecanismos_reales.py` aplica la version escalable DA-P^theta* en datos reales;
  el LP global de WP no se resuelve a escala de 99.890 familias.

Documentacion: `reports/paper/maching_learning_matching.md` (teoria, Wake County) y
`reports/paper/notas_metodologicas.md`.

### Outputs

Resultados, tablas y figuras en `reports/`. Decisiones metodológicas documentadas en `reports/paper/notas_metodologicas.md`.

---

## Notas

- Todos los scripts y comentarios están en español.
- Archivos grandes (imágenes >50 MB, matrices de embeddings) están en `.gitignore`.
- 382 colegios urbanos tras filtros (el dataset original incluye ~558 con rurales).
- `05c` usa selección manual curada de fotos (headings elegidos por inspección visual).
- `data/processed/poblacion_upz_2024.parquet` (de `00_fetch_poblacion_upz.py`) no lo
  consume ningun script del pipeline; se conserva como fuente de referencia.
- `models/*.joblib` provienen de un `04a_regresion.py` que ya no existe en el repo y
  ningun script los carga; se conservan solo como registro historico.
