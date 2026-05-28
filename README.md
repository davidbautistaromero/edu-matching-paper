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

Las imágenes GSV y Mapillary no se incluyen en el repositorio. Ver `scripts/gsv_config.py` y `scripts/mapillary_filtros.py` para configurar las descargas.

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

# ── Fase 2-3 — Features visuales (GPU recomendado) ────────────────────────
python scripts/02a_extract_embeddings.py
python scripts/02b_diagnose_embeddings.py
python scripts/02c_seg_cityscapes.py
python scripts/03b_nmf_topics.py                # NMF K=8 (modelo principal)
python scripts/03c_clip_features.py
python scripts/03d_imputacion_espacial.py

# ── Fase 4 — Estimación de demanda ────────────────────────────────────────
python scripts/04a_berry_ols.py              # Berry inversion + OLS (6 specs M1-M6)
python scripts/04b_blp.py                    # BLP-GMM: baseline + IV-BLP
python scripts/04c_build_capacidad.py        # capacidad escolar

# ── Fase 5 — Validación visual + simulación de familias ───────────────────
python scripts/05a_simular_distancias.py
python scripts/05b_expandir_familias.py      # expansión FEX_C + ruido en distancias
python scripts/05c_visual_index_validation.py  # figura top/bottom 5 (fotos curadas)

# ── Fase 6 — Preferencias (utilidad BLP) ──────────────────────────────────
python scripts/06_preferencias.py            # u_ij = δ_j + π₁·y_i·seg_z + λ₀·log1p(d) + λ₁·y_i·log1p(d) + ε

# ── Fase 7 — Mecanismos de matching (datos reales) ────────────────────────
python scripts/07_boston_mechanism.py
python scripts/07_da_mechanism.py
python scripts/07_sed_lex.py
python scripts/compare_mechanisms.py

# ── Fase 8 — Experimento sintético y robustez ─────────────────────────────
python scripts/08_datos_sinteticos.py
python scripts/09a_matching_sinteticos.py
python scripts/09b_robustez_gamma.py
```

> `03a_lda_topics.py` (LDA) está superado por NMF; no es necesario para reproducir resultados.

---

## Arquitectura

```
data/raw/          →  00_* scripts     →  data/processed/
data/processed/    →  01_build_dataset →  data/primary/colegios_features.geojson
GSV images         →  02-03 scripts    →  embeddings, segmentación, scores CLIP
colegios_features  →  04a_berry_ols    →  berry_delta_j.parquet (inversión Berry)
                   →  04b_blp          →  blp_delta_j.parquet (δ_j BLP, preferido)
                   →  04c              →  colegios_capacidad.parquet
familias + δ_j BLP →  05-06            →  preferencias_familias.parquet
                   →  07_*             →  data/results/matching_*.parquet
```

### Módulos compartidos

- **`matching_utils.py`** — Algoritmos `boston_mechanism` y `deferred_acceptance` como funciones puras con `priority_fn` configurable. También `compute_metrics` para evaluar eficiencia, equidad y sesgo visual.
- **`gsv_config.py`** — Parámetros de descarga GSV (N_HEADINGS=10, IMG_SIZE=640×640, FOV=90°).
- **`scripts/deeplabv3/`** — Implementación local DeepLabV3+ (backbone ResNet-101).

### Modelo estructural

**Berry OLS** (`04a`): inversión de Berry δ_j = log(s_j) − log(s₀), luego OLS con 6 especificaciones (M1–M6) que testean features visuales incrementalmente.

**BLP con micro-momentos** (`04b`):
```
u_ij = δ_j + π₁·yᵢ·seg_z_j + λ₀·log(1+d_ij) + λ₁·yᵢ·log(1+d_ij) + ε_ij
```
- θ = (π₁, λ₀, λ₁) estimados por GMM
- yᵢ = ingreso normalizado (N_ingpc / media)
- seg_z_j = seguridad percibida estandarizada (score CLIP)
- Dos specs: Baseline (Z=X) e IV-BLP (instrumentos BLP para q_j endógeno)
- First-stage F = 10.70; resultados robustos entre specs
- **Resultado clave:** seguridad percibida (+0.12) significativa; calidad académica q_j (−0.18) NO atrae demanda

**Preferencias** (`06`): usa directamente δ_j y θ del BLP. Genera rankings top-20 para 537K familias expandidas.

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

### Pendiente: WP-Rule (Weighted Polytope Rule)

Diseño de mecanismos vía StructSVM (Narasimhan, Agarwal & Parkes, 2016). Aprende pesos λ_ij sobre el politopo de matchings estables para aproximar un target de equidad. Documentado en:
- `reports/paper/maching_learning_matching.md` — teoría y resultados Wake County
- `reports/paper/notas_metodologicas.md` (Tarea 4) — plan de implementación para Bogotá

### Outputs

Resultados, tablas y figuras en `reports/`. Decisiones metodológicas documentadas en `reports/paper/notas_metodologicas.md`.

---

## Notas

- Todos los scripts y comentarios están en español.
- Archivos grandes (imágenes >50 MB, matrices de embeddings) están en `.gitignore`.
- 382 colegios urbanos tras filtros (el dataset original incluye ~558 con rurales).
- `05c` usa selección manual curada de fotos (headings elegidos por inspección visual).
