# Matching Escolar 

Analisis de mecanismos de asignacion escolar en Bogota con modelos de matching y senales visuales.

---

## Instalacion rapida

```powershell
cd C:\ruta\al\repo
.\setup.ps1
```

El script `setup.ps1` hace todo automaticamente:
1. Crea el entorno virtual `.venv`
2. Detecta si hay GPU NVIDIA disponible
   - **Con GPU:** instala torch con soporte CUDA 12.4 (recomendado — el full run tarda ~5-8 min)
   - **Sin GPU:** instala torch CPU (el full run puede tardar ~80 min)
3. Instala el resto de dependencias (`requirements.txt`)
4. Descarga el checkpoint DeepLabV3+ ResNet-101 (~449 MB) si no existe en `checkpoints/`

> **Nota:** El checkpoint no esta en git por su tamano. El setup lo descarga automaticamente
> desde Google Drive. Si la descarga falla, bajalo manualmente desde:
> https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb
> y guardalo en `checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth`

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
│   │   ├── estaciones_transmilenio.geojson
│   │   ├── paraderos_sitp.geojson
│   │   ├── parques_bogota.geojson
│   │   └── delitos_alto_impacto.geojson
│   ├── processed/                  <- Intermedios limpios, output de scripts 00_*
│   │   ├── colegios_dataset.geojson
│   │   ├── saber_bogota_merged.geojson
│   │   ├── demanda_clean.geojson
│   │   ├── matriculas_clean.geojson
│   │   ├── em2021_por_upz.csv
│   │   ├── sitp_clean.geojson
│   │   ├── parques_clean.geojson
│   │   ├── delitos_por_localidad.csv
│   │   └── competencia_privada_localidad.csv
│   ├── primary/                    <- Dataset maestro, output de 01_build_dataset.py
│   │   └── colegios_features.geojson
│   └── images/
│       ├── gsv/                    <- Imágenes Google Street View (558 sedes × 10 headings)
│       │   ├── {id_establecimiento}/
│       │   │   └── {id_sede}_{heading:03d}.jpg
│       │   ├── gsv_catalog.csv     <- Metadatos de descarga (ignorado en git)
│       │   └── mapa_cobertura_gsv.png
│       ├── mapillary/              <- Catálogo Mapillary (imágenes ignoradas en git)
│       │   ├── mapillary_catalog.csv  <- ignorado en git (>100 MB)
│       │   ├── resumen_fechas.csv     <- ignorado en git
│       │   ├── mapa_cobertura_total.png
│       │   └── mapa_cobertura_filtrada.png
│       └── embeddings/             <- Embeddings y tópicos visuales
│           ├── gsv_vgg19_raw.parquet          <- 1 fila por imagen (5,580 × 512)
│           ├── gsv_lda_K{k}.parquet           <- Proporciones de tópicos por establecimiento
│           ├── gsv_lda_K{k}_images.parquet    <- Proporciones de tópicos por imagen
│           ├── gsv_lda_K{k}_topics.json       <- Top features por tópico
│           └── diagnostico_embeddings.json    <- Resultados de pruebas de calidad
├── docs/                           <- Diccionarios de variables y documentacion de referencia
│   ├── em2021_diccionario.ods
│   └── em2021_variablesadicionales_diccionario.ods
├── models/
│   └── regretnet/                  <- Arquitectura y pesos del modelo RegretNet
├── notebooks/
│   └── 01_eda.ipynb                <- Analisis exploratorio de datos
├── reports/
│   ├── figures/
│   │   └── eda/                    <- Figuras del analisis exploratorio (8 figuras)
│   └── paper/                      <- Documento final
├── scripts/
│   ├── 00_fetch_geodata.py         <- Descarga geodatos SED, ICFES y parques
│   ├── 00_fetch_saber11_bogota.py  <- Descarga resultados ICFES
│   ├── 00_fetch_em2021_variables.py <- Descarga EM2021 (dos tablas)
│   ├── 00_colegios_csv_to_geojson.py <- Geocodificacion directorio SED
│   ├── 00_merge_saber_geojson.py   <- Unifica Saber 11 multi-ano
│   ├── 00_demand_capacity_colegios.py <- Limpia demanda y matricula
│   ├── 00_build_em2021_por_upz.py  <- Agrega EM2021 por UPZ
│   ├── 00_clean_sitp.py            <- Normaliza GeoJSON SITP (formato ESRI)
│   ├── 00_clean_parques.py         <- Convierte MAGNA-SIRGAS a WGS84
│   ├── 00_clean_delitos.py         <- Agrega delitos por localidad (suma CMH*CONT por localidad)
│   ├── 00_build_competencia_privada.py <- Calcula % sedes no oficiales por localidad
│   ├── 00_analyze_mapillary_colegios.py <- Catalogo Mapillary y mapas de cobertura
│   ├── 00_download_mapillary_colegios.py <- Descarga imágenes Mapillary
│   ├── 00_download_gsv_colegios.py <- Descarga imágenes Google Street View (fuente principal)
│   ├── 01_build_dataset.py         <- Integra todas las fuentes -> primary/
│   ├── 01_mapa_cobertura_gsv.py    <- Mapa de cobertura GSV
│   ├── 02_extract_embeddings.py    <- Extrae embeddings VGG19 por imagen (512d, sin agregar)
│   ├── 02a_diagnose_embeddings.py  <- Diagnóstico de calidad + selección d PCA (figura)
│   ├── 03_lda_topics.py            <- PCA(68d) + LDA sobre imágenes -> tópicos por establecimiento
│   ├── gsv_config.py               <- Parámetros GSV (modo prueba, headings, resolución)
│   └── mapillary_filtros.py        <- Parámetros de selección Mapillary
├── requirements.txt
└── README.md
```

---

## Extraccion de datos

### 1. Directorio de colegios oficiales - SED Bogota

| | |
|---|---|
| **Fuente** | Secretaria de Educacion del Distrito via datos abiertos Bogota |
| **Script** | `scripts/00_fetch_geodata.py` |
| **Output raw** | `data/raw/colegios_dataset.csv` |
| **Output processed** | `data/processed/colegios_dataset.geojson` <- generado por `00_colegios_csv_to_geojson.py` |
| **Contenido** | Nombre, direccion, coordenadas, localidad, UPZ, naturaleza juridica y caracteristicas academicas de cada sede educativa oficial |

---

### 2. Resultados Saber 11 - ICFES

| | |
|---|---|
| **Fuente** | ICFES via datos abiertos Bogota |
| **Script** | `scripts/00_fetch_saber11_bogota.py` |
| **Output raw** | `data/raw/saber11_bogota_2020_2022.csv` - `data/raw/pruebassaber2023.geojson` |
| **Output processed** | `data/processed/saber_bogota_merged.geojson` <- generado por `00_merge_saber_geojson.py` |
| **Contenido** | Puntaje promedio Saber 11 por establecimiento educativo para los anos 2020, 2022 y 2023. Variable `q_j` (calidad academica del colegio) |

---

### 3. Demanda y matricula por colegio - SED Bogota

| | |
|---|---|
| **Fuente** | Secretaria de Educacion del Distrito via datos abiertos Bogota |
| **Script** | `scripts/00_fetch_geodata.py` (descarga conjunta con directorio) |
| **Output raw** | `data/raw/demandacupos04_2024.geojson` - `data/raw/matriculatotal04_2024.geojson` |
| **Output processed** | `data/processed/demanda_clean.geojson` - `data/processed/matriculas_clean.geojson` <- generados por `00_demand_capacity_colegios.py` |
| **Contenido** | Numero de cupos demandados y matricula total por colegio. Se usa para construir `sobre_demanda_j = demanda / matricula`, variable dependiente de la regresion que estima `alpha` |

---

### 4. Encuesta Multiproposito 2021 - SDP / DANE

| | |
|---|---|
| **Fuente** | Secretaria Distrital de Planeacion via datos abiertos Bogota |
| **Script** | `scripts/00_fetch_em2021_variables.py` |
| **Output raw** | `data/raw/em2021_encuesta_principal.csv` - `data/raw/em2021_variables_adicionales.csv` |
| **Contenido** | Dos tablas descargadas por separado y cruzadas por `DIRECTORIO` / `directorio_hog` |

**Encuesta principal** (`em2021_encuesta_principal.csv`):

| Variable | Descripcion |
|---|---|
| `DIRECTORIO` | Llave de cruce entre tablas |
| `COD_UPZ_GRUPO` | UPZ de residencia del hogar |
| `COD_LOCALIDAD` | Localidad de residencia |
| `ESTRATO2021` | Estrato de muestreo |
| `NVCBP11AA` | Estrato para tarifa (real del hogar) |
| `FEX_C` | Factor de expansion muestral |

**Variables adicionales** (`em2021_variables_adicionales.csv`):

| Variable | Descripcion |
|---|---|
| `directorio_hog` | Llave de cruce entre tablas |
| `N_pobre_monetario` | Pobreza monetaria (0/1) |
| `N_pobre_extremo` | Pobreza extrema (0/1) |
| `N_pobre_ipm` | Indice de Pobreza Multidimensional (0/1) |
| `N_ingpc` | Ingreso per capita del hogar |
| `N_sin_cp` | Indice de capacidad de pago |
| `N_nper` | Numero de personas en el hogar |
| `N_gm_educ_hog` | Gasto mensual en educacion |
| `N_deficit_cuantitativo` | Deficit cuantitativo de vivienda (0/1) |
| `N_deficit_cualitativo` | Deficit cualitativo de vivienda (0/1) |
| `N_deficit_habitacional` | Deficit habitacional total (0/1) |

> Los diccionarios de variables estan en `docs/em2021_diccionario.ods` y `docs/em2021_variablesadicionales_diccionario.ods`.

---

### 5. Variables de control geograficas y de seguridad

Estas fuentes se descargan **manualmente** desde el portal de datos abiertos de Bogota y se ubican directamente en `data/raw/`. No hay script de descarga automatizado.

| Archivo | Fuente | Contenido |
|---|---|---|
| `estaciones_transmilenio.geojson` | Transmilenio S.A. | Estaciones troncales de TransMilenio. Control de accesibilidad al transporte publico |
| `paraderos_sitp.geojson` | Transmilenio S.A. | Paraderos SITP. Control de accesibilidad al transporte |
| `parques_bogota.geojson` | Secretaria Distrital de Planeacion | Parques del POT Bogota. Control de infraestructura verde del entorno |
| `delitos_alto_impacto.geojson` | Secretaria Distrital de Seguridad | Delitos de alto impacto por localidad. Control de seguridad del entorno |

**URL de descarga:** https://datosabiertos.bogota.gov.co

---

### 6. Imágenes de entorno de colegios - Google Street View *(fuente principal)*

| | |
|---|---|
| **Fuente** | Google Street View Static API |
| **Script** | `scripts/00_download_gsv_colegios.py` |
| **Config** | `scripts/gsv_config.py` (parámetros editables: `MODO_MUESTRA`, `N_HEADINGS`, resolución) |
| **Output catálogo** | `data/images/gsv/gsv_catalog.csv` *(ignorado en git — reproducible con el script)* |
| **Output imágenes** | `data/images/gsv/{id_establecimiento}/{id_sede}_{heading:03d}.jpg` *(ignoradas en git)* |
| **Output mapa** | `data/images/gsv/mapa_cobertura_gsv.png` |

**Parámetros de descarga:**
- `N_HEADINGS = 10` headings uniformes (0°, 36°, 72°, … 324°) para capturar la fachada desde múltiples ángulos
- Resolución: `640×640` px, `FOV = 90°`, `PITCH = 0`
- Reanudable: si `gsv_catalog.csv` existe, salta las imágenes ya descargadas

**Cobertura resultante:**
- Imágenes descargadas: **5,580** (558 sedes × 10 headings)
- Sedes cubiertas: **558 / 558 (100%)**
- Costo: ~$39 USD (dentro del crédito gratuito mensual de Google Cloud)

**Por qué GSV como fuente principal:** A diferencia de Mapillary (crowdsourced, calidad variable), GSV usa cámaras calibradas a altura estandarizada (~2.5 m) con resolución y encuadre consistentes. Esto reduce el ruido en los embeddings VGG19 que no proviene del colegio sino de la cámara o el ángulo.

**Modo prueba:** Activar `MODO_MUESTRA = True` en `gsv_config.py` para probar con `N_MUESTRA = 5` sedes antes del run completo. El catálogo de prueba se guarda como `gsv_catalog_muestra.csv`.

---

### 7. Imágenes de entorno de colegios - Mapillary *(referencia / exploración)*

| | |
|---|---|
| **Fuente** | Mapillary Graph API v4 (imágenes públicas con licencia CC) |
| **Scripts** | `scripts/00_analyze_mapillary_colegios.py` / `scripts/00_download_mapillary_colegios.py` |
| **Config** | `scripts/mapillary_filtros.py` (parámetros editables: fecha, ángulo, radio) |
| **Output catálogo** | `data/images/mapillary/mapillary_catalog.csv` *(ignorado en git — >100 MB)* |
| **Output resumen** | `data/images/mapillary/resumen_fechas.csv` *(ignorado en git)* |
| **Output mapas** | `data/images/mapillary/mapa_cobertura_total.png` / `mapa_cobertura_filtrada.png` |
| **Convención de nombre** | `{DANE12_EST}_{YYYY-MM-DD}_{image_id}.jpg` |

**Parámetros de búsqueda** (configurables en `mapillary_filtros.py`):
- Radio: `RADIO_M = 100` m alrededor del punto de cada colegio
- Fecha mínima: `FECHA_DESDE = "2020-01-01"`
- Ángulo máximo cámara→colegio: `ANGULO_MAX_DEG = 90°`
- Deduplicación por secuencia y espacial (10 m mínimo entre imágenes)
- Máximo por colegio: `N_MAX_POR_COLEGIO = 10`

**Cobertura del catálogo:**
- Imágenes en catálogo: ~89,856
- Sedes cubiertas: 292 / 558 (52%) tras filtros

**Nota:** Mapillary se mantiene como fuente secundaria para análisis de robustez. La fuente principal para los embeddings VGG19 es GSV (cobertura 100%).

---

### 8. Embeddings visuales de colegios

| | |
|---|---|
| **Fuente** | Imágenes GSV descargadas (5,580 imágenes — 558 sedes × 10 headings) |
| **Script** | `scripts/02_extract_embeddings.py` |
| **Output** | `data/images/embeddings/gsv_vgg19_raw.parquet` (5,580 filas × 512 features) |
| **Contenido** | Un vector de 512 dimensiones por imagen, sin agregar — la agregación ocurre en `03_lda_topics.py` |

**Metodología de extracción:**
- Modelo: VGG19 preentrenado en ImageNet, sin cabeza de clasificación
- Capa de salida: `block5_pool` + `AdaptiveAvgPool2d(1,1)` → vector de 512d no-negativo (ReLU)
- Preprocesamiento: `Resize(256)` → `CenterCrop(224)` → normalización ImageNet
- Procesamiento en lotes de 32 imágenes; tolerante a imágenes corruptas

**Por qué no se agrega aquí:** promediar los vectores por sede o establecimiento antes de LDA reduce artificialmente la variabilidad entre colegios. LDA aprende tópicos más estables sobre imágenes individuales; la agregación de proporciones ocurre *después* en espacio ya interpretable.

---

## Transformacion de datos

Los scripts de transformacion toman los archivos de `raw/` y producen versiones limpias y enriquecidas en `processed/`. Cada transformacion tiene una justificacion metodologica dentro del modelo de matching.

---

### T1. Directorio de colegios -> GeoJSON

**Script:** `scripts/colegios_csv_to_geojson.py`
**Input:** `data/raw/colegios_dataset.csv`
**Output:** `data/processed/colegios_dataset.geojson`

**Que hace:**
- Normaliza nombres de columnas (elimina tildes, espacios, mayusculas)
- Filtra solo colegios del sector **Oficial** -- el analisis se restringe a oferta publica
- Convierte coordenadas `coord_x` / `coord_y` (con coma decimal) a geometria Point en EPSG:4326
- Descarta registros sin coordenadas validas

**Por que:** El GeoJSON es la base espacial del analisis. Necesitamos la geometria de cada colegio para calcular distancias a equipamientos (TransMilenio, parques) y para asignar la UPZ de residencia a cada estudiante sintetico.

---

### T2. Resultados Saber 11 -> dataset unificado

**Script:** `scripts/merge_saber_geojson.py`
**Input:** `data/raw/saber11_bogota_2020_2022.csv` - `data/raw/pruebassaber2023.geojson`
**Output:** `data/processed/saber_bogota_merged.geojson`

**Que hace:**
- Agrega el CSV (2020-2022) por establecimiento y ano, promediando `punt_global`
- Pivota: genera columnas `punt_global_2020`, `punt_global_2022` por colegio
- Toma el GeoJSON 2023 como base georreferenciada y le une los puntajes historicos por codigo DANE
- Si hay duplicados por establecimiento (multiples jornadas), conserva el de menor `ORDEN_DE_S` y promedia puntajes

**Por que:** La variable `q_j` (calidad academica del colegio) se construye como el promedio de puntajes Saber 11 de los anos disponibles. Usar varios anos reduce el ruido de un ano atipico. El codigo DANE del establecimiento es la llave de cruce con el directorio de colegios.

---

### T3. Demanda y matricula -> variables limpias

**Script:** `scripts/demand_capacity_colegios.py`
**Input:** `data/raw/demandacupos04_2024.geojson` - `data/raw/matriculatotal04_2024.geojson`
**Output:** `data/processed/demanda_clean.geojson` - `data/processed/matriculas_clean.geojson`

**Que hace:**
- Filtra cada GeoJSON conservando solo los campos relevantes:
  - Demanda: `DANE12_EST`, `NOMBRE_EST`, `DTotal` (cupos solicitados)
  - Matricula: `DANE12_EST`, `NOMBRE_EST`, `NOMBRE_SED`, `ORDEN_DE_S`, `TMATRIC_GE` (matricula total)
- Descarta toda columna irrelevante para mantener archivos manejables

**Por que:** La variable `sobre_demanda_j = DTotal / TMATRIC_GE` es la variable dependiente de la regresion que estima `alpha` -- el peso que los hogares le dan a la apariencia visual del colegio versus su calidad academica. Un colegio con alta sobre-demanda es mas deseado de lo que su calidad objetiva justificaria, lo que puede reflejar sesgo visual.

---

### T4. Encuesta Multiproposito 2021 -> controles por UPZ

**Script:** `scripts/build_em2021_por_upz.py`
**Input:** `data/raw/em2021_encuesta_principal.csv` - `data/raw/em2021_variables_adicionales.csv`
**Output:** `data/processed/em2021_por_upz.csv`

**Que hace:**

1. **Join entre tablas:** La encuesta principal y las variables adicionales comparten un identificador de vivienda, pero con formato distinto (`DIRECTORIO = "166238"` vs `directorio_hog = "1662381"` -- el ultimo digito codifica el numero de hogar). Se reconstruye la llave truncando el ultimo caracter de `directorio_hog` antes del merge.

2. **Filtro geografico:** Se excluyen los ~65K hogares sin `COD_UPZ_GRUPO` asignado (municipios de Cundinamarca fuera de Bogota incluidos en la encuesta).

3. **Agregacion ponderada por UPZ:** Todas las metricas se calculan usando `FEX_C` (factor de expansion muestral) como peso. Sin ponderar, las UPZs con mayor intensidad de muestreo quedarian sobrerepresentadas.

**Variables construidas y su rol en el modelo:**

| Variable output | Fuente | Rol |
|---|---|---|
| `tasa_pobreza_monetaria` | Media ponderada de `N_pobre_monetario` | Control socioeconomico de la UPZ |
| `tasa_pobreza_extrema` | Media ponderada de `N_pobre_extremo` | Control socioeconomico |
| `tasa_ipm` | Media ponderada de `N_pobre_ipm` | Control multidimensional de pobreza |
| `ingreso_percapita_promedio` | Media ponderada de `N_ingpc` | Control de ingreso |
| `capacidad_pago_promedio` | Media ponderada de `N_sin_cp` | Control de capacidad adquisitiva |
| `tamano_hogar_promedio` | Media ponderada de `N_nper` | Control demografico |
| `gasto_educ_promedio` | Media ponderada de `N_gm_educ_hog` | Proxy de valoracion educativa -- hogares que gastan mas en educacion pueden ser menos susceptibles al sesgo visual |
| `tasa_deficit_cuantitativo` | Media ponderada de `N_deficit_cuantitativo` | Control de calidad del entorno fisico -- separa "el barrio se ve mal" de "el colegio se ve mal" |
| `tasa_deficit_cualitativo` | Media ponderada de `N_deficit_cualitativo` | Idem |
| `tasa_deficit_habitacional` | Media ponderada de `N_deficit_habitacional` | Resumen del entorno |
| `pct_estrato_1` ... `pct_estrato_6` | Proporcion ponderada de `NVCBP11AA` | Distribucion de estrato en la UPZ -- control y calibracion de datos sinteticos |
| `n_hogares_muestra` | Conteo sin ponderar | Indicador de confiabilidad de la celda UPZ |
| `poblacion_expandida` | Suma de `FEX_C` | Aproximacion al total de hogares reales en la UPZ |

> **Limitacion:** `COD_UPZ_GRUPO` en la EM2021 agrupa UPZs pequenas bajo un mismo codigo. El cruce posterior con el directorio de colegios puede dejar algunos sin match directo. Esta limitacion se documenta en el paper.

---

### T5. Delitos de alto impacto -> controles de seguridad por localidad

**Script:** `scripts/00_clean_delitos.py`
**Input:** `data/raw/delitos_alto_impacto.geojson`
**Output:** `data/processed/delitos_por_localidad.csv`

**Que hace:**
- Lee el GeoJSON (1 fila por localidad, columnas `CM**[YY]CONT` con conteos anuales por periodo)
- Suma las columnas de cada periodo por tipo de delito para obtener el acumulado real por localidad
- Estandariza el nombre de localidad para el cruce con el directorio de colegios (`CANDELARIA` -> `LA CANDELARIA`)
- Excluye la fila `SIN LOCALIZACION`

> **Nota:** El campo `CM**TOTAL` del GeoJSON fuente es el total de Bogota entera, no el de la localidad. Los valores correctos por localidad se obtienen sumando las columnas `CM**[YY]CONT`.

**Variables de salida:** `homicidios`, `lesiones_personales`, `hurto_personas`, `hurto_residencias`, `hurto_automotores`, `hurto_bicicletas`, `hurto_comercio`, `hurto_entidades`, `violencia_intrafam`, `delitos_sexuales`.

**Por que:** El entorno de seguridad afecta las preferencias de los hogares. Una localidad con alta criminalidad puede reducir la demanda de colegios aunque tengan buena calidad o apariencia. Incluirlo como control aísla el efecto de la señal visual.

---

### T6. Competencia del sector privado -> porcentaje por localidad

**Script:** `scripts/00_build_competencia_privada.py`
**Input:** `data/raw/colegios_dataset.csv` (directorio completo oficial + no oficial)
**Output:** `data/processed/competencia_privada_localidad.csv`

**Que hace:**
- Carga el directorio completo (sector Oficial y No Oficial)
- Agrupa por `nombre_localidad` y calcula:
  - `n_sedes_total`: total de sedes en la localidad
  - `n_sedes_no_oficial`: sedes del sector no oficial
  - `pct_no_oficial`: porcentaje de sedes no oficiales
- Normaliza el nombre de localidad para el cruce posterior

**Variables de salida:** `localidad_norm`, `n_sedes_total`, `n_sedes_no_oficial`, `pct_no_oficial`.

**Por que:** La intensidad competitiva del mercado educativo local puede moderar el efecto visual. En localidades con alta oferta privada, los hogares tienen mas opciones y pueden ser mas sensibles a senales de calidad -- tanto academica como visual. Se incluye como control en la regresion de `alpha` y como variable descriptiva en el EDA (Figura 4).

---

### T7. Extracción de embeddings VGG19

**Script:** `scripts/02_extract_embeddings.py`
**Input:** `data/images/gsv/{id_establecimiento}/*.jpg` + `data/images/gsv/gsv_catalog.csv`
**Output:** `data/images/embeddings/gsv_vgg19_raw.parquet`

**Que hace:**
- Filtra el catálogo GSV para quedarse solo con imágenes descargadas
- Carga VGG19 (ImageNet) y construye extractor `block5_pool → AvgPool → 512d`
- Procesa las 5,580 imágenes en lotes de 32, tolerante a archivos corruptos
- Guarda una fila por imagen con metadatos (`id_establecimiento`, `id_sede`, `heading`) y 512 features

**Por que la agregación no ocurre aquí:** agregar por sede/establecimiento antes de LDA suaviza los vectores y concentra los embeddings en una región más homogénea del espacio, lo que produce colapso de tópicos en LDA. La agregación se realiza en T9 una vez que las proporciones de tópicos ya son interpretables.

---

### T8. Diagnóstico de calidad de embeddings y selección de d PCA

**Script:** `scripts/02a_diagnose_embeddings.py`
**Input:** `data/images/embeddings/gsv_vgg19_raw.parquet`
**Outputs:** `data/images/embeddings/diagnostico_embeddings.json` · `reports/figures/pca_component_selection.png`

**Que hace:**
1. **Varianza por feature:** verifica que ninguna dimensión esté degenerada (var ≈ 0)
2. **Similitud coseno entre pares:** muestrea 5,000 pares aleatorios y reporta la distribución de similitudes — alerta si la media supera 0.85 (señal de colapso potencial en LDA)
3. **PCA + selección de d:** calcula hasta 150 PCs y construye la figura de sedimentación con el codo y d\* seleccionado

**Resultados sobre los datos actuales:**

| Test | Resultado | Estado |
|---|---|---|
| Varianza por feature | Var min: 0.000343, 0% features nulas | ✅ OK |
| Similitud coseno (imágenes) | Media: 0.61 | ✅ OK |
| PCA — codo | d = 18 (∼55% varianza) | — |
| PCA — umbral 70% | d = 37 | — |
| PCA — umbral 80% | d = 68 | ✅ seleccionado |
| PCA — umbral 90% | d = 141 | — |

**Criterio de selección de d\*:** se elige el umbral más alto alcanzado dentro de 150 PCs que sea ≥ 80%. Si ninguno se alcanza, se usa el 80% como piso. El codo (d=18) se reporta como referencia pero no se usa directamente porque captura solo ∼55% de la varianza visual.

**Por que d=68 y no d=37 ni d=141:** el tramo 70%→80% cuesta 31 PCs adicionales con una ganancia de 0.32% por PC — eficiente. El tramo 80%→90% cuesta 73 PCs con 0.14% por PC — ineficiente. d=68 está en el punto donde la curva entra en su zona plana, con N/d = 82 (cómodo para LDA).

---

### T9. Tópicos visuales con LDA

**Script:** `scripts/03_lda_topics.py`
**Input:** `data/images/embeddings/gsv_vgg19_raw.parquet`
**Outputs (por cada K ∈ {6, 8, 10}):**
- `data/images/embeddings/gsv_lda_K{k}_images.parquet` — proporciones por imagen
- `data/images/embeddings/gsv_lda_K{k}.parquet` — proporciones por establecimiento
- `data/images/embeddings/gsv_lda_K{k}_topics.json` — top features por tópico

**Que hace:**
1. **PCA(68):** reduce 512d → 68d y shift a no-negativo (requerido por LDA). Retiene el 80% de la varianza visual
2. **Normalización L1:** transforma cada vector a una distribución de probabilidad (suma = 1)
3. **LDA sobre imágenes individuales:** aprende K tópicos visuales con más datos y mayor variabilidad que si se corriera sobre promedios por establecimiento
4. **Agregación:** promedia las proporciones de tópicos por establecimiento → 1 vector de K dimensiones por colegio

**Por que LDA sobre imágenes y no sobre establecimientos:** con 5,580 imágenes la relación N/d = 82, suficiente para que LDA estime distribuciones estables. Correr LDA sobre los 306 vectores promediados (N/d = 4.5) producía colapso: todos los tópicos excepto uno recibían peso uniforme 1/K.

**Parámetro principal:** `K_DEFAULT = 8` tópicos. Se evalúan K ∈ {6, 8, 10} para análisis de robustez.

**Nota:** LDA colapsa en este dataset (std=0 en todos los tópicos para cualquier K). El prior de Dirichlet domina cuando los embeddings VGG19 de fachadas escolares son suficientemente homogéneos. Se reemplaza por NMF en T10.

---

### T10. Tópicos visuales con NMF (reemplaza LDA)

**Script:** `scripts/03_nmf_topics.py`
**Input:** `data/images/embeddings/gsv_vgg19_raw.parquet`
**Outputs (por cada K ∈ {6, 8, 10}):**
- `data/images/embeddings/gsv_nmf_K{k}_images.parquet` — proporciones por imagen
- `data/images/embeddings/gsv_nmf_K{k}.parquet` — proporciones por establecimiento
- `data/images/embeddings/gsv_nmf_K{k}_topics.json` — top features por tópico

**Por que NMF y no LDA:** NMF (Non-negative Matrix Factorization, Lee & Seung 1999) no asume distribuciones probabilísticas. Trabaja directamente sobre los features no-negativos (garantizados por ReLU en VGG19) y produce tópicos parts-based con varianza real entre establecimientos (std ≈ 0.05–0.12 vs std=0 en LDA).

**Parámetro principal:** `K_DEFAULT = 8`. Modelo principal en `gsv_nmf_K8.parquet`.

---

### T11. Segmentación semántica con DeepLabV3+ Cityscapes

**Script:** `scripts/02b_seg_cityscapes.py`
**Input:** `data/images/gsv/{id_establecimiento}/*.jpg`
**Outputs:**
- `data/images/segmentation/gsv_cs_raw.parquet` — proporciones por imagen
- `data/images/segmentation/gsv_cs_establecimiento.parquet` — proporciones por establecimiento
- `data/images/segmentation/diagnostico_primera_foto.png` — validación visual

**Que hace:**
- Pasa cada imagen por DeepLabV3+ ResNet-101 (Cityscapes, 19 clases) — cada píxel recibe una etiqueta semántica
- Agrupa las 19 clases en 7 categorías temáticas para la regresión:

| Grupo | Clases Cityscapes | Media (558 colegios) |
|---|---|---|
| `infraestructura_vial` | road, sidewalk | 33.1% |
| `edificacion` | building | 25.9% |
| `referencia` | sky, person, rider | 23.9% |
| `vegetacion` | vegetation, terrain | 9.6% |
| `cerramiento` | wall, fence | 3.7% |
| `vehiculos` | car, truck, bus, train, moto, bici | 2.7% |
| `mobiliario_urbano` | pole, traffic light, traffic sign | 1.0% |

- En regresión se excluye `referencia` para evitar multicolinealidad perfecta (las 7 suman 1)

**Checkpoint requerido:** `checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth` (~449 MB). El `setup.ps1` lo descarga automáticamente.

**Velocidad:** ~6 min para 5,580 imágenes con GPU RTX 4070.

**Por que Cityscapes:** entrenado específicamente en escenas urbanas (19 clases urbanas). Produce coeficientes directamente interpretables en regresión — "10pp más de vegetación → X% más sobredemanda" (Suel et al. 2019, Scientific Reports).

---

### T12. Features perceptuales con CLIP

**Script:** `scripts/02c_clip_features.py`
**Input:** `data/images/gsv/{id_establecimiento}/*.jpg`
**Outputs:**
- `data/images/clip/gsv_clip_raw.parquet` — scores por imagen
- `data/images/clip/gsv_clip_establecimiento.parquet` — scores por establecimiento

**Que hace:**
- Carga CLIP ViT-B/32 (Radford et al. 2021) — mapea imágenes y texto al mismo espacio de 512 dimensiones
- Para cada imagen calcula: `score = coseno(imagen, frase_positiva) − coseno(imagen, frase_negativa)`
- Las 4 dimensiones y sus frases son **fijas** para garantizar replicabilidad:

| Dimensión | Frase positiva | Frase negativa |
|---|---|---|
| `mantenimiento` | "a school building with a clean and well-maintained facade" | "a school building with a deteriorated and neglected facade" |
| `vegetacion_percibida` | "a school surrounded by trees and green areas" | "a school with no vegetation or green spaces around it" |
| `accesibilidad` | "a school with a welcoming and open entrance" | "a school with a closed, walled-off and unwelcoming entrance" |
| `seguridad_percibida` | "a school in a safe and calm street environment" | "a school in a dangerous and chaotic street environment" |

**Scores medios (558 colegios):** mantenimiento −0.012 · vegetacion −0.046 · accesibilidad −0.028 · seguridad −0.006

**Por que CLIP complementa Cityscapes:** Cityscapes mide *qué hay físicamente* en la imagen. CLIP mide *cómo se percibe* ese entorno — deterioro, apertura, seguridad — dimensiones subjetivas validadas en Naik et al. (2017) y Dubey et al. (2016) como predictores de comportamiento urbano.

**Velocidad:** ~54 segundos para 5,580 imágenes con GPU RTX 4070.

---

### T13. Imputación espacial de valores faltantes

**Script:** `scripts/03b_imputacion_espacial.py`
**Input:** `data/primary/colegios_features.geojson`
**Output:** `data/primary/colegios_features_imputed.geojson`

**Que hace:**
- Detecta automáticamente columnas numéricas con NaN (25 columnas, 661 valores faltantes)
- Para cada valor faltante: busca vecinos dentro de un radio de 2 km (haversine, BallTree)
  y imputa con la media de los vecinos con valor válido
- Fallback a mediana global si no hay vecinos en el radio (40 casos)
- Resultado: 306 establecimientos sin NaN en ninguna variable numérica

**Por que imputación espacial:** los colegios cercanos comparten entorno socioeconómico.
La media de vecindario es mejor estimador que la media global, especialmente para variables
de UPZ (pobreza, ingreso) que varían fuertemente por zona.

---

### T14. LASSO comparativo de métodos visuales

**Script:** `scripts/04_regresion.py`
**Variable dependiente:** `log(sobre_demanda_j)` — log del ratio demanda/matrícula
**Input:** `colegios_features_imputed.geojson` + outputs de T10, T11, T12
**Outputs:** `reports/lasso_comparativa.csv` · `reports/lasso_M4_coefs.csv`

**Por que LASSO y no OLS:** N=301, p_max=17. LASSO (Tibshirani 1996) penaliza coeficientes
pequeños a exactamente cero, produciendo selección automática de features y manejando
multicolinealidad (las proporciones Cityscapes suman 1).

**Controles (M0):** `puntaje_icfes_promedio` (media 2020/2022/2023) · `tasa_pobreza_monetaria` ·
`ingreso_percapita_promedio` · `dist_sitp_m` · `pct_no_oficial` · `hurto_personas` · `homicidios`

**Resultados (LassoCV, k=5):**

| Modelo | p_in | Activas | λ* | R²_adj | RMSE_cv |
|---|---|---|---|---|---|
| M0 — Baseline | 7 | 0 | 0.0139 | −0.024 | 0.088 |
| **M1 — NMF** | **15** | **9** | **0.0038** | **0.044** | **0.087** |
| M2 — Cityscapes | 13 | 4 | 0.0056 | −0.014 | 0.088 |
| M3 — CLIP | 11 | 0 | 0.0139 | −0.038 | 0.088 |
| M4 — Combinado | 17 | 0 | 0.0139 | −0.060 | 0.088 |

**Features seleccionadas por M1 (9):**
`topic_2` (−) · `pct_no_oficial` (+) · `topic_1` (+) · `puntaje_icfes_promedio` (+) ·
`topic_6` (−) · `hurto_personas` (+) · `dist_sitp_m` (−) · `topic_7` (−) · `ingreso_percapita_promedio` (+)

**Interpretación:** Los tópicos visuales NMF 1, 2, 6 y 7 tienen señal sobre sobredemanda
incluso controlando por calidad académica, pobreza, accesibilidad y competencia privada.
Cityscapes y CLIP no sobreviven la penalización LASSO — la señal visual está capturada
en los patrones latentes VGG19+NMF, no en proporciones semánticas ni índices perceptuales.

**Pendiente:** identificar qué representan visualmente topic_1, topic_2, topic_6 y topic_7
(inspección de imágenes con peso alto en cada tópico).
