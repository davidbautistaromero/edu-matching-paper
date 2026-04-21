# Matching Escolar - Paper AI

Analisis de mecanismos de asignacion escolar en Bogota con modelos de matching y senales visuales.

---

## Estructura del proyecto

```
paper-AI/
├── data/
│   ├── raw/                  <- Datos descargados tal como vienen de la fuente (no modificar)
│   │   ├── colegios_dataset.csv
│   │   ├── saber11_bogota_2020_2022.csv
│   │   ├── pruebassaber2023.geojson
│   │   ├── demandacupos04_2024.geojson
│   │   ├── matriculatotal04_2024.geojson
│   │   ├── em2021_encuesta_principal.csv
│   │   ├── em2021_variables_adicionales.csv
│   │   ├── estaciones_transmilenio.geojson
│   │   ├── paraderos_sitp.geojson
│   │   ├── parques_bogota.geojson          <- pendiente re-descarga
│   │   └── delitos_alto_impacto.geojson
│   ├── processed/            <- Intermedios limpios, output de scripts 00_*
│   │   ├── colegios_dataset.geojson
│   │   ├── saber_bogota_merged.geojson
│   │   ├── demanda_clean.geojson
│   │   ├── matriculas_clean.geojson
│   │   ├── em2021_por_upz.csv
│   │   ├── sitp_clean.geojson
│   │   └── delitos_por_localidad.csv
│   ├── primary/              <- Dataset maestro final, output de 01_build_dataset.py
│   │   └── colegios_features.geojson
│   └── images/
│       └── embeddings/       <- Embeddings VGG19 por colegio (equipo de imagenes)
├── docs/                     <- Diccionarios de variables y documentacion de referencia
│   ├── em2021_diccionario.ods
│   └── em2021_variablesadicionales_diccionario.ods
├── models/
│   └── regretnet/            <- Arquitectura y pesos del modelo RegretNet
├── notebooks/                <- Exploracion y analisis
├── reports/
│   ├── figures/              <- Graficas del paper
│   └── paper/                <- Documento final
├── scripts/
│   ├── 00_fetch_colegios_geo.py          <- descarga directorio SED
│   ├── 00_colegios_csv_to_geojson.py     <- limpieza y geocodificacion
│   ├── 00_fetch_saber11_bogota.py        <- descarga resultados ICFES
│   ├── 00_merge_saber_geojson.py         <- unifica Saber 11 multi-ano
│   ├── 00_demand_capacity_colegios.py    <- limpia demanda y matricula
│   ├── 00_fetch_em2021_variables.py      <- descarga EM2021 (dos tablas)
│   ├── 00_build_em2021_por_upz.py        <- agrega EM2021 por UPZ
│   ├── 00_clean_sitp.py                  <- normaliza GeoJSON SITP (formato ESRI)
│   ├── 00_clean_delitos.py               <- agrega delitos por localidad
│   └── 01_build_dataset.py               <- dataset maestro (primary/)
├── requirements.txt
└── README.md
```

> **Convencion de scripts:**
> - `00_*.py` -- extraccion y limpieza de fuentes individuales. Output en `raw/` o `processed/`.
> - `01_*.py` -- integracion. Une todos los processed en el dataset maestro. Output en `primary/`.
> - `02_*.py` y siguientes -- modelado (pendiente).

> **Nota:** `data/` no se versiona en git. Todos los archivos se reproducen corriendo los scripts en orden.

---

## Extraccion de datos

### 1. Directorio de colegios oficiales - SED Bogota

| | |
|---|---|
| **Script** | `00_fetch_colegios_geo.py` |
| **Fuente** | Secretaria de Educacion del Distrito via datos abiertos Bogota |
| **Output raw** | `data/raw/colegios_dataset.csv` |
| **Output processed** | `data/processed/colegios_dataset.geojson` <- `00_colegios_csv_to_geojson.py` |
| **Contenido** | Nombre, direccion, coordenadas, localidad, UPZ y caracteristicas academicas de cada sede oficial |

---

### 2. Resultados Saber 11 - ICFES

| | |
|---|---|
| **Script** | `00_fetch_saber11_bogota.py` |
| **Fuente** | ICFES via datos abiertos Bogota |
| **Output raw** | `data/raw/saber11_bogota_2020_2022.csv` - `data/raw/pruebassaber2023.geojson` |
| **Output processed** | `data/processed/saber_bogota_merged.geojson` <- `00_merge_saber_geojson.py` |
| **Contenido** | Puntaje promedio Saber 11 por establecimiento para 2020, 2022 y 2023. Variable `q_j` |

---

### 3. Demanda y matricula por colegio - SED Bogota

| | |
|---|---|
| **Script** | `00_fetch_colegios_geo.py` (descarga conjunta) |
| **Fuente** | Secretaria de Educacion del Distrito via datos abiertos Bogota |
| **Output raw** | `data/raw/demandacupos04_2024.geojson` - `data/raw/matriculatotal04_2024.geojson` |
| **Output processed** | `data/processed/demanda_clean.geojson` - `data/processed/matriculas_clean.geojson` <- `00_demand_capacity_colegios.py` |
| **Contenido** | Cupos demandados y matricula total por sede. Base para construir `sobre_demanda_j` |

---

### 4. Encuesta Multiproposito 2021 - SDP / DANE

| | |
|---|---|
| **Script** | `00_fetch_em2021_variables.py` |
| **Fuente** | Secretaria Distrital de Planeacion via datos abiertos Bogota |
| **Output raw** | `data/raw/em2021_encuesta_principal.csv` - `data/raw/em2021_variables_adicionales.csv` |
| **Output processed** | `data/processed/em2021_por_upz.csv` <- `00_build_em2021_por_upz.py` |
| **Contenido** | Indicadores socioeconomicos por UPZ: pobreza, ingreso, gasto educativo, deficit habitacional, distribucion de estrato |

**Encuesta principal** (`em2021_encuesta_principal.csv`):

| Variable | Descripcion |
|---|---|
| `DIRECTORIO` | Llave de cruce |
| `COD_UPZ_GRUPO` | UPZ de residencia |
| `ESTRATO2021` | Estrato de muestreo |
| `NVCBP11AA` | Estrato para tarifa (real) |
| `FEX_C` | Factor de expansion muestral |

**Variables adicionales** (`em2021_variables_adicionales.csv`):

| Variable | Descripcion |
|---|---|
| `directorio_hog` | Llave de cruce |
| `N_pobre_monetario` | Pobreza monetaria (0/1) |
| `N_pobre_extremo` | Pobreza extrema (0/1) |
| `N_pobre_ipm` | Indice de Pobreza Multidimensional (0/1) |
| `N_ingpc` | Ingreso per capita |
| `N_sin_cp` | Indice de capacidad de pago |
| `N_nper` | Numero de personas en el hogar |
| `N_gm_educ_hog` | Gasto mensual en educacion |
| `N_deficit_cuantitativo` | Deficit cuantitativo de vivienda (0/1) |
| `N_deficit_cualitativo` | Deficit cualitativo de vivienda (0/1) |
| `N_deficit_habitacional` | Deficit habitacional total (0/1) |

---

### 5. Variables de control geograficas y de seguridad

Descarga **manual** desde https://datosabiertos.bogota.gov.co -- ubicar en `data/raw/`.

| Archivo | Fuente | Script de limpieza | Output processed |
|---|---|---|---|
| `estaciones_transmilenio.geojson` | Transmilenio S.A. | (ninguno, formato estandar) | -- |
| `paraderos_sitp.geojson` | Transmilenio S.A. | `00_clean_sitp.py` | `sitp_clean.geojson` |
| `parques_bogota.geojson` | Sec. Distrital de Planeacion | pendiente | `parques_clean.geojson` |
| `delitos_alto_impacto.geojson` | Sec. Distrital de Seguridad | `00_clean_delitos.py` | `delitos_por_localidad.csv` |

> **Nota SITP:** El GeoJSON raw tiene formato ESRI (`geometry: {x, y}` sin `type`). `00_clean_sitp.py` lo convierte a GeoJSON estandar.
> **Nota delitos:** El dataset raw tiene los totales de la ciudad replicados en cada fila (no desagregados por localidad). Se usa como control de orden de magnitud. Limitacion documentada en el paper.

---

### 6. Embeddings visuales de colegios (pendiente - equipo de imagenes)

| | |
|---|---|
| **Responsable** | Otro miembro del grupo |
| **Output esperado** | `data/images/embeddings/embeddings.parquet` |
| **Contenido** | Embeddings VGG19 por colegio, para construir `v_j` en `02_visual_index.py` |

---

## Transformacion de datos

Los scripts `00_*` transforman cada fuente individual. El script `01_build_dataset.py` integra todo.

---

### T1. Colegios: sedes -> establecimiento + GeoJSON

**Script:** `00_colegios_csv_to_geojson.py`
**Input:** `data/raw/colegios_dataset.csv`
**Output:** `data/processed/colegios_dataset.geojson`

- Normaliza columnas, filtra sector Oficial, geocodifica coordenadas con coma decimal
- Descarta sedes sin coordenadas

---

### T2. Saber 11 -> q_j multi-ano

**Script:** `00_merge_saber_geojson.py`
**Input:** `saber11_bogota_2020_2022.csv` - `pruebassaber2023.geojson`
**Output:** `data/processed/saber_bogota_merged.geojson`

- Agrega por establecimiento y ano, promedia `punt_global`
- Pivota a una columna por ano: `punt_global_2020`, `punt_global_2022`, `puntaje_2023`
- Une con GeoJSON 2023 como base geografica por codigo DANE
- Usar varios anos reduce el ruido de resultados atipicos

---

### T3. Demanda y matricula -> variables limpias

**Script:** `00_demand_capacity_colegios.py`
**Input:** `demandacupos04_2024.geojson` - `matriculatotal04_2024.geojson`
**Output:** `demanda_clean.geojson` - `matriculas_clean.geojson`

- Conserva solo campos relevantes (DANE12_EST, DTotal, TMATRIC_GE)
- La matricula esta a nivel de sede -- se agrega a establecimiento en el build

---

### T4. EM2021 -> controles por UPZ

**Script:** `00_build_em2021_por_upz.py`
**Input:** `em2021_encuesta_principal.csv` - `em2021_variables_adicionales.csv`
**Output:** `data/processed/em2021_por_upz.csv`

1. **Join:** `directorio_hog` tiene un digito extra vs `DIRECTORIO` (numero de hogar). Se trunca el ultimo caracter antes del merge.
2. **Filtro:** Se excluyen ~65K hogares sin `COD_UPZ_GRUPO` (municipios de Cundinamarca).
3. **Agregacion ponderada por UPZ** usando `FEX_C` (factor de expansion muestral).

| Variable output | Rol en el modelo |
|---|---|
| `tasa_pobreza_monetaria` | Control socioeconomico |
| `tasa_ipm` | Control multidimensional de pobreza |
| `ingreso_percapita_promedio` | Control de ingreso |
| `capacidad_pago_promedio` | Control de capacidad adquisitiva |
| `gasto_educ_promedio` | Proxy de valoracion educativa |
| `tasa_deficit_cuantitativo/cualitativo` | Control de calidad del entorno fisico |
| `pct_estrato_1...6` | Distribucion de estrato -- control y calibracion sinteticos |
| `n_hogares_muestra` | Confiabilidad de la celda UPZ |

---

### T5. SITP -> GeoJSON estandar

**Script:** `00_clean_sitp.py`
**Input:** `data/raw/paraderos_sitp.geojson`
**Output:** `data/processed/sitp_clean.geojson`

- Convierte formato ESRI (`geometry: {x, y}`) a GeoJSON estandar (`Point`)
- 7,653 paraderos validos, 0 omitidos

---

### T6. Delitos -> tabla por localidad

**Script:** `00_clean_delitos.py`
**Input:** `data/raw/delitos_alto_impacto.geojson`
**Output:** `data/processed/delitos_por_localidad.csv`

- Agrega por localidad sumando todos los meses
- Normaliza nombres (elimina tildes, mayusculas) para join robusto
- Corrige `Candelaria` -> `La Candelaria` para alinear con directorio SED
- Columnas: `homicidios`, `lesiones_personales`, `hurto_personas`, `hurto_residencias`, `hurto_automotores`, `hurto_bicicletas`, `hurto_comercio`, `hurto_entidades`, `violencia_intrafam`, `delitos_sexuales`

---

### T7. Dataset maestro (BUILD)

**Script:** `01_build_dataset.py`
**Input:** todos los `processed/` anteriores
**Output:** `data/primary/colegios_features.geojson`

Unidad de analisis: **establecimiento educativo** (no sede).

| Paso | Operacion |
|---|---|
| 1 | Agrega sedes -> establecimiento, toma sede principal como punto geografico |
| 2 | Join con Saber 11, construye `q_j = media(2020, 2022, 2023)` |
| 3 | Agrega matricula por establecimiento (suma sedes), join con demanda |
| 4 | Construye `sobre_demanda_j = demanda / matricula` |
| 5 | Join controles EM2021 por UPZ |
| 6 | Calcula distancia al TM y SITP mas cercano (formula Haversine) |
| 7 | Join delitos por localidad (nombre normalizado) |

**Cobertura del dataset final (306 establecimientos):**

| Variable | Cobertura |
|---|---|
| `q_j` (Saber 11) | 279 / 306 |
| `sobre_demanda_j` | 295 / 306 |
| Controles EM2021 | 279 / 306 |
| `dist_transmilenio_m` | 306 / 306 |
| `dist_sitp_m` | 306 / 306 |
| Delitos por localidad | 306 / 306 |
| `dist_parque_m` | pendiente re-descarga |
| `v_j` (visual) | pendiente embeddings |
