# Matching Escolar — Paper AI

Análisis de mecanismos de asignación escolar en Bogotá con modelos de matching y señales visuales.

---

## Estructura del proyecto

```
paper-AI/
├── data/
│   ├── raw/                  ← Datos descargados tal como vienen de la fuente (no modificar)
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
│   ├── primary/              ← (reservado) dataset maestro de colegios con todas las variables
│   ├── processed/            ← Datos limpios y cruzados, output de los scripts
│   │   ├── colegios_dataset.geojson
│   │   ├── saber_bogota_merged.geojson
│   │   ├── demanda_clean.geojson
│   │   └── matriculas_clean.geojson
│   └── images/
│       └── embeddings/       ← Embeddings VGG19 por colegio (entregado por equipo de imágenes)
├── docs/                     ← Diccionarios de variables y documentación de referencia
│   ├── em2021_diccionario.ods
│   └── em2021_variablesadicionales_diccionario.ods
├── models/
│   └── regretnet/            ← Arquitectura y pesos del modelo RegretNet
├── notebooks/                ← Exploración y análisis
├── reports/
│   ├── figures/              ← Gráficas del paper
│   └── paper/                ← Documento final
├── scripts/                  ← Scripts de descarga, limpieza y procesamiento
│   ├── fetch_colegios_geo.py
│   ├── colegios_csv_to_geojson.py
│   ├── fetch_saber11_bogota.py
│   ├── merge_saber_geojson.py
│   ├── demand_capacity_colegios.py
│   └── fetch_em2021_variables.py
├── requirements.txt
└── README.md
```


---

## Extracción de datos

### 1. Directorio de colegios oficiales — SED Bogotá

| | |
|---|---|
| **Fuente** | Secretaría de Educación del Distrito vía [datos abiertos Bogotá](https://datosabiertos.bogota.gov.co) |
| **Script** | `scripts/fetch_colegios_geo.py` |
| **Output raw** | `data/raw/colegios_dataset.csv` |
| **Output processed** | `data/processed/colegios_dataset.geojson` ← generado por `colegios_csv_to_geojson.py` |
| **Contenido** | Nombre, dirección, coordenadas, localidad, UPZ, naturaleza jurídica y características académicas de cada sede educativa oficial |

---

### 2. Resultados Saber 11 — ICFES

| | |
|---|---|
| **Fuente** | ICFES vía datos abiertos Bogotá |
| **Script** | `scripts/fetch_saber11_bogota.py` |
| **Output raw** | `data/raw/saber11_bogota_2020_2022.csv` · `data/raw/pruebassaber2023.geojson` |
| **Output processed** | `data/processed/saber_bogota_merged.geojson` ← generado por `merge_saber_geojson.py` |
| **Contenido** | Puntaje promedio Saber 11 por establecimiento educativo para los años 2020, 2022 y 2023. Variable `q_j` (calidad académica del colegio) |

---

### 3. Demanda y matrícula por colegio — SED Bogotá

| | |
|---|---|
| **Fuente** | Secretaría de Educación del Distrito vía datos abiertos Bogotá |
| **Script** | `scripts/fetch_colegios_geo.py` (descarga conjunta con directorio) |
| **Output raw** | `data/raw/demandacupos04_2024.geojson` · `data/raw/matriculatotal04_2024.geojson` |
| **Output processed** | `data/processed/demanda_clean.geojson` · `data/processed/matriculas_clean.geojson` ← generados por `demand_capacity_colegios.py` |
| **Contenido** | Número de cupos demandados y matrícula total por colegio. Se usa para construir `sobre_demanda_j = demanda / matrícula`, variable dependiente de la regresión que estima `α` |

---

### 4. Encuesta Multipropósito 2021 — SDP / DANE

| | |
|---|---|
| **Fuente** | Secretaría Distrital de Planeación vía [datos abiertos Bogotá](https://datosabiertos.bogota.gov.co/dataset/encuesta-multiproposito-2021-sdp) |
| **Script** | `scripts/fetch_em2021_variables.py` |
| **Output raw** | `data/raw/em2021_encuesta_principal.csv` · `data/raw/em2021_variables_adicionales.csv` |
| **Contenido** | Dos tablas descargadas por separado y cruzadas por `DIRECTORIO` / `directorio_hog`: |

**Encuesta principal** (`em2021_encuesta_principal.csv`):

| Variable | Descripción |
|---|---|
| `DIRECTORIO` | Llave de cruce entre tablas |
| `COD_UPZ_GRUPO` | UPZ de residencia del hogar |
| `COD_LOCALIDAD` | Localidad de residencia |
| `ESTRATO2021` | Estrato de muestreo |
| `NVCBP11AA` | Estrato para tarifa (real del hogar) |
| `FEX_C` | Factor de expansión muestral |
| `NPCHP4` | Nivel educativo más alto del jefe del hogar |
| `NPCJP9AI` | Satisfacción con su educación (escala 0-10) |

**Variables adicionales** (`em2021_variables_adicionales.csv`):

| Variable | Descripción |
|---|---|
| `directorio_hog` | Llave de cruce entre tablas |
| `N_pobre_monetario` | Pobreza monetaria (0/1) |
| `N_pobre_extremo` | Pobreza extrema (0/1) |
| `N_pobre_ipm` | Índice de Pobreza Multidimensional (0/1) |
| `N_ingpc` | Ingreso per cápita del hogar |
| `N_sin_cp` | Índice de capacidad de pago |
| `N_nper` | Número de personas en el hogar |
| `N_gm_educ_hog` | Gasto mensual en educación |
| `N_deficit_cuantitativo` | Déficit cuantitativo de vivienda (0/1) |
| `N_deficit_cualitativo` | Déficit cualitativo de vivienda (0/1) |
| `N_deficit_habitacional` | Déficit habitacional total (0/1) |

> Los diccionarios de variables están en `docs/em2021_diccionario.ods` y `docs/em2021_variablesadicionales_diccionario.ods`.

---

### 5. Variables de control geográficas y de seguridad

Estas fuentes se descargan **manualmente** desde el portal de datos abiertos de Bogotá y se ubican directamente en `data/raw/`. No hay script de descarga automatizado.

| Archivo | Fuente | Contenido |
|---|---|---|
| `estaciones_transmilenio.geojson` | Transmilenio S.A. | Estaciones troncales de TransMilenio. Control de accesibilidad al transporte público |
| `paraderos_sitp.geojson` | Transmilenio S.A. | Paraderos SITP. Control de accesibilidad al transporte |
| `parques_bogota.geojson` | Secretaría Distrital de Planeación | Parques del POT Bogotá. Control de infraestructura verde del entorno |
| `delitos_alto_impacto.geojson` | Secretaría Distrital de Seguridad | Delitos de alto impacto por localidad. Control de seguridad del entorno |

**URL de descarga:** [datosabiertos.bogota.gov.co](https://datosabiertos.bogota.gov.co)

---


