# Matching Escolar 

Analisis de mecanismos de asignacion escolar en Bogota con modelos de matching y senales visuales.

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
│       └── embeddings/             <- Embeddings VGG19 por colegio (equipo de imagenes)
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
│   └── 01_build_dataset.py         <- Integra todas las fuentes -> primary/
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

### 6. Imagenes de entorno de colegios - Mapillary

| | |
|---|---|
| **Fuente** | Mapillary Graph API v4 (imagenes publicas con licencia CC) |
| **Script** | `scripts/fetch_mapillary_colegios.py` |
| **Output catalogo** | `data/images/mapillary/mapillary_catalog.csv` |
| **Output resumen** | `data/images/mapillary/resumen_fechas.csv` |
| **Output imagenes** | `data/images/mapillary/*.jpg` |
| **Convencion de nombre** | `{DANE12_EST}_{YYYY-MM-DD}_{image_id}.jpg` |

**Parametros de busqueda:**
- Radio: 100 m alrededor del punto de cada colegio (bounding box)
- Fecha minima: 2021-01-01

**Criterios de seleccion para descarga:**
- Se excluyen imagenes panoramicas (`is_pano = True`): representan el 55 % del catalogo
  y tienen distorsion equirectangular incompatible con VGG19 sin preprocesado adicional.
- Deduplicacion por secuencia: de cada recorrido de captura (`sequence`) se conserva
  unicamente la imagen mas cercana al colegio, eliminando pseudorreplicacion
  (los recorridos tienen ~47 fotogramas consecutivos casi identicos en promedio).

**Cobertura resultante:**
- Imagenes descargadas: ~2 200 (regulares, no redundantes)
- Colegios cubiertos: 244 / 407 (60 %)

**Limitacion documentada — colegios sin indice visual:**
163 colegios quedan fuera del indice visual `v_j`: 37 no tienen ninguna imagen
de Mapillary dentro de 100 m, y 102 solo tienen imagenes panoramicas que se
excluyen por incompatibilidad metodologica con VGG19. Estos colegios se tratan
como `v_j = NaN` en la regresion. Se verifica que la ausencia de cobertura no
este correlacionada sistematicamente con el nivel socioeconomico de la UPZ
(ver Apendice X), de modo que el sesgo de seleccion potencial es aleatorio
respecto a las variables de interes.

### 7. Embeddings visuales de colegios (pendiente - equipo de imagenes)

| | |
|---|---|
| **Responsable** | Otro miembro del grupo |
| **Input** | `data/images/mapillary/*.jpg` |
| **Output esperado** | `data/images/embeddings/embeddings.parquet` |
| **Contenido** | Embeddings VGG19 por colegio, usados para construir el indice visual `v_j` en `scripts/02_visual_index.py` |

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
