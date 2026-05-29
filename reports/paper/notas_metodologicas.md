# Notas metodológicas del proyecto

Decisiones de diseño, justificaciones, resultados intermedios y referencias del pipeline. Complementa el README (que solo documenta inputs/outputs) y el paper (que resume los hallazgos principales).

---

## Datos

### Encuesta Multipropósito 2021 — variables clave

**Encuesta principal** (`em2021_encuesta_principal.csv`):

| Variable | Descripción |
|---|---|
| `DIRECTORIO` | Llave de cruce entre tablas |
| `COD_UPZ_GRUPO` | UPZ de residencia del hogar |
| `COD_LOCALIDAD` | Localidad de residencia |
| `ESTRATO2021` | Estrato de muestreo |
| `NVCBP11AA` | Estrato para tarifa (real del hogar, 1–6) |
| `FEX_C` | Factor de expansión muestral |

**Variables adicionales** (`em2021_variables_adicionales.csv`):

| Variable | Descripción |
|---|---|
| `directorio_hog` | Llave de cruce (= `DIRECTORIO` + dígito de hogar) |
| `N_pobre_monetario` | Pobreza monetaria (0/1) |
| `N_pobre_extremo` | Pobreza extrema (0/1) |
| `N_pobre_ipm` | Índice de Pobreza Multidimensional (0/1) |
| `N_ingpc` | Ingreso per cápita del hogar (COP mensuales) |
| `N_sin_cp` | Índice de capacidad de pago |
| `N_nper` | Número de personas en el hogar |
| `N_gm_educ_hog` | Gasto mensual en educación |
| `N_deficit_cuantitativo` | Déficit cuantitativo de vivienda (0/1) |
| `N_deficit_cualitativo` | Déficit cualitativo de vivienda (0/1) |
| `N_deficit_habitacional` | Déficit habitacional total (0/1) |

> Diccionarios completos en `docs/em2021_diccionario.ods` y `docs/em2021_variablesadicionales_diccionario.ods`.

**Join entre tablas:** `DIRECTORIO = "166238"` vs `directorio_hog = "1662381"` — el último dígito codifica el número de hogar dentro de la vivienda. Se reconstruye la llave truncando el último carácter de `directorio_hog` antes del merge. Los ~65K hogares sin `COD_UPZ_GRUPO` (municipios de Cundinamarca fuera de Bogotá) se excluyen.

**Proxy de grupo SISBEN `s`** (construida en `05b_expandir_familias.py`): clasificación primaria por `N_ingpc` usando umbrales DANE 2024; fallback por `estrato_real` cuando `N_ingpc` es nulo.

| Grupo | Condición | Umbral LP (per cápita mensual) |
|---|---|---|
| A | Pobreza extrema | `N_ingpc < $227,220` |
| B | Pobreza moderada | `$227,220 ≤ N_ingpc < $460,198` |
| C | Vulnerable | `$460,198 ≤ N_ingpc < $897,987` |
| D | No priorizado | `N_ingpc ≥ $897,987` |

Fuentes: DANE (2024a) para LP extrema/moderada; DANE (2024b) para umbral de vulnerabilidad.

---

### Imágenes GSV — decisiones de diseño

- `N_HEADINGS = 10` headings uniformes (0°, 36°, …, 324°) para capturar fachada desde múltiples ángulos
- Resolución: `640×640 px`, `FOV = 90°`, `PITCH = 0`
- Cobertura: 5,580 imágenes · 558 sedes · 100%
- Costo: ~$39 USD (dentro del crédito mensual gratuito de Google Cloud)
- Reanudable: si `gsv_catalog.csv` existe, salta imágenes ya descargadas
- Modo prueba: `MODO_MUESTRA = True` en `gsv_config.py` → 5 sedes, catálogo guardado como `gsv_catalog_muestra.csv`

**Por qué GSV sobre Mapillary:** GSV usa cámaras calibradas a altura estandarizada (~2.5 m) con resolución y encuadre consistentes. Mapillary (crowdsourced) cubre solo 292/558 sedes (52%) tras filtros y tiene calidad variable. El ruido de cámara/ángulo contaminaba los embeddings VGG19.

**Cobertura Mapillary** (fuente secundaria / exploración):
- Radio: 100 m, fecha mínima 2020-01-01, ángulo cámara→colegio ≤ 90°
- Deduplicación por secuencia y espacial (10 m mínimo)
- Catálogo: ~89,856 imágenes · 292/558 sedes

---

## Transformaciones — justificaciones

### T1. Directorio SED → GeoJSON

Filtra sector **Oficial** únicamente (el análisis se restringe a oferta pública). Convierte `coord_x`/`coord_y` con coma decimal a geometría Point EPSG:4326. La geometría es la base espacial para calcular distancias a TransMilenio/parques y para asignar UPZ a estudiantes sintéticos.

### T3. Demanda y matrícula → `sobre_demanda_j`

`sobre_demanda_j = DTotal / TMATRIC_GE` es la variable dependiente de la regresión que estima α̂. Un colegio con alta sobredemanda es más deseado de lo que su calidad justificaría — posible señal de sesgo visual.

> **Nota de datos:** el campo `CM**TOTAL` en el GeoJSON de delitos es el total de Bogotá entera, no por localidad. Los valores correctos por localidad se obtienen sumando las columnas `CM**[YY]CONT`.

### T4. EM2021 → controles por UPZ

Todas las métricas se calculan con `FEX_C` como peso. Sin ponderar, las UPZs con mayor intensidad de muestreo quedarían sobrerepresentadas.

`COD_UPZ_GRUPO` en EM2021 agrupa UPZs pequeñas bajo un mismo código — el cruce con el directorio de colegios puede dejar algunos sin match directo. Esta limitación se documenta en el paper.

**Variables construidas y su rol:**

| Variable output | Rol en el modelo |
|---|---|
| `tasa_pobreza_monetaria` / `tasa_pobreza_extrema` | Control socioeconómico de la UPZ |
| `tasa_ipm` | Control multidimensional de pobreza |
| `ingreso_percapita_promedio` | Control de ingreso |
| `capacidad_pago_promedio` | Control de capacidad adquisitiva |
| `tamano_hogar_promedio` | Control demográfico |
| `gasto_educ_promedio` | Proxy de valoración educativa — hogares que más gastan pueden ser menos susceptibles al sesgo visual |
| `tasa_deficit_cuantitativo` / `tasa_deficit_cualitativo` | Separa "el barrio se ve mal" de "el colegio se ve mal" |
| `pct_estrato_1` … `pct_estrato_6` | Distribución de estrato en la UPZ — control y calibración de datos sintéticos |

### T6. Competencia privada

La intensidad competitiva local puede moderar el efecto visual. En localidades con alta oferta privada los hogares tienen más opciones y pueden ser más sensibles a señales de calidad. Variable descriptiva en EDA (Figura 4) y control en regresión.

### T7. Embeddings VGG19 — por qué no se agrega aquí

Promediar los vectores por sede/establecimiento antes de LDA suaviza los embeddings y concentra el espacio, produciendo colapso de tópicos. La agregación ocurre *después*, en espacio ya interpretable (proporciones de tópicos).

### T8. Selección de d PCA

**Resultados del diagnóstico sobre los datos actuales:**

| Test | Resultado | Estado |
|---|---|---|
| Varianza por feature | Var mín: 0.000343, 0% features nulas | ✅ OK |
| Similitud coseno (imágenes) | Media: 0.61 | ✅ OK |
| PCA — codo | d = 18 (∼55% varianza) | referencia |
| PCA — umbral 70% | d = 37 | — |
| PCA — umbral 80% | d = 68 | ✅ **seleccionado** |
| PCA — umbral 90% | d = 141 | — |

**Criterio:** umbral más alto ≥ 80% dentro de 150 PCs. El codo (d=18) se reporta como referencia pero captura solo ~55% de la varianza. El tramo 70%→80% cuesta 31 PCs con 0.32% de varianza por PC (eficiente); el tramo 80%→90% cuesta 73 PCs con 0.14%/PC (ineficiente). Con d=68: N/d = 82, cómodo para NMF.

### T9. LDA → descartado por colapso

LDA colapsa en este dataset (std=0 en todos los tópicos para cualquier K). El prior de Dirichlet domina cuando los embeddings VGG19 de fachadas escolares son suficientemente homogéneos entre imágenes.

**Por qué NMF sobre imágenes individuales y no sobre establecimientos:** con 5,580 imágenes la relación N/d = 82. Corriendo sobre los 306 vectores promediados: N/d = 4.5 → colapso (todos los tópicos peso uniforme 1/K). NMF no asume distribuciones probabilísticas y produce tópicos parts-based con varianza real (std ≈ 0.05–0.12).

### T11. Segmentación Cityscapes — grupos y estadísticas

| Grupo | Clases Cityscapes | Media (558 colegios) |
|---|---|---|
| `infraestructura_vial` | road, sidewalk | 33.1% |
| `edificacion` | building | 25.9% |
| `referencia` | sky, person, rider | 23.9% |
| `vegetacion` | vegetation, terrain | 9.6% |
| `cerramiento` | wall, fence | 3.7% |
| `vehiculos` | car, truck, bus, train, moto, bici | 2.7% |
| `mobiliario_urbano` | pole, traffic light, traffic sign | 1.0% |

`referencia` se excluye en regresión para evitar multicolinealidad perfecta (las 7 suman 1). Velocidad: ~6 min para 5,580 imágenes con GPU RTX 4070.

**Por qué Cityscapes sobre un modelo genérico:** entrenado en escenas urbanas, produce coeficientes interpretables — "10pp más de vegetación → X% más sobredemanda" (Suel et al. 2019).

### T12. Features perceptuales CLIP

Frases fijas para garantizar replicabilidad:

| Dimensión | Frase positiva | Frase negativa |
|---|---|---|
| `mantenimiento` | "a school building with a clean and well-maintained facade" | "a school building with a deteriorated and neglected facade" |
| `vegetacion_percibida` | "a school surrounded by trees and green areas" | "a school with no vegetation or green spaces around it" |
| `accesibilidad` | "a school with a welcoming and open entrance" | "a school with a closed, walled-off and unwelcoming entrance" |
| `seguridad_percibida` | "a school in a safe and calm street environment" | "a school in a dangerous and chaotic street environment" |

Scores medios (558 colegios): mantenimiento −0.012 · vegetacion −0.046 · accesibilidad −0.028 · seguridad −0.006. Velocidad: ~54 segundos con GPU RTX 4070.

**Por qué CLIP complementa Cityscapes:** Cityscapes mide *qué hay físicamente*; CLIP mide *cómo se percibe* — deterioro, apertura, seguridad. Cityscapes y CLIP no sobreviven el proceso de selección de variables (Lasso M1) — la señal visual queda capturada en los tópicos latentes NMF.

### T13. Imputación espacial

25 columnas con NaN, 661 valores faltantes. Radio de vecindad: 2 km (haversine, BallTree). Fallback a mediana global en 40 casos sin vecinos. Resultado: 306 establecimientos sin NaN.

Los colegios cercanos comparten entorno socioeconómico; la media de vecindario es mejor estimador que la media global para variables de UPZ (pobreza, ingreso) que varían fuertemente por zona.

---

## Regresión — modelo seleccionado (Ridge M1)

**Variable dependiente:** `log(sobre_demanda_j)`  
**Controles M0:** `puntaje_icfes_promedio` · `tasa_pobreza_monetaria` · `ingreso_percapita_promedio` · `dist_sitp_m` · `pct_no_oficial` · `hurto_personas` · `homicidios` · `n_oficiales_localidad` · `estrato_2`…`estrato_6`

**Estimadores comparados:** OLS · Ridge · LASSO · ElasticNet — todos con `Pipeline(StandardScaler)` para evitar data leakage, CV k=5.

OLS y Ridge: RMSE_cv ~0.14–0.16, sobreajuste (N=301, p hasta 23). LASSO y ElasticNet generalizan mejor fuera de muestra.

**Modelo seleccionado: Ridge M1** — R²_adj=0.203, Spearman(CV)=0.466, 408 obs. Ver tabla completa de coeficientes en `reports/tabla_ridge_m1.tex`.

**Features seleccionadas por M1-Lasso (referencia, 9 activas):**

| Variable | Coef | Tipo |
|---|---|---|
| `estrato_4` | +0.016 | Control |
| `topic_1` | +0.009 | Visual NMF |
| `topic_2` | −0.006 | Visual NMF |
| `estrato_3` | −0.003 | Control |
| `hurto_personas` | +0.002 | Control |
| `pct_no_oficial` | +0.002 | Control |
| `puntaje_icfes_promedio` | −0.002 | Control |
| `estrato_5` | −0.002 | Control |
| `topic_6` | −0.001 | Visual NMF |

topic_1 (+) y topic_2 (−) son los tópicos visuales con mayor señal. El coeficiente de `estrato_4` es el más alto — colegios en zonas de estrato 4 tienen mayor sobredemanda relativa. Cityscapes y CLIP no sobreviven ningún estimador.

**Pendiente:** identificar qué representan visualmente topic_1, topic_2 y topic_6 (inspección de imágenes con peso alto en cada tópico).

---

## Pipeline de simulación

### S1. Familias escolares — EM2021

Filtra personas 5–17 años en institución oficial (`NPCHP2=1`, `NPCHP12=1`), agrega a nivel hogar (`DIRECTORIO`) y cruza con encuesta principal para traer UPZ, localidad y estrato real (`NVCBP11AA`).

**Resultado:** 21,643 hogares con hijos en colegio oficial.

### S2. Ubicación en manzana y distancias

Para cada familia: identifica UPZ → selecciona manzana con mismo `estrato_real` (fallback al más cercano si no hay coincidencia exacta) → punto aleatorio en bbox del polígono (rejection sampling) → distancias Haversine a los 303 colegios con embeddings NMF.

**Resultado:** 13,568 familias ubicadas (87.5% de match — el 12.5% restante son UPZs 8xx periféricas sin polígono individual). Matriz de distancias: shape (13,568 × 303), min=0.01 km, max=32.7 km.

### S3. Modelo de utilidad y rankings

```
u_ij = β·X_j  −  α(y_i) · ln(1 + d_ij)  +  ε_ij
```

| Componente | Descripción |
|---|---|
| `β·X_j` | Score de calidad predicho por Ridge M1 — homogéneo entre hogares |
| `α(y_i) · ln(1+d_ij)` | Penalización distancia heterogénea por ingreso per cápita del hogar |
| `ε_ij ~ Gumbel(0,1)` | Ruido logístico (seed=42) |

**Penalización por distancia** — función continua del ingreso per cápita; γ calibrado imponiendo α(p10)/α(p90) = 3× (ratio de Hastings et al. 2009 entre p10 y p90 de la distribución de ingreso bogotana), donde p10 = $80,000 y p90 = $1,400,000 (ratio y_p90/y_p10 = 17.5×), de modo que γ = log(3)/log(17.5) = 0.384:

```
α(y_i) = α₀ · (ȳ / y_i)^γ
```

donde `y_i` = `N_ingpc` del hogar _i_ (COP mensuales), `ȳ` = ingreso medio muestral (≈ $507,000 según `familias_expandidas.parquet`), `α₀ = 1.0` y `γ = log(3)/log(17.5) = 0.384`. Por construcción, `α(ȳ) = α₀ = 1.0`.

**¿Por qué ingreso continuo en lugar de estrato?**

- **Estrato ≠ ingreso:** el estrato en Colombia clasifica la infraestructura del *barrio* para subsidiar servicios públicos, no el ingreso del hogar. La correlación empírica estrato–ingreso es positiva pero moderada; dos hogares en el mismo estrato pueden tener ingresos muy distintos.
- **Heterogeneidad intra-grupo:** el estrato discreto colapsa la variación continua del ingreso en seis categorías, perdiendo toda la dispersión dentro de cada grupo. Con ingreso continuo, cada hogar recibe su propia penalización.
- **Fundamento teórico:** la forma `(ȳ/y_i)^γ` se motiva directamente por la teoría de costo de oportunidad — el costo de transporte pesa *proporcionalmente más* cuando el presupuesto es menor. La especificación con estrato era una aproximación discreta de esta relación.

**α implícito por percentil de ingreso** (`ȳ ≈ $507,000`, `α(y) = 1.0 · (507,000/y)^0.384`):

| Percentil | y_i (COP/mes) | ȳ/y_i | α(y_i) |
|---|---|---|---|
| p10 | $80,000 | 6.34 | 1.74 |
| p25 | $180,000 | 2.82 | 1.40 |
| p50 | $380,000 | 1.33 | 1.11 |
| ȳ (media) | $507,000 | 1.00 | 1.00 |
| p75 | $720,000 | 0.70 | 0.86 |
| p90 | $1,400,000 | 0.36 | 0.58 |

**¿Por qué α₀ = 1.0?**

La familia con ingreso igual a la media acepta ir aproximadamente 1.7 km extra por +1 desviación estándar de calidad académica. Con la especificación log-distancia: `α(ȳ) · log(1+d) = 1` → `d = e^(1/α₀) − 1 = e^(1/1.0) − 1 ≈ 1.72 km`. Esto es consistente con la movilidad observada en Bogotá, donde la mayoría de familias elige colegios dentro de un radio de 2–3 km de su residencia.

**Choice set:** colegios de la misma localidad. Excepción: La Candelaria (localidad 17) puede elegir en localidades 3 (Santa Fe), 14 (Los Mártires) y 15 (Antonio Nariño).

**Resultado:** rankings top-20 por familia. Choice set promedio: 23.3 colegios.

---

## Pipeline de matching

### M1. Datos reales — diseño

- Capacidad = `round(matricula_total / 13)`, mínimo 5 (cohort anual estimado)
- Prioridad = distancia Haversine (réplica del criterio SED Bogotá)
- Choice set = localidad (heredado de `06_preferencias.py`)

**Resultados (datos reales — población expandida FEX_C, α continuo por N_ingpc):**

| Mecanismo | Asignados | Rank medio | Blocking pairs | corr(ingreso, a_j) | corr(ingreso, v_j) | Rechazo A-B (%) | Rechazo total (%) |
|---|---|---|---|---|---|---|---|
| Boston (BM) | 97,965 | **1.740** | **8,072** | 0.20 | 0.14 | 1.9% | 1.9% |
| DA (Gale-Shapley) | 97,970 | 2.100 | **0** ✓ | 0.21 | 0.13 | 2.4% | 1.9% |
| SED-lex | 97,968 | 1.872 | **0** ✓ | **0.17** | **0.13** | **0.0%** | 1.9% |

**Lectura:**

- **Eficiencia (rank medio):** BM es el más eficiente (1.74 ≈ primera opción promedio), DA el más costoso (2.10). SED-lex intermedio (1.87). El costo de strategy-proofness es ~0.36 posiciones de rank (BM→DA).
- **Estabilidad:** solo DA y SED-lex son estables (BP=0). BM genera 8,072 blocking pairs — pares familia-colegio donde ambos preferirían estar juntos pero no están asignados.
- **Equidad (corr ingreso-atractivo):** SED-lex tiene la correlación más baja (0.17) — las familias de menor ingreso acceden a colegios relativamente más atractivos. BM y DA ≈ 0.20-0.21.
- **Sesgo visual:** prácticamente idéntico entre mecanismos (0.13-0.14) — el mecanismo no suprime ni amplifica el peso de lo visual en la asignación final.
- **Rechazo grupos vulnerables (SISBEN A+B):** SED-lex = **0.0%** por diseño (prioridad lexicográfica). DA = 2.4% (peor que BM por mayor competencia inducida por strategy-proofness). Rechazo total idéntico (1.9%) — la capacidad del sistema es la restricción binding.

**Resultado central:** SED-lex domina en equidad e inclusión sin sacrificar eficiencia sustancialmente (+0.13 rank vs BM). El mecanismo actual de la SED (prioridad por distancia+vulnerabilidad) ya está bien diseñado en términos de equidad — el problema no es el mecanismo sino el sesgo en las preferencias declaradas.

### M2. Experimento sintético — modelo de utilidad

```
CON sesgo: u_ij  = q_j_std + (α̂ + γ_s_i) · v_j + ε_ij
SIN sesgo: u0_ij = q_j_std + ε_ij
```

**Construcción de v_j:**
```
Paso 1: OLS  →  log(SD_j) = β₀ + β₁·q_j_std + e_j
Paso 2: v_j_raw = e_j        (residuo = demanda no explicada por calidad)
Paso 3: v_j = z-score(v_j_raw)    ⟹  corr(v_j, q_j) = 0.000 por construcción
```

**Parámetros calibrados:**

| Parámetro | Valor | Fuente |
|---|---|---|
| `alpha_hat` (α̂) | 0.08793 | OLS `log(SD) ~ v_j` (p≈0) |
| `gamma_s` | γ₀/s, **γ₀=1** (caso base) | Forma funcional Hastings & Weinstein (2008) |
| `sigma` (σ) | 1.0 | Calibrado para competencia real |
| N estudiantes | 10,000 | Distribución real EM2021 |
| M colegios | 100 | Distribución uniforme en [0,1]² |

Validación: `corr(estrato, v_j_top1)` pasa de 0.004 (sin sesgo) a 0.030 (con sesgo) — amplificación ×7 del sesgo visual en la primera preferencia por estrato.

### M3. Resultados — 6 condiciones (BM/DA/SED × sesgo/verdad)

| Condición | corr(ingreso, q_j) | corr(ingreso, v_j) | Rank medio | Blocking pairs |
|---|---|---|---|---|
| BM-bias  | +0.027 | −0.092 | 17.82 | 3,229 |
| BM-true  | +0.006 | −0.002 | 16.53 | 2,922 |
| DA-bias  | −0.006 | −0.027 | 48.03 | 0 ✓ |
| DA-true  | −0.011 | −0.001 | 47.14 | 0 ✓ |
| SED-bias | −0.507 | −0.446 | 28.35 | 0 ✓ |
| SED-true | −0.538 | −0.035 | 27.28 | 0 ✓ |

**Δcalidad BM** (bias vs true): 0.02. **Δsesgo SED** (bias vs true): 0.41.

Hallazgos: BM genera ~3,200 blocking pairs; DA y SED son estables. SED amplifica el sesgo visual (Δcorr = 0.41) porque las familias vulnerables priorizadas eligen con sesgo sin fricción geográfica — el desempate por lotería (sintético) no suprime el sesgo; en datos reales lo suprime el desempate por distancia.

### M4. Robustez a γ₀

Grid: `γ₀ ∈ {0.25, 0.50, 0.75, 1.00, 1.25, 1.50}`, paso 0.25, mundo fijo seed=42. Δrank, Δq y Δv_j crecen monótonamente con γ₀. El ordenamiento proxy-SED > BM ≈ DA se mantiene para todo γ₀ en el rango — resultados cualitativos robustos.

---

## Referencias

### Mecanismos de matching y school choice
- Abdulkadiroğlu, A. & Sönmez, T. (2003). School Choice: A Mechanism Design Approach. *American Economic Review*, 93(3), 729–747.
- Gale, D. & Shapley, L. (1962). College Admissions and the Stability of Marriage. *American Mathematical Monthly*, 69(1), 9–15.

### Preferencias y distancia en school choice
- Hastings, J., Kane, T. & Staiger, D. (2009). Heterogeneous Preferences and the Efficacy of Public School Choice. *NBER Working Paper* 12145. [Fuente del ratio α(p10)/α(p90)=3×; γ=0.384 se calibra imponiendo este ratio sobre la distribución bogotana (y_p90/y_p10=17.5×)]
- Burgess, S., Greaves, E., Vignoles, A. & Wilson, D. (2015). What Parents Want: School Preferences and School Choice. *Economic Journal*, 125(587), 1262–1289. [Gradiente de distancia heterogéneo por estrato, especificación log-distancia]
- Gallego, F. & Hernando, A. (2009). School Choice in Chile: Looking at the Demand Side. *Pontificia Universidad Católica de Chile*, Working Paper. [Contexto LatAm: restricción geográfica como principal determinante en familias de bajos ingresos]
- Einav, L. & Levin, J. (2010). Empirical Industrial Organization: A Progress Report. *Journal of Economic Perspectives*, 24(2), 145–162. [Especificación log-log de parámetros heterogéneos]

### Mecanismos aprendidos
- Dütting, P., Feng, Z., Narasimhan, H., Parkes, D. & Ravindranath, S. (2019/2023). Optimal Auctions through Deep Learning. *ICML 2019 / Journal of the ACM*, 70(1).

### Features visuales urbanas
- Naik, N., Raskar, R. & Hidalgo, C. (2017). Computer Vision Uncovers Predictors of Physical Urban Change. *PNAS*, 114(29), 7571–7576.
- Dubey, A., Naik, N., Parikh, D., Raskar, R. & Hidalgo, C. (2016). Deep Learning the City: Quantifying Urban Perception at a Global Scale. *ECCV 2016*.
- Suel, E., Bhatt, S., Brauer, M., Flaxman, S. & Bhatt, S. (2019). Measuring Individual Wellbeing from Urban Street Images. *Scientific Reports*, 9, 15851.

### Visión computacional y embeddings
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
- Lee, D. & Seung, H. (2001). Algorithms for Non-negative Matrix Factorization. *NeurIPS 2000*.
- Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015* (VGG19).

### Econometría
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society B*, 58(1), 267–288.
- Zou, H. & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society B*, 67(2), 301–320.
- Long, J. & Ervin, L. (2000). Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model. *The American Statistician*, 54(3), 217–224.

### Estadísticas de pobreza y vulnerabilidad — DANE
- DANE (2024a). Boletín técnico: Pobreza Monetaria Colombia 2024. Umbrales LP extrema ($227,220) y LP moderada ($460,198) per cápita mensual.
- DANE (2024b). Comunicado de prensa: Clases Sociales Colombia 2024. Umbral de vulnerabilidad ($897,987) per cápita mensual.

---

## Extensiones pendientes — Semana 2026-05-22

### Tarea 1. Interpretación de tópicos NMF

Revisar `reports/figures/topic_{1..8}_top8.png` — las 8 imágenes con mayor peso en cada tópico. Asignar etiqueta semántica a cada tópico (e.g., "fachada bien mantenida", "entorno deteriorado", "vegetación densa", "vía sin pavimentar"). Documentar las etiquetas con justificación visual. Esto es prerequisito para la estimación de demanda — los tópicos entran como regresores individuales y sus coeficientes deben ser interpretables.

**Decisión de dimensionalidad: K=6 como especificación principal.**
- `gsv_nmf_K6.parquet` ya existe (408 × 7). No requiere regenerar embeddings.
- Reduce parámetros en BLP: 12 (6 β_k + 6 π_k) vs 16 con K=8. Más cómodo con N=306 colegios.
- Verificar que la señal visual se mantiene: correr `04a_regresion.py` con K=6 y confirmar que algún tópico es significativo.
- K=8 como especificación de robustez.
- También interpretar los tópicos de K=6 (generar `topic_{1..6}_top8.png` si no existen).

### Tarea 2. Estimación estructural de demanda

**Objetivo:** Reemplazar los parámetros calibrados (α₀=1, γ=0.384) con parámetros estimados de los datos reales de Bogotá.

**2a. Berry inversion (logit puro)**
Reformular la regresión actual (`04a_regresion.py`) como estimación estructural:
- Definir market shares: `s_j = matrícula_j / M_t` donde `M_t` = población escolar del mercado (localidad)
- Definir outside option: `s_0 = 1 - Σ s_j` (familias que van a privado o no asisten)
- Invertir: `log(s_j / s_0) = δ_j = β_q·q_j + Σ_k β_k·topic_k_j + controles + ξ_j`
- Los 6 tópicos NMF (K=6, especificación principal) entran como regresores individuales, NO como índice v_j agregado. K=8 como robustez.
- Reportar: elasticidades de distancia, WTP por calidad vs señal visual

**2b. Logit mixto (BLP/PyBLP)**
Estimar heterogeneidad en preferencias por ingreso:
```
u_ij = δ_j + α_base·d_ij + π·(ingreso_i × d_ij) + Σ_k π_k·(ingreso_i × topic_k_j) + ε_ij
```
- `agent_data`: familias EM2021 con ingreso (`N_ingpc`) como demográfico observado
- `product_data`: colegios × mercado (localidad) con tópicos NMF, q_j, controles
- `π` captura cómo el ingreso modifica sensibilidad a distancia — reemplaza la calibración `α(y_i) = (ȳ/y_i)^0.384`
- `π_k` captura qué tópicos visuales pesan más para familias de distintos ingresos — esto es el sesgo visual desagregado
- Tres especificaciones de robustez: (1) K=6 tópicos individuales (principal), (2) K=8 tópicos individuales, (3) PCA → v_j escalar
- Herramienta: PyBLP (Conlon y Gortmaker, 2020)
- No hay endogeneidad de precio (educación oficial es gratuita); la distancia es exógena

**Nota sobre endogeneidad:** No requerimos IVs para precio. El "precio" de acceso es la distancia, que es exógena condicional en la ubicación de la familia. Los instrumentos BLP clásicos (cost-shifters, Hausman) no aplican. Sin embargo, puede haber endogeneidad en ξ_j (calidad no observada): si colegios con buena gestión interna también se ven mejor físicamente, los β̂_k de los tópicos NMF estarían sesgados. **Estrategia:** empezar sin IV (reportar como asociación), luego como robustez instrumentar con IVs de diferenciación (Gandhi y Houde, 2020) — distancia en espacio de características entre colegios competidores. PyBLP los calcula con `pyblp.build_differentiation_instruments()`.

**Referencia clave:** Curso de Jeff Gortmaker y Ariel Pakes (Mixtape Sessions). Documentación en `reports/paper/demand_estimation_*.md`.

### Tarea 3. Monte Carlo con parámetros estimados

**Objetivo:** Reemplazar los parámetros calibrados en `08_datos_sinteticos.py` con los α̂, β̂, π̂ de la estimación estructural y validar robustez con múltiples réplicas.

**Diseño:**
- Recalibrar DGP sintético con parámetros estimados en Tarea 2
- 100 réplicas (seeds 1–100) × BM/DA/SED para cada configuración
- Reportar media ± IC 95% de: delta_rank, delta_q(ingreso), delta_v(ingreso)
- Robustez a γ₀ ∈ {0.25, 0.50, 0.75, 1.00, 1.25, 1.50} también con IC

**Dos escenarios de evaluación:**
1. **Correlación realista** (v_j, q_j correlacionados como en Bogotá) — valida que los resultados se mantienen en un mundo realista
2. **Ortogonal** (v_j ⊥ q_j por construcción) — identificación limpia del sesgo visual puro

**Métricas de equidad basadas en ingreso continuo** (`N_ingpc`), no en estrato ni grupo SISBEN. Justificación: SISBEN se deriva del ingreso; el ingreso continuo es estrictamente más informativo y captura variación within-group.

**Dependencia:** Requiere parámetros estimados de Tarea 2.

### Tarea 4. Mecanismo aprendido — Weighted Polytope Rule (WP-Rule)

**Objetivo:** Diseñar un mecanismo de asignación que sea estable por construcción y optimice equidad visual, superando a BM/DA/SED.

**Fundamento teórico (Narasimhan, Agarwal y Parkes, 2016; Roth, 1984):**
- Todo emparejamiento estable es un vértice de un politopo convexo (Teorema de Roth)
- Maximizar una función lineal sobre ese politopo siempre produce un vértice → matching estable garantizado
- Los pesos de esa función lineal se pueden aprender con datos

**Formulación:**
Para cada par (familia i, colegio j), se define un peso:
```
λ_ij(W) = a_ij · rank(familia i prefiere colegio j) + b_ij · rank(colegio j prefiere familia i) + c_ij
```
El matching WP-Rule resuelve:
```
y* = argmax_{y ∈ Ω_estable} Σ_ij λ_ij · y_ij
```
donde `Ω_estable` se define por: (i) racionalidad individual, (ii) no blocking pairs.

Casos especiales: `a` domina → DA student-optimal; `b` domina → DA school-optimal. Valores intermedios → tradeoff aprendido.

**Entrenamiento (StructSVM sobre sintéticos correlacionados):**
- Generar 1000 instancias de preferencias con DGP calibrado (parámetros estimados, v_j y q_j correlacionados como en Bogotá)
- Para cada instancia, calcular matching "ideal" no estable (Hungarian/LP que maximiza welfare − penalización·|corr(ingreso, v_j)|)
- Entrenar W = [a, b, c] para minimizar distancia de Hamming entre WP-Rule(W) y el matching ideal
- El modelo aprende la estructura realista de correlaciones

**Evaluación (sobre sintéticos ortogonales, 100 réplicas MC):**
- Aplicar los pesos λ* aprendidos sobre escenario ortogonal (v_j ⊥ q_j)
- Comparar BM/DA/SED/WP-learned
- Aquí cualquier corr(ingreso, v_j) es sesgo puro — identificación limpia
- Métrica: ¿WP reduce sesgo visual sin empeorar q_j asignado?

**Aplicación (datos reales de Bogotá):**
- Aplicar pesos λ* sobre las 97,968 familias expandidas
- Tabla comparativa final: BM vs DA vs SED-lex vs WP-learned
- Viabilidad operacional: el mecanismo es un LP, computable en segundos

**Implementación:** LP con PuLP/scipy para el matching por instancia; StructSVM (o red neuronal) para aprender W.

**Referencia:** Narasimhan, H., Agarwal, S. y Parkes, D. (2016). Automated Mechanism Design without Money via Machine Learning. Documentación en `reports/paper/maching_learning_matching.md`.

---

### Tarea 4b. Ajustes a WP-Rule (post implementación Jhoan, 2026-05-28)

**Contexto:** La rama `jhoan-svm` implementa 09c (WP-Rule + StructSVM), 09d (unicidad sintéticos), 09e (unicidad datos reales). Resultado principal: el retículo de matchings estables colapsa a un punto (brecha = 18-25 de 113,857 asignadas en datos reales), por lo que DA ≈ SED ≈ WP bajo prioridad-distancia. El sesgo vive en las preferencias, no en el mecanismo.

**Ajustes pendientes (en orden):**

#### 4b.1 Función objetivo corregida

La función target actual solo penaliza `corr(ingreso, v_j)`:

```
w_ij = q_j − μ · y_i^c · v_j
```

La función correcta incorpora tres objetivos:

```
w_ij = q_j − μ₁ · y_i^c · v_j − μ₂ · y_i^c · q_j
```

| Término | Objetivo | Meta |
|---|---|---|
| q_j | Maximizar calidad total asignada | ↑ |
| −μ₁ · y_i^c · v_j | Romper sesgo visual | corr(ingreso, v_j) → 0 |
| −μ₂ · y_i^c · q_j | Equidad compensatoria en calidad | corr(ingreso, q_j) ≤ 0 |

**Justificación:** el diagnóstico es el sesgo visual, pero el daño se mide en calidad. Ambas correlaciones importan. La equidad en calidad permite valores negativos (familias pobres accediendo a mejores colegios = compensación).

**Calibración de μ₁, μ₂:** no son arbitrarios. Calibrar μ₁ tal que corr(ingreso, v_j) ≈ 0 en el matching ideal, y μ₂ tal que corr(ingreso, q_j) alcance un target de equidad compensatoria. Complementar con análisis de sensibilidad barriendo (μ₁, μ₂) para mostrar la frontera eficiencia-equidad.

#### 4b.2 Features de pesos WP expandidos

Actual: `λ_ij(W) = a·rank_i + b·rank_j + d·(y_i^c · v_j)` (3 pesos)

Corregido: `λ_ij(W) = a·rank_i + b·rank_j + d₁·(y_i^c · v_j) + d₂·(y_i^c · q_j)` (4 pesos)

El StructSVM aprende (a, b, d₁, d₂) para acercarse al target corregido.

#### 4b.3 Baselines WP

Agregar a la comparativa:
- **WP equal-weights** (a=1, b=1, d₁=0, d₂=0): baseline sin entrenar. Si DA = WP-equal, confirma retículo trivial.
- **WP paper-original** (a, b aprendidos, d₁=d₂=0): especificación del paper de Narasimhan sin features de equidad. Baseline teórico.

#### 4b.4 Parámetros del DGP alineados con BLP

Actual: `ALPHA_HAT = 0.088, GAMMA0 = 1.0` (constantes ad-hoc, utilidad tipo power-law por estrato).

Corregido: usar los θ estimados del BLP:
- π₁ = −0.028 (interacción ingreso × seguridad percibida)
- λ₀ = +0.024 (penalización base distancia)
- λ₁ = −0.094 (interacción ingreso × distancia)

La utilidad sintética debe ser: `u_ij = δ_j + π₁·y_i·seg_z_j + λ₀·log(1+d_ij) + λ₁·y_i·log(1+d_ij) + ε_ij`, consistente con 06_preferencias.py.

#### 4b.5 Escala de entrenamiento

Actual: 24 familias × 6 colegios (puede ser insuficiente para generalizar).

Probar: 100×20 como mínimo, idealmente 500×50. Los pesos aprendidos en mercados pequeños pueden no transferir a 537K×382.

#### 4b.6 Restricción de acceso garantizado para bajos ingresos

Análogo al proxy SED que prioriza por grupo SISBEN: agregar una restricción dura al LP de WP que **garantice** asignación a familias vulnerables.

**Implementación (constraint adicional en el politopo):**
```
∀ i con SISBEN ∈ {A, B}: Σ_j x_ij = 1
```

Toda familia de grupo SISBEN A o B **debe** ser asignada. No es un peso aprendible — es una garantía de acceso.

**Variante soft (feature en los pesos):**
```
λ_ij(W) = a·rank_i + b·rank_j + d₁·(y_i^c · v_j) + d₂·(y_i^c · q_j) + e·𝟙[SISBEN_i ∈ {A,B}]
```
El StructSVM aprende e > 0 → prioriza asignar a SISBEN bajo. Pero la versión hard es preferida para el paper porque refleja la política real de la SED.

**Nota:** esta restricción interactúa con 4b.6 (unicidad bajo SISBEN). Si el acceso garantizado cambia la estructura del politopo, puede abrir el retículo.

#### 4b.7 Unicidad bajo prioridades SED

09e solo verifica unicidad con prioridad-distancia pura. Correr también con prioridad SED (grupo SISBEN + distancia). Si bajo prioridad SISBEN el retículo se abre (múltiples matchings estables), WP tendría espacio para optimizar — cambiando el resultado principal.

Grupos SISBEN calculados a partir de ingreso (umbrales DANE ya implementados en 09c):
```
A: ingreso < 227,220
B: 227,220 ≤ ingreso < 460,198
C: 460,198 ≤ ingreso < 897,987
D: ingreso ≥ 897,987
```

#### 4b.8 Paths y limpieza

Todos los paths en 09c/09d/09e están hardcodeados a `/content/edu-matching-paper/` (Colab). Cambiar a paths relativos con `Path(__file__).resolve().parent.parent` como el resto del repo.

**Orden de ejecución:** 4b.8 (limpieza) → 4b.4 (DGP) → 4b.1 (target) → 4b.2 (features) → 4b.6 (acceso garantizado) → 4b.3 (baselines) → 4b.5 (escala) → 4b.7 (unicidad SED).

---

### Resultados 4b — implementación y verificación (Jhoan Fuentes, 2026-05-29)

> Esta subsección documenta los resultados de ejecutar el plan 4b anterior. Todo
> el código quedó en `scripts/09c_wp_rule.py` (4b.1–4b.4, 4b.6, 4b.8),
> `scripts/09g_uniqueness_sed.py` (4b.7) y `scripts/09h_escala_reticulo.py` (4b.5).
> El orden de ejecución se reordenó: 4b.7 se corrió primero por ser el único
> experimento capaz de cambiar la conclusión (mueve el conjunto factible, no solo
> el objetivo).

**4b.1 + 4b.2 (objetivo de 3 términos + 4ª feature).** Implementados en 09c. El
StructSVM aprende d₂ < 0 (compensación por calidad) y la métrica corr(ingreso, q)
se acerca a 0 bajo el objetivo corregido. Pero el *matching* no cambia: WP coincide
con DA. Razón: el objetivo vive sobre el politopo de estables; si el politopo es
casi un punto, el argmax es invariante a la función que se maximice encima.

**4b.3 (baselines).** WP equal-weights (a=b=1, d=0, sin entrenar), WP paper-original
(d=0) y WP full coinciden con DA (≠DA ≈ 0, BP=0) en el mundo ortogonal. Que la WP
*sin entrenar* ya sea DA prueba que la coincidencia es geometría del politopo, no
producto del entrenamiento.

**4b.4 (DGP alineado con BLP).** El mercado de entrenamiento de 09c ahora muestrea
colegios/familias reales y usa la utilidad estructural del IV-BLP
(δ_j + π₁·y·seg_z + λ₀·log1p(d) + λ₁·y·log1p(d) + Gumbel). El resultado no depende
de los betas (corrobora 09f): WP = DA con cualquier parametrización del entrenamiento.

**4b.5 (escala — 09h).** Diámetro exacto del retículo (DA familia-opt vs colegio-opt,
método de 09e) sobre cientos de mercados. El diámetro *absoluto* crece con N
(0.53 → 1.15 → 2.28 en 24×6, 100×20, 500×50) pero el *relativo* (gap/N) se contrae
(2.21% → 1.15% → 0.46%). El colapso del retículo no es artefacto de N=24. En Bogotá
real (537k, ver 09e/09g) el diámetro es 0.0158% (18/113857).

**4b.6 (acceso garantizado SISBEN).** Implementadas ambas variantes en 09c: hard
(restricción Σ_j x_ij = 1 para SISBEN A/B en el ILP) y soft (feature e·1[SISBEN∈A/B]).
La hard es estable (BP=0, 0 infeasible) en el DGP de entrenamiento, pero su efecto es
nulo porque hay holgura de cupos (acc_AB = 1.000 ya sin la restricción). La restricción
solo mordería bajo escasez — que es el caso de los datos reales, no del entrenamiento.

**4b.7 (unicidad bajo prioridades SED — 09g).** El experimento decisivo. Bajo
prioridad-distancia el ancho del retículo es 18; bajo SED (SISBEN + distancia) es 12.
El retículo **no se abre**: se contrae. Ningún mecanismo estable redistribuye el sesgo
visual ni siquiera reordenando las prioridades a favor de los pobres. Hallazgo
adicional bajo escasez (120k cupos, 537k familias): cambiar la prioridad de distancia
a SED **reemplaza el 70% de quién obtiene cupo** (solo 30% de las familias asignadas
reciben cupo bajo ambos sistemas; 79.952 pierden y 79.952 ganan).

**Síntesis (M1, lectura de Jhoan).** Dado un sistema de prioridad fijo, los cuatro
mecanismos (BM/DA/SED/WP) colapsan al mismo punto: el retículo de estables es
proporcionalmente despreciable y la WP, estable por construcción, no puede alejarse
de DA aunque se le dé un objetivo de equidad completo. Cambiar el *sistema de
prioridad* sí reasigna masivamente (70% de los cupos). Conclusión de política: la
palanca está en a quién se prioriza (distancia vs vulnerabilidad), no en qué algoritmo
de matching se usa. El sesgo visual vive en las preferencias declaradas; ningún
mecanismo estable lo deshace.


### Tarea 5. Integración y paper

- Comparativa final de 4 mecanismos (BM/DA/SED/WP) con IC
- Narrativa de 3 actos: diagnóstico → cuantificación → solución
- Contribución triple: (1) metodológica (demanda + CV), (2) empírica (sesgo robusto), (3) diseño (WP-Rule)
- Actualizar slides y/o redactar working paper

### Dependencias

```
Tarea 1 (tópicos) ──────────────────────────┐
                                             ↓
Tarea 2 (estimación de demanda) ────────────→ requiere tópicos interpretados
                                             ↓
Tarea 3 (Monte Carlo) ─────────────────────→ requiere parámetros estimados
                                             ↓
Tarea 4 (WP-Rule) ─────────────────────────→ requiere DGP calibrado de Tarea 3
                                             ↓
Tarea 5 (integración) ─────────────────────→ requiere todo lo anterior
```

Tarea 1 es independiente y puede arrancar de inmediato.

### Tarea 6 (robustez). IV-GMM para estimación de demanda

**Contexto:** La estimación base (Tarea 2) reporta β̂_k como asociación. Si ξ_j (calidad no observada) está correlacionada con los tópicos NMF (colegios bien gestionados también se ven mejor), los coeficientes estarían sesgados.

**Instrumentos:** IVs de diferenciación (Gandhi y Houde, 2020) — para cada colegio j, calcular la distancia promedio en el espacio de características respecto a los colegios competidores del mismo mercado. PyBLP: `pyblp.build_differentiation_instruments()`.

**Implementación:** Agregar los instrumentos a la especificación PyBLP y re-estimar. Comparar β̂_k con y sin IV — si los signos y magnitudes se mantienen, la estimación base es robusta.

**Prioridad:** Baja — hacer solo si da tiempo después de las tareas 1-5. Los resultados sin IV ya son una mejora sustancial sobre la calibración.
