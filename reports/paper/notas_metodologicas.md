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
u_ij = β·X_j  −  α_s · ln(1 + d_ij)  +  ε_ij
```

| Componente | Descripción |
|---|---|
| `β·X_j` | Score de calidad predicho por Ridge M1 — homogéneo entre estratos |
| `α_s · ln(1+d_ij)` | Penalización distancia heterogénea por estrato |
| `ε_ij ~ Gumbel(0,1)` | Ruido logístico (seed=42) |

**Penalización por distancia** — power law calibrada con ratio α₁/α₆ = 3× (Hastings et al. 2009):

```
α_s = 0.30 / s^0.613
```

| Estrato | α_s |
|---|---|
| 1 | 0.300 |
| 2 | 0.196 |
| 3 | 0.153 |
| 4 | 0.128 |
| 5 | 0.112 |
| 6 | 0.100 |

**Choice set:** colegios de la misma localidad. Excepción: La Candelaria (localidad 17) puede elegir en localidades 3 (Santa Fe), 14 (Los Mártires) y 15 (Antonio Nariño).

**Resultado:** rankings top-20 por familia. Choice set promedio: 23.3 colegios.

---

## Pipeline de matching

### M1. Datos reales — diseño

- Capacidad = `round(matricula_total / 13)`, mínimo 5 (cohort anual estimado)
- Prioridad = distancia Haversine (réplica del criterio SED Bogotá)
- Choice set = localidad (heredado de `06_preferencias.py`)

**Resultados:**

| Mecanismo | Asignados | Eficiencia q | corr(estrato, q) | Rank medio | Blocking pairs |
|---|---|---|---|---|---|
| Boston (BM) | 13,568 (100%) | 258.72 | +0.193 | 1.18 | **8,194** |
| DA (Gale-Shapley) | 13,568 (100%) | 258.73 | +0.192 | 1.19 | **0** ✓ |
| SED-lex | 13,568 (100%) | — | — | — | **0** ✓ |

DA: 0 blocking pairs (estable por construcción). La diferencia de eficiencia BM vs DA es mínima — la capacidad total (45,119 cupos) supera ampliamente la demanda (13,568 familias). La competencia existe dentro de localidades, pero casi todos obtienen su primera o segunda preferencia. Grupos A-B: rechazo 0.0% en datos reales para SED-lex.

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
- Hastings, J., Kane, T. & Staiger, D. (2009). Heterogeneous Preferences and the Efficacy of Public School Choice. *NBER Working Paper* 12145. [Fuente de α₀=0.30 y ratio α₁/α₆=3×]
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
