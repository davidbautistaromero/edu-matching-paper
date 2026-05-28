# Estimación de Demanda Escolar: Documentación Técnica

*Documento de referencia para la redacción del paper. Cubre la inversión de Berry, OLS, 2SLS e instrumentos, y el BLP con micro-momentos.*

---

## 1. Marco general

Estimamos la demanda por colegios oficiales en Bogotá usando un modelo de elección discreta con preferencias heterogéneas. El objetivo es identificar cómo las señales visuales del entorno escolar (medidas con Computer Vision) afectan las decisiones de matrícula, y cuantificar la heterogeneidad por ingreso en la sensibilidad a distancia y apariencia.

El pipeline tiene dos etapas:

1. **Berry OLS + 2SLS** (`04a_berry_ols.py`): estimación agregada. Recupera la utilidad media δ_j de cada colegio y estima los coeficientes lineales β sobre características observables, incluyendo features visuales.

2. **BLP con micro-momentos** (`04b_blp.py`): estimación con heterogeneidad individual. Usa los δ_j de Berry como punto de partida y estima parámetros no lineales θ = (π₁, λ₀, λ₁) que capturan interacciones ingreso × seguridad percibida e ingreso × distancia.

---

## 2. Inversión de Berry (1994)

### 2.1 Definición de mercado

Definimos un mercado t como una **localidad** de Bogotá. Hay T = 19 mercados (localidades urbanas con colegios oficiales).

Para cada mercado t, definimos:

- **M_t** = tamaño total del mercado (familias potenciales)
- **s_j** = market share del colegio j = demanda_j / M_t
- **s₀** = outside option share (familias que no eligen ningún colegio oficial del mercado)

### 2.2 Cálculo de market shares

```
demanda_loc_t = Σ_j demanda_j    (sumada sobre todos los colegios de la localidad t)
M_t = demanda_loc_t / (1 - s₀_t)
s_j = demanda_j / M_t
```

Donde `s₀_t` se obtiene del ratio entre cupos no demandados y el tamaño total estimado de la localidad.

### 2.3 Inversión

Bajo un logit multinomial estándar, la utilidad media del colegio j se recupera analíticamente:

$$\delta_j = \log(s_j) - \log(s_0)$$

Esta es la "inversión de Berry" — transforma market shares observados en utilidades medias sin necesidad de optimización. Es exacta bajo logit; bajo modelos con heterogeneidad (BLP), sirve como punto de partida para el contraction mapping.

### 2.4 Datos

- **382 colegios urbanos** tras excluir rurales y establecimientos sin datos de demanda
- **19 localidades** como mercados
- Fuente de demanda: inscripciones a colegios oficiales (SED Bogotá)
- Outside share: s₀ calibrado por localidad

---

## 3. Berry OLS — Cuatro especificaciones

### 3.1 Modelo

$$\delta_j = X_j \beta + \xi_j$$

Donde:
- $X_j$ = vector de características observables del colegio j
- $\beta$ = coeficientes lineales
- $\xi_j$ = calidad no observada (residuo)

### 3.2 Variables

**Features visuales (CLIP):**

| Variable | Fuente | Descripción |
|---|---|---|
| `seguridad_percibida_z` | CLIP zero-shot | Probabilidad de "a safe-looking neighborhood" vs "an unsafe-looking neighborhood" |
| `vegetacion_percibida_z` | CLIP zero-shot | Probabilidad de "a green area with trees" vs "a barren area without vegetation" |
| `mantenimiento_z` | CLIP zero-shot | Probabilidad de "a well-maintained building" vs "a poorly maintained building" |
| `modernidad_z` | CLIP zero-shot | Probabilidad de "a modern building" vs "an old deteriorated building" |

**Features visuales (Cityscapes segmentación):**

| Variable | Fuente | Descripción |
|---|---|---|
| `infraestructura_vial_z` | DeepLabV3+ | % de píxeles clasificados como road + sidewalk |
| `cerramiento_z` | DeepLabV3+ | % de píxeles clasificados como fence + wall |

**Controles:**

| Variable | Descripción |
|---|---|
| `q_j_z` | Calidad académica: promedio estandarizado de puntajes ICFES (Saber 11) 2020-2023 |
| `log_homicidios_z` | log(1 + homicidios) en la localidad (seguridad objetiva) |
| `log_dist_sitp_z` | log(1 + distancia al paradero SITP más cercano) en metros |
| `pct_no_oficial_z` | % de matrícula no oficial en la localidad (competencia privada) |
| `es_tecnico` | Dummy: 1 si el colegio tiene modalidad técnica |

Todas las variables continuas están estandarizadas (z-score sobre la muestra de colegios).

### 3.3 Especificaciones

| Spec | Variables | R² | R²_adj |
|---|---|---|---|
| **M0** | Solo controles (q_j, homicidios, SITP, pct_no_oficial, es_tecnico) | baseline | baseline |
| **M1** | CLIP + controles | — | — |
| **M2** | Cityscapes + controles | — | — |
| **M3** | CLIP + Cityscapes + controles (completa) | — | — |

### 3.4 Resultados clave (M3, errores estándar HC1)

| Variable | Coeficiente | SE | p-valor | Significancia |
|---|---|---|---|---|
| `seguridad_percibida_z` | **+0.110** | 0.035 | 0.002 | *** |
| `vegetacion_percibida_z` | −0.002 | 0.037 | 0.957 | |
| `mantenimiento_z` | −0.048 | 0.046 | 0.293 | |
| `modernidad_z` | +0.003 | 0.040 | 0.946 | |
| `infraestructura_vial_z` | +0.019 | 0.031 | 0.543 | |
| `cerramiento_z` | +0.030 | 0.028 | 0.293 | |
| `q_j_z` | **−0.180** | 0.040 | <0.001 | *** |
| `log_homicidios_z` | **−0.435** | 0.036 | <0.001 | *** |
| `pct_no_oficial_z` | **−0.433** | 0.033 | <0.001 | *** |
| `es_tecnico` | **+0.439** | 0.088 | <0.001 | *** |
| `log_dist_sitp_z` | +0.018 | 0.027 | 0.504 | |

**Resultado central:** de las 6 features visuales, solo `seguridad_percibida` es significativa al 1%. Incrementar la percepción de seguridad en 1 desviación estándar aumenta δ_j en 0.110 — un efecto comparable en magnitud a `q_j` pero de signo opuesto.

**q_j negativo:** la calidad académica (puntaje ICFES) tiene coeficiente **negativo** (−0.180). Esto no significa que las familias prefieran peores colegios. Significa que, condicional en las demás variables, los colegios con mejor puntaje ICFES tienen **menor** sobredemanda observada — consistente con que la calidad académica es endógena (los colegios buenos atraen demanda → se llenan → la SED redistribuye familias a otros colegios, reduciendo la demanda marginal observada).

---

## 4. 2SLS — Instrumentando calidad académica

### 4.1 Problema de endogeneidad

`q_j` es endógena porque:
1. **Simultaneidad:** colegios con más demanda pueden atraer mejores profesores → better q_j
2. **Variable omitida:** reputación no observada correlacionada con q_j y con demanda
3. **Sesgo de selección mecánica:** la SED reasigna familias de colegios saturados a menos demandados, creando correlación artificial negativa entre calidad y demanda marginal

### 4.2 Instrumento: calidad media de rivales (BLP instruments)

Construimos un instrumento estilo BLP (Berry, Levinsohn & Pakes 1995):

$$Z_j^{IV} = \text{mean\_q\_rivals}_j = \sum_{k \neq j, k \in t} w_{jk} \cdot q_k$$

Donde:
- La suma es sobre todos los colegios k en la **misma localidad** t, excluyendo j
- $w_{jk} = \frac{1/d_{jk}}{\sum_{k'} 1/d_{jk'}}$ — pesos inversamente proporcionales a la distancia haversine entre colegios
- $d_{jk}$ = distancia haversine en km entre j y k (mínimo clipeado a 0.1 km)

**Lógica de exclusión:** la calidad promedio de los colegios rivales cercanos afecta la *calidad relativa* de j (y por tanto su demanda) solo a través de su efecto en q_j como variable de competencia, no directamente sobre las preferencias de los hogares. Es un instrumento de competencia estándar en IO.

### 4.3 First stage

```
q_j_z = γ₀ + γ₁·mean_q_rivals_z + γ₂·controles + ν_j
```

| Estadístico | Valor |
|---|---|
| F-statistic | **10.70** |
| R² | 0.187 |
| Coef. `mean_q_rivals_z` | +0.277 (p < 0.001) |

F > 10 → instrumento no débil según Stock & Yogo (2005).

### 4.4 Second stage (2SLS)

| Variable | OLS (M3) | 2SLS (M3-IV) |
|---|---|---|
| `seguridad_percibida_z` | +0.110*** | **+0.108***  |
| `q_j_z` | −0.180*** | **−0.278**  |
| `log_homicidios_z` | −0.435*** | −0.436*** |
| `pct_no_oficial_z` | −0.433*** | −0.405*** |
| `es_tecnico` | +0.439*** | +0.451*** |

**Interpretación del cambio en q_j:** el coeficiente se vuelve **más negativo** (−0.180 → −0.278). Esto sugiere un sesgo de atenuación en OLS: la endogeneidad sesgaba q_j hacia cero. El instrumento corrige esto y revela que el efecto negativo de calidad académica sobre demanda marginal es aún más pronunciado — reforzando la hipótesis de que la calidad académica NO es lo que atrae demanda en primera instancia.

**Los coeficientes visuales son estables:** `seguridad_percibida` pasa de +0.110 a +0.108, prácticamente idéntico. Esto es tranquilizador: las señales visuales no están contaminadas por la endogeneidad de q_j.

---

## 5. BLP con micro-momentos

### 5.1 Motivación

Berry OLS y 2SLS estiman coeficientes **agregados** — un β para todas las familias. Pero esperamos que la sensibilidad a distancia y a señales visuales varíe con el ingreso del hogar:
- Familias de ingreso alto pueden recorrer más distancia (tienen vehículo, pueden pagar transporte)
- Familias de ingreso alto pueden ser menos sensibles a señales visuales superficiales (tienen más información)

BLP (Berry, Levinsohn & Pakes, 1995) permite estimarlo con **heterogeneidad individual observable** a partir de micro-datos.

### 5.2 Modelo de utilidad

$$u_{ij} = \delta_j + \pi_1 \cdot y_i \cdot \text{seg}_{z,j} + \lambda_0 \cdot \log(1 + d_{ij}) + \lambda_1 \cdot y_i \cdot \log(1 + d_{ij}) + \varepsilon_{ij}$$

Donde:
- $\delta_j$ = utilidad media del colegio j (recuperada por contraction mapping)
- $y_i = N\_ingpc_i / \overline{N\_ingpc}$ = ingreso per cápita normalizado de la familia i
- $\text{seg}_{z,j}$ = seguridad percibida estandarizada (z-score) del colegio j
- $d_{ij}$ = distancia en km de la familia i al colegio j
- $\varepsilon_{ij} \sim$ Gumbel(0,1) (tipo I valor extremo)

**Parámetros no lineales (θ):**

| Parámetro | Interacción | Interpretación |
|---|---|---|
| $\pi_1$ | $y_i \times \text{seg}_{z,j}$ | ¿Los ricos valoran más/menos la seguridad percibida? |
| $\lambda_0$ | $\log(1+d_{ij})$ | Penalización base por distancia (común a todos) |
| $\lambda_1$ | $y_i \times \log(1+d_{ij})$ | ¿Los ricos penalizan más/menos la distancia? |

**Parámetros lineales (β):** recuperados por OLS en la segunda etapa:

$$\delta_j = X_j \beta + \xi_j$$

Con $X_j$ = (constante, seguridad_percibida_z, q_j_z, log_homicidios_z, log_dist_sitp_z, pct_no_oficial_z, es_tecnico).

### 5.3 Contraction mapping

Los $\delta_j$ no son observados directamente — se recuperan iterativamente para que las market shares predichas por el modelo coincidan con las observadas:

$$\delta_j^{(r+1)} = \delta_j^{(r)} + \log(s_j^{obs}) - \log(\hat{S}_j(\theta, \delta^{(r)}))$$

Donde $\hat{S}_j$ es la market share predicha:

$$\hat{S}_j(\theta, \delta) = \frac{1}{I} \sum_{i=1}^{I} \frac{\exp(\delta_j + \mu_{ij}(\theta))}{1 + \sum_{k} \exp(\delta_k + \mu_{ik}(\theta))}$$

con $\mu_{ij}(\theta) = \pi_1 \cdot y_i \cdot \text{seg}_{z,j} + \lambda_0 \cdot \log(1+d_{ij}) + \lambda_1 \cdot y_i \cdot \log(1+d_{ij})$

Convergencia: tolerancia $10^{-8}$, máximo 200 iteraciones.

### 5.4 Micro-momentos

Además de igualar market shares (condiciones de momentos agregados), usamos **micro-momentos** (Petrin 2002, BLP 2004) para mejorar la identificación:

**Momento de ingreso** $m_j^{obs}$: ingreso promedio ponderado por proximidad inversa

$$m_j^{obs} = \frac{\sum_{i} y_i \cdot FEX_i / d_{ij}}{\sum_{i} FEX_i / d_{ij}}$$

Captura el "sorting por ingreso": si familias ricas viven más cerca de colegios con alta seguridad percibida, $m_j$ será alto para esos colegios.

**Momento de distancia** $d_j^{obs}$: distancia promedio ponderada por factor de expansión

$$d_j^{obs} = \frac{\sum_{i} d_{ij} \cdot FEX_i}{\sum_{i} FEX_i}$$

**Predichos del modelo:**

$$m_j^{pred}(\theta) = \frac{\sum_i y_i \cdot P_{ij}(\theta)}{\sum_i P_{ij}(\theta)}, \qquad d_j^{pred}(\theta) = \frac{\sum_i d_{ij} \cdot P_{ij}(\theta)}{\sum_i P_{ij}(\theta)}$$

Donde $P_{ij}$ es la probabilidad de elección predicha por el modelo.

### 5.5 Función objetivo GMM

$$Q(\theta) = g(\theta)' W g(\theta) + \mu_{micro} \left[ \sum_j (m_j^{pred} - m_j^{obs})^2 + \sum_j (d_j^{pred} - d_j^{obs})^2 \right]$$

Donde $g(\theta) = Z' \xi(\theta)$ son las condiciones de momentos agregados estándar de BLP:

$$\xi_j(\theta) = \delta_j(\theta) - X_j \hat{\beta}(\theta)$$

$Z$ = matriz de instrumentos (variables exógenas + BLP instruments para q_j).
$W$ = matriz de pesos (usamos $W = (Z'Z)^{-1}$, un paso).

### 5.6 Instrumentos en BLP

**Especificación Baseline (Z = X):** sin instrumentos adicionales. $Z_j$ = las propias variables exógenas $X_j$.

**Especificación IV-BLP:** instrumenta q_j con `mean_q_rivals_z` (calidad media de rivales ponderada por 1/distancia, mismo instrumento que en Berry 2SLS).

$$Z_j^{IV} = [X_j, \text{mean\_q\_rivals}_z]$$

### 5.7 Muestra de familias

- **Fuente:** Encuesta Multipropósito de Bogotá 2021 (EM2021), expandida con `FEX_C`
- **537,031 familias** expandidas (con réplicas y ruido en distancias ±100m)
- **Muestra para BLP:** 300 familias por localidad (5,700 total), muestreo estratificado ponderado por FEX_C
- **y_i:** ingreso per cápita normalizado (N_ingpc / media global)
- **Distancias:** matriz haversine familia × colegio en km

### 5.8 Resultados

#### Parámetros no lineales (θ)

| Parámetro | Baseline | IV-BLP | Interpretación |
|---|---|---|---|
| $\pi_1$ (y_i × seg_z) | −0.0272 | **−0.0279** | Familias con mayor ingreso son *menos* sensibles a seguridad percibida |
| $\lambda_0$ (log dist) | +0.0224 | **+0.0242** | Coeficiente base de distancia (positivo pero pequeño — absorbido por δ_j) |
| $\lambda_1$ (y_i × log dist) | −0.0907 | **−0.0941** | Familias ricas penalizan *más* la distancia (contraintuitivo pero robusto) |

#### Parámetros lineales (β, OLS en segunda etapa)

| Variable | Baseline | IV-BLP |
|---|---|---|
| constante | −2.743 | −2.741 |
| `seguridad_percibida_z` | **+0.111** | **+0.112** |
| `q_j_z` | −0.184 | −0.184 |
| `log_homicidios_z` | −0.440 | −0.440 |
| `log_dist_sitp_z` | +0.018 | +0.018 |
| `pct_no_oficial_z` | −0.418 | −0.417 |
| `es_tecnico` | +0.445 | +0.445 |

#### Diagnósticos

| Métrica | Baseline | IV-BLP |
|---|---|---|
| GMM objective | 42.26 | 62.76 |
| Convergencia | ✓ | ✓ |
| RMSE micro-momento ingreso | 0.319 | 0.318 |
| RMSE micro-momento distancia | 0.095 | 0.098 |
| First-stage F (q_j) | — | **10.70** |

### 5.9 Interpretación económica

1. **Seguridad percibida importa** (β = +0.112, p < 0.01): un incremento de 1σ en la percepción de seguridad visual del entorno escolar aumenta la utilidad media del colegio en 0.112. Este efecto es **robusto** entre Berry OLS, 2SLS y BLP; no está contaminado por la endogeneidad de q_j.

2. **Calidad académica no atrae demanda** (β = −0.184): condicional en ubicación, seguridad y otros controles, los colegios con mejores puntajes ICFES no tienen mayor sobredemanda. El 2SLS amplifica este resultado (−0.278). Interpretación: la calidad académica es difícil de observar para las familias pre-matrícula; las señales visuales son más accesibles.

3. **Heterogeneidad por ingreso en seguridad** (π₁ = −0.028): familias de mayor ingreso son ligeramente *menos* sensibles a la seguridad percibida visual. El efecto es pequeño pero consistente — los ricos posiblemente usan otros canales de información (redes sociales, rankings, boca a boca) y dependen menos de la apariencia superficial.

4. **Heterogeneidad por ingreso en distancia** (λ₁ = −0.094): familias de mayor ingreso penalizan *más* la distancia. Esto es contraintuitivo si se piensa en acceso vehicular, pero consistente con que las familias ricas tienen más opciones cercanas (viven en localidades con más oferta educativa) y por tanto su costo de oportunidad de desplazarse es mayor.

5. **Homicidios y competencia privada dominan:** `log_homicidios` (−0.440) y `pct_no_oficial` (−0.417) son los efectos más grandes. Las familias evitan localidades violentas y la presencia de colegios privados reduce la demanda por oficiales.

---

## 6. Índice visual v_j

Dado que solo `seguridad_percibida` es significativa en la especificación completa (M3), el índice visual se define como:

$$v_j = \hat{\beta}_{seg} \cdot \text{seg}_{z,j} = 0.110 \cdot \text{seg}_{z,j}$$

Este índice unidimensional rankea los 382 colegios por atractivo visual percibido. Se valida en `05c_visual_index_validation.py` con inspección visual de los top 5 y bottom 5 (fotos GSV curadas manualmente).

---

## 7. De la estimación a las preferencias (06)

`06_preferencias.py` toma los outputs de 04b y genera listas de preferencia para cada familia:

$$u_{ij} = \delta_j^{BLP} + \pi_1 \cdot y_i \cdot \text{seg}_{z,j} + \lambda_0 \cdot \log(1+d_{ij}) + \lambda_1 \cdot y_i \cdot \log(1+d_{ij}) + \varepsilon_{ij}$$

Con:
- $\delta_j^{BLP}$ de `blp_delta_j.parquet` (spec IV-BLP)
- θ = (π₁, λ₀, λ₁) de `blp_results.csv` (spec iv_blp)
- $y_i$ = ingreso normalizado (N_ingpc / media global de 537K familias)
- $\text{seg}_{z,j}$ = estandarizada sobre los 382 colegios
- $\varepsilon_{ij} \sim$ Gumbel(0,1)

**Choice set:** familias eligen dentro de su localidad. Excepción: La Candelaria (localidad 17) puede elegir también en Santa Fe (3), Los Mártires (14) y Antonio Nariño (15) — por cercanía geográfica y pocos colegios propios.

**Output:** rankings top-20 por familia → `preferencias_familias.parquet` (537,031 familias × 20 preferencias).

---

## 8. Archivos generados

| Archivo | Script | Contenido |
|---|---|---|
| `data/primary/berry_delta_j.parquet` | 04a | δ_j, s_j, demanda_total por colegio |
| `reports/tables/berry_ols_specs.csv` | 04a | Coeficientes M0-M3 con SE y p-valores |
| `reports/tables/berry_2sls_m3.csv` | 04a | Coeficientes 2SLS (M3 con q_j instrumentada) |
| `data/primary/blp_delta_j.parquet` | 04b | δ_j refinados por BLP (spec IV preferida) |
| `reports/tables/blp_results.csv` | 04b | θ y β para ambas specs (baseline + IV) |
| `data/primary/preferencias_familias.parquet` | 06 | Top-20 rankings por familia (537K filas) |
| `data/processed/utilidades_familias.parquet` | 06 | Matriz completa u_ij (537K × 380) |
| `data/primary/vj_scores.csv` | 05c | v_j, rank y quintil por colegio |

---

## 9. Referencias metodológicas

- Berry, S. T. (1994). Estimating Discrete-Choice Models of Product Differentiation. *RAND Journal of Economics*, 25(2), 242-262.
- Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices in Market Equilibrium. *Econometrica*, 63(4), 841-890.
- Petrin, A. (2002). Quantifying the Benefits of New Products: The Case of the Minivan. *Journal of Political Economy*, 110(4), 705-729.
- Berry, S., Levinsohn, J., & Pakes, A. (2004). Differentiated Products Demand Systems from a Combination of Micro and Macro Data: The New Car Market. *Journal of Political Economy*, 112(1), 68-105.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In *Identification and Inference for Econometric Models*, Cambridge University Press.
- Hastings, J. S., Kane, T. J., & Staiger, D. O. (2009). Heterogeneous Preferences and the Efficacy of Public School Choice. Working Paper.
- Narasimhan, H., Agarwal, S., & Parkes, D. C. (2016). Automated Mechanism Design without Money via Machine Learning. *JMLR*, 17(1), 5765-5797.
