# BLP para Humanos 🧠

*Una guía para entender la estimación de demanda estilo BLP sin morir en el intento.*

---

## ¿Por qué estamos aquí?

Tenemos un problema concreto: las familias de Bogotá eligen colegio. Queremos saber **qué pesa** en esa decisión — calidad académica, distancia, o cómo se ve el colegio desde afuera.

Hasta ahora le pusimos números a mano: "la distancia pesa así, la calidad pesa así." Funcionó para la primera versión. Pero Álvaro nos dijo: **estímenlo.** Y tiene razón — si los datos pueden hablar, hay que dejarlos hablar.

BLP es el método estándar en economía para estimar estas preferencias. Lo inventaron Berry, Levinsohn y Pakes en 1995 para el mercado de autos, pero aplica a cualquier situación donde la gente elige entre opciones: hospitales, escuelas, candidatos políticos, cereales.

---

## Capítulo 1: La decisión de una familia

### El modelo mental

La familia de Doña Rosa en Ciudad Bolívar tiene que elegir colegio para su hijo. Ella observa:

- **Calidad académica** — ha escuchado que el Colegio San Carlos tiene buenos resultados en el ICFES
- **Distancia** — el San Carlos queda a 4 km, el Colegio Nuevo Horizonte a 800 metros
- **Cómo se ve** — Nuevo Horizonte tiene la fachada pintada y árboles; el San Carlos tiene muros grises y una calle sin pavimentar
- **Cosas que no observamos** — una vecina le dijo que el rector del San Carlos es excelente

En economía, escribimos esto como:

```
Utilidad de Rosa por el colegio j = 
    lo que TODOS valoran del colegio (calidad, apariencia, lo que no medimos)
  + lo que es ESPECÍFICO de Rosa (distancia desde su casa, su ingreso)
  + un shock aleatorio (el día que llenó el formulario estaba de buen humor)
```

O en notación:

```
u_ij = δ_j + μ_ij + ε_ij
```

Rosa elige el colegio que le da mayor utilidad. Simple.

---

## Capítulo 2: ¿Qué observamos nosotros?

No vemos la decisión de Rosa directamente. Lo que sí vemos es el **resultado agregado**: cuántas familias eligieron cada colegio. Eso son las **market shares** — la fracción de familias que fue a cada colegio.

```
s_j = (familias que eligieron j) / (total de familias en el mercado)
```

También hay familias que no eligieron ningún colegio oficial (fueron a privado, no asistieron). Esa es la **outside option**:

```
s_0 = 1 - Σ s_j
```

El tamaño de mercado M_t es la población escolar total de cada localidad. Esto importa porque define qué tan grande es s_0 — un mercado grande con pocos matriculados implica que muchas familias están eligiendo "no ir" a oficial.

---

## Capítulo 3: El truco de Berry (logit puro)

### La versión simple: todos son iguales

Empecemos asumiendo que todas las familias valoran las cosas igual. No hay μ_ij — solo δ_j (lo común) y ε_ij (el shock).

Berry (1994) demostró algo genial: si ε sigue una distribución Gumbel (tipo valor extremo), las shares tienen forma logit:

```
s_j = exp(δ_j) / (1 + Σ_k exp(δ_k))
```

Y al dividir s_j entre s_0:

```
s_j / s_0 = exp(δ_j)
```

Tomas logaritmo de ambos lados:

```
log(s_j) - log(s_0) = δ_j
```

**¡Es una regresión lineal!**

Del lado izquierdo: datos que calculamos de la matrícula.
Del lado derecho: las características del colegio.

```
log(s_j/s_0) = β_q·calidad_j + β_1·topic_1_j + ... + β_8·topic_8_j + controles + ξ_j
```

Corres OLS. Obtienes los β. Listo. Eso es Berry inversion.

### ¿Qué significan los coeficientes?

- **β_q = 0.5** → una desviación estándar más de calidad académica aumenta el log-odds de elección en 0.5
- **β_1 = 0.3** → topic_1 (fachada bonita) atrae familias
- **β_6 = −0.2** → topic_6 (entorno deteriorado) repele familias
- **ξ_j** → todo lo que el colegio tiene de atractivo que no medimos (el buen rector, el boca a boca)

### El problema

Supón que mejora el Colegio A (le pintan la fachada). En logit puro, **todos** los demás colegios pierden la misma proporción de familias. No importa si están al lado o al otro lado de la ciudad. No importa si son parecidos o completamente diferentes.

Esto es absurdo. Si pintan un colegio en Kennedy, los colegios de Kennedy deberían perder más familias que los de Usaquén. Y las familias pobres deberían reaccionar distinto que las ricas.

Esa es la limitación del logit puro. Se llama **IIA** (Independencia de Alternativas Irrelevantes). Es lo mismo que el problema del autobús rojo/autobús azul que mencionan las notas de Gortmaker.

---

## Capítulo 4: Heterogeneidad — cada familia es diferente

### La solución: dejar que los coeficientes varíen

En vez de un solo β para todos, permitimos que dependa del ingreso de la familia:

```
u_ij = δ_j + μ_ij + ε_ij
```

donde:

```
δ_j  = β_q·calidad_j + Σ β_k·topic_k_j + ξ_j     ← lo común
μ_ij = π_d·(ingreso_i × distancia_ij) + Σ π_k·(ingreso_i × topic_k_j)   ← lo heterogéneo
```

### ¿Qué captura cada π?

**π_d (ingreso × distancia):**
- Si π_d > 0: familias más ricas penalizan MENOS la distancia (tienen carro, pueden llevar al hijo)
- Si π_d < 0: familias más ricas penalizan MÁS (valoran más su tiempo)
- La literatura dice π_d > 0 consistentemente (Hastings et al., 2009)

**π_k (ingreso × tópico visual k):**
- Si π_6 < 0: familias más pobres son MÁS sensibles al deterioro visual (topic_6)
- Esto sería evidencia directa de que el sesgo visual es regresivo

### Ejemplo numérico inventado

Supón que estimamos π_6 = −0.3 y β_6 = −0.1:

```
Efecto de topic_6 para una familia rica (ingreso normalizado = +1):
  β_6 + π_6·(+1) = −0.1 + (−0.3)·(+1) = −0.4

Efecto de topic_6 para una familia pobre (ingreso normalizado = −1):
  β_6 + π_6·(−1) = −0.1 + (−0.3)·(−1) = +0.2
```

¡La familia pobre VALORA POSITIVAMENTE el deterioro! No porque le guste — sino porque los colegios deteriorados están cerca de su casa, y ella pesa más la distancia que la apariencia. El efecto neto de topic_6 confunde apariencia con accesibilidad.

Este tipo de cosas solo las ves con heterogeneidad. Con logit puro, β_6 = −0.1 para todos y te pierdes toda la historia.

---

## Capítulo 5: El problema computacional

### ¿Por qué no podemos hacer lo mismo que Berry?

Con heterogeneidad, las shares ya no se invierten analíticamente:

```
s_j = Σ_i w_i · exp(δ_j + μ_ij) / (1 + Σ_k exp(δ_k + μ_ik))
```

Esto es una **suma ponderada** sobre todos los tipos de familias. Cada familia tiene su propio μ_ij, así que no puedes factorizar exp(δ_j) afuera. No hay truco algebraico.

Para evaluar esta suma necesitas **integración numérica**: simulas 100-1000 familias representativas por mercado con distintos ingresos, calculas la probabilidad de cada una, y promedias. Eso es lo que PyBLP hace internamente.

---

## Capítulo 6: La contracción BLP (el corazón del algoritmo)

### Dos loops anidados

El algoritmo tiene dos niveles:

**Loop externo** — propones valores de π (los parámetros de heterogeneidad):
"Probemos con π_d = 0.2 y π_6 = −0.3"

**Loop interno** — dado ese π, buscas los δ_j que hacen que las shares del modelo coincidan con las shares observadas.

El loop interno es la **contracción de BLP**:

```
Empiezas con un guess de δ (por ejemplo, δ = log(s_j/s_0) del logit puro)

Repites:
    1. Calcula shares predichas: s_pred_j = Σ_i w_i · logit(δ_j + μ_ij(π))
    2. Actualiza: δ_j = δ_j + log(s_obs_j) - log(s_pred_j)
    3. Si |s_obs - s_pred| < tolerancia → convergió

Resultado: los δ̂_j que racionalizan los datos dado este π
```

**Intuición:** si un colegio tiene más share observada de la que el modelo predice, le subimos la δ (es más atractivo de lo que pensábamos). Si tiene menos, le bajamos. Eventualmente converge.

### De vuelta al loop externo

Una vez que tienes los δ̂_j, puedes recuperar ξ_j:

```
ξ̂_j = δ̂_j - β·X_j
```

El ξ̂_j es lo que **no explicamos** — la calidad no observada del colegio. Si nuestro modelo es bueno, ξ̂_j debería ser ruido. Si es grande y sistemático, nos falta algo.

El loop externo busca el π que minimiza la correlación entre ξ̂_j y los instrumentos:

```
Condición GMM: E[ξ_j · z_j] = 0
```

Esto dice: "la calidad no observada no debería estar correlacionada con los instrumentos." Si lo está, el π que propusiste es incorrecto.

### Diagrama completo

```
┌─────────────────────────────────────────────────┐
│ LOOP EXTERNO: buscar π*                         │
│                                                 │
│   Proponer π                                    │
│       ↓                                         │
│   ┌─────────────────────────────────────────┐   │
│   │ LOOP INTERNO: contracción BLP           │   │
│   │                                         │   │
│   │   Dado π, iterar δ hasta que            │   │
│   │   shares predichas = shares observadas  │   │
│   │                                         │   │
│   │   Resultado: δ̂_j                       │   │
│   └─────────────────────────────────────────┘   │
│       ↓                                         │
│   Regresión: δ̂_j = β·X_j + ξ̂_j → β̂          │
│       ↓                                         │
│   Condición GMM: g(π) = Σ ξ̂_j · z_j           │
│       ↓                                         │
│   ¿g(π) ≈ 0? → Sí: π* encontrado               │
│                 No: ajustar π, repetir          │
└─────────────────────────────────────────────────┘

Resultado final: β̂ (cuánto pesa cada feature)
                 π̂ (cómo varía por ingreso)
                 δ̂_j (utilidad de cada colegio)
                 ξ̂_j (calidad no observada)
```

---

## Capítulo 7: ¿Qué necesitamos para correrlo?

### Los datos (ya los tenemos todos)

| Input | Fuente | ¿Listo? |
|---|---|---|
| Market shares s_j | Matrícula / población escolar por localidad | Calcular de colegios_capacidad + población UPZ |
| Outside option s_0 | 1 − Σ s_j | Calcular |
| Características X_j | q_j, topic_1..8, controles | ✅ colegios_features.geojson |
| Distancia d_ij | Haversine familia→colegio | ✅ distancias_expandidas.parquet |
| Ingreso y_i | N_ingpc de EM2021 | ✅ familias_expandidas.parquet |
| Mercados t | Localidades de Bogotá | Definir (19-20 localidades) |

### La herramienta: PyBLP

```python
import pyblp

# 1. Formular el problema
problem = pyblp.Problem(
    product_formulations=(
        pyblp.Formulation('1 + q_j + topic_1 + ... + topic_8 + controles'),  # X1: δ
        pyblp.Formulation('0 + distancia'),  # X2: interactúa con demográficos
    ),
    agent_formulation=pyblp.Formulation('0 + ingreso'),  # demográficos
    product_data=product_data,   # colegios × mercado
    agent_data=agent_data,       # familias × mercado
)

# 2. Resolver
results = problem.solve()

# 3. Leer resultados
print(results.beta)   # β: utilidad media
print(results.pi)     # π: heterogeneidad por ingreso
```

PyBLP maneja la contracción, la integración numérica, los errores estándar, todo. Nosotros le pasamos los datos y la especificación.

---

## Capítulo 8: ¿Qué obtenemos al final?

### Tabla de parámetros (hipotética)

| Parámetro | Coef | SE | Interpretación |
|---|---|---|---|
| β_q | 0.45 | 0.08 | Calidad académica aumenta demanda ✓ |
| β_1 (fachada) | 0.28 | 0.11 | Fachada bonita atrae |
| β_6 (deterioro) | −0.19 | 0.09 | Deterioro repele |
| π_d (ingreso × dist) | 0.15 | 0.04 | Ricos penalizan menos la distancia |
| π_1 (ingreso × fachada) | 0.05 | 0.06 | No significativo — todos valoran igual la fachada |
| π_6 (ingreso × deterioro) | −0.22 | 0.07 | **Pobres más sensibles al deterioro** |

### ¿Qué historia cuentan estos números?

1. La calidad académica importa (β_q > 0), pero no tanto como la distancia
2. Las señales visuales tienen peso real en la demanda — no es ruido
3. **El sesgo visual es regresivo**: π_6 < 0 significa que familias de menores ingresos evitan más los colegios con entorno deteriorado, incluso controlando por distancia y calidad
4. La fachada importa para todos por igual (π_1 ≈ 0), pero el deterioro del entorno afecta desproporcionadamente a los pobres

Con estos parámetros estimados, ya no necesitamos calibrar nada a mano. El dato habló.

---

## Capítulo 9: Conexión con el resto del paper

```
Cap 5 (estimación) → α̂, β̂, π̂
        ↓
Cap 7 (sintéticos) → generamos familias con ESTOS parámetros
        ↓
Cap 7 (Monte Carlo) → 100 réplicas, resultados con IC
        ↓
Cap 6 (WP-Rule) → entrenamos mecanismo sobre sintéticos calibrados
        ↓
Cap 6 (datos reales) → aplicamos WP sobre las 97,968 familias
```

Todo el paper se ancla en la estimación. Sin ella, los parámetros son arbitrarios. Con ella, todo fluye de los datos.

---

## Resumen en una frase

**BLP toma los datos de quién eligió qué colegio, y le pone números a cuánto pesan la calidad, la apariencia y la distancia en esa decisión — permitiendo que familias ricas y pobres valoren las cosas distinto.**

---

## Lecturas recomendadas (en orden de accesibilidad)

1. **Nevo (2000)** — "A Practitioner's Guide..." — la guía más leída, con ejemplo paso a paso
2. **Conlon y Gortmaker (2020)** — "Best Practices..." — la guía moderna con PyBLP
3. **Berry (1994)** — el paper original de la inversión — corto y elegante
4. **Notas de Gortmaker** — `demand_estimation_*.md` en este repo — las clases del Mixtape

---

## Capítulo 10: Lo que implementamos, por qué falló, y cómo arreglarlo

*Esta sección documenta la implementación real del BLP para el paper de matching escolar: qué hicimos, qué resultados obtuvimos, por qué los parámetros de distancia no tienen sentido económico, y qué soluciones son factibles.*

---

### 10.1 Lo que implementamos

#### La especificación

```
u_ij = δ_j + π₁·y_i·seguridad_j + π₂·y_i·vegetacion_j + λ₀·d_ij + λ₁·y_i·d_ij + ε_ij
```

Cuatro parámetros no lineales:
- π₁: ¿familias ricas valoran más la seguridad visual percibida?
- π₂: ¿familias ricas valoran más la vegetación percibida?
- λ₀: ¿cuánto pesa la distancia para todos?
- λ₁: ¿los ricos sienten menos la distancia?

Las variables visuales (`seguridad_percibida`, `vegetacion_percibida`) vienen de CLIP — un modelo de visión computacional que compara imágenes de Street View contra frases como "a school in a safe and calm street environment" vs "a school in a dangerous and chaotic street environment". Son las dos únicas variables visuales que sobrevivieron la regresión de Berry OLS con controles (Capítulo 3 del pipeline).

Sin σ (no incluimos heterogeneidad aleatoria en gustos) porque d_ij ya es individual — cada familia tiene su propia distancia a cada colegio, lo que genera heterogeneidad directamente.

#### Los datos

- 369 colegios públicos en 19 localidades de Bogotá
- 5,700 familias muestreadas de la EM2021 (300 por localidad, ponderadas por FEX_C)
- Distancias Haversine familia→colegio (precalculadas, ~2.4 km promedio)
- Ingreso per cápita normalizado: y_i = N_ingpc / media(N_ingpc)
- Market shares de demanda administrativa SED 2024
- Outside share fija: s₀ = 0.05 en todas las localidades

#### El algoritmo

Exactamente lo descrito en los Capítulos 5 y 6: loop externo (L-BFGS-B) busca θ = (π₁, π₂, λ₀, λ₁), loop interno (contraction mapping) recupera δ_j, OLS de segunda etapa recupera β y ξ_j.

Además, implementamos dos **micro momentos** por colegio (BLP 2004, Petrin 2002):

1. **Momento de ingreso** (m_j): ingreso medio de familias ponderado por proximidad al colegio j
   - Observado: m_j^obs = Σ_i(y_i / d_ij) / Σ_i(1/d_ij)
   - Predicho: m_j^pred = Σ_i(y_i · s_ij) / Σ_i(s_ij)
   - Apunta a π₁, π₂

2. **Momento de distancia** (d_j): distancia media ponderada por FEX al colegio j
   - Observado: d_j^obs = Σ_i(d_ij · w_i) / Σ_i(w_i)
   - Predicho: d_j^pred = Σ_i(d_ij · s_ij) / Σ_i(s_ij)
   - Apunta a λ₁

El GMM combina momentos agregados (ξ'Z(Z'Z)⁻¹Z'ξ) con la suma de cuadrados de los micro momentos.

#### Los resultados

El modelo convergió en 21 iteraciones (130 evaluaciones). Los parámetros estimados:

| Parámetro | Estimación | Esperado | Sentido económico |
|---|---|---|---|
| π₁ (ingreso × seguridad) | −0.031 | Ambiguo | Ricos valoran menos la seguridad visual |
| π₂ (ingreso × vegetación) | −0.010 | Ambiguo | ~Neutro |
| λ₀ (distancia base) | **+0.120** | **Negativo** | ❌ "Todos prefieren colegios lejanos" |
| λ₁ (ingreso × distancia) | −0.202 | Positivo (Hastings) | ❌ Ricos penalizan más la distancia |

Los β lineales sí tienen sentido:

| Variable | β | Sentido |
|---|---|---|
| log_homicidios | −0.613 | ✅ Más violencia → menos demanda |
| es_técnico | +0.332 | ✅ Colegios técnicos atraen |
| seguridad_percibida | +0.061 | ✅ Calles seguras atraen |
| pct_no_oficial | +0.142 | ✅ Más competencia privada → los oficiales que sobreviven son mejores |
| q_j (calidad Saber 11) | −0.191 | ⚠️ Signo raro (ver discusión) |

---

### 10.2 Por qué falló: la intuición

Imagina que miras los datos desde arriba. Ves dos mundos:

**Ciudad Bolívar (sur, pobre):**
- 39 colegios oficiales, familias con ingreso bajo
- Casi nadie va a privado → share oficial ALTA
- La localidad es grande → distancias promedio altas (~3 km)
- Entorno visual deteriorado

**Chapinero (norte, rico):**
- 8 colegios oficiales, familias con ingreso alto
- La mayoría va a privado → share oficial BAJA
- La localidad es compacta → distancias promedio bajas (~1.5 km)
- Entorno visual bueno

El modelo ve: {share alta + distancia alta + ingreso bajo} en el sur y {share baja + distancia baja + ingreso alto} en el norte.

¿Qué concluye? "Las familias pobres van a colegios lejanos con más frecuencia que las ricas → la distancia no les molesta tanto → λ₀ > 0."

Pero eso es **falso**. La familia pobre no eligió ir lejos — simplemente no tiene alternativa privada. La share alta no revela preferencia por la distancia, revela **ausencia de outside option**.

Es como si vieras que los presos comen la comida de la cárcel todos los días y concluyeras que les encanta. No — es que no hay restaurante.

---

### 10.3 Por qué falló: la matemática

El problema se formaliza así. La share observada del colegio j en el mercado t es:

$$S_j^{obs} = \frac{\text{demanda}_j}{M_t}$$

donde $M_t$ es el tamaño del mercado (población escolar de la localidad). La outside share es:

$$s_{0t} = 1 - \sum_{j \in J_t} S_j^{obs}$$

La inversión de Berry da:

$$\delta_j = \ln(S_j^{obs}) - \ln(s_{0t})$$

**Problema 1: s₀ fija.** Usamos s₀ = 0.05 para todas las localidades. Pero en realidad:

- Chapinero: ~80% va a privado → s₀ ≈ 0.80 → ln(s₀) ≈ −0.22
- Ciudad Bolívar: ~5% va a privado → s₀ ≈ 0.05 → ln(s₀) ≈ −3.00

Con s₀ = 0.05 fija, el δ_j de un colegio en Chapinero sale artificialmente bajo (porque la share observada es baja pero la restamos contra un s₀ igual al de Ciudad Bolívar). Para compensar, ξ_j tiene que ser negativo en el norte y positivo en el sur.

**Problema 2: correlación ξ con ingreso.** Esa ξ_j sistemáticamente negativa en el norte correlaciona con las características del vecindario (rico, buenas visuales, distancias cortas). Los instrumentos Z = X no pueden romper esta correlación porque X incluye las señales visuales que también varían con el ingreso del barrio.

**Problema 3: λ₀ absorbe la confusión.** El optimizador encuentra que λ₀ > 0 reduce el error del modelo porque:

$$\underbrace{\text{share alta}}_{\text{sur}} \leftarrow \underbrace{\lambda_0 > 0}_{\text{"distancia atrae"}} \cdot \underbrace{d_{ij} \text{ grande}}_{\text{localidad grande}}$$

Es más barato para el GMM poner λ₀ positivo que ajustar 369 δ_j individualmente.

**Formalmente:** el supuesto de identificación $E[\xi_j \cdot Z_j] = 0$ se viola porque:

$$\text{Cov}(\xi_j, \text{ingreso}_{UPZ(j)}) \neq 0$$

El sorting residencial genera una correlación entre la calidad no observada del colegio y las características observables del vecindario. Sin un instrumento que rompa esta correlación, los parámetros de distancia absorben el sesgo.

---

### 10.4 Las soluciones: Capítulo H de la Encuesta Multipropósito

La EM2021 tiene un capítulo que no descargamos: **Capítulo H — Educación** (136 variables). Contiene información a nivel de persona sobre asistencia escolar, tipo de establecimiento (público/privado), medio de transporte, tiempo de desplazamiento y gasto en educación.

Con este capítulo podemos construir tres mejoras:

#### Solución A: Outside share variable por estrato y localidad

**Intuición.** El problema central es que tratamos todas las localidades como si tuvieran la misma proporción de familias que van a privado. Chapinero y Ciudad Bolívar no son comparables. Si sabemos cuántas familias de cada estrato en cada localidad eligen sector privado, podemos calcular:

$$s_{0,st} = \frac{\text{familias estrato } s \text{ en localidad } t \text{ que van a privado o no asisten}}{\text{total familias estrato } s \text{ en localidad } t}$$

**Matemática.** Con $s_{0t}$ variable, la inversión de Berry cambia:

$$\delta_j = \ln(S_j^{obs}) - \ln(s_{0t})$$

Si $s_{0t}$ es grande en Chapinero (mucho privado), $\ln(s_{0t})$ es menos negativo → $\delta_j$ sube → ya no necesita un $\xi_j$ negativo para compensar la share baja. Se rompe la correlación espuria entre $\xi_j$ y las características del vecindario.

**Qué necesitamos del Cap H:** Para cada persona de 5-17 años, saber si asiste a establecimiento público o privado, cruzado con el estrato y la localidad del hogar.

**Impacto esperado:** Alto. Este es probablemente el cambio más importante. Arregla directamente el Problema 1 y atenúa el Problema 2.

#### Solución B: Micro momento de modo de transporte

**Intuición.** El modelo actual no distingue entre "una familia que camina 2 km" y "una familia que toma bus 2 km". Pero el costo real es completamente distinto — caminar 2 km con un niño de 5 años es mucho peor que ir en bus. Y el modo de transporte correlaciona fuertemente con el ingreso: familias pobres caminan, familias ricas usan ruta escolar o carro.

**Matemática.** Construimos un micro momento:

$$\text{mm}_1 = P(\text{camina al colegio} \mid \text{estrato} \leq 2, \text{sector oficial})$$

vs

$$\text{mm}_2 = P(\text{camina al colegio} \mid \text{estrato} \geq 4, \text{sector oficial})$$

Si mm₁ >> mm₂, hay restricción de movilidad diferencial por ingreso. El momento análogo del modelo es:

$$\hat{\text{mm}}_1(\theta) = \frac{\sum_{i: y_i < \bar{y}} \sum_j s_{ij} \cdot \mathbf{1}[d_{ij} < 1\text{km}]}{\sum_{i: y_i < \bar{y}} \sum_j s_{ij}}$$

Esto apunta directamente a λ₁: la diferencia entre mm₁ y mm₂ identifica cómo el ingreso modifica la sensibilidad a la distancia — pero a través de una **restricción observable** (caminar vs no), no de una correlación espuria con el barrio.

**Qué necesitamos del Cap H:** Variable de medio de transporte al establecimiento educativo, cruzada con estrato y sector (público/privado).

**Impacto esperado:** Medio-alto. Identifica λ₁ de forma más limpia que el micro momento actual basado en Haversine.

#### Solución C: Instrumento de gasto en transporte escolar

**Intuición.** Si una familia reporta que gasta $50,000/mes en transporte escolar y otra gasta $0, eso revela directamente cuánto les cuesta la distancia. El gasto en transporte es un **precio sombra** de la distancia.

**Matemática.** Podemos construir el gasto promedio en transporte escolar por UPZ:

$$\overline{\text{gasto\_transporte}}_{UPZ} = \frac{\sum_{i \in UPZ} \text{gasto}_i \cdot \text{FEX}_i}{\sum_{i \in UPZ} \text{FEX}_i}$$

Este promedio es un instrumento válido para la distancia:
- **Relevante:** correlaciona con d_ij porque UPZs con colegios lejanos tienen mayor gasto en transporte. ✅
- **Exógeno:** cuánto gasta una familia promedio de la UPZ en transporte no afecta la calidad no observada ξ_j de un colegio específico. ✅ (El argumento de exclusión es que el gasto en transporte refleja la geografía de la UPZ, no la calidad del colegio.)

Con este instrumento, la condición de momentos:

$$E[\xi_j \cdot \overline{\text{gasto\_transporte}}_{UPZ(j)}] = 0$$

permite identificar λ₀ sin depender de la correlación directa entre distancia e ingreso.

**Qué necesitamos del Cap H:** Gasto del hogar en transporte escolar, cruzado con UPZ.

**Impacto esperado:** Medio. Es el instrumento más limpio para λ₀, pero requiere que haya suficiente variación en gasto de transporte dentro de cada localidad.

---

### 10.5 Plan de acción

| Paso | Qué hacer | Prioridad | Impacto |
|---|---|---|---|
| **1** | Descargar Capítulo H de la EM2021 del DANE | Alta | Habilita todo lo demás |
| **2** | Calcular s₀ por estrato × localidad | **Crítica** | Arregla el problema principal |
| **3** | Re-estimar BLP con s₀ variable | Alta | Verificar si λ₀ se corrige |
| **4** | Construir micro momento de transporte | Media | Mejora identificación de λ₁ |
| **5** | Construir instrumento de gasto | Media | Mejora identificación de λ₀ |
| **6** | Re-estimar BLP completo | Alta | Resultados finales |

El paso 2 es el más importante. Si con s₀ variable λ₀ sale negativo y λ₁ sale positivo, probablemente no necesitemos los pasos 4 y 5 — el problema era la outside share, no la falta de instrumentos.

---

### 10.6 ¿Qué pasa si nada funciona?

Si después de corregir la outside share y agregar micro momentos, los parámetros de distancia siguen sin sentido:

1. **Fijar λ₀ de la literatura** (Hastings et al. 2009: λ₀ ≈ −1.0) y estimar solo π₁, π₂, λ₁. Menos elegante pero honesto — muchos papers de IO fijan parámetros de fuentes externas cuando la identificación es débil.

2. **Reportar el resultado como evidencia de sorting.** Un λ₀ > 0 *es un resultado* — dice que con datos agregados y sin elecciones individuales, la distancia no se puede separar del sorting residencial. Eso motiva la extensión con datos del SIMAT.

3. **Separar la contribución del paper:** Berry OLS (Capítulo 3) sí funciona y produce resultados interpretables. Los π del BLP son un plus, pero el paper no depende de ellos para contar la historia principal: las señales visuales predicen demanda escolar.

La peor opción es forzar los números hasta que "salgan bien". Mejor reportar con honestidad y dejar la extensión como trabajo futuro con microdatos del SIMAT.
