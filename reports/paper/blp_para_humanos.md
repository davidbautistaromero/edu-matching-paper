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
