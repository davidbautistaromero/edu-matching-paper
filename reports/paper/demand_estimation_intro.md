# Demand Estimation — Mixtape Session
**Jeff Gortmaker and Ariel Pakes**

---

## Sobre el instructor

- Candidato a PhD en Economía (quinto año) en Harvard University.
- Objetivo: hacer la estimación estilo BLP más accesible a investigadores.
  - Artículos de mejores prácticas (Conlon y Gortmaker, 2020, 2023).
  - Paquete open-source de Python: **PyBLP**.

---

## Estructura del curso

Tres días, 6pm–9pm:

1. **Hoy:** Modelo BLP, logit puro, endogeneidad del precio.
2. **Miércoles:** Logit mixto, identificación, mejores prácticas numéricas.
3. **Viernes:** Micro BLP, datos de encuestas a consumidores, otras extensiones.

- Preguntas por Discord (se responden en tiempo real o después de clase).
- Tres ejercicios de programación, uno por día.
  - Los primeros dos se resuelven en vivo al inicio de los días 2 y 3.

---

## Lecturas recomendadas

**Guías modernas:**
1. Berry y Haile (2021)
2. Conlon y Gortmaker (2020)
3. Conlon y Gortmaker (2023)

**Guías fundacionales:**
1. Berry, Levinsohn y Pakes (1995)
2. Nevo (2000)
3. Petrin (2002)
4. Berry, Levinsohn y Pakes (2004)

> Ninguna es obligatoria para el curso, pero se recomienda revisarlas después.

---

## Ejemplo running

BLP sirve para analizar decisiones como compras de productos, visitas hospitalarias, elección de escuela, comportamiento de voto, etc.

Se usa principalmente para **análisis contrafactual** de situaciones que aún no han ocurrido (se necesita un modelo estructural cuando no basta estimar un efecto de tratamiento).

**Pregunta central:** ¿Qué pasaría si redujéramos a la mitad el precio de un producto importante?
- *Empresa:* ¿Más ventas o canibalización?
- *Regulador:* ¿Pérdida de ingresos por eliminar un impuesto?
- *Academia:* ¿Consecuencias sobre el bienestar?

---

## El modelo BLP

### Visión general

- Modelo de elección discreta: individuos escogen entre distintas alternativas.
- Elecciones ocurren en mercados $t \in T$ (períodos de tiempo, regiones geográficas, etc.).
- Cada mercado tiene individuos con tipos $i \in I_t$ (distintos demografías y preferencias).
- Los individuos enfrentan opciones $j \in J_t$ (productos, hospitales, candidatos, etc.).
- **Opción exterior** $j = 0$: no compra, sin tratamiento, sin voto, etc.

### Maximización de utilidad

$$\max_{j \in J_t \cup \{0\}} u_{ijt} = \delta_{jt} + \mu_{ijt} + \varepsilon_{ijt}$$

Los individuos escogen la alternativa que maximiza su utilidad indirecta $u_{ijt}$, descompuesta en tres partes:

| Componente | Descripción |
|---|---|
| $\delta_{jt}$ | **Utilidad media:** preferencia promedio entre todos los individuos del mercado. |
| $\mu_{ijt}$ | **Heterogeneidad sistemática:** diferencias por demografía u otras características. |
| $\varepsilon_{ijt}$ | **Heterogeneidad idiosincrática:** ruido superpuesto que facilita la estimación. |

Se parametrizan $\delta_{jt}$ y $\mu_{ijt}$, y se asume una distribución conveniente para $\varepsilon_{ijt}$.

### Cuotas de mercado agregadas

Supuesto: $\varepsilon_{ijt}$ sigue una distribución **valor extremo tipo I** iid (*logit shocks*), lo que produce probabilidades de elección logit multinomial:

$$s_{ijt} = \frac{\exp(\delta_{jt} + \mu_{ijt})}{\sum_{k \in J_t \cup \{0\}} \exp(\delta_{kt} + \mu_{ikt})}$$

Agregando sobre tipos de individuos con pesos $w_{it}$:

$$s_{jt} = \sum_{i \in I_t} w_{it} \cdot s_{ijt}$$

En los datos, se observan cantidades $q_{jt} = s_{jt} \cdot M_t$.

### Elección del tamaño de mercado

Se necesita dividir $q_{jt}$ por algún tamaño de mercado $M_t$ para obtener $s_{jt}$, pero frecuentemente no se observa la cantidad de la opción exterior $q_{0t}$.

- A veces es directo (ej. mercado de medicamentos = personas con la enfermedad).
- Generalmente es ambiguo e importante (ej. ¿cuántas decisiones de compra de cereal se toman por día en una ciudad?).
- Se deben probar distintos supuestos: a mayor tamaño, mayor sustitución hacia la opción exterior.

### Identificación y normalizaciones

La utilidad es invariante a transformaciones afines positivas, por lo que se necesitan dos normalizaciones:

a. **Nivel:** $u_{i0t} = \varepsilon_{i0t}$, es decir $\delta_{0t} = \mu_{i0t} = 0$ → estimaciones relativas a la utilidad de la opción exterior.  
b. **Escala:** $\text{Var}(\varepsilon_{ijt}) = \pi^2/6$ (ya normalizada al derivar las probabilidades) → estimaciones relativas a la escala del ruido.

---

## Estimación: Logit puro

### Modelo logit puro (sin heterogeneidad)

Caso más simple: $\mu_{ijt} = 0$. Las cuotas se simplifican:

$$s_{jt} = \frac{\exp \delta_{jt}}{1 + \sum_{k \in J_t} \exp \delta_{kt}}$$

Usando el resultado de Berry (1994), se pueden recuperar las utilidades medias:

$$\log \frac{s_{jt}}{s_{0t}} = \delta_{jt}$$

### Ecuación de estimación

$$\log \frac{s_{jt}}{s_{0t}} = \delta_{jt} = \alpha p_{jt} + x'_{jt}\beta + \xi_{jt}$$

- $p_{jt}$: precio del producto (ej. precio por porción de cereal).
- $x_{jt}$: características observables (constante, dummy "mushy", etc.).
- $\xi_{jt}$: calidad no observada (características no medidas, publicidad, shocks de demanda).

### Interpretación de parámetros

- **$\alpha$** está en "utils por dólar" → reportar elasticidades precio propias:

$$\eta_{jjt} = \frac{\partial \log q_{jt}}{\partial \log p_{jt}} = \alpha \cdot p_{jt} \cdot (1 - s_{jt})$$

- **$\beta$** está en "utils" → reportar la disposición a pagar: $\beta / \alpha$ (en dólares).

---

## Endogeneidad del precio

### El problema

En OLS, si un regresor está correlacionado con el error, su coeficiente es sesgado.

- Las firmas conocen más sobre la demanda que el investigador al fijar precios.
- Típicamente $\text{Cov}(p_{jt}, \xi_{jt}) > 0$, lo que sesga $\hat{\alpha}$ hacia cero.

### Efectos fijos

Agregar efectos fijos de producto ($\xi_j$) y mercado ($\xi_t$) elimina mucho sesgo cuando el precio está correlacionado con esos componentes de $\xi_{jt}$.

- Requiere múltiples observaciones por producto y mercado.
- Con datos de scanner modernos (miles de productos/mercados), se "absorben" iterativamente: **Stata:** Reghdfe · **R:** Fixest · **Python:** PyFixest / PyBLP+PyHDFE.
- Insuficiente si $\text{Cov}(p_{jt}, \Delta\xi_{jt}) > 0$.

### Variables instrumentales (IV)

Se necesita un instrumento $z_{jt}$ que cumpla:
- **Relevancia:** $\text{Cov}(p_{jt}, z_{jt}) \neq 0$
- **Exclusión:** $\text{Cov}(\xi_{jt}, z_{jt}) = 0$

Siempre correr una primera etapa: ¿el signo tiene sentido? ¿es fuerte?

**Instrumentos típicos para el precio:**

| Tipo | Descripción |
|---|---|
| **Cost-shifters** | Precios de insumos, aranceles, etc. |
| **Hausman** | Precio del mismo producto promediado en otros mercados. |
| **Waldfogel** | Características promedio de consumidores en mercados cercanos. |
| **BLP** | Características promedio $x_{kt}$ de productos competidores $k \neq j$. |

> Recomendación: comenzar con un solo instrumento, preferiblemente un cost-shifter si se tiene.

---

## Ejercicio de programación 1

1. Configurar Python y PyBLP.
2. Estimación logit puro.
3. Contrafactual de reducción de precio a la mitad.

**Punto crítico:** ¿Los patrones de sustitución estimados son razonables?

**Ejercicios suplementales (opcional):**
- Inferencia estadística.
- Modelar el lado de oferta.
- Verificar el código simulando datos.

---

## Referencias

- Berry, S. (1994). Estimating discrete-choice models of product differentiation. *RAND Journal of Economics*, 242–262.
- Berry, S. y Pakes, A. (2007). The pure characteristics demand model. *International Economic Review*, 48(4), 1193–1225.
- Berry, S., Levinsohn, J. y Pakes, A. (1995). Automobile prices in market equilibrium. *Econometrica*, 63(4), 841–890.
- Berry, S., Levinsohn, J. y Pakes, A. (2004). Differentiated products demand systems from a combination of micro and macro data. *Journal of Political Economy*, 112(1), 68–105.
- Berry, S. T. y Haile, P. A. (2021). Foundations of demand estimation. *Handbook of Industrial Organization*, Vol. 4, 1–62.
- Conlon, C. y Gortmaker, J. (2020). Best practices for differentiated products demand estimation with PyBLP. *RAND Journal of Economics*, 51(4), 1108–1161.
- Conlon, C. y Gortmaker, J. (2023). Incorporating micro data into differentiated products demand estimation with PyBLP.
- DellaVigna, S. y Gentzkow, M. (2019). Uniform pricing in US retail chains. *Quarterly Journal of Economics*, 134(4), 2011–2084.
- Nevo, A. (2000). A practitioner's guide to estimation of random-coefficients logit models of demand. *Journal of Economics & Management Strategy*, 9(4), 513–548.
- Petrin, A. (2002). Quantifying the benefits of new products: The case of the minivan. *Journal of Political Economy*, 110(4), 705–729.
- Waldfogel, J. (2003). Preference externalities. *RAND Journal of Economics*, 34(3), 557.