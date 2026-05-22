# Demand Estimation — Mixtape Session
*Jeff Gortmaker and Ariel Pakes*

---

## Resumen de la clase anterior

$$\hat{\theta} = \arg\min_{\theta} \; g(\theta)' W g(\theta)$$

donde

$$g(\theta) = \frac{1}{N} \sum_{t \in T} \sum_{j \in J_t} \underbrace{(\delta_{jt} - x'_{jt}\beta)}_{\xi_{jt}(\theta)} \cdot z_{jt}$$

sujeto a

$$s_{jt} = \sum_{i \in I_t} w_{it} \cdot \frac{\exp[\delta_{jt} + x'_{jt}(\Sigma\nu_{it} + \Pi y_{it})]}{1 + \sum_{k \in J_t} \exp[\delta_{kt} + x'_{kt}(\Sigma\nu_{it} + \Pi y_{it})]}$$

- Agregar heterogeneidad en preferencias $\mu_{ijt}$ produce patrones de sustitución más realistas.
  - La forma más común es $\mu_{ijt} = x'_{jt}(\Sigma\nu_{it} + \Pi y_{it})$ con $\nu_{it} \sim N(0, I)$ e $y_{it}$ de datos censales.
  - Implementa coeficientes aleatorios $\beta_{it} \sim N(\beta + \Pi y_{it}, \Sigma\Sigma')$ sobre las características $x_{jt}$.
- Esto requirió agregar datos del consumidor $i$ para complementar los datos de producto $j$ del día 1.

---

## Variación limitada entre mercados

- El ejercicio de reducción de precios parece más razonable con coeficientes aleatorios: los consumidores sustituyen hacia productos similares, especialmente en la dimensión de precio.
- Pero los coeficientes aleatorios que se pueden agregar están limitados por la variación en los datos.
- **No se puede** estimar crediblemente una desviación estándar en $\Sigma$ sobre el dummy *mushy*: los mismos cereales están en cada mercado, por lo que no hay variación en el conjunto de elección a lo largo de la dimensión *mushy*.
- **Tampoco se puede** estimar un parámetro en $\Pi$ sobre el log-ingreso solo: los efectos fijos de mercado son colineales con las medias de ingreso a nivel de mercado.

---

## Variación dentro del mercado

Sin mucha variación entre mercados, lo que realmente se necesita es **variación dentro del mercado**.

**"Micro data"** contiene información sobre elecciones individuales, no solo cantidades a nivel de mercado.

Fuentes típicas de encuestas a consumidores:
- Encuestas internas de empresas.
- Encuestas ad-hoc realizadas por académicos.
- Datasets de investigación de mercado (p. ej. NielsenIQ Consumer Panel).
- Agencias regulatorias como la autoridad antimonopolio del Reino Unido (Reynolds y Walters, 2008).

---

## Intuición con el ejemplo de cereales

Imaginemos encuestar personas en el supermercado que compraron cereal:

- **"¿Cuál fue su ingreso anual el año pasado?"**
  - Informativo sobre un parámetro $\pi_1$ sobre log-ingreso.
  - La media del ingreso de compradores de cereal captura cómo el ingreso desplaza la preferencia por cereal.

- **"¿Habría comprado otro cereal *mushy* si su primera elección no estaba disponible?"**
  - Informativo sobre una desviación estándar $\sigma$ en $\Sigma$ sobre el dummy *mushy*.
  - La sustitución dentro de *mushy* es precisamente lo que este parámetro aumentará.

---

## Estimador Micro BLP

Se extiende el estimador BLP incorporando estadísticas de encuestas a consumidores:

$$\hat{\theta} = \arg\min_{\theta} \; g(\theta)' W g(\theta), \quad g(\theta) = \begin{bmatrix} \frac{1}{N}\sum_{j,t}(\delta_{jt}(\theta) - x'_{jt}\beta) \cdot z_{jt} \\ f(v) - f(v(\theta)) \end{bmatrix}$$

donde:
1. $v_1$: ingreso medio de compradores de cereal → igualado a la predicción del modelo $v_1(\theta)$.
2. $v_2$: fracción de compradores de cereal que eligió un cereal *mushy* y elegiría otro *mushy* si su primera opción no estuviera disponible → análogo del modelo $v_2(\theta)$.

- El vector de estadísticas igualadas es $v = [v_1, v_2]'$.
- No tienen que ser medias: se puede igualar cualquier función suave $f(v)$ — ratios, correlaciones, etc.
- El estimador **"micro BLP"** resultante es ampliamente usado en organización industrial.

---

## Popularidad del Micro BLP

Popularizado inicialmente por Petrin (2002) y BLP (2004). Se ha utilizado en industrias como automóviles, comida rápida, computadoras personales, cigarrillos, café, seguros de salud, hogares de enfermería, cámaras digitales, cereal, escuelas, retail, camiones, preescolares, lavadoras y licores destilados, entre otros países: Estados Unidos, China, Bélgica, Francia y Chile.

Se usará el marco estandarizado de **PyBLP** de Conlon y Gortmaker (2023).

---

## Momentos Micro (*Micro Moments*)

Hay dos nuevos componentes:

1. **Estadísticas micro**: $f(v) = [f_1(v), \ldots, f_M(v)]'$
2. **Sus análogos en el modelo**: $f(v(\theta)) = [f_1(v(\theta)), \ldots, f_M(v(\theta))]'$

Se necesita $f(v) \to f(v(\theta))$ a medida que el micro dataset crece. Esto genera $m = 1, \ldots, M$ "momentos micro" distintos — muy diferentes de los "momentos agregados" $E[\xi_{jt} \cdot z_{jt}] = 0$.

---

## Estadísticas Micro

Cada estadística micro $f_m(v)$ es un resumen calculado sobre un micro dataset.

Los **micro datasets** $d \in D$ consisten en consumidores encuestados $n \in N_d$ que pueden:

1. Provenir de cada mercado $t_n \in T$ con probabilidad igual.
2. Ser de tipo $i_n \in I_{t_n}$ con probabilidad $w_{i_n t_n}$ (mismo peso que antes).
3. Elegir $j_n \in J_{t_n} \cup \{0\}$ con probabilidad $s_{i_n j_n t_n}$ (misma probabilidad logit que antes).
4. Ser seleccionados en la encuesta con probabilidad conocida $w^d_{i_n j_n t_n}$ (frecuentemente basada en elección).

Las estadísticas micro son funciones suaves de promedios de "partes micro":

$$v_p = \frac{1}{N_d} \sum_{n \in N_d} v^p_{i_n j_n t_n}$$

Diferentes pesos $w^d_{ijt}$, valores $v^p_{ijt}$ y funciones $f_m(\cdot)$ permiten capturar la mayoría de los resúmenes estadísticos.

---

## Ejemplo: estadística micro para $v_1$

Para igualar $v_1$ (ingreso medio de compradores de cereal), PyBLP necesita:

**Definir el micro dataset $d$:**
- Pesos de muestreo $w^d_{ijt} = \mathbf{1}\{j \neq 0\}$: solo compradores de cereal fueron encuestados.
- Se especifica una función para calcular una matriz de pesos por mercado $t$.
- También se especifica el número de observaciones de la encuesta $N_d = |N_d|$.

**Definir la parte micro $p$:**
- Valores micro $v^p_{ijt} = \text{income}_{it}$: así $v_p$ es el ingreso medio encuestado.
- Se especifica una segunda función para calcular una matriz de valores por mercado $t$.

**Definir el momento micro $m$:**
- La función identidad $f_m(v_p) = v_p$ simplemente iguala el ingreso medio encuestado.
- También se especifica el valor observado de la estadística $v_1$.

---

## Análogos del modelo

El análogo del modelo de una parte micro $v_p(\theta)$ es una esperanza condicional:

$$v_p(\theta) = f_m\left(\frac{\sum_{t \in T}\sum_{i \in I_t}\sum_{j \in J_t \cup \{0\}} w_{it} \cdot s_{ijt}(\theta) \cdot w^d_{ijt} \cdot v^p_{ijt}}{\sum_{t \in T}\sum_{i \in I_t}\sum_{j \in J_t \cup \{0\}} w_{it} \cdot s_{ijt}(\theta) \cdot w^d_{ijt}}\right)$$

Refleja el proceso generador de datos de los consumidores encuestados $n \in N_d$:
1. En mercado $t_n$ con probabilidad igual.
2. De tipo $i_n$ con probabilidad $w_{i_n t_n}$.
3. Elige $j_n$ con probabilidad $s_{i_n j_n t_n}$.
4. Seleccionado en la encuesta con probabilidad $w^d_{i_n j_n t_n}$.

---

## Elección de momentos micro

Para cada nuevo parámetro sin instrumento, se necesita un momento micro. Conviene comenzar con un solo momento que "apunte" al parámetro.

Si un parámetro puede estimarse con variación agregada **o** micro:
- Elegir la que parezca más "creíble" (frecuentemente la micro).
- Se pueden usar ambas: los momentos micro pueden reducir errores estándar grandes derivados de variación agregada limitada.

---

## Apuntando momentos micro

Caso simple: 1 característica $x_{jt}$ (p. ej. precio), 1 demográfico $y_{it}$ (p. ej. ingreso):

$$u_{ijt} = \beta_1 + \pi_1 y_{it} + (\beta_x + \pi_x y_{it})x_{jt} + \xi_{jt} + \varepsilon_{ijt}$$

| Parámetro | Momento que lo apunta | Intuición |
|---|---|---|
| $\pi_1$ | $E[y_{it} \mid j \neq 0]$ | Ingreso medio de compradores → cómo el ingreso desplaza la preferencia |
| $\pi_x$ | $E[y_{it} \cdot x_{jt} \mid j \neq 0]$ o $\text{Cov}(y_{it}, x_{jt} \mid j \neq 0)$ | Relación ingreso-precio → cómo el ingreso cambia la sensibilidad al precio |

> Los parámetros "lineales" $\beta_1$ o $\beta_x$ no son directamente informativos desde los micro datos: la utilidad media $\delta_{jt} = \beta_1 + \beta_x x_{jt} + \xi_{jt}$ ya está determinada por las cuotas de mercado $s_{jt}$.

---

## Datos de segunda elección

Para apuntar parámetros de heterogeneidad no observada en $\Sigma$, la literatura ha tenido éxito con **datos de segunda elección** (p. ej. BLP, 2004):

- La encuesta pregunta qué $k_n \neq j_n$ elegiría si su primera elección no estuviera disponible.
- Los pesos y valores micro ahora tienen un índice extra: $w^d_{ijkt}$ y $v^p_{ijkt}$.
- Las medidas directas de sustitución son muy informativas sobre $\Sigma$: cada segunda elección es como observar un mercado nuevo con la primera elección removida.

Con heterogeneidad no observada:

$$u_{ijt} = \beta_1 + \sigma_1\nu_{1it} + \pi_1 y_{it} + (\beta_x + \sigma_x\nu_{2it} + \pi_x y_{it})x_{jt} + \xi_{jt} + \varepsilon_{ijt}$$

| Parámetro | Momento que lo apunta |
|---|---|
| $\sigma_x$ | $P(\text{mushy}_{jt} \text{ y } \text{mushy}_{kt} \mid j \neq 0)$ o $\text{Cov}(x_{jt}, x_{kt} \mid j, k \neq 0)$ |
| $\sigma_1$ | $P(k = 0 \mid j \neq 0)$ (sustitución hacia el bien externo) |

---

## Sustitución exterior y tamaño de mercado

Estimar $\sigma_1$ es importante para la sustitución dentro-fuera del mercado (p. ej. efecto de un impuesto a todos los refrescos; crecimiento del mercado por innovación).

- Asumir un tamaño de mercado $M_t$ pequeño implica una cuota exterior $s_{0t}$ pequeña → alta calidad interior $\xi_{jt}$ → poca sustitución al bien externo en contrafactuales.
- Igualar directamente una razón de desvío exterior (*outside diversion ratio*) ayuda a disciplinar la sustitución exterior.
- Ver Zhang (2023) para más soluciones.

---

## Uso de más información

**¿Por qué usar resúmenes estadísticos y no la encuesta completa?**

Razones para preferir estadísticas resumen:
- **Costo**: más datos cuestan más; los resúmenes pueden ser gratuitos.
- **Confidencialidad**: los proveedores pueden querer proteger identidades.
- **Compatibilidad**: datos agregados y micro pueden venir de esquemas de muestreo distintos.
- **Claridad**: igualar una sola estadística deja en claro de dónde viene la identificación.

Sin embargo, **agregar más información puede aumentar considerablemente la precisión**.

---

## Máxima Verosimilitud (MLE)

Si solo se tuvieran micro datos, se podría trabajar con la log-verosimilitud:

$$\log L(\theta, \delta) = \sum_{n \in N_d} P(t_n, j_n, k_n, y_{i_n t_n} \mid n \in N_d;\; \theta, \delta)$$

Enfoque clásico en dos pasos:
1. Encontrar $\hat{\theta} = (\hat{\Sigma}, \hat{\Pi})$ y las utilidades medias $\hat{\delta}$ que maximizan $\log L(\theta, \delta)$.
2. Regresión IV de $\hat{\delta}_{jt}$ sobre $x_{jt}$ para recuperar los parámetros lineales $\hat{\beta}$.

Para un enfoque moderno, ver Grieco, Murry, Pinkse y Sagl (2023), que combinan ambos pasos y el verosímil de cuotas de mercado agregadas en un único objetivo (paquete `Grumps.jl` en Julia).

---

## Momentos micro óptimos

En micro BLP, los momentos micro óptimos igualan las condiciones de primer orden del MLE:

$$f^*(v) = \frac{1}{N_d} \sum_{n \in N_d} \frac{\partial P(t_n, j_n, k_n, y_{i_n t_n} \mid n \in N_d;\; \theta)}{\partial \theta}$$

- Usan **toda** la información en el micro dataset (Conlon y Gortmaker, 2023).
- Eficiencia estadística: intuición análoga a que MLE es eficiente.
- Relativamente fáciles de computar en PyBLP (pocas líneas de código).
- Como los IVs óptimos, se pueden actualizar junto con la matriz de ponderación en un segundo paso GMM.

---

## Ejercicio de Codificación 3

1. Incorporar momentos micro.
2. Estimación Micro BLP.
3. Evaluar mejoras al contrafactual de reducción de precios.

**Ejercicios suplementales:**
- Variar el tamaño de mercado.
- Momentos micro óptimos.
- Estimar un parámetro de anidamiento (*nesting parameter*).

---

## Referencias

- Berry, S., Levinsohn, J., y Pakes, A. (2004). "Differentiated products demand systems from a combination of micro and macro data: The new car market." *Journal of Political Economy*, 112(1), 68–105.
- Conlon, C. y Gortmaker, J. (2023). "Incorporating micro data into differentiated products demand estimation with PyBLP."
- Grieco, P., Murry, C., Pinkse, J., y Sagl, S. (2023). "Conformant and efficient estimation of discrete choice demand models."
- Petrin, A. (2002). "Quantifying the benefits of new products: The case of the minivan." *Journal of Political Economy*, 110(4), 705–729.
- Pinkse, J. et al. `Grumps.jl`. Disponible en https://github.com/NittanyLion/Grumps.jl
- Reynolds, G. y Walters, C. (2008). "The use of customer surveys for market definition and the competitive assessment of horizontal mergers." *Journal of Competition Law and Economics*, 4(2), 411–431.
- Zhang, L. (2023). "Identification and estimation of market size in discrete choice demand models."