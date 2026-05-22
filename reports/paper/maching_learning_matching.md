# Aplicaciones de Redes Profundas a la Teoría de Emparejamiento

**Álvaro J. Riascos Villegas** — Universidad de los Andes y Quantil  
*14 de mayo de 2026*

---

## Contenido

1. [Introducción](#1-introducción)
2. [¿Cómo pueden salir mal las cosas?](#2-cómo-pueden-salir-mal-las-cosas)
3. [Mercados de dos lados: Algoritmo de aceptación diferida](#3-mercados-de-dos-lados-algoritmo-de-aceptación-diferida)
   - Mercado de Matrimonios: Uno a uno
   - Comportamiento Estratégico
   - Admisiones a la Universidad: Muchos a uno
4. [¿Qué son las matemáticas?](#4-qué-son-las-matemáticas)
5. [Automated Mechanism Design](#5-automated-mechanism-design)
6. [Resultados](#6-resultados)

---

## 1. Introducción

- **Mercados de dos lados:** trabajadores y empresas, médicos y hospitales, estudiantes y escuelas públicas, donantes y receptores de riñones, vendedores y compradores de acciones, plataformas digitales.
- **Mercados de un lado:** estudiantes y dormitorios universitarios, subastas o licitaciones.
- En muchos casos no existe un mercado organizado y usar transferencias monetarias no es ético.
- Se estudian los mercados en los cuales no hay una transferencia de dinero.
- Es un ejemplo exitoso de la aplicación de la teoría de juegos.

---

## 2. ¿Cómo pueden salir mal las cosas?

### Mercado de matrimonios

Sea $M = \{m_1, \ldots, m_n\}$ y $W = \{w_1, \ldots, w_p\}$ el conjunto de hombres y mujeres.

- Cada hombre (o mujer) tiene preferencias por el otro grupo incluyendo una preferencia por estar solo (preferencias racionales: completas y transitivas). Se denota $P(m) = w_{i_1}, \ldots, m, \ldots, w_{i_n}$ las preferencias de $m$ de mayor a menor.
- Cuando hay indiferencia, por ejemplo entre $w_{i_2}$ y $w_{i_3}$, se denota $P(m) = w_{i_1}, [w_{i_2}, w_{i_3}], \ldots, m, \ldots, w_{i_n}$.
- También se usan $>_m$ y $\geq_m$ como preferencias estrictas o débiles.
- El conjunto de preferencias se denota: $P = \{P(m_1), \ldots, P(m_n), P(w_1), \ldots, P(w_p)\}$
- Un mercado de matrimonios es $(M, W, P)$.
- Una mujer $w$ es **aceptable** para $m$ si $w \geq_m m$.

### Definición: Emparejamiento y Racionalidad Individual

Una función 1-1 $\mu : M \cup W \to M \cup W$ tal que $\mu^2 = id$ y si $\mu(m) \neq m$, entonces $\mu(m) \in W$ (similar para las mujeres). Se dice que $\mu(m)$ es la pareja de $m$.

**Ejemplo:**

$$\mu = \begin{pmatrix} w_4 & w_1 & w_2 & w_3 & (m_5) \\ m_1 & m_2 & m_3 & m_4 & m_5 \end{pmatrix}$$

**Definición (Racionalidad Individual):** $\mu$ es individualmente racional si cada agente es aceptable para su pareja (i.e., si ningún agente bloquea de forma unilateral el emparejamiento).

### Definición: Estabilidad

Un emparejamiento $\mu$ es **estable** si no es bloqueado por ningún agente o pareja de agentes. Se dice que $\mu$ es bloqueado por una pareja $(m, w)$ si ambos prefieren estar emparejados entre sí a estar con la pareja que les corresponde en $\mu$: $m >_w \mu(w),\ w >_m \mu(m)$.

### Ejemplo de Estabilidad

$$P(m_1) = w_2, w_1, w_3 \qquad P(w_1) = m_1, m_3, m_2$$
$$P(m_2) = w_1, w_3, w_2 \qquad P(w_2) = m_3, m_1, m_2$$
$$P(m_3) = w_1, w_2, w_3 \qquad P(w_3) = m_1, m_3, m_2$$

El emparejamiento $\mu = \begin{pmatrix} w_1 & w_2 & w_3 \\ m_1 & m_2 & m_3 \end{pmatrix}$ **no es estable** ($(w_2, m_1)$ lo bloquean).

El emparejamiento $\mu = \begin{pmatrix} w_1 & w_2 & w_3 \\ m_1 & m_3 & m_2 \end{pmatrix}$ **es estable**. Corresponde al algoritmo de aceptación diferida en el que la mujer propone.

---

### Variaciones simples del modelo: problemas donde no existe emparejamiento estable

#### El problema de los compañeros de habitación (Gale y Shapley)

Mercado de un solo lado, muchos a uno. $n$ (par) personas deben emparejarse para ocupar $n/2$ habitaciones. Cada persona tiene preferencias sobre los otros $n-1$. Un emparejamiento es estable si no existen dos personas no emparejadas que ambas prefieran estar juntas a estar con la pareja que les tocó.

**Ejemplo con $n = 4$:**

$$p(a) = b, c, d \qquad p(b) = c, a, d \qquad p(c) = a, b, d \qquad p(d) = \text{arbitrario}$$

No puede existir un emparejamiento estable: alguien debe quedar con $d$, y esa persona es el preferido de alguno de los otros dos, quienes bloquearían el emparejamiento.

#### El problema de conformar una familia (Alkan)

Tres tipos de agentes: padres, madres e hijos. Un emparejamiento es la conformación de una familia (padre, madre, un hijo). Un conjunto $(m, w, c)$ bloquea un emparejamiento si cada miembro prefiere conformar esa familia a la familia con la que está emparejado.

Con tres hombres, tres mujeres y tres hijos y preferencias específicas (ver detalle), cualquier posible emparejamiento resulta bloqueado — no existe emparejamiento estable.

#### Muchos a uno: Trabajadores a firmas

Mercado de dos lados, muchos a uno. Las firmas tienen preferencias por subconjuntos de trabajadores; los trabajadores tienen preferencias sobre las firmas. Una firma $F$ y un conjunto de trabajadores $C$ bloquean un emparejamiento si $F$ prefiere $C$ al conjunto con el que está emparejada y cada trabajador de $C$ prefiere $F$ a la firma que le tocó.

**Ejemplo con dos firmas y tres trabajadores:**

$$P(F_1) = \{w_1,w_3\}, \{w_1,w_2\}, \{w_2,w_3\}, \{w_1\}, \{w_2\}$$
$$P(F_2) = \{w_1,w_3\}, \{w_2,w_3\}, \{w_1,w_2\}, \{w_3\}, \{w_1\}, \{w_2\}$$
$$P(w_1) = F_2, F_1 \qquad P(w_2) = F_2, F_1 \qquad P(w_3) = F_1, F_2$$

Todo emparejamiento individualmente racional es bloqueado — no existe emparejamiento estable.

---

## 3. Mercados de dos lados: Algoritmo de aceptación diferida

### 3.1 Mercado de Matrimonios: Uno a uno

**Teorema (Gale y Shapley):** Existe un emparejamiento estable para todo mercado de matrimonios.

#### Algoritmo de aceptación diferida

1. Cada hombre le propone matrimonio a la mujer que más prefiere entre las aceptables para él.
2. Las mujeres rechazan las ofertas de todos los hombres que no son aceptables o que no son los preferidos entre las ofertas recibidas; se comprometen temporalmente con la mejor oferta.
3. Los hombres rechazados hacen ofertas a la siguiente mujer más preferida entre las que no los han rechazado anteriormente y son aceptables.
4. Se repite hasta que ningún hombre es rechazado.
5. El resultado es un emparejamiento estable.

**Demostración:** Si no fuera estable, habría una pareja que bloquea. El hombre de esa pareja prefiere a esa mujer más que a la que le tocó, por lo que debió haberle propuesto y fue rechazado por ella. Pero si ella lo rechazó, terminó en el algoritmo con un hombre mejor (las mujeres solo mejoran en las rondas). Contradicción. $\square$

**Observaciones:**
- Si hay indiferencia, usar cualquier regla predefinida para elegir.
- El algoritmo siempre termina porque ningún hombre le propone más de una vez a una misma mujer.
- El resultado es individualmente racional.

#### Ejemplo

$$P(m_1) = w_1,w_2,w_3,w_4 \qquad P(w_1) = m_2,m_3,m_1,m_4,m_5$$
$$P(m_2) = w_4,w_2,w_3,w_1 \qquad P(w_2) = m_3,m_1,m_2,m_4,m_5$$
$$P(m_3) = w_4,w_3,w_1,w_2 \qquad P(w_3) = m_5,m_4,m_1,m_2,m_3$$
$$P(m_4) = w_1,w_4,w_3,w_2 \qquad P(w_4) = m_1,m_4,m_5,m_2,m_3$$
$$P(m_5) = w_1,w_2,w_4$$

Cuando el hombre propone: $\mu_M = \begin{pmatrix} w_1 & w_2 & w_3 & w_4 & (m_5) \\ m_1 & m_2 & m_3 & m_4 & m_5 \end{pmatrix}$

Cuando la mujer propone: $\mu_W = \begin{pmatrix} w_4 & w_1 & w_2 & w_3 & (m_5) \\ m_1 & m_2 & m_3 & m_4 & m_5 \end{pmatrix}$

**Definición:** $\mu$ estable es **óptimo para los hombres** si ningún hombre prefiere otro emparejamiento estable.

**Teorema (Gale-Shapley, Optimalidad):** Si las preferencias son estrictas, siempre existe un emparejamiento óptimo para los hombres (el emparejamiento en el que los hombres proponen).

> **Nota:** Cuando las preferencias no son estrictas, puede no existir un emparejamiento estable óptimo para ningún lado del mercado.

**Optimalidad de Pareto:** La asignación óptima para los hombres es óptima de Pareto en un sentido débil para los hombres (resultado análogo para las mujeres).

### 3.2 Comportamiento Estratégico en el Mercado de Matrimonios

Cuando el mecanismo es el algoritmo DA con hombres proponiendo, **los hombres no tienen incentivos a manipular** sus preferencias reportadas (resultado análogo para las mujeres cuando ellas proponen). Sin embargo, el lado que no propone sí puede tener incentivos a mentir.

**Ejemplo de manipulación:** Con las preferencias verdaderas, el resultado del DA (hombres proponen) empareja $w_1$ con $m_1$. Si $w_1$ reporta preferencias manipuladas (declara a $m_1$ inaceptable), el nuevo resultado la empareja con $m_3$ — una mejor pareja según sus preferencias verdaderas.

### 3.3 Admisiones a la Universidad: Muchos a uno

Conjunto de estudiantes $S = \{s_1, \ldots, s_n\}$ y universidades $U = \{u_1, \ldots, u_p\}$. Cada universidad $i$ tiene una cuota $q_i$ de estudiantes que pueden ingresar.

**Extensión del algoritmo:** Las universidades (análogas a mujeres) pueden aceptar temporalmente estudiantes hasta su cuota. En cada ronda rechazan estudiantes inaceptables y aceptan temporalmente hasta su cuota. La asignación resultante es estable.

**Con preferencias estrictas, la asignación es óptima para los estudiantes.**

**Demostración de optimalidad para estudiantes:** Por inducción se muestra que el DA nunca rechaza una universidad "posible" para un estudiante (posible = existe algún emparejamiento estable en el que ese estudiante va a esa universidad). Si en alguna ronda un estudiante es rechazado por $u$ que prefiere temporalmente a $s_1, \ldots, s_m$, asumir que eso es posible lleva a contradicción con la hipótesis de inducción o con la estabilidad de $\mu$.

**No optimalidad para las universidades:** A diferencia del caso uno a uno, en el caso muchos a uno el DA (universidades proponen) puede no ser óptimo de Pareto en sentido débil para las universidades.

---

## 4. ¿Qué son las matemáticas?

> *What, then, to raise the old question once more, is mathematics? The answer, it appears, is that any argument which is carried with sufficient precision is mathematical.*
>
> — Gale and Shapley. 1962. *College Admissions and the Stability of Marriage*

---

## 5. Automated Mechanism Design

### Modelo

- $N = \{1, \ldots, n\}$ conjunto de agentes.
- $\Omega$ conjunto de resultados.
- $\succ$ relación de preferencia estricta. El conjunto de todas las relaciones de preferencia $\mathcal{P}$.
- Un **mecanismo** es una función $f : \mathcal{P} \to \Omega$.
- Los agentes reportan sus preferencias (posiblemente manipuladas) y el mecanismo determina el resultado.
- Las preferencias de los agentes se generan con una distribución $\mathcal{D}$.

Sea $g : \mathcal{P} \to \Omega$ un objetivo que puede no satisfacer las restricciones de diseño. El problema es encontrar un mecanismo $f : \mathcal{P} \to \Omega$ que sí satisfaga las restricciones de diseño (e.g., estabilidad) y sea lo más aproximado a $g$. La cercanía se mide con una función $D : \Omega \times \Omega \to \mathbb{R}$.

**El problema de diseño de mecanismos:**

$$\min_{f \in \mathcal{F}}\ \mathbb{E}_{\succ \sim \mathcal{D}}[D(g(\succ), f(\succ))] \tag{11}$$

Versión empírica (generando $m$ muestras):

$$\min_{f \in \mathcal{F}}\ \frac{1}{m} \sum_{i=1}^{m} D(g(\succ^i), f(\succ^i)) \tag{12}$$

> Referencia: *Automated mechanism design without money via machine learning.* 2016. Narasimhan, Agarwal, Parkes.

### Representación Lineal de Emparejamientos Estables

Un emparejamiento se representa como una matriz booleana $y \in \{0,1\}^{n \times m}$ donde $y_{hm} = 1$ sii $(h, m)$ están emparejados, con filas y columnas sumando $\leq 1$.

El conjunto de emparejamientos estables $\Omega(\succ)$ satisface:

$$y_{hm} = 0,\quad (h,m) \notin A(\succ) \tag{13}$$

$$y_{hm} + \sum_{h' \succ_m h} y_{h'm} + \sum_{m' \succ_h m} y_{hm'} \geq 1 \tag{14}$$

La restricción (13) garantiza racionalidad individual; la (14) se viola solo cuando una pareja bloquea.

**Teorema (Roth):** Un emparejamiento es estable si y solo si es un punto extremo del politopo $\hat{\Omega}(\succ)$ (la relajación continua de $\Omega(\succ)$).

### Weighted Polytope Rules (WP-Rule)

**Definición:** Dada una función de pesos $\lambda : \mathcal{P} \to \mathbb{R}^{n \times n}$:

$$f_{WP}(\succ;\, \lambda) = \operatorname{argmax}_{y \in \hat{\Omega}(\succ)}\ \sum_m \sum_h \lambda_{mh}(\succ)\, y_{mh} \tag{15}$$

**Teorema (Weighted Polytope Rules):** $f_{WP}(\succ, \lambda)$ son estables y contienen las asignaciones que se obtienen del algoritmo de aceptación diferida.

#### Función de pesos parametrizada

Se define la **función de rango**:

$$\operatorname{rank}(\succ_h, m) = |\{m' \in M : m \succ_h m'\}| \tag{16}$$
$$\operatorname{rank}(\succ_m, h) = |\{h' \in H : h \succ_m h'\}| \tag{17}$$

Casos especiales:
- $\lambda_{hm}(\succ) = \operatorname{rank}(\succ_h, m)$ → emparejamiento óptimo para hombres
- $\lambda_{hm}(\succ) = \operatorname{rank}(\succ_m, h)$ → emparejamiento óptimo para mujeres

Forma parametrizada que interpola entre ambos extremos:

$$\lambda_{hm}(\succ;\, W) = a_{hm}\operatorname{rank}(\succ_h,m) + b_{hm}\operatorname{rank}(\succ_m,h) + c_{hm} \tag{18}$$

con $W = [a, b, c]$, $a, b, c \in \mathbb{R}^{n \times n}$.

El problema empírico se resuelve usando SVM estructural (StructSVM). En principio pueden utilizarse Redes Neuronales Profundas.

---

## 6. Resultados

### Datos Sintéticos

- Problema uno-a-uno con **10 hospitales y 10 doctores**; preferencias con posible indiferencia (ties).
- Las preferencias se generan con parámetro $\alpha \in [0,1]$ que controla la correlación:
  - $\alpha$ alto → preferencias muy correlacionadas → **menos emparejamientos estables**
  - $\alpha$ bajo → preferencias heterogéneas → mayor flexibilidad para aproximar el target
- **Dos reglas objetivo no estables (targets):**
  - *Equal-weighted Hungarian:* maximiza $\sum[\operatorname{rank}(h,m) + \operatorname{rank}(m,h)]$
  - *Diversity-inducing Hungarian:* igual, más un bono por emparejar grupos específicos
- 1000 preferencias generadas, divididas en entrenamiento, validación y prueba (1/3 cada una).

**Interpretación:**

- A mayor $\alpha$, el error crece para todos los métodos.
- Para la regla *equal-weighted*: StructSVM-WP recupera automáticamente una WP-Rule con pesos iguales, superando a las reglas DA.
- Para la regla *diversity-inducing*: la WP-Rule entrenada supera a los tres baselines incluso con $\alpha$ grande.
- StructSVM-WP domina consistentemente a: DA doctor propone, DA hospital propone, y WP pesos iguales sin entrenar.

### Datos Reales: Elección de Escuelas (Wake County, NC)

- **Datos:** 37 escuelas y 5504 estudiantes del sistema escolar público de Wake County, Carolina del Norte.
- Se estima un modelo Plackett-Luce de preferencias; se generan preferencias para **5 escuelas, 100 estudiantes** (cuota de 25 por escuela).
- **Indiferencias:** estudiantes con 1–2 opciones reportadas; escuelas con rangos agrupados en 20 categorías.
- **Cuatro reglas objetivo no estables:**
  - **BM:** Mecanismo de Boston
  - **EH:** Hungarian con pesos iguales
  - **MH:** Hungarian con pesos mayores a 1/3 de estudiantes con estatus de minoría
  - **DH:** Hungarian diversity-inducing
- 200 ejemplos generados, divididos en entrenamiento y prueba (100 cada uno).

**Resultados — Error de Hamming en prueba (menor es mejor):**

| Método | BM | EH | MH | DH |
|--------|---:|---:|---:|---:|
| StructSVM-WP | **27.2** | 36.0 | **22.1** | **47.7** |
| DA (estudiante propone) | 28.4 | 46.5 | 33.4 | 57.3 |
| WP pesos iguales | 29.7 | **34.3** | 23.0 | 53.2 |

- StructSVM-WP supera a los baselines en **3 de 4** reglas objetivo.
- Para EH, la WP con pesos iguales ya aproxima bien el target (el propio target asigna pesos iguales).