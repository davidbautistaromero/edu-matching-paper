# Contexto: Paper de Matching Escolar en Bogotá — Sesión de trabajo BLP

## 1. Descripción del proyecto

Paper de economía sobre elección escolar en Bogotá. El objetivo es estimar cómo familias de distinto nivel de ingreso reaccionan a **señales visuales del entorno escolar** (fachadas + entorno inmediato capturado con Google Street View en 360° sobre la ubicación del colegio).

---

## 2. Datos disponibles

| Dato | Descripción |
|---|---|
| ~370 colegios públicos | 19 localidades de Bogotá |
| Demanda agregada | Total de solicitudes por colegio — **no elecciones individuales** |
| Variables visuales | Tópicos de embeddings CLIP sobre imágenes Street View (fachada + entorno inmediato) |
| Calidad académica | Puntaje Saber 11 |
| Controles | Homicidios, transporte, zona |
| Encuesta Multipropósito 2021 | ~13,500 familias con ingreso per cápita, estrato, ubicación lat/lon, UPZ |
| Distancias $d_{ij}$ | Calculadas por Haversine, familia → colegio. Las familias fueron ubicadas aleatoriamente en manzanas de su estrato/UPZ |
| Precios | No hay — todos los colegios son públicos y gratuitos |

**Restricción clave de diseño**: las familias solo pueden elegir colegios dentro de su propia localidad. Esto define los 19 mercados y el denominador de las cuotas de mercado.

---

## 3. Parámetros de interés

- $\pi_k$: cómo varía el peso del tópico visual $k$ en la utilidad según el ingreso de la familia
  - $\pi_k > 0$ → familias ricas valoran más el tópico $k$
  - $\pi_k < 0$ → familias pobres reaccionan más al tópico $k$
- $\lambda_1$: cómo varía la sensibilidad a la distancia según el ingreso
  - $\lambda_1 > 0$ → familias ricas sienten menos el costo de la distancia

---

## 4. El modelo de utilidad

$$u_{ij} = \underbrace{\beta_1 \text{Saber11}_j + \beta_2 \text{homicidios}_j + \beta_3 \text{transporte}_j + \sum_k \beta_k x_{jk}}_{\delta_j \text{ — igual para todas las familias}} + \underbrace{\sum_k \pi_k y_i x_{jk}}_{\text{¿ricos valoran más el tópico }k\text{?}} + \underbrace{\lambda_0 d_{ij}}_{\text{costo base de distancia}} + \underbrace{\lambda_1 y_i d_{ij}}_{\text{¿ricos sienten menos la distancia?}} + \epsilon_{ij}$$

Donde:
- $\delta_j$ es la utilidad media del colegio $j$ — igual para todas las familias
- $x_{jk}$ son los tópicos visuales del entorno escolar
- $y_i$ es el ingreso per cápita de la familia $i$
- $d_{ij}$ es la distancia familia $i$ → colegio $j$
- $\epsilon_{ij}$ es Gumbel iid

**Nota importante**: no se incluye un $\nu$ de heterogeneidad aleatoria adicional porque $d_{ij}$ ya es individual — genera heterogeneidad directamente sin necesitar simulación de gustos latentes.

### Qué estima cada parámetro

| Parámetro | Pregunta que responde |
|---|---|
| $\beta_k$ | ¿En promedio, todos prefieren colegios con tópico $k$? |
| $\pi_k$ | ¿Esa preferencia es mayor si la familia es rica? |
| $\lambda_0$ | ¿En promedio, todos evitan colegios lejanos? |
| $\lambda_1$ | ¿Ese rechazo a la distancia es menor si la familia es rica? |

---

## 5. Estrategia de estimación: BLP con micro-momentos

### 5.1 Por qué BLP

- La heterogeneidad en preferencias hace la cuota de mercado no lineal en parámetros — no hay forma cerrada
- BLP resuelve esto con una contraction mapping que recupera $\delta_j$ sin necesitar elecciones individuales
- La variante BLP (2004) con micro-momentos permite identificar heterogeneidad usando microdatos de una encuesta externa

### 5.2 La contraction mapping

Para cada localidad (mercado), se itera:

$$\delta_j^{(h+1)} = \delta_j^{(h)} + \ln S_j^{obs} - \ln \hat{S}_j(\delta^{(h)})$$

Donde $\hat{S}_j = \frac{1}{N}\sum_i s_{ij}$ y $s_{ij} = \frac{\exp(\delta_j + \sum_k \pi_k y_i x_{jk} + \lambda_0 d_{ij} + \lambda_1 y_i d_{ij})}{1 + \sum_\ell \exp(\delta_\ell + \sum_k \pi_k y_i x_{\ell k} + \lambda_0 d_{i\ell} + \lambda_1 y_i d_{i\ell})}$

Esto recupera $\delta_j$ como un número por colegio, sin descomponerlo todavía.

### 5.3 Recuperación de parámetros

Una vez convergida la contraction mapping:

$$\xi_j = \delta_j - \beta_1 \text{Saber11}_j - \beta_2 \text{homicidios}_j - \beta_3 \text{transporte}_j - \sum_k \beta_k x_{jk}$$

Los $\beta$ salen de una regresión IV de $\delta_j$ sobre las características del colegio.

### 5.4 GMM y momentos

El estimador minimiza $\xi(\theta)' Z (Z'Z)^{-1} Z' \xi(\theta)$ con dos tipos de momentos:

**Momentos de demanda agregada:**
$$E[\xi_j \cdot Z_j] = 0$$

**Micro-momentos (BLP 2004):**
$$E[m_j^{pred}(\theta) - m_j^{obs}] = 0$$
$$E[d_j^{pred}(\theta) - d_j^{obs}] = 0$$

Donde:
- $m_j^{obs}$ = ingreso promedio ponderado por $1/d_{ij}$ de familias de la Encuesta Multipropósito en la misma localidad del colegio $j$
- $d_j^{obs}$ = distancia promedio ponderada de esas mismas familias al colegio $j$
- El primer micro-momento ancla $\pi_k$
- El segundo micro-momento ancla $\lambda_1$ y evita confundir los dos parámetros

### 5.5 El algoritmo completo

```
Loop externo — optimizador busca (πk, λ₀, λ₁):

    1. Dado (πk, λ₀, λ₁), calcular sij para cada familia y colegio
    
    2. Contraction mapping por localidad:
       δj converge cuando Σᵢ sij = Sj observado
    
    3. Regresión IV de δj sobre características del colegio
       → obtener βk y ξj
    
    4. Evaluar momentos GMM:
       g(θ) = ξj · Zj                          (demanda agregada)
              mj_pred - mj_obs                  (micro-momento ingreso)
              dj_pred - dj_obs                  (micro-momento distancia)
    
    5. Actualizar (πk, λ₀, λ₁) para reducir criterio GMM

Repetir hasta convergencia
```

### 5.6 Instrumentos candidatos

| Instrumento | Lógica |
|---|---|
| Número de colegios en la localidad | Variación en el outside option |
| Distancia al segundo colegio más cercano | Intensidad de competencia local |
| Tópicos visuales del competidor más cercano | Afecta sustitución pero no $\xi_j$ |

---

## 6. Problemas de identificación — estado actual

| Problema | Riesgo | Estado |
|---|---|---|
| Sorting residencial dentro de localidad | **Alto — principal amenaza** | Pendiente test de balance |
| Confusión entre $\pi_k$ y $\lambda_1$ | Alto | Mitigado con dos micro-momentos |
| $m_j^{obs}$ mide vecino, no elector | Medio | **Resuelto** por restricción a localidad |
| Denominador del mercado | Medio | **Resuelto** — familias de la localidad |
| Tópicos capturan vecindario, no colegio | Medio | **Reenmarcado** (ver sección 7) |
| Variación de tópicos dentro de localidad | Medio | Análisis preliminar sugiere que no hay concentración |

### Test de balance pendiente

Regresión de tópicos sobre ingreso promedio de la UPZ:

$$x_{jk} = \gamma_0 + \gamma_1 \bar{y}_{UPZ(j)} + \eta_j$$

Si $\gamma_1$ es pequeño e insignificante → argumento fuerte de que la infraestructura visual no siguió al ingreso (plausible dado que es infraestructura pública histórica).

---

## 7. Decisión conceptual: "señales visuales del entorno escolar"

Las imágenes de Street View capturan fachada + entorno inmediato del colegio. Esto es **una fortaleza, no una limitación**, porque:

- Las familias no evalúan solo el edificio — evalúan el entorno completo: seguridad percibida de la calle, estado del andén, tipo de comercio, zonas verdes
- La distinción entre "infraestructura del colegio" y "entorno del colegio" es artificial desde el punto de vista de la decisión familiar
- Hay respaldo en la literatura de *neighborhood effects* en educación

**Concepto adoptado**: *señales visuales del entorno escolar* (no "infraestructura interna del colegio")

Lo que **no** se puede afirmar: que los tópicos miden calidad interna del plantel.

---

## 8. Referencias clave identificadas

- Berry, Levinsohn & Pakes (1995) — BLP original
- Berry, Levinsohn & Pakes (2004) — micro-momentos, *JPE*
- Petrin (2002) — micro-momentos con encuesta externa, *JPE* — caso análogo al nuestro
- Gowrisankaran & Town (1999) — elección hospitalaria sin precios, análogo estructural
- Neilson (2021) — sistemas escolares latinoamericanos con problemas similares de identificación
- Hastings, Kane & Staiger (2005) — elección escolar con heterogeneidad por ingreso

---

## 9. Pasos pendientes

1. Confirmar heterogeneidad de tópicos visuales dentro de cada localidad
2. Correr test de balance: $x_{jk}$ sobre $\bar{y}_{UPZ(j)}$
3. Construir $m_j^{obs}$ y $d_j^{obs}$ desde la Encuesta Multipropósito
4. Implementar contraction mapping con distancia individual
5. Definir instrumentos finales y testear relevancia
