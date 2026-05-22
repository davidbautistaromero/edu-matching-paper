# Estructura del paper

**Título tentativo:** Señales Visuales, Estimación de Demanda y Diseño de Mecanismos en la Asignación Escolar de Bogotá

---

## Abstract

Las familias que eligen colegio público observan señales visuales (fachadas, entorno urbano) que no reflejan calidad académica. Estimamos un modelo de demanda escolar que incorpora estas señales — extraídas de imágenes Google Street View mediante NMF sobre embeddings VGG19 — y cuantificamos su efecto heterogéneo por ingreso. Los resultados, validados con simulaciones Monte Carlo, muestran que el sesgo visual es robusto y regresivo: familias de menores ingresos son más sensibles a señales de deterioro urbano. Proponemos un mecanismo de asignación aprendido (Weighted Polytope Rule) que es estable por construcción y reduce el sesgo visual sin sacrificar eficiencia. Evaluamos este mecanismo contra Boston, Deferred Acceptance y el sistema actual de la Secretaría de Educación de Bogotá sobre datos reales de 97,968 familias.

---

## 1. Introducción

- La asignación escolar es un problema de matching: familias tienen preferencias, colegios tienen capacidad
- Las preferencias declaradas no reflejan solo calidad — incorporan señales observables del entorno físico
- En Bogotá, la infraestructura urbana está espacialmente segregada por ingreso → las señales visuales amplifican la desigualdad
- Tres preguntas:
  1. ¿Cuánto pesan las señales visuales en la demanda escolar?
  2. ¿Es ese efecto heterogéneo por ingreso?
  3. ¿Se puede diseñar un mecanismo que mitigue el sesgo visual sin perder estabilidad?
- Contribución triple: metodológica (CV + estimación estructural), empírica (cuantificación robusta), diseño (mecanismo aprendido)

---

## 2. Contexto institucional

- Sistema de asignación escolar de Bogotá (Resolución 1587/2025 de la SED)
- Prioridad lexicográfica: grupo SISBEN (A > B > C > D) → distancia → lotería
- 306 establecimientos oficiales, ~100K familias
- Segregación espacial: colegios en el sur tienen peor infraestructura visual Y concentran familias de menores ingresos

---

## 3. Datos

### 3.1 Colegios
- Directorio SED: ubicación, matrícula, capacidad
- ICFES Saber 11: puntaje promedio (proxy de calidad académica q_j)
- Demanda y matrícula: sobredemanda como proxy de preferencia revelada
- Controles urbanos: delitos, SITP, parques, competencia privada, estrato por localidad

### 3.2 Imágenes y features visuales
- Google Street View: 5,580 imágenes (558 sedes × 10 headings)
- Pipeline: VGG19 → PCA(68d) → NMF(K=8) → 8 tópicos visuales por establecimiento
- Interpretación de tópicos (sección 4.1)
- Alternativas descartadas: LDA (colapsa), Cityscapes y CLIP (no sobreviven regularización)

### 3.3 Familias
- Encuesta Multipropósito 2021 (EM2021): 21,643 hogares con hijos en colegio oficial
- Variables clave: ingreso per cápita (N_ingpc), estrato real, UPZ, localidad
- Expansión con factor FEX_C → 97,968 familias representativas
- Ubicación en manzana por estrato + distancias Haversine a 303 colegios

---

## 4. Señales visuales de los colegios

### 4.1 Tópicos NMF — interpretación
- Descripción de los 8 tópicos con las imágenes de mayor peso
- Etiquetas semánticas y ejemplos visuales

### 4.2 Regresión reducida
- Ridge M1: log(sobredemanda) ~ q_j + tópicos NMF + controles
- topic_1 (+) y topic_2 (−) son los tópicos con mayor señal
- R²_adj = 0.203, Spearman(CV) = 0.466
- Las señales visuales tienen poder predictivo sobre la demanda incluso controlando por calidad académica

---

## 5. Estimación estructural de demanda

### 5.1 Modelo
- Utilidad: u_ij = δ_j + μ_ij + ε_ij
- Utilidad media: δ_j = β_q·q_j + Σ_k β_k·topic_k_j + controles + ξ_j
- Heterogeneidad: μ_ij = α·d_ij + π·(ingreso_i × d_ij) + Σ_k π_k·(ingreso_i × topic_k_j)
- ε_ij ~ Gumbel(0,1)

### 5.2 Logit puro (Berry inversion)
- Market shares: s_j = matrícula_j / M_t (mercado = localidad)
- Outside option: s_0 = familias en privado o no asisten
- Inversión: log(s_j/s_0) = δ_j
- Estimación de β_q, β_k por OLS/IV

### 5.3 Logit mixto (BLP)
- Implementación con PyBLP
- Agent data: familias EM2021 con ingreso como demográfico
- Estimación de π (interacción ingreso × distancia) y π_k (interacción ingreso × tópico visual)
- Identificación: variación cross-localidad en composición de colegios y familias
- No hay endogeneidad de precio (educación gratuita)

### 5.4 Resultados
- Elasticidades de distancia por decil de ingreso
- WTP por calidad vs señal visual
- ¿Qué tópicos pesan más para familias de bajos ingresos? → diagnóstico desagregado del sesgo

---

## 6. Mecanismos de asignación

### 6.1 Mecanismos evaluados
- **Boston Mechanism (BM):** eficiente pero manipulable, genera blocking pairs
- **Deferred Acceptance (DA):** strategy-proof, estable, student-optimal entre los estables
- **SED-lex:** prioridad SISBEN lexicográfica + distancia (mecanismo actual de Bogotá)
- **WP-Rule:** mecanismo aprendido, estable por construcción, optimiza equidad visual

### 6.2 Resultados sobre datos reales
- Tabla comparativa: asignados, rank medio, blocking pairs, corr(ingreso, q_j), corr(ingreso, v_j)
- BM/DA/SED: el mecanismo no suprime el sesgo visual — el problema está en las preferencias
- WP-Rule: reduce corr(ingreso, v_j) sin empeorar eficiencia ni estabilidad

---

## 7. Experimento sintético

### 7.1 Diseño
- DGP calibrado con parámetros estimados de la sección 5
- Dos escenarios: correlación realista (v_j, q_j) y ortogonal (v_j ⊥ q_j)
- Monte Carlo: 100 réplicas por configuración

### 7.2 Resultados
- Media ± IC 95% de sesgo visual, equidad, ranking por mecanismo
- Escenario ortogonal: identificación limpia — cualquier corr(ingreso, v_j) es sesgo puro
- WP-Rule domina en reducción de sesgo manteniendo estabilidad

### 7.3 Robustez
- Sensibilidad a γ₀ ∈ {0.25, ..., 1.50} con IC
- El ordenamiento WP > SED > DA > BM en equidad visual se mantiene para todo γ₀

---

## 8. Discusión

- El sesgo visual es un canal de transmisión de desigualdad urbana hacia desigualdad educativa
- Los mecanismos existentes (incluso SED-lex) no están diseñados para corregir sesgos en preferencias
- WP-Rule es operacionalmente viable (LP, computable en segundos) y domina a los mecanismos existentes
- Limitaciones: outside option simplificada, mercados definidos por localidad, NMF vs alternativas de CV más recientes
- Implicaciones de política: la inversión en infraestructura escolar visible puede tener efectos sobre la equidad en la asignación

---

## 9. Conclusión

- Tres contribuciones: (1) integración de CV y estimación de demanda escolar, (2) cuantificación robusta del sesgo visual y su regresividad, (3) mecanismo aprendido que mitiga el sesgo
- Agenda futura: RegretNet (redes neuronales para mecanismos), extensión a otras ciudades, datos longitudinales

---

## Referencias

(ver `notas_metodologicas.md` para lista completa actual)

---

## Apéndices

- A. Detalles del pipeline visual (VGG19, PCA, NMF, Cityscapes, CLIP)
- B. Derivación del modelo de utilidad y calibración de α
- C. Restricciones del politopo estable (formulación LP de la WP-Rule)
- D. Tablas completas de resultados Monte Carlo
- E. Figuras adicionales: tópicos NMF, mapas de segregación, curvas de robustez
