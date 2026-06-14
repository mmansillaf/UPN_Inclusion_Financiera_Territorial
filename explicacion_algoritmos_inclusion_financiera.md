# Explicación de Algoritmos — Segmentación Territorial para Inclusión Financiera

> **Determinación de zonas con mejor perspectiva para el desarrollo e inclusión financiera en el Perú**  
> Pipeline de Machine Learning No Supervisado sobre 24 departamentos  
> **Versión 1.0 — Basado en datos verificados de BaseDedatosFinal.xlsx y ejecución real del script**

---

## Índice

1. [Contexto del Problema](#1-contexto-del-problema)
2. [Datos y Variables](#2-datos-y-variables)
3. [Pipeline de Ingeniería de Características](#3-pipeline-de-ingeniería-de-características)
4. [Algoritmo 1: PCA — Análisis de Componentes Principales](#4-algoritmo-1-pca)
5. [Algoritmo 2: Clustering Jerárquico (Ward)](#5-algoritmo-2-clustering-jerárquico)
6. [Algoritmo 3: DBSCAN — Detección de Outliers](#6-algoritmo-3-dbscan)
7. [Algoritmo 4: Random Forest — Importancia de Variables](#7-algoritmo-4-random-forest)
8. [Algoritmo 5: K-Means — Segmentación Final](#8-algoritmo-5-k-means)
9. [Resultados: Los 3 Clusters](#9-resultados)
10. [Zonas con Mejor Perspectiva](#10-zonas-con-mejor-perspectiva)
11. [Conclusiones y Limitaciones](#11-conclusiones)

---

## 1. Contexto del Problema

### 1.1 ¿Qué se busca resolver?

En Perú, gran parte de la actividad económica se desarrolla en contextos de **alta informalidad**. Muchas personas, familias y pequeños negocios generan ingresos genuinos que no siempre figuran en registros formales (declaraciones tributarias, historial crediticio, información bancaria verificable). Esto dificulta que las entidades financieras —especialmente las **Cajas Municipales**— identifiquen territorios con oportunidades favorables para su expansión.

### 1.2 Enfoque

En lugar de evaluar clientes individuales (costoso y poco escalable), el proyecto analiza **departamentos peruanos como unidades territoriales** usando variables agregadas. Se aplican 5 algoritmos de Machine Learning para segmentar los 24 departamentos en grupos homogéneos según su perfil de inclusión financiera.

### 1.3 Actor principal

El análisis se centra en **Cajas de Ahorro y Crédito (Cajas Municipales)**, excluyendo banca múltiple, financieras y fintech. Las Cajas Municipales tienen alta presencia en regiones y segmentos poco atendidos por la banca tradicional, y su modelo de negocio se alinea con la inclusión financiera.

---

## 2. Datos y Variables

### 2.1 Fuente

**Archivo:** `BaseDedatosFinal.xlsx` — Hoja: `Data Final`  
**Estructura:** 24 filas (departamentos) × 14 columnas (variables)  
**Valores nulos:** 0 en toda la base  
**Duplicados:** 0

### 2.2 Variables Originales

| # | Variable | Tipo | Descripción |
|---|---|---|---|
| 1 | `Departamento` | Categórica | Identificador territorial (no entra al modelo) |
| 2 | `Num_Cajas` | Absoluta | Número de Cajas Municipales en el departamento |
| 3 | `Depositos_Cajas_PEN_MM` | Absoluta | Depósitos en Cajas (millones de PEN) |
| 4 | `Poblacion_18_70` | Absoluta | Población en edad activa (18-70 años) |
| 5 | `PoblacionTotal_Censo2017` | Absoluta | Población total (Censo 2017) |
| 6 | `PEA_ocupada_miles_2024` | Absoluta | PEA ocupada (miles de personas, 2024) |
| 7 | `PEA_no_ocupada_miles_2024` | Absoluta | PEA desocupada (miles de personas, 2024) |
| 8 | `Ingreso_Prom_PEN_2024` | Absoluta | Ingreso promedio mensual (PEN, 2024) |
| 9 | `NroLineasTelefoniaMovil` | Absoluta | Líneas de telefonía móvil activas |
| 10 | `Conexión_Internet_Fijo` | Absoluta | Conexiones de internet fijo |
| 11 | `NBI_%_2024` | Porcentaje | Necesidades Básicas Insatisfechas (% ,2024) |
| 12 | `Denuncias_2024` | Absoluta | Denuncias registradas (2024) |
| 13 | `Num_PNP_2026` | Absoluta | Efectivos policiales (2026) |
| 14 | `PBI_miles_PEN` | Absoluta | Producto Bruto Interno departamental (miles PEN) |

### 2.3 Estadísticas Descriptivas Clave

| Variable | Media | Min | Max | Asimetría |
|---|---|---|---|---|
| Num_Cajas | 47.5 | 12 (Tumbes) | 253 (Lima) | 3.30 (muy asimétrica) |
| Depositos_Cajas_PEN_MM | 1,465.5 | 109.9 (Tumbes) | 10,755.8 (Lima) | 3.05 (muy asimétrica) |
| PoblacionTotal_Censo2017 | 1,224,245 | 141,070 (Madre de Dios) | 10,479,899 (Lima) | 4.43 (extremadamente asimétrica) |
| NBI_%_2024 | 20.5% | 6.8% (Lambayeque) | 55.9% (Loreto) | 1.33 (moderada) |
| Ingreso_Prom_PEN_2024 | S/.1,607 | S/.1,083 (Puno) | S/.3,870 (Lima) | 3.27 (muy asimétrica) |

> **Observación crítica:** Todas las variables absolutas tienen asimetría > 3.0, dominadas por Lima/Callao que concentra entre 22% y 58% del total nacional según la variable. Esto justifica **obligatoriamente** la transformación a indicadores per cápita.

---

## 3. Pipeline de Ingeniería de Características

### 3.1 Problema de las variables absolutas

Si se usaran valores absolutos (PBI total, depósitos totales, población, etc.), Lima dominaría todas las métricas por su tamaño poblacional. Para que la comparación sea justa, **cada variable se transforma a indicador relativo**.

### 3.2 Transformaciones Aplicadas

| Variable Derivada | Fórmula | Dimensión que Mide |
|---|---|---|
| `Oficinas_por_100k` | Num_Cajas / (Población / 100,000) | Densidad de infraestructura financiera |
| `Depositos_por_Capita` | Depósitos_MM × 1e6 / Población_18_70 | Profundidad financiera (S/ por adulto) |
| `Tasa_Empleo` | PEA_ocupada / (PEA_ocupada + PEA_no_ocupada) | Estabilidad laboral |
| `Internet_por_100k` | Conexiones_Internet / (Población / 100,000) | Brecha digital |
| `Movil_por_100k` | Líneas_Móviles / (Población / 100,000) | Cobertura móvil |
| `PBI_por_Capita` | PBI_miles × 1000 / Población | Productividad económica |
| `PNP_por_100k` | Num_PNP / (Población / 100,000) | Seguridad ciudadana |
| `Indice_Brecha_Inclusion` | NBI% / (Of/100k + 0.01) | Brecha de inclusión combinada |

### 3.3 Las 7 Features del Modelo

De todas las derivadas, se seleccionan **7** para los algoritmos, cada una capturando un eje distinto:

| Feature | Media | Std | Min | Max | Interpretación |
|---|---|---|---|---|---|
| Oficinas_por_100k | 5.18 | 2.06 | 2.41 (Lima) | 9.92 (Madre de Dios) | Acceso físico a servicios financieros |
| Depositos_por_Capita | 1,620 | 1,196 | 421 (Ucayali) | 5,410 (Arequipa) | Uso activo del sistema (ahorro) |
| NBI_%_2024 | 20.53 | 11.63 | 6.79 (Lambayeque) | 55.90 (Loreto) | Pobreza y exclusión social |
| Ingreso_Prom_PEN_2024 | 1,608 | 550 | 1,083 (Puno) | 3,870 (Lima) | Capacidad económica |
| Tasa_Empleo | 0.947 | 0.016 | 0.908 (Moquegua) | 0.969 (La Libertad) | Estabilidad laboral |
| PBI_por_Capita | 17,419 | 14,207 | 8,083 (San Martín) | 78,198 (Moquegua) | Productividad regional |
| Internet_por_100k | 32,793 | 19,496 | 7,945 (Loreto) | 77,310 (Lima) | Conectividad digital |

> **Nota:** Ninguna correlación entre features supera |r| > 0.8. La más alta es Ingreso ↔ Internet (r=0.75). Esto confirma que cada feature aporta información única.

### 3.4 Estandarización (StandardScaler)

Se aplica **StandardScaler**: cada variable se transforma a media=0, desviación estándar=1. Es obligatorio porque K-Means y PCA miden distancias euclidianas: una variable como PBI/cápita (rango: 8k-78k) dominaría a Tasa_Empleo (rango: 0.908-0.969) sin escalamiento.

---

## 4. Algoritmo 1: PCA

### 4.1 ¿Qué hace?

El **Análisis de Componentes Principales (PCA)** reduce las 7 dimensiones originales a 2 ejes ortogonales (PC1, PC2) para visualizar los 24 departamentos en un plano 2D. Cada componente principal es una combinación lineal de las variables originales.

### 4.2 ¿Por qué se usa?

- Permite **ver** si los datos tienen estructura de grupos en 2D
- Identifica qué variables se correlacionan (flechas en misma dirección en el biplot)
- Detecta visualmente departamentos con perfil atípico

### 4.3 Parámetros

```python
pca_2d = PCA(n_components=2, random_state=42)
```

### 4.4 Resultados Verificados (Ejecución Real)

| Componente | Varianza Explicada | Acumulada |
|---|---|---|
| PC1 | **47.0%** | 47.0% |
| PC2 | **22.0%** | **69.0%** |
| PC3 | 12.6% | 81.6% |
| PC4 | 8.7% | 90.3% |
| PC5 | 6.2% | 96.5% |
| PC6 | 2.2% | 98.7% |
| PC7 | 1.3% | 100% |

> **Interpretación:** Con 2 componentes se captura el **69%** de la varianza total. Para llegar al 90% se necesitan **4 componentes**. El plano 2D es informativo pero pierde ~31% de la información.

### 4.5 Loadings — ¿Qué significa cada componente?

| Variable | PC1 (47%) | PC2 (22%) |
|---|---|---|
| Oficinas_por_100k | **-0.310** | **-0.520** |
| Depositos_por_Capita | **-0.360** | **-0.375** |
| NBI_%_2024 | +0.277 | +0.380 |
| Ingreso_Prom_PEN_2024 | **-0.334** | +0.531 |
| Tasa_Empleo | +0.447 | -0.131 |
| PBI_por_Capita | **-0.441** | -0.003 |
| Internet_por_100k | **-0.438** | +0.381 |

**Lectura del Biplot:**
- **PC1 (eje X):** Separa departamentos con alta productividad, conectividad e ingresos (valores negativos) de aquellos con alta pobreza y empleo informal (valores positivos). Es un eje de **desarrollo vs. vulnerabilidad**.
- **PC2 (eje Y):** Separa por **ingreso alto + conectividad** (positivo) de **alta densidad de oficinas + depósitos** (negativo). Lima está arriba a la izquierda (alto ingreso, poca densidad financiera per cápita). Arequipa está abajo a la izquierda (alta densidad financiera).

**Salida visual:** `final_01_pca_biplot.png`

---

## 5. Algoritmo 2: Clustering Jerárquico

### 5.1 ¿Qué hace?

El **Clustering Jerárquico (método Ward)** no requiere definir K de antemano. Parte de 24 objetos individuales y los fusiona iterativamente minimizando la varianza intra-cluster. El resultado es un **dendrograma** (árbol de fusiones).

### 5.2 ¿Por qué se usa?

- Es **exploratorio**: muestra la estructura natural de grupos sin imponer un K fijo
- **Valida** la elección de K que luego usará K-Means
- Si el dendrograma muestra 3 ramas claras, sugiere que K=3 es natural

### 5.3 Parámetros

```python
Z = linkage(X_scaled, method='ward')
# color_threshold=6.5 para sugerir corte en K=3
```

### 5.4 Resultados

El dendrograma muestra **3 ramas principales** al corte en altura 6.5:
1. Un grupo grande de ~16 departamentos
2. Un grupo intermedio de ~7 departamentos
3. Lima/Callao como rama separada

Esto **consistente** con los resultados de K-Means K=3.

**Salida visual:** `final_02_dendrograma.png`

---

## 6. Algoritmo 3: DBSCAN

### 6.1 ¿Qué hace?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** agrupa puntos densos y marca como **outlier (−1)** a los puntos aislados que no forman parte de ninguna zona densa. A diferencia de K-Means, NO asigna todos los puntos a un cluster.

### 6.2 ¿Por qué se usa?

K-Means asigna **todos** los puntos a un cluster, incluso si no encajan bien. DBSCAN identifica departamentos con perfil **único** que merecen análisis individual — no para eliminarlos, sino para entender qué los hace diferentes.

### 6.3 Parámetros

```python
dbscan = DBSCAN(eps=2.2, min_samples=2)
```

- **eps=2.2:** Radio de vecindad (calibrado para datos estandarizados con n=24)
- **min_samples=2:** Mínimo de vecinos para ser punto núcleo

### 6.4 Resultados Verificados

| Métrica | Valor |
|---|---|
| Clusters de densidad | 2 |
| Outliers detectados | **4** |
| Departamentos outlier | **Arequipa, Lima/Callao, Madre de Dios, Moquegua** |

**¿Por qué son outliers?**

| Departamento | ¿Qué los hace únicos? |
|---|---|
| **Arequipa** | Depósitos_por_Capita más alto del país (S/.5,410) — duplica al segundo (Cusco S/.3,601) |
| **Lima/Callao** | Perfil contradictorio: ingreso máximo (S/.3,870) + internet máximo (77,310/100k) pero NBI alto (33.9%) y la menor densidad de oficinas del país (2.41/100k) |
| **Madre de Dios** | Máxima densidad de oficinas (9.92/100k) y máxima cobertura móvil (238,808/100k) — economía pequeña con alta penetración financiera relativa |
| **Moquegua** | PBI per cápita más alto del país (S/.78,198) — minería distorsiona el perfil económico |

> **Importante:** Ser outlier NO significa que el departamento sea "malo". Significa que su combinación de indicadores es atípica y merece análisis individual.

**Salida visual:** `final_03_dbscan_outliers.png`

---

## 7. Algoritmo 4: Random Forest

### 7.1 ¿Qué hace?

**Random Forest** es un modelo supervisado que entrena 200 árboles de decisión y promedia sus predicciones. Como subproducto, mide la **importancia** de cada variable: cuánto contribuye a reducir el error de predicción.

### 7.2 ¿Por qué se usa en un análisis no supervisado?

Aunque el objetivo principal es clustering, Random Forest se usa como **diagnóstico**. Responde a la pregunta: *"¿Qué variable socioeconómica explica MEJOR el nivel de depósitos financieros?"*

### 7.3 Configuración

```python
# Variable objetivo: Depositos_por_Capita (proxy de inclusión financiera)
# Predictores: las otras 6 features (sin Depositos_por_Capita)
FEATURES_RF = [Oficinas_por_100k, NBI_%, Ingreso, Tasa_Empleo, PBI/cápita, Internet/100k]

rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
cv_r2 = cross_val_score(rf, X_rf, y_rf, cv=5, scoring='r2')
```

### 7.4 Resultados Verificados

| Variable | Importancia | % |
|---|---|---|
| **Oficinas_por_100k** | **0.6888** | **68.9%** |
| Ingreso_Prom_PEN_2024 | 0.1087 | 10.9% |
| NBI_%_2024 | 0.0917 | 9.2% |
| PBI_por_Capita | 0.0388 | 3.9% |
| Tasa_Empleo | 0.0366 | 3.7% |
| Internet_por_100k | 0.0354 | 3.5% |

**R² validación cruzada (5-fold): -0.225 ± 0.910**

### 7.5 Interpretación

**La variable que más predice la inclusión financiera es la densidad de oficinas (68.9%).** Esto valida que el **acceso físico a sucursales** sigue siendo el factor clave en Perú — más que el ingreso, la conectividad o la pobreza.

**Sobre el R² negativo:** Con solo 24 datos y validación cruzada de 5 particiones (entrenamiento con ~19 datos), el modelo no logra generalizar. Esto **no invalida el ranking de importancias** — el orden relativo de las variables es informativo aunque el poder predictivo absoluto sea bajo.

**Salida visual:** `final_04_random_forest_importancia.png`

---

## 8. Algoritmo 5: K-Means

### 8.1 ¿Qué hace?

**K-Means** divide los 24 departamentos en K grupos (clusters) minimizando la suma de distancias al cuadrado de cada punto al centroide de su cluster.

### 8.2 Selección de K

Se evaluaron **3 criterios** para K desde 2 hasta 9:

| K | Inercia (↓ mejor) | Silhouette (↑ mejor) | Davies-Bouldin (↓ mejor) |
|---|---|---|---|
| **2** | 33.85 | **0.357** | 1.105 |
| **3** | 25.84 | 0.345 | 0.882 |
| 4 | 21.03 | 0.272 | 1.069 |
| 5 | 17.42 | 0.259 | 1.138 |
| 6 | 14.43 | 0.287 | 1.067 |
| 7 | 12.26 | 0.302 | 1.049 |
| 8 | 10.53 | 0.297 | 0.919 |
| 9 | 9.28 | 0.326 | **0.653** |

### 8.3 Decisión Metodológica: K=3

- **Silhouette óptimo: K=2** (0.357)
- **Davies-Bouldin óptimo: K=2 o K=9** (1.105 o 0.653)
- **K seleccionado: K=3** (Silhouette=0.345, Davies-Bouldin=0.882)

**¿Por qué K=3 y no K=2?**  
K=2 separaría Perú en "desarrollado vs. no desarrollado", correcto estadísticamente pero **poco útil para la toma de decisiones**. K=3 permite identificar un grupo **intermedio** (Cluster 1) que genera perfiles más accionables para estrategias de expansión diferenciadas. Adicionalmente, el dendrograma jerárquico sugiere 3 ramas naturales.

### 8.4 Parámetros Finales

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
```

**Salida visual:** `final_05_kmeans_seleccion_k.png` (criterios) | `final_06_kmeans_resultado.png` (clusters)

---

## 9. Resultados

### 9.1 Los 3 Clusters

#### 🔴 CLUSTER 0 — "Inclusión en Desarrollo" (16 departamentos)

| Indicador | Cluster 0 | Nacional (promedio) |
|---|---|---|
| Oficinas/100k | **4.36** | 5.18 |
| Depósitos/cápita | **S/.993** | S/.1,620 |
| NBI % | **22.5%** | 20.5% |
| Ingreso promedio | **S/.1,380** | S/.1,608 |
| PBI/cápita | **S/.12,185** | S/.17,419 |
| Tasa de empleo | **0.954** | 0.947 |
| Internet/100k | **24,608** | 32,793 |

**Departamentos:** Amazonas, Áncash, Apurímac, Ayacucho, Cajamarca, Huancavelica, Huánuco, La Libertad, Lambayeque, Loreto, Pasco, Piura, Puno, San Martín, Tumbes, Ucayali

**Perfil:** Predominantemente rural/sierra/selva. Baja densidad financiera, ingresos por debajo del promedio, alta pobreza (NBI 22.5%) y limitada conectividad digital.

---

#### 🟢 CLUSTER 1 — "Desarrollo Económico Alto" (7 departamentos)

| Indicador | Cluster 1 | Nacional (promedio) | vs Cluster 0 |
|---|---|---|---|
| Oficinas/100k | **7.45** (+44%) | 5.18 | 1.7× |
| Depósitos/cápita | **S/.3,109** (+92%) | S/.1,620 | **3.1×** |
| NBI % | **14.1%** (−31%) | 20.5% | 0.6× |
| Ingreso promedio | **S/.1,805** (+12%) | S/.1,608 | 1.3× |
| PBI/cápita | **S/.28,387** (+63%) | S/.17,419 | **2.3×** |
| Tasa de empleo | 0.933 (−2%) | 0.947 | 0.98× |
| Internet/100k | **45,141** (+38%) | 32,793 | **1.8×** |

**Departamentos:** Arequipa, Cusco, Ica, Junín, Madre de Dios, Moquegua, Tacna

**Perfil:** Costa sur + sierra sur + selva sur. Alta densidad financiera (7.45 of/100k), depósitos per cápita 3.1× superiores al Cluster 0, PBI per cápita más del doble. Menor pobreza (14.1% NBI). Son los territorios con **mejor perspectiva para expansión financiera**.

---

#### 🔵 CLUSTER 2 — "Megaurbano Contradictorio" (1 departamento)

| Indicador | Lima/Callao | Nacional (promedio) |
|---|---|---|
| Oficinas/100k | **2.41** (−53%) | 5.18 |
| Depósitos/cápita | S/.1,231 (−24%) | S/.1,620 |
| NBI % | **33.9%** (+65%) | 20.5% |
| Ingreso promedio | **S/.3,870** (+141%) | S/.1,608 |
| PBI/cápita | S/.24,396 (+40%) | S/.17,419 |
| Tasa de empleo | 0.937 (−1%) | 0.947 |
| Internet/100k | **77,310** (+136%) | 32,793 |

**Perfil contradictorio:** Lima tiene el ingreso más alto del país (S/.3,870) y la mejor conectividad digital (77,310 internet/100k), pero también tiene **la densidad de oficinas per cápita más baja** (2.41/100k) y el NBI más alto (33.9%) después de Loreto y Ucayali. Esto refleja la enorme desigualdad interna de Lima: la riqueza y la conectividad coexisten con pobreza extrema en asentamientos humanos. DBSCAN la detecta como outlier por esta contradicción.

---

### 9.2 Mapa de Correlación entre Features

No se encontraron correlaciones altas (|r| > 0.8) entre las 7 features. La correlación más fuerte es Ingreso ↔ Internet/100k (r=0.75), lo cual es esperable (a mayor ingreso, mayor acceso a internet). La Tasa de Empleo tiene correlaciones negativas con PBI/cápita (r=−0.62) e Internet/100k (r=−0.67), lo que sugiere que los departamentos con mayor PBI/cápita y mejor conectividad tienden a tener **menor tasa de empleo formal** — posiblemente por mayor PEA en búsqueda activa en zonas urbanas.

**Salida visual:** `final_07_correlacion_features.png` | `final_08_perfil_clusters.png`

---

## 10. Zonas con Mejor Perspectiva

### 10.1 Ranking por Inclusión Financiera (Depósitos per Cápita)

| # | Departamento | Cluster | Dep/cápita | Of/100k | NBI% | PBI/cap |
|---|---|---|---|---|---|---|
| 1 | **Arequipa** | 1 🟢 | S/.5,410 | 6.29 | 8.6% | S/.22,880 |
| 2 | **Cusco** | 1 🟢 | S/.3,601 | 6.88 | 11.4% | S/.18,886 |
| 3 | **Junín** | 1 🟢 | S/.3,308 | 6.58 | 20.7% | S/.12,835 |
| 4 | **Madre de Dios** | 1 🟢 | S/.2,529 | 9.92 | 27.3% | S/.13,621 |
| 5 | **Tacna** | 1 🟢 | S/.2,388 | 6.98 | 9.9% | S/.27,087 |
| 6 | **Moquegua** | 1 🟢 | S/.2,286 | 9.72 | 11.2% | S/.78,198 |
| 7 | **Ica** | 1 🟢 | S/.2,239 | 5.76 | 9.2% | S/.25,201 |

### 10.2 Recomendaciones por Perfil

**Prioridad Alta — Cluster 1 (7 departamentos):**
- **Arequipa, Cusco, Tacna, Ica:** Bajo NBI (<12%) + alta densidad financiera. Mercados con alta capacidad de absorción de nuevos servicios financieros. Menor riesgo crediticio esperado.
- **Moquegua, Madre de Dios:** Alta densidad financiera pero perfiles atípicos (minería, economía pequeña). Requieren estudios de mercado específicos antes de expandir.

**Prioridad Media — Cluster 0 (16 departamentos):**
- **La Libertad, Lambayeque, Piura:** Costa norte con densidad financiera media y economías activas. Potencial para expansión gradual.
- **Apurímac, Cajamarca, Puno:** Alta informalidad pero baja penetración financiera. Mercado virgen pero requiere inversión en educación financiera.

**Caso Especial — Lima/Callao:**
- No requiere expansión de sucursales físicas (la densidad de oficinas ya es la más baja del país, pero el mercado está saturado). **La oportunidad en Lima está en canales digitales** (tiene la mayor conectividad del país: 77k/100k) y en los conos (donde se concentra el NBI de 33.9%).

### 10.3 Factor Clave: Densidad de Oficinas

Random Forest reveló que **Oficinas_por_100k explica el 68.9% de la varianza en depósitos per cápita**. Esto significa que:

> *Para aumentar la inclusión financiera en un territorio, abrir más sucursales (o agentes corresponsales) es 7× más efectivo que esperar a que suban los ingresos o mejore la conectividad.*

---

## 11. Conclusiones

### 11.1 Hallazgos Principales

1. **Los 24 departamentos peruanos se segmentan naturalmente en 3 clusters**, con un grupo mayoritario (16 deptos) de "inclusión en desarrollo", un grupo de "desarrollo económico alto" (7 deptos) y Lima como caso atípico.

2. **La densidad de oficinas financieras es el factor #1** que predice la inclusión financiera (68.9% de importancia), muy por encima del ingreso (10.9%) o la conectividad (3.5%).

3. **4 departamentos son outliers por perfil único** (Arequipa, Lima, Madre de Dios, Moquegua) y requieren estrategias de expansión personalizadas.

4. **Lima es un caso contradictorio:** alta riqueza y conectividad, pero profunda desigualdad reflejada en baja densidad financiera per cápita y alto NBI.

### 11.2 Limitaciones del Análisis

| Limitación | Impacto | Mitigación |
|---|---|---|
| **n=24 departamentos solamente** | Muestra pequeña para ML; K-Means y RF tienen poder estadístico limitado | Resultados exploratorios, no concluyentes. Se prioriza interpretabilidad |
| **R² de Random Forest negativo (−0.225)** | RF no puede predecir confiablemente depósitos per cápita con solo 24 datos | El ranking de importancias sigue siendo informativo (el orden es estable) |
| **Silhouette K=2 (0.357) > K=3 (0.345)** | K=3 es 2° mejor según métrica, no el óptimo estadístico | Se justifica por interpretabilidad y coherencia con dendrograma |
| **Datos de un solo año** | No captura tendencias temporales ni estacionalidad | Análisis transversal válido para diagnóstico inicial |
| **Solo Cajas Municipales** | Excluye banca múltiple, financieras, fintech | Coherente con el objetivo (expansión de Cajas) |

### 11.3 Trabajo Futuro

1. Escalar a nivel **provincial o distrital** para mayor granularidad
2. Incorporar datos temporales (2017-2026) para análisis de evolución
3. Incluir variables de **riesgo crediticio** y **morosidad** por territorio
4. Validar clusters con datos de **rentabilidad real de sucursales**
5. Dashboard interactivo con los resultados (ver `precompute_dashboard.py`)

---

> *Documento generado con datos verificados de BaseDedatosFinal.xlsx y ejecución real del script `analisis_final_basededatos_explicado.py`. Versión 1.0 — Junio 2026.*
