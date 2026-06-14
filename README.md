<div align="center">

# 🏦 Segmentación Territorial para Inclusión Financiera

### Machine Learning No Supervisado para la Expansión Sostenible de Servicios Financieros Formales en el Perú

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?style=flat&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.10-11557C?style=flat)
![Status](https://img.shields.io/badge/status-v1.0--release-2A9D8F?style=flat)
![License](https://img.shields.io/badge/license-academic-6C757D?style=flat)

<img src="final_06_kmeans_resultado.png" alt="K-Means Clustering — 3 clusters en espacio PCA" width="680"/>

**Universidad Privada del Norte** · Maestría en Ciencia de Datos & IA · Curso: Machine Learning  
**Versión 1.0** — Junio 2026

---

</div>

## 📑 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Participantes](#-participantes)
- [Pipeline Técnico](#-pipeline-técnico)
- [Algoritmos Implementados](#-algoritmos-implementados)
- [Resultados: Los 3 Clusters](#-resultados-los-3-clusters)
- [Métricas del Código](#-métricas-del-código)
- [Cómo Usar](#-cómo-usar)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Documentación Adicional](#-documentación-adicional)
- [Cómo Citar](#-cómo-citar)

---

## 🎯 Descripción del Proyecto

Este proyecto aplica **5 algoritmos de Machine Learning** para segmentar los **24 departamentos del Perú** según su nivel de inclusión financiera, utilizando **14 variables demográficas, económicas, digitales, financieras, laborales y sociales**.

El objetivo es identificar **zonas con mejor perspectiva para la expansión de Cajas Municipales**, usando aprendizaje no supervisado que descubre patrones territoriales sin necesidad de etiquetas previas.

**Problema de fondo:** En Perú, gran parte de la economía opera en informalidad — personas y negocios generan ingresos que no figuran en registros bancarios ni tributarios. Esto hace que la expansión financiera tradicional (basada en historial crediticio) sea insuficiente. La alternativa: analizar **territorios enteros** mediante señales agregadas.

🔑 **Hallazgo principal:** La **densidad de oficinas financieras** explica el **68.9%** de la variación en inclusión financiera entre departamentos — 7× más que el ingreso promedio o la conectividad digital.

---

## 👥 Participantes

| | Integrante | Código | Rol |
|---:|---|---|---|
| 1 | **Abad Panta, Cristina Elizabeth** | N00558600 | Data Scientist |
| 2 | **Arroyo Estredo, Jiang Alberto** | N00557773 | Data Scientist |
| 3 | **Garcia Yuffra, Juan Angel** | N00563512 | Data Scientist |
| 4 | **Mansilla Florez, Christian Serge Moises** | N00558117 | Data Scientist |
| 5 | **Melgarejo Alcedo, Elen** | N00561314 | Data Scientist |
| 6 | **Velarde Fernandez, Liliana** | N00554157 | Data Scientist |

**Docente:** Dario Condor Callupe

---

## 🔬 Pipeline Técnico

| Fase | Descripción | Detalle |
|---|---|---|
| **1** 📥 Carga | `BaseDedatosFinal.xlsx` → hoja `Data Final` | 24 deptos × 14 vars, 0 nulos |
| **2** ⚙️ Feature Engineering | 8 variables derivadas (per cápita, tasas, ratios) | Normalización por población |
| **3** 📐 Estandarización | `StandardScaler` → media=0, std=1 | Obligatorio para K-Means + PCA |
| **4a** 🔍 PCA | Reducción a 2D (69% varianza explicada) | Biplot + scree plot |
| **4b** 🌳 Clust. Jerárquico | Dendrograma Ward | Validación de K natural |
| **4c** 📍 DBSCAN | Detección de outliers por densidad | eps=2.2, min_samples=2 |
| **4d** 🌲 Random Forest | Importancia de variables (diagnóstico) | Objetivo: Depósitos/cápita |
| **5** 🎯 K-Means | Segmentación final K=3 | Silhouette=0.345, DB=0.882 |
| **6** 📤 Exportación | 8 PNG + 1 CSV con resultados | 300 dpi, UTF-8 |

---

## 🧠 Algoritmos Implementados

| Algoritmo | Propósito | Métrica Clave | Salida |
|---|---|---|---|
| **PCA** | Visualizar 7 dimensiones en 2D | 69% varianza explicada (PC1+PC2) | `final_01_pca_biplot.png` |
| **Jerárquico (Ward)** | Validar estructura natural de grupos | Corte a altura 6.5 → 3 ramas | `final_02_dendrograma.png` |
| **DBSCAN** | Detectar perfiles únicos (outliers) | 4 outliers: AQP, LIM, MDD, MOQ | `final_03_dbscan_outliers.png` |
| **Random Forest** | Ranking de importancia de variables | Of/100k = **68.9%** (R² CV = −0.225) | `final_04_random_forest_importancia.png` |
| **K-Means (K=3)** | Segmentación final en 3 clusters | Silhouette=0.345 · DB=0.882 | `final_06_kmeans_resultado.png` |

---

## 📊 Resultados: Los 3 Clusters

### 🔴 Cluster 0 — "Inclusión en Desarrollo" (16 deptos)

| Indicador | Valor | vs. Promedio Nacional |
|---|---|---|
| Oficinas/100k | 4.36 | −16% |
| Depósitos/cápita | S/.993 | **−39%** |
| NBI % | 22.5% | +10% |
| Ingreso promedio | S/.1,380 | −14% |
| PBI/cápita | S/.12,185 | −30% |
| Internet/100k | 24,608 | −25% |

**Departamentos:** Amazonas, Áncash, Apurímac, Ayacucho, Cajamarca, Huancavelica, Huánuco, La Libertad, Lambayeque, Loreto, Pasco, Piura, Puno, San Martín, Tumbes, Ucayali

---

### 🟢 Cluster 1 — "Desarrollo Económico Alto" (7 deptos) ⭐

| Indicador | Valor | vs. Nacional | vs. Cluster 0 |
|---|---|---|---|
| Oficinas/100k | 7.45 | **+44%** | **1.7×** |
| Depósitos/cápita | S/.3,109 | **+92%** | **3.1×** |
| NBI % | 14.1% | −31% | 0.6× |
| Ingreso promedio | S/.1,805 | +12% | 1.3× |
| PBI/cápita | S/.28,387 | **+63%** | **2.3×** |
| Internet/100k | 45,141 | +38% | **1.8×** |

**Departamentos:** Arequipa, Cusco, Ica, Junín, Madre de Dios, Moquegua, Tacna

---

### 🔵 Cluster 2 — "Megaurbano Contradictorio" (1 depto)

| Indicador | Valor | vs. Nacional |
|---|---|---|
| Oficinas/100k | **2.41** (mínimo país) | −53% |
| Depósitos/cápita | S/.1,231 | −24% |
| NBI % | **33.9%** | +65% |
| Ingreso promedio | **S/.3,870** (máximo país) | **+141%** |
| Internet/100k | **77,310** (máximo país) | **+136%** |

**Departamento:** Lima/Callao (outlier por perfil contradictorio)

---

### 📍 Outliers Detectados por DBSCAN

| Departamento | ¿Por qué es único? |
|---|---|
| **Arequipa** | Depósitos/cápita más alto del país (S/.5,410) — duplica al segundo |
| **Lima/Callao** | Máximo ingreso + máxima conectividad, pero mínimo Of/100k y NBI alto |
| **Madre de Dios** | Máxima densidad de oficinas (9.92/100k) + máxima cobertura móvil |
| **Moquegua** | PBI/cápita más alto (S/.78,198) — minería distorsiona el perfil |

---

## 📐 Métricas del Código

| Métrica | Valor |
|---|---|
| Script principal | `analisis_final_basededatos_explicado.py` |
| Líneas totales | **1,008** |
| Líneas de código | 542 |
| Líneas de comentarios | 327 (documentación inline) |
| Modelos implementados | 5 (PCA, Jerárquico, DBSCAN, RF, K-Means) |
| Visualizaciones generadas | 8 gráficos PNG a 300 dpi |
| Archivos de salida | 1 CSV (24 deptos × 15 columnas) |
| Tiempo de ejecución | **~21 segundos** en CPU estándar |
| Dependencias | pandas 2.3, numpy 2.4, scikit-learn 1.8, matplotlib 3.10, scipy 1.17, seaborn 0.13, openpyxl |
| Reproducibilidad | Garantizada (random_state=42, n_init=30) |

### Script auxiliar

| Archivo | Líneas | Propósito |
|---|---|---|
| `precompute_dashboard.py` | 187 | Precomputa datos para dashboard interactivo (PCA 3D, K-Means K=2..6, Hopkins) |

---

## 🚀 Cómo Usar

### Requisitos

```bash
pip install -r requirements.txt
```

### Ejecutar el pipeline completo

```bash
cd Github/
python3 analisis_final_basededatos_explicado.py
```

Esto generará:
- 8 archivos PNG en la misma carpeta
- 1 archivo CSV (`final_resultados_clustering.csv`)

### Precomputar datos para dashboard

```bash
python3 precompute_dashboard.py > dashboard_data.json
```

---

## 📁 Estructura del Repositorio

```
📂 Github/
├── 📄 README.md                              # Esta página
├── 📄 requirements.txt                       # Dependencias Python
├── 📄 BaseDedatosFinal.xlsx                  # Dataset: 24 deptos × 14 vars
├── 📄 analisis_final_basededatos_explicado.py # Pipeline completo (1008 LOC)
├── 📄 precompute_dashboard.py                # Precompute para dashboard
├── 📄 Machine Learning - Grupo7 - 13_Jun_26 v1.docx  # Informe académico
├── 📄 final_resultados_clustering.csv        # Resultados con clusters
├── 📄 explicacion_algoritmos_inclusion_financiera.md  # Documento técnico detallado
│
├── 🖼️ final_01_pca_biplot.png                # PCA: scree + biplot
├── 🖼️ final_02_dendrograma.png               # Dendrograma jerárquico
├── 🖼️ final_03_dbscan_outliers.png           # DBSCAN: outliers
├── 🖼️ final_04_random_forest_importancia.png # Importancia de variables
├── 🖼️ final_05_kmeans_seleccion_k.png        # Selección de K óptimo
├── 🖼️ final_06_kmeans_resultado.png          # K-Means K=3
├── 🖼️ final_07_correlacion_features.png      # Matriz de correlación
├── 🖼️ final_08_perfil_clusters.png           # Perfiles de cluster
├── 🖼️ explicacion_pipeline_completo.png      # Diagrama del pipeline
└── 🖼️ explicacion_sae_concepto.png           # Diagrama SAE
```

---

## 📖 Documentación Adicional

Para una explicación detallada de cada algoritmo, variables, limitaciones y fundamentos técnicos, consulta:

➡️ **[explicacion_algoritmos_inclusion_financiera.md](explicacion_algoritmos_inclusion_financiera.md)**

Incluye: justificación de cada modelo, métricas verificadas, interpretación de loadings PCA, ranking de importancia de variables, y análisis de limitaciones (R² negativo, n=24, Silhouette subóptimo).

---

## 📄 Cómo Citar

```bibtex
@misc{grupo7_upn_2026_segmentacion,
  title  = {Segmentación Territorial de Departamentos Peruanos para la Expansión de Servicios Financieros Formales mediante Machine Learning No Supervisado},
  author = {Abad Panta, Cristina and Arroyo Estredo, Jiang and Garcia Yuffra, Juan and Mansilla Florez, Christian and Melgarejo Alcedo, Elen and Velarde Fernandez, Liliana},
  year   = {2026},
  school = {Universidad Privada del Norte},
  note   = {Curso de Machine Learning — Maestría en Ciencia de Datos & IA}
}
```

---

<div align="center">

**Universidad Privada del Norte** · Maestría en Ciencia de Datos & Inteligencia Artificial  
**Curso:** Machine Learning · **Docente:** Dario Condor Callupe  
**Perú** — Junio 2026

[⬆ Volver al inicio](#-segmentación-territorial-para-inclusión-financiera)

</div>
