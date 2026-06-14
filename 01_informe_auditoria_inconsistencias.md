# INFORME DE AUDITORÍA — Inconsistencias Encontradas
## Proyecto: Segmentación Territorial para Inclusión Financiera
## Fuentes: BaseDedatosFinal.xlsx | analisis_final_basededatos_explicado.py | Word Grupo7

---

### INCONSISTENCIA 1 — Word vs Script: ¿PCA selecciona variables o solo visualiza?

**Word (línea 109):**
> "El modelo operará sobre las variables que fueron seleccionadas por PCA"

Esto sugiere que K-Means recibe como entrada las **componentes PCA** (las variables reducidas por PCA).

**Script REAL (líneas 240-244, 776-777):**
```python
X = df[FEATURES].values          # 7 variables originales
X_scaled = scaler.fit_transform(X) # estandarizadas
...
kmeans = KMeans(n_clusters=K_FINAL)
etiquetas_km = kmeans.fit_predict(X_scaled)  # K-Means sobre X_scaled, NO sobre PCA
```

**Verificado ejecutando el script:** K-Means recibe `X_scaled` (7 features estandarizadas), no las componentes PCA. PCA se usa solo para visualización (biplot) y como eje del gráfico de resultados. No hay reducción de dimensionalidad antes de K-Means.

**Severidad: MEDIA** — El Word describe una arquitectura (PCA como selector de variables → K-Means) que no corresponde al código real.

---

### INCONSISTENCIA 2 — Word vs Script: Random Forest como "no adecuado" pero implementado

**Word (línea 102):**
> "no resulta adecuado utilizar modelos supervisados como regresión logística, árboles de decisión, Random Forest o XGBoost"

**Script REAL (líneas 603-613):**
```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_depth=5, ...)
rf.fit(X_rf, y_rf)
```

El script SÍ entrena y ejecuta RandomForestRegressor. Aunque el script documenta (líneas 586-597) que se usa como "diagnóstico" y no como modelo principal, el Word no lo aclara y lo descarta explícitamente, creando una contradicción directa.

**Severidad: BAJA** — El uso de RF como diagnóstico está bien justificado en el script, pero el Word necesita corregir la redacción para no descartarlo.

---

### INCONSISTENCIA 3 — Word vs Script: Lima/Callao como "valores superiores"

**Word:**
> "Lima/Callao puede concentrar valores significativamente superiores en población, depósitos, conectividad, PBI, denuncias y presencia institucional"

**Datos reales verificados del Excel:**
| Variable | Lima vs país | Realidad |
|---|---|---|
| Of/100k | 2.41 (mínimo nacional) | **Es la más baja**, no superior |
| NBI % | 33.9% (tercero más alto) | **Es pobreza alta**, no superior |
| Dep/cápita | S/.1,231 (debajo del promedio S/.1,620) | **Está por debajo del promedio** |
| Ingreso | S/.3,870 (máximo nacional) | ✅ Correcto: muy superior |
| Internet/100k | 77,310 (máximo nacional) | ✅ Correcto: muy superior |

El perfil de Lima NO es "valores superiores en todo" — es un perfil **contradictorio**: alta conectividad digital, alto ingreso, pero pobreza extrema (33.9% NBI) y la menor densidad de oficinas financieras per cápita del país (2.41/100k). DBSCAN la detecta como outlier precisamente por esto, no por ser "superior".

**Severidad: MEDIA** — La caracterización de Lima en el Word es imprecisa y puede llevar a conclusiones erróneas sobre por qué es outlier.

---

### INCONSISTENCIA 4 — Word no reporta ninguna métrica numérica

**Word:** No contiene ningún valor numérico de resultados. Describe la metodología pero nunca reporta:
- Silhouette Score real (K=2: 0.357, K=3: 0.345)
- Davies-Bouldin real (K=3: 0.882)
- Varianza explicada por PCA (PC1=47%, PC2=22%, total 69%)
- R² de Random Forest (-0.225 ± 0.910)
- Composición de los 3 clusters
- Outliers detectados por DBSCAN (Arequipa, Lima, Madre de Dios, Moquegua)

**Severidad: ALTA** — Un informe de machine learning sin métricas de resultado no permite evaluar la calidad del modelo.

---

### INCONSISTENCIA 5 — Silhouette óptimo es K=2 (0.357) pero se fuerza K=3 (0.345)

**Script documenta (líneas 688-693):**
```python
# Silhouette Score óptimo puede ser K=2, pero se fuerza K=3 por coherencia
# interpretativa. K=2 separaría Perú en "desarrollado vs. no desarrollado",
# lo cual es correcto estadísticamente pero poco útil para la toma de decisiones.
```

Esto es una decisión metodológica explícita y documentada en el código. El Word no la menciona.

**Severidad: BAJA** — Decisión válida pero debe documentarse en el Word.

---

### INCONSISTENCIA 6 — Random Forest R² negativo no documentado

**Script (líneas 594-597):**
> "El R² de validación cruzada suele ser negativo con n=24 (pocos datos para entrenar). Eso NO invalida el ranking de importancias — el orden de las variables es útil aunque el poder predictivo absoluto sea bajo."

**Resultado real:** R² CV = -0.225 ± 0.910

El Word no menciona que Random Forest tiene R² negativo (predice peor que la media). La sección de "Validación cruzada" en el Word solo dice "estimará el desempeño" sin reportar el resultado real.

**Severidad: MEDIA** — Un R² negativo sin aclaración puede malinterpretarse como error del modelo.

---

### INCONSISTENCIA 7 — precompute_dashboard.py duplica feature engineering

El archivo `precompute_dashboard.py` implementa su propio feature engineering con ligeras diferencias:
- Usa openpyxl directo en vez de pandas read_excel
- Incluye Movil_por_100k, PNP_por_100k, Brecha_Inclusion como features (el script principal los calcula pero NO los usa como features del modelo)
- El script principal usa exactamente 7 features; el dashboard calcula 9 derivadas

**Severidad: BAJA** — Scripts con propósitos diferentes (análisis vs dashboard) es aceptable, pero la duplicación puede generar desfases si se modifican las transformaciones.

---

### RESUMEN

| # | Inconsistencia | Severidad | Fuente |
|---|---|---|---|
| 1 | Word dice "variables seleccionadas por PCA" → Script usa 7 features originales | MEDIA | Word vs Script |
| 2 | Word descarta RF como "no adecuado" → Script implementa RF | BAJA | Word vs Script |
| 3 | Word describe Lima como "valores superiores" → Datos muestran perfil contradictorio | MEDIA | Word vs Excel |
| 4 | Word no reporta ninguna métrica numérica de resultados | ALTA | Word |
| 5 | Silhouette óptimo K=2, se fuerza K=3 sin documentar en Word | BAJA | Script vs Word |
| 6 | R² negativo (-0.225) no documentado en Word | MEDIA | Script vs Word |
| 7 | precompute_dashboard.py duplica feature engineering | BAJA | Script vs Script |
