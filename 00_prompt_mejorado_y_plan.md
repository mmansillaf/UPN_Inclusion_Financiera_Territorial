# Prompt Mejorado

```
Analiza los 3 archivos del proyecto de segmentación territorial de departamentos peruanos
(BaseDedatosFinal.xlsx, Machine Learning - Grupo7 - 13_Jun_26 v1.docx,
analisis_final_basededatos_explicado.py) y genera un documento técnico estructurado con:

1. Contexto del problema: ¿por qué se segmentan los 24 departamentos del Perú para
   expansión de servicios financieros formales (Cajas Municipales)?

2. Pipeline de feature engineering: variables originales (14 columnas del Excel),
   transformaciones aplicadas (per cápita, tasas, ratios), y las 7 features finales
   del modelo con justificación de cada una.

3. Explicación didáctica de los 5 algoritmos implementados:
   - PCA: reducción de dimensionalidad, scree plot, biplot, varianza explicada
   - Clustering Jerárquico (Ward): dendrograma, validación visual de K
   - DBSCAN: detección de outliers por densidad, parámetros eps/min_samples
   - Random Forest: importancia de variables para predecir Depositos_por_Capita
   - K-Means: selección de K (Elbow + Silhouette + Davies-Bouldin), clustering final K=3

4. Resultados obtenidos: composición de cada cluster, perfil socioeconómico,
   departamentos por grupo, outliers detectados.

5. Interpretación final: ¿qué zonas tienen mejor perspectiva para inclusión
   financiera y desarrollo? Sustentar con datos de los clusters.

Entregar en un solo documento markdown (.md) con tablas de variables, métricas
de cada modelo, perfiles de cluster, y conclusiones accionables para una Caja
Municipal.
```

---

# Plan de Trabajo

```
╔══════════════════════════════════════════════════════════════════╗
║ PLAN — Documento: Segmentación Territorial para Inclusión Fin. ║
║ Archivos fuente: Github/                                       ║
║ Salida:       explicacion_algoritmos_inclusion_financiera.md   ║
╚══════════════════════════════════════════════════════════════════╝

FASE 1 — Extraer datos del Excel
  [ ] Leer BaseDedatosFinal.xlsx → hoja "Data Final"
  [ ] Tabla de 24 departamentos × 14 variables originales
  [ ] Estadísticas descriptivas (min, max, media por variable)

FASE 2 — Sintetizar el documento Word
  [ ] Extraer definición del problema, marco teórico
  [ ] Extraer justificación de los algoritmos (del informe del grupo)

FASE 3 — Analizar el script Python (ya leído)
  [ ] Documentar cada paso del pipeline (6 pasos, 5 modelos)
  [ ] Extraer métricas: varianza PCA, Silhouette, Davies-Bouldin,
      importancias RF, outliers DBSCAN
  [ ] Extraer composición de los 3 clusters finales

FASE 4 — Construir el documento explicativo
  [ ] Sección 1: Presentación del problema
  [ ] Sección 2: Datos y variables (tabla original + derivadas)
  [ ] Sección 3: Pipeline de ingeniería de características
  [ ] Sección 4: Algoritmo por algoritmo (5 modelos):
       - ¿Qué hace?   - ¿Por qué se usa?   - ¿Qué nos dice?
  [ ] Sección 5: Resultados (tabla de clusters, perfiles, mapa)
  [ ] Sección 6: Zonas con mejor perspectiva (top 3 departamentos)
  [ ] Sección 7: Conclusiones para toma de decisiones

FASE 5 — Revisión y entrega
  [ ] Verificar consistencia datos vs. documento
  [ ] Ajustar formato markdown
  [ ] Mostrar resumen al usuario
```
