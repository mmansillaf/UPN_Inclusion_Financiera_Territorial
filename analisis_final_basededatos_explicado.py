# =============================================================================
# ANÁLISIS DE INCLUSIÓN FINANCIERA — BASE DE DATOS FINAL
# Archivo: analisis_final_basededatos.py
# Fuente:  BaseDedatosFinal.xlsx  (hoja: "Data Final")
# Autores: Grupo de Machine Learning
# =============================================================================
#
# ─────────────────────────────────────────────────────────────────────────────
# ¿QUÉ HACE ESTE SCRIPT?
# ─────────────────────────────────────────────────────────────────────────────
# Toma los 24 departamentos del Perú con 13 variables financieras, económicas,
# demográficas y sociales, y aplica 5 algoritmos de Machine Learning para
# segmentarlos según su nivel de inclusión financiera:
#
#   1. PCA ──── Reduce 7 dimensiones a 2 ejes para visualizar los datos.
#   2. Jerárquico ── Dendrograma que muestra la estructura natural de grupos.
#   3. DBSCAN ──── Detecta departamentos con perfil único (outliers).
#   4. Random Forest ── Mide qué variable predice mejor la inclusión financiera.
#   5. K-Means ── Segmentación final en 3 clusters interpretables.
#
# SALIDAS: 8 gráficos PNG + 1 CSV con clusters asignados.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

# Forzar codificación UTF-8 en consola Windows para que los tildes se muestren
# correctamente (evita UnicodeEncodeError en cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin ventana — evita errores de Tkinter en consola
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Algoritmos de clustering y reducción dimensional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Modelo supervisado para medir importancia de variables
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Clustering jerárquico (librería SciPy, separada de sklearn)
from scipy.cluster.hierarchy import dendrogram, linkage

# Elementos gráficos adicionales
from matplotlib.patches import Patch, Circle

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Directorio donde vive este script (se usa para construir rutas absolutas)
DIRECTORIO = os.path.dirname(os.path.abspath(__file__))

# Ruta al archivo Excel con la base de datos final
ARCHIVO_DATOS = os.path.join(DIRECTORIO, 'BaseDedatosFinal.xlsx')

# Nombre de la hoja dentro del Excel
HOJA_DATOS = 'Data Final'

# Prefijo para todos los archivos de salida de esta versión
PREFIJO = 'final_'

# Paleta de colores para los clusters (hasta 5 clusters posibles)
COLORES_CLUSTER = ['#E63946', '#2A9D8F', '#457B9D', '#F4A261', '#8338EC']

# Semilla aleatoria para reproducibilidad (los resultados serán siempre iguales)
SEMILLA = 42

print("=" * 80)
print("ANÁLISIS DE INCLUSIÓN FINANCIERA — BASE DE DATOS FINAL")
print("K-Means + PCA + Clustering Jerárquico + DBSCAN + Random Forest")
print("=" * 80)


# =============================================================================
# PASO 1: CARGA Y PREPARACIÓN DE DATOS
# =============================================================================
# Se carga la hoja "Data Final" del Excel (24 departamentos, 13 variables).
# La columna "Departamento" se estandariza a mayúsculas (DPTO_KEY) para usarla
# como etiqueta en los gráficos sin problemas de mayúsculas/minúsculas.
# =============================================================================
#
# ── ¿POR QUÉ ESTE PASO? ─────────────────────────────────────────────────────
# Los datos vienen de una base depurada por el grupo. Solo cargamos, limpiamos
# tildes (ó→o para evitar errores de encoding en Windows) y creamos la clave
# de departamento. No hay valores nulos que imputar.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/6] Cargando datos desde BaseDedatosFinal.xlsx (hoja: Data Final)...")

df = pd.read_excel(ARCHIVO_DATOS, sheet_name=HOJA_DATOS)

# Renombrar la columna con tilde (ó) para evitar problemas de encoding en Windows
# 'Conexión_Internet_Fijo' → 'Conexion_Internet_Fijo'
df.rename(columns=lambda c: c.strip()
                              .replace('ó', 'o').replace('ú', 'u')
                              .replace('á', 'a').replace('é', 'e')
                              .replace('í', 'i'),
          inplace=True)

# Crear clave de departamento en mayúsculas para usarla como etiqueta en gráficos
df['DPTO_KEY'] = df['Departamento'].str.upper().str.strip()

print(f"    [OK] {len(df)} departamentos cargados")
print(f"    [OK] {df.isnull().sum().sum()} valores nulos en toda la base")
print(f"    [OK] Columnas: {[c for c in df.columns if c not in ['Departamento', 'DPTO_KEY']]}")


# =============================================================================
# PASO 2: INGENIERÍA DE VARIABLES (FEATURE ENGINEERING)
# =============================================================================
# Las variables originales no son comparables directamente entre departamentos
# porque tienen escalas y poblaciones muy distintas (ej. Lima tiene 11M hab.
# y Madre de Dios 140k hab.). Para que el análisis sea justo, normalizamos
# cada indicador por habitante o como tasa.
# =============================================================================
#
# ── ¿POR QUÉ TRANSFORMAR? ───────────────────────────────────────────────────
# Si usáramos valores absolutos (PBI total, denuncias totales, etc.), Lima
# dominaría todas las métricas por su tamaño poblacional. Al convertir a
# indicadores per cápita o por cada 100k habitantes, medimos intensidad o
# prevalencia — no volumen absoluto. Esto permite comparar regiones de
# distinto tamaño en igualdad de condiciones.
#
# Por ejemplo: "Oficinas_por_100k" mide cuántas oficinas hay por cada 100 mil
# personas, no cuántas oficinas hay en total. Un departamento pequeño puede
# tener más cobertura relativa que uno grande.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Calculando indicadores derivados (Feature Engineering)...")

# --- Infraestructura financiera ---
# Cuántas oficinas de Cajas hay por cada 100,000 habitantes.
# Permite comparar regiones de distinto tamaño poblacional.
df['Oficinas_por_100k'] = (
    df['Num_Cajas'] / (df['PoblacionTotal_Censo2017'] / 100_000)
)

# --- Profundidad financiera ---
# Cuántos soles en depósitos corresponden a cada persona en edad activa (18-70 años).
# Un valor alto indica que la gente usa activamente los servicios de ahorro.
df['Depositos_por_Capita'] = (
    df['Depositos_Cajas_PEN_MM'] * 1_000_000 / df['Poblacion_18_70']
)

# --- Tasa de empleo ---
# Proporción de la PEA que está efectivamente empleada.
# PEA total = PEA ocupada + PEA no ocupada (desempleados que buscan trabajo).
df['PEA_total'] = df['PEA_ocupada_miles_2024'] + df['PEA_no_ocupada_miles_2024']
df['Tasa_Empleo'] = df['PEA_ocupada_miles_2024'] / df['PEA_total']

# --- Conectividad digital ---
# Conexiones de internet fijo por cada 100,000 habitantes.
# Proxy de acceso a banca digital y fintech.
df['Internet_por_100k'] = (
    df['Conexion_Internet_Fijo'] / (df['PoblacionTotal_Censo2017'] / 100_000)
)

# --- Conectividad móvil ---
# Líneas de telefonía móvil por cada 100,000 habitantes.
# Relevante para pagos móviles e inclusión digital.
df['Movil_por_100k'] = (
    df['NroLineasTelefoniaMovil'] / (df['PoblacionTotal_Censo2017'] / 100_000)
)

# --- PBI per cápita ---
# Producto Bruto Interno por habitante (en soles).
# Mide la productividad económica total de la región.
# PBI_miles_PEN está en miles de soles → multiplicamos ×1000 para obtener soles.
df['PBI_por_Capita'] = (
    df['PBI_miles_PEN'] * 1_000 / df['PoblacionTotal_Censo2017']
)

# --- Seguridad por habitante ---
# Número de efectivos PNP por cada 100,000 habitantes.
# Variable contextual: la seguridad influye en el clima para hacer negocios
# y en la disposición de la gente a movilizarse hacia oficinas bancarias.
df['PNP_por_100k'] = (
    df['Num_PNP_2026'] / (df['PoblacionTotal_Censo2017'] / 100_000)
)

# --- Índice de brecha de inclusión ---
# Relaciona la pobreza (NBI) con la cobertura de oficinas.
# Un valor alto indica: mucha pobreza + pocas oficinas = alta brecha de inclusión.
# El +0.01 evita división por cero en caso extremo de 0 oficinas.
df['Indice_Brecha_Inclusion'] = (
    df['NBI_%_2024'] / (df['Oficinas_por_100k'] + 0.01)
)

print("    [OK] Oficinas_por_100k        — densidad de infraestructura bancaria")
print("    [OK] Depositos_por_Capita     — profundidad financiera (S/ por persona activa)")
print("    [OK] Tasa_Empleo              — PEA ocupada / PEA total")
print("    [OK] Internet_por_100k        — conectividad digital fija")
print("    [OK] Movil_por_100k           — cobertura de telefonía móvil")
print("    [OK] PBI_por_Capita           — productividad económica regional")
print("    [OK] PNP_por_100k             — densidad policial (seguridad)")
print("    [OK] Indice_Brecha_Inclusion  — NBI relativo a cobertura bancaria")


# =============================================================================
# PASO 3: SELECCIÓN DE FEATURES Y ESTANDARIZACIÓN
# =============================================================================
# De todas las variables derivadas se eligen 7 para los modelos. Cada una
# aporta una dimensión distinta e independiente del fenómeno de inclusión
# financiera.
#
# La estandarización (StandardScaler) transforma cada variable para que tenga
# media=0 y desviación estándar=1. Es OBLIGATORIA porque:
#   - K-Means y PCA miden distancias: una variable en miles (PBI) dominaría
#     a una en decimales (Tasa_Empleo) si no se escala.
#   - Con estandarización, todas las variables tienen el mismo "peso" inicial.
# =============================================================================

print("\n[3/6] Seleccionando features y aplicando estandarización (StandardScaler)...")

# Las 7 features del modelo — cada una captura un eje distinto del fenómeno
FEATURES = [
    'Oficinas_por_100k',      # Acceso físico a servicios financieros
    'Depositos_por_Capita',   # Uso activo del sistema financiero (ahorro)
    'NBI_%_2024',             # Pobreza y exclusión social
    'Ingreso_Prom_PEN_2024',  # Capacidad económica de la población
    'Tasa_Empleo',            # Estabilidad del mercado laboral
    'PBI_por_Capita',         # Productividad económica de la región
    'Internet_por_100k',      # Brecha digital (acceso a banca digital)
]

# Extraer la matriz de datos (solo las columnas de features)
X = df[FEATURES].values

# Aplicar estandarización: (valor - media) / desviación_estándar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"    [OK] {len(df)} departamentos × {len(FEATURES)} features")
print(f"    [OK] StandardScaler aplicado — media≈0, std≈1 por variable")
print(f"\n    Estadísticas de las features ANTES de escalar:")
print(f"    {'Variable':<30} {'Min':>10} {'Max':>10} {'Media':>10}")
print(f"    {'-'*62}")
for feat in FEATURES:
    v = df[feat]
    print(f"    {feat:<30} {v.min():>10.2f} {v.max():>10.2f} {v.mean():>10.2f}")


# =============================================================================
# MODELO 1: PCA — ANÁLISIS DE COMPONENTES PRINCIPALES
# =============================================================================
# PCA reduce las 7 dimensiones originales a 2 ejes ortogonales (PC1, PC2)
# para poder visualizar los departamentos en un plano 2D. También indica
# cuánta información (varianza) captura cada componente.
#
# Adicionalmente se genera un BIPLOT: superpone los puntos (departamentos)
# con los vectores de cada variable original, mostrando cómo se relacionan
# las variables y dónde se posiciona cada región.
# =============================================================================
#
# ── ¿QUÉ APORTA PCA? ────────────────────────────────────────────────────────
# 1) SABER si los datos se pueden representar en 2D sin perder mucha info.
#    Si PC1+PC2 explican >60% de la varianza, el plano 2D es confiable.
# 2) IDENTIFICAR qué variables se correlacionan (flechas en misma dirección).
# 3) DETECTAR visualmente si hay grupos naturales (departamentos cercanos
#    en el plano PCA tienden a tener perfiles similares).
#
# El BIPLOT combina ambas informaciones:
#   - Puntos = departamentos en el espacio reducido
#   - Flechas = dirección de cada variable original
#     (una flecha larga = variable muy influyente en PC1 o PC2)
#     (dos flechas en mismo sentido = variables correlacionadas positivamente)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4a/6] MODELO 1: PCA (Análisis de Componentes Principales)...")

# PCA completo para el Scree Plot (cuántos componentes se necesitan)
pca_full = PCA()
pca_full.fit(X_scaled)
varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

# Número de componentes necesarios para explicar el 90% de la varianza
n_comp_90 = int(np.argmax(varianza_acumulada >= 0.90)) + 1

# PCA en 2D para visualización y como base para gráficos posteriores
pca_2d = PCA(n_components=2, random_state=SEMILLA)
X_pca = pca_2d.fit_transform(X_scaled)

# Guardar las coordenadas PCA en el dataframe para uso posterior
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

var_exp = pca_2d.explained_variance_ratio_
print(f"    [OK] Varianza explicada: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, Total={var_exp.sum():.1%}")
print(f"    [OK] Componentes para 90% de varianza: {n_comp_90}")

# Tabla de loadings: cuánto contribuye cada variable original a PC1 y PC2
print(f"\n    Loadings (contribución de cada variable a cada componente):")
print(f"    {'Variable':<30} {'PC1':>8} {'PC2':>8}")
print(f"    {'-'*48}")
for i, feat in enumerate(FEATURES):
    print(f"    {feat:<30} {pca_2d.components_[0, i]:>+8.3f} {pca_2d.components_[1, i]:>+8.3f}")

# ---- Gráfico PCA: Scree Plot + Biplot ----
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle('Análisis de Componentes Principales (PCA)\nBase de Datos Final — 24 Departamentos del Perú',
             fontsize=14, fontweight='bold', y=1.01)

# Panel izquierdo: Scree Plot
# Barras = varianza de cada componente; línea roja = varianza acumulada
n_comp = len(pca_full.explained_variance_ratio_)
colores_scree = ['#E63946' if i < n_comp_90 else '#ADB5BD' for i in range(n_comp)]
axes[0].bar(range(1, n_comp + 1),
            pca_full.explained_variance_ratio_ * 100,
            color=colores_scree, edgecolor='black', alpha=0.85)
axes[0].plot(range(1, n_comp + 1), varianza_acumulada * 100,
             'ko-', linewidth=2, markersize=5, label='Varianza acumulada')
axes[0].axhline(90, color='red', linestyle='--', linewidth=1.5, label='Umbral 90%')
axes[0].axvline(n_comp_90 + 0.5, color='orange', linestyle=':', linewidth=2,
                label=f'{n_comp_90} comp. = 90%')
axes[0].set_xlabel('Componente Principal', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Varianza Explicada (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Scree Plot\n(barras rojas = componentes seleccionados)', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticks(range(1, n_comp + 1))

# Panel derecho: Biplot
# Coloreamos los puntos por NBI (pobreza): rojo=más pobre, verde=menos pobre
norm = plt.Normalize(df['NBI_%_2024'].min(), df['NBI_%_2024'].max())
scatter = axes[1].scatter(
    df['PCA1'], df['PCA2'],
    c=df['NBI_%_2024'], cmap='RdYlGn_r', norm=norm,
    s=150, alpha=0.85, edgecolors='black', linewidth=1, zorder=3
)
cb = plt.colorbar(scatter, ax=axes[1], shrink=0.8)
cb.set_label('NBI % 2024 (rojo = más pobreza)', fontsize=9)

# Dibujar los vectores de las variables originales (loadings escalados)
escala_vectores = 2.8
loadings = pca_2d.components_.T  # Forma: (n_features, 2)
for j, feat in enumerate(FEATURES):
    lx, ly = loadings[j, 0] * escala_vectores, loadings[j, 1] * escala_vectores
    axes[1].annotate('', xy=(lx, ly), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    # Texto del vector ligeramente desplazado para no solaparse con la flecha
    axes[1].text(lx * 1.18, ly * 1.18,
                 feat.replace('_', '\n'), fontsize=7.5,
                 color='navy', fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

# Etiquetas de departamentos
for _, row in df.iterrows():
    axes[1].annotate(
        row['DPTO_KEY'],
        xy=(row['PCA1'], row['PCA2']),
        xytext=(4, 4), textcoords='offset points',
        fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.65)
    )

# Líneas de referencia en el origen
axes[1].axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
axes[1].axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
axes[1].set_xlabel(f'PC1  —  {var_exp[0]:.1%} de varianza', fontsize=11, fontweight='bold')
axes[1].set_ylabel(f'PC2  —  {var_exp[1]:.1%} de varianza', fontsize=11, fontweight='bold')
axes[1].set_title('Biplot: Departamentos y Variables\n(color = nivel de pobreza NBI)', fontsize=11)
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
ruta_pca = os.path.join(DIRECTORIO, f'{PREFIJO}01_pca_biplot.png')
plt.savefig(ruta_pca, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}01_pca_biplot.png")


# =============================================================================
# MODELO 2: CLUSTERING JERÁRQUICO (DENDROGRAMA)
# =============================================================================
# El clustering jerárquico no requiere definir el número de clusters de
# antemano. Parte de 24 objetos individuales y los va fusionando en grupos
# según su similitud hasta llegar a un solo gran grupo.
#
# Método Ward: minimiza la varianza total dentro de cada cluster al fusionar.
# Es el método más robusto y produce clusters compactos.
#
# El resultado es un DENDROGRAMA (árbol): la altura de cada unión indica
# qué tan distintos son los grupos que se están fusionando.
# =============================================================================
#
# ── ¿POR QUÉ USAR JERÁRQUICO ANTES DE K-MEANS? ──────────────────────────────
# El dendrograma es EXPLORATORIO: nos muestra si hay una estructura natural
# de grupos, sin imponer un K fijo. Si el dendrograma muestra 3 ramas claras
# a una altura definida, eso sugiere que K=3 es un número natural de clusters.
#
# Se usa como VALIDACIÓN PREVIA de la elección de K que luego hará K-Means.
# Si el dendrograma sugiere K=3 y K-Means da un buen Silhouette con K=3,
# la segmentación es más confiable que si solo usáramos un método.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4b/6] MODELO 2: Clustering Jerárquico (método Ward)...")

# Calcular la matriz de enlace jerárquico con método Ward
# Recibe los datos estandarizados (X_scaled)
Z = linkage(X_scaled, method='ward')

fig, ax = plt.subplots(figsize=(15, 8))

# Dibujar el dendrograma
# color_threshold: las ramas por encima de este valor se colorean en gris
# (indica que forman grupos separados)
dendro = dendrogram(
    Z,
    labels=df['DPTO_KEY'].values,
    leaf_rotation=75,
    leaf_font_size=9,
    color_threshold=6.5,
    above_threshold_color='#6C757D',
    ax=ax
)

# Línea de corte sugerida para K=3 grupos
ax.axhline(y=6.5, color='#E63946', linestyle='--', linewidth=2.5,
           label='Corte sugerido → K=3 grupos')

ax.set_title(
    'Clustering Jerárquico — Método Ward\n'
    'Base de Datos Final | 24 Departamentos del Perú',
    fontsize=13, fontweight='bold'
)
ax.set_xlabel('Departamento', fontsize=11, fontweight='bold')
ax.set_ylabel('Distancia de fusión (Ward)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.25, axis='y')

# Sombrear el área por debajo del corte para destacar los grupos
ax.axhspan(0, 6.5, alpha=0.04, color='green')

plt.tight_layout()
ruta_dendro = os.path.join(DIRECTORIO, f'{PREFIJO}02_dendrograma.png')
plt.savefig(ruta_dendro, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}02_dendrograma.png")


# =============================================================================
# MODELO 3: DBSCAN — DETECCIÓN DE OUTLIERS
# =============================================================================
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) agrupa
# puntos que están en zonas "densas" del espacio y marca como OUTLIER (-1) a
# los puntos aislados que no forman parte de ningún grupo denso.
#
# Parámetros:
#   eps=2.2 : radio de vecindad. Dos puntos son "vecinos" si su distancia
#             euclidiana (en espacio estandarizado) es menor a eps.
#   min_samples=2 : mínimo de vecinos para que un punto sea "core point"
#                   (núcleo de un cluster). Con n=24, valores bajos son
#                   adecuados.
#
# Un outlier en DBSCAN NO significa que el departamento sea "malo".
# Significa que su combinación de indicadores es única, diferente a todos
# los demás — es un caso que merece atención individualizada.
# =============================================================================
#
# ── ¿POR QUÉ DBSCAN ADEMÁS DE K-MEANS? ──────────────────────────────────────
# K-Means ASIGNA todos los puntos a un cluster, incluso si un punto no encaja
# bien en ningún grupo. DBSCAN, en cambio, DEJA FUERA los puntos que no
# pertenecen a ninguna zona densa. Esto permite identificar departamentos con
# perfil atípico que distorsionarían los clusters de K-Means.
#
# Por ejemplo: si Lima/Callao es un outlier, significa que su perfil es tan
# extremo que merece análisis aparte — no debería forzarse a compartir cluster
# con otros departamentos.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4c/6] MODELO 3: DBSCAN (detección de outliers y densidad)...")

# eps=2.2 calibrado empíricamente para este dataset estandarizado de n=24
dbscan = DBSCAN(eps=2.2, min_samples=2)
etiquetas_db = dbscan.fit_predict(X_scaled)

# Agregar etiqueta al dataframe (-1 = outlier)
df['DBSCAN_Cluster'] = etiquetas_db

n_outliers   = int((etiquetas_db == -1).sum())
n_clusters_db = len(set(etiquetas_db)) - (1 if -1 in etiquetas_db else 0)

print(f"    [OK] Clusters de densidad detectados: {n_clusters_db}")
print(f"    [OK] Outliers identificados:          {n_outliers}")
if n_outliers > 0:
    outliers_lista = df[df['DBSCAN_Cluster'] == -1]['DPTO_KEY'].values
    print(f"    [OK] Departamentos outlier: {', '.join(outliers_lista)}")

# ---- Gráfico DBSCAN en espacio PCA 2D ----
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle('DBSCAN — Detección de Outliers y Clusters por Densidad\nBase de Datos Final | 24 Departamentos del Perú',
             fontsize=13, fontweight='bold', y=1.01)

# Colores: rojo para outliers, colores distintos para cada cluster
paleta_db = {-1: '#E63946', 0: '#2A9D8F', 1: '#457B9D', 2: '#F4A261', 3: '#8338EC'}
colores_puntos = [paleta_db.get(int(l), '#CCCCCC') for l in etiquetas_db]

# Panel izquierdo: scatter en espacio PCA
ax = axes[0]
ax.scatter(df['PCA1'], df['PCA2'], c=colores_puntos,
           s=200, alpha=0.85, edgecolors='black', linewidth=1.5, zorder=3)
for _, row in df.iterrows():
    ax.annotate(row['DPTO_KEY'],
                xy=(row['PCA1'], row['PCA2']),
                xytext=(5, 4), textcoords='offset points',
                fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.6))

# Leyenda dinámica: solo muestra etiquetas que existen en los datos
leyenda_db = [Patch(facecolor='#E63946', edgecolor='black',
                    label=f'Outliers — perfil único (n={n_outliers})')]
for c in sorted(set(etiquetas_db) - {-1}):
    n_c = int((etiquetas_db == c).sum())
    leyenda_db.append(Patch(facecolor=paleta_db.get(c, '#CCCCCC'),
                            edgecolor='black', label=f'Cluster DBSCAN {c} (n={n_c})'))
ax.legend(handles=leyenda_db, loc='best', fontsize=9)
ax.set_xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=10, fontweight='bold')
ax.set_ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=10, fontweight='bold')
ax.set_title('Proyección en espacio PCA', fontsize=11)
ax.grid(True, alpha=0.25)

# Panel derecho: tabla resumen de outliers con sus indicadores clave
ax2 = axes[1]
ax2.axis('off')
if n_outliers > 0:
    df_out = df[df['DBSCAN_Cluster'] == -1][
        ['DPTO_KEY', 'NBI_%_2024', 'Ingreso_Prom_PEN_2024',
         'PBI_por_Capita', 'Oficinas_por_100k', 'Depositos_por_Capita',
         'Tasa_Empleo', 'Internet_por_100k']
    ].copy()
    df_out.columns = ['Depto', 'NBI %', 'Ingreso\nProm S/.', 'PBI/\nCápita', 'Ofic\n/100k', 'Dep/\nCápita S/.', 'Tasa\nEmpleo', 'Internet\n/100k']
    df_out = df_out.round(1)
    tabla = ax2.table(
        cellText=df_out.values,
        colLabels=df_out.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.1, 1, 0.75]
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    # Cabecera en rojo (color de outliers)
    for j in range(len(df_out.columns)):
        tabla[0, j].set_facecolor('#E63946')
        tabla[0, j].set_text_props(color='white', fontweight='bold')
    ax2.set_title('Indicadores de los departamentos outlier', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
ruta_dbscan = os.path.join(DIRECTORIO, f'{PREFIJO}03_dbscan_outliers.png')
plt.savefig(ruta_dbscan, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}03_dbscan_outliers.png")


# =============================================================================
# MODELO 4: RANDOM FOREST — IMPORTANCIA DE VARIABLES
# =============================================================================
# Random Forest es un modelo supervisado que entrena 200 árboles de decisión
# y luego promedia sus predicciones. Como subproducto mide la IMPORTANCIA de
# cada variable: cuánto contribuye a reducir el error de predicción.
#
# Variable objetivo: Depositos_por_Capita
#   Se usa como proxy de "nivel de inclusión financiera" — regiones con más
#   depósitos per cápita usan más activamente el sistema financiero.
#
# Variables predictoras: las 6 features restantes (sin Depositos_por_Capita)
#
# La validación cruzada (5-fold) divide los 24 datos en 5 partes, entrena
# en 4 y evalúa en la restante — repitiendo 5 veces. Produce un R² más
# confiable que el R² simple (que siempre sería artificialmente alto en
# entrenamiento).
# =============================================================================
#
# ── ¿POR QUÉ RANDOM FOREST EN UN ANÁLISIS NO SUPERVISADO? ───────────────────
# Aunque el objetivo principal del script es clustering (no supervisado),
# Random Forest sirve aquí como DIAGNÓSTICO. Responde a la pregunta:
# "¿Qué variable socioeconómica explica MEJOR el nivel de depósitos
# financieros?" Si "Oficinas_por_100k" es la más importante (como suele
# ocurrir), eso valida que el acceso físico a sucursales es el factor
# clave de la inclusión financiera. Si "Internet_por_100k" lo fuera,
# la inclusión digital sería más relevante que la presencial.
#
# El R² de validación cruzada suele ser negativo con n=24 (pocos datos
# para entrenar). Eso NO invalida el ranking de importancias — el orden
# de las variables es útil aunque el poder predictivo absoluto sea bajo.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4d/6] MODELO 4: Random Forest (importancia de variables)...")

# Features predictoras: todas excepto la variable objetivo
FEATURES_RF = [f for f in FEATURES if f != 'Depositos_por_Capita']
X_rf = df[FEATURES_RF].values
y_rf = df['Depositos_por_Capita'].values

# Entrenar el Random Forest
# n_estimators=200: 200 árboles (más árboles = más estable, pero más lento)
# max_depth=5: profundidad máxima de cada árbol (controla el sobreajuste)
rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                           random_state=SEMILLA, n_jobs=-1)
rf.fit(X_rf, y_rf)

# Validación cruzada con 5 particiones
cv_r2 = cross_val_score(rf, X_rf, y_rf, cv=5, scoring='r2')

# Serie de importancias ordenadas de mayor a menor
importancias = pd.Series(rf.feature_importances_, index=FEATURES_RF)
importancias_ord = importancias.sort_values(ascending=False)

print(f"    [OK] R² validación cruzada (5-fold): {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
print(f"    [OK] Importancia por variable:")
for feat, imp in importancias_ord.items():
    barra = '█' * int(imp * 50)
    print(f"       {feat:<30} {imp:.4f} ({imp*100:5.1f}%)  {barra}")

# ---- Gráfico de importancias ----
fig, ax = plt.subplots(figsize=(11, 6))

# Colorear la barra más importante en rojo, el resto en verde-azulado
colores_rf = ['#E63946' if imp == importancias_ord.max() else '#2A9D8F'
              for imp in importancias_ord.sort_values(ascending=True)]

barras = ax.barh(
    importancias_ord.sort_values(ascending=True).index,
    importancias_ord.sort_values(ascending=True).values,
    color=colores_rf, edgecolor='black', height=0.6
)

# Etiquetas numéricas al final de cada barra
for bar, val in zip(barras, importancias_ord.sort_values(ascending=True).values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}  ({val*100:.1f}%)',
            va='center', fontsize=9.5, fontweight='bold', color='#212529')

ax.set_xlabel('Importancia Gini (suma total = 1.0)', fontsize=11, fontweight='bold')
ax.set_title(
    f'Random Forest — Importancia de Variables\n'
    f'Variable objetivo: Depositos_por_Capita  |  R² CV = {cv_r2.mean():.3f}',
    fontsize=12, fontweight='bold'
)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, importancias_ord.max() * 1.35)

# Nota al pie explicando la interpretación
fig.text(0.5, -0.04,
         'La barra más larga es la variable que más influye en predecir el nivel de depósitos per cápita.',
         ha='center', fontsize=9, style='italic', color='#495057')

plt.tight_layout()
ruta_rf = os.path.join(DIRECTORIO, f'{PREFIJO}04_random_forest_importancia.png')
plt.savefig(ruta_rf, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}04_random_forest_importancia.png")


# =============================================================================
# MODELO 5: K-MEANS — SELECCIÓN DE K ÓPTIMO
# =============================================================================
# K-Means divide los datos en K grupos minimizando la varianza intra-cluster.
# El desafío es elegir el K correcto. Se usan 3 criterios complementarios:
#
# 1. MÉTODO DEL CODO (Elbow): graficar la inercia (suma de distancias al
#    cuadrado al centroide). La inercia siempre baja al aumentar K; el "codo"
#    (quiebre de la curva) sugiere el K donde agregar más clusters ya no
#    mejora significativamente.
#
# 2. COEFICIENTE DE SILUETA (Silhouette Score): mide qué tan bien asignado
#    está cada punto a su cluster (−1 a +1, mayor es mejor). Se maximiza.
#
# 3. ÍNDICE DAVIES-BOULDIN: mide la relación entre dispersión intra-cluster
#    y separación inter-cluster (menor es mejor). Se minimiza.
#
# Se usa n_init=30 para que K-Means pruebe 30 inicializaciones distintas y
# no se quede en un mínimo local (solución subóptima).
#
# ── NOTA SOBRE K=3 ──────────────────────────────────────────────────────────
# El Silhouette Score óptimo puede ser K=2, pero se fuerza K=3 por coherencia
# interpretativa. K=2 separaría Perú en "desarrollado vs. no desarrollado",
# lo cual es correcto estadísticamente pero poco útil para la toma de
# decisiones. K=3 permite identificar un grupo intermedio y genera perfiles
# más accionables para estrategias de expansión diferenciadas.
# ─────────────────────────────────────────────────────────────────────────────
# =============================================================================

print("\n[5/6] MODELO 5: K-Means — selección de K óptimo y clustering final...")
print("      Evaluando K de 2 a 9...")

inertias       = []
sil_scores     = []
db_scores      = []
K_range        = range(2, 10)  # Probar K desde 2 hasta 9

for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=SEMILLA, n_init=30)
    labels_tmp = km_tmp.fit_predict(X_scaled)
    inertias.append(km_tmp.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_tmp))
    db_scores.append(davies_bouldin_score(X_scaled, labels_tmp))

# Identificar el K óptimo según cada criterio
best_k_sil = list(K_range)[int(np.argmax(sil_scores))]
best_k_db  = list(K_range)[int(np.argmin(db_scores))]

print(f"    [OK] Silhouette Score  — mejor K: {best_k_sil} (score={max(sil_scores):.3f})")
print(f"    [OK] Davies-Bouldin    — mejor K: {best_k_db} (score={min(db_scores):.3f})")

# ---- Gráfico de criterios de selección de K ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('K-Means — Criterios para Selección del K Óptimo\nBase de Datos Final | 24 Departamentos del Perú',
             fontsize=13, fontweight='bold', y=1.02)

# Panel 1: Método del Codo
axes[0].plot(K_range, inertias, 'o-', color='#457B9D', linewidth=2.5, markersize=9,
             markerfacecolor='white', markeredgewidth=2)
axes[0].fill_between(K_range, inertias, alpha=0.1, color='#457B9D')
axes[0].set_xlabel('K — Número de clusters', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Inercia (suma dist² al centroide)', fontsize=11, fontweight='bold')
axes[0].set_title('Método del Codo\n(buscar el quiebre de la curva)', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(K_range)

# Panel 2: Silhouette Score
axes[1].plot(K_range, sil_scores, 'o-', color='#2A9D8F', linewidth=2.5, markersize=9,
             markerfacecolor='white', markeredgewidth=2)
axes[1].axvline(best_k_sil, color='#E63946', linestyle='--', linewidth=2.5,
                label=f'Óptimo: K={best_k_sil}  ({max(sil_scores):.3f})')
axes[1].fill_between(K_range, sil_scores, alpha=0.1, color='#2A9D8F')
axes[1].set_xlabel('K — Número de clusters', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Silhouette Score (↑ mejor)', fontsize=11, fontweight='bold')
axes[1].set_title('Coeficiente de Silueta\n(mayor = clusters más definidos)', fontsize=11)
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(K_range)

# Panel 3: Davies-Bouldin Index
axes[2].plot(K_range, db_scores, 'o-', color='#F4A261', linewidth=2.5, markersize=9,
             markerfacecolor='white', markeredgewidth=2)
axes[2].axvline(best_k_db, color='#E63946', linestyle='--', linewidth=2.5,
                label=f'Óptimo: K={best_k_db}  ({min(db_scores):.3f})')
axes[2].fill_between(K_range, db_scores, alpha=0.1, color='#F4A261')
axes[2].set_xlabel('K — Número de clusters', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index (↓ mejor)', fontsize=11, fontweight='bold')
axes[2].set_title('Índice Davies-Bouldin\n(menor = clusters más separados)', fontsize=11)
axes[2].legend(fontsize=10, loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(K_range)

plt.tight_layout()
ruta_k_sel = os.path.join(DIRECTORIO, f'{PREFIJO}05_kmeans_seleccion_k.png')
plt.savefig(ruta_k_sel, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}05_kmeans_seleccion_k.png")

# =============================================================================
# K-MEANS FINAL CON K SELECCIONADO
# =============================================================================
# Se elige K=3 como balance entre los criterios anteriores y el dendrograma.
# Con K=3 se obtienen grupos interpretables e internamente coherentes,
# adecuados para análisis de políticas públicas con n=24 departamentos.
# =============================================================================

K_FINAL = 3
print(f"\n    Ejecutando K-Means final con K={K_FINAL} (n_init=30, seed={SEMILLA})...")

kmeans = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=30)
etiquetas_km = kmeans.fit_predict(X_scaled)

df['Cluster'] = etiquetas_km

sil_final = silhouette_score(X_scaled, etiquetas_km)
db_final  = davies_bouldin_score(X_scaled, etiquetas_km)

print(f"    [OK] Silhouette Score (K={K_FINAL}): {sil_final:.3f}  [0=malo → 1=perfecto]")
print(f"    [OK] Davies-Bouldin  (K={K_FINAL}): {db_final:.3f}  [menor es mejor]")
print(f"\n    Composición de clusters:")
for i in range(K_FINAL):
    subset = df[df['Cluster'] == i]
    print(f"\n    CLUSTER {i} — {len(subset)} departamentos:")
    print(f"      {', '.join(subset['DPTO_KEY'].values)}")
    print(f"      Ofic/100k={subset['Oficinas_por_100k'].mean():.2f} | "
          f"Dep/cap=S/.{subset['Depositos_por_Capita'].mean():,.0f} | "
          f"NBI={subset['NBI_%_2024'].mean():.1f}% | "
          f"Ingreso=S/.{subset['Ingreso_Prom_PEN_2024'].mean():,.0f} | "
          f"PBI/cap=S/.{subset['PBI_por_Capita'].mean():,.0f} | "
          f"Empleo={subset['Tasa_Empleo'].mean():.3f} | "
          f"Internet/100k={subset['Internet_por_100k'].mean():,.0f}")

# ---- Gráfico K-Means resultado en espacio PCA ----
fig, ax = plt.subplots(figsize=(15, 11))

# Dibujar puntos coloreados por cluster
color_puntos_km = [COLORES_CLUSTER[int(c)] for c in etiquetas_km]
ax.scatter(df['PCA1'], df['PCA2'],
           c=color_puntos_km, s=420,
           alpha=0.80, edgecolors='black', linewidth=2, zorder=3)

# Proyectar los centroides al espacio PCA para graficarlos
centroides_pca = pca_2d.transform(kmeans.cluster_centers_)
ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
           c='gold', s=750, marker='*',
           edgecolors='black', linewidth=2, zorder=5, label='Centroides')

# Dibujar círculos de contorno alrededor de cada cluster
for i in range(K_FINAL):
    pts = df[df['Cluster'] == i]
    if len(pts) > 1:
        distancias = np.sqrt(
            (pts['PCA1'] - centroides_pca[i, 0])**2 +
            (pts['PCA2'] - centroides_pca[i, 1])**2
        )
        radio = distancias.max() * 1.18
    else:
        radio = 0.6
    circulo = Circle(
        (centroides_pca[i, 0], centroides_pca[i, 1]), radio,
        fill=False, edgecolor=COLORES_CLUSTER[i],
        linewidth=2.5, linestyle='--', alpha=0.75
    )
    ax.add_patch(circulo)

# Etiquetas de departamentos
for _, row in df.iterrows():
    ax.annotate(
        row['DPTO_KEY'],
        xy=(row['PCA1'], row['PCA2']),
        xytext=(6, 5), textcoords='offset points',
        fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.6)
    )

# Leyenda con nombre interpretativo de cada cluster
nombres_cluster = [''] * K_FINAL
for i in range(K_FINAL):
    subset = df[df['Cluster'] == i]
    nbi_prom = subset['NBI_%_2024'].mean()
    ing_prom = subset['Ingreso_Prom_PEN_2024'].mean()
    pbi_prom = subset['PBI_por_Capita'].mean()
    # Asignar nombre según perfil del cluster
    if nbi_prom > 25:
        nombres_cluster[i] = f'Cluster {i}: Alta exclusión financiera (n={len(subset)})'
    elif pbi_prom > 20000 or ing_prom > 2000:
        nombres_cluster[i] = f'Cluster {i}: Desarrollo económico alto (n={len(subset)})'
    else:
        nombres_cluster[i] = f'Cluster {i}: Inclusión en desarrollo (n={len(subset)})'

leyenda_km = [Patch(facecolor=COLORES_CLUSTER[i], edgecolor='black', label=nombres_cluster[i])
              for i in range(K_FINAL)]
leyenda_km.append(ax.scatter([], [], c='gold', s=500, marker='*',
                              edgecolors='black', linewidth=2, label='Centroides'))
ax.legend(handles=leyenda_km, loc='best', fontsize=10.5,
          framealpha=0.9, edgecolor='gray')

ax.set_xlabel(f'PC1  —  {var_exp[0]:.1%} de varianza', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2  —  {var_exp[1]:.1%} de varianza', fontsize=12, fontweight='bold')
ax.set_title(
    f'K-Means (K={K_FINAL}) — Segmentación de Inclusión Financiera\n'
    f'Silhouette={sil_final:.3f}  |  Davies-Bouldin={db_final:.3f}  |  Base: BaseDedatosFinal.xlsx',
    fontsize=13, fontweight='bold', pad=20
)
ax.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()
ruta_km = os.path.join(DIRECTORIO, f'{PREFIJO}06_kmeans_resultado.png')
plt.savefig(ruta_km, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}06_kmeans_resultado.png")


# =============================================================================
# PASO 6: GRÁFICOS COMPLEMENTARIOS Y EXPORTACIÓN
# =============================================================================

print("\n[6/6] Generando gráficos complementarios y exportando resultados...")

# ---- Mapa de calor de correlaciones ----
# Muestra la correlación de Pearson entre todas las features del modelo.
# Permite detectar variables redundantes (correlación > 0.9) y confirmar
# que cada variable aporta información única al modelo.

df_corr = df[FEATURES].copy()
corr_matrix = df_corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Solo triángulo inferior

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    mask=mask,
    ax=ax,
    square=True,
    linewidths=0.6,
    cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Pearson (−1 a +1)'},
    annot_kws={'size': 9.5}
)
ax.set_title(
    'Matriz de Correlación entre Features del Modelo\n'
    'Base de Datos Final | (triangular: solo mitad inferior)',
    fontsize=12, fontweight='bold'
)
plt.xticks(rotation=35, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
ruta_corr = os.path.join(DIRECTORIO, f'{PREFIJO}07_correlacion_features.png')
plt.savefig(ruta_corr, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}07_correlacion_features.png")

# ---- Gráfico de perfiles de clusters (Radar / Barras agrupadas) ----
# Muestra las medias de cada feature por cluster en barras agrupadas,
# usando los datos estandarizados para que todas las variables sean comparables.
# Permite entender el "perfil" de cada grupo de un vistazo.

medias_clusters = np.zeros((K_FINAL, len(FEATURES)))
for i in range(K_FINAL):
    idx = df[df['Cluster'] == i].index
    medias_clusters[i] = X_scaled[idx].mean(axis=0)

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(FEATURES))
ancho = 0.25
for i in range(K_FINAL):
    offset = (i - 1) * ancho
    barras = ax.bar(x + offset, medias_clusters[i], ancho,
                    label=f'Cluster {i}  (n={int((df["Cluster"]==i).sum())})',
                    color=COLORES_CLUSTER[i], edgecolor='black', alpha=0.85)

ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Variable (estandarizada: media=0, std=1)', fontsize=11, fontweight='bold')
ax.set_ylabel('Valor medio estandarizado (Z-score)', fontsize=11, fontweight='bold')
ax.set_title('Perfil de Cada Cluster — Medias Estandarizadas por Variable\n'
             'Barras > 0: el cluster está por encima del promedio nacional',
             fontsize=12, fontweight='bold')
etiquetas_ejes = [f.replace('_', '\n') for f in FEATURES]
ax.set_xticks(x)
ax.set_xticklabels(etiquetas_ejes, fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
ruta_perfil = os.path.join(DIRECTORIO, f'{PREFIJO}08_perfil_clusters.png')
plt.savefig(ruta_perfil, dpi=300, bbox_inches='tight')
plt.close()
print(f"    [OK] Gráfico guardado: {PREFIJO}08_perfil_clusters.png")

# ---- Exportar CSV con todos los resultados ----
# Unir columnas auxiliares (movil, PNP, índice brecha) al resultado final
df_resultado = df[[
    'DPTO_KEY', 'Cluster', 'DBSCAN_Cluster',
    'Oficinas_por_100k', 'Depositos_por_Capita',
    'NBI_%_2024', 'Ingreso_Prom_PEN_2024',
    'Tasa_Empleo', 'PBI_por_Capita', 'Internet_por_100k',
    'Movil_por_100k', 'PNP_por_100k', 'Indice_Brecha_Inclusion',
    'PCA1', 'PCA2'
]].copy().round(4)

ruta_csv = os.path.join(DIRECTORIO, 'final_resultados_clustering.csv')
df_resultado.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
print(f"    [OK] CSV exportado: final_resultados_clustering.csv")


# =============================================================================
# RESUMEN FINAL EN CONSOLA
# =============================================================================

print("\n" + "=" * 80)
print("RESUMEN DE RESULTADOS POR CLUSTER")
print("=" * 80)

for i in range(K_FINAL):
    subset = df[df['Cluster'] == i].copy()
    print(f"\nCLUSTER {i} — {len(subset)} departamentos")
    print(f"  Departamentos: {', '.join(subset['DPTO_KEY'].values)}")
    print(f"  Indicadores promedio:")
    print(f"    Oficinas por 100k hab. : {subset['Oficinas_por_100k'].mean():.2f}")
    print(f"    Depositos per capita   : S/. {subset['Depositos_por_Capita'].mean():,.0f}")
    print(f"    NBI %                  : {subset['NBI_%_2024'].mean():.1f}%")
    print(f"    Ingreso promedio       : S/. {subset['Ingreso_Prom_PEN_2024'].mean():,.0f}")
    print(f"    PBI per capita         : S/. {subset['PBI_por_Capita'].mean():,.0f}")
    print(f"    Tasa de empleo         : {subset['Tasa_Empleo'].mean():.3f}")
    print(f"    Internet por 100k      : {subset['Internet_por_100k'].mean():,.0f}")

print("\n" + "=" * 80)
print("ARCHIVOS GENERADOS:")
print("=" * 80)
archivos_generados = [
    (f'{PREFIJO}01_pca_biplot.png',          'Scree plot + Biplot PCA con vectores de carga'),
    (f'{PREFIJO}02_dendrograma.png',          'Dendrograma clustering jerarquico Ward'),
    (f'{PREFIJO}03_dbscan_outliers.png',      'Outliers y clusters por densidad (DBSCAN)'),
    (f'{PREFIJO}04_random_forest_importancia.png', 'Importancia de variables (Random Forest)'),
    (f'{PREFIJO}05_kmeans_seleccion_k.png',   'Criterios Elbow + Silueta + Davies-Bouldin'),
    (f'{PREFIJO}06_kmeans_resultado.png',     'Resultado final K-Means K=3 en espacio PCA'),
    (f'{PREFIJO}07_correlacion_features.png', 'Matriz de correlacion entre features'),
    (f'{PREFIJO}08_perfil_clusters.png',      'Perfil de cada cluster (medias estandarizadas)'),
    ('final_resultados_clustering.csv',       'Dataset completo con etiquetas de cluster'),
]
for archivo, descripcion in archivos_generados:
    print(f"  [OK] {archivo:<45} — {descripcion}")
print("=" * 80)
