#!/usr/bin/env python3
"""Extraer datos derivados y estadísticas para auditoría."""
import pandas as pd
import numpy as np

df = pd.read_excel('BaseDedatosFinal.xlsx', sheet_name='Data Final')
df.columns = [c.replace('ó','o').replace('ú','u').replace('á','a').replace('é','e').replace('í','i') for c in df.columns]

# Feature engineering exacto del script
df['Oficinas_por_100k'] = df['Num_Cajas'] / (df['PoblacionTotal_Censo2017'] / 100_000)
df['Depositos_por_Capita'] = df['Depositos_Cajas_PEN_MM'] * 1_000_000 / df['Poblacion_18_70']
pea_total = df['PEA_ocupada_miles_2024'] + df['PEA_no_ocupada_miles_2024']
df['Tasa_Empleo'] = df['PEA_ocupada_miles_2024'] / pea_total
df['Internet_por_100k'] = df['Conexion_Internet_Fijo'] / (df['PoblacionTotal_Censo2017'] / 100_000)
df['Movil_por_100k'] = df['NroLineasTelefoniaMovil'] / (df['PoblacionTotal_Censo2017'] / 100_000)
df['PBI_por_Capita'] = df['PBI_miles_PEN'] * 1000 / df['PoblacionTotal_Censo2017']
df['PNP_por_100k'] = df['Num_PNP_2026'] / (df['PoblacionTotal_Censo2017'] / 100_000)
df['Indice_Brecha_Inclusion'] = df['NBI_%_2024'] / (df['Oficinas_por_100k'] + 0.01)

FEATURES = ['Oficinas_por_100k','Depositos_por_Capita','NBI_%_2024','Ingreso_Prom_PEN_2024','Tasa_Empleo','PBI_por_Capita','Internet_por_100k']

print("=" * 140)
print("VARIABLES DERIVADAS — Todos los departamentos")
print("=" * 140)
hdr = f"{'Departamento':<20} {'Of/100k':>9} {'Dep/cap':>10} {'NBI%':>7} {'Ingreso':>9} {'T.Empl':>7} {'PBI/cap':>10} {'Int/100k':>9} {'Mov/100k':>9} {'PNP/100k':>9} {'Brecha':>8}"
print(hdr)
print("-" * 140)
for _, row in df.iterrows():
    print(f"{row['Departamento']:<20} {row['Oficinas_por_100k']:>9.2f} {row['Depositos_por_Capita']:>10,.0f} {row['NBI_%_2024']:>6.1f}% {row['Ingreso_Prom_PEN_2024']:>9,.0f} {row['Tasa_Empleo']:>7.3f} {row['PBI_por_Capita']:>10,.0f} {row['Internet_por_100k']:>9,.0f} {row['Movil_por_100k']:>9,.0f} {row['PNP_por_100k']:>9.0f} {row['Indice_Brecha_Inclusion']:>8.2f}")

print("\n\nESTADÍSTICAS DE FEATURES")
print("=" * 80)
for f in FEATURES:
    print(f"\n  {f}:")
    print(f"    Media: {df[f].mean():>12.4f}  Std: {df[f].std():>12.4f}")
    print(f"    Min:   {df[f].min():>12.4f}  Max: {df[f].max():>12.4f}")

# Correlación
print("\n\nMATRIZ DE CORRELACIÓN")
print("=" * 80)
corr = df[FEATURES].corr()
print(corr.round(3).to_string())

print("\n\nCORRELACIONES ALTAS (|r| > 0.8):")
mask = (abs(corr) > 0.8) & (abs(corr) < 1.0)
found = False
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        if mask.iloc[i, j]:
            print(f"  {FEATURES[i]} <-> {FEATURES[j]}: r = {corr.iloc[i,j]:.3f}")
            found = True
if not found:
    print("  Ninguna correlacion > 0.8 entre las 7 features")

# Top 5 departamentos por Depositos_por_Capita (proxy de inclusión financiera)
print("\n\nTOP 5 DEPARTAMENTOS — Mayor inclusión financiera (Depositos_por_Capita)")
print("=" * 80)
top5 = df.sort_values('Depositos_por_Capita', ascending=False).head(5)
for _, r in top5.iterrows():
    print(f"  {r['Departamento']:<20} Dep/cap=S/.{r['Depositos_por_Capita']:>8,.0f}  Of/100k={r['Oficinas_por_100k']:.2f}  PBI/cap=S/.{r['PBI_por_Capita']:>8,.0f}  NBI={r['NBI_%_2024']:.1f}%  Internet/100k={r['Internet_por_100k']:,.0f}")

print("\n\nBOTTOM 5 — Menor inclusión financiera")
bottom5 = df.sort_values('Depositos_por_Capita').head(5)
for _, r in bottom5.iterrows():
    print(f"  {r['Departamento']:<20} Dep/cap=S/.{r['Depositos_por_Capita']:>8,.0f}  Of/100k={r['Oficinas_por_100k']:.2f}  PBI/cap=S/.{r['PBI_por_Capita']:>8,.0f}  NBI={r['NBI_%_2024']:.1f}%  Internet/100k={r['Internet_por_100k']:,.0f}")
