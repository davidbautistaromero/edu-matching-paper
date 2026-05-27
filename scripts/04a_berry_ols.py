#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04a_berry_ols.py
================
Paso 1 del modelo de demanda escolar: inversion de Berry (1994) + OLS.
Construye el indice visual v_j como combinacion lineal de features visuales
estimados en la especificacion completa (M3).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

# =============================================================================
# RUTAS
# =============================================================================

BASE        = Path(__file__).resolve().parent.parent
COLEGIOS    = BASE / 'data' / 'primary' / 'colegios_features_imputed.geojson'
CLIP_PATH   = BASE / 'data' / 'images' / 'clip' / 'gsv_clip_establecimiento.parquet'
CS_PATH     = BASE / 'data' / 'images' / 'segmentation' / 'gsv_cs_establecimiento.parquet'
EXCL_PATH   = BASE / 'data' / 'raw' / 'excluded_schools.csv'
OUT_TABLES  = BASE / 'reports' / 'tables'
OUT_PRIMARY = BASE / 'data' / 'primary'

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PRIMARY.mkdir(parents=True, exist_ok=True)

CLIP_VARS = ['mantenimiento', 'vegetacion_percibida', 'modernidad', 'seguridad_percibida']
CS_VARS   = ['infraestructura_vial', 'cerramiento']  # excluye edificacion (colineal con modernidad), vegetacion (colineal con vegetacion_percibida), vehiculos, mobiliario_urbano, referencia
CTRL_CONT = ['q_j', 'log_homicidios', 'log_dist_sitp', 'pct_no_oficial']
CTRL_BIN  = ['es_tecnico']

# =============================================================================
# PASO 1: CARGA Y FUSION
# =============================================================================

print("Cargando datos...")

gdf = gpd.read_file(COLEGIOS)
df  = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
df['id_establecimiento'] = df['id_establecimiento'].astype(str).str.strip()

excluidos = pd.read_csv(EXCL_PATH)['id_establecimiento'].astype(str).str.strip().tolist()

clip = pd.read_parquet(CLIP_PATH)
clip['id_establecimiento'] = clip['id_establecimiento'].astype(str).str.strip()
clip = clip[~clip['id_establecimiento'].isin(excluidos)]

cs = pd.read_parquet(CS_PATH)
cs['id_establecimiento'] = cs['id_establecimiento'].astype(str).str.strip()
cs = cs[~cs['id_establecimiento'].isin(excluidos)]

df = df.merge(clip[['id_establecimiento'] + CLIP_VARS], on='id_establecimiento', how='inner')
df = df.merge(cs[['id_establecimiento'] + CS_VARS],   on='id_establecimiento', how='inner')

print(f"Establecimientos tras merge: {len(df):,}")

# =============================================================================
# PASO 2: INVERSION DE BERRY
# =============================================================================

df['demanda_total']   = pd.to_numeric(df['demanda_total'], errors='coerce')
df['codigo_localidad'] = df['codigo_localidad'].astype(str).str.strip()
df['s0_localidad']    = pd.to_numeric(df['s0_localidad'], errors='coerce')
df = df[df['demanda_total'] > 0].copy()

s0_por_loc = (
    df[['codigo_localidad', 's0_localidad']]
    .drop_duplicates()
    .sort_values('codigo_localidad')
)
print("s0 por localidad:")
print(s0_por_loc.to_string(index=False))

demanda_loc = df.groupby('codigo_localidad')['demanda_total'].transform('sum')
M_t         = demanda_loc / (1 - df['s0_localidad'])

df['s_j']     = df['demanda_total'] / M_t
df['delta_j'] = np.log(df['s_j']) - np.log(df['s0_localidad'])
df = df[df['s_j'] > 0].copy()

print(f"Escuelas con s_j > 0: {len(df):,}")
print(f"Mercados (localidades): {df['codigo_localidad'].nunique()}")

# =============================================================================
# PASO 3: CONSTRUCCION DE VARIABLES
# =============================================================================

for col in ['q_j', 'dist_sitp_m', 'homicidios', 'pct_no_oficial'] + CLIP_VARS + CS_VARS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['log_homicidios'] = np.log1p(df['homicidios'])
df['log_dist_sitp']  = np.log1p(df['dist_sitp_m'])
df['es_tecnico']     = df['caracter_media'].str.contains('cnico', case=False, na=False).astype(float)

CONTINUAS = CLIP_VARS + CS_VARS + CTRL_CONT
todas_vars = CONTINUAS + CTRL_BIN

df = df.dropna(subset=['delta_j'] + todas_vars).reset_index(drop=True)
print(f"Muestra final tras dropna: {len(df):,}")

# Estandarizar variables continuas
for col in CONTINUAS:
    mu, sd = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sd

CLIP_Z = [f'{c}_z' for c in CLIP_VARS]
CS_Z   = [f'{c}_z' for c in CS_VARS]
CTRL_Z = [f'{c}_z' for c in CTRL_CONT] + CTRL_BIN

localidad_arr = df['codigo_localidad'].values
y = df['delta_j']

# =============================================================================
# PASO 4: CUATRO ESPECIFICACIONES OLS
# =============================================================================

def run_ols(y, X_vars, df, localidad_arr):
    X = sm.add_constant(df[X_vars].copy())
    modelo = sm.OLS(y, X).fit()
    res_hc1    = modelo.get_robustcov_results(cov_type='HC1')
    res_clust  = modelo.get_robustcov_results(cov_type='cluster', groups=localidad_arr)
    return modelo, res_hc1, res_clust

specs = {
    'M0': CTRL_Z,
    'M1': CLIP_Z + CTRL_Z,
    'M2': CS_Z   + CTRL_Z,
    'M3': CLIP_Z + CS_Z + CTRL_Z,
}

resultados = {}
for nombre, vars_spec in specs.items():
    mod, hc1, clust = run_ols(y, vars_spec, df, localidad_arr)
    resultados[nombre] = {'mod': mod, 'hc1': hc1, 'clust': clust, 'vars': vars_spec}
    print(f"\n{nombre}: N={len(df):,}  R²={mod.rsquared:.4f}  R²_adj={mod.rsquared_adj:.4f}")

# =============================================================================
# PASO 5: TABLA COMPARATIVA
# =============================================================================

todas_variables = ['const'] + list(dict.fromkeys(
    v for s in specs.values() for v in ['const'] + s
))

filas = []
for var in ['const'] + CLIP_Z + CS_Z + CTRL_Z:
    fila = {'variable': var}
    for nombre, res in resultados.items():
        hc1 = res['hc1']
        idx = list(res['hc1'].model.exog_names).index(var) if var in res['hc1'].model.exog_names else None
        if idx is not None:
            fila[f'{nombre}_coef'] = hc1.params[idx]
            fila[f'{nombre}_se']   = hc1.bse[idx]
            fila[f'{nombre}_pval'] = hc1.pvalues[idx]
        else:
            fila[f'{nombre}_coef'] = np.nan
            fila[f'{nombre}_se']   = np.nan
            fila[f'{nombre}_pval'] = np.nan
    filas.append(fila)

tabla = pd.DataFrame(filas)

# Fila de R²
r2_fila = {'variable': 'R2'}
r2adj_fila = {'variable': 'R2_adj'}
n_fila = {'variable': 'N'}
for nombre, res in resultados.items():
    r2_fila[f'{nombre}_coef']    = res['mod'].rsquared
    r2adj_fila[f'{nombre}_coef'] = res['mod'].rsquared_adj
    n_fila[f'{nombre}_coef']     = len(df)
    for suf in ['_se', '_pval']:
        r2_fila[f'{nombre}{suf}']    = np.nan
        r2adj_fila[f'{nombre}{suf}'] = np.nan
        n_fila[f'{nombre}{suf}']     = np.nan

tabla = pd.concat([tabla, pd.DataFrame([r2_fila, r2adj_fila, n_fila])], ignore_index=True)

# Imprimir tabla
print('\n' + '=' * 90)
print('RESULTADOS OLS — CUATRO ESPECIFICACIONES (SE HC1)')
print('=' * 90)
print(f"{'Variable':<35} {'M0':>12} {'M1':>12} {'M2':>12} {'M3':>12}")
print('-' * 90)

def fmt_cell(coef, pval):
    if np.isnan(coef):
        return '—'
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    return f'{coef:.4f}{sig}'

def fmt_se(se):
    if np.isnan(se):
        return ''
    return f'({se:.4f})'

for _, row in tabla.iterrows():
    if row['variable'] in ('R2', 'R2_adj', 'N'):
        vals = [f"{row[f'{n}_coef']:.4f}" if not np.isnan(row[f'{n}_coef']) else '—'
                for n in specs]
        print(f"{row['variable']:<35} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")
    else:
        coefs = [fmt_cell(row[f'{n}_coef'], row[f'{n}_pval']) for n in specs]
        ses   = [fmt_se(row[f'{n}_se']) for n in specs]
        print(f"{row['variable']:<35} {coefs[0]:>12} {coefs[1]:>12} {coefs[2]:>12} {coefs[3]:>12}")
        print(f"{'':35} {ses[0]:>12} {ses[1]:>12} {ses[2]:>12} {ses[3]:>12}")

print('=' * 90)
print('Nota: *** p<0.01, ** p<0.05, * p<0.1. SE HC1 robustos.')

tabla.to_csv(OUT_TABLES / 'berry_ols_specs.csv', index=False, encoding='utf-8-sig')
print(f'\nTabla guardada: {OUT_TABLES / "berry_ols_specs.csv"}')



# Guardar delta_j
delta_out = df[['id_establecimiento', 'codigo_localidad', 'delta_j', 's_j', 'demanda_total']].copy()
delta_out.to_parquet(OUT_PRIMARY / 'berry_delta_j.parquet', index=False)
print(f'Delta_j guardado: {OUT_PRIMARY / "berry_delta_j.parquet"}')
