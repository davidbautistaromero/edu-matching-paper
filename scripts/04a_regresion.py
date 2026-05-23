#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04a_regresion.py
================
Estimacion de demanda escolar via logit agregado de Berry (1994) y
logit mixto BLP con interacciones de ingreso via PyBLP.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# =============================================================================
# RUTAS
# =============================================================================

BASE        = Path(__file__).resolve().parent.parent
COLEGIOS    = BASE / 'data' / 'primary' / 'colegios_features_imputed.geojson'
NMF_PATH    = BASE / 'data' / 'images' / 'embeddings' / 'gsv_nmf_K6.parquet'
OUT_TABLES  = BASE / 'reports' / 'tables'

OUT_TABLES.mkdir(parents=True, exist_ok=True)

OUTSIDE_SHARE = 0.05
NMF_FEATURES  = [f'topic_{i}' for i in range(1, 7)]

# =============================================================================
# PASO 1: CARGA Y FUSION
# =============================================================================

gdf = gpd.read_file(COLEGIOS)
df  = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
df['id_establecimiento'] = df['id_establecimiento'].astype(str).str.strip()

nmf = pd.read_parquet(NMF_PATH)
nmf['id_establecimiento'] = nmf['id_establecimiento'].astype(str).str.strip()

df = df.merge(nmf[['id_establecimiento'] + NMF_FEATURES],
              on='id_establecimiento', how='inner')

print(f'Establecimientos tras merge NMF: {len(df):,}')

# =============================================================================
# PASO 2: CUOTAS DE MERCADO (INVERSION DE BERRY)
# =============================================================================

df['demanda_total'] = pd.to_numeric(df['demanda_total'], errors='coerce')
df['codigo_localidad'] = df['codigo_localidad'].astype(str).str.strip()

df = df[df['demanda_total'] > 0].copy()

demanda_loc = df.groupby('codigo_localidad')['demanda_total'].transform('sum')
M_t         = demanda_loc / (1 - OUTSIDE_SHARE)

df['M_t']    = M_t
df['s_j']    = df['demanda_total'] / M_t
df['s_0']    = OUTSIDE_SHARE
df['delta_j'] = np.log(df['s_j']) - np.log(df['s_0'])

df = df[df['s_j'] > 0].copy()
print(f'Escuelas con s_j > 0: {len(df):,}')
print(f'Mercados (localidades): {df["codigo_localidad"].nunique()}')

# =============================================================================
# PASO 3: CONSTRUCCION DE VARIABLES
# =============================================================================

for col in ['q_j', 'dist_sitp_m', 'hurto_personas', 'homicidios',
            'puntaje_2023', 'punt_global_2022', 'punt_global_2020']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['log_dist_sitp']          = np.log1p(df['dist_sitp_m'])
df['log_hurto']              = np.log1p(df['hurto_personas'])
df['log_homicidios']         = np.log1p(df['homicidios'])
df['es_rural']               = df['zona'].isin(['RURAL', 'EXPANSION']).astype(float)
df['es_tecnico']             = df['caracter_media'].str.contains('cnico', case=False, na=False).astype(float)
df.loc[df['caracter_media'].isin(['Sin informacion', 'Sin información']), 'es_tecnico'] = np.nan

CONTINUAS = NMF_FEATURES + ['q_j', 'log_dist_sitp', 'log_hurto', 'log_homicidios']
TODAS_X   = CONTINUAS + ['es_rural', 'es_tecnico']

df = df.dropna(subset=['delta_j'] + TODAS_X).reset_index(drop=True)
print(f'Muestra final tras dropna: {len(df):,}')

# Estandarizar variables continuas
for col in CONTINUAS:
    mu, sigma    = df[col].mean(), df[col].std()
    df[f'{col}_norm'] = (df[col] - mu) / sigma

df['q_j_norm'] = df['q_j_norm']

VARS_REG = [f'{c}_norm' for c in CONTINUAS] + ['es_rural', 'es_tecnico']

X_raw = df[VARS_REG].copy()
y     = df['delta_j']

X = sm.add_constant(X_raw)
localidad_arr = df['codigo_localidad'].values

# =============================================================================
# PASO 4: OLS CON SE ROBUSTOS
# =============================================================================

modelo_ols     = sm.OLS(y, X).fit()
res_hc1        = modelo_ols.get_robustcov_results(cov_type='HC1')
res_cluster    = modelo_ols.get_robustcov_results(
    cov_type='cluster', groups=localidad_arr
)

print('\n' + '=' * 70)
print('REGRESION LOGIT BERRY — OLS')
print('=' * 70)
print(f'N escuelas:   {len(df):,}')
print(f'N mercados:   {df["codigo_localidad"].nunique()}')
print(f'R²:           {modelo_ols.rsquared:.4f}')
print(f'R² ajustado:  {modelo_ols.rsquared_adj:.4f}')
print(f'F-statistic:  {modelo_ols.fvalue:.3f}  (p={modelo_ols.f_pvalue:.4g})')
print()

# Tabla de coeficientes combinando ambos SE
nombres = X.columns.tolist()
coefs   = modelo_ols.params
se_hc1  = res_hc1.bse
se_clus = res_cluster.bse
t_hc1   = res_hc1.tvalues
p_hc1   = res_hc1.pvalues
ci      = res_hc1.conf_int()

tabla_coefs = pd.DataFrame({
    'variable':    nombres,
    'coef':        np.asarray(coefs),
    'se_hc1':      np.asarray(se_hc1),
    'se_cluster':  np.asarray(se_clus),
    't_stat_hc1':  np.asarray(t_hc1),
    'p_value_hc1': np.asarray(p_hc1),
    'ci95_lo':     np.asarray(ci)[:, 0],
    'ci95_hi':     np.asarray(ci)[:, 1],
})

print(f"{'Variable':<30} {'Coef':>10} {'SE_HC1':>10} {'SE_Clust':>10} "
      f"{'t(HC1)':>9} {'p(HC1)':>9}")
print('-' * 82)
for _, r in tabla_coefs.iterrows():
    sig = '***' if r['p_value_hc1'] < 0.01 else ('**' if r['p_value_hc1'] < 0.05 else
          ('*' if r['p_value_hc1'] < 0.1 else ''))
    print(f"{r['variable']:<30} {r['coef']:>10.4f} {r['se_hc1']:>10.4f} "
          f"{r['se_cluster']:>10.4f} {r['t_stat_hc1']:>9.3f} {r['p_value_hc1']:>9.4f} {sig}")

sig_topics = [f'topic_{i}_norm' for i in range(1, 7)
              if tabla_coefs.loc[tabla_coefs['variable'] == f'topic_{i}_norm',
                                 'p_value_hc1'].values[0] < 0.05]
print(f'\nTopics significativos al 5%: {sig_topics if sig_topics else "ninguno"}')

# =============================================================================
# PASO 5: OUTPUTS
# =============================================================================

# --- Tabla de coeficientes ---
tabla_coefs.to_csv(OUT_TABLES / 'berry_logit_ols.csv', index=False, encoding='utf-8-sig')
print(f'\nTabla guardada: {OUT_TABLES / "berry_logit_ols.csv"}')

# --- Tabla de cuotas de mercado ---
market_df = df[['id_establecimiento', 'codigo_localidad', 'demanda_total', 'M_t', 's_j', 'delta_j']].copy()
market_df.to_csv(OUT_TABLES / 'berry_market_shares.csv', index=False, encoding='utf-8-sig')
print(f'Tabla guardada: {OUT_TABLES / "berry_market_shares.csv"}')

# =============================================================================
# PASO 6: BERRY LOGIT EXTENDIDO — INTERACCIONES UPZ-INGRESO
# =============================================================================

print('\n' + '=' * 62)
print('BERRY LOGIT EXTENDIDO — INTERACCIONES UPZ-INGRESO')
print('=' * 62)

# --------------------------------------------------------------------------
# 6.1  Ingreso ponderado por UPZ
# --------------------------------------------------------------------------
FAMILIAS_PATH = BASE / 'data' / 'processed' / 'familias_expandidas.parquet'
try:
    familias_raw = pd.read_parquet(FAMILIAS_PATH, engine='pyarrow')
    print(f'Familias cargadas: {len(familias_raw):,}')
except Exception as e:
    print(f'Parquet fallido ({e}). Usando CSV slim...')
    familias_raw = pd.read_csv('/tmp/blp_data/familias_expandidas_slim.csv')
    print(f'Familias cargadas (CSV): {len(familias_raw):,}')

for col in ['N_ingpc', 'FEX_C']:
    if col in familias_raw.columns:
        familias_raw[col] = pd.to_numeric(familias_raw[col], errors='coerce')

familias_raw = familias_raw.dropna(subset=['N_ingpc', 'FEX_C'])
familias_raw = familias_raw[familias_raw['N_ingpc'] > 0].copy()

# UPZ-level income (primary)
upz_income_map = {}
if 'COD_UPZ_GRUPO' in familias_raw.columns:
    fam_upz = familias_raw.copy()
    fam_upz['COD_UPZ_GRUPO'] = pd.to_numeric(fam_upz['COD_UPZ_GRUPO'], errors='coerce')
    fam_upz = fam_upz.dropna(subset=['COD_UPZ_GRUPO'])
    fam_upz['COD_UPZ_GRUPO'] = fam_upz['COD_UPZ_GRUPO'].astype(int)
    fam_upz['_inc_w'] = fam_upz['N_ingpc'] * fam_upz['FEX_C']
    y_upz = (
        fam_upz.groupby('COD_UPZ_GRUPO')[['_inc_w', 'FEX_C']].sum()
        .assign(y_upz=lambda d: d['_inc_w'] / d['FEX_C'])
        [['y_upz']].reset_index()
    )
    y_upz['y_upz_norm'] = y_upz['y_upz'] / y_upz['y_upz'].mean()
    upz_income_map = dict(zip(y_upz['COD_UPZ_GRUPO'], y_upz['y_upz_norm']))
    print(f'UPZs con ingreso: {len(y_upz):,}')
else:
    print('COD_UPZ_GRUPO no disponible — se usará fallback por localidad')

# Localidad-level income (fallback)
familias_raw['COD_LOCALIDAD'] = pd.to_numeric(
    familias_raw['COD_LOCALIDAD'], errors='coerce'
)
familias_raw = familias_raw.dropna(subset=['COD_LOCALIDAD'])
familias_raw['COD_LOCALIDAD'] = familias_raw['COD_LOCALIDAD'].astype(int).astype(str)
familias_raw['_inc_w'] = familias_raw['N_ingpc'] * familias_raw['FEX_C']
y_loc = (
    familias_raw.groupby('COD_LOCALIDAD')[['_inc_w', 'FEX_C']].sum()
    .assign(y_loc=lambda d: d['_inc_w'] / d['FEX_C'])
    [['y_loc']].reset_index()
)
y_loc['y_loc_norm'] = y_loc['y_loc'] / y_loc['y_loc'].mean()
loc_income_map = dict(zip(y_loc['COD_LOCALIDAD'], y_loc['y_loc_norm']))
print(f'Localidades con ingreso: {len(y_loc):,}')

# --------------------------------------------------------------------------
# 6.2  Distancia haversine media familia → colegio por localidad
# --------------------------------------------------------------------------

df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

fam_coords_ok = {'lat', 'lon'}.issubset(familias_raw.columns)
if fam_coords_ok:
    familias_raw['lat'] = pd.to_numeric(familias_raw['lat'], errors='coerce')
    familias_raw['lon'] = pd.to_numeric(familias_raw['lon'], errors='coerce')
    fam_for_dist = familias_raw.dropna(subset=['lat', 'lon']).copy()
else:
    fam_for_dist = pd.DataFrame()

N_SAMPLE  = 500
dist_rows = []

for loc in df['codigo_localidad'].astype(str).str.strip().unique():
    esc_loc = df[
        (df['codigo_localidad'].astype(str).str.strip() == loc) &
        df['lat'].notna() & df['lon'].notna()
    ][['id_establecimiento', 'lat', 'lon']]
    if len(esc_loc) == 0:
        continue

    if len(fam_for_dist) > 0:
        fam_loc = fam_for_dist[fam_for_dist['COD_LOCALIDAD'] == loc]
    else:
        fam_loc = pd.DataFrame()

    if len(fam_loc) == 0:
        for eid in esc_loc['id_establecimiento']:
            dist_rows.append({'id_establecimiento': eid, 'd_bar': np.nan})
        continue

    fex = fam_loc['FEX_C'].values.astype(float) if 'FEX_C' in fam_loc.columns else np.ones(len(fam_loc))
    fex = np.where(~np.isfinite(fex) | (fex <= 0), 1.0, fex)
    fex = fex / fex.sum()

    rng    = np.random.default_rng(42)
    n_draw = min(N_SAMPLE, len(fam_loc))
    idx    = rng.choice(len(fam_loc), size=n_draw, replace=False, p=fex)
    sample = fam_loc.iloc[idx]
    flats  = np.radians(sample['lat'].values.astype(float))
    flons  = np.radians(sample['lon'].values.astype(float))

    R_KM = 6371.0
    for _, esc in esc_loc.iterrows():
        slat = np.radians(float(esc['lat']))
        slon = np.radians(float(esc['lon']))
        dphi = flats - slat
        dlam = flons - slon
        a    = np.sin(dphi / 2)**2 + np.cos(slat) * np.cos(flats) * np.sin(dlam / 2)**2
        d_km = R_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist_rows.append({'id_establecimiento': esc['id_establecimiento'],
                          'd_bar': float(np.mean(d_km))})

dist_df = pd.DataFrame(dist_rows)
med_d = dist_df['d_bar'].median()
dist_df['d_bar']         = dist_df['d_bar'].fillna(med_d)
dist_df['log_dist_mean'] = np.log1p(dist_df['d_bar'])

df = df.merge(dist_df[['id_establecimiento', 'log_dist_mean']],
              on='id_establecimiento', how='left')
df['log_dist_mean'] = df['log_dist_mean'].fillna(df['log_dist_mean'].median())
print(f'log_dist_mean — media={df["log_dist_mean"].mean():.3f}, '
      f'N válidos={df["log_dist_mean"].notna().sum():,}')

# --------------------------------------------------------------------------
# 6.3  Ingreso UPZ → cada colegio
# --------------------------------------------------------------------------
df['_upz_num'] = pd.to_numeric(df['codigo_upz'], errors='coerce')

def _map_upz_income(x):
    if pd.isna(x):
        return np.nan
    return upz_income_map.get(int(x), np.nan)

df['y_upz_norm'] = df['_upz_num'].apply(_map_upz_income) if upz_income_map else np.nan

# Fallback: localidad-level income
loc_key = df['codigo_localidad'].astype(str).str.strip()
mask_miss = df['y_upz_norm'].isna()
df.loc[mask_miss, 'y_upz_norm'] = loc_key[mask_miss].map(loc_income_map)
df['y_upz_norm'] = df['y_upz_norm'].fillna(1.0)  # global mean fallback

n_upz_ok  = int((~mask_miss).sum())
n_loc_fb  = int(mask_miss.sum() - (df['y_upz_norm'] == 1.0).sum())
print(f'Escuelas con ingreso UPZ: {n_upz_ok:,}  (fallback localidad: {n_loc_fb:,})')

# --------------------------------------------------------------------------
# 6.4  Términos de interacción
# --------------------------------------------------------------------------
TOPIC_NORM = [f'topic_{k}_norm' for k in range(1, 7)]
for k in range(1, 7):
    df[f'topic_{k}_norm_x_income'] = df[f'topic_{k}_norm'] * df['y_upz_norm']
df['log_dist_mean_x_income'] = df['log_dist_mean'] * df['y_upz_norm']

# --------------------------------------------------------------------------
# 6.5  Regresión extendida (OLS, HC1)
# --------------------------------------------------------------------------
INTR_VARS = [f'topic_{k}_norm_x_income' for k in range(1, 7)] + ['log_dist_mean_x_income']
EXT_VARS  = (
    TOPIC_NORM + INTR_VARS +
    ['log_dist_mean', 'q_j_norm', 'log_dist_sitp_norm',
     'log_homicidios_norm', 'es_tecnico']
)

df_ext = df.dropna(subset=['delta_j'] + EXT_VARS).reset_index(drop=True)
X_ext  = sm.add_constant(df_ext[EXT_VARS].copy())
y_ext  = df_ext['delta_j']

modelo_ext  = sm.OLS(y_ext, X_ext).fit()
res_ext_hc1 = modelo_ext.get_robustcov_results(cov_type='HC1')

_cols    = X_ext.columns.tolist()
coef_ext = dict(zip(_cols, res_ext_hc1.params))
se_ext   = dict(zip(_cols, res_ext_hc1.bse))
pval_ext = dict(zip(_cols, res_ext_hc1.pvalues))

# --------------------------------------------------------------------------
# 6.6  Tabla comparativa y estadísticos
# --------------------------------------------------------------------------

def _base_beta(var):
    row = tabla_coefs[tabla_coefs['variable'] == var]
    return float(row['coef'].values[0]) if len(row) > 0 else np.nan

print('\n' + '=' * 68)
print('Berry Logit: Base vs Extended (UPZ-Income Interactions)')
print('=' * 68)
print(f"{'Variable':<32} {'Base b':>8} {'Extd b':>9} {'g(income)':>11} {'p-val(g)':>10}")
print('-' * 68)

for k in range(1, 7):
    v_main = f'topic_{k}_norm'
    v_int  = f'topic_{k}_norm_x_income'
    base_b = _base_beta(v_main)
    ext_b  = coef_ext.get(v_main, np.nan)
    gamma  = coef_ext.get(v_int,  np.nan)
    pv     = pval_ext.get(v_int,  np.nan)
    sig    = '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.1 else ''))
    print(f"  topic_{k:<27} {base_b:>8.3f} {ext_b:>9.3f} {gamma:>11.3f} {pv:>10.4f} {sig}")

ext_bd  = coef_ext.get('log_dist_mean', np.nan)
gamma_d = coef_ext.get('log_dist_mean_x_income', np.nan)
pv_d    = pval_ext.get('log_dist_mean_x_income', np.nan)
sig_d   = '***' if pv_d < 0.01 else ('**' if pv_d < 0.05 else ('*' if pv_d < 0.1 else ''))
print(f"  {'log_dist_mean':<30} {'—':>8} {ext_bd:>9.3f} {gamma_d:>11.3f} {pv_d:>10.4f} {sig_d}")

print('-' * 68)
print('  Controls (q_j_norm, log_dist_sitp_norm, log_homicidios_norm, es_tecnico): incluidos')
print('=' * 68)
print('\nInterpretacion:')
print('  g_k > 0 => UPZs con mayor ingreso valoran topic_k MAS')
print('  g_k < 0 => UPZs con menor ingreso valoran topic_k MAS (senal visual importa mas para pobres)')
print('  g_d > 0 => mayor ingreso menos sensible a distancia (consistente con Hastings)')

# R² comparison
print(f'\nR² base (PASO 4):    {modelo_ols.rsquared:.4f}')
print(f'R² extendido:         {modelo_ext.rsquared:.4f}')

# F-test for joint significance of interaction terms
intr_idx = [_cols.index(v) for v in INTR_VARS]
R_mat    = np.zeros((len(INTR_VARS), len(_cols)))
for i, ci in enumerate(intr_idx):
    R_mat[i, ci] = 1.0

f_test   = res_ext_hc1.wald_test(R_mat, use_f=True, scalar=True)
f_stat   = float(f_test.statistic)
f_pval   = float(f_test.pvalue)
print(f'\nF-test conjunto interacciones: F={f_stat:.3f},  p={f_pval:.4g}'
      f'  (df={len(INTR_VARS)},{int(res_ext_hc1.df_resid)})')

n_upz_ext = int(df_ext['_upz_num'].dropna().nunique()) if '_upz_num' in df_ext.columns else 'N/A'
print(f'\nN observaciones:           {len(df_ext):,}')
print(f'N mercados (localidades):  {df_ext["codigo_localidad"].nunique()}')
print(f'N UPZs en muestra:         {n_upz_ext}')

# --------------------------------------------------------------------------
# 6.7  Guardar outputs
# --------------------------------------------------------------------------
ci_ext = res_ext_hc1.conf_int()
ext_coef_table = pd.DataFrame({
    'variable':  _cols,
    'coef':      list(coef_ext.values()),
    'se_hc1':    list(se_ext.values()),
    't_stat':    np.asarray(res_ext_hc1.tvalues),
    'p_value':   list(pval_ext.values()),
    'ci95_lo':   np.asarray(ci_ext)[:, 0],
    'ci95_hi':   np.asarray(ci_ext)[:, 1],
})
ext_coef_table.to_csv(OUT_TABLES / 'berry_extended_results.csv', index=False, encoding='utf-8-sig')
print(f'\nTabla guardada: {OUT_TABLES / "berry_extended_results.csv"}')

inc_rows = []
for v in INTR_VARS:
    g   = float(coef_ext.get(v, np.nan))
    s   = float(se_ext.get(v,   np.nan))
    p   = float(pval_ext.get(v, np.nan))
    t   = g / s if (np.isfinite(s) and s != 0) else np.nan
    if np.isfinite(g):
        interp = ('UPZs ricas lo valoran MÁS'   if g >  0.05 else
                  'UPZs pobres lo valoran MÁS'   if g < -0.05 else
                  'efecto ingreso neutro')
    else:
        interp = 'n/a'
    inc_rows.append({'variable': v, 'gamma': g, 'se_hc1': s,
                     't_stat': t, 'p_value': p, 'interpretation': interp})

pd.DataFrame(inc_rows).to_csv(
    OUT_TABLES / 'berry_income_interactions.csv', index=False, encoding='utf-8-sig'
)
print(f'Tabla guardada: {OUT_TABLES / "berry_income_interactions.csv"}')
