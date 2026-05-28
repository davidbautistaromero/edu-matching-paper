#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_blp.py
==========
BLP con micro-momentos para demanda escolar en Bogota.
Estima interacciones ingreso x senales visuales (CLIP) e ingreso x distancia.

Ref: reports/paper/blp_contexto_paper.md

Modelo:
  u_ij = delta_j + pi1*y_i*seg_j + lam0*d_ij + lam1*y_i*d_ij + eps_ij

Micro-momentos (BLP 2004, Petrin 2002):
  m_j_obs = ingreso medio ponderado por proximidad (1/d_ij)
  d_j_obs = distancia media ponderada por FEX_C
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm as sp_norm

# =============================================================================
# RUTAS
# =============================================================================

BASE        = Path(__file__).resolve().parent.parent
DELTA_PATH  = BASE / 'data' / 'primary'    / 'berry_delta_j.parquet'
COLEGIOS    = BASE / 'data' / 'primary'    / 'colegios_features_imputed.geojson'
CLIP_PATH   = BASE / 'data' / 'images'     / 'clip' / 'gsv_clip_establecimiento.parquet'
FAM_PATH    = BASE / 'data' / 'processed'  / 'familias_expandidas.parquet'
DIST_PATH   = BASE / 'data' / 'processed'  / 'distancias_expandidas.parquet'
EXCL_PATH   = BASE / 'data' / 'raw'        / 'excluded_schools.csv'

OUT_TABLES  = BASE / 'reports' / 'tables'
OUT_PRIMARY = BASE / 'data' / 'primary'
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PRIMARY.mkdir(parents=True, exist_ok=True)

OUTSIDE_SHARE = 0.05
N_SAMPLE_LOC  = 300
SEED          = 42

CLIP_VARS = ['seguridad_percibida']
CTRL_CONT = ['q_j', 'log_homicidios', 'log_dist_sitp', 'pct_no_oficial']
CTRL_BIN  = ['es_tecnico']

# =============================================================================
# PASO 1: COLEGIOS — carga, exclusion rural, merge delta_j y CLIP
# =============================================================================

print("Cargando datos de colegios...")

excluidos = pd.read_csv(EXCL_PATH)['id_establecimiento'].astype(str).str.strip().tolist()

gdf = gpd.read_file(COLEGIOS)
df  = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
df['id_establecimiento'] = df['id_establecimiento'].astype(str).str.strip()
df = df[~df['id_establecimiento'].isin(excluidos)].copy()
print(f"  Colegios urbanos: {len(df):,}")

delta = pd.read_parquet(DELTA_PATH)
delta['id_establecimiento'] = delta['id_establecimiento'].astype(str).str.strip()
df = df.merge(delta[['id_establecimiento', 'delta_j', 's_j']], on='id_establecimiento', how='inner')
print(f"  Colegios con delta_j: {len(df):,}")

clip = pd.read_parquet(CLIP_PATH)
clip['id_establecimiento'] = clip['id_establecimiento'].astype(str).str.strip()
clip = clip[~clip['id_establecimiento'].isin(excluidos)]
df = df.merge(clip[['id_establecimiento'] + CLIP_VARS], on='id_establecimiento', how='inner')
print(f"  Colegios tras merge CLIP: {len(df):,}")

# =============================================================================
# PASO 2: VARIABLES DE COLEGIO Y ESTANDARIZACION
# =============================================================================

for col in ['q_j', 'dist_sitp_m', 'homicidios', 'pct_no_oficial'] + CLIP_VARS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['log_homicidios'] = np.log1p(df['homicidios'])
df['log_dist_sitp']  = np.log1p(df['dist_sitp_m'])
df['es_tecnico']     = df['caracter_media'].str.contains('cnico', case=False, na=False).astype(float)

df = df.dropna(subset=CLIP_VARS + CTRL_CONT + CTRL_BIN).reset_index(drop=True)
print(f"  Colegios tras dropna: {len(df):,}")
print(f"  Mercados (localidades): {df['codigo_localidad'].nunique()}")

# Estandarizar continuas (z-score en muestra de colegios)
for col in CLIP_VARS + ['q_j', 'log_homicidios', 'log_dist_sitp', 'pct_no_oficial']:
    mu, sd = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sd

CLIP_Z = [f'{c}_z' for c in CLIP_VARS]
CTRL_Z = [f'{c}_z' for c in ['q_j', 'log_homicidios', 'log_dist_sitp', 'pct_no_oficial']] + CTRL_BIN

# BLP instruments for q_j: calidad de rivales ponderada por 1/distancia (por localidad)
from sklearn.metrics import pairwise_distances

_coords = np.deg2rad(df[['lat', 'lon']].values.astype(float))
_dist_km = pairwise_distances(_coords, metric='haversine') * 6371.0
np.fill_diagonal(_dist_km, np.inf)

_locs = df['codigo_localidad'].apply(
    lambda x: str(int(float(x))).zfill(2) if pd.notna(x) and str(x).strip() != '' else None
).values
_q_vals = df['q_j_z'].values

_rival_q_w = []
for i in range(len(df)):
    mask = (_locs == _locs[i])
    mask[i] = False
    others_idx = np.where(mask)[0]
    if len(others_idx) > 0:
        d = _dist_km[i, others_idx]
        w = 1.0 / np.maximum(d, 0.1)
        w = w / w.sum()
        _rival_q_w.append(np.dot(w, _q_vals[others_idx]))
    else:
        _rival_q_w.append(0.0)

df['mean_q_rivals'] = _rival_q_w
mu, sd = df['mean_q_rivals'].mean(), df['mean_q_rivals'].std()
df['mean_q_rivals_z'] = (df['mean_q_rivals'] - mu) / sd if sd > 0 else 0.0

# First-stage: q_j_z ~ mean_q_rivals_z + other exogenous
import statsmodels.api as sm
_iv_vars = ['mean_q_rivals_z']
_fs_X = sm.add_constant(df[_iv_vars +
                             ['log_homicidios_z', 'log_dist_sitp_z', 'pct_no_oficial_z', 'es_tecnico']
                            + CLIP_Z])
_fs_y = df['q_j_z']
_fs_mod = sm.OLS(_fs_y, _fs_X).fit()
_fs_F = _fs_mod.fvalue
print(f"\n  FIRST STAGE: q_j_z ~ instruments + controls")
print(f"    F-statistic: {_fs_F:.2f}  (>10 = strong)")
print(f"    R2: {_fs_mod.rsquared:.4f}")
for iv in _iv_vars:
    print(f"    {iv}: coef={_fs_mod.params[iv]:+.4f} (p={_fs_mod.pvalues[iv]:.4f})")

print("\nEstadisticos variables estandarizadas (colegios):")
for v in CLIP_Z + CTRL_Z[:-1]:
    print(f"  {v:<35} mean={df[v].mean():+.4f}  std={df[v].std():.4f}"
          f"  [{df[v].min():.2f}, {df[v].max():.2f}]")

# =============================================================================
# PASO 3: FAMILIAS — ingreso normalizado, muestra 300 por localidad
# =============================================================================

print("\nCargando familias...")
fam = pd.read_parquet(FAM_PATH)
fam['_orig_idx']    = fam.index                                          # para slicear dist despues
fam['COD_LOCALIDAD'] = fam['COD_LOCALIDAD'].astype(float).astype(int).astype(str).str.zfill(2)
fam['N_ingpc']       = pd.to_numeric(fam['N_ingpc'], errors='coerce')
fam = fam.dropna(subset=['N_ingpc', 'FEX_C']).copy()

mean_ingpc  = fam['N_ingpc'].mean()
fam['y_i']  = fam['N_ingpc'] / mean_ingpc
print(f"  Total familias: {len(fam):,}  |  ingreso medio referencia: {mean_ingpc:,.0f}")

# Muestreo estratificado: N_SAMPLE_LOC familias por localidad ponderado por FEX_C
rng    = np.random.default_rng(SEED)
grupos = []
for loc, grp in fam.groupby('COD_LOCALIDAD'):
    pesos = grp['FEX_C'].clip(lower=0)
    total = pesos.sum()
    if total == 0:
        continue
    pesos = pesos / total
    n_sel = min(N_SAMPLE_LOC, len(grp))
    idx   = rng.choice(grp.index, size=n_sel, replace=False, p=pesos.values)
    grupos.append(grp.loc[idx])

fam_sample = pd.concat(grupos).reset_index(drop=True)
print(f"  Familias en muestra: {len(fam_sample):,}")
print(f"  Localidades cubiertas: {fam_sample['COD_LOCALIDAD'].nunique()}")
print(f"  y_i — mean={fam_sample['y_i'].mean():.4f}  std={fam_sample['y_i'].std():.4f}"
      f"  [{fam_sample['y_i'].min():.4f}, {fam_sample['y_i'].max():.4f}]")
print("\n  Familias por localidad:")
print(fam_sample.groupby('COD_LOCALIDAD').size().to_string())

# =============================================================================
# PASO 4: MATRIZ DE DISTANCIAS (familias x colegios)
# =============================================================================

print("\nCargando matriz de distancias (puede tardar)...")
dist_full = pd.read_parquet(DIST_PATH)
print(f"  Dimensiones originales: {dist_full.shape}")

escuelas_ok   = df['id_establecimiento'].tolist()
escuelas_dist = [c for c in escuelas_ok if c in dist_full.columns]
orig_idx      = fam_sample['_orig_idx'].values

dist_sub = dist_full.loc[orig_idx, escuelas_dist].copy()
print(f"  Escuelas cubiertas por distancias: {len(escuelas_dist):,} / {len(escuelas_ok):,}")
print(f"  Dimension submatriz (I x J): {dist_sub.shape}")
print(f"  Distancia media (m): {dist_sub.values.mean():,.0f}")
print(f"  Distancia mediana (m): {np.median(dist_sub.values):,.0f}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RESUMEN CARGA BLP")
print("=" * 60)
print(f"  Colegios (J):              {len(df):>6,}")
print(f"  Familias en muestra (I):   {len(fam_sample):>6,}")
print(f"  Localidades/mercados (T):  {df['codigo_localidad'].nunique():>6,}")
print(f"  OUTSIDE_SHARE:             {OUTSIDE_SHARE}")
print(f"  Escuelas con distancias:   {len(escuelas_dist):>6,}")
print(f"  Escuelas sin distancias:   {len(escuelas_ok) - len(escuelas_dist):>6,}")
print("=" * 60)

# =============================================================================
# PASO 5: CONSTRUCCION DE market_data POR LOCALIDAD
# =============================================================================

print("\nConstruyendo market_data por localidad...")

# Normalizar codigo_localidad al mismo formato zfill(2) que fam_sample
df['_loc_key'] = df['codigo_localidad'].apply(
    lambda x: str(int(float(x))).zfill(2) if pd.notna(x) and str(x).strip() != '' else None
)

# Indice por id_establecimiento para alineacion eficiente con .loc
df_idx = df.set_index('id_establecimiento')

market_data = {}

for t, fam_t in fam_sample.groupby('COD_LOCALIDAD'):
    # Colegios del mercado t (solo los que tienen distancias disponibles)
    df_t  = df[df['_loc_key'] == t]
    j_all = df_t['id_establecimiento'].tolist()
    j_ids = [j for j in j_all if j in dist_sub.columns]
    if not j_ids:
        continue

    # Indices originales de las familias (usados como row labels en dist_sub)
    fam_orig_idx = fam_t['_orig_idx'].values

    # Submatriz de distancias: filas=familias (por _orig_idx), cols=escuelas del mercado
    dist_m = dist_sub.loc[fam_orig_idx, j_ids]   # metros

    dist_km_arr  = dist_m.values
    dist_log_arr = np.log1p(dist_km_arr)

    # Arrays de colegio alineados por j_ids via .loc
    df_t_al   = df_idx.loc[j_ids]
    seg_z_arr = df_t_al['seguridad_percibida_z'].values
    s_obs_arr = df_t_al['s_j'].values
    delta_arr = df_t_al['delta_j'].values

    fam_y_arr = fam_t['y_i'].values
    fex_w_arr = fam_t['FEX_C'].values

    market_data[t] = dict(
        J_ids      = j_ids,
        seg_z      = seg_z_arr,
        s_obs      = s_obs_arr,
        delta_init = delta_arr,
        fam_y      = fam_y_arr,
        fex_w      = fex_w_arr,
        dist_km    = dist_km_arr,
        dist_log   = dist_log_arr,
        dist_raw   = dist_km_arr,   # alias para micro-momentos
    )

    J = len(j_ids)
    I = len(fam_orig_idx)
    print(f"  Mercado {t}: J={J:>3}  I={I:>4}  dist_media={dist_km_arr.mean():5.2f} km")

print(f"\nTotal mercados en market_data: {len(market_data)}")

# =============================================================================
# PASO 6: FUNCIONES BLP
# =============================================================================

def compute_shares(theta, market_data):
    """
    Para cada mercado t, calcula cuotas predichas S_hat_j = mean_i(s_ij).

    theta = (pi1, lam0, lam1)
      pi1  : y_i * seg_z_j
      lam0 : log(1 + d_ij)
      lam1 : y_i * log(1 + d_ij)

    Returns dict {market -> S_hat array shape (J,)}
    """
    pi1, lam0, lam1 = theta
    shares = {}
    for t, md in market_data.items():
        delta  = md['delta_init']          # (J,)
        seg_z  = md['seg_z']               # (J,)
        log_d  = md['dist_log']            # (I, J)
        y_i    = md['fam_y'][:, None]      # (I, 1)

        # mu_ij: (I, J)
        mu = pi1 * y_i * seg_z + lam0 * log_d + lam1 * y_i * log_d

        # clip antes de exp para evitar overflow
        exponents = np.clip(delta + mu, -500, 500)  # (I, J)
        exp_ij    = np.exp(exponents)
        denom     = 1.0 + exp_ij.sum(axis=1, keepdims=True)  # (I, 1)
        s_ij      = exp_ij / denom                            # (I, J)
        shares[t] = s_ij.mean(axis=0)                         # (J,)
    return shares


def contraction_mapping(theta, market_data, tol=1e-8, max_iter=200):
    """
    Berry (1994) contraction: delta_j <- delta_j + log(s_obs) - log(S_hat).

    Itera hasta que max |delta_new - delta| < tol o se agote max_iter.
    Returns (delta_dict, n_iters) donde delta_dict = {market -> delta array}.
    """
    # Copiar deltas iniciales
    delta_dict = {t: md['delta_init'].copy() for t, md in market_data.items()}

    md_working = {
        t: dict(md, delta_init=delta_dict[t]) for t, md in market_data.items()
    }

    n_iters = 0
    for iteration in range(max_iter):
        n_iters += 1
        max_change = 0.0
        shares = compute_shares(theta, md_working)

        for t, md in md_working.items():
            s_obs   = md['s_obs']
            S_hat   = shares[t]
            delta_new = md['delta_init'] + np.log(s_obs + 1e-15) - np.log(S_hat + 1e-15)
            change    = np.abs(delta_new - md['delta_init']).max()
            if change > max_change:
                max_change = change
            md['delta_init'] = delta_new

        if max_change < tol:
            break

    for t, md in md_working.items():
        delta_dict[t] = md['delta_init']

    return delta_dict, n_iters


# =============================================================================
# TEST: contraction con theta=(0, -0.5, 0)
# =============================================================================

print("\n" + "=" * 60)
print("TEST contraction_mapping — theta=(0, -0.5, 0)")
print("=" * 60)

theta_test   = (0.0, -0.5, 0.0)
delta_conv, n_it = contraction_mapping(theta_test, market_data)
print(f"  Convergido en {n_it} iteraciones")

for t in sorted(delta_conv)[:3]:
    vals = delta_conv[t]
    parts = [f"delta[{k}]={vals[k]:.6f}" for k in range(min(3, len(vals)))]
    print(f"  Mercado {t}: " + "  ".join(parts))

# =============================================================================
# MICRO-MOMENTOS PREDICHOS
# =============================================================================

def compute_micro_moments_pred(theta, market_data, delta_dict=None):
    """
    Tras contraction_mapping, para cada escuela j:
      m_j_pred = sum_i(y_i * s_ij) / sum_i(s_ij)
      d_j_pred = sum_i(d_ij_raw * s_ij) / sum_i(s_ij)  [dist_km, no log]

    Returns (m_pred, d_pred): dicts {j_id -> float}.
    """
    if delta_dict is None:
        delta_dict, _ = contraction_mapping(theta, market_data)

    pi1, lam0, lam1 = theta
    m_pred = {}
    d_pred = {}

    for t, md in market_data.items():
        delta  = delta_dict[t]           # (J,)
        seg_z  = md['seg_z']             # (J,)
        log_d  = md['dist_log']          # (I, J)
        d_raw  = md['dist_km']           # (I, J)
        y_1d   = md['fam_y']             # (I,)
        j_ids  = md['J_ids']

        mu        = (pi1 * y_1d[:, None] * seg_z
                     + lam0 * log_d
                     + lam1 * y_1d[:, None] * log_d)
        exp_ij    = np.exp(np.clip(delta + mu, -500, 500))   # (I, J)
        s_ij      = exp_ij / (1.0 + exp_ij.sum(axis=1, keepdims=True))  # (I, J)
        sum_s     = s_ij.sum(axis=0) + 1e-15                # (J,)

        m_j = (y_1d[:, None] * s_ij).sum(axis=0) / sum_s   # (J,)
        d_j = (d_raw * s_ij).sum(axis=0) / sum_s            # (J,)

        for k, jid in enumerate(j_ids):
            m_pred[jid] = m_j[k]
            d_pred[jid] = d_j[k]

    return m_pred, d_pred


# =============================================================================
# MICRO-MOMENTOS OBSERVADOS
# =============================================================================

def compute_micro_moments_obs(market_data):
    """
    Para cada escuela j (sin depender de theta):
      m_j_obs = sum_i(y_i / d_ij_raw) / sum_i(1 / d_ij_raw)
      d_j_obs = sum_i(d_ij_raw * w_i) / sum_i(w_i)   [w_i = FEX_C o 1/I]

    Returns (m_obs, d_obs): dicts {j_id -> float}.
    """
    m_obs = {}
    d_obs = {}

    for t, md in market_data.items():
        d_raw = md['dist_km']    # (I, J)
        y_1d  = md['fam_y']     # (I,)
        j_ids = md['J_ids']
        I     = len(y_1d)

        w = md['fex_w'] if 'fex_w' in md else np.ones(I) / I   # (I,)

        d_safe  = d_raw + 1e-9   # evitar division por cero
        inv_d   = 1.0 / d_safe   # (I, J)

        m_num = (y_1d[:, None] * inv_d).sum(axis=0)   # (J,)
        m_den = inv_d.sum(axis=0) + 1e-15              # (J,)
        m_j   = m_num / m_den

        d_num = (d_raw * w[:, None]).sum(axis=0)       # (J,)
        d_den = w.sum() + 1e-15
        d_j   = d_num / d_den

        for k, jid in enumerate(j_ids):
            m_obs[jid] = m_j[k]
            d_obs[jid] = d_j[k]

    return m_obs, d_obs


# =============================================================================
# ESTADISTICOS DE MICRO-MOMENTOS OBSERVADOS
# =============================================================================

print("\n" + "=" * 60)
print("MICRO-MOMENTOS OBSERVADOS")
print("=" * 60)

m_obs_dict, d_obs_dict = compute_micro_moments_obs(market_data)

m_obs_arr = np.array(list(m_obs_dict.values()))
d_obs_arr = np.array(list(d_obs_dict.values()))

print(f"  m_obs (ingreso ponderado 1/d) — mean={m_obs_arr.mean():.4f}  std={m_obs_arr.std():.4f}"
      f"  [{m_obs_arr.min():.4f}, {m_obs_arr.max():.4f}]")
print(f"  d_obs (distancia ponderada FEX) — mean={d_obs_arr.mean():.4f}  std={d_obs_arr.std():.4f}"
      f"  [{d_obs_arr.min():.4f}, {d_obs_arr.max():.4f}]")

# =============================================================================
# PASO 7: CONSTRUCCION DE X_mat, Z_mat, Z_iv Y school_ids
# =============================================================================

print("\n" + "=" * 60)
print("CONSTRUYENDO X_mat, Z_mat, Z_iv")
print("=" * 60)

# Orden de escuelas: todas las que aparecen en market_data, mercado por mercado (sorted)
school_ids = []
for _t in sorted(market_data.keys()):
    school_ids.extend(market_data[_t]['J_ids'])

assert len(school_ids) == len(set(school_ids)), "school_ids contiene duplicados"
print(f"  Total escuelas en market_data: {len(school_ids):,}")

# Alinear df con school_ids para construir matrices
df_ord = df.set_index('id_establecimiento').loc[school_ids].reset_index()

# X_mat: constante + 7 variables de colegio (mismas que en 04a_berry_ols)
_X_cols = ['seguridad_percibida_z',
           'q_j_z', 'log_homicidios_z', 'log_dist_sitp_z', 'pct_no_oficial_z', 'es_tecnico']

X_mat = np.column_stack([np.ones(len(df_ord))] + [df_ord[c].values for c in _X_cols])

# Z_baseline: Z = X (no instruments)
Z_baseline = X_mat.copy()

# Z_iv: replace q_j_z with mean_q_rivals_z (1/d weighted, by localidad)
_Z_iv_cols = ['seguridad_percibida_z',
              'mean_q_rivals_z',
              'log_homicidios_z', 'log_dist_sitp_z', 'pct_no_oficial_z', 'es_tecnico']
Z_iv = np.column_stack([np.ones(len(df_ord))] + [df_ord[c].values for c in _Z_iv_cols])

# s_obs_all alineado con school_ids
s_obs_all = df.set_index('id_establecimiento').loc[school_ids, 's_j'].values

print(f"  X_mat shape: {X_mat.shape}  (J x {X_mat.shape[1]}: const + {X_mat.shape[1]-1} vars)")
print(f"  Z_baseline shape: {Z_baseline.shape}  (Z = X)")
print(f"  Z_iv shape: {Z_iv.shape}  (Z with BLP instruments)")

assert not np.isnan(X_mat).any(), "NaN en X_mat"
assert not np.isnan(Z_baseline).any(), "NaN en Z_baseline"
assert not np.isnan(Z_iv).any(), "NaN en Z_iv"

# Mapeo escuela -> (mercado, posicion_en_J_ids) para extraer delta eficientemente
_school_to_pos = {}
for _t in sorted(market_data.keys()):
    for _k, _jid in enumerate(market_data[_t]['J_ids']):
        _school_to_pos[_jid] = (_t, _k)

# =============================================================================
# PASO 8: FUNCION OBJETIVO GMM
# =============================================================================

_call_count = [0]


def gmm_objective(theta, market_data, m_obs_dict, d_obs_dict, X_mat, Z_mat, s_obs_all, school_ids):
    _call_count[0] += 1
    pi1, lam0, lam1 = theta
    delta_dict, n_it = contraction_mapping(theta, market_data)
    delta_vec = np.array([
        delta_dict[_school_to_pos[jid][0]][_school_to_pos[jid][1]]
        for jid in school_ids
    ])
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta    = XtX_inv @ (X_mat.T @ delta_vec)
    xi      = delta_vec - X_mat @ beta
    ZtZ_inv    = np.linalg.inv(Z_mat.T @ Z_mat)
    agg_moment = xi @ Z_mat @ ZtZ_inv @ Z_mat.T @ xi
    m_pred, d_pred = compute_micro_moments_pred(theta, market_data, delta_dict=delta_dict)
    common_ids   = [jid for jid in school_ids if jid in m_pred and jid in m_obs_dict]
    m_diff       = np.array([m_pred[j] - m_obs_dict[j] for j in common_ids])
    d_diff       = np.array([d_pred[j] - d_obs_dict[j] for j in common_ids])
    micro_moment = (m_diff ** 2).sum() + (d_diff ** 2).sum()
    total = agg_moment + micro_moment
    if _call_count[0] <= 5:
        print(f"  [eval {_call_count[0]:>3}] theta=({theta[0]:+.4f}, {theta[1]:+.4f}, "
              f"{theta[2]:+.4f})  "
              f"agg={agg_moment:.6f}  micro={micro_moment:.6f}  total={total:.6f}"
              f"  (CM iters={n_it})")
    return total


# =============================================================================
# PASO 9: FUNCION DE ESTIMACION REUTILIZABLE
# =============================================================================

def run_blp_estimation(Z_mat, label, market_data, m_obs_dict, d_obs_dict,
                       X_mat, s_obs_all, school_ids):
    """Run full BLP-GMM estimation with given Z_mat. Returns results dict."""
    global _call_count
    _call_count[0] = 0

    print("\n" + "=" * 60)
    print(f"ESTIMACION: {label}")
    print("=" * 60)

    _bounds = [(-3.0, 3.0), (-5.0, 5.0), (-3.0, 3.0)]
    _x0     = np.array([0.0, -0.5, 0.0])
    print(f"  x0     = {_x0.tolist()}")
    print(f"  Z shape: {Z_mat.shape}")

    result = optimize.minimize(
        gmm_objective, _x0,
        args=(market_data, m_obs_dict, d_obs_dict, X_mat, Z_mat, s_obs_all, school_ids),
        method='L-BFGS-B', bounds=_bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False},
    )

    theta_hat = tuple(result.x)
    pi1, lam0, lam1 = result.x

    print(f"\n  Convergencia: {result.success}")
    print(f"  Evaluaciones: {result.nfev}  |  Objetivo: {result.fun:.6f}")
    print(f"  theta: pi1={pi1:+.4f}  lam0={lam0:+.4f}  lam1={lam1:+.4f}")

    # Contraction final + OLS
    delta_dict, n_it = contraction_mapping(theta_hat, market_data)
    delta_vec = np.array([
        delta_dict[_school_to_pos[jid][0]][_school_to_pos[jid][1]]
        for jid in school_ids
    ])
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta    = XtX_inv @ (X_mat.T @ delta_vec)
    xi      = delta_vec - X_mat @ beta

    _beta_names = ['constante'] + _X_cols
    print(f"\n  {'Variable':<40} {'beta':>12}")
    print(f"  {'-'*40} {'-'*12}")
    for bname, bval in zip(_beta_names, beta):
        print(f"  {bname:<40} {bval:>+12.6f}")

    # Micro-momentos
    m_pred, d_pred = compute_micro_moments_pred(theta_hat, market_data)
    common = [j for j in school_ids if j in m_pred and j in m_obs_dict]
    m_err = np.array([m_pred[j] - m_obs_dict[j] for j in common])
    d_err = np.array([d_pred[j] - d_obs_dict[j] for j in common])
    print(f"\n  Micro-momentos: RMSE_m={np.sqrt((m_err**2).mean()):.4f}  RMSE_d={np.sqrt((d_err**2).mean()):.4f}")
    print("=" * 60)

    return {
        'label': label,
        'result': result,
        'theta': result.x,
        'beta': beta,
        'beta_names': _beta_names,
        'delta': delta_vec,
        'xi': xi,
        'rmse_m': np.sqrt((m_err**2).mean()),
        'rmse_d': np.sqrt((d_err**2).mean()),
    }


# =============================================================================
# PASO 10: DOS ESTIMACIONES — BASELINE vs IV-BLP
# =============================================================================

res_base = run_blp_estimation(
    Z_baseline, "BASELINE (Z=X)", market_data, m_obs_dict, d_obs_dict,
    X_mat, s_obs_all, school_ids)

res_iv = run_blp_estimation(
    Z_iv, "IV-BLP (BLP instruments)", market_data, m_obs_dict, d_obs_dict,
    X_mat, s_obs_all, school_ids)

# =============================================================================
# PASO 11: TABLA COMPARATIVA
# =============================================================================

print("\n" + "=" * 70)
print("COMPARACION: BASELINE vs IV-BLP")
print("=" * 70)

_theta_labels = ['pi1  (y_i x seg_z)',
                 'lam0 (log_d)', 'lam1 (y_i x log_d)']

print(f"  {'':40s} {'Baseline':>12} {'IV-BLP':>12}")
print(f"  {'-'*40} {'-'*12} {'-'*12}")
print(f"  {'--- Non-linear (theta) ---':40s}")
for i, lab in enumerate(_theta_labels):
    print(f"  {lab:40s} {res_base['theta'][i]:>+12.6f} {res_iv['theta'][i]:>+12.6f}")

print(f"\n  {'--- Linear (beta) ---':40s}")
for i, bname in enumerate(res_base['beta_names']):
    print(f"  {bname:40s} {res_base['beta'][i]:>+12.6f} {res_iv['beta'][i]:>+12.6f}")

print(f"\n  {'--- Diagnostics ---':40s}")
print(f"  {'GMM objective':40s} {res_base['result'].fun:>12.4f} {res_iv['result'].fun:>12.4f}")
print(f"  {'Convergence':40s} {str(res_base['result'].success):>12s} {str(res_iv['result'].success):>12s}")
print(f"  {'RMSE micro-m (income)':40s} {res_base['rmse_m']:>12.4f} {res_iv['rmse_m']:>12.4f}")
print(f"  {'RMSE micro-d (distance)':40s} {res_base['rmse_d']:>12.4f} {res_iv['rmse_d']:>12.4f}")
print(f"  {'First-stage F (q_j)':40s} {'---':>12s} {_fs_F:>12.2f}")
print("=" * 70)

# =============================================================================
# PASO 12: GUARDAR RESULTADOS (IV-BLP como preferido)
# =============================================================================

print("\nGuardando resultados...")

_rows = []
for spec, res in [('baseline', res_base), ('iv_blp', res_iv)]:
    for pname, pval in zip(['pi1', 'lam0', 'lam1'], res['theta']):
        _rows.append({'spec': spec, 'parametro': pname, 'estimacion': pval, 'tipo': 'no_lineal'})
    for bname, bval in zip(res['beta_names'], res['beta']):
        _rows.append({'spec': spec, 'parametro': f'beta_{bname}', 'estimacion': bval, 'tipo': 'lineal'})

_df_results = pd.DataFrame(_rows)
_results_path = OUT_TABLES / 'blp_results.csv'
_df_results.to_csv(_results_path, index=False)
print(f"  Guardado: {_results_path}")

# Use IV-BLP as preferred specification for downstream
_df_delta = pd.DataFrame({
    'id_establecimiento': school_ids,
    'delta_j_blp':        res_iv['delta'],
    'xi_j':               res_iv['xi'],
})
_delta_path = OUT_PRIMARY / 'blp_delta_j.parquet'
_df_delta.to_parquet(_delta_path, index=False)
print(f"  Guardado: {_delta_path}  (IV-BLP preferred)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
