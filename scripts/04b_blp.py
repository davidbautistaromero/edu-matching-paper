#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_blp.py
==========
BLP con micro-momentos para demanda escolar en Bogota.
Estima interacciones ingreso x senales visuales (CLIP) e ingreso x distancia.

Ref: reports/paper/blp_contexto_paper.md

Modelo:
  u_ij = delta_j + pi1*y_i*seg_j + pi2*y_i*veg_j + lam0*d_ij + lam1*y_i*d_ij + eps_ij

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

CLIP_VARS = ['seguridad_percibida', 'vegetacion_percibida']
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
df.loc[df['caracter_media'].isin(['Sin informacion', 'Sin información']), 'es_tecnico'] = np.nan

df = df.dropna(subset=CLIP_VARS + CTRL_CONT + CTRL_BIN).reset_index(drop=True)
print(f"  Colegios tras dropna: {len(df):,}")
print(f"  Mercados (localidades): {df['codigo_localidad'].nunique()}")

# Estandarizar continuas (z-score en muestra de colegios)
for col in CLIP_VARS + ['q_j', 'log_homicidios', 'log_dist_sitp', 'pct_no_oficial']:
    mu, sd = df[col].mean(), df[col].std()
    df[f'{col}_z'] = (df[col] - mu) / sd

CLIP_Z = [f'{c}_z' for c in CLIP_VARS]
CTRL_Z = [f'{c}_z' for c in ['q_j', 'log_homicidios', 'log_dist_sitp', 'pct_no_oficial']] + CTRL_BIN

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
    veg_z_arr = df_t_al['vegetacion_percibida_z'].values
    s_obs_arr = df_t_al['s_j'].values
    delta_arr = df_t_al['delta_j'].values

    fam_y_arr = fam_t['y_i'].values
    fex_w_arr = fam_t['FEX_C'].values

    market_data[t] = dict(
        J_ids      = j_ids,
        seg_z      = seg_z_arr,
        veg_z      = veg_z_arr,
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

    theta = (pi1, pi2, lam0, lam1)
      pi1  : y_i * seg_z_j
      pi2  : y_i * veg_z_j
      lam0 : log(1 + d_ij)
      lam1 : y_i * log(1 + d_ij)

    Returns dict {market -> S_hat array shape (J,)}
    """
    pi1, pi2, lam0, lam1 = theta
    shares = {}
    for t, md in market_data.items():
        delta  = md['delta_init']          # (J,)
        seg_z  = md['seg_z']               # (J,)
        veg_z  = md['veg_z']               # (J,)
        log_d  = md['dist_log']            # (I, J)
        y_i    = md['fam_y'][:, None]      # (I, 1)

        # mu_ij: (I, J)
        mu = pi1 * y_i * seg_z + pi2 * y_i * veg_z + lam0 * log_d + lam1 * y_i * log_d

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
# TEST: contraction con theta=(0, 0, -0.5, 0)
# =============================================================================

print("\n" + "=" * 60)
print("TEST contraction_mapping — theta=(0, 0, -0.5, 0)")
print("=" * 60)

theta_test   = (0.0, 0.0, -0.5, 0.0)
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

    pi1, pi2, lam0, lam1 = theta
    m_pred = {}
    d_pred = {}

    for t, md in market_data.items():
        delta  = delta_dict[t]           # (J,)
        seg_z  = md['seg_z']             # (J,)
        veg_z  = md['veg_z']             # (J,)
        log_d  = md['dist_log']          # (I, J)
        d_raw  = md['dist_km']           # (I, J)
        y_1d   = md['fam_y']             # (I,)
        j_ids  = md['J_ids']

        mu        = (pi1 * y_1d[:, None] * seg_z
                     + pi2 * y_1d[:, None] * veg_z
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
# PASO 7: CONSTRUCCION DE X_mat, Z_mat Y school_ids
# =============================================================================

print("\n" + "=" * 60)
print("CONSTRUYENDO X_mat y Z_mat")
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
_X_cols = ['seguridad_percibida_z', 'vegetacion_percibida_z',
           'q_j_z', 'log_homicidios_z', 'log_dist_sitp_z', 'pct_no_oficial_z', 'es_tecnico']

X_mat = np.column_stack([np.ones(len(df_ord))] + [df_ord[c].values for c in _X_cols])
Z_mat = X_mat.copy()   # Z = X (instrumentos iguales a regresores)

# s_obs_all alineado con school_ids
s_obs_all = df.set_index('id_establecimiento').loc[school_ids, 's_j'].values

print(f"  X_mat shape: {X_mat.shape}  (J x 8: const + 7 vars)")
print(f"  Z_mat shape: {Z_mat.shape}  (Z = X)")

assert not np.isnan(X_mat).any(), "NaN en X_mat"
assert not np.isnan(Z_mat).any(), "NaN en Z_mat"

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
    """
    Objetivo GMM-BLP combinando momento agregado (Berry 1994) y micro-momentos.

    1. Contraction mapping -> delta_j
    2. OLS: delta = X @ beta + xi
    3. Momento agregado: xi' Z (Z'Z)^{-1} Z' xi
    4. Micro-momentos: sum((m_pred-m_obs)^2) + sum((d_pred-d_obs)^2)
    """
    _call_count[0] += 1

    # --- Contraction mapping: delta por mercado ---
    delta_dict, n_it = contraction_mapping(theta, market_data)

    # --- Apilar delta en el orden de school_ids ---
    delta_vec = np.array([
        delta_dict[_school_to_pos[jid][0]][_school_to_pos[jid][1]]
        for jid in school_ids
    ])

    # --- OLS: delta = X @ beta + xi ---
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta    = XtX_inv @ (X_mat.T @ delta_vec)
    xi      = delta_vec - X_mat @ beta

    # --- Momento agregado (forma cuadratica GMM con W = (Z'Z)^{-1}) ---
    ZtZ_inv    = np.linalg.inv(Z_mat.T @ Z_mat)
    agg_moment = xi @ Z_mat @ ZtZ_inv @ Z_mat.T @ xi

    # --- Micro-momentos (reusar delta_dict del CM) ---
    m_pred, d_pred = compute_micro_moments_pred(theta, market_data, delta_dict=delta_dict)

    common_ids   = [jid for jid in school_ids if jid in m_pred and jid in m_obs_dict]
    m_diff       = np.array([m_pred[j] - m_obs_dict[j] for j in common_ids])
    d_diff       = np.array([d_pred[j] - d_obs_dict[j] for j in common_ids])
    micro_moment = (m_diff ** 2).sum() + (d_diff ** 2).sum()

    total = agg_moment + micro_moment

    if _call_count[0] <= 5:
        print(f"  [eval {_call_count[0]:>3}] theta=({theta[0]:+.4f}, {theta[1]:+.4f}, "
              f"{theta[2]:+.4f}, {theta[3]:+.4f})  "
              f"agg={agg_moment:.6f}  micro={micro_moment:.6f}  total={total:.6f}"
              f"  (CM iters={n_it})")

    return total


# =============================================================================
# PASO 9: OPTIMIZACION L-BFGS-B
# =============================================================================

print("\n" + "=" * 60)
print("OPTIMIZACION GMM — L-BFGS-B")
print("=" * 60)

_call_count[0] = 0   # resetear antes de la optimizacion

_bounds = [(-3.0, 3.0), (-3.0, 3.0), (-5.0, 5.0), (-3.0, 3.0)]  # pi1, pi2, lam0, lam1
_x0     = np.array([0.0, 0.0, -0.5, 0.0])

print(f"  x0     = {_x0.tolist()}")
print(f"  bounds = {_bounds}")
print(f"\n  Primeras 5 evaluaciones de la funcion objetivo:")

_result = optimize.minimize(
    gmm_objective,
    _x0,
    args=(market_data, m_obs_dict, d_obs_dict, X_mat, Z_mat, s_obs_all, school_ids),
    method='L-BFGS-B',
    bounds=_bounds,
    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False},
)

print("\n" + "=" * 60)
print("RESULTADO OPTIMIZACION GMM-BLP")
print("=" * 60)
print(f"  Convergencia: {_result.success}  ({_result.message})")
print(f"  Iteraciones:  {_result.nit}")
print(f"  Evaluaciones: {_result.nfev}")
print(f"  Objetivo min: {_result.fun:.6f}")
print(f"\n  theta_hat:")
_theta_names = [
    'pi1  (y_i * seg_z)',
    'pi2  (y_i * veg_z)',
    'lam0 (log_d)',
    'lam1 (y_i * log_d)',
]
for _name, _val in zip(_theta_names, _result.x):
    print(f"    {_name:<30}: {_val:+.6f}")
print("=" * 60)

# =============================================================================
# PASO 10: EXTRAER PARAMETROS Y TABLA RESUMEN
# =============================================================================

pi1, pi2, lam0, lam1 = _result.x

print("\n" + "=" * 60)
print("PARAMETROS NO-LINEALES ESTIMADOS")
print("=" * 60)
print(f"  {'Parámetro':<35} {'Estimación':>12}")
print(f"  {'-'*35} {'-'*12}")
print(f"  {'pi1  (ingreso × seguridad_z)':<35} {pi1:>+12.6f}")
print(f"  {'pi2  (ingreso × vegetacion_z)':<35} {pi2:>+12.6f}")
print(f"  {'lam0 (log_distancia)':<35} {lam0:>+12.6f}")
print(f"  {'lam1 (ingreso × log_distancia)':<35} {lam1:>+12.6f}")
print("=" * 60)

# =============================================================================
# PASO 11: CONTRACTION FINAL + OLS -> BETA LINEAL
# =============================================================================

print("\nEjecutando contraction mapping final con theta_hat...")

_theta_hat = tuple(_result.x)
_delta_final, _n_it_final = contraction_mapping(_theta_hat, market_data)
print(f"  Convergido en {_n_it_final} iteraciones")

# Apilar delta en orden de school_ids
_delta_vec_final = np.array([
    _delta_final[_school_to_pos[jid][0]][_school_to_pos[jid][1]]
    for jid in school_ids
])

# OLS: delta = X @ beta + xi
_XtX_inv_final = np.linalg.inv(X_mat.T @ X_mat)
_beta_final    = _XtX_inv_final @ (X_mat.T @ _delta_vec_final)
_xi_final      = _delta_vec_final - X_mat @ _beta_final

_beta_names = ['constante'] + _X_cols

print("\n" + "=" * 60)
print("PARAMETROS LINEALES (beta) — OLS sobre delta_j final")
print("=" * 60)
print(f"  {'Variable':<40} {'beta':>12}")
print(f"  {'-'*40} {'-'*12}")
for _bname, _bval in zip(_beta_names, _beta_final):
    print(f"  {_bname:<40} {_bval:>+12.6f}")
print("=" * 60)

# =============================================================================
# PASO 12: MICRO-MOMENTOS FINALES — COMPARACION PRED vs OBS
# =============================================================================

print("\nCalculando micro-momentos finales...")

_m_pred_final, _d_pred_final = compute_micro_moments_pred(_theta_hat, market_data)

_common = [j for j in school_ids if j in _m_pred_final and j in m_obs_dict]
_m_err  = np.array([_m_pred_final[j] - m_obs_dict[j]  for j in _common])
_d_err  = np.array([_d_pred_final[j] - d_obs_dict[j]  for j in _common])

print("\n" + "=" * 60)
print("MICRO-MOMENTOS: PRED vs OBS")
print("=" * 60)
print(f"  Escuelas evaluadas: {len(_common):,}")
print(f"\n  Ingreso ponderado (m_j):")
print(f"    Error medio (pred - obs):  {_m_err.mean():+.6f}")
print(f"    MAE:                       {np.abs(_m_err).mean():.6f}")
print(f"    RMSE:                      {np.sqrt((_m_err**2).mean()):.6f}")
print(f"\n  Distancia ponderada (d_j):")
print(f"    Error medio (pred - obs):  {_d_err.mean():+.6f}")
print(f"    MAE:                       {np.abs(_d_err).mean():.6f}")
print(f"    RMSE:                      {np.sqrt((_d_err**2).mean()):.6f}")
print("=" * 60)

# =============================================================================
# PASO 13: GUARDAR RESULTADOS
# =============================================================================

print("\nGuardando resultados...")

# --- blp_results.csv ---
_rows = []
for _pname, _pval in zip(['pi1', 'pi2', 'lam0', 'lam1'], [pi1, pi2, lam0, lam1]):
    _rows.append({'parametro': _pname, 'estimacion': _pval, 'tipo': 'no_lineal'})
for _bname, _bval in zip(_beta_names, _beta_final):
    _rows.append({'parametro': f'beta_{_bname}', 'estimacion': _bval, 'tipo': 'lineal'})

_df_results = pd.DataFrame(_rows)
_results_path = OUT_TABLES / 'blp_results.csv'
_df_results.to_csv(_results_path, index=False)
print(f"  Guardado: {_results_path}")

# --- blp_delta_j.parquet ---
_df_delta = pd.DataFrame({
    'id_establecimiento': school_ids,
    'delta_j_blp':        _delta_vec_final,
    'xi_j':               _xi_final,
})
_delta_path = OUT_PRIMARY / 'blp_delta_j.parquet'
_df_delta.to_parquet(_delta_path, index=False)
print(f"  Guardado: {_delta_path}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RESUMEN FINAL BLP")
print("=" * 60)
print(f"  Convergencia GMM:          {_result.success}")
print(f"  Objetivo GMM final:        {_result.fun:.6f}")
print(f"  Evaluaciones función obj:  {_result.nfev}")
print(f"  theta_hat: pi1={pi1:+.4f}  pi2={pi2:+.4f}  lam0={lam0:+.4f}  lam1={lam1:+.4f}")
print(f"  delta_j — mean={_delta_vec_final.mean():.4f}  std={_delta_vec_final.std():.4f}"
      f"  [{_delta_vec_final.min():.4f}, {_delta_vec_final.max():.4f}]")
print(f"  xi_j    — mean={_xi_final.mean():.4f}  std={_xi_final.std():.4f}"
      f"  [{_xi_final.min():.4f}, {_xi_final.max():.4f}]")
print(f"  Error medio m_j:           {_m_err.mean():+.6f}")
print(f"  Error medio d_j:           {_d_err.mean():+.6f}")
print(f"  Archivos guardados:")
print(f"    {_results_path}")
print(f"    {_delta_path}")
print("=" * 60)
