"""
08c_robustez_gamma.py
=====================
Análisis de robustez del experimento sintético a diferentes valores de gamma_s.

El parámetro gamma_s = GAMMA_BASE / s controla qué tan fuerte es el sesgo visual
de cada estrato. GAMMA_BASE = 0.26 es el valor calibrado desde los datos reales.

Este script corre el experimento para seis valores de GAMMA_BASE:
    [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    γ₀=1 es el caso base de referencia (paso 0.25)

Para cada valor: corre los tres mecanismos (BM, DA, proxy-SED) con y sin sesgo,
calcula delta = corr(estrato, v_asignado) [bias] - corr(estrato, v_asignado) [true],
y guarda los resultados en una tabla comparativa.

Outputs:
    reports/robustez_gamma.csv     — tabla completa de resultados
    reports/figures/robustez_gamma.png — figura 3x1: delta por mecanismo vs gamma
"""

import json
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

os.chdir(Path(__file__).resolve().parent.parent)
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
COL_P   = ROOT / 'data' / 'primary' / 'colegios_features_imputed.geojson'
FAM_P   = ROOT / 'data' / 'processed' / 'familias_expandidas.parquet'
REP_DIR = ROOT / 'reports'
FIG_DIR = ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Parámetros fijos (iguales a 08b_datos_sinteticos.py) ─────────────────────
N_STUDENTS     = 10_000
M_SCHOOLS      = 100
RATIO_D_O      = 1.16
RHO            = 0.00
SIGMA          = 1.0
SEED           = 42
ALPHA_0        = 1.0
GAMMA_POW      = np.log(3) / np.log(17.5)  # ≈ 0.384 — log(3)/log(17.5) donde 17.5 = y_p90/y_p10 = 1400000/80000
CAPACIDAD_TOTAL = round(N_STUDENTS / RATIO_D_O)   # ≈ 862
N_REPS         = 1   # mundo fijo coincide con seed=42 de 09_matching_sinteticos.py

# ── Grid de robustez ──────────────────────────────────────────────────────────
GAMMA_BASE_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]   # γ₀=1 como caso base; paso 0.25

# ── 1. Cargar inputs reales ───────────────────────────────────────────────────
log.info('Cargando inputs reales...')
gdf   = gpd.read_file(COL_P)
q_all = gdf['q_j'].dropna().values.astype(float)
mu_q  = float(q_all.mean())
std_q = float(q_all.std())

fam_df = pd.read_parquet(FAM_P)
fam_df['sisben_cat'] = pd.to_numeric(fam_df['sisben_cat'], errors='coerce').astype('Int64')
fam_df['N_ingpc']    = pd.to_numeric(fam_df['N_ingpc'],    errors='coerce')

vc = fam_df['sisben_cat'].dropna().astype(int).value_counts()
strato_counts = {int(c): int(n) for c, n in vc.items() if int(c) in range(4)}
total_fam     = sum(strato_counts.values())
strato_dist   = {c: strato_counts[c] / total_fam for c in sorted(strato_counts)}

# Pool de ingresos reales por categoría SISBEN
ingpc_pool = {
    c: fam_df.loc[fam_df['sisben_cat'] == c, 'N_ingpc'].dropna().values
    for c in range(4)
}

# alpha: continuo por N_ingpc individual (igual que 06_preferencias.py)
# gamma: sigue indexado por sisben_cat (susceptibilidad al sesgo visual)
_estrato_proxy = {0: 1, 1: 2, 2: 3, 3: 5}  # para gamma únicamente

# y_bar: media del ingreso real de familias expandidas
y_bar = float(fam_df['N_ingpc'].mean())

log.info(f'  mu_q={mu_q:.4f}  std_q={std_q:.4f}')
log.info(f'  y_bar (ingreso medio real) = ${y_bar:,.0f}')

# ── 2. Funciones auxiliares ───────────────────────────────────────────────────

def generar_colegios(rng):
    q_j     = rng.normal(mu_q, std_q, M_SCHOOLS)
    q_j_std = (q_j - mu_q) / std_q
    eta     = rng.normal(0, 1, M_SCHOOLS)
    v_j     = RHO * q_j_std + np.sqrt(1 - RHO**2) * eta
    cap_base = rng.integers(15, 40, M_SCHOOLS).astype(float)
    cap_base = cap_base * CAPACIDAD_TOTAL / cap_base.sum()
    cap_base = np.clip(np.round(cap_base).astype(int), 5, None)
    diff = CAPACIDAD_TOTAL - cap_base.sum()
    if diff != 0:
        cap_base[np.argmax(cap_base)] += diff
    coord = rng.uniform(size=(M_SCHOOLS, 2))
    return pd.DataFrame({
        'id_establecimiento': [f'COL_{j:03d}' for j in range(M_SCHOOLS)],
        'q_j': q_j, 'q_j_std': q_j_std, 'v_j': v_j,
        'capacidad': cap_base, 'coord_x': coord[:, 0], 'coord_y': coord[:, 1],
    })


def generar_estudiantes(rng):
    cats  = sorted(strato_dist.keys())   # 0,1,2,3 (sisben_cat)
    probs = np.array([strato_dist[c] for c in cats])
    probs = probs / probs.sum()
    sisben = rng.choice(cats, size=N_STUDENTS, p=probs).astype(int)
    # Ingreso continuo sorteado del pool empírico por categoría
    ingpc = np.zeros(N_STUDENTS, dtype=float)
    for c in cats:
        mask = sisben == c
        pool = ingpc_pool[c]
        if len(pool) > 0:
            ingpc[mask] = rng.choice(pool, size=mask.sum(), replace=True)
    return sisben, ingpc


def compute_corr_v(asignados, v_by_id, estrato_arr):
    """Correlación Pearson entre estrato y v_j del colegio asignado."""
    assigned_mask = asignados >= 0
    if assigned_mask.sum() < 10:
        return np.nan
    v_assigned = np.array([v_by_id.get(str(a), np.nan) for a in asignados])
    mask = assigned_mask & ~np.isnan(v_assigned)
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(estrato_arr[mask], v_assigned[mask])[0, 1])


def boston_mechanism(pref_matrix, cap_arr):
    """BM simplificado: rondas con aceptación irrevocable."""
    N, M = pref_matrix.shape
    assignment = np.full(N, -1, dtype=int)
    remaining_cap = cap_arr.copy().astype(int)
    assigned = np.zeros(N, dtype=bool)
    for round_idx in range(M):
        proposals = {}
        for i in range(N):
            if assigned[i]: continue
            # Primera opción no intentada aún
            col = pref_matrix[i, round_idx]
            if col not in proposals:
                proposals[col] = []
            proposals[col].append(i)
        for col, applicants in proposals.items():
            cap = remaining_cap[col]
            if cap <= 0: continue
            accepted = applicants[:cap]
            for i in accepted:
                assignment[i] = col
                assigned[i] = True
            remaining_cap[col] -= len(accepted)
        if assigned.all():
            break
    return assignment


def da_mechanism(pref_matrix, cap_arr):
    """DA / Gale-Shapley (student-proposing) con prioridad por orden de llegada."""
    N, M = pref_matrix.shape
    assignment   = np.full(N, -1, dtype=int)
    next_prop    = np.zeros(N, dtype=int)
    school_queue = {j: [] for j in range(M)}
    cap          = cap_arr.copy().astype(int)
    free         = list(range(N))
    while free:
        next_free = []
        for i in free:
            if next_prop[i] >= M:
                continue
            j = pref_matrix[i, next_prop[i]]
            next_prop[i] += 1
            school_queue[j].append(i)
        for j in range(M):
            if not school_queue[j]: continue
            # Prioridad: orden de llegada (posición en lista de propuesta)
            accepted = school_queue[j][:cap[j]]
            rejected = school_queue[j][cap[j]:]
            assignment[accepted] = j
            next_free.extend(rejected)
            school_queue[j] = accepted
        free = [i for i in next_free if next_prop[i] < M]
    return assignment


CAT_OFFSET = 1_000_000  # igual que en 09_matching_sinteticos.py

def proxy_sed(pref_matrix, cap_arr, sisben_arr, lottery):
    """DA con prioridad lexicográfica: sisben_cat primero, lotería como desempate.
    Idéntico a priority_sed() de 09_matching_sinteticos.py."""
    N, M = pref_matrix.shape
    assignment   = np.full(N, -1, dtype=int)
    next_prop    = np.zeros(N, dtype=int)
    school_queue = {j: [] for j in range(M)}
    cap          = cap_arr.copy().astype(int)
    free         = list(range(N))

    while free:
        next_free = []
        for i in free:
            if next_prop[i] >= M: continue
            j = pref_matrix[i, next_prop[i]]
            next_prop[i] += 1
            school_queue[j].append(i)
        for j in range(M):
            if not school_queue[j]: continue
            q_sorted = sorted(school_queue[j],
                              key=lambda i: int(sisben_arr[i]) * CAT_OFFSET + lottery[j][i])
            accepted = q_sorted[:cap[j]]
            rejected = q_sorted[cap[j]:]
            assignment[accepted] = j
            next_free.extend(rejected)
            school_queue[j] = accepted
        free = [i for i in next_free if next_prop[i] < M]
    return assignment


# ── 3. Loop principal de robustez ─────────────────────────────────────────────
log.info('=' * 60)
log.info(f'Robustez gamma: {GAMMA_BASE_GRID} | {N_REPS} reps por valor')
log.info('=' * 60)

results = []

# Acumuladores indexados por (gamma_base, mecanismo)
rep_results = {
    (g, m): {'rank_bias': [], 'rank_true': [],
             'q_bias':    [], 'q_true':    [],
             'vj_bias':   [], 'vj_true':   []}
    for g in GAMMA_BASE_GRID
    for m in ['BM', 'DA', 'proxy-SED']
}

# Loop externo: réplica (mundo fijo)
for rep in range(N_REPS):
    rng = np.random.default_rng(SEED + rep * 100)

    colegios              = generar_colegios(rng)
    sisben_arr, ingpc_arr = generar_estudiantes(rng)
    estrato_arr           = sisben_arr

    school_ids = colegios['id_establecimiento'].values
    q_by_id    = colegios.set_index('id_establecimiento')['q_j_std'].to_dict()
    v_by_id    = colegios.set_index('id_establecimiento')['v_j'].to_dict()
    q_j_std    = colegios['q_j_std'].values
    v_j        = colegios['v_j'].values
    cap_arr    = colegios['capacidad'].values

    coord_est   = rng.uniform(size=(N_STUDENTS, 2))
    coord_col   = colegios[['coord_x', 'coord_y']].values
    dist_matrix = cdist(coord_est, coord_col).astype(np.float32)

    # Lotería de prioridad por colegio — igual que en 09_matching_sinteticos.py
    lottery = {}
    for j in range(M_SCHOOLS):
        perm = rng.permutation(N_STUDENTS)
        lottery[j] = {int(idx): int(rank) for rank, idx in enumerate(perm)}

    # alpha continuo por ingreso individual
    ingpc_safe = np.where(np.isnan(ingpc_arr) | (ingpc_arr <= 0), y_bar, ingpc_arr)
    alpha_i    = ALPHA_0 * (y_bar / ingpc_safe) ** GAMMA_POW
    dist_pen   = alpha_i[:, None] * np.log1p(dist_matrix)
    eps      = rng.gumbel(0, SIGMA, size=(N_STUDENTS, M_SCHOOLS))

    # Preferencias sin sesgo — iguales para todos los gamma de esta réplica
    U_true    = q_j_std[None, :] - dist_pen + eps
    pref_true = np.argsort(-U_true, axis=1)

    # Loop interno: solo cambia gamma
    for gamma_base in GAMMA_BASE_GRID:
        gamma_s = {c: gamma_base / _estrato_proxy[c] for c in range(4)}
        gamma_i = np.array([gamma_s[int(c)] for c in sisben_arr])

        U_bias    = q_j_std[None, :] - dist_pen + gamma_i[:, None] * v_j[None, :] + eps
        pref_bias = np.argsort(-U_bias, axis=1)

        for mec_name, pref_b, pref_t in [
            ('BM',        pref_bias, pref_true),
            ('DA',        pref_bias, pref_true),
            ('proxy-SED', pref_bias, pref_true),
        ]:
            if mec_name == 'BM':
                asgn_b = boston_mechanism(pref_b, cap_arr.copy())
                asgn_t = boston_mechanism(pref_t, cap_arr.copy())
            elif mec_name == 'DA':
                asgn_b = da_mechanism(pref_b, cap_arr.copy())
                asgn_t = da_mechanism(pref_t, cap_arr.copy())
            else:
                asgn_b = proxy_sed(pref_b, cap_arr.copy(), sisben_arr, lottery)
                asgn_t = proxy_sed(pref_t, cap_arr.copy(), sisben_arr, lottery)

            def mean_rank(asgn, pref):
                ranks = []
                for i, j in enumerate(asgn):
                    if j < 0: continue
                    row_pref = list(pref[i])
                    try:   ranks.append(row_pref.index(j) + 1)
                    except ValueError: pass
                return float(np.mean(ranks)) if ranks else np.nan

            rb = mean_rank(asgn_b, pref_b)
            rt = mean_rank(asgn_t, pref_t)
            rep_results[(gamma_base, mec_name)]['rank_bias'].append(rb)
            rep_results[(gamma_base, mec_name)]['rank_true'].append(rt)

            q_b = np.array([q_by_id.get(school_ids[a], np.nan) if a >= 0 else np.nan for a in asgn_b])
            q_t = np.array([q_by_id.get(school_ids[a], np.nan) if a >= 0 else np.nan for a in asgn_t])
            mask_low = (sisben_arr <= 1)
            rep_results[(gamma_base, mec_name)]['q_bias'].append(float(np.nanmean(q_b[mask_low])) if mask_low.sum() > 5 else np.nan)
            rep_results[(gamma_base, mec_name)]['q_true'].append(float(np.nanmean(q_t[mask_low])) if mask_low.sum() > 5 else np.nan)

            vb = np.array([v_by_id.get(school_ids[a], np.nan) if a >= 0 else np.nan for a in asgn_b])
            vt = np.array([v_by_id.get(school_ids[a], np.nan) if a >= 0 else np.nan for a in asgn_t])
            mask_b = ~np.isnan(vb) & ~np.isnan(ingpc_arr)
            mask_t = ~np.isnan(vt) & ~np.isnan(ingpc_arr)
            cb = float(np.corrcoef(ingpc_arr[mask_b], vb[mask_b])[0,1]) if mask_b.sum() > 10 else np.nan
            ct = float(np.corrcoef(ingpc_arr[mask_t], vt[mask_t])[0,1]) if mask_t.sum() > 10 else np.nan
            rep_results[(gamma_base, mec_name)]['vj_bias'].append(cb)
            rep_results[(gamma_base, mec_name)]['vj_true'].append(ct)

    log.info(f'  Réplica {rep+1}/{N_REPS} completada')

def safe_mean(lst): return round(float(np.mean([x for x in lst if not np.isnan(x)])), 5) if any(not np.isnan(x) for x in lst) else np.nan
def safe_std(lst):  return round(float(np.std( [x for x in lst if not np.isnan(x)])), 5) if any(not np.isnan(x) for x in lst) else np.nan

for gamma_base in GAMMA_BASE_GRID:
    log.info(f'\ngamma_base={gamma_base:.2f}')
    for mec_name in ['BM', 'DA', 'proxy-SED']:
        rr = rep_results[(gamma_base, mec_name)]
        rb = safe_mean(rr['rank_bias']); rt = safe_mean(rr['rank_true'])
        qb = safe_mean(rr['q_bias']);    qt = safe_mean(rr['q_true'])
        vb = safe_mean(rr['vj_bias']);   vt = safe_mean(rr['vj_true'])

        row = {
            'gamma_base':     gamma_base,
            'mecanismo':      mec_name,
            'delta_rank':     round(rb - rt, 5) if not (np.isnan(rb) or np.isnan(rt)) else np.nan,
            'delta_rank_std': safe_std([b - t for b, t in zip(rr['rank_bias'], rr['rank_true']) if not (np.isnan(b) or np.isnan(t))]),
            'delta_q':        round(qb - qt, 5) if not (np.isnan(qb) or np.isnan(qt)) else np.nan,
            'delta_q_std':    safe_std([b - t for b, t in zip(rr['q_bias'], rr['q_true']) if not (np.isnan(b) or np.isnan(t))]),
            'delta_vj':       round(vb - vt, 5) if not (np.isnan(vb) or np.isnan(vt)) else np.nan,
            'delta_vj_std':   safe_std([b - t for b, t in zip(rr['vj_bias'], rr['vj_true']) if not (np.isnan(b) or np.isnan(t))]),
            'n_reps': N_REPS,
        }
        results.append(row)
        log.info(f'  {mec_name:<12} Δrank={row["delta_rank"]:+.4f} | Δq(E1-E2)={row["delta_q"]:+.4f} | Δvj={row["delta_vj"]:+.4f}')

# ── 4. Guardar tabla ──────────────────────────────────────────────────────────
rob_df = pd.DataFrame(results)
rob_df.to_csv(REP_DIR / 'robustez_gamma.csv', index=False, encoding='utf-8')
log.info(f'\nGuardado: reports/robustez_gamma.csv ({len(rob_df)} filas)')

# ── 5. Figura — 3 paneles por métrica, 3 líneas por mecanismo ────────────────
# ── Estilo paper (idéntico a 09_matching_sinteticos.py) ──────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': 'white', 'figure.dpi': 150,
})

COLORS  = {'BM': '#2166ac', 'DA': '#d6604d', 'proxy-SED': '#4dac26'}
MARKERS = {'BM': 'o',       'DA': 's',        'proxy-SED': '^'}
LABELS  = {'BM': 'Boston (BM)', 'DA': 'Gale-Shapley (DA)', 'proxy-SED': 'proxy-SED Bogotá'}

METRICAS = [
    ('delta_rank', 'delta_rank_std',
     r'$\Delta$ rank medio ($\text{sesgo} - \text{sin sesgo}$)',
     r'(a) Eficiencia — $\Delta$ rank medio'),
    ('delta_q',    'delta_q_std',
     r'$\Delta\,\bar{q}_j\,(\text{SISBEN A+B})$ estandarizado',
     r'(b) Calidad asignada a grupos vulnerables'),
    ('delta_vj',   'delta_vj_std',
     r'$\Delta\,\rho(\text{ingreso},\, v_j^{\text{asignado}})$',
     r'(c) Sesgo visual — $\Delta\,\text{corr}(\text{ingreso},\,v_j)$'),
]

x_vals = sorted(rob_df['gamma_base'].unique())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

for ax, (col, col_std, ylabel, title) in zip(axes, METRICAS):
    for mec in ['BM', 'DA', 'proxy-SED']:
        sub = rob_df[rob_df['mecanismo'] == mec].sort_values('gamma_base')
        x  = sub['gamma_base'].values
        y  = sub[col].values
        ye = sub[col_std].values

        ax.plot(x, y, linestyle='-', marker=MARKERS[mec],
                color=COLORS[mec], linewidth=1.6, markersize=6,
                label=LABELS[mec], zorder=3)

    ax.axhline(0, color='#444', linewidth=0.8, linestyle='--', zorder=1)
    ax.axvline(1.0, color='#888', linewidth=0.8, linestyle=':', zorder=1)
    ax.text(1.01, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.001,
            r'$\gamma_0$', ha='left', va='bottom', fontsize=8.5,
            color='#666', style='italic')

    ax.set_xlabel(r'$\gamma_{\mathrm{base}}$', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10, loc='left', pad=6)
    ax.set_xticks(x_vals)
    ax.tick_params(labelsize=9)

fig.suptitle(
    r'Análisis de robustez: efecto del sesgo visual a distintos valores de $\gamma_s = \gamma_{\mathrm{base}}/s$',
    fontsize=10.5, y=1.06
)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03),
           ncol=3, fontsize=8.5, frameon=False)
fig.text(0.5, -0.04,
         fr'$N={N_STUDENTS:,}$ estudiantes $\times$ $M={M_SCHOOLS}$ colegios $\times$ {N_REPS} réplicas por $\gamma$. '
         r'Línea punteada vertical: $\gamma_0 = 1$ (caso base).',
         ha='center', fontsize=8.5, color='#444')

plt.tight_layout()
plt.savefig(FIG_DIR / 'robustez_gamma.png', dpi=150, bbox_inches='tight')
plt.close()
log.info('Guardado: reports/figures/robustez_gamma.png')

# ── 6. Resumen ────────────────────────────────────────────────────────────────
log.info('\n' + '=' * 60)
log.info('RESUMEN — Robustez a gamma')
log.info('=' * 60)
log.info(f'{"gamma":>6}  {"Mecanismo":>12}  {"Δrank":>8}  {"Δq(E1-2)":>10}  {"Δvj":>8}')
log.info('-' * 52)
for _, row in rob_df.iterrows():
    log.info(f'{row["gamma_base"]:>6.2f}  {row["mecanismo"]:>12}  '
             f'{row["delta_rank"]:>+8.4f}  {row["delta_q"]:>+10.4f}  {row["delta_vj"]:>+8.4f}')
