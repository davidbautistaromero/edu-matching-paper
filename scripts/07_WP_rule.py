"""
07_WP_rule.py
=============
Entrena la Weighted Polytope Rule (WP-Rule) vía StructSVM
(Narasimhan, Agarwal & Parkes 2016).

Diseño:
  - Genera mercados sintéticos Bogotá-like (ρ = ρ_Bogotá) para train/valid/test
  - Usa ρ=0 como test diagnóstico, no como distribución principal de entrenamiento
  - Calibra prioridad ingreso-distancia-visual y μ₁ para corrección visual
  - Entrena StructSVM → aprende W = (a, b, d_v)
  - Guarda pesos y parámetros en reports/wp_calibracion.json

Inputs:
  reports/tables/blp_results.csv
  data/primary/colegios_features_imputed.geojson
  data/images/clip/gsv_clip_establecimiento.parquet
  data/processed/familias_expandidas.parquet

Output:
  reports/wp_calibracion.json  — contiene w_learned, w_baseline, w_equal, μ₁*, params
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BLP_P       = ROOT / "reports" / "tables" / "blp_results.csv"
COLEGIOS_P  = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
CLIP_P      = ROOT / "data" / "images" / "clip" / "gsv_clip_establecimiento.parquet"
FAM_P       = ROOT / "data" / "processed" / "familias_expandidas.parquet"
OUT_REP     = ROOT / "reports"

# ── Constantes ───────────────────────────────────────────────────────────────
SISBEN_AB       = {0, 1}
SEED            = 42
N_STUDENTS      = 100
M_SCHOOLS       = 20
N_TRAIN         = 80
N_VALID         = 30
N_TEST          = 30

MU1_GRID        = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
THETA_GRID      = [-2.0, -1.0, -0.5, -0.2, -0.1, -0.05,
                   0.0,
                   0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
TARGET_V_CORR   = 0.0
V_BIAS_WEIGHT    = 1.0
MU_REG           = 1e-3
PRIO_INC_WEIGHT  = 1.0
PRIO_DIST_WEIGHT = 1.0

SVM_ITER        = 50
SVM_LR          = 0.1
SVM_REG         = 1e-3

FEATURE_NAMES = ["rank_pref", "priority", "inc_v"]
FEAT_SCALE = np.ones(len(FEATURE_NAMES))


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def income_distance_priority(inc, dist, v, theta):
    """Higher value means higher school priority.

    Low income and short distance raise priority. The theta term adjusts priority
    by the income-safety channel to target zero visual sorting.
    """
    inc_c = _zscore(inc)
    dist_score = -_zscore(dist)  # lower distance -> higher priority
    base = (PRIO_INC_WEIGHT * (-inc_c)[None, :]
            + PRIO_DIST_WEIGHT * dist_score.T)
    visual_adjustment = theta * (v[:, None] * inc_c[None, :])
    return base - visual_adjustment


def apply_income_distance_priority(mk, theta):
    mk["prio"] = income_distance_priority(mk["inc"], mk["dist"], mk["v"], theta)
    mk["theta_priority"] = float(theta)
    return mk


# =============================================================================
# Cargar parámetros
# =============================================================================

def cargar_parametros():
    blp = pd.read_csv(BLP_P)
    iv = blp[blp["spec"] == "iv_blp"].set_index("parametro")["estimacion"]
    pi1  = float(iv["pi1"])
    lam0 = float(iv["lam0"])
    lam1 = float(iv["lam1"])
    beta_const = float(iv.get("beta_constante", -2.741))
    beta_seg   = float(iv.get("beta_seguridad_percibida_z", 0.112))
    beta_q     = float(iv.get("beta_q_j_z", -0.184))

    log.info(f"  BLP θ: π₁={pi1:+.5f}, λ₀={lam0:+.5f}, λ₁={lam1:+.5f}")
    log.info(f"  BLP β: const={beta_const:.3f}, seg={beta_seg:+.4f}, q={beta_q:+.4f}")

    return dict(pi1=pi1, lam0=lam0, lam1=lam1,
                beta_const=beta_const, beta_seg=beta_seg, beta_q=beta_q)


def cargar_distribuciones_reales():
    import geopandas as gpd

    gdf = gpd.read_file(COLEGIOS_P)
    q_all = pd.to_numeric(gdf["q_j"], errors="coerce").dropna().values
    mu_q, std_q = float(q_all.mean()), float(q_all.std())

    clip = pd.read_parquet(CLIP_P)
    clip["id_establecimiento"] = clip["id_establecimiento"].astype(str).str.strip()
    gdf["id_establecimiento"] = gdf["id_establecimiento"].astype(str).str.strip()
    merged = gdf.merge(clip[["id_establecimiento", "seguridad_percibida"]],
                       on="id_establecimiento", how="inner")
    merged["q_j"] = pd.to_numeric(merged["q_j"], errors="coerce")
    merged = merged.dropna(subset=["q_j", "seguridad_percibida"])
    rho_real = float(np.corrcoef(merged["seguridad_percibida"], merged["q_j"])[0, 1])

    fam = pd.read_parquet(FAM_P)
    fam["N_ingpc"] = pd.to_numeric(fam["N_ingpc"], errors="coerce")
    fam["sisben_cat"] = pd.to_numeric(fam.get("sisben_cat"), errors="coerce").astype("Int64")

    ingpc_pool, sisben_counts = {}, {}
    for c in range(4):
        mask = fam["sisben_cat"] == c
        vals = fam.loc[mask, "N_ingpc"].dropna().values
        if len(vals) == 0:
            vals = fam["N_ingpc"].dropna().values
        ingpc_pool[c] = vals
        sisben_counts[c] = int(mask.sum())

    total = sum(sisben_counts.values())
    sisben_dist = {c: sisben_counts[c] / total for c in sorted(sisben_counts)}
    mean_ingpc = float(fam["N_ingpc"].mean())

    log.info(f"  q_j: μ={mu_q:.4f}, σ={std_q:.4f}")
    log.info(f"  ρ(seg_z, q_j) real = {rho_real:+.4f}")
    log.info(f"  Ingreso medio = ${mean_ingpc:,.0f}")
    for c, p in sisben_dist.items():
        log.info(f"  SISBEN {c}: {p:.1%} | pool n={len(ingpc_pool[c]):,}")

    return dict(mu_q=mu_q, std_q=std_q, rho_real=rho_real,
                ingpc_pool=ingpc_pool, sisben_dist=sisben_dist,
                mean_ingpc=mean_ingpc)


# =============================================================================
# Generador de mercados sintéticos
# =============================================================================

def generar_mercado(rng, params, distrib, n, m, rho):
    PI1, LAM0, LAM1 = params["pi1"], params["lam0"], params["lam1"]
    B_CONST, B_SEG, B_Q = params["beta_const"], params["beta_seg"], params["beta_q"]

    q_raw = rng.normal(distrib["mu_q"], distrib["std_q"], m)
    q_z = (q_raw - distrib["mu_q"]) / distrib["std_q"]

    eta = rng.standard_normal(m)
    v_z = rho * q_z + np.sqrt(max(1e-9, 1 - rho**2)) * eta
    v_z = (v_z - v_z.mean()) / (v_z.std() + 1e-9)

    xi_j = rng.normal(0, 0.3, m)
    delta_j = B_CONST + B_Q * q_z + B_SEG * v_z + xi_j

    ratio_do = 1.16
    cap_total = round(n / ratio_do)
    cap_base = rng.integers(max(1, cap_total // m - 5),
                            cap_total // m + 6, m).astype(float)
    cap_base = np.clip(np.round(cap_base * cap_total / cap_base.sum()).astype(int), 1, None)
    diff = cap_total - cap_base.sum()
    if diff != 0:
        cap_base[np.argmax(cap_base)] += diff

    coord_sch = rng.uniform(size=(m, 2))

    cats = sorted(distrib["sisben_dist"].keys())
    probs = np.array([distrib["sisben_dist"][c] for c in cats])
    probs = probs / probs.sum()
    sisben_arr = rng.choice(cats, size=n, p=probs)

    ingpc_arr = np.zeros(n)
    for c in cats:
        mask = sisben_arr == c
        pool = distrib["ingpc_pool"][c]
        ingpc_arr[mask] = rng.choice(pool, size=mask.sum(), replace=True)

    y_i = ingpc_arr / distrib["mean_ingpc"]
    coord_fam = rng.uniform(size=(n, 2))
    dist_matrix = cdist(coord_fam, coord_sch).astype(np.float32)
    dist_log = np.log1p(dist_matrix)

    eps = rng.gumbel(0, 1, size=(n, m))
    U = (delta_j[None, :]
         + PI1 * y_i[:, None] * v_z[None, :]
         + LAM0 * dist_log
         + LAM1 * y_i[:, None] * dist_log
         + eps)

    pref_rank = (-U).argsort(axis=1).argsort(axis=1) + 1

    estrato = np.searchsorted(np.quantile(ingpc_arr, [1/6, 2/6, 3/6, 4/6, 5/6]),
                              ingpc_arr) + 1

    mk = dict(
        n=n, m=m,
        q=q_z, v=v_z, delta_j=delta_j, cap=cap_base, coord_sch=coord_sch,
        inc=ingpc_arr, y_i=y_i, sisben=sisben_arr, estrato=estrato,
        coord_fam=coord_fam,
        pref_rank=pref_rank, dist=dist_matrix,
        rho_effective=float(np.corrcoef(q_z, v_z)[0, 1]),
    )
    return apply_income_distance_priority(mk, theta=0.0)


# =============================================================================
# Restricciones del politopo
# =============================================================================

def _stab_constraints(mk):
    """Rothblum (1992) stability constraints for many-to-one matching.
    For each (i, j):
      cap_j * sum_{j': i prefers j' >= j} x_{ij'}
      + sum_{i': j prioritizes i' > i} x_{i'j} >= cap_j
    Rewritten as: -LHS <= -cap_j  (for A_ub form).
    """
    n, m = mk["n"], mk["m"]
    pr, prio = mk["pref_rank"], mk["prio"]
    R, b = [], []
    for i in range(n):
        for j in range(m):
            cap_j = float(mk["cap"][j])
            row = np.zeros(n * m)
            for jj in np.where(pr[i] <= pr[i, j])[0]:
                row[i * m + jj] += cap_j
            for ii in np.where(prio[j] > prio[j, i])[0]:
                row[ii * m + j] += 1.0
            R.append(-row)
            b.append(-cap_j)
    return np.array(R), np.array(b)


def _assign_constraints(mk):
    n, m = mk["n"], mk["m"]
    R, b = [], []
    for i in range(n):
        r = np.zeros(n * m)
        r[i * m:(i + 1) * m] = 1.0
        R.append(r)
        b.append(1.0)
    for j in range(m):
        r = np.zeros(n * m)
        r[j::m] = 1.0
        R.append(r)
        b.append(float(mk["cap"][j]))
    return np.array(R), np.array(b)


def _access_constraints(mk):
    n, m = mk["n"], mk["m"]
    R, b = [], []
    for i in range(n):
        if int(mk["sisben"][i]) in SISBEN_AB:
            r = np.zeros(n * m)
            r[i * m:(i + 1) * m] = 1.0
            R.append(r)
            b.append(1.0)
    if not R:
        return np.zeros((0, n * m)), np.zeros(0)
    return np.array(R), np.array(b)


def build_cache(mk):
    return {
        "st": _stab_constraints(mk),
        "as": _assign_constraints(mk),
        "acc": _access_constraints(mk),
    }


def stable_argmax(mk, weights, cache, hard_access=False):
    A_ub = np.vstack([cache["as"][0], cache["st"][0]])
    b_ub = np.concatenate([cache["as"][1], cache["st"][1]])

    if hard_access:
        Aeq, beq = cache["acc"]
        if Aeq.shape[0] > 0:
            A_ub = np.vstack([A_ub, -Aeq])
            b_ub = np.concatenate([b_ub, -beq])

    res = linprog(c=-weights.ravel(),
                  A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, 1)] * (mk["n"] * mk["m"]),
                  integrality=np.ones(mk["n"] * mk["m"], dtype=int),
                  method="highs")

    if not res.success:
        return None

    x = np.round(res.x).reshape(mk["n"], mk["m"])
    return [int(np.argmax(x[i])) if x[i].max() > 0.5 else None
            for i in range(mk["n"])]


# =============================================================================
# WP-Rule — pesos, features, target
# =============================================================================

N_FEATS = len(FEATURE_NAMES)


def _inc_c(mk):
    return (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)


def _feature_matrices(mk):
    inc_c = _inc_c(mk)
    return [
        -mk["pref_rank"].astype(float),
        mk["prio"].T.astype(float),
        np.outer(inc_c, mk["v"]),
    ]


def _std_matrix(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _normalize_w(w):
    nrm = np.linalg.norm(w)
    return w / nrm if nrm > 1e-9 else w

def wp_weights(mk, w):
    """WP score: rank, income-distance priority, and visual-bias correction."""
    mats = _feature_matrices(mk)
    out = np.zeros((mk["n"], mk["m"]))
    for k, mat in enumerate(mats):
        out += w[k] * mat / FEAT_SCALE[k]
    return out


def wp_feats(mk, cols):
    mats = _feature_matrices(mk)
    vals = []
    for k, mat in enumerate(mats):
        vals.append(sum(mat[i, cols[i]] / FEAT_SCALE[k]
                        for i in range(mk["n"])
                        if cols[i] is not None))
    return np.array(vals, dtype=float)


def wp_target(mk, mu1, cache):
    """Income-distance WP target LP with visual-bias correction."""
    rank_term, prio_term, yv_term = [
        _std_matrix(x) for x in _feature_matrices(mk)
    ]
    w = rank_term + prio_term - mu1 * yv_term

    # Restricciones: asignación y cupos. No imponemos acceso duro SED.
    A_ub = cache["as"][0]
    b_ub = cache["as"][1]

    res = linprog(c=-w.ravel(),
                  A_ub=A_ub, b_ub=b_ub,
                  bounds=(0, 1), method="highs")
    if not res.success:
        return None
    x = res.x.reshape(mk["n"], mk["m"])
    return [int(np.argmax(x[i])) if x[i].max() > 0.5 else None
            for i in range(mk["n"])]


def matching_metrics(mk, cols):
    matched = np.array([c is not None for c in cols])
    cc = np.array([c if c is not None else 0 for c in cols])

    def corr(x, y):
        if matched.sum() < 3:
            return np.nan
        return float(np.corrcoef(x[matched], y[matched])[0, 1])

    rank = np.array([
        mk["pref_rank"][i, cc[i]] if matched[i] else mk["m"] + 1
        for i in range(mk["n"])
    ])

    return dict(
        sesgo_inc_v=corr(mk["inc"], mk["v"][cc]),
        corr_inc_q=corr(mk["inc"], mk["q"][cc]),
        q_medio=float(mk["q"][cc][matched].mean()) if matched.any() else np.nan,
        rank_medio=float(rank.mean()),
        asignados=float(matched.mean()),
    )


def summarize_matchings(markets, cols_list):
    rows = [matching_metrics(mk, cols) for mk, cols in zip(markets, cols_list)
            if cols is not None]
    if not rows:
        return {}
    return {
        k: float(np.nanmean([r[k] for r in rows]))
        for k in rows[0]
    }


def make_markets(rng, params, distrib, n_markets, rho, label):
    markets, caches = [], []
    for k in range(n_markets):
        mk = generar_mercado(rng, params, distrib, N_STUDENTS, M_SCHOOLS, rho=rho)
        markets.append(mk)
        caches.append(build_cache(mk))
        if (k + 1) % 20 == 0:
            log.info(f"  {label}: {k+1}/{n_markets} mercados generados")
    return markets, caches


def rebuild_caches(markets, theta):
    for mk in markets:
        apply_income_distance_priority(mk, theta)
    return [build_cache(mk) for mk in markets]


def calibrar_theta(markets):
    best = dict(theta=0.0, score=np.inf, summary={})
    candidates = []
    w_probe = _normalize_w(np.array([1.0, 1.0, 0.0]))
    for theta in THETA_GRID:
        caches = rebuild_caches(markets, theta)
        cols = [stable_argmax(mk, wp_weights(mk, w_probe), c, hard_access=False)
                for mk, c in zip(markets, caches)]
        summary = summarize_matchings(markets, cols)
        if not summary:
            continue
        score = abs(summary["sesgo_inc_v"] - TARGET_V_CORR)
        cand = dict(theta=theta, score=score, summary=summary)
        candidates.append(cand)
        if score < best["score"]:
            best = cand
        log.info(
            f"  theta={theta:>5.2f} -> sesgo={summary['sesgo_inc_v']:+.4f} "
            f"rank={summary['rank_medio']:.2f}"
        )
    for cand in sorted(candidates, key=lambda x: x["score"])[:5]:
        s = cand["summary"]
        log.info(
            f"    cand theta={cand['theta']:>5.2f} score={cand['score']:.4f} "
            f"sesgo={s['sesgo_inc_v']:+.4f} rank={s['rank_medio']:.2f}"
        )
    return best


def calibrar_mu(markets, caches):
    best = dict(mu1=0.0, score=np.inf, summary={})
    candidates = []
    for mu1 in MU1_GRID:
        targets = [wp_target(mk, mu1, c)
                   for mk, c in zip(markets, caches)]
        summary = summarize_matchings(markets, targets)
        if not summary:
            continue
        sesgo_v = summary["sesgo_inc_v"]
        score = V_BIAS_WEIGHT * abs(sesgo_v - TARGET_V_CORR) + MU_REG * mu1
        candidates.append(dict(mu1=mu1, score=score, summary=summary))
        if score < best["score"]:
            best = dict(mu1=mu1, score=score, summary=summary)
        log.info(f"  μ₁={mu1:>5.1f} → sesgo_v={sesgo_v:+.4f} score={score:.4f}")
    for cand in sorted(candidates, key=lambda x: x["score"])[:5]:
        s = cand["summary"]
        log.info(
            f"    cand μ₁={cand['mu1']:>5.1f} "
            f"score={cand['score']:.4f} "
            f"sesgo_v={s['sesgo_inc_v']:+.4f}"
        )
    return best


# =============================================================================
# StructSVM
# =============================================================================

def train_svm(markets, targets, caches, n_iter=SVM_ITER, lr=SVM_LR, reg=SVM_REG,
              early_tol=1e-5, early_patience=5, active_mask=None, initial_w=None):
    if active_mask is None:
        active_mask = np.ones(N_FEATS)
    else:
        active_mask = np.array(active_mask, dtype=float)

    if initial_w is None:
        w = np.array([1.0, 1.0, 0.0], dtype=float)
    else:
        w = np.array(initial_w, dtype=float)
    w = _normalize_w(w * active_mask)
    stale = 0
    w_prev = w.copy()

    for it in range(n_iter):
        grad = np.zeros(N_FEATS)
        for mk, yh, c in zip(markets, targets, caches):
            if yh is None:
                continue
            aug = wp_weights(mk, w).astype(float)
            for i in range(mk["n"]):
                if yh[i] is not None:
                    aug[i, yh[i]] -= 1.0

            ystar = stable_argmax(mk, aug, c, hard_access=False)
            if ystar is None:
                continue
            grad += wp_feats(mk, ystar) - wp_feats(mk, yh)

        grad = (grad / len(markets) + reg * w) * active_mask
        w = w - lr * grad
        w = _normalize_w(w * active_mask)

        if (it + 1) % 10 == 0:
            log.info(f"  iter {it+1}/{n_iter}: w={np.round(w, 4)}")

        # Early stopping
        if np.linalg.norm(w - w_prev) < early_tol:
            stale += 1
            if stale >= early_patience:
                log.info(f"  Early stop at iter {it+1} (no change for {early_patience} iters)")
                break
        else:
            stale = 0
        w_prev = w.copy()

    return w


# =============================================================================
# Main
# =============================================================================

def main():
    global FEAT_SCALE

    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ── Paso 1 ───────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("PASO 1 — Cargando parámetros estimados")
    log.info("=" * 70)
    params = cargar_parametros()
    distrib = cargar_distribuciones_reales()
    rho_real = distrib["rho_real"]

    # ── Paso 2 ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("PASO 2 — Generando splits sintéticos calibrados a Bogotá")
    log.info("=" * 70)

    train_markets, train_caches = make_markets(
        rng, params, distrib, N_TRAIN, rho_real, "train Bogotá-like")
    valid_markets, valid_caches = make_markets(
        rng, params, distrib, N_VALID, rho_real, "valid Bogotá-like")
    test_markets, test_caches = make_markets(
        rng, params, distrib, N_TEST, rho_real, "test Bogotá-like")
    diag_markets, diag_caches = make_markets(
        rng, params, distrib, N_TEST, 0.0, "test diagnóstico ρ=0")

    # Diagnóstico: correlaciones en las preferencias (antes de asignar)
    _diag_yv, _diag_yq = [], []
    for mk in train_markets:
        # Para cada familia, ponderar v_j y q_j por la utilidad implícita (softmax)
        # Proxy simple: correlación ingreso con v/q del colegio preferido (#1)
        top_j = mk["pref_rank"].argmin(axis=1)  # colegio más preferido
        _diag_yv.append(float(np.corrcoef(mk["inc"], mk["v"][top_j])[0, 1]))
        _diag_yq.append(float(np.corrcoef(mk["inc"], mk["q"][top_j])[0, 1]))
    log.info(f"  DIAGNÓSTICO preferencias (top-1):")
    log.info(f"    corr(ingreso, v_top1) = {np.mean(_diag_yv):+.4f}")
    log.info(f"    corr(ingreso, q_top1) = {np.mean(_diag_yq):+.4f}")

    # ── Paso 2b ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("PASO 2b — Calibrando prioridad ingreso-distancia-visual")
    log.info("=" * 70)

    best_theta = calibrar_theta(valid_markets)
    THETA = best_theta["theta"]
    log.info(
        f"  → theta*={THETA:+.2f} "
        f"(sesgo_v={best_theta['summary']['sesgo_inc_v']:+.4f})"
    )
    train_caches = rebuild_caches(train_markets, THETA)
    valid_caches = rebuild_caches(valid_markets, THETA)
    test_caches = rebuild_caches(test_markets, THETA)
    diag_caches = rebuild_caches(diag_markets, THETA)

    # ── Paso 2c ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("PASO 2c — Calibrando μ₁ en validación Bogotá-like")
    log.info("=" * 70)

    best_mu = calibrar_mu(valid_markets, valid_caches)
    MU1 = best_mu["mu1"]
    log.info(
        f"  → μ₁*={MU1} "
        f"(sesgo_v={best_mu['summary']['sesgo_inc_v']:+.4f})"
    )

    log.info("  Calculando targets finales...")
    train_targets = [wp_target(mk, MU1, c)
                     for mk, c in zip(train_markets, train_caches)]

    FEAT_SCALE = np.ones(N_FEATS)
    FEAT_SCALE = np.array([
        np.std([wp_feats(mk, list(yh))[k]
                for mk, yh in zip(train_markets, train_targets)])
        for k in range(N_FEATS)
    ]) + 1e-9
    log.info(f"  FEAT_SCALE = {np.round(FEAT_SCALE, 2)}")

    # ── Paso 3 ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("PASO 3 — Entrenando WP-Rule (StructSVM)")
    log.info("=" * 70)

    w_learned = train_svm(train_markets, train_targets, train_caches)
    log.info(f"  w* learned {FEATURE_NAMES} = {np.round(w_learned, 4)}")

    # Baseline aprendido: usa preferencias y prioridades, sin corrección visual.
    baseline_mask = np.array([1, 1, 0], dtype=float)
    w_baseline = train_svm(
        train_markets, train_targets, train_caches,
        active_mask=baseline_mask,
        initial_w=np.array([1.0, 1.0, 0.0]),
    )
    log.info(f"  w* baseline (sin d_v) = {np.round(w_baseline, 4)}")

    # Baseline sin entrenar: equal weights sobre preferencia y prioridad.
    w_equal = _normalize_w(np.array([1.0, 1.0, 0.0]))
    log.info(f"  w  equal   (a,b,0) = {np.round(w_equal, 4)}")

    log.info("")
    log.info("  Validación WP sobre mercados Bogotá-like:")
    for name, w in [("WP_learned", w_learned), ("WP_baseline", w_baseline), ("WP_equal", w_equal)]:
        cols = [stable_argmax(mk, wp_weights(mk, w), c, hard_access=False)
                for mk, c in zip(valid_markets, valid_caches)]
        summary = summarize_matchings(valid_markets, cols)
        log.info(
            f"    {name:10} sesgo={summary.get('sesgo_inc_v', np.nan):+.4f} "
            f"corr_yq={summary.get('corr_inc_q', np.nan):+.4f} "
            f"q={summary.get('q_medio', np.nan):+.4f} "
            f"rank={summary.get('rank_medio', np.nan):.2f}"
        )

    log.info("")
    log.info("  Tests preparados (se ejecutan al correr el script):")
    for label, mks, cas in [
        ("Bogotá-like", test_markets, test_caches),
        ("Diagnóstico ρ=0", diag_markets, diag_caches),
    ]:
        cols = [stable_argmax(mk, wp_weights(mk, w_learned), c, hard_access=False)
                for mk, c in zip(mks, cas)]
        summary = summarize_matchings(mks, cols)
        log.info(
            f"    {label:15} WP_learned sesgo={summary.get('sesgo_inc_v', np.nan):+.4f} "
            f"corr_yq={summary.get('corr_inc_q', np.nan):+.4f}"
        )

    # ── Guardar ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("Guardando wp_calibracion.json")
    log.info("=" * 70)

    cal = {
        "n_students": N_STUDENTS,
        "m_schools": M_SCHOOLS,
        "n_train": N_TRAIN,
        "n_valid": N_VALID,
        "n_test": N_TEST,
        "rho_train": round(rho_real, 5),
        "rho_valid": round(rho_real, 5),
        "rho_test": round(rho_real, 5),
        "rho_diagnostic": 0.0,
        "rho_real": round(rho_real, 5),
        "mu1": MU1,
        "theta_priority": THETA,
        "theta_grid": THETA_GRID,
        "priority_model": "income_distance_visual",
        "priority_income_weight": PRIO_INC_WEIGHT,
        "priority_distance_weight": PRIO_DIST_WEIGHT,
        "target_v_corr": TARGET_V_CORR,
        "v_bias_weight": V_BIAS_WEIGHT,
        "mu_reg": MU_REG,
        "svm_iter": SVM_ITER,
        "svm_lr": SVM_LR,
        "svm_reg": SVM_REG,
        "feature_names": FEATURE_NAMES,
        "feat_scale": [round(float(x), 5) for x in FEAT_SCALE],
        "seed": SEED,
        "betas_blp": {k: round(v, 6) for k, v in params.items()},
        "distribuciones": {
            "mu_q": round(distrib["mu_q"], 5),
            "std_q": round(distrib["std_q"], 5),
            "rho_real": round(rho_real, 5),
            "mean_ingpc": round(distrib["mean_ingpc"], 2),
            "sisben_dist": {str(k): round(v, 5) for k, v in distrib["sisben_dist"].items()},
        },
        "w_learned": [round(float(x), 5) for x in w_learned],
        "w_baseline": [round(float(x), 5) for x in w_baseline],
        "w_equal": [round(float(x), 5) for x in w_equal],
    }
    out_path = OUT_REP / "wp_calibracion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)
    log.info(f"  {out_path}")
    log.info(f"\n[tiempo total: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()


