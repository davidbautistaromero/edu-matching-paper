"""
08_simulacion_mecanismos.py
===========================
Evaluacion de mecanismos de asignacion escolar sobre mercados sinteticos.

Carga los pesos WP entrenados por 07_WP_rule.py y:
  1. Barrido de rho = {0, 0.2, 0.4, 0.6, 0.8, 1.0}
     -> Como escala el sesgo visual con la correlacion v-q
  2. Comparacion final en rho = rho_real aprox. 0.115
     -> BM, DA, SED-lex, WP_learned

Inputs:
  reports/wp_calibracion.json  (de 07_WP_rule.py)
  reports/tables/blp_results.csv
  data/primary/colegios_features_imputed.geojson
  data/images/clip/gsv_clip_establecimiento.parquet
  data/processed/familias_expandidas.parquet

Outputs:
  reports/tables/wp_rho_sweep.csv
  reports/tables/wp_rule_results.csv
  reports/figures/matching/wp_rho_sweep.png
  reports/figures/matching/wp_comparacion_final.png
"""

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import matching_utils as mu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- Paths --------------------------------------------------------------------
CAL_P       = ROOT / "reports" / "wp_calibracion.json"
BLP_P       = ROOT / "reports" / "tables" / "blp_results.csv"
COLEGIOS_P  = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
CLIP_P      = ROOT / "data" / "images" / "clip" / "gsv_clip_establecimiento.parquet"
FAM_P       = ROOT / "data" / "processed" / "familias_expandidas.parquet"
OUT_TABLES  = ROOT / "reports" / "tables"
OUT_FIGS    = ROOT / "reports" / "figures" / "matching"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# -- Constantes ---------------------------------------------------------------
SISBEN_AB   = {0, 1}
SEED        = 123       # distinto al de entrenamiento
N_EVAL      = 60
RHO_SWEEP   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PRIO_INC_WEIGHT  = 1.0
PRIO_DIST_WEIGHT = 1.0


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def income_distance_priority(inc, dist, v, theta):
    inc_c = _zscore(inc)
    dist_score = -_zscore(dist)
    base = (PRIO_INC_WEIGHT * (-inc_c)[None, :]
            + PRIO_DIST_WEIGHT * dist_score.T)
    visual_adjustment = theta * (v[:, None] * inc_c[None, :])
    return base - visual_adjustment


def apply_income_distance_priority(mk, theta):
    mk["prio"] = income_distance_priority(mk["inc"], mk["dist"], mk["v"], theta)
    mk["theta_priority"] = float(theta)
    return mk


# =============================================================================
# Reutilizar funciones de 07 (generador, restricciones, LP)
# =============================================================================
# Importamos las funciones compartidas directamente.
# En produccion, esto iria en un modulo compartido.
# Por ahora, copiamos las esenciales.

def cargar_parametros():
    blp = pd.read_csv(BLP_P)
    iv = blp[blp["spec"] == "iv_blp"].set_index("parametro")["estimacion"]
    return dict(
        pi1=float(iv["pi1"]), lam0=float(iv["lam0"]), lam1=float(iv["lam1"]),
        beta_const=float(iv.get("beta_constante", -2.741)),
        beta_seg=float(iv.get("beta_seguridad_percibida_z", 0.112)),
        beta_q=float(iv.get("beta_q_j_z", -0.184)),
    )


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

    return dict(mu_q=mu_q, std_q=std_q, rho_real=rho_real,
                ingpc_pool=ingpc_pool, sisben_dist=sisben_dist,
                mean_ingpc=mean_ingpc)


def generar_mercado(rng, params, distrib, n, m, rho, theta_priority=0.0):
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

    mk = dict(
        n=n, m=m,
        q=q_z, v=v_z, delta_j=delta_j, cap=cap_base,
        inc=ingpc_arr, y_i=y_i, sisben=sisben_arr,
        pref_rank=pref_rank, dist=dist_matrix,
        rho_effective=float(np.corrcoef(q_z, v_z)[0, 1]),
    )
    return apply_income_distance_priority(mk, theta_priority)


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
            # All schools j' that i ranks at least as good as j
            for jj in np.where(pr[i] <= pr[i, j])[0]:
                row[i * m + jj] += cap_j
            # All students i' that j strictly prioritizes over i
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


def wp_weights(mk, w, feat_scale):
    a, b, dv = w
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    return (a * (-mk["pref_rank"]) / feat_scale[0]
            + b * mk["prio"].T / feat_scale[1]
            + dv * np.outer(inc_c, mk["v"]) / feat_scale[2])


# =============================================================================
# Metricas
# =============================================================================

def compute_metrics(mk, cols):
    matched = np.array([c is not None for c in cols])
    cc = np.array([c if c is not None else 0 for c in cols])

    def corr(x, y):
        m = matched
        if m.sum() < 3:
            return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    rank = np.array([
        mk["pref_rank"][i, cc[i]] if matched[i] else mk["m"] + 1
        for i in range(mk["n"])
    ])

    ab_mask = np.array([int(s) in SISBEN_AB for s in mk["sisben"]])
    acc_ab = float(matched[ab_mask].mean()) if ab_mask.any() else np.nan

    return dict(
        sesgo_inc_v=corr(mk["inc"], mk["v"][cc]),
        equidad_inc_q=corr(mk["inc"], mk["q"][cc]),
        q_medio=float(mk["q"][cc][matched].mean()) if matched.any() else np.nan,
        rank_medio=float(rank.mean()),
        acc_ab=acc_ab,
    )


def priority_distance_fn(mk):
    """Priority convention for matching_utils: lower value means higher priority."""
    return lambda i, sid, _mk=mk: float(_mk["dist"][i, int(sid[1:])])


def priority_id_fn(mk):
    """Income-distance priority. mk['prio'] is higher-is-better; utils expects lower-is-better."""
    return lambda i, sid, _mk=mk: -float(_mk["prio"][int(sid[1:]), i])


def priority_sed_lex_fn(mk):
    """SED-like lexicographic priority: lower SISBEN category, then shorter distance."""
    return lambda i, sid, _mk=mk: (
        float(_mk["sisben"][i]) * 1e6 + float(_mk["dist"][i, int(sid[1:])])
    )


def run_mecanismos(mk):
    n, m = mk["n"], mk["m"]
    pl = [[f"C{j}" for j in np.argsort(mk["pref_rank"][i])] for i in range(n)]
    sc = {f"C{j}": int(mk["cap"][j]) for j in range(m)}
    pf = priority_distance_fn(mk)
    ps = priority_sed_lex_fn(mk)
    return {
        "BM":  mu.boston_mechanism(pl, sc, pf),
        "DA":  mu.deferred_acceptance(pl, sc, pf),
        "SED-lex": mu.deferred_acceptance(pl, sc, ps),
    }


def cols_from_matching(matching):
    return [int(s[1:]) if s is not None else None for s in matching]


# =============================================================================
# Evaluacion de un conjunto de mercados
# =============================================================================

def evaluar(rng, params, distrib, rho, n_eval, w_learned, w_baseline, w_equal, feat_scale,
            theta_priority=0.0,
            only_wp=False):
    """Evalua mecanismos en n_eval mercados con correlacion rho.
    Si only_wp=True, solo corre variantes WP (para barrido de rho)."""
    n_s = params.get("_n_students", 100)
    m_s = params.get("_m_schools", 20)

    if only_wp:
        mecanismos = ["WP_learned", "WP_baseline", "WP_equal"]
    else:
        mecanismos = ["BM", "DA", "SED-lex", "WP_learned"]

    keys = ["sesgo_inc_v", "equidad_inc_q", "q_medio", "rank_medio", "acc_ab", "bp"]
    resultados = {mec: {k: [] for k in keys} for mec in mecanismos}

    for k in range(n_eval):
        mk = generar_mercado(rng, params, distrib, n_s, m_s,
                             rho=rho, theta_priority=theta_priority)
        cache = build_cache(mk)

        pl = [[f"C{j}" for j in np.argsort(mk["pref_rank"][i])] for i in range(mk["n"])]
        sc = {f"C{j}": int(mk["cap"][j]) for j in range(mk["m"])}
        pf = priority_distance_fn(mk)
        pi = priority_id_fn(mk)
        ps = priority_sed_lex_fn(mk)

        if not only_wp:
            res_clasicos = run_mecanismos(mk)
            for mec_name, matching in res_clasicos.items():
                cols = cols_from_matching(matching)
                m_dict = compute_metrics(mk, cols)
                prio_fn = ps if mec_name == "SED-lex" else pf
                bp = mu.count_blocking_pairs(matching, pl, sc, prio_fn)
                for kk in m_dict:
                    resultados[mec_name][kk].append(m_dict[kk])
                resultados[mec_name]["bp"].append(bp)

        wp_variants = {"WP_learned": w_learned, "WP_baseline": w_baseline, "WP_equal": w_equal}
        for wp_name, w in wp_variants.items():
            if wp_name not in mecanismos:
                continue
            cols = stable_argmax(mk, wp_weights(mk, w, feat_scale), cache, hard_access=False)
            if cols is None:
                continue
            m_dict = compute_metrics(mk, cols)
            matching_str = [f"C{c}" if c is not None else None for c in cols]
            bp = mu.count_blocking_pairs(matching_str, pl, sc, pi)
            for kk in m_dict:
                resultados[wp_name][kk].append(m_dict[kk])
            resultados[wp_name]["bp"].append(bp)

    return mecanismos, resultados


def ci(vals):
    v = np.array(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), 1.96 * v.std() / np.sqrt(len(v))


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # -- Cargar calibracion -----------------------------------------------
    log.info("=" * 70)
    log.info("Cargando wp_calibracion.json")
    log.info("=" * 70)

    with open(CAL_P, "r", encoding="utf-8") as f:
        cal = json.load(f)

    w_learned = np.array(cal["w_learned"])
    w_baseline = np.array(cal.get("w_baseline", cal.get("w_paper", np.zeros(3))))
    w_equal = np.array(cal["w_equal"])
    feat_scale = np.array(cal["feat_scale"])
    rho_real = cal["rho_real"]
    if cal.get("priority_model") != "income_distance_visual" or "theta_priority" not in cal:
        raise ValueError(
            "wp_calibracion.json no corresponde a la nueva WP ingreso-distancia. "
            "Corre primero scripts/07_WP_rule.py para calibrar theta_priority."
        )
    theta_priority = float(cal["theta_priority"])

    if any(len(x) != 3 for x in [w_learned, w_baseline, w_equal, feat_scale]):
        raise ValueError(
            "wp_calibracion.json usa una especificacion WP antigua. "
            "Corre primero scripts/07_WP_rule.py para generar pesos de 3 features."
        )

    log.info(f"  w_learned = {np.round(w_learned, 4)}")
    log.info(f"  w_baseline = {np.round(w_baseline, 4)}")
    log.info(f"  w_equal   = {np.round(w_equal, 4)}")
    log.info(f"  feat_scale = {np.round(feat_scale, 2)}")
    log.info(f"  rho_real = {rho_real:.4f}")
    log.info(f"  theta_priority = {theta_priority:+.2f}")

    # -- Cargar datos -----------------------------------------------------
    params = cargar_parametros()
    distrib = cargar_distribuciones_reales()
    params["_n_students"] = cal["n_students"]
    params["_m_schools"] = cal["m_schools"]

    # -- Paso 1: Barrido de rho -------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info(f"PASO 1 -- Barrido de rho: {RHO_SWEEP}")
    log.info("=" * 70)

    sweep_rows = []
    for rho_test in RHO_SWEEP:
        log.info(f"\n  rho = {rho_test:.2f}")
        mecanismos, resultados = evaluar(
            rng, params, distrib, rho_test, N_EVAL,
            w_learned, w_baseline, w_equal, feat_scale,
            theta_priority=theta_priority, only_wp=True)

        for mec in mecanismos:
            r = resultados[mec]
            if not r["q_medio"]:
                continue
            row = dict(
                rho=rho_test,
                mecanismo=mec,
                sesgo_inc_v=round(float(np.nanmean(r["sesgo_inc_v"])), 5),
                equidad_inc_q=round(float(np.nanmean(r["equidad_inc_q"])), 5),
                q_medio=round(float(np.nanmean(r["q_medio"])), 5),
                rank_medio=round(float(np.nanmean(r["rank_medio"])), 4),
                acc_ab=round(float(np.nanmean(r["acc_ab"])), 5),
                bp_mean=round(float(np.nanmean(r["bp"])), 2),
            )
            sweep_rows.append(row)
            log.info(f"    {mec:12} sesgo={row['sesgo_inc_v']:+.4f}  "
                     f"equid={row['equidad_inc_q']:+.4f}  "
                     f"q={row['q_medio']:+.4f}  "
                     f"rank={row['rank_medio']:.2f}  "
                     f"BP={row['bp_mean']:.1f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_TABLES / "wp_rho_sweep.csv", index=False, encoding="utf-8-sig")
    log.info(f"\n  Saved: {OUT_TABLES / 'wp_rho_sweep.csv'}")

    # -- Figura: barrido de ? ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mec in ["WP_learned", "WP_baseline", "WP_equal"]:
        sub = sweep_df[sweep_df["mecanismo"] == mec]
        axes[0].plot(sub["rho"], sub["sesgo_inc_v"], "o-", label=mec)
        axes[1].plot(sub["rho"], sub["equidad_inc_q"], "o-", label=mec)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("rho(v, q)")
    axes[0].set_ylabel("corr(ingreso, v_asignado)")
    axes[0].set_title("Sesgo visual vs correlacion")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("rho(v, q)")
    axes[1].set_ylabel("corr(ingreso, q_asignado)")
    axes[1].set_title("Equidad en calidad vs correlacion")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "wp_rho_sweep.png", dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {OUT_FIGS / 'wp_rho_sweep.png'}")

    # -- Paso 2: Comparacion final en rho_real ----------------------------
    log.info("")
    log.info("=" * 70)
    log.info(f"PASO 2 -- Comparacion final (rho={rho_real:.3f}, {N_EVAL} mercados)")
    log.info("=" * 70)

    mecanismos, resultados = evaluar(
        rng, params, distrib, rho_real, N_EVAL,
        w_learned, w_baseline, w_equal, feat_scale,
        theta_priority=theta_priority)

    header = f"{'mec':12}{'sesgo(y,v)':>14}{'equid(y,q)':>14}{'q_med':>9}{'rank':>8}{'acc_AB':>9}{'BP':>7}"
    log.info(header)
    log.info("-" * len(header))

    rows = []
    for mec in mecanismos:
        r = resultados[mec]
        if not r["q_medio"]:
            continue
        sv_m, sv_ci = ci(r["sesgo_inc_v"])
        eq_m, eq_ci = ci(r["equidad_inc_q"])
        q_m = np.nanmean(r["q_medio"])
        rk_m, rk_ci = ci(r["rank_medio"])
        ac_m, ac_ci = ci(r["acc_ab"])
        bp_m = np.nanmean(r["bp"])

        log.info(f"{mec:12}{sv_m:>+8.3f}+/-{sv_ci:.3f}"
                 f"{eq_m:>+8.3f}+/-{eq_ci:.3f}"
                 f"{q_m:>+9.3f}{rk_m:>8.2f}{ac_m:>9.3f}{bp_m:>7.1f}")

        rows.append(dict(
            mecanismo=mec,
            sesgo_inc_v_mean=round(sv_m, 5), sesgo_inc_v_ci=round(sv_ci, 5),
            equidad_inc_q_mean=round(eq_m, 5), equidad_inc_q_ci=round(eq_ci, 5),
            q_medio=round(q_m, 5),
            rank_medio_mean=round(rk_m, 4), rank_medio_ci=round(rk_ci, 4),
            acc_ab_mean=round(ac_m, 5),
            bp_mean=round(bp_m, 2),
        ))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_TABLES / "wp_rule_results.csv", index=False, encoding="utf-8-sig")
    log.info(f"\n  Saved: {OUT_TABLES / 'wp_rule_results.csv'}")

    log.info(f"\n[tiempo total: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()

