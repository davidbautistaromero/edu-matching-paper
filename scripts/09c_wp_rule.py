# =============================================================================
#  09c_wp_rule.py  —  Mecanismo aprendido (WP-Rule) vía StructSVM
#  Cuarto mecanismo del paper, comparado contra BM / DA / SED sobre el
#  mundo sintético. Reutiliza matching_utils.py (BM, DA, blocking pairs).
#
#  Teoría: el conjunto de matchings estables es un politopo íntegro
#  (Vande Vate 1989; Rothblum 1992; Roth, Rothblum & Vande Vate 1993; caso
#  con capacidades = politopo de b-matching estable). Maximizar una función
#  lineal sobre él entrega un vértice => matching estable. Los pesos de esa
#  función se aprenden (Narasimhan, Agarwal & Parkes 2016, marco estilo SVM).
#
#  Especificación de pesos (extiende la del md con una feature de equidad):
#      lambda_ij(w) = a·rank_familia(j) + b·rank_colegio(i)
#                     + d·(ingreso_c_i · v_j)
#  El tercer término es necesario: sin él los pesos sólo ven rangos de
#  preferencia y son ciegos al sesgo visual (WP colapsa exactamente a DA).
# =============================================================================
import numpy as np, os, sys
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# --- importar las funciones de matching de David ----------------------------
sys.path.insert(0, "/content/edu-matching-paper/scripts")
sys.path.insert(0, ".")
import matching_utils as mu

ALPHA_HAT, GAMMA0 = 0.08793, 1.0
DANE = (227220.0, 460198.0, 897987.0)
FEAT_SCALE = np.array([1.0, 1.0, 1.0])

def sisben_group(inc):
    return 1 if inc < DANE[0] else 2 if inc < DANE[1] else 3 if inc < DANE[2] else 4

# --- calibración desde los datos reales de David (con fallback) -------------
def calibrar():
    base = "/content/edu-matching-paper/data/primary"
    try:
        import pandas as pd
        col = pd.read_parquet(f"{base}/sinteticos_b_colegios.parquet")
        est = pd.read_parquet(f"{base}/sinteticos_b_estudiantes.parquet")
        rho = float(np.corrcoef(col["q_j_std"], col["v_j"])[0, 1])
        inc_pool = est["N_ingpc"].to_numpy(float)
        inc_pool = inc_pool[inc_pool > 0]
        print(f"calibrado desde datos: rho(q,v)={rho:+.3f} | "
              f"ingreso n={len(inc_pool)} mediana={np.median(inc_pool):,.0f}")
        return rho, (lambda n, r: r.choice(inc_pool, n, replace=True))
    except Exception as e:
        print(f"[fallback, sin datos: {e}]")
        return 0.40, (lambda n, r: r.lognormal(np.log(380000), 0.9, n))

# --- generador de mercado pequeño (fiel a md M2) ----------------------------
def generar_mercado(n, m, rho, rng, inc_sampler):
    inc = inc_sampler(n, rng)
    s = np.array([sisben_group(x) for x in inc])
    estrato = np.searchsorted(np.quantile(inc, [1/6,2/6,3/6,4/6,5/6]), inc) + 1
    q = rng.standard_normal(m); z = rng.standard_normal(m)
    v = rho * q + np.sqrt(max(1e-9, 1 - rho**2)) * z
    v = (v - v.mean()) / (v.std() + 1e-9); q = (q - q.mean()) / (q.std() + 1e-9)
    gum = rng.gumbel(0, 1, size=(n, m))
    U = q[None, :] + (ALPHA_HAT + GAMMA0 / s)[:, None] * v[None, :] + gum
    pref_rank = (-U).argsort(1).argsort(1) + 1
    prio = np.vstack([rng.permutation(n).argsort() + 1 for _ in range(m)])
    cap = np.full(m, int(np.ceil(n / m)) + 1)
    return dict(n=n, m=m, inc=inc, s=s, estrato=estrato, q=q, v=v,
                pref_rank=pref_rank, prio=prio, cap=cap)

# --- restricciones del politopo estable -------------------------------------
def _stab(mk):
    n, m, pr, prio, cap = mk["n"], mk["m"], mk["pref_rank"], mk["prio"], mk["cap"]
    R, b = [], []
    for i in range(n):
        for j in range(m):
            row = np.zeros(n * m)
            for jj in np.where(pr[i] <= pr[i, j])[0]: row[i*m+jj] += cap[j]
            for ii in np.where(prio[j] < prio[j, i])[0]: row[ii*m+j] += 1.0
            R.append(-row); b.append(-float(cap[j]))
    return np.array(R), np.array(b)

def _assign(mk):
    n, m = mk["n"], mk["m"]; R, b = [], []
    for i in range(n):
        r = np.zeros(n*m); r[i*m:(i+1)*m] = 1.0; R.append(r); b.append(1.0)
    for j in range(m):
        r = np.zeros(n*m); r[j::m] = 1.0; R.append(r); b.append(float(mk["cap"][j]))
    return np.array(R), np.array(b)

def stable_argmax(mk, weights, cache):
    """ILP binario -> matching estable de máximo peso (columnas enteras)."""
    A = np.vstack([cache["as"][0], cache["st"][0]])
    b = np.concatenate([cache["as"][1], cache["st"][1]])
    res = milp(c=-weights.ravel(), constraints=LinearConstraint(A, -np.inf, b),
               integrality=np.ones(mk["n"]*mk["m"]), bounds=Bounds(0, 1))
    x = np.round(res.x).reshape(mk["n"], mk["m"])
    return [int(np.argmax(x[i])) if x[i].max() > 0.5 else None for i in range(mk["n"])]

def wp_weights(mk, w):
    a, b, d = w; s0, s1, s2 = FEAT_SCALE
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    return (a*mk["pref_rank"]/s0 + b*mk["prio"].T/s1
            + d*np.outer(inc_c, mk["v"])/s2)

def feats(mk, cols):
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    return np.array([sum(mk["pref_rank"][i, cols[i]] for i in range(mk["n"])),
                     sum(mk["prio"][cols[i], i] for i in range(mk["n"])),
                     sum(inc_c[i]*mk["v"][cols[i]] for i in range(mk["n"]))], float)

# --- target ideal (transporte: calidad - mu·ingreso·v) ----------------------
def target(mk, mu_pen, cache):
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    w = mk["q"][None, :] - mu_pen * np.outer(inc_c, mk["v"])
    res = linprog(c=-w.ravel(), A_ub=cache["as"][0], b_ub=cache["as"][1],
                  bounds=(0, 1), method="highs")
    return res.x.reshape(mk["n"], mk["m"]).argmax(1)

# --- StructSVM (subgradiente) -----------------------------------------------
def train(markets, targets, caches, n_iter=40, lr=0.1, reg=1e-3):
    w = np.array([-1.0, 0.0, 0.0])
    for _ in range(n_iter):
        g = np.zeros(3)
        for mk, yh, c in zip(markets, targets, caches):
            aug = wp_weights(mk, w).astype(float)
            for i in range(mk["n"]): aug[i, yh[i]] -= 1.0   # inferencia aumentada
            ystar = stable_argmax(mk, aug, c)
            g += feats(mk, ystar) - feats(mk, yh)
        g = g/len(markets) + reg*w
        w = w - lr*g; nrm = np.linalg.norm(w)
        if nrm > 1e-9: w /= nrm
    return w

# --- métrica: sesgo, calidad, rank ------------------------------------------
def metrics(mk, cols):
    matched = np.array([c is not None for c in cols])
    cc = np.array([c if c is not None else 0 for c in cols])
    def corr(x, y): return float(np.corrcoef(x[matched], y[matched])[0,1]) if matched.sum()>2 else np.nan
    rank = np.array([mk["pref_rank"][i, cc[i]] if matched[i] else mk["m"]+1 for i in range(mk["n"])])
    return dict(sesgo_inc_v=corr(mk["inc"], mk["v"][cc]),
                sesgo_estrato_v=corr(mk["estrato"].astype(float), mk["v"][cc]),
                q_medio=float(mk["q"][cc][matched].mean()), rank_medio=float(rank.mean()))

def n_estables(mk, cache, rng, n_dir=12):
    seen = set()
    for _ in range(n_dir):
        w = rng.standard_normal(3); w /= np.linalg.norm(w)
        cols = stable_argmax(mk, wp_weights(mk, w), cache)
        if all(c is not None for c in cols): seen.add(tuple(cols))
    return len(seen)

# ============================================================================
if __name__ == "__main__":
    import time; t0 = time.time()
    rng = np.random.default_rng(42)
    RHO, inc_sampler = calibrar()
    N, M, MU = 24, 6, 8.0
    R_TRAIN, R_EVAL = 50, 40

    # entrenar sobre mundo correlacionado (realista)
    mks, tgs, cas = [], [], []
    for _ in range(R_TRAIN):
        mk = generar_mercado(N, M, RHO, rng, inc_sampler)
        c = {"st": _stab(mk), "as": _assign(mk)}
        mks.append(mk); cas.append(c); tgs.append(target(mk, MU, c))
    FEAT_SCALE = np.array([np.std([feats(mk, list(yh))[k] for mk, yh in zip(mks, tgs)])
                           for k in range(3)]) + 1e-9
    print("FEAT_SCALE (rank_i, rank_j, fair) =", np.round(FEAT_SCALE, 2))
    w_star = train(mks, tgs, cas, n_iter=40, lr=0.1)
    print(f"w* (a, b, d) = ({w_star[0]:+.3f}, {w_star[1]:+.3f}, {w_star[2]:+.3f})\n")

    # evaluar sobre mundo ORTOGONAL (identificación limpia del sesgo puro)
    acc = {k: {m: [] for m in ["sesgo_inc_v","sesgo_estrato_v","q_medio","rank_medio","bp"]}
           for k in ["BM","DA","SED","WP"]}
    nstab = []
    for _ in range(R_EVAL):
        mk = generar_mercado(N, M, 0.0, rng, inc_sampler)
        c = {"st": _stab(mk), "as": _assign(mk)}
        pl, sc, pf = (lambda mk: (
            [[f"C{j}" for j in np.argsort(mk["pref_rank"][i])] for i in range(mk["n"])],
            {f"C{j}": int(mk["cap"][j]) for j in range(mk["m"])},
            (lambda i, sid: float(mk["prio"][int(sid[1:]), i]))))(mk)
        s_arr = mk["s"]; ps = lambda i, sid, _s=s_arr: _s[i]*1e6 + pf(i, sid)
        cof = lambda a: [int(s[1:]) if s is not None else None for s in a]
        res = {"BM": mu.boston_mechanism(pl, sc, pf),
               "DA": mu.deferred_acceptance(pl, sc, pf),
               "SED": mu.deferred_acceptance(pl, sc, ps)}
        wp_cols = stable_argmax(mk, wp_weights(mk, w_star), c)
        for k, a in res.items():
            m = metrics(mk, cof(a))
            bp = mu.count_blocking_pairs(a, pl, sc, ps if k=="SED" else pf)
            for kk in m: acc[k][kk].append(m[kk])
            acc[k]["bp"].append(bp)
        mwp = metrics(mk, wp_cols)
        bpwp = mu.count_blocking_pairs([f"C{cc}" if cc is not None else None for cc in wp_cols], pl, sc, pf)
        for kk in mwp: acc["WP"][kk].append(mwp[kk])
        acc["WP"]["bp"].append(bpwp)
        nstab.append(n_estables(mk, c, rng))

    def ci(v):
        v = np.array(v, float); v = v[~np.isnan(v)]
        return v.mean(), 1.96*v.std()/np.sqrt(len(v))
    print("=== mundo ortogonal, media ± IC95 sobre", R_EVAL, "mercados ===")
    print(f"{'mec':4}{'sesgo(ing,v)':>20}{'sesgo(estr,v)':>20}{'q_medio':>11}{'rank':>13}{'BP':>7}")
    for k in ["BM","DA","SED","WP"]:
        a = acc[k]
        f = lambda key: "%+.3f±%.3f" % ci(a[key])
        print(f"{k:4}{f('sesgo_inc_v'):>20}{f('sesgo_estrato_v'):>20}"
              f"{np.mean(a['q_medio']):>+11.3f}{f('rank_medio'):>13}{np.mean(a['bp']):>7.1f}")
    print(f"\ndiagnóstico: # de matchings estables distintos por mercado = "
          f"{np.mean(nstab):.2f} (mediana {np.median(nstab):.0f})")
    print(f"[tiempo total: {time.time()-t0:.1f}s]")
