# =============================================================================
#  09c_wp_rule.py  —  Mecanismo aprendido (WP-Rule) via StructSVM
#  Cuarto mecanismo del paper, comparado contra BM / DA / SED. Reutiliza
#  matching_utils.py (BM, DA, blocking pairs).
#
#  Teoria: el conjunto de matchings estables es un politopo integro
#  (Vande Vate 1989; Rothblum 1992; Roth, Rothblum & Vande Vate 1993; caso
#  con capacidades = politopo de b-matching estable). Maximizar una funcion
#  lineal sobre el entrega un vertice => matching estable. Los pesos se aprenden
#  (Narasimhan, Agarwal & Parkes 2016, marco estilo SVM).
#
#  --- Tarea 4b (post comentarios David, md notas_metodologicas) -------------
#  4b.8  paths relativos (ROOT = Path(__file__)...), sin hardcoding /content/
#  4b.4  DGP alineado con BLP: el mercado de entrenamiento muestrea colegios y
#        familias REALES; la utilidad usa la formula y los betas del IV-BLP:
#          u_ij = delta_j + pi1*y_i*seg_z_j + lam0*log1p(d_ij)
#                 + lam1*y_i*log1p(d_ij) + eps,  eps~Gumbel(0,1)
#        (pi1,lam0,lam1) de reports/tables/blp_results.csv fila iv_blp.
#  4b.1  funcion objetivo de 3 terminos (calidad + romper sesgo visual +
#        equidad compensatoria en calidad):
#          w_ij = q_j - mu1*(y_i^c*v_j) - mu2*(y_i^c*q_j)
#  4b.2  4 features de pesos: lambda_ij = a*rank_i + b*rank_j
#        + d1*(y_i^c*v_j) + d2*(y_i^c*q_j)
#  4b.6  acceso garantizado SISBEN A/B. Dos variantes:
#        hard  -> restriccion Sum_j x_ij = 1 para todo i con SISBEN in {A,B}
#        soft  -> feature extra e*1[SISBEN_i in {A,B}] en los pesos
#  4b.3  baselines: WP equal-weights (a=b=1, d=0) y WP paper-original (d=0)
# =============================================================================
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# --- 4b.8: paths relativos --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, ".")
import matching_utils as mu

PRIMARY = ROOT / "data" / "primary"
CLIP_P  = ROOT / "data" / "images" / "clip" / "gsv_clip_establecimiento.parquet"
CAP_P   = ROOT / "data" / "primary" / "colegios_capacidad.parquet"
FU_P    = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
GEO_P   = ROOT / "data" / "primary" / "colegios_features_imputed.geojson"
BLP_P   = ROOT / "reports" / "tables" / "blp_results.csv"

DANE = (227220.0, 460198.0, 897987.0)            # umbrales SISBEN (DANE)
FEAT_SCALE = np.array([1.0, 1.0, 1.0, 1.0])      # 4 features (4b.2); recalibrado en runtime
SISBEN_AB = {1, 2}                                # grupos A,B = acceso garantizado (4b.6)

def sisben_group(inc):
    return 1 if inc < DANE[0] else 2 if inc < DANE[1] else 3 if inc < DANE[2] else 4

def _snorm(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

_P = 0.017453292519943295
def _hav(la1, lo1, la2, lo2):
    dlat = (la2 - la1) * _P; dlon = (lo2 - lo1) * _P
    a = np.sin(dlat/2)**2 + np.cos(la1*_P)*np.cos(la2*_P)*np.sin(dlon/2)**2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# --- 4b.4: betas del IV-BLP -------------------------------------------------
def cargar_betas():
    import pandas as pd
    bp = pd.read_csv(BLP_P)
    iv = bp[bp["spec"] == "iv_blp"].set_index("parametro")["estimacion"]
    return float(iv["pi1"]), float(iv["lam0"]), float(iv["lam1"])

# --- 4b.4: insumos reales (delta_j, seg_z, q_j, coords, capacidad, ingresos) -
def cargar_reales():
    import pandas as pd, geopandas as gpd
    delta = pd.read_parquet(PRIMARY / "blp_delta_j.parquet")
    delta["id_establecimiento"] = _snorm(delta["id_establecimiento"])
    clip = pd.read_parquet(CLIP_P)
    clip["id_establecimiento"] = _snorm(clip["id_establecimiento"])
    sg = clip["seguridad_percibida"]; clip["seg_z"] = (sg - sg.mean()) / sg.std()
    cap = pd.read_parquet(CAP_P).drop_duplicates("id_establecimiento")
    cap["id_establecimiento"] = _snorm(cap["id_establecimiento"])
    gdf = gpd.read_file(GEO_P)[["id_establecimiento", "q_j"]]
    gdf["id_establecimiento"] = _snorm(gdf["id_establecimiento"])
    sch = (delta.merge(clip[["id_establecimiento", "seg_z"]], on="id_establecimiento", how="inner")
                .merge(cap[["id_establecimiento", "lat", "lon", "capacidad"]], on="id_establecimiento", how="inner")
                .merge(gdf, on="id_establecimiento", how="inner"))
    sch = sch[sch["capacidad"].notna() & sch["q_j"].notna()].reset_index(drop=True)
    fu = pd.read_parquet(FU_P)
    fu["DIRECTORIO"] = _snorm(fu["DIRECTORIO"])
    return sch, fu

# --- 4b.4: mercado de entrenamiento = submuestra REAL, utilidad BLP ----------
def generar_mercado(sch, fu, mean_ingpc, betas, n, m, rng):
    PI1, LAM0, LAM1 = betas
    d_lat = sch["lat"].to_numpy(float); d_lon = sch["lon"].to_numpy(float)
    delta_j = sch["delta_j_blp"].to_numpy(float)
    seg_z   = sch["seg_z"].to_numpy(float)
    q_real  = sch["q_j"].to_numpy(float)
    f_inc = np.asarray([np.nan])  # placeholder; reasignado abajo
    import pandas as pd
    f_inc = pd.to_numeric(fu["N_ingpc"], errors="coerce").to_numpy(float)
    f_inc = np.where(np.isnan(f_inc) | (f_inc <= 0), mean_ingpc, f_inc)
    f_lat = fu["lat"].to_numpy(float); f_lon = fu["lon"].to_numpy(float)

    fi = rng.choice(len(fu), n, replace=False)
    si = rng.choice(len(sch), m, replace=False)
    inc = f_inc[fi]
    yi  = inc / mean_ingpc
    d = _hav(f_lat[fi][:, None], f_lon[fi][:, None], d_lat[si][None, :], d_lon[si][None, :])
    ld = np.log1p(d)
    # 4b.4: utilidad estructural BLP
    U = (delta_j[si][None, :] + PI1*yi[:, None]*seg_z[si][None, :]
         + LAM0*ld + LAM1*yi[:, None]*ld + rng.gumbel(0, 1, size=(n, m)))
    pref_rank = (-U).argsort(1).argsort(1) + 1
    # prioridad sintetica (loteria) para el mundo de entrenamiento
    prio = np.vstack([rng.permutation(n).argsort() + 1 for _ in range(m)])
    cap = np.full(m, int(np.ceil(n / m)) + 1)
    # q_j y v_j (=seg_z) estandarizados sobre la submuestra, para target/metricas (4b.1)
    q = (q_real[si] - q_real[si].mean()) / (q_real[si].std() + 1e-9)
    v = (seg_z[si] - seg_z[si].mean()) / (seg_z[si].std() + 1e-9)
    s = np.array([sisben_group(x) for x in inc])
    estrato = np.searchsorted(np.quantile(inc, [1/6,2/6,3/6,4/6,5/6]), inc) + 1
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

# --- 4b.6 hard: igualdad Sum_j x_ij = 1 para SISBEN A/B (acceso garantizado) -
def _access_eq(mk):
    """Filas de IGUALDAD: cada familia SISBEN A/B debe quedar asignada."""
    n, m = mk["n"], mk["m"]; R, b = [], []
    for i in range(n):
        if int(mk["s"][i]) in SISBEN_AB:
            r = np.zeros(n*m); r[i*m:(i+1)*m] = 1.0; R.append(r); b.append(1.0)
    if not R:
        return np.zeros((0, n*m)), np.zeros(0)
    return np.array(R), np.array(b)

def stable_argmax(mk, weights, cache, hard_access=False):
    """ILP binario -> matching estable de maximo peso.
       hard_access=True agrega la restriccion dura de acceso SISBEN A/B (4b.6 hard)."""
    A_ub = np.vstack([cache["as"][0], cache["st"][0]])
    b_ub = np.concatenate([cache["as"][1], cache["st"][1]])
    cons = [LinearConstraint(A_ub, -np.inf, b_ub)]
    if hard_access:
        Aeq, beq = cache.get("acc", _access_eq(mk))
        if Aeq.shape[0] > 0:
            cons.append(LinearConstraint(Aeq, beq, beq))   # igualdad
    res = milp(c=-weights.ravel(), constraints=cons,
               integrality=np.ones(mk["n"]*mk["m"]), bounds=Bounds(0, 1))
    if res.x is None:
        return None                                        # infeasible (p.ej. hard sin holgura)
    x = np.round(res.x).reshape(mk["n"], mk["m"])
    return [int(np.argmax(x[i])) if x[i].max() > 0.5 else None for i in range(mk["n"])]

# --- 4b.2: 4 features (rank_i, rank_j, inc*v, inc*q) [+ soft access opcional] -
def wp_weights(mk, w, soft_access=False):
    a, b, d1, d2 = w[0], w[1], w[2], w[3]
    s0, s1, s2, s3 = FEAT_SCALE[:4]
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    W = (a*mk["pref_rank"]/s0 + b*mk["prio"].T/s1
         + d1*np.outer(inc_c, mk["v"])/s2
         + d2*np.outer(inc_c, mk["q"])/s3)
    if soft_access and len(w) >= 5:
        e = w[4]
        ab = np.array([1.0 if int(s) in SISBEN_AB else 0.0 for s in mk["s"]])
        W = W + e * ab[:, None]                            # 4b.6 soft: premia asignar A/B
    return W

def feats(mk, cols, soft_access=False):
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    n = mk["n"]
    base = [sum(mk["pref_rank"][i, cols[i]] for i in range(n)),
            sum(mk["prio"][cols[i], i]      for i in range(n)),
            sum(inc_c[i]*mk["v"][cols[i]]   for i in range(n)),
            sum(inc_c[i]*mk["q"][cols[i]]   for i in range(n))]
    if soft_access:
        ab = np.array([1.0 if int(s) in SISBEN_AB else 0.0 for s in mk["s"]])
        base.append(sum(ab[i] for i in range(n) if cols[i] is not None))
    return np.array(base, float)

# --- 4b.1: target ideal de 3 terminos ---------------------------------------
def target(mk, mu1, mu2, cache):
    inc_c = (mk["inc"] - mk["inc"].mean()) / (mk["inc"].std() + 1e-9)
    w = (mk["q"][None, :]
         - mu1 * np.outer(inc_c, mk["v"])
         - mu2 * np.outer(inc_c, mk["q"]))
    res = linprog(c=-w.ravel(), A_ub=cache["as"][0], b_ub=cache["as"][1],
                  bounds=(0, 1), method="highs")
    return res.x.reshape(mk["n"], mk["m"]).argmax(1)

# --- StructSVM (subgradiente) -----------------------------------------------
def train(markets, targets, caches, n_iter=40, lr=0.1, reg=1e-3, ndim=4, soft_access=False):
    w = np.zeros(ndim); w[0] = -1.0
    for _ in range(n_iter):
        g = np.zeros(ndim)
        for mk, yh, c in zip(markets, targets, caches):
            aug = wp_weights(mk, w, soft_access).astype(float)
            for i in range(mk["n"]): aug[i, yh[i]] -= 1.0
            ystar = stable_argmax(mk, aug, c)
            g += feats(mk, ystar, soft_access) - feats(mk, yh, soft_access)
        g = g/len(markets) + reg*w
        w = w - lr*g; nrm = np.linalg.norm(w)
        if nrm > 1e-9: w /= nrm
    return w

# --- metricas: sesgo, equidad en calidad, calidad, rank ---------------------
def metrics(mk, cols):
    matched = np.array([c is not None for c in cols])
    cc = np.array([c if c is not None else 0 for c in cols])
    def corr(x, y):
        return float(np.corrcoef(x[matched], y[matched])[0,1]) if matched.sum() > 2 else np.nan
    rank = np.array([mk["pref_rank"][i, cc[i]] if matched[i] else mk["m"]+1 for i in range(mk["n"])])
    return dict(sesgo_inc_v=corr(mk["inc"], mk["v"][cc]),
                sesgo_estrato_v=corr(mk["estrato"].astype(float), mk["v"][cc]),
                equidad_inc_q=corr(mk["inc"], mk["q"][cc]),     # 4b.1: objetivo corr<=0
                q_medio=float(mk["q"][cc][matched].mean()),
                rank_medio=float(rank.mean()),
                # acceso SISBEN A/B logrado (4b.6): fraccion de A/B asignadas
                acc_ab=float(np.mean([matched[i] for i in range(mk["n"]) if int(mk["s"][i]) in SISBEN_AB])
                             if any(int(s) in SISBEN_AB for s in mk["s"]) else np.nan))

# ============================================================================
if __name__ == "__main__":
    import time, pandas as pd
    t0 = time.time()
    rng = np.random.default_rng(42)
    betas = cargar_betas()
    print(f"4b.4 betas IV-BLP: pi1={betas[0]:+.5f}, lam0={betas[1]:+.5f}, lam1={betas[2]:+.5f}")
    sch, fu = cargar_reales()
    mean_ingpc = float(pd.to_numeric(fu["N_ingpc"], errors="coerce").mean())
    print(f"colegios reales={len(sch)}, familias reales={len(fu)}, mean_ingpc={mean_ingpc:,.0f}")

    N, M, MU1, MU2 = 24, 6, 8.0, 4.0
    R_TRAIN, R_EVAL = 50, 40

    # --- entrenamiento (mundo BLP real) ---
    mks, cas = [], []
    for _ in range(R_TRAIN):
        mk = generar_mercado(sch, fu, mean_ingpc, betas, N, M, rng)
        c = {"st": _stab(mk), "as": _assign(mk), "acc": _access_eq(mk)}
        mks.append(mk); cas.append(c)
    tgs = [target(mk, MU1, MU2, c) for mk, c in zip(mks, cas)]
    FEAT_SCALE = np.array([np.std([feats(mk, list(yh))[k] for mk, yh in zip(mks, tgs)])
                           for k in range(4)]) + 1e-9
    print("FEAT_SCALE (rank_i, rank_j, inc*v, inc*q) =", np.round(FEAT_SCALE, 2))

    # WP full (4b.1+4b.2): 4 features, objetivo 3 terminos
    w_full = train(mks, tgs, cas, ndim=4)
    print(f"\n4b.1+4b.2  w* full (a,b,d1,d2) = {np.round(w_full,3)}")

    # 4b.3 baselines
    w_equal = np.array([1.0, 1.0, 0.0, 0.0])                       # equal-weights, sin entrenar
    tgs_po = [target(mk, 0.0, 0.0, c) for mk, c in zip(mks, cas)]  # paper-original: solo calidad
    FS_full = FEAT_SCALE.copy()
    FEAT_SCALE = np.array([np.std([feats(mk, list(yh))[k] for mk, yh in zip(mks, tgs_po)])
                           for k in range(4)]) + 1e-9
    w_po = train(mks, tgs_po, cas, ndim=4); w_po = np.array([w_po[0], w_po[1], 0.0, 0.0])
    print(f"4b.3  w* paper-original (a,b,0,0) = {np.round(w_po,3)}")
    print(f"4b.3  w  equal-weights  (1,1,0,0) = {np.round(w_equal,3)}")

    # --- evaluacion en mundo ORTOGONAL (sesgo visual puro) ---
    # mundo ortogonal: re-muestrea reales pero descorrelaciona v de q permutando v
    def mk_ortogonal(seed_rng):
        mk = generar_mercado(sch, fu, mean_ingpc, betas, N, M, seed_rng)
        mk["v"] = mk["v"][seed_rng.permutation(M)]   # rompe corr(q,v) en la submuestra
        return mk

    mecs = ["BM","DA","SED","WP","WP_eq","WP_po","WP_hard","WP_soft"]
    keys = ["sesgo_inc_v","sesgo_estrato_v","equidad_inc_q","q_medio","rank_medio","acc_ab","bp"]
    acc = {k: {kk: [] for kk in keys} for k in mecs}
    fam_vs_da = {k: 0 for k in ["WP","WP_eq","WP_po","WP_hard","WP_soft"]}
    hard_infeasible = 0

    # entrenar WP_soft (5 features) una vez
    FEAT_SCALE = np.array(list(FS_full) + [1.0])
    tgs_soft = tgs   # mismo target de 3 terminos; el acceso entra por la feature
    FEAT_SCALE = np.array([np.std([feats(mk, list(yh), True)[k] for mk, yh in zip(mks, tgs_soft)])
                           for k in range(5)]) + 1e-9
    FS_soft = FEAT_SCALE.copy()
    w_soft = train(mks, tgs_soft, cas, ndim=5, soft_access=True)
    print(f"4b.6 soft  w* (a,b,d1,d2,e) = {np.round(w_soft,3)}  (e>0 => premia acceso A/B)")

    for _ in range(R_EVAL):
        mk = mk_ortogonal(rng)
        c = {"st": _stab(mk), "as": _assign(mk), "acc": _access_eq(mk)}
        pl = [[f"C{j}" for j in np.argsort(mk["pref_rank"][i])] for i in range(mk["n"])]
        sc = {f"C{j}": int(mk["cap"][j]) for j in range(mk["m"])}
        pf = lambda i, sid: float(mk["prio"][int(sid[1:]), i])
        s_arr = mk["s"]; ps = lambda i, sid, _s=s_arr: _s[i]*1e6 + pf(i, sid)
        cof = lambda a: [int(s[1:]) if s is not None else None for s in a]

        res = {"BM": mu.boston_mechanism(pl, sc, pf),
               "DA": mu.deferred_acceptance(pl, sc, pf),
               "SED": mu.deferred_acceptance(pl, sc, ps)}
        da_cols = cof(res["DA"])

        FEAT_SCALE = FS_full
        wp_cols   = stable_argmax(mk, wp_weights(mk, w_full), c)
        wpeq_cols = stable_argmax(mk, wp_weights(mk, w_equal), c)
        wppo_cols = stable_argmax(mk, wp_weights(mk, w_po), c)
        wphard    = stable_argmax(mk, wp_weights(mk, w_full), c, hard_access=True)
        FEAT_SCALE = FS_soft
        wpsoft    = stable_argmax(mk, wp_weights(mk, w_soft, soft_access=True), c)
        FEAT_SCALE = FS_full

        if wphard is None: hard_infeasible += 1

        wp_map = {"WP": wp_cols, "WP_eq": wpeq_cols, "WP_po": wppo_cols,
                  "WP_hard": wphard, "WP_soft": wpsoft}

        for k, a in res.items():
            m = metrics(mk, cof(a))
            bp = mu.count_blocking_pairs(a, pl, sc, ps if k=="SED" else pf)
            for kk in m: acc[k][kk].append(m[kk])
            acc[k]["bp"].append(bp)
        for k, cols in wp_map.items():
            if cols is None: continue
            m = metrics(mk, cols)
            bp = mu.count_blocking_pairs([f"C{cc}" if cc is not None else None for cc in cols], pl, sc, pf)
            for kk in m: acc[k][kk].append(m[kk])
            acc[k]["bp"].append(bp)
            fam_vs_da[k] += sum(1 for x, y in zip(cols, da_cols) if x != y)

    def ci(v):
        v = np.array(v, float); v = v[~np.isnan(v)]
        return (v.mean(), 1.96*v.std()/np.sqrt(len(v))) if len(v) else (np.nan, np.nan)

    print("\n=== mundo ortogonal, media +- IC95 sobre", R_EVAL, "mercados ===")
    print(f"{'mec':8}{'sesgo(inc,v)':>16}{'equid(inc,q)':>16}{'q_med':>9}{'rank':>8}{'acc_AB':>9}{'BP':>7}{'!=DA':>7}")
    for k in mecs:
        a = acc[k]
        if not a["q_medio"]: 
            print(f"{k:8}  (sin datos / infeasible)"); continue
        fvd = fam_vs_da.get(k, "-")
        print(f"{k:8}{'%+.3f'%ci(a['sesgo_inc_v'])[0]:>16}{'%+.3f'%ci(a['equidad_inc_q'])[0]:>16}"
              f"{np.mean(a['q_medio']):>+9.3f}{np.mean(a['rank_medio']):>8.2f}"
              f"{ci(a['acc_ab'])[0]:>9.3f}{np.mean(a['bp']):>7.1f}{str(fvd):>7}")

    print(f"\n4b.6 hard: mercados infeasible = {hard_infeasible}/{R_EVAL}")
    if hard_infeasible == 0:
        bp_hard = np.mean(acc['WP_hard']['bp']) if acc['WP_hard']['bp'] else float('nan')
        print(f"           BP medio WP_hard = {bp_hard:.1f} "
              f"({'estable (BP=0) -> hard viable como en el md' if bp_hard==0 else 'BP>0 -> hard ROMPE estabilidad'})")
    print(f"[tiempo total: {time.time()-t0:.1f}s]")
