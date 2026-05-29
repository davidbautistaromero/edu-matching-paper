# =============================================================================
#  09h_escala_reticulo.py  —  Diametro del reticulo vs ESCALA (Tarea 4b.5)
#
#  Pregunta (md 4b.5 + comentario de David "24 podria ser pequeno"): el colapso
#  del reticulo de matchings estables que recupera DA == WP (09c), ¿es un
#  artefacto de mercados chicos (N=24), o se sostiene a escala?
#
#  Metodo: en vez de sondear el politopo con un ILP (no escala, branch-and-bound
#  se atora), se mide el DIAMETRO EXACTO del reticulo como en 09e/09g: la
#  diferencia entre el matching optimo-FAMILIAS (DA familia-proponente,
#  matching_utils.deferred_acceptance) y el optimo-COLEGIOS (DA colegio-
#  proponente). Ambas DA corren en milisegundos a cualquier escala. El gap entre
#  los extremos = # familias que difieren = diametro exacto del reticulo.
#
#  El mercado sintetico replica el DGP de 09c (utilidad estructural BLP sobre
#  submuestras reales). Se barren tres escalas con muchos mercados por escala.
#
#  RESULTADO (semilla fija, DGP-BLP de 09c):
#     24x6   (200 mercados): diametro medio 0.53 | trivial 80% | 2.21% de N
#     100x20 (200 mercados): diametro medio 1.15 | trivial 68% | 1.15% de N
#     500x50 (100 mercados): diametro medio 2.28 | trivial 53% | 0.46% de N
#  El diametro ABSOLUTO crece con N, pero el RELATIVO (gap / N) se CONTRAE.
#  El reticulo NO es un punto, pero su tamano relativo decae con la escala; en el
#  mercado real de Bogota (537k, 09e/09g) el diametro es 0.0158% (18/113857).
#  => el colapso NO es artefacto de N=24: a mayor escala, menor margen relativo
#  para que un mecanismo estable se aleje de DA. Corrobora M1 a escala.
# =============================================================================
import sys, time
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, ".")
import matching_utils as mu
# reutiliza el DGP-BLP real de 09c (cargar_betas, cargar_reales, generar_mercado)
import importlib.util
_spec = importlib.util.spec_from_file_location("c09", ROOT / "scripts" / "09c_wp_rule.py")
_src = open(ROOT / "scripts" / "09c_wp_rule.py").read().split("if __name__")[0]
c09 = importlib.util.module_from_spec(_spec)
exec(_src, c09.__dict__)

# --- extremos del reticulo via las dos DA (mismo criterio de 09e/09g) -------
def da_familia(mk):
    """DA familia-proponente -> optimo familias (matching_utils)."""
    pl = [[f"C{j}" for j in np.argsort(mk["pref_rank"][i])] for i in range(mk["n"])]
    sc = {f"C{j}": int(mk["cap"][j]) for j in range(mk["m"])}
    pf = lambda i, sid: float(mk["prio"][int(sid[1:]), i])
    a = mu.deferred_acceptance(pl, sc, pf)
    return [int(s[1:]) if s is not None else None for s in a]

def da_colegio(mk):
    """DA colegio-proponente -> optimo colegios. prio[j][i] menor = mas prioridad."""
    n, m = mk["n"], mk["m"]; prio, cap = mk["prio"], mk["cap"]; pref_rank = mk["pref_rank"]
    order = {j: np.argsort(prio[j]) for j in range(m)}
    ptr = {j: 0 for j in range(m)}; held = [None]*n; cnt = {j: 0 for j in range(m)}
    q = deque(range(m)); inq = set(range(m))
    while q:
        j = q.popleft(); inq.discard(j); O = order[j]
        while cnt[j] < cap[j] and ptr[j] < n:
            i = int(O[ptr[j]]); ptr[j] += 1; cur = held[i]
            if cur is None: held[i] = j; cnt[j] += 1
            elif pref_rank[i, j] < pref_rank[i, cur]:
                held[i] = j; cnt[j] += 1; cnt[cur] -= 1
                if cur not in inq: inq.add(cur); q.append(cur)
    return held

def diametro(mk):
    mf = da_familia(mk); mc = da_colegio(mk)
    af = sum(x is not None for x in mf); ac = sum(x is not None for x in mc)
    gap = sum(1 for a, b in zip(mf, mc) if a != b)
    return gap, af, ac

def correr_escala(sch, fu, mean_ingpc, betas, N, M, R, seed):
    rng = np.random.default_rng(seed)
    t0 = time.time(); gaps = []; mismatch = 0
    for _ in range(R):
        mk = c09.generar_mercado(sch, fu, mean_ingpc, betas, N, M, rng)
        gap, af, ac = diametro(mk)
        gaps.append(gap)
        if af != ac: mismatch += 1
    gaps = np.array(gaps)
    print(f"\n=== {N}x{M}, {R} mercados  [{time.time()-t0:.1f}s] ===")
    print(f"  diametro del reticulo (DA-fam vs DA-col):")
    print(f"     media={gaps.mean():.2f}  mediana={np.median(gaps):.0f}  "
          f"min={gaps.min()}  max={gaps.max()}")
    print(f"  reticulo TRIVIAL (gap=0, estable unico): "
          f"{(gaps==0).sum()}/{R} ({100*(gaps==0).mean():.0f}%)")
    print(f"  diametro RELATIVO (gap / N), media: {100*gaps.mean()/N:.2f}%")
    if mismatch: print(f"  [aviso: {mismatch} mercados con |asignados| distinto entre extremos -> revisar]")
    return gaps

# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    betas = c09.cargar_betas()
    print(f"betas IV-BLP: pi1={betas[0]:+.5f}, lam0={betas[1]:+.5f}, lam1={betas[2]:+.5f}")
    sch, fu = c09.cargar_reales()
    mean_ingpc = float(pd.to_numeric(fu["N_ingpc"], errors="coerce").mean())
    print(f"colegios reales={len(sch)}, familias reales={len(fu)}, mean_ingpc={mean_ingpc:,.0f}")

    print("\n" + "="*70)
    print("DIAMETRO DEL RETICULO vs ESCALA  (metodo exacto de 09e, sin ILP)")
    print("="*70)
    g24  = correr_escala(sch, fu, mean_ingpc, betas, 24,  6,  200, seed=2024)
    g100 = correr_escala(sch, fu, mean_ingpc, betas, 100, 20, 200, seed=2025)
    g500 = correr_escala(sch, fu, mean_ingpc, betas, 500, 50, 100, seed=2026)

    rel24  = 100*g24.mean()/24
    rel100 = 100*g100.mean()/100
    rel500 = 100*g500.mean()/500
    print("\n" + "="*70)
    print("Lectura: el diametro ABSOLUTO crece con N (media {:.2f} -> {:.2f} -> {:.2f}),"
          .format(g24.mean(), g100.mean(), g500.mean()))
    print("pero el RELATIVO (gap/N) se CONTRAE ({:.2f}% -> {:.2f}% -> {:.2f}%)."
          .format(rel24, rel100, rel500))
    print("El colapso del reticulo NO es artefacto de mercados chicos (N=24). En")
    print("Bogota real (537k familias, 09e/09g) el diametro es 0.0158% (18/113857).")
    print("A mayor escala, menor margen relativo para que un mecanismo estable se")
    print("aleje de DA. Corrobora M1 a escala.")
    print(f"[tiempo total: {time.time()-t0:.1f}s]")
