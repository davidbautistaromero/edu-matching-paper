# =============================================================================
#  09j_acceso_sisben_real.py  —  Acceso de familias vulnerables bajo escasez real
#
#  Complementa 4b.6 / 4b.7: la restriccion de acceso (4b.6) se evaluo en el mundo
#  sintetico de 09c, que tiene HOLGURA de cupos -> ahi resulto inocua (todas las
#  A/B ya entraban). Este script la evalua donde SI importa: los datos reales de
#  Bogota, con escasez aguda (120k cupos para 537k familias, ~21% asignadas).
#
#  Mide cuantas familias SISBEN A/B quedan SIN cupo bajo cada prioridad
#  (distancia vs SED=SISBEN+distancia), desagregado por grupo SISBEN.
#
#  HALLAZGO (semilla fija, datos reales):
#    A+B sin cupo: distancia = 233872 -> SED = 183071  (SED mete 50.801 A/B mas)
#  Pero el desagregado revela que SED NO ayuda parejo a los vulnerables:
#    SISBEN A: 20.9% -> 89.1%  (el grupo mas pobre entra masivamente)
#    SISBEN B: 21.2% ->  7.4%  (CAE: queda peor que bajo distancia)
#    SISBEN C: 21.6% ->  0.3%
#    SISBEN D: 20.5% ->  0.0%
#  La prioridad SED redistribuye DENTRO de los vulnerables, hacia los MAS pobres
#  (A), a costa de expulsar a B/C/D. No es Pareto-mejor para los vulnerables.
#
#  Implicacion para 4b.6: bajo escasez, la prioridad SED logra el acceso del grupo
#  A pero hunde a B/C/D. Una restriccion de acceso con piso para B (no solo A)
#  SI tendria un rol que la prioridad pura no cumple. La tension de politica real
#  es: priorizar a los mas pobres tiene un costo sobre los pobres del medio.
# =============================================================================
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, ".")
import matching_utils as mu

_spec = importlib.util.spec_from_file_location("mod09g", ROOT / "scripts" / "09g_uniqueness_sed.py")
g = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(g)

def asignacion(D, LOT, sisben=None):
    order = g.build_order(D, LOT, sisben)
    BIG = np.int32(D["N"]+1); tab = {}
    for s in D["school_ids"]:
        r = np.full(D["N"], BIG, np.int32); o = order[s]
        if o.size: r[o] = np.arange(o.size, dtype=np.int32)
        tab[s] = r
    prio = lambda i, sid: float(tab[sid][i])
    return mu.deferred_acceptance(D["pref_lists"], D["school_cap"], prio)

def tasa_acceso(matching, sis, N, label):
    asignada = np.array([x is not None for x in matching])
    print(f"\n[{label}]")
    print(f"  asignadas totales: {asignada.sum()} de {N} ({100*asignada.mean():.1f}%)")
    for grp, nom in [(1,"A"),(2,"B"),(3,"C"),(4,"D")]:
        mask = sis == grp; n = int(mask.sum()); asig = int(asignada[mask].sum())
        print(f"  SISBEN {nom}: {asig}/{n} con cupo ({100*asig/max(n,1):.1f}%) | sin cupo: {n-asig}")
    ab = (sis==1)|(sis==2)
    sin_ab = int((~asignada[ab]).sum())
    print(f"  >>> A+B SIN cupo: {sin_ab} de {int(ab.sum())} familias vulnerables")
    return sin_ab

if __name__ == "__main__":
    t0 = time.time()
    print("cargando datos reales...")
    D = g.cargar()
    sis = D["fam_sisben"].astype(int)
    LOT = np.arange(D["N"], dtype=np.int64)

    print("\n" + "="*64)
    print("ACCESO de familias vulnerables (SISBEN A/B) por prioridad, DATOS REALES")
    print("="*64)
    m_dist = asignacion(D, LOT, sisben=None)
    m_sed  = asignacion(D, LOT, sisben=D["fam_sisben"])
    sin_dist = tasa_acceso(m_dist, sis, D["N"], "PRIORIDAD = DISTANCIA")
    sin_sed  = tasa_acceso(m_sed,  sis, D["N"], "PRIORIDAD = SED (SISBEN)")

    print("\n" + "="*64)
    print(f"A+B sin cupo:  distancia = {sin_dist}  ->  SED = {sin_sed}  (delta = {sin_sed-sin_dist:+d})")
    print("Lectura: SED mete mas A/B en total, pero redistribuye DENTRO de los")
    print("vulnerables: el grupo A entra (89%) a costa de expulsar a B (cae a 7%),")
    print("C y D. No es Pareto-mejor para los vulnerables -> una restriccion con piso")
    print("para B tendria un rol que la prioridad pura no cumple. La tension real es:")
    print("priorizar a los mas pobres cuesta acceso de los pobres del medio.")
    print(f"[tiempo total: {time.time()-t0:.1f}s]")
