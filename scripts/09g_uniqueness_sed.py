# =============================================================================
#  09g_uniqueness_sed.py  —  Unicidad del reticulo bajo prioridades SED (Tarea 4b.7)
#
#  Pregunta (md 4b.7): 09e verifica unicidad del matching estable solo bajo
#  prioridad-DISTANCIA. Si bajo prioridad SED (grupo SISBEN + distancia) el
#  reticulo se ABRE, la WP-Rule tendria espacio para optimizar y el resultado
#  principal (M1) cambiaria. Este script lo decide.
#
#  Metodo (identico a 09e): el conjunto estable es un reticulo cuyos extremos son
#  el optimo-familias (DA familia-proponente) y el optimo-colegios (DA colegio-
#  proponente). El ANCHO = # familias que difieren entre extremos = diametro
#  EXACTO del reticulo. Se corre DOS veces con la MISMA loteria de desempate
#  (indice), cambiando SOLO la clave de prioridad:
#     (1) distancia pura   -> control, debe reproducir el ~18 de 09e
#     (2) SED = SISBEN + distancia (SISBEN primario, distancia secundaria)
#  Asi cualquier cambio en el ancho es atribuible a SED y no al sorteo.
#
#  SISBEN se calcula del ingreso N_ingpc con los umbrales DANE; ingreso faltante
#  o <=0 se imputa con la media (consistente con 06_preferencias.py y 09f).
#
#  RESULTADO (semilla fija, datos reales de Bogota):
#     ancho distancia = 18 | ancho SED = 12.
#  El reticulo NO se abre bajo SED: se CONTRAE. Reordenar prioridades a favor de
#  los pobres (SISBEN A/B delante del 78% restante) no le da margen a WP para
#  redistribuir; lo aprieta. Corrobora y REFUERZA M1: ningun mecanismo estable
#  redistribuye el sesgo visual, ni siquiera bajo las prioridades reales del SED.
#  Efecto de cambiar la PRIORIDAD (distancia -> SED) bajo escasez (120k cupos,
#  537k familias): solo 33905 familias (30% de las asignadas) reciben cupo bajo
#  AMBOS sistemas; 79952 pierden el cupo y otras 79952 lo ganan. Cambiar la
#  prioridad reemplaza el 70% de QUIEN obtiene cupo. => la palanca de politica
#  es la PRIORIDAD (a quien se prioriza), no el MECANISMO (DA/SED/WP coinciden
#  dado un sistema de prioridad fijo, ancho 12-18).
# =============================================================================
import sys, time
from pathlib import Path
from collections import deque, Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, ".")
import matching_utils as mu

PREF_P  = ROOT / "data" / "primary"   / "preferencias_familias.parquet"
CAP_P   = ROOT / "data" / "primary"   / "colegios_capacidad.parquet"
COORD_P = ROOT / "data" / "processed" / "familias_ubicadas.parquet"
CAP_COL = "capacidad"
DANE = (227220.0, 460198.0, 897987.0)

def snorm(s): return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
def norm_id(x):
    s = str(x).strip(); return s[:-2] if s.endswith(".0") else s
_P = 0.017453292519943295
def hav_vec(la1, lo1, la2, lo2):
    dlat = (la2-la1)*_P; dlon = (lo2-lo1)*_P
    a = np.sin(dlat/2)**2 + np.cos(la1*_P)*np.cos(la2*_P)*np.sin(dlon/2)**2
    return 12742.0*np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

# ----------------------------------------------------------------- carga
def cargar():
    df_pref = pd.read_parquet(PREF_P)
    pref_cols = [c for c in df_pref.columns if str(c).lower().startswith("pref_")]
    directorios = (snorm(df_pref["DIRECTORIO"]).to_numpy() if "DIRECTORIO" in df_pref.columns
                   else snorm(pd.Series(df_pref.index)).to_numpy())
    N = len(df_pref)
    raw = df_pref[pref_cols].to_numpy(dtype=object)

    df_cap = pd.read_parquet(CAP_P).drop_duplicates(subset=["id_establecimiento"])
    df_cap = df_cap[df_cap[CAP_COL].notna()].copy()
    df_cap["id_establecimiento"] = snorm(df_cap["id_establecimiento"])
    school_ids = df_cap["id_establecimiento"].tolist()
    sch_idx = {s: k for k, s in enumerate(school_ids)}
    school_cap = {s: int(c) for s, c in zip(school_ids, df_cap[CAP_COL])}
    sch_lat = df_cap["lat"].to_numpy(float); sch_lon = df_cap["lon"].to_numpy(float)
    M = len(school_ids)

    df_co = pd.read_parquet(COORD_P).copy()
    df_co["DIRECTORIO"] = snorm(df_co["DIRECTORIO"])
    df_co = df_co.drop_duplicates(subset=["DIRECTORIO"]).set_index("DIRECTORIO")
    fam_lat = df_co["lat"].reindex(directorios).to_numpy(float)
    fam_lon = df_co["lon"].reindex(directorios).to_numpy(float)

    # SISBEN por familia (imputacion de media, igual que 06/09f)
    inc_dir = pd.to_numeric(df_co["N_ingpc"], errors="coerce").to_numpy(float)
    mean_inc = float(np.nanmean(inc_dir))
    inc_safe = np.where(np.isnan(inc_dir) | (inc_dir <= 0), mean_inc, inc_dir)
    sis_dir = np.where(inc_safe < DANE[0], 1, np.where(inc_safe < DANE[1], 2,
              np.where(inc_safe < DANE[2], 3, 4)))
    mean_group = (1 if mean_inc < DANE[0] else 2 if mean_inc < DANE[1]
                  else 3 if mean_inc < DANE[2] else 4)
    df_co["_sis"] = sis_dir
    fam_sisben = df_co["_sis"].reindex(directorios).to_numpy(float)
    fam_sisben = np.where(np.isnan(fam_sisben), mean_group, fam_sisben)

    # listas de preferencia con strings canonicos
    canon = {s: s for s in school_ids}
    pref_lists = []
    for row in raw:
        pl = []
        for x in row:
            if x is None: continue
            cs = canon.get(norm_id(x))
            if cs is not None: pl.append(cs)
        pref_lists.append(pl)
    stu_rank = [{s: r for r, s in enumerate(pl)} for pl in pref_lists]
    cand = {s: [] for s in school_ids}
    for i, pl in enumerate(pref_lists):
        for s in pl: cand[s].append(i)
    cand = {s: np.asarray(v, np.int64) for s, v in cand.items()}

    return dict(N=N, M=M, directorios=directorios, school_ids=school_ids, sch_idx=sch_idx,
                school_cap=school_cap, sch_lat=sch_lat, sch_lon=sch_lon,
                fam_lat=fam_lat, fam_lon=fam_lon, fam_sisben=fam_sisben,
                mean_inc=mean_inc, mean_group=mean_group,
                pref_lists=pref_lists, stu_rank=stu_rank, cand=cand)

# ----------------------------------------------------------------- motores
def build_order(D, tiebreak, sisben=None):
    """Orden estricto de cada colegio sobre sus candidatos.
       lexsort usa la ULTIMA clave como PRIMARIA:
         sin SED -> (tiebreak, distancia)             => distancia primaria
         con SED -> (tiebreak, distancia, sisben)      => SISBEN primaria"""
    order = {}
    for s in D["school_ids"]:
        fa = D["cand"][s]
        if fa.size == 0: order[s] = fa; continue
        k = D["sch_idx"][s]
        d = hav_vec(D["fam_lat"][fa], D["fam_lon"][fa], D["sch_lat"][k], D["sch_lon"][k])
        d = np.where(np.isnan(d), np.inf, d)
        keys = (tiebreak[fa], d) if sisben is None else (tiebreak[fa], d, sisben[fa])
        order[s] = fa[np.lexsort(keys)]
    return order

def college_da(D, order):
    """DA colegio-proponente -> optimo para colegios."""
    N = D["N"]; held = [None]*N
    cnt = {s: 0 for s in D["school_ids"]}; ptr = {s: 0 for s in D["school_ids"]}
    inq = set(D["school_ids"]); q = deque(D["school_ids"])
    stu_rank = D["stu_rank"]; cap = D["school_cap"]
    while q:
        s = q.popleft(); inq.discard(s)
        O = order[s]; L = O.shape[0]
        while cnt[s] < cap[s] and ptr[s] < L:
            i = int(O[ptr[s]]); ptr[s] += 1; cur = held[i]
            if cur is None: held[i] = s; cnt[s] += 1
            elif stu_rank[i][s] < stu_rank[i][cur]:
                held[i] = s; cnt[s] += 1; cnt[cur] -= 1
                if cur not in inq: inq.add(cur); q.append(cur)
    return held

def run_pair(D, tiebreak, label, sisben=None):
    N = D["N"]
    order = build_order(D, tiebreak, sisben)
    BIG = np.int32(N+1); tab = {}
    for s in D["school_ids"]:
        r = np.full(N, BIG, np.int32); o = order[s]
        if o.size: r[o] = np.arange(o.size, dtype=np.int32)
        tab[s] = r
    def prio(i, sid): return float(tab[sid][i])
    t1 = time.time(); mS = mu.deferred_acceptance(D["pref_lists"], D["school_cap"], prio); t2 = time.time()
    mC = college_da(D, order); t3 = time.time()
    aS = sum(x is not None for x in mS); aC = sum(x is not None for x in mC)
    gap = sum(1 for a, b in zip(mS, mC) if a != b); den = aS if aS else N
    bpS = mu.count_blocking_pairs(mS, D["pref_lists"], D["school_cap"], prio)
    bpC = mu.count_blocking_pairs(mC, D["pref_lists"], D["school_cap"], prio)
    print(f"\n[{label}]  DA-fam {t2-t1:.1f}s | DA-col {t3-t2:.1f}s")
    print(f"  asignadas: fam-opt={aS}, col-opt={aC} "
          f"({'rural-hospitals OK' if aS==aC else 'DIFIEREN -> bug de desempate'})")
    print(f"  estabilidad: BP(fam)={bpS}, BP(col)={bpC} "
          f"({'ambos estables OK' if bpS==0 and bpC==0 else 'NO estable -> revisar'})")
    print(f"  ANCHO reticulo (fam-opt vs col-opt) = {gap}  "
          f"({100*gap/N:.4f}% de N, {100*gap/den:.4f}% de asignadas)")
    print(f"  -> {'TRIVIAL (estable unico)' if gap==0 else 'NO trivial'}")
    return mS, gap

# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("="*72 + "\nCARGA\n" + "="*72)
    D = cargar()
    print(f"N={D['N']}, M={D['M']}, cupos={sum(D['school_cap'].values())}, "
          f"sin_coord={int(np.isnan(D['fam_lat']).sum())}")
    print(f"mean_ingpc={D['mean_inc']:,.0f} -> grupo_media={D['mean_group']}")
    print("SISBEN expandido:", dict(sorted(Counter(D['fam_sisben'].astype(int)).items())))
    print(f"[carga: {time.time()-t0:.1f}s]")

    LOT = np.arange(D["N"], dtype=np.int64)   # misma loteria para ambas corridas

    print("\n" + "="*72)
    print("(1) PRIORIDAD = DISTANCIA  (control: debe reproducir el ~18 de 09e)")
    print("="*72)
    mS_dist, gap_dist = run_pair(D, LOT, "distancia | loteria=indice")

    print("\n" + "="*72)
    print("(2) PRIORIDAD = SED  (SISBEN primario, distancia secundaria)")
    print("="*72)
    mS_sed, gap_sed = run_pair(D, LOT, "SED=SISBEN+distancia | loteria=indice",
                               sisben=D["fam_sisben"])

    # --- efecto de CAMBIAR el sistema de prioridad (distancia -> SED) ---
    # NO es ancho del reticulo (eso es intra-sistema). Esto es entre-sistemas:
    # bajo escasez, cambiar la prioridad cambia QUIEN obtiene cupo.
    asg_D = set(i for i, x in enumerate(mS_dist) if x is not None)
    asg_S = set(i for i, x in enumerate(mS_sed)  if x is not None)
    inter = asg_D & asg_S
    cambian = sum(1 for i in inter if mS_dist[i] != mS_sed[i])
    print("\n" + "="*72)
    print(f"ANCHO del reticulo (intra-sistema):   distancia = {gap_dist}    SED = {gap_sed}")
    print("-"*72)
    print("EFECTO DE CAMBIAR LA PRIORIDAD (distancia -> SED), entre-sistemas:")
    print(f"  asignadas bajo distancia: {len(asg_D)}")
    print(f"  asignadas bajo SED:       {len(asg_S)}")
    print(f"  reciben cupo en AMBOS:    {len(inter)} ({100*len(inter)/len(asg_D):.0f}% de los asignados)")
    print(f"  pierden cupo al pasar a SED: {len(asg_D - asg_S)}")
    print(f"  ganan cupo al pasar a SED:   {len(asg_S - asg_D)}")
    print(f"  de los {len(inter)} en ambos, cambian de colegio: {cambian} ({100*cambian/max(len(inter),1):.0f}%)")
    print("-"*72)
    print("Lectura: el reticulo NO se abre bajo SED (se contrae, 18 -> 12). Ningun")
    print("mecanismo estable redistribuye el sesgo visual ni bajo prioridades SED.")
    print("El reshuffle masivo muestra que la palanca de politica es la PRIORIDAD")
    print("(quien tiene preferencia), no el MECANISMO (DA/SED/WP coinciden dado un")
    print("sistema de prioridad fijo). Refuerza M1.")
    print(f"[total: {time.time()-t0:.1f}s]")
