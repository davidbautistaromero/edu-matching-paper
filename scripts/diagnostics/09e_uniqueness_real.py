# =============================================================================
#  09e_uniqueness_real.py
#  ¿El conjunto de matchings estables es único sobre los DATOS REALES de Bogotá?
#
#  Sube a escala real (537031 familias expandidas FEX_C, 416 colegios) el
#  criterio de retículo de 09d. Prioridad = distancia Haversine (la misma que
#  usa 07_da_mechanism.py para generar matching_da). El conjunto estable es un
#  retículo cuyos extremos son el matching óptimo para FAMILIAS (DA familia-
#  proponente, matching_utils.deferred_acceptance) y el óptimo para COLEGIOS
#  (college_da, abajo). Si coinciden, el matching estable es único.
#
#  Sutileza que obliga el diseño: hay ~40 agentes por DIRECTORIO compartiendo
#  coordenada exacta, luego distancia exacta, luego EMPATES masivos de prioridad.
#  Para que el criterio del retículo aplique (requiere prioridad ESTRICTA) ambas
#  DA rompen el empate con el MISMO orden: distancia, y luego una lotería fija.
#  Se corre bajo dos loterías para verificar que el ancho es robusto al sorteo.
#
#  Lectura: ancho ~0 bajo ambas loterías => el conjunto estable colapsa también
#  en el mercado real, así que todo mecanismo estable bajo prioridad-distancia
#  (DA, SED, y la WP-Rule de 09c) coincide. Corrobora M1: el sesgo vive en las
#  preferencias declaradas, no en el mecanismo. cross_lot mide cuántas familias
#  equidistantes cambian de cupo según el sorteo (racionamiento bajo escasez),
#  un efecto del desempate, NO evidencia sobre el sesgo en preferencias.
# =============================================================================
from pathlib import Path
import sys, time
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matching_utils as mu

BASE = str(Path(__file__).resolve().parents[2] / "data")
ARCHIVO_PREF  = f"{BASE}/primary/preferencias_familias.parquet"
ARCHIVO_CAP   = f"{BASE}/primary/colegios_capacidad.parquet"
ARCHIVO_COORD = f"{BASE}/processed/familias_ubicadas.parquet"
CAP_COL = "capacidad"

def snorm(s): return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
def norm_id(x):
    s = str(x).strip(); return s[:-2] if s.endswith(".0") else s
_P = 0.017453292519943295
def hav_vec(la1, lo1, la2, lo2):
    dlat=(la2-la1)*_P; dlon=(lo2-lo1)*_P
    a=np.sin(dlat/2)**2+np.cos(la1*_P)*np.cos(la2*_P)*np.sin(dlon/2)**2
    return 12742.0*np.arcsin(np.sqrt(np.clip(a,0.0,1.0)))

# ----------------------------------------------------------------- carga
t0=time.time(); print("="*72+"\nCARGA\n"+"="*72)
df_pref=pd.read_parquet(ARCHIVO_PREF)
pref_cols=[c for c in df_pref.columns if str(c).lower().startswith("pref_")]
directorios=(snorm(df_pref["DIRECTORIO"]).to_numpy() if "DIRECTORIO" in df_pref.columns
             else snorm(pd.Series(df_pref.index)).to_numpy())
N=len(df_pref)
raw=df_pref[pref_cols].to_numpy(dtype=object); del df_pref

df_cap=pd.read_parquet(ARCHIVO_CAP).drop_duplicates(subset=["id_establecimiento"])
df_cap=df_cap[df_cap[CAP_COL].notna()].copy()
df_cap["id_establecimiento"]=snorm(df_cap["id_establecimiento"])
school_ids=df_cap["id_establecimiento"].tolist()
sch_idx={s:k for k,s in enumerate(school_ids)}
school_cap={s:int(c) for s,c in zip(school_ids,df_cap[CAP_COL])}
sch_lat=df_cap["lat"].to_numpy(float); sch_lon=df_cap["lon"].to_numpy(float)
M=len(school_ids)

df_co=pd.read_parquet(ARCHIVO_COORD).copy()
df_co["DIRECTORIO"]=snorm(df_co["DIRECTORIO"])
df_co=df_co.drop_duplicates(subset=["DIRECTORIO"]).set_index("DIRECTORIO")
fam_lat=df_co["lat"].reindex(directorios).to_numpy(float)
fam_lon=df_co["lon"].reindex(directorios).to_numpy(float)
n_dir=pd.Index(directorios).nunique()
print(f"N={N}, DIRECTORIOs unicos={n_dir} (=> {N/n_dir:.1f} agentes/coordenada, "
      f"fuente de los empates), M={M}, cupos={sum(school_cap.values())}, "
      f"sin coord={int(np.isnan(fam_lat).sum())}")

# listas de preferencia con strings canonicos (ahorra RAM: M objetos compartidos)
canon={s:s for s in school_ids}
pref_lists=[]
for row in raw:
    pl=[]
    for x in row:
        if x is None: continue
        cs=canon.get(norm_id(x))
        if cs is not None: pl.append(cs)
    pref_lists.append(pl)
del raw
stu_rank=[{s:r for r,s in enumerate(pl)} for pl in pref_lists]

# candidatos por colegio (agentes que lo listan)
cand={s:[] for s in school_ids}
for i,pl in enumerate(pref_lists):
    for s in pl: cand[s].append(i)
cand={s:np.asarray(v,np.int64) for s,v in cand.items()}
print(f"[carga: {time.time()-t0:.1f}s]")

# ----------------------------------------------------------------- motores
def build_order(tiebreak):
    """Orden de cada colegio sobre sus candidatos: distancia, empate por tiebreak."""
    order={}
    for s in school_ids:
        fa=cand[s]
        if fa.size==0: order[s]=fa; continue
        k=sch_idx[s]; d=hav_vec(fam_lat[fa],fam_lon[fa],sch_lat[k],sch_lon[k])
        d=np.where(np.isnan(d),np.inf,d)
        order[s]=fa[np.lexsort((tiebreak[fa], d))]
    return order

def college_da(order):
    """DA colegio-proponente -> matching estable optimo para colegios."""
    held=[None]*N; cnt={s:0 for s in school_ids}; ptr={s:0 for s in school_ids}
    inq=set(school_ids); q=deque(school_ids)
    while q:
        s=q.popleft(); inq.discard(s)
        O=order[s]; cap=school_cap[s]; L=O.shape[0]
        while cnt[s]<cap and ptr[s]<L:
            i=int(O[ptr[s]]); ptr[s]+=1; cur=held[i]
            if cur is None: held[i]=s; cnt[s]+=1
            elif stu_rank[i][s]<stu_rank[i][cur]:
                held[i]=s; cnt[s]+=1; cnt[cur]-=1
                if cur not in inq: inq.add(cur); q.append(cur)
    return held

def run_pair(tiebreak, label):
    """Corre ambas DA bajo el MISMO orden estricto y mide el ancho del reticulo."""
    order=build_order(tiebreak)
    BIG=np.int32(N+1); tab={}
    for s in school_ids:
        r=np.full(N,BIG,np.int32); o=order[s]
        if o.size: r[o]=np.arange(o.size,dtype=np.int32)
        tab[s]=r
    def prio(i,sid): return float(tab[sid][i])   # rank estricto: sin empates
    t1=time.time(); mS=mu.deferred_acceptance(pref_lists,school_cap,prio); t2=time.time()
    mC=college_da(order); t3=time.time()
    aS=sum(x is not None for x in mS); aC=sum(x is not None for x in mC)
    gap=sum(1 for a,b in zip(mS,mC) if a!=b); den=aS if aS else N
    print(f"\n[{label}] DA-familia {t2-t1:.1f}s, DA-colegio {t3-t2:.1f}s")
    print(f"  asignadas: familia-opt={aS}, colegio-opt={aC} "
          f"({'rural-hospitals OK' if aS==aC else 'DIFIEREN, revisar'})")
    print(f"  brecha familia-opt vs colegio-opt = {gap}  "
          f"({100*gap/N:.4f}% de N, {100*gap/den:.4f}% de las asignadas)")
    print(f"  reticulo {'TRIVIAL (estable unico)' if gap==0 else 'NO trivial, ancho='+str(gap)}")
    del tab
    return mS, gap

if __name__ == "__main__":
    print("\n"+"="*72+"\nRETICULO CON DESEMPATE CONSISTENTE (distancia, luego loteria)\n"+"="*72)
    mS1, gap1 = run_pair(np.arange(N,dtype=np.int64), "loteria=indice")
    mS2, gap2 = run_pair(np.random.RandomState(0).permutation(N).astype(np.int64),
                         "loteria=permutacion(seed0)")

    a1=sum(x is not None for x in mS1)
    cross_lot=sum(1 for a,b in zip(mS1,mS2) if a!=b)
    print(f"\nsensibilidad a la loteria (familia-opt L1 vs L2): {cross_lot} familias "
          f"({100*cross_lot/max(a1,1):.2f}% de asignadas) cambian de colegio segun el sorteo")
    print("-"*60)
    print("Lectura: ancho ~0 bajo ambas loterias => el conjunto estable colapsa")
    print("tambien en el mercado real de Bogota bajo prioridad-distancia, asi que")
    print("DA, SED y la WP-Rule (09c) coinciden. Corrobora M1: la palanca son las")
    print("preferencias, no el mecanismo. cross_lot = racionamiento por sorteo entre")
    print("familias equidistantes bajo escasez (efecto del desempate, no del sesgo).")
    print(f"\n[total: {time.time()-t0:.1f}s]")
