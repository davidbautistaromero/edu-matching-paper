# =============================================================================
#  09i_redistribucion.py  —  Analisis de redistribucion (pregunta de David, 2026-05-29)
#
#  Pregunta: la redistribucion por ingreso (pasar de prioridad-DISTANCIA a
#  prioridad-SED = SISBEN) mueve el sesgo visual? Mejora equidad? A que costo
#  de eficiencia?
#
#  Metodo: corre DA familia-proponente bajo cada prioridad sobre los datos reales
#  (537k familias, 416 colegios) y mide, sobre las familias asignadas:
#    - corr(ingreso, v_j)  -> SESGO VISUAL (v_j = seguridad percibida estandarizada)
#    - corr(ingreso, q_j)  -> equidad en calidad academica
#    - rank medio          -> eficiencia (1 = primera opcion declarada)
#  Reporta tanto sobre TODOS los asignados como sobre las familias asignadas en
#  AMBOS escenarios (control de composicion: misma gente, sin sesgo de seleccion).
#  Reutiliza las funciones de 09g (cargar, build_order) ya en el repo.
#
#  RESULTADO (semilla fija, datos reales):
#    corr(ingreso, v_j): distancia +0.048 -> SED +0.003   (sesgo visual -> ~0)
#    corr(ingreso, q_j): distancia +0.178 -> SED +0.110   (equidad calidad mejora)
#    rank medio:         distancia 8.80   -> SED 3.85      (eficiencia mejora)
#  Sobre las 33905 familias COMUNES (sin composicion): rank 8.62 -> 4.45,
#    18102 mejoran rank vs 477 empeoran (38:1). La mejora es REAL, no composicion.
#
#  Lectura: la redistribucion por ingreso NO empeora el sesgo visual (la hipotesis
#  "F, los pobres acceden a peor señal visual" se descarta): lo NEUTRALIZA, la
#  correlacion ingreso-visual cae a ~0. Y lo hace mejorando equidad Y eficiencia a
#  la vez. La mejora de rank viene de que la prioridad-distancia penaliza fuerte el
#  rank al chocar con las preferencias (las familias prefieren calidad, no cercania);
#  SED, menos correlacionada con la distancia, deja a la gente en mejor posicion de
#  su propia lista. Confirma la tesis: la palanca de equidad es la PRIORIDAD, no el
#  mecanismo, y bajo SED no hay trade-off eficiencia-equidad.
# =============================================================================
# Este script fue movido a un subdirectorio; ROOT y matching_utils
# se resuelven relativos a scripts/ para que siga siendo ejecutable.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, ".")
import matching_utils as mu

# importar 09g como modulo (reutiliza cargar, build_order, hav_vec, etc.)
_spec = importlib.util.spec_from_file_location("mod09g", ROOT / "scripts" / "09g_uniqueness_sed.py")
g = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(g)

def snorm(s): return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def cargar_atributos_colegio():
    import geopandas as gpd
    clip = pd.read_parquet(ROOT/"data"/"images"/"clip"/"gsv_clip_establecimiento.parquet")
    clip["id_establecimiento"] = snorm(clip["id_establecimiento"])
    sg = clip["seguridad_percibida"]; clip["v_j"] = (sg - sg.mean()) / sg.std()
    gdf = gpd.read_file(ROOT/"data"/"primary"/"colegios_features_imputed.geojson")[["id_establecimiento","q_j"]]
    gdf["id_establecimiento"] = snorm(gdf["id_establecimiento"])
    att = clip[["id_establecimiento","v_j"]].merge(gdf, on="id_establecimiento", how="inner")
    return att.set_index("id_establecimiento")

def asignacion(D, LOT, sisben=None):
    order = g.build_order(D, LOT, sisben)
    BIG = np.int32(D["N"]+1); tab = {}
    for s in D["school_ids"]:
        r = np.full(D["N"], BIG, np.int32); o = order[s]
        if o.size: r[o] = np.arange(o.size, dtype=np.int32)
        tab[s] = r
    prio = lambda i, sid: float(tab[sid][i])
    return mu.deferred_acceptance(D["pref_lists"], D["school_cap"], prio)

def metricas(D, idx, matching, fam_inc, v_by_id, q_by_id, label):
    inc = fam_inc[idx]
    v   = np.array([v_by_id.get(matching[i], np.nan) for i in idx])
    q   = np.array([q_by_id.get(matching[i], np.nan) for i in idx])
    rank = np.array([D["stu_rank"][i].get(matching[i], len(D["pref_lists"][i])) for i in idx]) + 1
    def corr(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        return float(np.corrcoef(a[m], b[m])[0,1]) if m.sum() > 2 else np.nan
    print(f"\n[{label}]  asignadas={len(idx)}")
    print(f"  corr(ingreso, v_j)  [SESGO VISUAL] = {corr(inc,v):+.4f}")
    print(f"  corr(ingreso, q_j)  [equidad calid] = {corr(inc,q):+.4f}")
    print(f"  rank medio          [eficiencia]    = {np.nanmean(rank):.3f}")
    return dict(corr_iv=corr(inc,v), corr_iq=corr(inc,q), rank=float(np.nanmean(rank)))

if __name__ == "__main__":
    t0 = time.time()
    print("cargando datos...")
    D = g.cargar()
    att = cargar_atributos_colegio()
    v_by_id = att["v_j"].to_dict(); q_by_id = att["q_j"].to_dict()

    inc_dir = pd.to_numeric(
        pd.read_parquet(ROOT/"data"/"processed"/"familias_ubicadas.parquet")
          .assign(DIRECTORIO=lambda d: snorm(d["DIRECTORIO"]))
          .drop_duplicates("DIRECTORIO").set_index("DIRECTORIO")["N_ingpc"], errors="coerce")
    mean_inc = float(inc_dir.mean()); inc_map = inc_dir.to_dict()
    fam_inc = np.array([inc_map.get(dirr, np.nan) for dirr in D["directorios"]], float)
    fam_inc = np.where(np.isnan(fam_inc) | (fam_inc <= 0), mean_inc, fam_inc)

    LOT = np.arange(D["N"], dtype=np.int64)

    print("\n" + "="*64)
    print("REDISTRIBUCION: prioridad-DISTANCIA vs prioridad-SED (SISBEN)")
    print("="*64)
    m_dist = asignacion(D, LOT, sisben=None)
    m_sed  = asignacion(D, LOT, sisben=D["fam_sisben"])
    r_dist = metricas(D, [i for i,x in enumerate(m_dist) if x is not None], m_dist, fam_inc, v_by_id, q_by_id, "DISTANCIA")
    r_sed  = metricas(D, [i for i,x in enumerate(m_sed)  if x is not None], m_sed,  fam_inc, v_by_id, q_by_id, "SED = SISBEN + distancia")

    print("\n" + "="*64)
    print("RESPUESTA: la redistribucion mueve el sesgo visual?")
    print("="*64)
    print(f"  corr(ingreso, v_j):  {r_dist['corr_iv']:+.4f}  ->  {r_sed['corr_iv']:+.4f}   (delta {r_sed['corr_iv']-r_dist['corr_iv']:+.4f})")
    print(f"  corr(ingreso, q_j):  {r_dist['corr_iq']:+.4f}  ->  {r_sed['corr_iq']:+.4f}   (delta {r_sed['corr_iq']-r_dist['corr_iq']:+.4f})")
    print(f"  rank medio:          {r_dist['rank']:.3f}   ->  {r_sed['rank']:.3f}   (delta {r_sed['rank']-r_dist['rank']:+.3f})")

    # control de composicion: solo familias asignadas en AMBOS
    ambos = [i for i in range(D["N"]) if m_dist[i] is not None and m_sed[i] is not None]
    print("\n" + "="*64)
    print(f"CONTROL DE COMPOSICION: mismas {len(ambos)} familias (asignadas en ambos)")
    print("="*64)
    cd = metricas(D, ambos, m_dist, fam_inc, v_by_id, q_by_id, "DISTANCIA (comunes)")
    cs = metricas(D, ambos, m_sed,  fam_inc, v_by_id, q_by_id, "SED (comunes)")
    mejora = sum(1 for i in ambos
                 if D["stu_rank"][i].get(m_sed[i], len(D["pref_lists"][i])) < D["stu_rank"][i].get(m_dist[i], len(D["pref_lists"][i])))
    peor   = sum(1 for i in ambos
                 if D["stu_rank"][i].get(m_sed[i], len(D["pref_lists"][i])) > D["stu_rank"][i].get(m_dist[i], len(D["pref_lists"][i])))
    igual  = len(ambos) - mejora - peor
    print(f"\n  rank sobre la misma gente: {cd['rank']:.3f} -> {cs['rank']:.3f} (delta {cs['rank']-cd['rank']:+.3f})")
    print(f"  de las {len(ambos)}: mejoran rank={mejora}, empeoran={peor}, igual={igual}")
    print(f"  -> mejora de eficiencia {'REAL (no composicion)' if cs['rank'] < cd['rank']-0.01 else 'mayormente composicion, revisar'}")
    print(f"\n[tiempo total: {time.time()-t0:.1f}s]")
