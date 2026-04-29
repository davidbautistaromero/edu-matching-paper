"""
14_comparativa_mecanismos.py
============================
Recomputa metricas para los 4 mecanismos sobre el mismo universo y con la
misma quality_col, para que la tabla del paper sea comparable.

Inputs:
  data/results/matching_da.parquet       (David, DA puro)
  data/results/matching_bm.parquet       (David, Boston)
  data/results/matching_sed_lex.parquet  (Jhoan, SED lex)
  data/results/matching_sed_dist.parquet (Jhoan, DA-dist)

Output:
  reports/matching_comparativa_global.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from matching_utils import compute_metrics

FAM_P = ROOT / "data" / "processed" / "familias_expandidas.parquet"
COL_P = ROOT / "data" / "primary"   / "colegios_capacidad.parquet"
DIST_P = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
PREF_P = ROOT / "data" / "primary"   / "preferencias_familias.parquet"

print("Cargando datos base...")
fam  = pd.read_parquet(FAM_P).reset_index(drop=True)
col  = pd.read_parquet(COL_P)
col["id_establecimiento"] = col["id_establecimiento"].astype(str)
school_info = col.set_index("id_establecimiento")
school_cap  = school_info["capacidad"].to_dict()
dist = pd.read_parquet(DIST_P).reset_index(drop=True)
pref = pd.read_parquet(PREF_P).reset_index(drop=True)

# Universo comun: drop NaN estrato
mask = fam["estrato_real"].notna().values
fam  = fam.loc[mask].reset_index(drop=True)
dist = dist.loc[mask].reset_index(drop=True)
pref = pref.loc[mask].reset_index(drop=True)
print(f"  Universo comun: {len(fam):,} familias")

estrato_arr = fam["estrato_real"].astype(int).values
valid_schools = set(school_info.index)

dist_cols = [c for c in dist.columns if c in valid_schools]
dist_arr = dist[dist_cols].values.astype(np.float32)
school_col_idx = {sid: k for k, sid in enumerate(dist_cols)}

pref_cols = [c for c in pref.columns if c.startswith("pref_")]
pref_lists = [
    [s for s in row if isinstance(s, str) and s in valid_schools]
    for row in pref[pref_cols].astype(str).values
]

def priority_dist(i, sid):
    k = school_col_idx.get(sid)
    return np.inf if k is None else float(dist_arr[i, k])

# Mapeo de DIRECTORIO->indice posicional en fam
fam_idx = pd.Series(np.arange(len(fam)), index=fam["DIRECTORIO"].values)

mechanisms = {
    "BM":      ROOT / "data/results/matching_bm.parquet",
    "DA":      ROOT / "data/results/matching_da.parquet",
    "SED-lex": ROOT / "data/results/matching_sed_lex.parquet",
    "SED-dist":ROOT / "data/results/matching_sed_dist.parquet",
}

CAT_OFFSET = 1_000_000
categoria = (estrato_arr > 2).astype(int)

def priority_lex(i, sid):
    k = school_col_idx.get(sid)
    if k is None: return np.inf
    return float(categoria[i]) * CAT_OFFSET + float(dist_arr[i, k])

priority_by_mech = {
    "BM":       priority_dist,
    "DA":       priority_dist,
    "SED-lex":  priority_lex,
    "SED-dist": priority_dist,
}

rows = []
for name, p in mechanisms.items():
    pfn = priority_by_mech[name]
    print(f"\nProcesando {name}...")
    df = pd.read_parquet(p)
    if "DIRECTORIO" in df.columns:
        df = df.reset_index(drop=True)
    # Reproyectar al universo comun, mismo orden que fam
    if len(df) != len(fam):
        df = df[df["DIRECTORIO"].isin(fam["DIRECTORIO"].values)].reset_index(drop=True)
    # Asegurar mismo orden por DIRECTORIO posicional
    if not (df["DIRECTORIO"].values == fam["DIRECTORIO"].values).all():
        df = df.set_index("DIRECTORIO").loc[fam["DIRECTORIO"].values].reset_index()
    asgn = df["id_establecimiento"].astype("object").where(df["id_establecimiento"].notna(), None).tolist()
    # Convertir floats que vienen como str de NaN a None
    asgn = [None if (s is None or (isinstance(s, float) and np.isnan(s)) or str(s) in ("nan", "None", "")) else str(s) for s in asgn]

    met = compute_metrics(
        asgn, pref_lists, school_cap, pfn,
        school_info, quality_col="q_j", visual_col="sobre_demanda_j",
        estrato_arr=estrato_arr, label=name,
    )
    rows.append(met)

out = pd.DataFrame(rows)
OUT_P = ROOT / "reports" / "matching_comparativa_global.csv"
out.to_csv(OUT_P, index=False)
print(f"\nGuardado: {OUT_P}")
print("\n" + "="*100)
print("TABLA COMPARATIVA (mismo universo, q_j = Saber 11)")
print("="*100)
cols_show = ["condicion", "n_asignados", "eficiencia_q", "equidad_corr",
             "sesgo_visual", "rank_medio", "blocking_pairs"]
print(out[cols_show].to_string(index=False))
