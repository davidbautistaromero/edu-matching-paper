"""
09b_matching_sinteticos_nmf.py
==============================
Versión paralela de 09_matching_sinteticos.py que corre los 4 condiciones de
matching (BM-bias, BM-true, DA-bias, DA-true) sobre los datos sintéticos
generados con v_j construido desde los tópicos NMF reales (08b_datos_sinteticos_nmf.py).

La lógica es idéntica a 09_matching_sinteticos.py. Sólo cambian los archivos
de entrada (prefijo sinteticos_nmf_) y los archivos de salida (prefijo nmf_).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from matching_utils import (
    boston_mechanism,
    compute_metrics,
    deferred_acceptance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "results"
REP_DIR = ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

log.info("=" * 60)
log.info("Paso 1 — Cargando datos sintéticos NMF...")

pref_bias = pd.read_parquet(ROOT / "data/primary/sinteticos_nmf_preferencias.parquet")
pref_true = pd.read_parquet(ROOT / "data/primary/sinteticos_nmf_pref_sin_sesgo.parquet")
col_df    = pd.read_parquet(ROOT / "data/primary/sinteticos_nmf_colegios.parquet")
est_df    = pd.read_parquet(ROOT / "data/primary/sinteticos_nmf_estudiantes.parquet")

with open(REP_DIR / "sinteticos_nmf_calibracion.json", encoding="utf-8") as f:
    cal = json.load(f)

N = len(est_df)
M = len(col_df)

col_df     = col_df.set_index("id_establecimiento")
school_ids = col_df.index.tolist()
estrato    = est_df["estrato"].values

school_cap = col_df["capacidad_sintetica"].to_dict()

log.info(f"  N={N} estudiantes | M={M} colegios | "
         f"cap_total={sum(school_cap.values())} | alpha_hat={cal['alpha_hat']:.5f}")

log.info("Paso 2 — Generando lotería de prioridad (seed=42)...")

rng_lot = np.random.default_rng(SEED)
lottery: dict[str, dict[int, int]] = {}
for sid in school_ids:
    perm = rng_lot.permutation(N)
    lottery[sid] = {int(student_idx): int(rank) for rank, student_idx in enumerate(perm)}

def priority_fn(student_idx: int, school_id: str) -> int:
    return lottery[school_id][student_idx]

log.info("Paso 3 — Preparando listas de preferencias...")

valid_schools = set(school_ids)

def to_pref_lists(pref_df: pd.DataFrame) -> list[list[str]]:
    pref_cols = [c for c in pref_df.columns if c.startswith("pref_")]
    return [
        [s for s in row.values if s in valid_schools]
        for _, row in pref_df[pref_cols].iterrows()
    ]

pref_lists_bias = to_pref_lists(pref_bias)
pref_lists_true = to_pref_lists(pref_true)

CONDITIONS = [
    ("bm_bias", boston_mechanism,    pref_lists_bias, "BM-bias-NMF"),
    ("bm_true", boston_mechanism,    pref_lists_true, "BM-true-NMF"),
    ("da_bias", deferred_acceptance, pref_lists_bias, "DA-bias-NMF"),
    ("da_true", deferred_acceptance, pref_lists_true, "DA-true-NMF"),
]

log.info("=" * 60)
log.info("Paso 4 — Ejecutando 4 condiciones de matching...")

results     = []
assignments = {}

for key, fn, plists, label in CONDITIONS:
    log.info(f"  Corriendo {label}...")
    asgn = fn(plists, school_cap, priority_fn)
    met = compute_metrics(
        assignment=asgn,
        pref_lists=plists,
        school_cap=school_cap,
        priority_fn=priority_fn,
        school_info=col_df,
        quality_col="q_j_std",
        visual_col="v_j",
        estrato_arr=estrato,
        label=label,
    )
    results.append(met)
    assignments[key] = asgn

log.info("=" * 60)
log.info("Paso 5 — Guardando resultados...")

for key, asgn in assignments.items():
    df = pd.DataFrame({
        "id_estudiante"      : est_df["id_estudiante"],
        "estrato"            : estrato,
        "id_establecimiento" : asgn,
    }).merge(
        col_df[["q_j", "q_j_std", "v_j", "sobre_demanda_j"]].reset_index(),
        on="id_establecimiento",
        how="left",
    )
    fname = f"sinteticos_nmf_{key}.parquet"
    df.to_parquet(OUT_DIR / fname, index=False)
    log.info(f"  {fname}")

comp_df = pd.DataFrame(results)
comp_df.to_csv(REP_DIR / "sinteticos_nmf_matching_comparison.csv", index=False)
log.info("  sinteticos_nmf_matching_comparison.csv")

log.info("=" * 60)
log.info("TABLA COMPARATIVA — Datos sintéticos con v_j post-Lasso (NMF)")
log.info("=" * 60)
cols_show = ["condicion", "n_asignados", "eficiencia_q", "equidad_corr",
             "sesgo_visual", "rank_medio", "blocking_pairs"]
print(comp_df[cols_show].to_string(index=False))
log.info("Done.")
