"""
05b_expandir_familias.py
========================
Expande las familias encuestadas usando el factor de expansión FEX_C.
Cada familia se replica round(FEX_C) veces para representar la población real.

Estrategia:
  - Cada réplica hereda todos los atributos de la familia original
    (UPZ, localidad, estrato, coordenadas) — no se mueve geográficamente.
  - Las distancias de cada réplica = distancias originales + ruido U(-0.1, 0.1) km
    clippeado a max(0, d) para evitar distancias negativas.
  - El ruido es equivalente a estar en una manzana adyacente (~±100m),
    garantizando que la réplica permanezca dentro de la misma UPZ y estrato.

Inputs:
  data/processed/familias_ubicadas.parquet       — familias simuladas con FEX_C
  data/processed/familias_distancias.parquet     — matriz de distancias (n_fam × n_col)

Outputs:
  data/processed/familias_expandidas.parquet     — familias expandidas (N_real × atributos)
  data/processed/distancias_expandidas.parquet   — matriz de distancias expandida (N_real × n_col)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Raiz del repositorio, derivada de la ubicacion de este script.
# Evita rutas absolutas y hace el pipeline reproducible en cualquier maquina.
_ROOT = Path(__file__).resolve().parent.parent


SEED = 42
NOISE_KM = 0.1   # ±100m de perturbación en distancias

FAM_PATH   = _ROOT / "data" / "processed" / "familias_ubicadas.parquet"
DIST_PATH  = _ROOT / "data" / "processed" / "familias_distancias.parquet"
FAM_OUT    = _ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_OUT   = _ROOT / "data" / "processed" / "distancias_expandidas.parquet"

rng = np.random.default_rng(SEED)

# Umbrales DANE 2024 — ingreso per cápita mensual (pesos corrientes)
SISBEN_UMBRALES = [227_220, 460_198, 897_987]

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Cargando familias_ubicadas.parquet...")
fam = pd.read_parquet(FAM_PATH)
print(f"  Familias originales: {len(fam):,}")

print("Cargando familias_distancias.parquet...")
dist = pd.read_parquet(DIST_PATH)
n_colegios = dist.shape[1]
print(f"  Dimensiones distancias: {dist.shape}")

# ── Calcular réplicas por familia ─────────────────────────────────────────────
fam["n_replicas"] = fam["FEX_C"].apply(lambda x: max(1, round(float(x))))
total_replicas = fam["n_replicas"].sum()
print(f"\n  Total réplicas (población representada): {total_replicas:,}")
print(f"  Factor expansión medio: {fam['FEX_C'].astype(float).mean():.1f}")

# ── Expandir familias ─────────────────────────────────────────────────────────
print("\nExpandiendo familias...")
fam_exp = fam.loc[fam.index.repeat(fam["n_replicas"])].copy()
fam_exp = fam_exp.drop(columns=["n_replicas"]).reset_index(drop=True)

# Índice original para asociar distancias
fam_exp["_idx_original"] = fam.index.repeat(fam["n_replicas"]).values

print(f"  Familias expandidas: {len(fam_exp):,}")

# ── Expandir distancias con ruido ─────────────────────────────────────────────
print("Expandiendo distancias con ruido U(-0.1, 0.1) km...")
dist_vals = dist.values.astype(np.float32)
col_names = dist.columns.tolist()

# Indexar distancias originales para cada réplica
dist_base = dist_vals[fam_exp["_idx_original"].values]  # (N_real, n_col)

# Agregar ruido y clippear a >= 0
noise = rng.uniform(-NOISE_KM, NOISE_KM, size=dist_base.shape).astype(np.float32)
dist_exp = np.clip(dist_base + noise, 0, None)

print(f"  Dimensiones distancias expandidas: {dist_exp.shape}")
print(f"  Min distancia: {dist_exp.min():.4f} km  |  Max: {dist_exp.max():.4f} km")

# ── Guardar ───────────────────────────────────────────────────────────────────
print("\nGuardando outputs...")
fam_exp = fam_exp.drop(columns=["_idx_original"])

# ── Calcular sisben_cat una sola vez ──────────────────────────────────────────
# cat 0=A (extrema), 1=B (moderada), 2=C (vulnerable), 3=D (no priorizado)
# Fallback: si N_ingpc nulo → E1→A, E2→B, E3→C, E4-6→D
ingpc   = pd.to_numeric(fam_exp.get("N_ingpc",      pd.Series(dtype=float)), errors="coerce").values
estrato = pd.to_numeric(fam_exp.get("estrato_real", pd.Series(dtype=float)), errors="coerce").fillna(3).clip(1, 6).values

fam_exp["sisben_cat"] = np.where(
    ~np.isnan(ingpc),
    np.where(ingpc < SISBEN_UMBRALES[0], 0,
    np.where(ingpc < SISBEN_UMBRALES[1], 1,
    np.where(ingpc < SISBEN_UMBRALES[2], 2, 3))),
    np.where(estrato <= 1, 0,
    np.where(estrato <= 2, 1,
    np.where(estrato <= 3, 2, 3)))
).astype(int)

labels = ["A (extrema)", "B (moderada)", "C (vulnerable)", "D (no priorizado)"]
for cat_i, label in enumerate(labels):
    n = (fam_exp["sisben_cat"] == cat_i).sum()
    print(f"  SISBEN {label}: {n:,} ({n/len(fam_exp)*100:.1f}%)")

fam_exp.to_parquet(FAM_OUT, index=False)
size_fam = FAM_OUT.stat().st_size / 1e6
print(f"  {FAM_OUT.name}  ({len(fam_exp):,} filas, {size_fam:.1f} MB)")

dist_exp_df = pd.DataFrame(dist_exp, columns=col_names)
dist_exp_df.to_parquet(DIST_OUT, index=False)
size_dist = DIST_OUT.stat().st_size / 1e6
print(f"  {DIST_OUT.name}  ({dist_exp_df.shape}, {size_dist:.1f} MB)")

print("\nDone.")
