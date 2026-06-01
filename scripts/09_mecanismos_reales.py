"""
09_mecanismos_reales.py
=======================
Compara mecanismos de asignacion sobre datos reales de Bogota.

Este script usa las familias de primer ingreso (~99K) y los cupos reales
(~120K). Las restricciones de localidad entran a traves de las listas de
preferencias generadas previamente: `preferencias_familias.parquet` ya contiene
el choice set local (misma localidad, con la excepcion de La Candelaria).

Mecanismos:
  - BM: Boston con prioridad geografica.
  - DA: Deferred Acceptance con prioridad geografica.
  - SED-lex: DA con prioridad lexicografica SISBEN + distancia.
  - WP-priority: DA con la prioridad ingreso-distancia-visual calibrada por
    07_WP_rule.py. No reentrena WP ni resuelve el LP completo a escala real.

Inputs:
  data/primary/preferencias_familias.parquet
  data/processed/familias_expandidas.parquet
  data/processed/distancias_expandidas.parquet
  data/primary/colegios_capacidad.parquet
  data/images/clip/gsv_clip_establecimiento.parquet
  reports/wp_calibracion.json

Outputs:
  data/results/matching_real_{bm,da,sed_lex,wp_priority}.parquet
  reports/tables/mecanismos_reales_results.csv
  reports/figures/matching/mecanismos_reales_comparison.png
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matching_utils import (
    boston_mechanism,
    count_blocking_pairs,
    deferred_acceptance,
    mean_rank_obtained,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PREF_P = ROOT / "data" / "primary" / "preferencias_familias.parquet"
FAM_P = ROOT / "data" / "processed" / "familias_expandidas.parquet"
DIST_P = ROOT / "data" / "processed" / "distancias_expandidas.parquet"
CAP_P = ROOT / "data" / "primary" / "colegios_capacidad.parquet"
CLIP_P = ROOT / "data" / "images" / "clip" / "gsv_clip_establecimiento.parquet"
CAL_P = ROOT / "reports" / "wp_calibracion.json"

OUT_RESULTS = ROOT / "data" / "results"
OUT_TABLES = ROOT / "reports" / "tables"
OUT_FIGS = ROOT / "reports" / "figures" / "matching"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

CAPACITY_MIN = 5
CAT_OFFSET = 1_000_000.0
PREF_TOP_K = 20


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _clean_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _clean_id_value(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def load_wp_calibration() -> dict:
    with open(CAL_P, "r", encoding="utf-8") as f:
        cal = json.load(f)
    if cal.get("priority_model") != "income_distance_visual":
        raise ValueError(
            "wp_calibracion.json no corresponde al modelo income_distance_visual. "
            "Corre 07_WP_rule.py antes de usar este script."
        )
    if "theta_priority" not in cal:
        raise ValueError("wp_calibracion.json no contiene theta_priority.")
    return cal


def load_real_market() -> dict:
    log.info("=" * 70)
    log.info("PASO 1 -- Cargando datos reales")
    log.info("=" * 70)

    pref_df = pd.read_parquet(PREF_P)
    fam_df = pd.read_parquet(FAM_P)
    dist_df = pd.read_parquet(DIST_P)
    cap_df = pd.read_parquet(CAP_P)
    clip_df = pd.read_parquet(CLIP_P)

    cap_df["id_establecimiento"] = _clean_id_series(cap_df["id_establecimiento"])
    clip_df["id_establecimiento"] = _clean_id_series(clip_df["id_establecimiento"])
    dist_df.columns = [_clean_id_value(c) for c in dist_df.columns]

    n0 = min(len(pref_df), len(fam_df), len(dist_df))
    pref_df = pref_df.iloc[:n0].reset_index(drop=True)
    fam_df = fam_df.iloc[:n0].reset_index(drop=True)
    dist_df = dist_df.iloc[:n0].reset_index(drop=True)

    mask = (pd.to_numeric(fam_df["n_hijos_ingreso"], errors="coerce") > 0).to_numpy()
    pref_df = pref_df.loc[mask].reset_index(drop=True)
    fam_df = fam_df.loc[mask].reset_index(drop=True)
    dist_df = dist_df.loc[mask].reset_index(drop=True)

    log.info(f"  Familias totales alineadas : {n0:,}")
    log.info(f"  Familias primer ingreso    : {len(fam_df):,}")

    required_cols = [
        "capacidad", "a_j", "a_j_visual", "q_j", "sobre_demanda_j",
        "nombre_localidad",
    ]
    missing = [c for c in required_cols if c not in cap_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en colegios_capacidad.parquet: {missing}")

    school_info = cap_df.set_index("id_establecimiento")[required_cols].copy()
    school_info = school_info.merge(
        clip_df[["id_establecimiento", "seguridad_percibida"]].drop_duplicates(
            "id_establecimiento"
        ).set_index("id_establecimiento"),
        left_index=True,
        right_index=True,
        how="left",
    )
    school_info["capacidad"] = (
        pd.to_numeric(school_info["capacidad"], errors="coerce")
        .fillna(0)
        .clip(lower=CAPACITY_MIN)
        .astype(int)
    )

    pref_cols = [f"pref_{k}" for k in range(1, PREF_TOP_K + 1) if f"pref_{k}" in pref_df.columns]
    schools_in_prefs = {
        _clean_id_value(s)
        for col in pref_cols
        for s in pref_df[col].dropna().astype(str)
    }
    schools_in_dist = set(dist_df.columns.astype(str))
    schools_in_info = set(school_info.index.astype(str))
    schools_in_clip = set(
        school_info.index[school_info["seguridad_percibida"].notna()].astype(str)
    )
    valid_schools = schools_in_prefs & schools_in_dist & schools_in_info & schools_in_clip
    if not valid_schools:
        raise ValueError("No hay colegios validos en la interseccion prefs/distancias/capacidad.")

    dist_cols = [c for c in dist_df.columns if c in valid_schools]
    dist_df = dist_df[dist_cols]
    dist_matrix = dist_df.to_numpy(dtype=np.float32)
    school_col_idx = {sid: k for k, sid in enumerate(dist_cols)}
    school_info = school_info.loc[dist_cols].copy()
    school_cap = school_info["capacidad"].to_dict()

    pref_lists = []
    for _, row in pref_df[pref_cols].iterrows():
        prefs = []
        for x in row.values:
            if pd.isna(x):
                continue
            sid = _clean_id_value(x)
            if sid in school_col_idx:
                prefs.append(sid)
        pref_lists.append(prefs)

    estrato = (
        pd.to_numeric(fam_df["estrato_real"], errors="coerce")
        .fillna(3)
        .clip(1, 6)
        .astype(int)
        .to_numpy()
    )
    sisben = (
        pd.to_numeric(fam_df["sisben_cat"], errors="coerce")
        .fillna(3)
        .clip(0, 3)
        .astype(int)
        .to_numpy()
        if "sisben_cat" in fam_df.columns
        else np.where(estrato <= 1, 0, np.where(estrato <= 2, 1, np.where(estrato <= 3, 2, 3)))
    )
    ingreso = pd.to_numeric(fam_df["N_ingpc"], errors="coerce").to_numpy(float)
    ingreso = np.where(np.isnan(ingreso) | (ingreso <= 0), np.nanmean(ingreso), ingreso)

    log.info(f"  Colegios validos          : {len(dist_cols):,}")
    log.info(f"  Capacidad total           : {sum(school_cap.values()):,}")
    log.info(f"  Ratio cupos/familias      : {sum(school_cap.values()) / len(fam_df):.2f}x")
    log.info(f"  Longitud media prefs      : {np.mean([len(p) for p in pref_lists]):.1f}")

    return {
        "fam_df": fam_df,
        "school_info": school_info,
        "school_cap": school_cap,
        "pref_lists": pref_lists,
        "dist_matrix": dist_matrix,
        "school_col_idx": school_col_idx,
        "estrato": estrato,
        "sisben": sisben,
        "ingreso": ingreso,
    }


def make_priorities(D: dict, theta_priority: float) -> dict:
    dist_matrix = D["dist_matrix"]
    school_col_idx = D["school_col_idx"]
    sisben = D["sisben"]
    ingreso_z = _zscore(D["ingreso"])
    dist_z = _zscore(dist_matrix)
    visual_z_by_col = _zscore(
        pd.to_numeric(D["school_info"]["seguridad_percibida"], errors="coerce").to_numpy(float)
    )

    def priority_dist(i: int, sid: str) -> float:
        col = school_col_idx.get(sid)
        return float(dist_matrix[i, col]) if col is not None else np.inf

    def priority_sed_lex(i: int, sid: str) -> float:
        col = school_col_idx.get(sid)
        if col is None:
            return np.inf
        return float(sisben[i]) * CAT_OFFSET + float(dist_matrix[i, col])

    def priority_wp(i: int, sid: str) -> float:
        """Lower is better. This is -P^theta_ij from 07_WP_rule.py."""
        col = school_col_idx.get(sid)
        if col is None:
            return np.inf
        return float(
            ingreso_z[i]
            + dist_z[i, col]
            + theta_priority * visual_z_by_col[col] * ingreso_z[i]
        )

    return {
        "BM": priority_dist,
        "DA": priority_dist,
        "SED-lex": priority_sed_lex,
        "WP-priority": priority_wp,
    }


def assigned_values(assignment: list[str | None], school_info: pd.DataFrame, col: str) -> np.ndarray:
    return np.array([
        school_info.loc[sid, col] if sid is not None and sid in school_info.index else np.nan
        for sid in assignment
    ], dtype=float)


def mechanism_metrics(
    label: str,
    assignment: list[str | None],
    priority_fn,
    D: dict,
) -> dict:
    matched = np.array([sid is not None for sid in assignment])
    q = assigned_values(assignment, D["school_info"], "q_j")
    v = assigned_values(assignment, D["school_info"], "seguridad_percibida")
    inc = D["ingreso"]
    sisben = D["sisben"]

    row = {
        "mecanismo": label,
        "n_familias": len(assignment),
        "n_asignados": int(matched.sum()),
        "n_sin_asignar": int((~matched).sum()),
        "rechazo_total": float((~matched).mean()),
        "corr_ingreso_visual": _safe_corr(inc[matched], v[matched]),
        "corr_ingreso_calidad": _safe_corr(inc[matched], q[matched]),
        "q_medio": float(np.nanmean(q[matched])),
        "rank_medio": mean_rank_obtained(assignment, D["pref_lists"]),
        "blocking_pairs": count_blocking_pairs(
            assignment, D["pref_lists"], D["school_cap"], priority_fn
        ),
    }
    for cat, name in [(0, "A"), (1, "B"), (2, "C"), (3, "D")]:
        mask = sisben == cat
        row[f"n_sisben_{name}"] = int(mask.sum())
        row[f"acceso_sisben_{name}"] = float(matched[mask].mean()) if mask.sum() else np.nan
    return row


def save_matching(label: str, assignment: list[str | None], D: dict) -> None:
    fname = {
        "BM": "matching_real_bm.parquet",
        "DA": "matching_real_da.parquet",
        "SED-lex": "matching_real_sed_lex.parquet",
        "WP-priority": "matching_real_wp_priority.parquet",
    }[label]
    fam = D["fam_df"]
    out = pd.DataFrame({
        "DIRECTORIO": fam["DIRECTORIO"].values,
        "COD_LOCALIDAD": fam["COD_LOCALIDAD"].values if "COD_LOCALIDAD" in fam.columns else np.nan,
        "estrato_real": D["estrato"],
        "sisben_cat": D["sisben"],
        "N_ingpc": D["ingreso"],
        "id_establecimiento": assignment,
    }).merge(
        D["school_info"][[
            "a_j", "a_j_visual", "seguridad_percibida", "q_j",
            "sobre_demanda_j", "nombre_localidad",
        ]]
        .reset_index(),
        on="id_establecimiento",
        how="left",
    )
    out.to_parquet(OUT_RESULTS / fname, index=False)
    log.info(f"  Saved: {OUT_RESULTS / fname}")


def plot_results(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]

    panels = [
        ("corr_ingreso_visual", "corr(ingreso, visual)"),
        ("corr_ingreso_calidad", "corr(ingreso, calidad)"),
        ("rank_medio", "Rank medio"),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.bar(results["mecanismo"], results[col], color=colors[:len(results)])
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Mecanismos sobre datos reales de Bogota", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_FIGS / "mecanismos_reales_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")


def main() -> None:
    t0 = time.time()
    cal = load_wp_calibration()
    theta = float(cal["theta_priority"])
    w = np.asarray(cal.get("w_learned", []), dtype=float)
    log.info(f"WP calibration: theta_priority={theta:+.4f}, w_learned={w}")

    D = load_real_market()
    priorities = make_priorities(D, theta)

    mechanisms = [
        ("BM", boston_mechanism, priorities["BM"]),
        ("DA", deferred_acceptance, priorities["DA"]),
        ("SED-lex", deferred_acceptance, priorities["SED-lex"]),
        ("WP-priority", deferred_acceptance, priorities["WP-priority"]),
    ]

    rows = []
    log.info("=" * 70)
    log.info("PASO 2 -- Ejecutando mecanismos")
    log.info("=" * 70)
    for label, engine, prio in mechanisms:
        t1 = time.time()
        log.info(f"  {label}: running...")
        assignment = engine(D["pref_lists"], D["school_cap"], prio)
        log.info(f"  {label}: done in {time.time() - t1:.1f}s")
        rows.append(mechanism_metrics(label, assignment, prio, D))
        save_matching(label, assignment, D)

    results = pd.DataFrame(rows)
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    results[numeric_cols] = results[numeric_cols].round(6)
    out_csv = OUT_TABLES / "mecanismos_reales_results.csv"
    results.to_csv(out_csv, index=False)
    plot_results(results)

    log.info("=" * 70)
    log.info("RESULTADOS -- datos reales")
    log.info("=" * 70)
    log.info("\n" + results[[
        "mecanismo", "n_asignados", "rechazo_total", "corr_ingreso_visual",
        "corr_ingreso_calidad", "q_medio", "rank_medio", "blocking_pairs",
    ]].to_string(index=False))
    log.info(f"Saved: {out_csv}")
    log.info(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
