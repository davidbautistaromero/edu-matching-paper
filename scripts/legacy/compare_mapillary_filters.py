#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_mapillary_filters.py
============================
Aplica cinco configuraciones de filtros de forma progresiva sobre
mapillary_catalog.csv, descarga las mismas escuelas de referencia en
todos los configs y genera grillas comparativas para evaluar visualmente
la calidad de cada estrategia.

Las escuelas de referencia se eligen como la intersección de colegios
cubiertos por TODOS los configs, garantizando comparación justa.

Todos los configs aplican siempre (no son comparables):
  · Solo imágenes de tipo perspective (excluye fisheye, equirectangular, spherical)
  · Sin panorámicas (is_pano = False)
  · Fecha ≥ FECHA_DESDE

Configuraciones comparables (cada una suma un filtro al anterior):
  0_baseline   Dedup por secuencia
  1_distancia  + distancia ≤ MAX_DIST_M metros
  2_heading    + foto apunta hacia el colegio (±MAX_ANGULO_DIFF°)
  3_sectores   + dedup por sector angular (reemplaza dedup por secuencia)
  4_techo      + máximo MAX_IMGS_POR_COLEGIO imágenes por colegio

SALIDAS
-------
  data/images/mapillary/comparacion_filtros/
  ├── fotos/
  │   ├── 0_baseline/   … 4_techo/    fotos descargadas por config
  ├── grid_<config>.png                escuelas ref × imgs en esa config
  ├── grid_cruzado.png                 escuelas ref × configs (1 img/celda)
  ├── comparacion_metricas.png
  └── comparacion_tabla.csv

USO
---
  python scripts/compare_mapillary_filters.py

  Ajusta PARÁMETROS. Con FORZAR_DESCARGA = False los archivos ya
  descargados se reutilizan (re-runs rápidos).

  NOTA: para que el filtro camera_type funcione el catálogo debe haberse
  generado con la versión actualizada de analyze_mapillary_colegios.py
  (que incluye camera_types=perspective en la query a la API y el campo
  camera_type en el CSV). Si el catálogo es anterior, el filtro se omite
  automáticamente sin romper el script.

REQUISITOS
  pip install aiohttp matplotlib pandas numpy
"""

# ---------------------------------------------------------------------------
# Stdlib
# ---------------------------------------------------------------------------
import asyncio
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Terceros
# ---------------------------------------------------------------------------
import aiohttp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# =============================================================================
# PARÁMETROS — edita solo esta sección
# =============================================================================

_ROOT         = Path(__file__).resolve().parents[2]
RUTA_CATALOGO = _ROOT / "data" / "images" / "mapillary" / "mapillary_catalog.csv"
RUTA_SALIDA   = _ROOT / "data" / "images" / "mapillary" / "comparacion_filtros"

# Total de colegios del dataset (para calcular % cobertura).
N_COLEGIOS_TOTAL = 407

# ── Escuelas de referencia ───────────────────────────────────────────────────
# Colegios usados para la comparación. Se eligen de la intersección de
# colegios cubiertos por TODOS los configs (comparación justa).
N_ESCUELAS_REF     = 6       # filas en las grillas
N_IMGS_POR_ESCUELA = 3       # imágenes por escuela en la grilla por config
RANDOM_SEED        = 42

# ── Descarga ─────────────────────────────────────────────────────────────────
FORZAR_DESCARGA = False      # True → re-descarga aunque los archivos existan
MAX_CONCURRENT  = 10

# ── Filtros de calidad (aplican en TODOS los configs) ────────────────────────
FECHA_DESDE = "2021-01-01"   # "" o None para desactivar

# ── Config 1 ─────────────────────────────────────────────────────────────────
MAX_DIST_M = 50              # metros

# ── Config 2 ─────────────────────────────────────────────────────────────────
MAX_ANGULO_DIFF = 60         # grados ± tolerancia heading foto→colegio

# ── Config 3 ─────────────────────────────────────────────────────────────────
N_SECTORES = 4               # 4 = cuadrantes 90°; 8 = octantes 45°

# ── Config 4 ─────────────────────────────────────────────────────────────────
MAX_IMGS_POR_COLEGIO = 4

# =============================================================================


# ---------------------------------------------------------------------------
# Helpers geométricos
# ---------------------------------------------------------------------------

def _bearing(lat1, lon1, lat2, lon2) -> float:
    dlon = math.radians(lon2 - lon1)
    r1, r2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(r2)
    y = math.cos(r1) * math.sin(r2) - math.sin(r1) * math.cos(r2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _diff_ang(a1, a2) -> float:
    d = abs(a1 - a2) % 360
    return d if d <= 180 else 360 - d


# ---------------------------------------------------------------------------
# Funciones de filtro atómicas
# ---------------------------------------------------------------------------

def f_pano_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_pano"] = df["is_pano"].astype(str).str.strip().str.lower() == "true"
    return df


def f_fecha(df):
    return df[df["fecha"] >= FECHA_DESDE].copy() if FECHA_DESDE else df.copy()


def f_sin_pano(df):
    return df[~df["is_pano"]].copy()


def f_solo_perspectiva(df):
    # Si el catálogo fue descargado sin camera_type (versión anterior del script),
    # no se puede filtrar y se devuelven todas las filas sin modificar.
    if "camera_type" not in df.columns:
        return df.copy()
    return df[df["camera_type"].astype(str).str.lower() == "perspective"].copy()


def f_dedup_seq(df):
    return (df.sort_values("distancia_m", na_position="last")
              .drop_duplicates(subset=["dane_est", "sequence"], keep="first")
              .reset_index(drop=True))


def f_dist(df):
    return df[df["distancia_m"] <= MAX_DIST_M].copy()


def f_heading(df):
    df = df.dropna(subset=["compass_angle", "lat_img", "lon_img",
                            "lat_colegio", "lon_colegio"]).copy()
    bear = df.apply(
        lambda r: _bearing(r.lat_img, r.lon_img, r.lat_colegio, r.lon_colegio), axis=1)
    diff = df.apply(lambda r: _diff_ang(r.compass_angle, bear[r.name]), axis=1)
    return df[diff <= MAX_ANGULO_DIFF].copy()


def f_dedup_sector(df):
    df = df.dropna(subset=["compass_angle"]).copy()
    df["_sec"] = (df["compass_angle"] // (360 / N_SECTORES)).astype(int)
    out = (df.sort_values("distancia_m", na_position="last")
             .drop_duplicates(subset=["dane_est", "_sec"], keep="first"))
    return out.drop(columns=["_sec"]).reset_index(drop=True)


def f_techo(df):
    return (df.sort_values("distancia_m", na_position="last")
              .groupby("dane_est", group_keys=False)
              .head(MAX_IMGS_POR_COLEGIO)
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Pipelines progresivos
# ---------------------------------------------------------------------------

def _base(df):
    """Filtros de calidad obligatorios presentes en todos los configs."""
    return f_solo_perspectiva(f_sin_pano(f_fecha(df)))


def p0(df): return f_dedup_seq(_base(df))
def p1(df): return f_dist(p0(df))
def p2(df): return f_heading(p1(df))
def p3(df): return f_dedup_sector(f_heading(f_dist(_base(df))))
def p4(df): return f_techo(p3(df))


CONFIGS = [
    {"id": "0_baseline",  "label": "0·Baseline",
     "desc": f"fecha≥{FECHA_DESDE} · solo perspectiva · dedup secuencia", "pipeline": p0},
    {"id": "1_distancia", "label": "1·+Distancia",
     "desc": f"+ distancia ≤ {MAX_DIST_M} m",                            "pipeline": p1},
    {"id": "2_heading",   "label": "2·+Heading",
     "desc": f"+ heading ±{MAX_ANGULO_DIFF}°",                           "pipeline": p2},
    {"id": "3_sectores",  "label": "3·+Sectores",
     "desc": f"+ dedup {N_SECTORES} sectores angulares",                  "pipeline": p3},
    {"id": "4_techo",     "label": "4·+Techo",
     "desc": f"+ máx {MAX_IMGS_POR_COLEGIO} imgs/colegio",               "pipeline": p4},
]


# ---------------------------------------------------------------------------
# Métricas por configuración
# ---------------------------------------------------------------------------

def calcular_metricas(cfg_id, label, desc, df_sel) -> dict:
    n    = len(df_sel)
    cols = df_sel.groupby("dane_est")["image_id"].count()
    nc   = len(cols)

    df_a  = df_sel.dropna(subset=["compass_angle", "lat_img", "lon_img"])
    pct_h = None
    if len(df_a):
        bear = df_a.apply(
            lambda r: _bearing(r.lat_img, r.lon_img, r.lat_colegio, r.lon_colegio), axis=1)
        diff = df_a.apply(lambda r: _diff_ang(r.compass_angle, bear[r.name]), axis=1)
        pct_h = round((diff <= MAX_ANGULO_DIFF).mean() * 100, 1)

    return {
        "config":         cfg_id,
        "label":          label,
        "descripcion":    desc,
        "n_imgs":         n,
        "n_colegios":     nc,
        "pct_colegios":   round(nc / N_COLEGIOS_TOTAL * 100, 1),
        "imgs_median":    round(cols.median(), 1) if nc else 0,
        "imgs_max":       int(cols.max()) if nc else 0,
        "imgs_std":       round(cols.std(), 1) if nc > 1 else 0,
        "dist_mediana_m": round(df_sel["distancia_m"].median(), 1) if n else 0,
        "dist_max_m":     round(df_sel["distancia_m"].max(), 1) if n else 0,
        "pct_heading_ok": pct_h,
    }


# ---------------------------------------------------------------------------
# Descarga asíncrona
# ---------------------------------------------------------------------------

async def _fetch(session, sem, url, ruta: Path) -> bool:
    if not FORZAR_DESCARGA and ruta.exists():
        return True
    async with sem:
        for intento in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 200:
                        ruta.write_bytes(await r.read())
                        return True
                    if r.status == 429:
                        await asyncio.sleep(2 ** intento * 3)
                        continue
                    return False
            except asyncio.CancelledError:
                return False
            except Exception:
                if intento == 2:
                    return False
                await asyncio.sleep(2 ** intento)
    return False


async def descargar_lote(filas: list, carpeta: Path) -> list:
    carpeta.mkdir(parents=True, exist_ok=True)
    sem   = asyncio.Semaphore(MAX_CONCURRENT)
    rutas = [None] * len(filas)

    async with aiohttp.ClientSession() as session:
        async def _t(i, fila):
            ruta = carpeta / fila["nombre_archivo"]
            ok   = await _fetch(session, sem, fila["url_descarga"], ruta)
            rutas[i] = ruta if ok else None

        await asyncio.gather(*[_t(i, f) for i, f in enumerate(filas)])

    return rutas


# ---------------------------------------------------------------------------
# Selección de imágenes para las escuelas de referencia
# ---------------------------------------------------------------------------

def imgs_para_escuelas(df_sel: pd.DataFrame, danes_ref: list) -> pd.DataFrame:
    """
    Para cada colegio en danes_ref devuelve las N_IMGS_POR_ESCUELA imágenes
    más cercanas disponibles en df_sel, ordenadas por (colegio, distancia).
    """
    df_sel = df_sel.dropna(subset=["url_descarga"])
    sub    = df_sel[df_sel["dane_est"].isin(danes_ref)]
    orden  = {d: i for i, d in enumerate(danes_ref)}
    return (sub.sort_values(["dane_est", "distancia_m"])
               .groupby("dane_est", group_keys=False, sort=False)
               .head(N_IMGS_POR_ESCUELA)
               .assign(_ord=lambda d: d["dane_est"].map(orden))
               .sort_values(["_ord", "distancia_m"])
               .drop(columns=["_ord"])
               .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Grillas visuales
# ---------------------------------------------------------------------------

def _leer_img(ruta: Path):
    try:
        return mpimg.imread(str(ruta))
    except Exception:
        return None


def grilla_config(cfg_label, cfg_desc, m, danes_ref,
                  nombre_colegios, rutas, ruta_png) -> None:
    """
    Filas = escuelas de referencia · Columnas = imágenes (N_IMGS_POR_ESCUELA).
    """
    ncols = N_IMGS_POR_ESCUELA
    nrows = len(danes_ref)
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).reshape(nrows, ncols)

    for col_i in range(ncols):
        axes[0, col_i].set_title(f"Img {col_i + 1}", fontsize=8)

    for row_i, dane in enumerate(danes_ref):
        nombre = nombre_colegios.get(dane, dane)[:32]
        imgs_dane = sorted(
            [(k, v) for k, v in rutas.items()
             if k[0] == cfg_label and k[1] == dane],
            key=lambda x: x[0][2],
        )
        for col_i in range(ncols):
            ax   = axes[row_i, col_i]
            ruta = imgs_dane[col_i][1] if col_i < len(imgs_dane) else None
            if ruta and ruta.exists():
                img = _leer_img(ruta)
                if img is not None:
                    ax.imshow(img)
                    ax.axis("off")
                    continue
            ax.set_facecolor("#e8e8e8")
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#aaa")
            ax.axis("off")

        axes[row_i, 0].set_ylabel(nombre, fontsize=6, rotation=90, labelpad=4)

    titulo = (f"{cfg_label}  ·  {cfg_desc}\n"
              f"{m['n_imgs']:,} imgs totales  ·  "
              f"{m['n_colegios']} colegios ({m['pct_colegios']:.1f}%)  ·  "
              f"dist_med {m['dist_mediana_m']:.1f} m")
    plt.suptitle(titulo, fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=130, bbox_inches="tight")
    plt.close()


def grilla_cruzada(cfg_labels, danes_ref, nombre_colegios, rutas, ruta_png) -> None:
    """
    Filas = escuelas de referencia · Columnas = configs (imagen más cercana).
    Celdas vacías indican que esa escuela no tiene cobertura en ese config.
    """
    nrows = len(danes_ref)
    ncols = len(cfg_labels)
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
    axes = np.array(axes).reshape(nrows, ncols)

    for col_i, lbl in enumerate(cfg_labels):
        axes[0, col_i].set_title(lbl, fontsize=8, pad=4)

    for row_i, dane in enumerate(danes_ref):
        nombre = nombre_colegios.get(dane, dane)[:30]
        for col_i, lbl in enumerate(cfg_labels):
            ax   = axes[row_i, col_i]
            ruta = rutas.get((lbl, dane, 0))
            if ruta and ruta.exists():
                img = _leer_img(ruta)
                if img is not None:
                    ax.imshow(img)
                    ax.axis("off")
                    continue
            ax.set_facecolor("#e8e8e8")
            ax.text(0.5, 0.5, "sin cobertura\nen este filtro",
                    ha="center", va="center", fontsize=6.5,
                    transform=ax.transAxes, color="#888")
            ax.axis("off")

        axes[row_i, 0].set_ylabel(nombre, fontsize=6.5, rotation=90, labelpad=4)

    plt.suptitle(
        f"Mismos {len(danes_ref)} colegios · imagen más cercana por configuración",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Grilla cruzada → {ruta_png.name}")


# ---------------------------------------------------------------------------
# Panel cuantitativo
# ---------------------------------------------------------------------------

def panel_metricas(tabla: pd.DataFrame, ruta_png: Path, total_cat: int) -> None:
    labels  = tabla["label"].tolist()
    x       = np.arange(len(labels))
    colores = ["#7f8c8d", "#3498db", "#e67e22", "#9b59b6", "#27ae60"]

    def _bar(ax, vals, titulo, ylabel, fmt="{:.0f}"):
        bars = ax.bar(x, vals, color=colores)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals],
                     fontsize=7, padding=3)
        ax.set_title(titulo, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    _bar(fig.add_subplot(gs[0, 0]), tabla["n_imgs"],
         "Imágenes seleccionadas", "N imgs", "{:,.0f}")
    _bar(fig.add_subplot(gs[0, 1]), tabla["pct_colegios"],
         "% colegios cubiertos", "%", "{:.1f}%")

    ax3 = fig.add_subplot(gs[0, 2])
    b3  = ax3.bar(x, tabla["imgs_median"], color=colores,
                   yerr=tabla["imgs_std"], capsize=4,
                   error_kw={"elinewidth": 1})
    ax3.bar_label(b3, labels=[f"{v:.1f}" for v in tabla["imgs_median"]],
                  fontsize=7, padding=3)
    ax3.set_title("Mediana imgs/colegio (±σ)", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax3.set_ylabel("Imgs/colegio", fontsize=8)
    ax3.yaxis.grid(True, alpha=0.4)

    _bar(fig.add_subplot(gs[1, 0]), tabla["dist_mediana_m"],
         "Distancia mediana foto→colegio", "Metros", "{:.1f} m")

    ax5 = fig.add_subplot(gs[1, 1])
    pct_h = tabla["pct_heading_ok"].fillna(0).tolist()
    b5 = ax5.bar(x, pct_h, color=colores)
    ax5.bar_label(b5, labels=[f"{v:.1f}%" for v in pct_h], fontsize=7, padding=3)
    ax5.set_ylim(0, 110)
    ax5.set_title(f"% imgs con heading ±{MAX_ANGULO_DIFF}°", fontsize=10)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax5.set_ylabel("%", fontsize=8)
    ax5.yaxis.grid(True, alpha=0.4)
    ax5.axhline(100, color="gray", lw=0.8, ls="--")

    _bar(fig.add_subplot(gs[1, 2]), tabla["imgs_max"],
         "Máx imgs en un colegio", "Imgs", "{:.0f}")

    fig.suptitle(
        f"Comparación progresiva de filtros — Mapillary × Colegios Bogotá\n"
        f"Catálogo base: {total_cat:,} imgs · {N_COLEGIOS_TOTAL} colegios",
        fontsize=11, y=1.01,
    )
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Panel métricas → {ruta_png.name}")


# ---------------------------------------------------------------------------
# Tabla consola
# ---------------------------------------------------------------------------

def _tabla_consola(tabla: pd.DataFrame) -> None:
    hdrs   = ["Config", "Imgs", "Colegios", "%Cobertura",
              "Med/col", "Máx", "Dist(m)", "%Heading"]
    cols   = ["label", "n_imgs", "n_colegios", "pct_colegios",
              "imgs_median", "imgs_max", "dist_mediana_m", "pct_heading_ok"]
    widths = [20, 10, 9, 12, 8, 6, 10, 11]
    sep    = "─" * sum(widths)
    print(f"\n{'═'*sum(widths)}")
    print("  COMPARACIÓN DE FILTROS")
    print(f"{'═'*sum(widths)}")
    print("  " + "".join(h.ljust(w) for h, w in zip(hdrs, widths)))
    print(f"  {sep}")
    for _, r in tabla.iterrows():
        vals = [
            str(r["label"])[:widths[0]-1],
            f"{int(r['n_imgs']):,}",
            str(int(r["n_colegios"])),
            f"{r['pct_colegios']:.1f}%",
            str(r["imgs_median"]),
            str(int(r["imgs_max"])),
            f"{r['dist_mediana_m']:.1f}",
            f"{r['pct_heading_ok']:.1f}%" if r["pct_heading_ok"] is not None else "—",
        ]
        print("  " + "".join(v.ljust(w) for v, w in zip(vals, widths)))
    print(f"{'═'*sum(widths)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    # ── Cargar y preparar catálogo ────────────────────────────────────────
    print(f"Cargando catálogo: {RUTA_CATALOGO.name} …")
    df = pd.read_csv(RUTA_CATALOGO, dtype={"dane_est": str})
    df = f_pano_norm(df)
    df = df.dropna(subset=["lon_img", "lat_img"])
    total_cat = len(df)

    tiene_camera_type = "camera_type" in df.columns
    print(f"  {total_cat:,} imágenes · {df['dane_est'].nunique()} colegios")
    if not tiene_camera_type:
        print("  [aviso] El catálogo no tiene columna camera_type. "
              "Regenera con FORZAR_RECATALOGO=True para activar el filtro de perspectiva.")
    print()

    # ── Aplicar pipelines ─────────────────────────────────────────────────
    print("Aplicando pipelines …")
    resultados = []
    cfg_rich   = []
    for cfg in CONFIGS:
        df_sel = cfg["pipeline"](df)
        m      = calcular_metricas(cfg["id"], cfg["label"], cfg["desc"], df_sel)
        resultados.append(m)
        cfg_rich.append({**cfg, "df_sel": df_sel, "metricas": m})
        print(f"  {cfg['label']:<18} {m['n_imgs']:>7,} imgs  "
              f"{m['n_colegios']:>4} colegios ({m['pct_colegios']:.1f}%)  "
              f"dist_med {m['dist_mediana_m']:.1f} m")

    tabla = pd.DataFrame(resultados)
    _tabla_consola(tabla)
    tabla.to_csv(RUTA_SALIDA / "comparacion_tabla.csv", index=False)

    # ── Panel cuantitativo ────────────────────────────────────────────────
    print("Generando panel de métricas …")
    panel_metricas(tabla, RUTA_SALIDA / "comparacion_metricas.png", total_cat)

    # ── Elegir colegios de referencia (intersección de TODOS los configs) ─
    danes_comun = None
    for cfg in cfg_rich:
        danes = set(cfg["df_sel"]["dane_est"].unique())
        danes_comun = danes if danes_comun is None else danes_comun & danes

    rng = random.Random(RANDOM_SEED)
    danes_ref = sorted(danes_comun)
    rng.shuffle(danes_ref)
    danes_ref = danes_ref[:N_ESCUELAS_REF]

    nombre_col = (df[df["dane_est"].isin(danes_ref)]
                  .drop_duplicates("dane_est")
                  .set_index("dane_est")["nombre_est"]
                  .to_dict())

    print(f"\nColegios de referencia ({len(danes_ref)}):")
    for d in danes_ref:
        print(f"  {d}  {nombre_col.get(d, '')}")

    # ── Descargar las mismas escuelas en cada config ───────────────────────
    # Clave: (cfg_label, dane_est, slot) donde slot=0 es la imagen más cercana
    rutas_master = {}

    print("\nDescargando fotos de referencia …")
    for cfg in cfg_rich:
        carpeta = RUTA_SALIDA / "fotos" / cfg["id"]
        carpeta.mkdir(parents=True, exist_ok=True)

        df_ref  = imgs_para_escuelas(cfg["df_sel"], danes_ref)
        if df_ref.empty:
            print(f"  {cfg['label']}: sin imágenes para los colegios de referencia")
            continue

        filas = df_ref.to_dict("records")
        print(f"  {cfg['label']}: descargando {len(filas)} imgs …", end="", flush=True)
        rutas_lista = asyncio.run(descargar_lote(filas, carpeta))
        print(f" {sum(1 for r in rutas_lista if r)}/{len(filas)} OK")

        dane_slots: dict = {}
        for fila, ruta in zip(filas, rutas_lista):
            dane = fila["dane_est"]
            slot = dane_slots.get(dane, 0)
            rutas_master[(cfg["label"], dane, slot)] = ruta
            dane_slots[dane] = slot + 1

    # ── Grilla por config ─────────────────────────────────────────────────
    print("\nGenerando grillas por config …")
    for cfg in cfg_rich:
        ruta_png = RUTA_SALIDA / f"grid_{cfg['id']}.png"
        grilla_config(
            cfg_label       = cfg["label"],
            cfg_desc        = cfg["desc"],
            m               = cfg["metricas"],
            danes_ref       = danes_ref,
            nombre_colegios = nombre_col,
            rutas           = rutas_master,
            ruta_png        = ruta_png,
        )
        print(f"  → grid_{cfg['id']}.png")

    # ── Grilla cruzada ────────────────────────────────────────────────────
    print("\nGenerando grilla cruzada …")
    grilla_cruzada(
        cfg_labels      = [c["label"] for c in cfg_rich],
        danes_ref       = danes_ref,
        nombre_colegios = nombre_col,
        rutas           = rutas_master,
        ruta_png        = RUTA_SALIDA / "grid_cruzado.png",
    )

    print(f"\nTodos los archivos en:\n  {RUTA_SALIDA}\n")
    print("Archivos clave:")
    print("  grid_cruzado.png          → mismos colegios × todos los filtros")
    print("  grid_<config>.png         → detalle de cada config (N imgs/colegio)")
    print("  comparacion_metricas.png  → panel cuantitativo")
    print("  comparacion_tabla.csv     → tabla numérica\n")


if __name__ == "__main__":
    main()
