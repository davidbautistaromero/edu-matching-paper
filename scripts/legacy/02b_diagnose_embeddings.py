#!/usr/bin/env python3
"""
02a_diagnose_embeddings.py
Diagnóstico de calidad de embeddings antes de LDA y selección de d_PCA.

Pruebas:
  1. Varianza por feature
  2. Similitud coseno entre pares de imágenes
  3. PCA explained variance + figura para justificar elección de d

Input:
  data/images/embeddings/gsv_vgg19_raw.parquet

Outputs:
  data/images/embeddings/diagnostico_embeddings.json
  figures/pca_component_selection.png
"""

EMBEDDINGS_PATH  = 'data/images/embeddings/gsv_vgg19_raw.parquet'
OUTPUT_JSON      = 'data/images/embeddings/diagnostico_embeddings.json'
FIGURE_PATH      = 'reports/figures/pca_component_selection'   # sin extensión, se guarda .png

N_SAMPLE_PAIRS   = 5000
PCA_N_COMPONENTS  = 150  # cuántas PCs calcular para el scree plot
MIN_VAR_THRESHOLD = 0.80  # umbral mínimo aceptable; si los superiores no se alcanzan, se usa este

# =============================================================================
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# =============================================================================
# ESTILO PAPER
# =============================================================================
PAPER_RC = {
    'font.family':        'serif',
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'lines.linewidth':    1.4,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
}

# =============================================================================
# CARGA
# =============================================================================
def load_embeddings(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    log.info('=' * 60)
    log.info('DIAGNÓSTICO DE EMBEDDINGS')
    log.info('=' * 60)
    log.info(f'Cargando: {path}')

    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feat_cols].values.astype(np.float32)

    n_img = len(df)
    n_est = df['id_establecimiento'].nunique() if 'id_establecimiento' in df.columns else '?'
    log.info(f'Imágenes: {n_img:,}  |  Establecimientos: {n_est}  |  Features: {len(feat_cols)}')
    log.info(f'Rango: min={X.min():.4f}  max={X.max():.4f}')
    return df, X


# =============================================================================
# TEST 1: VARIANZA POR FEATURE
# =============================================================================
def test_feature_variance(X: np.ndarray) -> dict:
    log.info('')
    log.info('─' * 60)
    log.info('TEST 1: Varianza por feature')
    log.info('─' * 60)

    d = X.shape[1]
    feat_var = X.var(axis=0)
    n_zero      = int((feat_var == 0).sum())
    n_near_zero = int((feat_var < 1e-6).sum())
    n_low       = int((feat_var < feat_var.mean() * 0.01).sum())
    pct_nz      = 100 * n_near_zero / d

    log.info(f'Varianza media:    {feat_var.mean():.6f}')
    log.info(f'Varianza mediana:  {np.median(feat_var):.6f}')
    log.info(f'Varianza max/min:  {feat_var.max():.6f} / {feat_var.min():.6f}')
    log.info(f'Var == 0:          {n_zero}/{d} ({100*n_zero/d:.1f}%)')
    log.info(f'Var < 1e-6:        {n_near_zero}/{d} ({pct_nz:.1f}%)')
    log.info(f'Var < 1% media:    {n_low}/{d} ({100*n_low/d:.1f}%)')

    alert = bool(pct_nz > 20)
    log.info(f'→ {"⚠️  ALERTA" if alert else "✅ OK"}: {pct_nz:.1f}% features con varianza casi nula')

    return {
        'media': float(feat_var.mean()),
        'mediana': float(np.median(feat_var)),
        'max': float(feat_var.max()),
        'min': float(feat_var.min()),
        'n_var_cero': n_zero,
        'n_var_near_zero': n_near_zero,
        'pct_near_zero': float(pct_nz),
        'alerta': alert,
    }


# =============================================================================
# TEST 2: SIMILITUD COSENO
# =============================================================================
def test_cosine_similarity(X: np.ndarray) -> dict:
    log.info('')
    log.info('─' * 60)
    log.info('TEST 2: Similitud coseno entre pares de imágenes')
    log.info('─' * 60)

    n = len(X)
    X_l2 = normalize(X, norm='l2')

    if n <= 200:
        log.info(f'N={n} ≤ 200 → similitud exacta (todos los pares)')
        sim_matrix = cosine_similarity(X_l2)
        idx = np.triu_indices(n, k=1)
        sims = sim_matrix[idx]
    else:
        rng = np.random.default_rng(42)
        ia = rng.integers(0, n, size=N_SAMPLE_PAIRS)
        ib = rng.integers(0, n, size=N_SAMPLE_PAIRS)
        mask = ia != ib
        ia, ib = ia[mask], ib[mask]
        sims = (X_l2[ia] * X_l2[ib]).sum(axis=1)
        log.info(f'N={n} > 200 → muestreo de {len(sims):,} pares aleatorios')

    cos_mean   = float(sims.mean())
    cos_median = float(np.median(sims))
    cos_p95    = float(np.percentile(sims, 95))

    log.info(f'Media:    {cos_mean:.4f}')
    log.info(f'Mediana:  {cos_median:.4f}')
    log.info(f'P95:      {cos_p95:.4f}')
    log.info(f'Min/Max:  {sims.min():.4f} / {sims.max():.4f}')

    alert = bool(cos_mean > 0.85)
    log.info(f'→ {"⚠️  ALERTA" if alert else "✅ OK"}: similitud media = {cos_mean:.4f} '
             f'(umbral >0.85 → embeddings muy similares)')

    return {
        'media': cos_mean,
        'mediana': cos_median,
        'p95': cos_p95,
        'min': float(sims.min()),
        'max': float(sims.max()),
        'n_pares': int(len(sims)),
        'alerta': alert,
    }


# =============================================================================
# TEST 3: PCA EXPLAINED VARIANCE + FIGURA
# =============================================================================
def test_pca_variance(X: np.ndarray, n_components: int) -> dict:
    log.info('')
    log.info('─' * 60)
    log.info('TEST 3: PCA — selección de dimensionalidad')
    log.info('─' * 60)

    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)

    ev    = pca.explained_variance_ratio_
    cumev = np.cumsum(ev)

    log.info(f'PC1: {100*ev[0]:.1f}%  |  PC2: {100*ev[1]:.1f}%  |  PC1+PC2: {100*cumev[1]:.1f}%')

    # Para cada umbral, reportar solo si fue alcanzado dentro de n_comp
    thresholds = {}
    for t in [70, 80, 90, 95]:
        idx = int(np.searchsorted(cumev, t / 100))
        if idx < n_comp:
            thresholds[t] = idx + 1
            log.info(f'PCs para {t}% varianza: {idx + 1}')
        else:
            thresholds[t] = None
            log.info(f'PCs para {t}% varianza: >  {n_comp} (no alcanzado)')

    # Elbow: d donde la ganancia marginal cae por debajo del 1%
    marginal_gain = np.diff(cumev)
    elbow_candidates = np.where(marginal_gain < 0.01)[0]
    d_elbow = int(elbow_candidates[0]) + 1 if len(elbow_candidates) else n_comp
    log.info(f'Codo (ganancia < 1% por PC adicional): d = {d_elbow}')

    # d* = exactamente el umbral mínimo (MIN_VAR_THRESHOLD);
    # si no se alcanza dentro de n_comp, cae al codo
    min_t = int(MIN_VAR_THRESHOLD * 100)
    if thresholds.get(min_t) is not None:
        d_recommended = thresholds[min_t]
        criterion = f'≥{min_t}% varianza'
    else:
        d_recommended = d_elbow
        criterion = f'codo (umbral {min_t}% no alcanzado en {n_comp} PCs)'
    log.info(f'Recomendación ({criterion}): d* = {d_recommended}')

    alert = bool(ev[0] > 0.70)
    log.info(f'→ {"⚠️  ALERTA" if alert else "✅ OK"}: PC1 = {100*ev[0]:.1f}% '
             f'(umbral >70% → colapso dimensional)')

    return {
        'explained_variance_ratio': [float(v) for v in ev],
        'cumulative_explained_variance': [float(v) for v in cumev],
        'pc1_pct': float(100 * ev[0]),
        'pc2_pct': float(100 * ev[1]),
        'n_pcs_70pct': thresholds.get(70),
        'n_pcs_80pct': thresholds.get(80),
        'n_pcs_90pct': thresholds.get(90),
        'n_pcs_95pct': thresholds.get(95),
        'd_elbow': d_elbow,
        'd_recommended': d_recommended,
        'alerta': alert,
    }, pca


def plot_pca_selection(pca_results: dict, output_path: str):
    """
    Genera figura paper-style con scree plot + varianza acumulada.
    """
    ev      = np.array(pca_results['explained_variance_ratio']) * 100
    cumev   = np.array(pca_results['cumulative_explained_variance']) * 100
    d_rec   = min(pca_results['d_recommended'], len(ev))
    d_elbow = min(pca_results['d_elbow'], len(ev))
    n_show  = len(ev)
    x       = np.arange(1, n_show + 1)

    C_BAR   = '#4878CF'   # azul steel
    C_CUM   = '#C44E52'   # rojo oscuro
    C_VREC  = '#2a2a2a'   # d* — negro
    C_ELBOW = '#E68A2E'   # codo — naranja

    thresh_styles = {
        90: ('#4d4d4d', '-'),
        80: ('#7f7f7f', '--'),
        70: ('#b3b3b3', ':'),
    }

    with plt.rc_context(PAPER_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(9, 3.8),
            gridspec_kw={'wspace': 0.42},
        )

        # ── Panel A: Scree plot ────────────────────────────────────────────
        ax1.bar(x, ev, color=C_BAR, alpha=0.85, width=0.75, zorder=2)

        # Línea codo
        ax1.axvline(d_elbow, color=C_ELBOW, lw=1.1, ls=':', zorder=3)
        ax1.text(d_elbow + 1, ev.max() * 0.92,
                 f'codo\n$d={d_elbow}$',
                 va='top', ha='left', fontsize=7, color=C_ELBOW)

        # Línea d*
        ax1.axvline(d_rec, color=C_VREC, lw=1.2, ls='--', zorder=4)
        ax1.text(d_rec + 1, ev.max() * 0.72,
                 f'$d^*={d_rec}$',
                 va='top', ha='left', fontsize=7, color=C_VREC)

        ax1.set_xlabel('Componente principal')
        ax1.set_ylabel('Varianza explicada (%)')
        ax1.set_title('(A) Gráfico de sedimentación')
        ax1.set_xlim(0.2, n_show + 0.8)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(25))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax1.tick_params(axis='x', rotation=45)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

        # ── Panel B: Varianza acumulada ───────────────────────────────────
        ax2.plot(x, cumev, color=C_CUM, lw=1.6, zorder=3)
        ax2.fill_between(x, cumev, alpha=0.08, color=C_CUM)

        # Líneas de umbral con etiquetas a la izquierda
        for thresh, (color, ls) in thresh_styles.items():
            ax2.axhline(thresh, color=color, ls=ls, lw=0.9, zorder=2)
            ax2.text(2, thresh + 1.2, f'{thresh}%',
                     ha='left', va='bottom', fontsize=7, color=color)

        # Línea codo
        ax2.axvline(d_elbow, color=C_ELBOW, lw=1.1, ls=':', zorder=4)
        ax2.annotate(
            f'codo $d={d_elbow}$\n({cumev[d_elbow-1]:.0f}%)',
            xy=(d_elbow, cumev[d_elbow - 1]),
            xytext=(d_elbow + 8, cumev[d_elbow - 1] - 14),
            fontsize=7, color=C_ELBOW,
            arrowprops=dict(arrowstyle='->', color=C_ELBOW, lw=0.8),
        )

        # Línea d* — la flecha ancla en la intersección con el umbral mínimo
        min_thresh_pct = int(MIN_VAR_THRESHOLD * 100)
        ax2.axvline(d_rec, color=C_VREC, lw=1.2, ls='--', zorder=5)
        ax2.annotate(
            f'$d^*={d_rec}$ ({min_thresh_pct}%)',
            xy=(d_rec, min_thresh_pct),
            xytext=(d_rec - 35, min_thresh_pct - 18),
            fontsize=7, color=C_VREC,
            arrowprops=dict(arrowstyle='->', color=C_VREC, lw=0.8),
        )

        ax2.set_xlabel('Número de componentes principales')
        ax2.set_ylabel('Varianza explicada acumulada (%)')
        ax2.set_title('(B) Varianza explicada acumulada')
        ax2.set_xlim(0.2, n_show + 0.8)
        ax2.set_ylim(0, 103)
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(25))
        ax2.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax2.tick_params(axis='x', rotation=45)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

        # Nota al pie — más pequeña, sin backslashes
        d_elbow_var    = f'{cumev[d_elbow-1]:.0f}'
        min_thresh_pct = int(MIN_VAR_THRESHOLD * 100)
        fig.text(
            0.5, -0.02,
            f'Características VGG19 block5_pool (512d), fachadas de colegios GSV, Bogotá. '
            f'Codo en $d={d_elbow}$ ({d_elbow_var}% var.); '
            f'$d^*={d_rec}$ seleccionado con umbral de {min_thresh_pct}% de varianza explicada.',
            ha='center', va='top', fontsize=5, color='#666666',
            style='italic',
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{output_path}.png')
        plt.close(fig)

    log.info(f'Figura guardada: {output_path}.png')


# =============================================================================
# MAIN
# =============================================================================
def main():
    _, X = load_embeddings(EMBEDDINGS_PATH)
    results = {'n': int(len(X)), 'd': int(X.shape[1])}

    results['varianza_features'] = test_feature_variance(X)
    results['similitud_coseno']  = test_cosine_similarity(X)

    pca_results, _ = test_pca_variance(X, PCA_N_COMPONENTS)
    results['pca'] = pca_results

    # Figura
    log.info('')
    log.info('─' * 60)
    log.info('Generando figura de selección de componentes PCA...')
    log.info('─' * 60)
    plot_pca_selection(pca_results, FIGURE_PATH)

    # Resumen
    log.info('')
    log.info('=' * 60)
    log.info('RESUMEN')
    log.info('=' * 60)
    alertas = [k for k, v in results.items() if isinstance(v, dict) and v.get('alerta')]
    if alertas:
        log.info(f'⚠️  Alertas: {", ".join(alertas)}')
        log.info('   → LDA probablemente colapsará sin PCA previo.')
    else:
        log.info('✅ Sin alertas.')

    d_rec = pca_results['d_recommended']
    log.info(f'Dimensionalidad recomendada para LDA: PCA_N_COMPONENTS = {d_rec}')
    log.info('')
    log.info('Pasos sugeridos:')
    log.info(f'  1. Fijar PCA_N_COMPONENTS = {d_rec} en 03_lda_topics.py')
    log.info('  2. Correr python scripts/03_lda_topics.py')

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f'JSON guardado: {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
