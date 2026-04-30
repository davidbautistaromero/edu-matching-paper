"""
topic_viz.py
============
Grid de top-6 colegios por tópico Ridge M1 relevante.
- topic_5 (+0.012) — positivo: más sobre-demanda
- topic_2 (-0.009) — más negativo: menos sobre-demanda
- topic_1 (-0.006) — segundo negativo

Una imagen por colegio (id_establecimiento): se toma el heading
con mayor peso en ese tópico para cada colegio, luego top-6 colegios únicos.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT    = Path(__file__).resolve().parent.parent
NMF_P   = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K8_images.parquet"
IMG_DIR = ROOT / "data" / "images" / "gsv"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 8

# Coeficientes Ridge M1 por tópico
COEFS = {
    "topic_1": -0.0060,
    "topic_2": -0.0092,
    "topic_3": -0.0012,
    "topic_4": -0.0031,
    "topic_5": +0.0120,
    "topic_6": -0.0043,
    "topic_7": +0.0018,
    "topic_8": +0.0021,
}

TOPICS = [
    (col, "top", f"{col}  [{coef:+.4f}]  {'↑' if coef > 0 else '↓'} sobre-demanda")
    for col, coef in sorted(COEFS.items())
]

df = pd.read_parquet(NMF_P)

# Reconstruir path de imagen
def img_path(row):
    return IMG_DIR / str(row["id_establecimiento"]) / \
           f"{row['id_establecimiento']}_{int(row['heading']):03d}.jpg"

df["img_path"] = df.apply(img_path, axis=1)
df = df[df["img_path"].apply(lambda p: p.exists())].copy()

# Filtrar imágenes vacías por tamaño de archivo (< 5KB = en blanco o corrupta)
df = df[df["img_path"].apply(lambda p: p.stat().st_size > 5000)].copy()
print(f"Imágenes válidas (>5KB): {len(df):,}")

for col, rank_dir, title in TOPICS:
    # Para cada colegio, quedarse con el heading de mayor peso en ese tópico
    best_per_school = (
        df.loc[df.groupby("id_establecimiento")[col].idxmax()]
        .reset_index(drop=True)
    )

    # Ordenar por peso, tomar candidatos extra para poder filtrar blancos
    BUFFER = TOP_N * 4
    if rank_dir == "top":
        candidates = best_per_school.nlargest(BUFFER, col)
    else:
        candidates = best_per_school.nsmallest(BUFFER, col)

    # Filtrar imágenes en blanco (std < 15 sobre píxeles 0-255)
    import numpy as _np
    def not_blank(path):
        try:
            img = mpimg.imread(str(path))
            return float(_np.std(img.astype(float))) >= 15.0
        except Exception:
            return False

    candidates = candidates[candidates["img_path"].apply(not_blank)].head(TOP_N)
    top = candidates

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        img = mpimg.imread(str(row["img_path"]))
        ax.imshow(img)
        ax.set_title(
            f"id={row['id_establecimiento']}  {int(row['heading'])*10}°\n"
            f"peso={row[col]:.3f}",
            fontsize=8
        )
        ax.axis("off")

    plt.tight_layout()
    fname = f"{col}_{rank_dir}{TOP_N}.png"
    out = OUT_DIR / fname
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out.name}")

print("Listo.")
