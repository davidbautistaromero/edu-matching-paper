"""
topic_viz.py
============

Dos figuras por tópico: top-8 (mayor peso) y bottom-8 (menor peso).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT    = Path(__file__).resolve().parents[2]
NMF_P   = ROOT / "data" / "images" / "embeddings" / "gsv_nmf_K6_images.parquet"
IMG_DIR = ROOT / "data" / "images" / "gsv"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N  = 8
TOPICS = [f"topic_{k}" for k in range(1, 7)]

df = pd.read_parquet(NMF_P)

def img_path(row):
    return IMG_DIR / str(row["id_establecimiento"]) / \
           f"{row['id_establecimiento']}_{int(row['heading']):03d}.jpg"

df["img_path"] = df.apply(img_path, axis=1)
df = df[df["img_path"].apply(lambda p: p.exists())].copy()
df = df[df["img_path"].apply(lambda p: p.stat().st_size > 5000)].copy()
print(f"Imágenes válidas (>5KB): {len(df):,}")

def not_blank(path):
    try:
        img = mpimg.imread(str(path))
        return float(np.std(img.astype(float))) >= 15.0
    except Exception:
        return False

BUFFER = TOP_N * 4

for col in TOPICS:
    k = col.split("_")[1]

    best_per_school = (
        df.loc[df.groupby("id_establecimiento")[col].idxmax()]
        .reset_index(drop=True)
    )

    for rank_dir in ("top", "bottom"):
        if rank_dir == "top":
            candidates = best_per_school.nlargest(BUFFER, col)
            suptitle = f"Topic {k} — Highest Weight (top 8)"
            fname = f"topic_{k}_top8.png"
        else:
            candidates = best_per_school.nsmallest(BUFFER, col)
            suptitle = f"Topic {k} — Lowest Weight (bottom 8)"
            fname = f"topic_{k}_bottom8.png"

        candidates = candidates[candidates["img_path"].apply(not_blank)].head(TOP_N)

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

        for ax, (_, row) in zip(axes.flat, candidates.iterrows()):
            img = mpimg.imread(str(row["img_path"]))
            ax.imshow(img)
            ax.set_title(
                f"id={row['id_establecimiento']} {int(row['heading'])*10}°"
                f" w={row[col]:.3f}",
                fontsize=8
            )
            ax.axis("off")

        plt.tight_layout()
        out = OUT_DIR / fname
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Guardado: {out.name}")

print("Listo.")
