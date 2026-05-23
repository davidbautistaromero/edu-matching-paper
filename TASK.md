# TASK: Add exclusion filter to 03b and 03c, then run them

## Context
Branch `paper-def-fixes`. We have `data/raw/excluded_schools.csv` with 23 rural school IDs.
The raw embeddings parquet already has `id_establecimiento` column.

## Step 1: Add filter to `scripts/03b_nmf_topics.py`
After loading `gsv_vgg19_raw.parquet`, add the same exclusion filter pattern:
1. Read `data/raw/excluded_schools.csv`  
2. Filter out rows where `id_establecimiento` matches excluded IDs (cast both to str, strip whitespace)
3. Log: "Excluded {n} images from {m} rural schools, {remaining} images remaining"
4. Continue with the filtered dataframe

## Step 2: Add filter to `scripts/03c_clip_features.py`
Same pattern as 03b — read exclusion list, filter the input data, log counts.
Check what input this script reads (probably the catalog or embeddings) and filter accordingly.

## Step 3: Run both scripts
1. Run `python scripts/03b_nmf_topics.py` — this generates new NMF topics (K=6,8,10)
2. Run `python scripts/03c_clip_features.py` — this generates new CLIP features

Both should complete and save updated parquet files.

## Step 4: Commit
Commit changes with message: `feat: add rural exclusion filter to 03b/03c, regenerate embeddings`

## Rules
- Only modify 03b and 03c (add filter near top after data load)
- DO run both scripts to regenerate outputs
- Do NOT modify any other scripts
- Do NOT delete TASK.md when done
- If a script fails, report the error — do not try to fix other scripts
