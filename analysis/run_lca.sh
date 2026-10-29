#!/bin/bash
# Latent class analyses (Supplementary Note 8, Fig. 2, Figure S8-1). Run from the repository root; see README for run times.
set -e
PY=${PY:-venv/bin/python}
D=analysis_dataset.csv
R=analysis
mkdir -p out
# primary specification: four-level coding, 16 items (items 12, 15, 16 dropped), 100 starts, BLRT with 200 bootstrap samples
$PY -I $R/lca.py $D out/lca_primary --coding four --starts 100 --drop 12,15,16 --labels primary --blrt 200 --blrt-starts 10
# alternative specifications compared in Table S8-8
$PY -I $R/lca.py $D out/lca_19items_3levels                                           # three-level coding, 19 items, 20 starts
$PY -I $R/lca.py $D out/lca_19items_4levels        --coding four --starts 100          # four-level coding, 19 items
$PY -I $R/lca.py $D out/lca_17items_4levels  --coding four --starts 100 --drop 15,16   # four-level coding, 17 items
$PY -I $R/lca.py $D out/lca_16items_3levels      --coding primary --starts 100 --drop 12,15,16   # three-level coding, 16 items
$PY -I $R/application_counts.py $D out/application_counts              # person-level counts and models (also in run_analysis.sh)
# tables of Supplementary Note 8 from the runs above (reads the output directories relative to the working directory;
# Table S8-1 uses out/item_screen from run_analysis.sh)
(cd out && ../$PY -I ../$R/note8_tables.py ../$D note8_tables.md)
echo "LCA DONE"
