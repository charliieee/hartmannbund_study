#!/bin/bash
# Reproduces all outputs except the latent class diagnostics (see README). Run from the repository root.
set -e
PY=${PY:-venv/bin/python}
D=analysis_dataset.csv
R=analysis
mkdir -p out
$PY -I $R/regression.py $D out/regression
$PY -I $R/application_counts.py $D out/application_counts
$PY -I $R/adoption_groups.py $D out/adoption_groups
$PY -I $R/topic_structure.py $D out/topic_structure
$PY -I $R/nonuser_reasons.py $D out/nonuser_reasons $R/q30_author_review.csv
$PY -I $R/no_intention_group.py $D out/no_intention_group
$PY -I $R/exposure_markers.py $D out/exposure_markers
$PY -I $R/early_career.py $D out/early_career
$PY -I $R/provision_form.py $D out/provision_form
$PY -I $R/providers_by_sector.py $D out/providers_by_sector
$PY -I $R/unsure_group.py $D out/unsure_group
$PY -I $R/content_measures.py $D out/content_measures
$PY -I $R/age_by_format.py $D out/age_by_format
$PY -I $R/item_screen.py $D out/item_screen
$PY -I $R/fig1.py $D out/fig1
$PY -I $R/fig5.py $D out/nonuser_reasons/q30_interested_coded.csv out/nonuser_reasons/q30_no_intention_coded.csv out/fig5
echo "ALL DONE"
