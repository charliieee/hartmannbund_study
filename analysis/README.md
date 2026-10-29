# analysis/ — scripts behind the tables and figures

*Physicians' artificial intelligence training needs in continuing medical education: a nationwide survey across adoption groups* (Perret, Siebert, Baumann, Schinle; npj Digital Medicine).

Data preparation and the descriptive analyses are in the repository root (`prepare_data.py`, `results_R2.py` to `results_R6.py`). This folder holds the scripts for everything else: the multivariable regression and its sensitivity analyses, the latent class analysis, the four adoption groups, the coding of free-text reasons, and Figures 1, 2, 4 and 5. Every script reads `analysis_dataset.csv`, the output of `prepare_data.py` (n = 362; MD5 `1d6da73c5735535ec9f077f8a3c44f4c`), and writes to `out/<script name>/`. The analysis scripts end with built-in checksums that abort the run if a number reported in the article or the Supplementary Information is not reproduced ("All checksums OK"); `check_numbers.py` then recomputes 224 named numbers of the article and the Supplementary Information from the output files.

## Where each table and figure comes from

| Reported | Script | Output |
|---|---|---|
| Fig. 1 | `fig1.py` | `out/fig1/fig1.png` |
| Fig. 2; Figure S8-1 | `lca.py` (primary specification, see below) | `out/lca_primary/fig2.png`, `figS8_1.png` |
| Fig. 3 | `results_R4.py` (repository root) | — |
| Fig. 4 | `adoption_groups.py` | `out/adoption_groups/fig4.png` |
| Fig. 5 | `fig5.py` (every number asserted against `adoption_groups.py` and `nonuser_reasons.py`) | `out/fig5/fig5.png` |
| Table 2; Tables S3-1, S3-2, S3-4 | `regression.py` | `out/regression/table2.csv`, `note3_tableA.csv`, `note3_tableB.csv`, `note3_item_level.csv` |
| Table 3 (sector column) and Table S10-9 | `providers_by_sector.py` | `out/providers_by_sector/providers_by_sector.csv` |
| Table 4; Table S10-2b | `content_measures.py` | `out/content_measures/content_measures.csv`, `content_measures_tests.csv` |
| Table S5-1 | `age_by_format.py` | `out/age_by_format/age_by_format.csv` |
| Tables S6-1 to S6-3; S10-1, S10-2 | `adoption_groups.py` | `out/adoption_groups/A_omnibus.csv`, `B_pairwise.csv`, `C_risk_differences.csv`, `D_trend.csv`, `E_composition.csv` |
| Table S8-1 | `item_screen.py` | `out/item_screen/item_screen.csv` |
| Tables S8-2 to S8-10 | `lca.py` (five runs, `run_lca.sh`), assembled by `note8_tables.py` | `out/lca_*/`, `out/note8_tables.md` |
| Table S8-11; Results RQ2 (application counts, use by unfamiliarity, provision × training) | `application_counts.py` | `out/application_counts/` |
| Note 10, topic structure (principal components) | `topic_structure.py` | `out/topic_structure/` |
| Table S10-3 | `no_intention_group.py` | `out/no_intention_group/1_composition.csv` |
| Tables S10-4, S10-5; Results RQ4 (free-text reasons) | `nonuser_reasons.py` with `q30_author_review.csv` | `out/nonuser_reasons/` |
| Table S10-6; Discussion (exposure markers) | `exposure_markers.py` | `out/exposure_markers/` |
| Table S10-7; Discussion (early-career physicians) | `early_career.py` | `out/early_career/` |
| Table S10-8; Discussion (form of employer-provided AI) | `provision_form.py` | `out/provision_form/` |
| Table S10-10; Results RQ1 (respondents unsure of their AI use) | `unsure_group.py` | `out/unsure_group/` |
| consistency check of the manuscript and Supplementary Information against all outputs | `check_numbers.py` | log only |

## Running

```
python3.13 -m venv venv
venv/bin/pip install -r analysis/requirements.txt
bash analysis/run_analysis.sh     # everything except the latent class analysis; a few minutes
bash analysis/run_lca.sh          # latent class analysis; about 45 minutes, most of it the bootstrap likelihood-ratio tests
```

Versions used: Python 3.13.17, pandas 2.3.3, numpy 2.2.6, scipy 1.18.1, statsmodels 0.15.0, StepMix 3.0.0, scikit-learn 1.9.1, matplotlib 3.11.2. Scripts are run from the repository root with `python -I`. `run_lca.sh` runs the primary specification (four response states, 16 items, 100 starts, 200 bootstrap samples) and the four alternative specifications compared in Table S8-8, then `note8_tables.py`, which needs `out/item_screen/` from `run_analysis.sh`.

**pandas 3.** The scripts in this folder also run under pandas 3.0.x. The repository-root script `prepare_data.py` does not: its strip rule for non-breaking spaces tests `col.dtype == "object"`, which is false for the `str` dtype that pandas 3 uses for text, so two comparisons fail silently and its checksums abort ("provision yes: 0 != 105"). `prepare_data_pandas3.patch` replaces the test by `pd.api.types.is_string_dtype(col) or pd.api.types.is_object_dtype(col)`, which behaves identically under pandas 2 and 3 (`git apply analysis/prepare_data_pandas3.patch`).

## Free-text coding (item 30)

`nonuser_reasons.py` codes the free-text reasons of the 149 non-users against ten categories defined before coding (rule-based first pass, documented second pass). `q30_author_review.csv` records the changes made by the authors on reading every answer (file, id, codes before, codes after, author, date, note); passed as the third argument, it is applied after the rule-based coding and before any count, and the checksums in the script refer to the coding with the review applied. The numbers in the article are those with the review applied.
