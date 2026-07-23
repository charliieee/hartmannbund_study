# AI training needs assessment among physicians in Germany

Analysis code for: *Consensus on format, divergence on content: a needs assessment for artificial intelligence training among physicians in Germany*  **[Sophie-Charlotte Perret, Jonah Siebert, Jan Baumann Markus Schinle; Journal, DOI after acceptance]**.

Cross-sectional online survey (n = 362). The code reproduces all numbers, tables, and figures in the manuscript.

## Repository structure

```
prepare_data.py        Data preparation: raw cleaned export -> analysis_dataset.csv
                       (all variable definitions, recoding rules, built-in checksums)
results_R2_fig1.py     RQ1: descriptives + Figure 1 (use, areas, training, barriers)
results_R3_table2.py   RQ2: logistic regression + sensitivity analyses (Table 2)
results_R4_fig2.py     RQ3: training preferences (Figure 2)
results_R5_fig3.py     RQ4a: segment comparisons with BH-FDR correction (Figure 3)
results_R6_fig4.py     RQ4b: latent class analysis (Figure 4)
requirements.txt       Python dependencies
codebook/              Variable definitions, recoding rules, analysis samples (incl. flow diagram)
questionnaire/         Full survey instrument (German original)
figures/               Script-generated figures (Fig. 1-4)
supplement/            Mirror of the published supplementary materials (added upon acceptance)
```

## How to run

```bash
pip install -r requirements.txt
python prepare_data.py        # aborts if any checksum fails
python results_R2_fig1.py     # then any results script, in any order
```

Requires Python >= 3.10 with pandas, scipy, statsmodels, stepmix, matplotlib, openpyxl.

`prepare_data.py` validates nine checksums (e.g., n = 362 total; 341 with defined AI-use status; 192 users; 322 regression sample). If a checksum fails, your input data differ from the version used in the manuscript.

## Data

The survey data are **not** included in this repository. Data are available from the corresponding author on reasonable request, subject to ethics approval. Place `DataCleaned.csv` (cp1252, semicolon-separated) next to the scripts to run the pipeline.

## Citation

**[Citation + DOI after acceptance]**

## License
Code: MIT. The questionnaire and supplementary materials are published with the article.
