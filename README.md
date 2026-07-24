# A nationwide needs assessment for artificial intelligence training in continuing medical education

Analysis code for: *Consensus on delivery, divergence on content: physicians' artificial intelligence training needs in continuing medical education*  **[Sophie-Charlotte Perret, Jonah Siebert, Jan Baumann Markus Schinle; Journal, DOI after acceptance]**.

Cross-sectional online survey (n = 362). The code reproduces all numbers, tables, and figures in the manuscript.

## Repository structure
This is the Supplementary Document `supplement_document.pdf` handed in when submitting the paper. 

| # | document | content |
|------|--------|---------|
| S1 | STROBE Flowchart | with STROBE Flowchart and Rules for Data Cleaning |
| S2 | Participants Characteristics | Tables about the Participants and Comparision to Bundesärztekammer Statistic |
| S3 | Sensitivity Analysis | Sensitivity Analysis for the Multivariable logistic regression of professional AI use (n = 322) in RQ2. Addition to Tab. 2. |
| S4 | Codebook | free-text coding,regression categories, correction for 3 inconsistent records |
| S5 | RQ3 | Age Group x Format, additional information for RQ3 |
| S6 | Segment Comparision |Segment comparison of all preference options: %, χ², Cramér’s V, adj. p + pairwise post-hoc tests (Fisher, BH) with effect sizes|
| S7 | Questionnaires |Questionnaire first in German (original) then in English (translated).  |
| S8 | LCA |Sensitivity Analysis |
| S9 | CHERRIES Checkliste |Checklist for Reporting Results of Internet E-Surveys|

### Pipeline
Scripts are run from the repository root in this order:

```
prepare_data.py        Data preparation: raw cleaned export -> analysis_dataset.csv
                       (all variable definitions, recoding rules, built-in checksums)
results_R2_fig1.py     RQ1: descriptives + Figure 1 (use, areas, training, barriers)
results_R3_table2.py   RQ2: logistic regression + sensitivity analyses (Table 2)
results_R4_fig2.py     RQ3: training preferences (Figure 2)
results_R5_fig3.py     RQ4a: segment comparisons with BH-FDR correction (Figure 3)
results_R6_fig4.py     RQ4b: latent class analysis (Figure 4)
requirements.txt       Python dependencies
```

### How to run

```bash
pip install -r requirements.txt
python prepare_data.py        # aborts if any checksum fails
python results_R2_fig1.py     # then any results script, in any order
```

Requires Python >= 3.10 with pandas, scipy, statsmodels, stepmix, matplotlib, openpyxl.

`prepare_data.py` validates nine checksums (e.g., n = 362 total; 341 with defined AI-use status; 192 users; 322 regression sample). If a checksum fails, your input data differ from the version used in the manuscript.

### Data

The survey data are **not** included in this repository. Data are available from the corresponding author on reasonable request, subject to ethics approval. In case of reasonable request, place `DataCleaned.csv` (cp1252, semicolon-separated) next to the scripts to run the pipeline.

## Citation

**[Citation + DOI after acceptance]**

## License
Code: MIT. The questionnaire and supplementary materials are published with the article.
