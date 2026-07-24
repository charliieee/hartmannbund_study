# A nationwide needs assessment for artificial intelligence training in continuing medical education

Analysis code for: *Consensus on delivery, divergence on content: physicians' artificial intelligence training needs in continuing medical education*  **[Sophie-Charlotte Perret, Jonah Siebert, Jan Baumann Markus Schinle; Journal, DOI after acceptance]**.

Cross-sectional online survey (n = 362). The code reproduces all numbers, tables, and figures in the manuscript.

## Repository structure
This repository contains the Supplementary Information, the Pipeline and Information regarding the original data.


### Supplementary Information 
This is the Supplementary Document `supplement_document.pdf` handed in when submitting the paper. 
| #  | Document                     | Content                                                                                                   |
|----|------------------------------|-----------------------------------------------------------------------------------------------------------|
| S1 | STROBE Flowchart             | STROBE participant-flow diagram and data-cleaning rules                                                    |
| S2 | Participant Characteristics  | Full sample distributions and comparison with Bundesärztekammer physician statistics                      |
| S3 | Sensitivity Analysis         | Sensitivity analyses for the multivariable logistic regression of professional AI use (n = 322; adds to Table 2) |
| S4 | Codebook                     | Free-text recoding, regression variable coding, and correction of three inconsistent records              |
| S5 | RQ3 — Age × Format           | Preferred formats by age group; additional information for RQ3                                             |
| S6 | Segment Comparison           | All preference options: %, χ², Cramér's V, BH-adjusted p, and pairwise post-hoc tests (Fisher, BH) with effect sizes |
| S7 | Questionnaire                | Full instrument in German (original), followed by the English translation                                 |
| S8 | Latent Class Analysis        | Four-class solution (sensitivity analysis)                                                                 |
| S9 | CHERRIES Checklist           | Checklist for Reporting Results of Internet E-Surveys                                                      |

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
