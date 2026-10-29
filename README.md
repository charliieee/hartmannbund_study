# A nationwide needs assessment for artificial intelligence training in continuing medical education

TITLE\
    *Physicians’ artificial intelligence training needs in continuing medical education: a nationwide survey across adoption groups* 

AUTHORS\
    Sophie-Charlotte Perret, University of Cambridge (0009-0009-0044-4633)\
    Jonah Siebert, Offenburg University of Applied Sciences (0009-0001-7136-1175)\
    Jan Baumann, Hartmannbund, Association of Doctors in Germany (0000-0003-3679-7341)\
    Markus Schinle, Offenburg University of Applied Sciences (0000-0002-3132-0280)

VERSION\
    1.1\
    Last updated: 27 October 2026

--------------------------------------------------------------------------------
## 1. OVERVIEW OF THE STUDY
--------------------------------------------------------------------------------

This repository documents the data preparation and analysis reported in the accompanying short communication, a secondary analysis of one item of a cross-sectional survey of 363 physicians in Germany conducted between December 2025 and March 2026. 

Respondents rated 19 digital tools on a six-point frequency scale (daily, weekly, monthly, less often, "would like to, but not available", "unclear what this is"). The paper reports the item "Differentialdiagnostik bzw. Clinical-Decision-Support (CDSS)" against the 18 other tools, its association with age, and its relation to self-reported AI use for diagnostic tasks.

The repository provides the materials referenced in the manuscript as "Supplements", the analysis code, and the aggregated tables behind Figure 1 and Supplements S2 and S3.

Individual-level microdata are NOT contained in this repository for privacy reasons. See Section 6, Data availability.

--------------------------------------------------------------------------------
## 2. Repository structure
--------------------------------------------------------------------------------
This repository contains the Supplementary Information, the Pipeline and Information regarding the original data, a cross-sectional online survey.


### Supplementary Information 
This is the Supplementary Document `supplement_document.pdf` handed in when submitting the paper. 

| # | Document | Content |
|---|---|---|
| 1 | Supplementary Note 1 | Participant flow and exclusion rules |
| 2 | Supplementary Note 2 | Participant characteristics and comparison with the physician population |
| 3 | Supplementary Note 3 | Multivariate logistic regression of professional AI use |
| 4 | Supplementary Note 4 | Codebook: free-text recoding, variable definitions and correction rules |
| 5 | Supplementary Note 5 | Preferred format by age group (RQ3) |
| 6 | Supplementary Note 6 | Training preferences across four adoption groups: omnibus tests, pairwise contrasts and equivalence bounds |
| 7 | Supplementary Note 7 | Questionnaire |
| 8 | Supplementary Note 8 | Latent class analysis of digital application use: item screen, model selection, convergence, classification certainty, the four-class solution, and sensitivity to coding and item set |
| 9 | Supplementary Note 9 | CHERRIES checklist (Checklist for Reporting Results of Internet E-Surveys)|
| 10 | Supplementary Note 10 | Adoption groups in detail: the shape of group differences, the two non-user groups, preferences by exposure, early-career physicians, the form of employer-provided AI, providers by sector, and respondents unsure of their AI use|

--------------------------------------------------------------------------------
### 3. SOFTWARE ENVIRONMENT
--------------------------------------------------------------------------------

  Language:   Python 3.12
  Libraries:  pandas, scipy (scipy.stats), statsmodels, stepmix, matplotlib, openpyxl
   
  Exact versions are pinned in code/requirements.txt. To create a matching environment:

        python3.12 -m venv venv
        source venv/bin/activate
        pip install -r code/requirements.txt
        

--------------------------------------------------------------------------------
### 4. REPRODUCING THE ANALYSIS
--------------------------------------------------------------------------------
Scripts are run from the repository root in this order:

```
prepare_data.py        Data preparation: raw cleaned export -> analysis_dataset.csv
                       (all variable definitions, recoding rules, built-in checksums)
results_R2.py     RQ1: descriptives + Figure 1 (use, areas, training, barriers)
results_R3.py   RQ2: logistic regression + sensitivity analyses (Table 2)
results_R4.py     RQ3: training preferences (Figure 2)
results_R5.py     RQ4a: segment comparisons with BH-FDR correction (Figure 3)
results_R6.py     RQ4b: latent class analysis (Figure 4)
requirements.txt       Python dependencies
```

--------------------------------------------------------------------------------
### 5. DATA AVAILABILITY
--------------------------------------------------------------------------------

   The survey data are **not** included in this repository. Data are available from the corresponding author on reasonable         request, subject to ethics approval. In case of reasonable request, place `DataCleaned.csv` (cp1252, semicolon-separated)       next to the scripts to run the pipeline.


--------------------------------------------------------------------------------
### 6. ETHICS
--------------------------------------------------------------------------------

    The study received favourable ethics opinions from the University of
    Cambridge (Ref. 736, 15 May 2025) and Offenburg University of Applied
    Sciences (Ref. 04-25, 27 June 2025). Data collection was conducted in
    compliance with the General Data Protection Regulation (GDPR). A privacy
    notice and participant information sheet preceded the questionnaire;
    participation was conditional on informed consent.


--------------------------------------------------------------------------------
### 7. LICENCE
--------------------------------------------------------------------------------
   Code: MIT. The questionnaire and supplementary materials are published with the article.


--------------------------------------------------------------------------------
### 8. CITATION 
--------------------------------------------------------------------------------
   See CITATION.cff. Paper: to be updated


--------------------------------------------------------------------------------
### 9. CONTACT
--------------------------------------------------------------------------------
   Corresponding Authors:
    Sophie Perret, University of Cambridge, scp64@cam.ac.uk 

    Co-Authors: 
    Jonah Siebert, Offenburg University of Applied Sciences, jonah.siebert@hs-offenburg.de 
    Jan Baumann, Hartmannbund e.V.  
    Markus Schinle, Offenburg University of Applied Sciences, markus.schinle@hs-offenburg.de 
================================================================================

