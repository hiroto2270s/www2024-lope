# Survival Analysis in HIV-Positive Patients: Treatment Efficacy and Contributing Factors
### Overview
This project evaluates survival outcomes for HIV-positive patients from the AIDS Clinical Trials Group Study 175 dataset, analysing the effects of treatment regimens and clinical/biological factors. Kaplan-Meier survival curves and Cox Proportional Hazards models are employed to identify significant predictors of survival.

### Dataset
The dataset originates from Kaggle: AIDS Clinical Trials Group Study 175. It includes data on 2,139 patients and 24 features, covering baseline characteristics, treatment groups, and survival outcomes.

### Key Features
#### Survival Analysis Techniques:
- Kaplan-Meier curves to visualise survival probabilities.
- Log-rank tests to compare group survival differences.
- Cox Proportional Hazards models to assess predictors.
  
#### Variables Studied:
- Clinical: CD4/CD8 cell counts, symptomatic status, off-treatment indicator.
- Treatment: Four regimens, including ZDV, ddI, and combination therapies.

### Findings
- Combination therapies and ddI outperform ZDV alone in survival probability.
- Symptomatic status, CD4 counts and off-treatment status are significant predictors of survival.
- Model evaluation reveals violations of the proportional hazards assumption, emphasising the need for advanced modelling.

### Repository Structure
- data/: AIDS_ClinicalTrial_GroupStudy175.csv contains the Kaggle dataset.
- scripts/: ACTG_code.R includes the R script.
- docs/: Final report (ACTG_report.pdf).
