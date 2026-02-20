# Big Data (Coursework + Final Project)

Coursework and final project for a Big Data class (University of Chicago). The repo contains assignments and a capstone project focused on predictive modeling using a Google Play Store dataset.

## Capstone: Google Play Store Predictive Modeling
**Goal:** Predict app success using app metadata + user reviews (primary target: **log(Installs)**).

**Data (before/after cleaning):**
- Apps: **10,841 → 10,356**
- Reviews: **64,295 → 37,432**

**Approach:**
- Built an end-to-end R pipeline: cleaning → feature engineering → modeling → evaluation
- Trained and compared: **LASSO**, **pruned decision tree**, **Random Forest**
- Evaluated using **20× random 80/20 train-test splits** with out-of-sample metrics

**Result:** Random Forest achieved **median out-of-sample R² ≈ 0.8** for predicting **log-installs**.

Full write-up (PDF): [Big Data Final – Cao, Lekkala, Zhang](Final%20Project/Big%20Data%20Final%20-%20Cao%2C%20Lekkala%2C%20Zhang.pdf)

## Repository Structure
- `Final Project/` — capstone project (paper + code)
- `hw2/ ... hw7/` — weekly assignments and exercises

## Tech Stack
- **R** (tidyverse, modeling, visualization)
- RMarkdown/Quarto (reports)
- Git for version control

## Notes
Datasets used in the final project are sourced from a public Google Play Store / reviews dataset (hosted on Kaggle).  
See the Final Project for details and reproducibility instructions.
The paper documents preprocessing choices, modeling setup, and evaluation details.
