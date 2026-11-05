# Bank Marketing: Predicting Term-Deposit Subscriptions

CSCI 3345 Machine Learning — Fall 2026

**Team Members:** Hiram Chen, Patrick Liong, Braden Johnsen

## Project Overview

This project investigates whether machine-learning models can predict customer subscriptions to bank term deposits using information available before a marketing call. We compare logistic regression and decision trees, examine the effect of target leakage from call duration, and evaluate generalization across time periods.

## Dataset

We use the UCI Bank Marketing dataset (`bank-additional-full.csv`), containing 41,188 observations, 20 input variables, and a binary subscription target.

Dataset: https://doi.org/10.24432/C5K306

## Methods

- Majority-class baseline
- L2-regularized logistic regression
- Depth-tuned decision tree
- Duration leakage ablation
- Random versus chronological splitting

Our primary metric is PR-AUC (average precision), with ROC-AUC used for comparisons across time periods.

## Reproducing Results

The dataset is publicly available through the UCI Machine Learning Repository.

The data pipeline is in `src/01_data_pipeline.ipynb`. Processed train, validation, and test matrices are in `data/processed/`. See `data/README.md` for the cleaning decisions, the split, and how to load the arrays. The majority-class baseline scores 0.887 accuracy and 0.113 validation PR-AUC.

## Team Responsibilities

- **Hiram Chen:** Data pipeline, cleaning, splits, preprocessing fitted on training rows, majority baseline.
- **Braden Johnsen:** Logistic regression, duration ablation, threshold and top-10% metrics.
- **Patrick Liong:** Decision tree, time-split ablation, bootstrap intervals.
- **All Members:** Final test evaluation and the report.
