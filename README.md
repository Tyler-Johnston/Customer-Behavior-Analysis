# Customer Behavior Analysis via Logistic Regression and SVMs

## Overview

In this project, we used Logistic Regression and Support Vector Machines (SVMs) to model two binary classification problems:

1. Online Shopper Purchase Intent
2. Bank Customer Churn

The goal was to analyze patterns in consumer behavior and determine which features were most useful in predicting purchase decisions and customer churn. We evaluated multiple kernel types for the SVMs and used class balancing techniques to improve model accuracy.

---

## Datasets

### Online Shopper Purchase Intent

- [Source](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
- Target variable: `Revenue` (True/False)
- Features include:
  - Bounce rate, exit rate, page value
  - Month, visitor type, weekend
  - Product interactions (e.g., informational and product-related page visits)

### Bank Customer Churn

- [Source](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- Target variable: `Churn` (1 = leaving, 0 = staying)
- Features include:
  - Age, balance, credit score, country
  - Active member status, gender

---

## Methods and Techniques

- Logistic Regression
- Support Vector Machines (linear, polynomial, and RBF kernels)
- Correlation analysis
- Feature selection
- One-hot encoding and data normalization
- Class weight balancing to address class imbalance

---

## Results Summary

### Shopper Purchase Intent

| Model | Features | F1 Score (Class 0 / Class 1) |
|-------|----------|------------------------------|
| Logistic Regression | PageValues vs. ExitRates | 0.63 / 0.94 |
| SVM (Linear) | PageValues vs. BounceRates | 0.65 / 0.93 |
| SVM (RBF) | PageValues vs. ExitRates | 0.66 / 0.93 |

The RBF SVM performed best overall, and class weight tuning significantly improved prediction of the purchasing class.

---

### Bank Customer Churn

| Model | Features | F1 Score (Churn / No Churn) |
|-------|----------|------------------------------|
| Logistic Regression | Age + Balance | 0.49 / 0.83 |
| SVM (Linear) | Age + Balance | 0.50 / 0.83 |
| SVM (Polynomial) | Age + Balance | 0.52 / 0.82 |
| SVM (RBF) | Age + Balance | 0.55 / 0.87 |

RBF again showed the strongest performance, with clear decision boundaries forming in feature space using age versus balance. Without class weighting, models were heavily biased toward the majority class (no churn).

---

## Tools Used

- Python
- scikit-learn
- pandas
- NumPy
- matplotlib

---

## Key Takeaways

- Page value, bounce rate, and exit rate were the strongest predictors of purchase intent.
- Age and balance produced clear decision boundaries for bank churn.
- Class weighting was essential for both datasets to avoid majority-class bias.
- SVM with an RBF kernel gave the best results across both tasks.

[Full Analysis](https://github.com/Tyler-Johnston/Customer-Behavior-Analysis/blob/main/Project%207%20-%20Logistic%20Regression%20and%20SVMs.pdf)
