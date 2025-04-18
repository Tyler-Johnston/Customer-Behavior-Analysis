# Customer Behavior Analysis via Logistic Regression & SVMs

## 📌 Overview

In this project, we used **Logistic Regression** and **Support Vector Machines (SVMs)** to model two binary classification problems:

1. **Online Shopper Purchase Intent**  
2. **Bank Customer Churn**

Our goal was to analyze patterns in consumer behavior and determine which features were most useful in predicting purchase decisions and customer churn. We evaluated multiple kernel types for SVMs and used class balancing techniques to improve model accuracy.

---

## 🧠 Datasets

### 1️⃣ Online Shopper Purchase Intent  
- [Source](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
- Predict target variable `Revenue` (True/False)
- Features include:  
  - Bounce Rate, Exit Rate, Page Value  
  - Month, Visitor Type, Weekend  
  - Product interactions (e.g., informational/product-related pages)

### 2️⃣ Bank Customer Churn  
- [Source](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- Predict target variable `Churn` (1 = leaving, 0 = staying)
- Features include:  
  - Age, Balance, Credit Score, Country  
  - Active Member Status, Gender  

---

## 🧪 Methods & Techniques

- **Logistic Regression**
- **Support Vector Machines (Linear, Polynomial, RBF Kernels)**
- **Correlation analysis**
- **Feature selection**
- **One-Hot Encoding & Data Normalization**
- **Class weight balancing to address class imbalance**

---

## 📊 Results Summary

### 🛒 Shopper Purchase Intent

| Model | Features | F1 Score (Class 0 / Class 1) |
|-------|----------|------------------------------|
| Logistic Regression | PageValues vs ExitRates | 0.63 / 0.94 |
| SVM (Linear) | PageValues vs BounceRates | 0.65 / 0.93 |
| SVM (RBF) | PageValues vs ExitRates | **0.66 / 0.93** |

- **RBF SVM** performed best overall  
- Class weight tuning significantly improved prediction of "purchasing" class

---

### 🏦 Bank Customer Churn

| Model | Features | F1 Score (Churn / No Churn) |
|-------|----------|------------------------------|
| Logistic Regression | Age + Balance | 0.49 / 0.83 |
| SVM (Linear) | Age + Balance | 0.50 / 0.83 |
| SVM (Polynomial) | Age + Balance | 0.52 / 0.82 |
| SVM (RBF) | Age + Balance | **0.55 / 0.87** |

- RBF again showed strongest performance  
- Strong visual decision boundaries formed in feature space using age vs. balance  
- Without class weighting, models biased heavily toward majority class (no churn)

---

## 🧰 Tools Used

- Python  
- scikit-learn  
- pandas  
- NumPy  
- matplotlib  

---

## 🎯 Key Takeaways

- **PageValue**, **BounceRate**, and **ExitRate** were top predictors of purchase intent
- **Age** and **Balance** showed clear decision boundaries for bank churn
- **Class weighting** was crucial for both datasets to avoid majority-class bias
- **SVM with RBF kernel** provided best results across both tasks

---

## 📎 Links
- 📊 [Shopper Dataset on Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
- 📊 [Churn Dataset on Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
