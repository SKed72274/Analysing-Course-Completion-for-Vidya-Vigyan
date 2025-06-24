# 🎓 Predicting Course Completion for Vidya Vigyan

*A Data-Driven Approach to Student Success Using Statistical and Machine Learning Models*

[![View on Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/subhashreekedia/notebookc6cfa13a52)

---

## 📘 Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Objective](#objective)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Statistical Analyses](#statistical-analyses)
- [Feature Engineering](#feature-engineering)
- [Threshold-Based Binarization](#threshold-based-binarization)
- [Feature Selection & Transformation](#feature-selection--transformation)
- [Model Training](#model-training)
- [Ensemble Models](#ensemble-models)
- [Evaluation](#evaluation)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)
- [Tools and Libraries](#tools-and-libraries)
- [Appendix](#appendix)
- [Authors](#authors)

---

## 📌 Introduction

This project explores a predictive modeling pipeline for identifying students likely to complete online courses on the Vidya Vigyan platform. The solution leverages a blend of statistical inference, feature engineering, and advanced machine learning models.

---

## 📂 Project Description

Built as part of the **Vista 2024 – Data Beyond Boundaries** competition, this solution helps understand learner behavior and suggests interventions to improve course completion rates using explainable data-driven methods.

---

## 🎯 Objective

To predict whether a student will complete a course based on their engagement and performance metrics, and to recommend personalized strategies to improve learning outcomes.

---

## 📊 Dataset Description

The dataset includes learner activity features like:
- `EngagementHours`
- `ContentConsumed`
- `AssessmentsTaken`
- `PerformanceMetric`
- `ProgressPercentage`
- `AccessMode`
- `LearningPathwayType`

---

## 🔍 Exploratory Data Analysis (EDA)

- Statistical summaries and distribution plots
- Joint plots and contour plots
- Histogram stack plots
- Correlation matrix
- Survival analysis to evaluate engagement patterns
- Conditional probability heatmap
- ANOVA to assess feature group variability

---

## 📈 Statistical Analyses

- **ANOVA**: Revealed key differences in features between completing and non-completing learners.
- **Survival Analysis**: Completion probability increases significantly with engagement time above 20 hours.
- **Eta Squared (η²)**: Used to measure effect size and relevance of features.

---

## 🧪 Feature Engineering

Generated insightful composite features:
- `InvolvementMetric = ContentConsumed × AssessmentsTaken`
- `ExamProficiencyIndex = AssessmentsTaken × PerformanceMetric`
- `AcademicAchievementFactor = PerformanceMetric × ProgressPercentage`
- `StudyDedicationIndex = ProgressPercentage × EngagementHours`
- `EngagementIndex = AssessmentsTaken × ProgressPercentage`

Techniques used:
- Feature interactions
- Square root, square, and cube transformations
- Relevance checks using correlation matrix and η²

---

## ⚙️ Threshold-Based Binarization

To simplify and enhance model interpretability, we used **Partial Dependence Plots (PDPs)** to identify optimal thresholds for key features. These thresholds were then used to convert continuous features into **binary variables** (e.g., "high engagement" vs. "low engagement").

This binary feature set was used to train a **Bernoulli Naive Bayes model**, which surprisingly yielded **very strong performance** due to the simplified yet meaningful feature space.

---

## 🧹 Feature Selection & Transformation

- Dropped non-informative features like `AccessMode`
- Removed multicollinear composite features
- Applied feature scaling and binarization based on PDPs

---

## 🤖 Model Training

Baseline Models:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM
- Gaussian Naive Bayes (low performance due to non-normal data)
- Artificial Neural Network

Binary Data Model:
- **Bernoulli Naive Bayes** using thresholded features

---

## 🧠 Ensemble Models

To further optimize performance, we used:
- **Voting Classifier**
- **Stacking Classifier** with Logistic Regression as meta-model (✅ Final Model)
- **Bagging** with XGBoost

Final accuracy from the Stacking Classifier: **~96%**

---

## 📏 Evaluation

- **Accuracy** = 0.96
- **F1 Score** = 0.95
- ROC-AUC (tested but did not provide extra insights)

---

## 💡 Key Insights

- Course success correlates highly with:
  - Engagement Hours > 35
  - Content Consumed between 7–10 units
  - Assessments Taken > 4
  - Performance Metric > 75%
- No significant impact of `AccessMode` or `LearningPathwayType`
- Some students spend 80–90 hours but still drop off, highlighting the need for **motivational triggers**

---

## 🚀 Recommendations

- **Gamification**:
  - Milestone-based rewards after 50% course completion
  - Performance-based leaderboards
  - Threshold-based gaming incentives using PDPs

- **UX/UI Enhancements**:
  - More interactive platform (web + mobile)
  - Community chat or feedback module

- **Personalized Support**:
  - Use model insights to nudge low-engagement users
  - Send motivational emails based on behavioral clusters

---

## 🛠 Tools and Libraries

- **Python**
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- **EDA & Visuals**
  - Joint plots, correlation matrix, contour plots
- **ML Techniques**
  - PDPs, eta squared, stacking, bagging, voting

---

## 📎 Appendix

### Acronyms
- **EDA**: Exploratory Data Analysis  
- **ANOVA**: Analysis of Variance  
- **PDP**: Partial Dependence Plot  
- **SVM**: Support Vector Machine  
- **XGBoost**: Extreme Gradient Boosting  
- **η² (Eta Squared)**: Measure of feature effect size  

### Methodologies
- **Interaction Features**
- **Square Root / Squaring / Cubing**
- **Binary Conversion from PDP Thresholds**
- **Ensemble Learning Techniques**

🔗 [View Kaggle Notebook](https://www.kaggle.com/code/subhashreekedia/notebookc6cfa13a52)

---
