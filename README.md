# 🌧️ Predictive Models for Rainfall Estimation in Junín

## 📌 Overview

This project addresses the challenge of rainfall prediction in the Junín region, where a high proportion of non-rain records introduces significant bias in traditional forecasting models. To overcome this, we developed and evaluated two alternative approaches aimed at improving predictive accuracy and event detection.

## 🎯 Objectives

- Mitigate data imbalance caused by the dominance of non-rain records.
- Explore hybrid and multiclass classification strategies for precipitation modeling.
- Assess model performance in detecting meaningful rainfall events.

## ⚙️ Methodology

We implemented and compared the following predictive frameworks:

### 🔹 Hybrid Model: Binary Classification + Regression
- **Step 1:** Binary classification to determine whether rainfall will occur.
- **Step 2:** Regression to estimate the volume of precipitation if rain is predicted.
- **Strength:** Enables fine-grained estimation of rainfall quantity.

### 🔸 Balanced Multiclass Classification Model
- **Approach:** Direct prediction of discrete precipitation levels using a multiclass classifier.
- **Technique:** Data balancing methods applied to address class imbalance.
- **Strength:** Achieved superior overall performance, especially in detecting significant rainfall events.

## 📊 Key Findings

- The hybrid model provides detailed volume estimates but is less effective in general event detection.
- The balanced multiclass model outperforms in overall accuracy and excels at identifying impactful rainfall occurrences.

## 🧰 Technologies Used

- Python (scikit-learn, pandas, NumPy)
- Data preprocessing and balancing techniques
- Evaluation metrics: Precision, Recall, F1-score

## 🚀 Future Work

- Integrate temporal features for short-term forecasting.
- Explore ensemble methods to combine strengths of both models.
- Extend the framework to other regions with similar data imbalance issues.

## 👥 Authors

- Lino, P.
- Calderón, C.
- Chavarria, M.
- Izarra, L.
- Mascco, P.
