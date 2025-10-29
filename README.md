# 🌧️ Rainfall Prediction in the Junín Region - Balanced Binary Classification Model

## 📝 Project Abstract

This project addresses the challenge of predicting **moderate/heavy rain** events in the Junín Region (Peru), a critical climatic factor for agricultural planning. The task is formulated as a severely **imbalanced binary classification problem** (most records are "No/Light Rain").

The study evaluates two Machine Learning approaches: a **Balanced Statistical Model** (Logistic Regression) and a **Neural Network Classifier** (MLPClassifier). To mitigate class imbalance, the **SMOTE (Synthetic Minority Over-sampling Technique)** method was applied exclusively to the training set, prioritizing the maximization of **Recall** for the critical minority class.

The **MLPClassifier with SMOTE** demonstrated superior performance, achieving a **Balanced Accuracy of 0.87** and a **Recall of 0.94** for the moderate/heavy rain class. This result validates the use of advanced non-linear models combined with oversampling techniques for identifying rare and critical precipitation events.

---

## 🚀 Key Models and Results

The primary objective was to maximize the **Recall** (sensitivity) metric for the **moderate/heavy rain class (Class 1)**, as a False Negative (missing an actual rainfall event) carries a significantly higher cost for the agricultural sector than a False Positive (a false alarm). A threshold of **2.5 mm** was used to define "moderate/heavy rain," adapted from the WMO's rainfall intensity classifications.

### Performance Comparison (on Test Set)

| Model | Recall (Class 1) | Balanced Accuracy | Precision (Class 1) | F1-Score (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| **MLP with SMOTE** | **0.94** | **0.868** | 0.05 | 0.09 |
| **LogReg with SMOTE** | 0.83 | 0.805 | 0.04 | 0.07 |
| LogReg without SMOTE | 0.83 | 0.794 | 0.04 | 0.07 |
| MLP without SMOTE | 0.00 | 0.50 | 0.00 | 0.00 |

> The **MLPClassifier with SMOTE** was selected as the optimal model for providing the best performance on the most critical metric: **Recall** (0.94). The **SMOTE technique proved essential**, as without it, the MLP completely failed to predict any rainfall events.

---

## 🛠️ Methodology and Data Preprocessing

### 1. Data Source
Hourly data on **meteorological variables** (Air Temperature, Relative Humidity, Atmospheric Pressure, Wind Speed and Direction) and **irradiance data** were obtained from the IGP (Instituto Geofísico del Perú) for the period **2021-2023**.

### 2. Key Preprocessing Steps

* **Dataset Integration:** Primary (EMA) and auxiliary (BSRN) datasets were merged using common date and time columns (year, month, day, hour).
* **Handling Missing Values:** Rows with null values were removed.
* **Future Prediction Variable Creation:** A **one-hour forward shift (`shift=1`)** was applied to the precipitation variable (`RR`), ensuring current meteorological variables predict rainfall one hour ahead.
* **Temporal Filtering:** The analysis was limited to the **December, January, February, and March** months to focus on the peak rainy season and reduce the high proportion of zero values.
* **Feature Selection by Collinearity:** Variables 'diffus\_SW\_rad', 'global\_SW\_rad', and 'reflec\_SW\_rad' were excluded due to physical redundancy.
* **Binarization of the Target Variable:** The Precipitation variable (`RR`) was transformed into a binary variable:
    * **Class 0:** "No Rain/Light Rain" ($\text{RR} < 2.5\text{ mm}$).
    * **Class 1:** "Moderate/Heavy Rain" ($\text{RR} \geq 2.5\text{ mm}$).

### 3. Class Imbalance Handling (SMOTE)

* The dataset was first split into training (75%) and test (25%) sets.
* **SMOTE (Synthetic Minority Over-sampling Technique)** was applied **exclusively to the training data**. This generates new synthetic samples for the minority class (rain days) by interpolating between nearest neighbors, thereby preventing data leakage into the test set.

---

## 💻 Code Snippet (Python)

Below are the steps for data processing and the implementation of the MLPClassifier with SMOTE.

### 1. Preprocessing, Binarization, and Data Splitting
```python
# Data loading and merging dataframes (df_base and df_aux1) ...
# df_final = df_base.merge(df_aux1, on=['year', 'month', 'day', 'hour'], how='inner') 
# df_final.dropna()

# 1. Apply temporal shift
shift = 1
# The precipitation (RR) at hour t+1 becomes the new 'RR' column for row t
df1 = df_final.drop(columns=['RR']).iloc[:-1*shift]
df2 = df_final['RR'].drop(df_final.index[0:shift])
df_total = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

# 2. Filter for peak rainy season months (Dec, Jan, Feb, Mar)
selected_months = [12, 1, 2, 3]
df_total = df_total[df_total['month'].isin(selected_months)]

# 3. Drop redundant/temporal variables 
columns_drop = ['year', 'day', 'month', 'hour', 'reflec_SW_rad', 'diffus_SW_rad', 'global_SW_rad']
df_total.drop(columns=columns_drop, inplace=True)

# 4. Binarize the target variable (RR)
bins = [0, 2.5, np.inf] # Class 0: < 2.5mm, Class 1: >= 2.5mm
labels = [0, 1]
df_total['rain_intensity'] = pd.cut(df_total['RR'], bins=bins, labels=labels, right=False)
df_total.drop(columns='RR', inplace=True)

# 5. Split the dataset
X = df_total.drop('rain_intensity', axis=1)
y = df_total['rain_intensity']
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
# Initial training set distribution: Counter({0: 4958, 1: 54})