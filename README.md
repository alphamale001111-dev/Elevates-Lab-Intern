# Elevates Lab Internship - Machine Learning Tasks

This repository contains internship task solutions for Machine Learning and Data Science.

---

## 📂 Repository Structure

```
├── .gitignore
├── README.md
└── Task 1/
    ├── Task1_Data_Cleaning_Preprocessing.ipynb
    ├── cleaned_titanic.csv
    ├── data_preprocessing.py
    ├── download_data.py
    ├── generate_titanic_data.py
    ├── titanic.csv
    └── plots/
        ├── 1_missing_values_heatmap.png
        ├── 2_outliers_before.png
        ├── 3_outliers_after.png
        ├── 4_feature_distributions.png
        └── 5_correlation_matrix.png
```

---

## 🎯 Task 1: Data Cleaning & Preprocessing

### Objective
Clean and prepare raw dataset (Titanic Dataset) for Machine Learning model training.

### Key Highlights
1. **Exploratory Data Analysis (EDA):** Inspected data types, non-null counts, missing value patterns.
2. **Missing Value Imputation:** Imputed `Age` with Median, `Embarked` with Mode, and created `Has_Cabin` flag.
3. **Categorical Encoding:** Converted `Sex` via Binary Label Encoding and `Embarked` via One-Hot Dummy Encoding.
4. **Outlier Detection & Removal:** Applied Interquartile Range (**IQR**) filtering on `Fare` and `Age` continuous numerical features.
5. **Feature Scaling:** Applied `StandardScaler` (Z-score) and `MinMaxScaler` normalization.

---

## 🚀 How to Run Locally

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
   ```

3. Run the data preprocessing script:
   ```bash
   python "Task 1/data_preprocessing.py"
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook "Task 1/Task1_Data_Cleaning_Preprocessing.ipynb"
   ```
