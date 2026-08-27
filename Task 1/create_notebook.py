import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# Task 1: Data Cleaning & Preprocessing
**Objective:** Learn how to clean and prepare raw data for Machine Learning models using Python, Pandas, NumPy, Matplotlib, and Seaborn.

---

## Steps Overview
1. **Import Dataset & Explore Basic Info:** Inspect data dimensions, data types, missing values, and statistical summary.
2. **Handle Missing Values:** Impute missing numerical and categorical values, drop redundant or sparse columns.
3. **Categorical Feature Encoding:** Convert categorical features into numerical format (Label Encoding & One-Hot Encoding).
4. **Outlier Detection & Handling:** Visualize outliers using Boxplots and filter out extreme values using Interquartile Range (IQR).
5. **Feature Scaling (Normalization & Standardization):** Apply `StandardScaler` and `MinMaxScaler` to scale numerical features.
"""))

# Setup
cells.append(nbf.v4.new_code_cell("""import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Set plot style
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)
print("Libraries imported successfully!")
"""))

# Step 1
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Import Dataset & Explore Basic Info
Load raw dataset and analyze structure, missing values, and summary statistics."""))

cells.append(nbf.v4.new_code_cell("""# Load Dataset
df = pd.read_csv("titanic.csv")

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\\n")
print("First 5 Rows:")
display(df.head())

print("\\nData Types & Info:")
df.info()

print("\\nMissing Values Count:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])

print("\\nStatistical Summary:")
display(df.describe().T)

# Plot missing values heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Values Heatmap (Raw Data)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Step 2
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Handle Missing Values
- Impute missing numerical values (`Age`) using median imputation.
- Impute missing categorical values (`Embarked`) using mode imputation.
- Handle high-missing feature (`Cabin`) by creating a binary flag (`Has_Cabin`).
- Drop non-informative identifier columns (`PassengerId`, `Name`, `Ticket`)."""))

cells.append(nbf.v4.new_code_cell("""df_clean = df.copy()

# 1. Median Imputation for Age
age_median = df_clean['Age'].median()
df_clean['Age'] = df_clean['Age'].fillna(age_median)
print(f"Imputed 'Age' with median: {age_median:.2f}")

# 2. Mode Imputation for Embarked
embarked_mode = df_clean['Embarked'].mode()[0]
df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)
print(f"Imputed 'Embarked' with mode: {embarked_mode}")

# 3. Create Has_Cabin binary indicator and drop Cabin
df_clean['Has_Cabin'] = df_clean['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df_clean.drop(columns=['Cabin'], inplace=True)

# 4. Drop identifier columns
df_clean.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

print(f"Remaining Missing Values in Dataset: {df_clean.isnull().sum().sum()}")
display(df_clean.head())
"""))

# Step 3
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Categorical Feature Encoding
- **Binary Label Encoding:** Convert `Sex` (`male`/`female`) to binary numbers (0/1).
- **One-Hot Encoding:** Convert nominal feature `Embarked` (`S`, `C`, `Q`) into dummy variables."""))

cells.append(nbf.v4.new_code_cell("""# Binary Label Encoding for Sex
le_sex = LabelEncoder()
df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])

# One-Hot Encoding for Embarked
df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True, dtype=int)

print("Data types after encoding:")
print(df_clean.dtypes)
display(df_clean.head())
"""))

# Step 4
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Visualize & Handle Outliers (IQR Method)
Visualize outliers using boxplots for numerical features (`Age`, `Fare`) and remove extreme outliers outside $Q1 - 1.5 \\times IQR$ and $Q3 + 1.5 \\times IQR$."""))

cells.append(nbf.v4.new_code_cell("""# Visualize Boxplots BEFORE Outlier Removal
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df_clean['Age'], ax=axes[0], color='skyblue')
axes[0].set_title("Boxplot of Age (Before Outlier Removal)", fontweight='bold')

sns.boxplot(y=df_clean['Fare'], ax=axes[1], color='salmon')
axes[1].set_title("Boxplot of Fare (Before Outlier Removal)", fontweight='bold')
plt.tight_layout()
plt.show()

# Function to remove outliers using IQR
def remove_outliers_iqr(dataframe, columns):
    df_out = dataframe.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_len = len(df_out)
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
        print(f"[{col}] Removed {initial_len - len(df_out)} outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df_out

df_no_outliers = remove_outliers_iqr(df_clean, ['Fare', 'Age'])
print(f"Dataset rows reduced from {len(df_clean)} to {len(df_no_outliers)}")

# Visualize Boxplots AFTER Outlier Removal
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df_no_outliers['Age'], ax=axes[0], color='lightgreen')
axes[0].set_title("Boxplot of Age (After Outlier Removal)", fontweight='bold')

sns.boxplot(y=df_no_outliers['Fare'], ax=axes[1], color='lightcoral')
axes[1].set_title("Boxplot of Fare (After Outlier Removal)", fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Step 5
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Feature Scaling (Standardization & Normalization)
Scale continuous numerical features (`Age`, `Fare`, `SibSp`, `Parch`) using:
1. **StandardScaler (Z-score Normalization):** Rescales data to $\\mu=0, \\sigma=1$.
2. **MinMaxScaler:** Rescales data to standard range $[0, 1]$."""))

cells.append(nbf.v4.new_code_cell("""num_features = ['Age', 'Fare', 'SibSp', 'Parch']

# 1. Standardization
scaler = StandardScaler()
df_scaled = df_no_outliers.copy()
df_scaled[num_features] = scaler.fit_transform(df_scaled[num_features])

# 2. Normalization
minmax_scaler = MinMaxScaler()
df_normalized = df_no_outliers.copy()
df_normalized[num_features] = minmax_scaler.fit_transform(df_normalized[num_features])

print("Standardized Numerical Features Summary:")
display(df_scaled[num_features].describe().round(3).T)

# Export cleaned & preprocessed dataset
df_scaled.to_csv("cleaned_titanic.csv", index=False)
print("Saved final preprocessed data to 'cleaned_titanic.csv'")

# Visualize Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Feature Correlation Matrix (Cleaned Data)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

nb.cells = cells

with open('Task1_Data_Cleaning_Preprocessing.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook 'Task1_Data_Cleaning_Preprocessing.ipynb' created successfully!")
