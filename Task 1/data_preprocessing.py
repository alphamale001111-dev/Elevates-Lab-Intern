import os
import matplotlib
matplotlib.use('Agg') # Headless backend for fast rendering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Set style
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

print("="*60)
print(" STEP 1: IMPORT DATASET & EXPLORE BASIC INFO ")
print("="*60)

df = pd.read_csv("titanic.csv")
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types & Non-Null Counts:")
print(df.info())

print("\nMissing Values Count:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])

print("\nStatistical Summary (Numerical Features):")
print(df.describe().T)

# Plot missing values visualization
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Values Heatmap (Raw Data)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/1_missing_values_heatmap.png", dpi=300)
plt.close()

print("\n" + "="*60)
print(" STEP 2: HANDLE MISSING VALUES ")
print("="*60)

# Make a working copy
df_clean = df.copy()

# 1. Impute missing Age with median
age_median = df_clean['Age'].median()
df_clean['Age'] = df_clean['Age'].fillna(age_median)
print(f"- Imputed missing 'Age' values with median: {age_median:.2f}")

# 2. Impute missing Embarked with mode
embarked_mode = df_clean['Embarked'].mode()[0]
df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)
print(f"- Imputed missing 'Embarked' values with mode: '{embarked_mode}'")

# 3. Handle Cabin: high missing percentage (~77%), create Cabin_Known binary feature or drop Cabin
df_clean['Has_Cabin'] = df_clean['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df_clean.drop(columns=['Cabin'], inplace=True)
print("- Created 'Has_Cabin' binary column and dropped raw 'Cabin' feature")

# 4. Drop irrelevant identifier columns for ML model prep (Name, Ticket, PassengerId)
identifiers = ['PassengerId', 'Name', 'Ticket']
df_clean.drop(columns=identifiers, inplace=True)
print(f"- Dropped identifier columns: {identifiers}")

print(f"\nRemaining Missing Values: {df_clean.isnull().sum().sum()}")

print("\n" + "="*60)
print(" STEP 3: CONVERT CATEGORICAL FEATURES TO NUMERICAL ENCODING ")
print("="*60)

print("Categorical features before encoding:")
print(df_clean.select_dtypes(include=['object']).columns.tolist())

# Label Encoding for Sex
le_sex = LabelEncoder()
df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
print(f"- Binary Encoded 'Sex': {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# One-Hot Encoding for Embarked
df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True, dtype=int)
print("- One-Hot Encoded 'Embarked' (created dummy columns)")

print("\nData Types after Encoding:")
print(df_clean.dtypes)

print("\n" + "="*60)
print(" STEP 4: VISUALIZE AND HANDLE OUTLIERS (IQR METHOD) ")
print("="*60)

# Plot Outliers BEFORE removal
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df_clean['Age'], ax=axes[0], color='skyblue')
axes[0].set_title("Boxplot of Age (Before Outlier Removal)", fontweight='bold')

sns.boxplot(y=df_clean['Fare'], ax=axes[1], color='salmon')
axes[1].set_title("Boxplot of Fare (Before Outlier Removal)", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/2_outliers_before.png", dpi=300)
plt.close()

# Calculate IQR for Fare and Age
print("Outlier Detection using Interquartile Range (IQR):")

def remove_outliers_iqr(dataframe, columns):
    df_out = dataframe.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_count = len(df_out)
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
        removed = initial_count - len(df_out)
        print(f"  - {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}. Removed {removed} outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")
    return df_out

df_no_outliers = remove_outliers_iqr(df_clean, ['Fare', 'Age'])
print(f"\nDataset size before outlier removal: {len(df_clean)}")
print(f"Dataset size after outlier removal: {len(df_no_outliers)}")

# Plot Outliers AFTER removal
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df_no_outliers['Age'], ax=axes[0], color='lightgreen')
axes[0].set_title("Boxplot of Age (After Outlier Removal)", fontweight='bold')

sns.boxplot(y=df_no_outliers['Fare'], ax=axes[1], color='lightcoral')
axes[1].set_title("Boxplot of Fare (After Outlier Removal)", fontweight='bold')
plt.tight_layout()
plt.savefig("plots/3_outliers_after.png", dpi=300)
plt.close()

print("\n" + "="*60)
print(" STEP 5: NORMALIZE & STANDARDIZE NUMERICAL FEATURES ")
print("="*60)

num_features = ['Age', 'Fare', 'SibSp', 'Parch']

# 1. Standardization (StandardScaler - mean 0, std 1)
scaler = StandardScaler()
df_scaled = df_no_outliers.copy()
df_scaled[num_features] = scaler.fit_transform(df_scaled[num_features])
print("Standardized Numerical Features using StandardScaler (Mean=0, Std=1):")
print(df_scaled[num_features].describe().round(3).T[['mean', 'std', 'min', 'max']])

# 2. Normalization (MinMaxScaler - scale [0, 1])
minmax_scaler = MinMaxScaler()
df_normalized = df_no_outliers.copy()
df_normalized[num_features] = minmax_scaler.fit_transform(df_normalized[num_features])
print("\nNormalized Numerical Features using MinMaxScaler (Range [0, 1]):")
print(df_normalized[num_features].describe().round(3).T[['mean', 'std', 'min', 'max']])

# Save final processed dataset (Standardized version)
cleaned_csv_path = "cleaned_titanic.csv"
df_scaled.to_csv(cleaned_csv_path, index=False)
print(f"\nFinal cleaned & preprocessed dataset saved to: '{cleaned_csv_path}'")

# Plot feature distributions after scaling
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_scaled['Age'], kde=True, ax=axes[0,0], color='blue')
axes[0,0].set_title("Scaled Age Distribution", fontweight='bold')

sns.histplot(df_scaled['Fare'], kde=True, ax=axes[0,1], color='green')
axes[0,1].set_title("Scaled Fare Distribution", fontweight='bold')

sns.histplot(df_scaled['SibSp'], kde=True, ax=axes[1,0], color='purple')
axes[1,0].set_title("Scaled SibSp Distribution", fontweight='bold')

sns.histplot(df_scaled['Parch'], kde=True, ax=axes[1,1], color='orange')
axes[1,1].set_title("Scaled Parch Distribution", fontweight='bold')

plt.tight_layout()
plt.savefig("plots/4_feature_distributions.png", dpi=300)
plt.close()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Feature Correlation Matrix (Cleaned & Scaled Data)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/5_correlation_matrix.png", dpi=300)
plt.close()

print("\n" + "="*60)
print(" DATA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY! ")
print("="*60)
