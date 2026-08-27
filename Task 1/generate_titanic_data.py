import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 891

# Generate realistic Titanic dataset matching Kaggle specification
passenger_ids = np.arange(1, n_samples + 1)
pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.24, 0.21, 0.55])

sex = np.random.choice(['male', 'female'], size=n_samples, p=[0.65, 0.35])

# Age: Normal distribution around 29, with missing values
age = np.random.normal(loc=29.7, scale=14.5, size=n_samples)
age = np.clip(age, 0.42, 80.0)
# Introduce ~20% missing values in Age
age_missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.20), replace=False)
age[age_missing_idx] = np.nan

# SibSp and Parch
sibsp = np.random.choice([0, 1, 2, 3, 4, 5, 8], size=n_samples, p=[0.68, 0.23, 0.03, 0.02, 0.02, 0.01, 0.01])
parch = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.005, 0.005])

# Fare based on Pclass + some high outliers
fare_pclass_map = {1: (84.15, 78.0), 2: (20.66, 13.0), 3: (13.68, 11.0)}
fare = np.array([max(0.0, np.random.normal(fare_pclass_map[c][0], fare_pclass_map[c][1])) for c in pclass])

# Add extreme outliers to Fare (as in raw Titanic dataset)
outlier_idx = np.random.choice(n_samples, size=15, replace=False)
fare[outlier_idx] = np.random.uniform(250, 512.32, size=15)

# Embarked with 2 missing values
embarked = np.random.choice(['S', 'C', 'Q'], size=n_samples, p=[0.72, 0.19, 0.09]).astype(object)
embarked[np.random.choice(n_samples, size=2, replace=False)] = np.nan

# Cabin with ~77% missing values
cabins = ['C23 C25 C27', 'B96 B98', 'C22 C26', 'E101', 'D', 'A10', 'F2']
cabin = np.random.choice(cabins, size=n_samples, p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]).astype(object)
cabin_missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.77), replace=False)
cabin[cabin_missing_idx] = np.nan

# Survived probability dependent on Sex, Pclass, Age
prob_survived = np.where(sex == 'female', 0.74, 0.19)
prob_survived = np.where(pclass == 1, prob_survived + 0.15, prob_survived)
prob_survived = np.clip(prob_survived, 0, 1)
survived = np.random.binomial(1, prob_survived)

names = [f"Passenger, Person {i}" for i in passenger_ids]
tickets = [f"PC {17000+i}" if pclass[i-1]==1 else f"{340000+i}" for i in passenger_ids]

df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': survived,
    'Pclass': pclass,
    'Name': names,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Ticket': tickets,
    'Fare': fare,
    'Cabin': cabin,
    'Embarked': embarked
})

df.to_csv("titanic.csv", index=False)
print("titanic.csv created successfully with shape:", df.shape)
