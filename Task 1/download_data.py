import seaborn as sns
import pandas as pd

print("Loading Titanic dataset from seaborn...")
df = sns.load_dataset('titanic')

# Rename columns to standard Kaggle format for full consistency
column_mapping = {
    'survived': 'Survived',
    'pclass': 'Pclass',
    'sex': 'Sex',
    'age': 'Age',
    'sibsp': 'SibSp',
    'parch': 'Parch',
    'fare': 'Fare',
    'embarked': 'Embarked',
    'class': 'Class',
    'deck': 'Cabin',
    'embark_town': 'Embark_Town',
    'alive': 'Alive',
    'alone': 'Alone'
}
df.rename(columns=column_mapping, inplace=True)

# Add synthetic PassengerId, Name, Ticket to match Kaggle raw structure exactly if missing
df.insert(0, 'PassengerId', range(1, len(df) + 1))
df['Name'] = [f"Passenger_{i}" for i in df['PassengerId']]
df['Ticket'] = [f"TICK_{1000+i}" for i in df['PassengerId']]

df.to_csv("titanic.csv", index=False)
print(f"Successfully saved titanic.csv with {len(df)} rows and {len(df.columns)} columns!")
