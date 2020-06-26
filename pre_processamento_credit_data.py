import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]

# apagar a coluna

base.drop('age', 1, inplace=True)

base.drop(base[base.age < 0].index, inplace=True)

base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc(base.age < 0, 'age') = 40.92