import pandas as pd
base = pd.read_csv('risco_credito2.csv')

previsores = base.iloc[:, 0: 4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])

from sklearn.linear_model import LogisticRegression

classificador = LogisticRegression()
classificador.fit(previsores, classe)

print(classificador.intercept_)

resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])
resultado2 = classificador.predict_proba([[0,0,1,2], [3,0,0,0]])

print(resultado2)
