# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:50:03 2020

@author: User
"""


import pandas as pd
base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:, 0: 4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
