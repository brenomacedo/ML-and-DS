# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:07:10 2020

@author: Breno
"""


import pandas as pd
base = pd.read_csv('census.csv')

previsores =  base.iloc[:, 0:14].values
race = base.iloc[:, 8:9]
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
labelencoder_race = LabelEncoder()
# labels = labelencoder_previsores.fit_transform(previsores[:, 1])

race[:, 0] = labelencoder_race.fit_transform(race[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

onehotencoder = OneHotEncoder(categorical_features=[8])
previsores = onehotencoder.fit_transform(previsores).toarray()