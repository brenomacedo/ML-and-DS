# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:07:10 2020

@author: Breno
"""


import pandas as pd
base = pd.read_csv('census.csv')

previsores =  base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
# labels = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])

