#Desenvolvimento do algoritmo svm
#Utilizado:
#bag of words, 

import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer as count_vect
from sklearn.model_selection import train_test_split


falsa = pd.read_csv(r'../scraper/noticia_falsa/csv/boatos_org.csv')
verdadeira = pd.read_csv(r'../scraper/noticia_verdadeira/csv/elpais.csv')


falsa = falsa.drop(columns=['quant', 'tema'])

#dropando itens para que ambos datasets tenham mesmo tamanho
#ate arranjar mais dados para o outro
verdadeira = verdadeira.drop(verdadeira.index[900:])

verdadeira = verdadeira.drop(columns=['quant'])

# 1 para verdadeiro, 0 para falso

falsa['label'] = 1
verdadeira['label'] = 0

print(falsa.head(5))

print(verdadeira.head(5))

dados = pd.concat([verdadeira,falsa])

print(dados)

label_encoder = preprocessing.LabelEncoder()

y = label_encoder.fit_transform(dados.label.values)
x = dados.titulo_noticia.values

#Stratify = retorna mesma proporção
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=43, test_size=.33, stratify=y)


