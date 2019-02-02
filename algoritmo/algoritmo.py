#Desenvolvimento do algoritmo svm
#Utilizado:
#bag of words, 

import os
import collections

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer as count_vect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

import nltk

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

print(collections.Counter(y_train))

collections.Counter(y_test)

print(x_train.shape)
print(x_test.shape)

#TF-IDF

#Baixando o 'bag of words'
#nltk.download('stopwords')

pt_stopwords = set(nltk.corpus.stopwords.words('portuguese'))

tfidf = TfidfVectorizer(min_df = 3, strip_accents = 'unicode', max_features = 3000,
                        analyzer = 'word', token_pattern = '\w{1,}',
                        ngram_range = (1,3), sublinear_tf = 1, stop_words = pt_stopwords)

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

#Testando com Regressao Logistica

classificador = LogisticRegression()
classificador.fit(x_train_tfidf, y_train)

preditor_lr = classificador.predict_proba(x_test_tfidf)

print('Acuracia do modelo LR: ',classificador.score(x_test_tfidf, y_test))

#Testando com Naive Bayes
clf = MultinomialNB().fit(x_train_tfidf,y_train)

preditor_nb = clf.predict(x_test_tfidf)


print('Acuracia do modelo NB: ',np.mean(preditor_nb == y_test))

#Testando com SVM

clf_svm = svm.SVC(gamma=0.001)
clf_svm.fit(x_train_tfidf, y_train)

clf_svm.predict(x_test_tfidf)

print('Acuracia do modelo SVM: ', clf_svm.score(x_test_tfidf, y_test, sample_weight=None))