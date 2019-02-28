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
from sklearn.externals import joblib

import nltk

falsa = pd.read_csv(r'../scraper/noticia_falsa/csv/boatos_org.csv')
verdadeira = pd.read_csv(r'../scraper/noticia_verdadeira/csv/elpais.csv')


falsa = falsa.drop(columns=['quant', 'tema'])

#dropando itens para que ambos datasets tenham mesmo tamanho
#ate arranjar mais dados para o outro
verdadeira = verdadeira.drop(verdadeira.index[2714:])

verdadeira = verdadeira.drop(columns=['quant'])

#Atribuindo a classe classificadora
# 1 para falso, 0 para verdadeiro

falsa['label'] = 1
verdadeira['label'] = 0

#print(falsa.head(5))

#print(verdadeira.head(5))

#Jutando os dois datasets em um so

dados = pd.concat([verdadeira,falsa])

#print(dados)

#Label encoder transforma valores categoricos em numericos

label_encoder = preprocessing.LabelEncoder()

y = label_encoder.fit_transform(dados.label.values)
x = dados.titulo_noticia.values

#print('printando x',x)

#Stratify = retorna mesma proporção
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=43, test_size=.33, stratify=y)

#print(collections.Counter(y_train))

#arq3 = "y_test.sav"
#joblib.dump(y_test, arq3)



collections.Counter(y_test)

#print(x_train.shape)
#print(x_test.shape)

#TF-IDF

#Baixando o 'bag of words'
#nltk.download('stopwords')

#Vetorizando palavras para numeros

pt_stopwords = set(nltk.corpus.stopwords.words('portuguese'))

tfidf = TfidfVectorizer(min_df = 3, strip_accents = 'unicode', max_features = 3000,
                        analyzer = 'word', token_pattern = '\w{1,}',
                        ngram_range = (1,3), sublinear_tf = 1, stop_words = pt_stopwords)


x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

#################################
#Testando com Regressao Logistica
#################################


#testando = tfidf.transform(texto_teste)

#print('TESTANDO', testando)

classificador = LogisticRegression()
classificador.fit(x_train_tfidf, y_train)

preditor_lr = classificador.predict_proba(x_test_tfidf)



print('Acuracia do modelo Regressão Logística: '
	,classificador.score(x_test_tfidf, y_test))

'''

#Salvando o modelo

arquivo = "modelo_reg_log.sav"
joblib.dump(classificador, arquivo)
'''
arq2 = "tfid_saved.sav"
joblib.dump(tfidf, arq2)

arquivo = "modelo_reg_log.sav"
joblib.dump(classificador, arquivo)


#########################
#Testando com Naive Bayes
#########################
'''
clf = MultinomialNB().fit(x_train_tfidf,y_train)

preditor_nb = clf.predict(x_test_tfidf)


arquivo = "modelo_naive.sav"
joblib.dump(clf, arquivo)
print('Modelo Naive Bayes salvo')

print('Acuracia do modelo Naive Bayes: ',np.mean(preditor_nb == y_test))

#################
#Testando com SVM
#################

clf_svm = svm.SVC(gamma=0.001)
clf_svm.fit(x_train_tfidf, y_train)

clf_svm.predict(x_test_tfidf)

print('Acuracia do modelo SVM: ', clf_svm.score(x_test_tfidf, y_test, sample_weight=None))

'''