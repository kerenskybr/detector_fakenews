#Algoritmo usando agora o corpo das noticias


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
from sklearn.linear_model import LinearRegression

import nltk

falsa = pd.read_csv(r'../scraper/noticia_falsa/csv_novo/boatos.org.csv')

ciencia = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/ciencia.csv') 
cultura = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/cultura.csv')
economia = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/economia.csv')
esportes = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/esportes.csv')
estilo = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/estilo.csv')
internacional = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/internacional.csv')
politica = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/politica.csv')
tecnologia = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/tecnologia.csv')

verdadeira = pd.concat([ciencia, cultura, economia, esportes, estilo, internacional, politica, tecnologia])

verdadeira = verdadeira.drop(columns=['id', 'titulo', 'url'])

falsa = falsa.drop(columns=['id', 'link', 'timestamp'])


#Removendo números do dataframe

verdadeira['corpo'] = verdadeira['corpo'].str.replace(r'\d+','')

falsa['corpo'] = falsa['corpo'].str.replace(r'\d+', '')


#Transformando letras todas para minúsculas

verdadeira['corpo'] = verdadeira['corpo'].str.lower()

falsa['corpo'] = falsa['corpo'].str.lower()

#Removendo ruídos

verdadeira['corpo'] = verdadeira['corpo'].str.replace('(){$%[^\w\s]','')

falsa['corpo'] = falsa['corpo'].str.replace('(){$%[^\w\s]','')

#Mantendo apenas os 1000 primeiros caracteres
verdadeira['corpo'] = verdadeira['corpo'].map(lambda x: str(x)[:1000])

#Aleatorizando as linhas, para que os assuntos se misturem
verdadeira.sort_values(by='corpo', inplace=True)

#Criando um novo indice apos randomizar as linhas
verdadeira.reset_index(drop=True, inplace=True)


#Deixando ambos datasets do mesmo tamanho
verdadeira_droped = verdadeira.drop(verdadeira.index[1374:])

#Subistituindo NaN values por espaços
verda = verdadeira_droped.fillna(' ')

#Atribuindo a classe classificadora
# 1 para falso, 0 para verdadeiro

falsa['label'] = 1
verda['label'] = 0

#Label encoder transforma valores categoricos em numericos

label_encoder = preprocessing.LabelEncoder()

#Jutando os dois datasets em um so

dados = pd.concat([verda,falsa], axis=0)

y = label_encoder.fit_transform(dados.label.values)
x = dados.corpo.values

#Dividindo entre teste e treino
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
                        ngram_range = (1,3), sublinear_tf = 1, encoding='utf-8', stop_words = pt_stopwords)


#x_train = tfidf.fit_transform(x_train['corpo'].values.astype('U'))


x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

#################################
#Testando com Regressao Logistica
#################################

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
'''
arq2 = "tfid_saved.sav"
joblib.dump(tfidf, arq2)

arquivo = "modelo_reg_log.sav"
joblib.dump(classificador, arquivo)
'''

#########################
#Testando com Naive Bayes
#########################

clf = MultinomialNB().fit(x_train_tfidf,y_train)

preditor_nb = clf.predict(x_test_tfidf)

'''
arquivo = "modelo_naive.sav"
joblib.dump(clf, arquivo)
print('Modelo Naive Bayes salvo')
'''
print('Acuracia do modelo Naive Bayes: ',np.mean(preditor_nb == y_test))

#################
#Testando com SVM
#################

clf_svm = svm.SVC(gamma=0.001)
clf_svm.fit(x_train_tfidf, y_train)

clf_svm.predict(x_test_tfidf)

print('Acuracia do modelo SVM: ', clf_svm.score(x_test_tfidf, y_test, sample_weight=None))

reg = LinearRegression().fit(x_train_tfidf, y_train)
print('Acuracia do modelo Regressao Linear: ', reg.score(x_train_tfidf, y_train)*100)