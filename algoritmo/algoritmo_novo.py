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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier


import nltk


from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

#Criando dataframes para as noticias falsas
#Creating dataframes to fake news

#falsa_boatos = pd.read_csv(r'../scraper/noticia_falsa/csv_novo/boatos.org.csv')
#falsa_ff = pd.read_csv(r'../scraper/noticia_falsa/fato_ou_fake.csv')

falsa = pd.read_csv(r'../scraper/noticia_falsa/csv_falso_final.csv')

#Criando data frames para as noticias verdadeiras
#Creating dataframes for real news

ciencia = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/ciencia.csv') 
cultura = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/cultura.csv')
economia = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/economia.csv')
esportes = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/esportes.csv')
estilo = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/estilo.csv')
internacional = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/internacional.csv')
politica = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/politica.csv')
tecnologia = pd.read_csv(r'../scraper/noticia_verdadeira/csv_novo/tecnologia.csv')

verdadeira = pd.concat([ciencia, cultura, economia, esportes, estilo, internacional, politica, tecnologia])

#falsa_boatos = falsa_boatos.drop(columns=['id','link','timestamp'])

#falsa_ff = falsa_ff.drop(columns=['id','titulo', 'corpo', 'url'])

#falsa_ff['corpo_titulo'] = falsa_ff['titulo'].map(str) + falsa_ff['corpo'].map(str)



#falsa_ff['corpo'] =  falsa_ff['corpo_titulo'] + falsa_boatos['corpo']

#falsa = pd.concat([falsa_ff, falsa_boatos], axis=0, sort=True)

#Excluido colunas desnecesssarias
verdadeira = verdadeira.drop(columns=['id', 'titulo', 'url'])

falsa = falsa['corpo']

falsa = falsa.reset_index()

#print('shape noticias verdadeiras',verdadeira.shape)
#print('shape noticias falsa',falsa.shape)

#print('Head noticia falsa', falsa.head())






#Removendo números e espaços do dataframe
#Removing numbers

verdadeira['corpo'] = verdadeira['corpo'].str.replace(r'\d+',' ')

falsa['corpo'] = falsa['corpo'].str.replace(r'\d+', ' ')


#Transformando letras todas para minúsculas
#Transforming all data to lower case

verdadeira['corpo'] = verdadeira['corpo'].str.lower()

falsa['corpo'] = falsa['corpo'].str.lower()

#Removendo ruídos

verdadeira['corpo'] = verdadeira['corpo'].str.replace('|""=(){$%[^\w\s]','')

falsa['corpo'] = falsa['corpo'].str.replace('|""=(){$%[^\w\s]','')

#Mantendo apenas os 1000 primeiros caracteres
verdadeira['corpo'] = verdadeira['corpo'].map(lambda x: str(x)[:500])

falsa['corpo'] = falsa['corpo'].map(lambda y: str(y)[:500])

#print(verdadeira['corpo'].str.len())

#print(falsa['corpo'].str.len())

#Aleatorizando as linhas, para que os assuntos se misturem
verdadeira.sort_values(by='corpo', inplace=True)

#Criando um novo indice apos randomizar as linhas
verdadeira.reset_index(drop=True, inplace=True)

#Deixando ambos datasets do mesmo tamanho
verdadeira_droped = verdadeira.drop(verdadeira.index[1744:])


#Subistituindo NaN values por espaços ou excluindo (deu no mesmo)
#verda = verdadeira_droped.fillna('')
verda = verdadeira_droped.dropna()

#falsa = falsa.dropna()

#Atribuindo a classe classificadora
# 1 para falso, 0 para verdadeiro

falsa['label'] = 1
verda['label'] = 0

falsa = falsa.drop(columns=['index'])

print('shape noticias verdadeiras depois drop',verda.shape)
print('shape noticias falsa depois drop',falsa.shape)

#Label encoder transforma valores categoricos em numericos
#Label encoder transform categorical data into numerical data

#label_encoder = preprocessing.LabelEncoder()

#Jutando os dois datasets em um so

dados = pd.concat([verda,falsa], axis=0, sort=True)

#y = label_encoder.fit_transform(dados.label.values)
y = dados.label.values
x = dados.corpo.values.astype('U')


#Dividindo entre teste e treino
#Stratify = retorna mesma proporção
x_train, x_test, y_train, y_test = train_test_split(x, y, 
	random_state=43, test_size=.15, stratify=y)

#print(collections.Counter(y_train))

#arq3 = "y_test.sav"
#joblib.dump(y_test, arq3)

#collections.Counter(y_test)

#print(x_train.shape)
#print(x_test.shape)

#TF-IDF

#Baixando o 'bag of words'
#nltk.download('stopwords')

#Vetorizando palavras para numeros

pt_stopwords = set(nltk.corpus.stopwords.words('portuguese'))

tfidf = TfidfVectorizer(min_df = 1, strip_accents = 'unicode', max_features = 3000,
						analyzer = 'word', ngram_range = (1,3), sublinear_tf = 1, 
						encoding='utf-8', stop_words = pt_stopwords)


#x_train = tfidf.fit_transform(x_train['corpo'].values.astype('U'))


x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

##############################
# Plots
##############################
'''
n_samples = 30
degrees = [1, 4, 15]

plt.figure(figsize=(140, 50))
for i in range(len(degrees)):
		ax = plt.subplot(1, len(degrees), i + 1)
		plt.setp(ax, xticks=(), yticks=())

		polynomial_features = PolynomialFeatures(degree=degrees[i],
																						 include_bias=False)
		linear_regression = LinearRegression()
		pipeline = Pipeline([("polynomial_features", polynomial_features),
												 ("linear_regression", linear_regression)])
		pipeline.fit(x[:, np.newaxis], y)

		# Evaluate the models using crossvalidation
		scores = cross_val_score(pipeline, x[:, np.newaxis], y,
														 scoring="neg_mean_squared_error", cv=10)

		X_test = np.linspace(0, 1, 100)
		plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
		plt.plot(X_test, true_fun(X_test), label="True function")
		plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.xlim((0, 1))
		plt.ylim((-2, 2))
		plt.legend(loc="best")
		plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
				degrees[i], -scores.mean(), scores.std()))
plt.show()
'''
#################################
#Testando com Regressao Logistica
#################################

classificador = LogisticRegression()
classificador.fit(x_train_tfidf, y_train)

preditor_lr = classificador.predict_proba(x_test_tfidf)

print('Acuracia do modelo Regressão Logística: '
	,classificador.score(x_test_tfidf, y_test))

clf_log_reg = make_pipeline(TfidfVectorizer(), LogisticRegression())


scores_a = cross_val_score(clf_log_reg, x, y, cv=5)
print('Validação Cruzada Reg Log', scores_a) 


mse_lr = mean_squared_error(y_test, classificador.predict(x_test_tfidf))
print("MSE_LR: %.4f" % mse_lr)

#scores = cross_val_score(preditor_lr, x_train_tfidf, x_test_tfidf, cv=5)

#print('Validação cruzada',scores)



#Salvando o modelo

arquivo = "modelo_reg_log.sav"
joblib.dump(classificador, arquivo)

'''
arq2 = "tfid_saved.sav"
joblib.dump(tfidf, arq2)

arquivo = "modelo_reg_log.sav"
joblib.dump(classificador, arquivo)
'''

#########################
#Testando com Naive Bayes
#########################

clf_nb = MultinomialNB().fit(x_train_tfidf,y_train)

preditor_nb = clf_nb.predict(x_test_tfidf)

print('Acuracia do modelo Naive Bayes: ',np.mean(preditor_nb == y_test))

clf_naive = make_pipeline(TfidfVectorizer(), MultinomialNB())

scores = cross_val_score(clf_naive, x, y, cv=5)
print('Validação Cruzada Naive', scores) 

mse_nb = mean_squared_error(y_test, clf_nb.predict(x_test_tfidf))
print("MSE_NB: %.4f" % mse_nb)


arquivo = "modelo_naive.sav"
joblib.dump(clf_nb, arquivo)
print('Modelo Naive Bayes salvo')


#################
#Testando com SVM
#################
'''
clf_svm = svm.SVC(gamma=0.001, kernel='linear')
clf_svm.fit(x_train_tfidf, y_train)

clf_svm.predict(x_test_tfidf)

print('Acuracia do modelo SVM: ', clf_svm.score(x_test_tfidf, y_test, sample_weight=None))

mse_svm = mean_squared_error(y_test, clf_svm.predict(x_test_tfidf))
print("MSE_SVM: %.4f" % mse_svm)

arquivo = "teste_svm.sav"
joblib.dump(clf_svm, arquivo)
'''

tfidf_save = "tfid_teste.sav"
joblib.dump(tfidf, tfidf_save)

# Acerto Miseravi # 
texto = ["No dia 18 de dezembro de 2018 um meteoroide do tamanho de um ônibus explodiu na atmosfera a Terra com um impacto energético de 10 bombas atômicas. Esta foi a segunda maior explosão desde que a NASA começou a registrar esses impactos há 30 anos. O maior impacto de um meteoroide já registado foi o de fevereiro de 2013 sobre a Rússia. A explosão mais recente teve apenas 40% da liberação de energia da anterior. Apesar de toda esta intensidade, ninguém viu a explosão. Ao contrário do meteoroide de 2013 que foi visto, registrado e sentido por milhares de pessoas na cidade de Chelyabinsk, o impacto de 2018 ocorreu sobre o Mar de Bering entre a Sibéria e o Alasca. Esta região é bastante isolada."]

#Acerto miseravi # 
#texto = ["Tomar chá de erva-doce é a melhor forma de se curar da gripe H1N1 já que a fórmula do Tamiflu, principal remédio para o tratamento "]

# Acerto miseravi # texto = ['O presidente Jair Bolsonaro foi o grande vencedor do Prêmio Esso de Jornalismo na categoria Fakenews. Bolsonaro ganhou com o trabalho "A repórter que.']

#texto = ['Uma senhora foi agredida no Rio de Janeiro ao gritar Bolsonaro. A foto da senhora com o rosto machucado choca o eleitor. Mas o que choca mesmo é que a senhora é Beatriz Segall, morta no início de setembro.']

#texto = ['Em visita aos estados unidos, o presidente Jair bolsonaro da o cu para donald trump']

#texto = texto.str.replace('""=(){$%[^\w\s]','')

#texto = texto.str.lower()

carrega_modelo = joblib.load('modelo_reg_log.sav')
carrega_tfidf = joblib.load('tfid_teste.sav')

texto_fit = carrega_tfidf.transform(texto)

prev = carrega_modelo.predict(texto_fit)

if prev == 0:
    print('Noticia Classificada como verdadeira')
else:
    print('Noticia Classificada como Falsa')

'''
reg = LinearRegression().fit(x_train_tfidf, y_train)
print('Acuracia do modelo Regressao Linear: ', reg.score(x_train_tfidf, y_train)*100)
'''
'''
clf = make_pipeline(TfidfVectorizer(), svm.SVC(kernel='linear'))

scores = cross_val_score(clf, x, y, cv=5)
print('Validação Cruzada svm', scores) 
'''

'''
params = {'n_estimators': 333, 'max_depth': 8, 'min_samples_split': 2,
					'learning_rate': 0.001, 'random_state': 0, 'loss':'deviance'}

clf_gra = GradientBoostingClassifier(**params).fit(x_train_tfidf, y_train)
print(clf_gra.score(x_test_tfidf, y_test))

mse = mean_squared_error(y_test, clf_gra.predict(x_test_tfidf))
print("MSE: %.4f" % mse)

'''
'''
arquivo = "teste_gra.sav"
joblib.dump(clf_gra, arquivo)

tfidf_save = "tfid_gra.sav"
joblib.dump(tfidf, tfidf_save)
'''

'''
carrega_modelo = joblib.load('teste_gra.sav')
carrega_tfidf = joblib.load('tfid_gra.sav')

texto_fit = carrega_tfidf.transform(texto)

prev = carrega_modelo.predict(texto_fit)


print('Noticia teste predito como: ', prev)
'''
# Plot training deviance

# compute test set deviance

'''
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf_gra.staged_predict(x_test_tfidf)):
		test_score[i] = clf_svm.loss_(y_test, y_pred)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf_svm.train_score_, 'b-',
				 label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
				 label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

plt.show()
'''