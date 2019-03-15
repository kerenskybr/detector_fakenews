from detectorfakenews import app
from detectorfakenews.forms import FormConsulta

from flask import render_template, url_for

from bs4 import BeautifulSoup

from urllib.request import urlopen 

import requests
import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import selenium as se
from selenium import webdriver

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer as count_vect

import re

def web_driver(path_to_driver, url):
	'''Função que carrega o driver do chrome'''

	options = webdriver.ChromeOptions()
	options.add_argument('headless')

	driver = webdriver.Chrome('../chrome_driver/chromedriver', options=options)
	driver.get(url)
	source = driver.page_source
	
	return BeautifulSoup(source, 'html.parser')


def remove_html_tags(text):
    """Remove html tags from a string"""
    
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


@app.route('/', methods=['GET', 'POST'])
def index():

	prev = []

	titulo = []

	##
	corpo = []

	exibe_titulo = ''

	exibe_corpo = ''

	carrega_modelo = joblib.load(os.path.join(app.root_path, 'saves/logistic_regression/modelo_reg_log.sav'))	

	tfidf_load = joblib.load(os.path.join(app.root_path, 'saves/logistic_regression/tfid_saved.sav'))	

	form = FormConsulta()

	if form.validate_on_submit():
		
		url = form.noticia.data

		resposta = requests.get(url)

		soup = BeautifulSoup(resposta.text, 'html.parser')

		titulo = [soup.title.string]

		exibe_titulo = str(titulo)[2:-2]
		
		for corpo in soup.find_all(id='cuerpo_noticia'):
			try:
				corpo_texto = str(corpo.p)
				
				exibe_corpo = remove_html_tags(corpo_texto)

			except:
				exibe_corpo = "Desculpe, não consegui encontrar o texto desta notícia."

		#exibe_corpo = str(exibe_corpo)[13:100] + ' . . .'
		
		texto_fit = tfidf_load.transform(titulo)
		
		prev = carrega_modelo.predict(texto_fit)

		#print(carrega_modelo.score(texto_fit, yteste))
		

	return render_template('index.html', form=form, exibe_titulo=exibe_titulo, exibe_corpo=exibe_corpo, prev=prev)


@app.route('/notas_versao')
def notas_versao():
	pass