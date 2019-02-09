from detectorfakenews import app

from flask import render_template, url_for

from detectorfakenews.forms import FormConsulta

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


import nltk

def web_driver(path_to_driver, url):
	'''Função que carrega o driver do chrome'''

	options = webdriver.ChromeOptions()
	options.add_argument('headless')

	driver = webdriver.Chrome('/home/roger/Documents/detector_fakenews/chrome_driver/chromedriver', 
									options=options)
	driver.get(url)
	source = driver.page_source
	
	return BeautifulSoup(source, 'html.parser')


@app.route('/', methods=['GET', 'POST'])
def index():

	prev = []

	titulo = []

	exibe_titulo = ''

	carrega_modelo = joblib.load(os.path.join(app.root_path, 'saves/modelo_reg_log.sav'))	

	tfidf_load = joblib.load(os.path.join(app.root_path, 'saves/tfid_saved.sav'))	

	yteste = joblib.load(os.path.join(app.root_path, 'saves/y_test.sav'))	


	form = FormConsulta()

	if form.validate_on_submit():
		
		url = form.noticia.data

		resposta = requests.get(url)

		soup = BeautifulSoup(resposta.text, 'html.parser')

		titulo = [soup.title.string]

		print(titulo)

		exibe_titulo = str(titulo)[2:-2]

		texto_fit = tfidf_load.transform(titulo)
		
		prev = carrega_modelo.predict(texto_fit)

		#print(carrega_modelo.score(texto_fit, yteste))
		

	return render_template('index.html', form=form, exibe_titulo=exibe_titulo, prev=prev)