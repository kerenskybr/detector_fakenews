#Scraper para o site boatos.org
import os
import re
import time

from bs4 import BeautifulSoup
from urllib.request import urlopen

import selenium as se
from selenium import webdriver

import pandas as pd


def web_driver(path_to_driver, url):
	options = webdriver.ChromeOptions()
	options.add_argument('headless')

	driver = webdriver.Chrome('/home/roger/Documents/detector_fakenews/chrome_driver/chromedriver', 
									options=options)
	driver.get(url)
	source = driver.page_source
	
	return BeautifulSoup(source, 'html.parser')


lista_link = []
lista_titulo = []
lista_tema = []


path = r'/home/roger/Documents/detector_fakenews/scraper/datas_invertidas.txt'
dates = pd.read_csv(path, header=None, encoding="ISO-8859-1", sep=',')

with open("/home/roger/Documents/detector_fakenews/scraper/datas_invertidas.txt", "r") as ins:
    for line in ins:
	    
        a = web_driver(r'/home/roger/Documents/detector_fakenews/chrome_driver/chromedriver', 
        				r'http://www.boatos.org/' + str(line))

        try:

            for x in a.findAll('h2',{'class':'entry-title'}):
                s = str(x)

                #achar link da noticia
                a_pattern = r'www.(\w|\W){1,1000}.html'
                regex = re.search(a_pattern, s)
                link = s[regex.start():regex.end()]
                lista_link.append(link)

                print(lista_link)
                
                #achar titulo da noticia
                
                lista_titulo.append(x.getText())

                print(lista_titulo)
                
                #achar tema da noticia
                
                theme_pattern = r'org/(\w|\W){1,1000}/'
                regex_theme = re.search(theme_pattern, link)
                lista_tema.append(link[regex_theme.start():regex_theme.end()][4:-1])
                print(lista_tema)
                print('Trabalhando...')
            
                df = pd.DataFrame()
                df['Título da Notícia'] = lista_titulo
                df['Temas'] = lista_tema
                df['link'] = lista_link

                df.to_csv('boatos_org.csv')
        except:
            re.search('Ops!',str(a))
            break
        
df = pd.DataFrame()
df['Título da Notícia'] = lista_titulo
df['Tema'] = lista_tema
df['URL'] = lista_link

df.to_csv('boatos_org.csv')

