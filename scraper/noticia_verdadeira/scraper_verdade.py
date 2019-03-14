#Scraper para o site folha el pais br

from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
import os
import re
import time
import pandas as pd

url = urlopen('https://brasil.elpais.com/seccion/politica')
bsObj =  BeautifulSoup(url, "html.parser" )

tags = pd.read_csv(r'/home/roger/Documents/detector_fakenews/scraper/elpais_tags.txt',header=None, sep=',')

tags.iloc[1,0]

for url in tags.iloc[:,1]:
    print(url)

list_url = []
list_title = []
list_news =[]
list_type = []

var = True

for url in tags.iloc[:,1][2:]:
    
    t = 0
    k = 1
    
    while var == True:
        time.sleep
        link = url + '/' + str(k)

        try:
            read_url = urlopen(link)
            bsObj =  BeautifulSoup(read_url, "html.parser" )
        except:
            var = False
            continue
       
        try:
            for obj in bsObj.findAll('h2',{'class':'articulo-titulo'}):
                print(obj.getText())
                list_title.append(obj.getText())

                for link in obj.children:
                    print('https:' + link.get('href'))
                    list_url.append('https:' + link.get('href'))
                    print(len(list_url)==len(list_title))
        except:
            continue
            
        k += 1
        print(k)
    t+=1
    var = True
    print(t)

csv = pd.DataFrame()

csv['title'] = list_title
csv['url'] = list_url

csv.to_csv('el_pais.csv')
