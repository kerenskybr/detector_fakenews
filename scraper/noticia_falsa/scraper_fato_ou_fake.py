#Scraper para o site fato ou fake

from bs4 import BeautifulSoup

from urllib.request import urlopen

import requests
import time
import csv

import pandas as pd


headers = {'User-Agent': 'Py_Scraper2'}

#https://g1.globo.com/fato-ou-fake/index/feed/pagina-7.ghtml

# "https://g1.globo.com/fato-ou-fake/" 

link = "https://g1.globo.com/fato-ou-fake/index/feed/pagina-" 

lista_titulo = []
lista_corpo = []
lista_url = []

var = True
pagina = 1

lista = []

print('Inicializando...')

csv = pd.DataFrame(columns=['titulo', 'url', 'corpo'])

while var:

    print('Trabalhando [...]')

    endereco = str(link) + str(pagina) + '.ghtml'

    print(endereco)

    resposta = requests.get(endereco, headers=headers) 

    #print(resposta.status_code)

    soup = BeautifulSoup(resposta.text, 'html.parser')
    
    pagina +=1

    
    #39 paginas
    if pagina >= 39:
        s1 = pd.Series(lista_titulo, name='titulo')
        s2 = pd.Series(lista_corpo, name='corpo')
        s3 = pd.Series(lista_url, name='url')

        final = pd.concat([s1, s2, s3], axis=1)

        final.to_csv('fato_ou_fake.csv')
        
        break
    
    try:

        #for titulo in soup.find_all('div',{'class':'feed-post-body-title gui-color-primary gui-color-hover '}):
        for titulo in soup.find_all('div',{'class':'feed-post-body-title gui-color-primary gui-color-hover '}):
            

            print('TITULO:', titulo.getText())
            lista_titulo.append(titulo.getText())                

            for end in titulo.children:

                visita = end.a['href']
                lista_url.append(visita)

                print('ENDEREÇO', visita)       

                #Coloca o link da materia em uma var nova
                link_2 = urlopen(str(visita))
                #Faz o parser da nova url
                soup_2 = BeautifulSoup(link_2, "html.parser")

                for corpo in soup_2.find_all('div', {'class':'mc-column content-text active-extra-styles active-capital-letter'}):
                    
                    print('CORPO', corpo.getText().replace('\n',''))
                    lista_corpo.append(corpo.getText().replace('\n', ''))
                   
    except:

        continue

