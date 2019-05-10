#Scraper para o site ceticismo_politico. Site famoso por propagar noticias
#falsas e achismos

from bs4 import BeautifulSoup

from urllib.request import urlopen

import requests
import time
import csv

import pandas as pd


headers = {'User-Agent': 'Py_Scraper2'}

link = "https://ceticismopolitico.com/page" 

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

    endereco = str(link) + '/' + str(pagina)

    print(endereco)

    resposta = requests.get(endereco, headers=headers) 

    print(resposta.status_code)

    soup = BeautifulSoup(resposta.text, 'html.parser')
    
    pagina +=1

    

    if pagina >= 25:
        s1 = pd.Series(lista_titulo, name='titulo')
        s2 = pd.Series(lista_corpo, name='corpo')
        s3 = pd.Series(lista_url, name='url')

        final = pd.concat([s1, s2, s3], axis=1)

        final.to_csv('ceticismo_politico.csv')
        
        break
    
    try:

        for titulo in soup.find_all('h3',{'class':'loop-title'}):
            

            print('TITULO:', titulo.getText())
            lista_titulo.append(titulo.getText())

            #csv.append({'titulo': lista_titulo}, ignore_index=True)
                

            for end in titulo.children:

                visita = end.get('href')
                lista_url.append(visita)

                print('ENDEREÇO', visita)       

                #Coloca o link da materia em uma var nova
                link_2 = urlopen(str(visita))
                #Faz o parser da nova url
                soup_2 = BeautifulSoup(link_2, "html.parser")

                for corpo in soup_2.find_all('div', {'class':'entry clearfix'}):
                    
                    print('CORPO', corpo.getText('p').replace('\n',''))
                    lista_corpo.append(corpo.getText('p').replace('\n', ''))

    except:

        continue

