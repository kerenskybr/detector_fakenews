from bs4 import BeautifulSoup

from urllib.request import urlopen

import requests
import time
import csv

import pandas as pd

#link = urlopen("https://brasil.elpais.com/seccion/tecnologia")

#soup = BeautifulSoup(link, "html.parser")

headers = {'User-Agent': 'Py_Scraper'}

link = "https://brasil.elpais.com/seccion/tecnologia" 

lista_titulo = []
lista_corpo = []
lista_url = []

var = True
pagina = 3 #30

lista = []

print('Inicializando...')


while var:

    endereco = str(link) + '/' + str(pagina)

    resposta = requests.get(endereco, headers=headers)    

    soup = BeautifulSoup(resposta.text, 'html.parser')
   
    try:
        for titulo in soup.find_all('h2',{'class':'articulo-titulo'}):
            try:

                print('TITULO:', titulo.getText())
                lista_titulo.append(titulo.getText())
            except:
                lista_titulo.append('NaN')

            for end in titulo.children:
                try:
                    print('ENDEREÇO')
                    print('https:' + end.get('href'))

                    visita = 'https:' + end.get('href')
                    lista_url.append(visita)
                except:
                    lista_url.append('NaN')
                
                
                #Coloca o link da materia em uma var nova
                link_2 = urlopen(str(visita))
                #Faz o parser da nova url
                soup_2 = BeautifulSoup(link_2, "html.parser")

                for corpo in soup_2.find_all('div', {'id':'cuerpo_noticia'}):
                    try:
                        print('CORPO', corpo.getText().replace('\n',''))
                        lista_corpo.append(corpo.getText().replace('\n', ''))
                    except:
                        lista_corpo.append('NaN')
       
    except:
        continue
        '''
        Caso algo saia errado ou a página chegue
        ao final, o arquivo é gravado
        '''
    print(endereco)
    print('##########################PAGINA##################', pagina)
          
    pagina-=1

    if pagina <= 0:
        break




#listass.to_csv('lista.csv')
'''
csv = pd.DataFrame()

csv['titulo'] = lista_titulo
csv['corpo'] = lista_corpo
csv['url'] = lista_url

csv.to_csv('el_pais_full.csv')



'''
s1 = pd.Series(lista_titulo, name='titulo')
s2 = pd.Series(lista_corpo, name='corpo')
s3 = pd.Series(lista_url, name='url')

s1.to_csv('s1.csv')
s2.to_csv('s2.csv')
s3.to_csv('s3.csv')