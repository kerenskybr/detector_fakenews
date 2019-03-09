from bs4 import BeautifulSoup

from urllib.request import urlopen

import requests

import csv

import pandas as pd

#link = urlopen("https://brasil.elpais.com/seccion/tecnologia")

#soup = BeautifulSoup(link, "html.parser")

headers = {'User-Agent': 'Py_Scraper'}

link = "https://www.boatos.org/entretenimento/page" 

lista_titulo = []
lista_corpo = []
lista_url = []

var = True
pagina = 17

lista = []

print('Inicializando...')


while var:

    endereco = str(link) + '/' + str(pagina)

    resposta = requests.get(endereco, headers=headers)    

    soup = BeautifulSoup(resposta.text, 'html.parser')
   
    try:
        for titulo in soup.find_all('h2',{'class':'entry-title'}):
            try:

                print('TITULO:', titulo.getText().replace('\n', ''))
                lista_titulo.append(titulo.getText().replace('\n', ''))
               
            except:
                lista_titulo.append('NaN')

            for end in titulo.find_all('a'):
                try:
                    print('ENDEREÇO')
                    print(end.get('href'))

                    visita = str(end.get('href'))
                    lista_url.append(visita)

                    #print('VISITA',visita)
                except:
                    lista_url.append('NaN')
                
                
                #Coloca o link da materia em uma var nova
                link_2 = requests.get(visita)
                #Faz o parser da nova url
                #print('LINK 2',link_2)

                soup_2 = BeautifulSoup(link_2.text, "html.parser")

                #print(soup_2.find_all('p'))
                #lista_corpo.append(soup_2.find_all('p'))
                
                for corpo in soup_2.find('div', {'id':'content'}):
                    try:
                        print('CORPO', corpo.getText().replace('\n',''))
                        #print('CORPO', corpo.getText().replace('\n',''))
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
    print('PAGINA: ', pagina)
          
    pagina-=17

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

final = pd.concat([s1, s2, s3], axis=1)

final.to_csv('boatos_org_mundo.csv')
