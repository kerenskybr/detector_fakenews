from bs4 import BeautifulSoup

from urllib.request import urlopen

link = urlopen("https://brasil.elpais.com/seccion/tecnologia/29")

soup = BeautifulSoup(link, "html.parser")


for titulo in soup.find_all('h2',{'class':'articulo-titulo'}):
    print('TITULO:', titulo.getText())

    for end in titulo.children:
        print('ENDEREÇO')
        print('https:' + end.get('href'))

        visita = 'https:' + end.get('href')
        #print('VISITA', visita)

        link_2 = urlopen(str(visita))
        #print('LINK', link)

        soup_2 = BeautifulSoup(link_2, "html.parser")

        for corpo in soup_2.find_all('div', {'id':'cuerpo_noticia'}):
            print('Terceiro loop')
            print('CORPO', corpo.getText().split())
            
