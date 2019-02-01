# Pequeno script para criar as datas que serão
# usadas no scraper do site boatos.org

# junho 2013 - janeiro 2019
# formato = 2013/06
import os 
import os.path
import csv 

ano = 2019

mes = 2

f= open("datas_invertidas.txt","w+")

while mes <= 13:
	
	data = str(ano) + '/0' + str(mes)	
	f.write(data + '\n')
	print(data)
	mes+=1

	if mes == 13:
		mes = 1
		ano-=1

	if ano == 2012:
		break