# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 21:20:10 2023

@author: Rodrigo
"""

import pandas as pd
import glob
import os

# Juntar ficheiros dentro da mesma pasta
wd = os.getcwd()
wd = os.path.abspath(os.path.join(wd, os.pardir))
path = os.chdir(wd)
path = os.path.join(wd,'Original Data','StandVirtual')
all_files = glob.glob(os.path.join(path , "*.csv"))

li = []
for filename in all_files:
    try:
        dfs = pd.read_csv(filename, index_col=None, header=0)
        li.append(dfs)
    except pd.errors.ParserError as e:
        print(f"Error parsing {filename}: {str(e)}")
        continue

df = pd.concat(li, axis=0, ignore_index=True)

df.info()
df.describe()
df.nunique()

# Novos nomes para colunas
df.drop('Gancho de Reboque', axis = 1, inplace = True)

##### Se erro, Adicionar novas colunas aqui
novas = 'Anunciante','Marca','Modelo','Versao','Combustivel','Mes de Registo','Ano',\
'Quilometros','Cilindrada [cm3]','Potencia [cv]','Valor Fixo','IVA Discriminado','IVA Dedutivel',\
'Possibilidade de Finaciamento','Segmento','Cor','Tipo de Caixa','Numero de Portas','Lotacao',\
'Traccao','Garantia de stand - incluida no preco [meses]','Numero de Registos','Origem',\
'Livro de Revisoes Completo','Nao Fumador','Condicao','Preco','ID Anuncio','Data Anuncio',\
'Consumo Urbano','Tipo de Cor','Aceita Retoma','Numero de Mudancas','Classe do Veiculo',\
'Emissoes CO2 [g/km]','Duas Chaves','Consumo Extra Urbano','Bateria','Autonomia Maxima',\
'IUC','Sub-modelo','Garantia de Fabrica ate','Inspecao valida ate','Filtro de Particulas',\
'Ou Ate','Consumo Combinado','Autonomia Electrica [km]','Consumo [kWh/100km]',\
'Capacidade da Bateria [kWh]','Classico','Site','Portugal','Valor Sem ISV','Link','Data',\
    'Tempo de carregamento', 'VIN'

df.columns
df.columns = novas

# Limpeza das Colunas
df.drop_duplicates(subset=['ID Anuncio'], inplace=True)
df.dropna(subset=['Site'], inplace=True)

Meses = {'Janeiro':1,
'Fevereiro':2,
'Março':3,
'Abril':4,
'Maio':5,
'Junho':6,
'Julho':7,
'Agosto':8,
'Setembro':9,
'Outubro':10,
'Novembro':11,
'Dezembro':12}

df['Mes de Registo'] = df['Mes de Registo'].replace(Meses)
df['Mes de Registo'] = pd.to_numeric(df['Mes de Registo'], errors = 'coerce')

df['Quilometros'] = df['Quilometros'].str.replace('km', '').str.replace(' ', '')
df['Quilometros'] = pd.to_numeric(df['Quilometros'])
df['Quilometros'] = pd.to_numeric(df['Quilometros'], errors = 'coerce').fillna(0)

df.loc[df['Combustivel'].isin(['Eléctrico']), 'Cilindrada [cm3]'] = df.loc[df['Combustivel'].isin(['Eléctrico']), 'Cilindrada [cm3]'].fillna(0)
df['Cilindrada [cm3]'] = df['Cilindrada [cm3]'].astype(str).str.replace(' ','').str.replace('cm3','')
df['Cilindrada [cm3]'] = pd.to_numeric(df['Cilindrada [cm3]'])

df['Potencia [cv]'] = df['Potencia [cv]'].astype(str).str.replace(' cv','')
df['Potencia [cv]'] = pd.to_numeric(df['Potencia [cv]'])

df['Garantia de stand - incluida no preco [meses]'] = df['Garantia de stand - incluida no preco [meses]'].str.replace(' Meses','').str.replace(' ','').fillna(0)
df['Garantia de stand - incluida no preco [meses]'] = pd.to_numeric(df['Garantia de stand - incluida no preco [meses]'])

df['Preco'] = df['Preco'].str.replace(' EUR','').str.replace(' ','').str.replace(',','.')
df['Preco'] = pd.to_numeric(df['Preco'])

df['Consumo Urbano'] = df['Consumo Urbano'].str.replace(' l/100km','').str.replace(',','.')
df['Consumo Urbano'] = pd.to_numeric(df['Consumo Urbano'], errors = 'coerce')

df['Consumo Extra Urbano'] = df['Consumo Extra Urbano'].str.replace(' l/100km','').str.replace(',','.')
df['Consumo Extra Urbano'] = pd.to_numeric(df['Consumo Extra Urbano'], errors = 'coerce')

df['Consumo Combinado'] = df['Consumo Combinado'].str.replace(' l/100km','').str.replace(',','.')
df['Consumo Combinado'] = pd.to_numeric(df['Consumo Combinado'], errors = 'coerce')

df['Emissoes CO2 [g/km]'] = df['Emissoes CO2 [g/km]'].str.replace(' g/km','').str.replace(',','.')
df['Emissoes CO2 [g/km]'] = pd.to_numeric(df['Emissoes CO2 [g/km]'], errors = 'coerce')

df['IUC'] = df['IUC'].str.replace(' €','').str.replace(',','.')
df['IUC'] = pd.to_numeric(df['IUC'], errors = 'coerce')

df['Consumo [kWh/100km]'] = df['Consumo [kWh/100km]'].str.replace(' kWh/100km','').str.replace(',','.')
df['Consumo [kWh/100km]'] = pd.to_numeric(df['Consumo [kWh/100km]'], errors = 'coerce')

df['Autonomia Maxima'] = df['Autonomia Maxima'].str.replace(' km','').str.replace(',','.')
df['Autonomia Maxima'] = pd.to_numeric(df['Autonomia Maxima'], errors = 'coerce')

df['Autonomia Electrica [km]'] = df['Autonomia Electrica [km]'].str.replace(' km','').str.replace(',','.')
df['Autonomia Electrica [km]'] = pd.to_numeric(df['Autonomia Electrica [km]'], errors = 'coerce')

df['Capacidade da Bateria [kWh]'] = df['Capacidade da Bateria [kWh]'].str.replace(' kWh','').str.replace(',','.')
df['Capacidade da Bateria [kWh]'] = pd.to_numeric(df['Capacidade da Bateria [kWh]'], errors = 'coerce')

df['Classico'] = df['Classico'].fillna('Nao')

df['Site'] = df['Site'].fillna('StandVirtual')
df['Portugal'] = df['Portugal'].fillna('Yes')

df = df.reset_index(drop=True)

df.info() # Dtype=object?
df.describe()
df.nunique()

df.drop(['Valor Fixo', 'IVA Discriminado','IVA Dedutivel','Possibilidade de Finaciamento',
         'Traccao','Lotacao','Numero de Registos','Livro de Revisoes Completo','Nao Fumador',
         'Tipo de Cor','Aceita Retoma', 'Duas Chaves','Classe do Veiculo','Garantia de Fabrica ate',
         'Inspecao valida ate','Filtro de Particulas','Ou Ate','Tempo de carregamento', 'VIN'], axis = 1, inplace = True)

file = os.path.join('Prepared Data','StandVirtual_cleaned.csv')
path = os.getcwd()
full_path = os.path.join(path, file)
df.to_csv(full_path, index = False)

#######################################
# import pandas as pd
# from docx import Document
# from io import StringIO

# # Assume you have a DataFrame

# # Create a StringIO object
# output = StringIO()
# df.info(buf=output)
# info_str = output.getvalue()

# # Now you can use info_str as any other string
# doc = Document()
# doc.add_paragraph(info_str)
# doc.save('df_info.docx')
