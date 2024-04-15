# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:38:05 2023

@author: Rodrigo

"""

import pandas as pd
import glob
import os

# Juntar ficheiros dentro da mesma pasta
os.chdir(r'C:\Users\Rodrigo\Documents\GitHub\ImportedCars\Original Data\AutoScout24')
wd = os.getcwd()
all_files = glob.glob(os.path.join(wd , "*.csv"))

li = []
for filename in all_files:
    try:
        dfs = pd.read_csv(filename, index_col=None, header=0,sep = ',')
        li.append(dfs)
    except pd.errors.ParserError as e:
        print(f"Error parsing {filename}: {str(e)}")
        continue

df = pd.concat(li, axis=0, ignore_index=True)

# Limpeza das colunas
df.drop_duplicates(subset=['ID'], inplace=True)
df.dropna(subset=['First Registration'], inplace=True) # Redudancia para apagar eventuais erros de extracao de dados
df.dropna(subset=['Site'], inplace=True) # Redudancia para apagar eventuais erros de extracao de dados

# Colocar valores corretos nas colunas corretas (g/km e l/100 km) 
df['Fuel Consumption'].fillna('', inplace=True)
df['CO2 Emissions'].fillna('', inplace=True)
df.loc[df['Fuel Consumption'].str.contains('Electricity'),'Fuel Consumption'] = ''

mask = df['Fuel Consumption'].str.contains(r'\bg/km\b')
df.loc[mask, 'CO2 Emissions'] = df.loc[mask, 'Fuel Consumption']
df.loc[mask, 'Fuel Consumption'] = ''

mask = df['CO2 Emissions'].str.contains(r'\bl/100 km\b')
df.loc[mask, 'Fuel Consumption'] = df.loc[mask, 'CO2 Emissions']
df.loc[mask, 'CO2 Emissions'] = ''

# Colocar valores de fuel consumption que tenham valores de autonomia a zero
mask = df['Fuel Consumption'].str.match(r'^\d+\s*km$')
df.loc[mask, 'Fuel Consumption'] = ''

mask = df['Fuel Consumption'].str.match(r'^A')
df.loc[mask, 'Fuel Consumption'] = ''

mask = df['CO2 Emissions'].str.match(r'^A\+*$|^C|^B')
df.loc[mask, 'CO2 Emissions'] = ''

# mask.value_counts()
# filtered_df = df[mask]

# 
df['Gearbox'] = df['Gearbox'].fillna('Automática')

# Apagar linhas que tenham outros tipos de combustiveis
df = df[~df['Fuel Type'].str.contains('CNG')].reset_index(drop=True)
df = df[~df['Fuel Type'].str.contains('LPG')].reset_index(drop=True)
df = df[~df['Fuel Type'].str.contains('-')].reset_index(drop=True)
df = df[~df['Fuel Type'].str.contains('Hydrogen')].reset_index(drop=True)
df = df[~df['Fuel Type'].str.contains('Others')].reset_index(drop=True)
df = df[~df['Fuel Type'].str.contains('Ethanol')].reset_index(drop=True)

# Substituir valores para correspondentes a StandVirtual
df['Seller'] = df['Seller'].replace('Private seller', 'Particular')
df['Seller'] = df['Seller'].replace('Dealer', 'Profissional')
df['Make'] = df['Make'].replace('Alpina', 'Alpine')
df['Make'] = df['Make'].replace('DS Automobiles', 'DS')
df['Make'] = df['Make'].replace('Rolls-Royce', 'Rolls Royce')
df['Make'] = df['Make'].replace('Citroen', 'Citroën')
df['Make'] = df['Make'].replace('Volkswagen', 'VW')
df['Fuel Type'] = df['Fuel Type'].replace('Electric', 'Eléctrico')
df['Fuel Type'] = df['Fuel Type'].replace('Gasoline', 'Gasolina')
df['Fuel Type'] = df['Fuel Type'].replace('Electric/Diesel', 'Híbrido (Diesel)')
df['Fuel Type'] = df['Fuel Type'].replace('Electric/Gasoline', 'Híbrido (Gasolina)')
df['Gearbox'] = df['Gearbox'].replace('Automatic', 'Automática')
df['Gearbox'] = df['Gearbox'].replace('Semi-Automatic', 'Automática')

df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').str.replace(',', '').str.split('-').apply(lambda x: x[0])
df['Price'] = pd.to_numeric(df['Price'])

df['Mileage'] = df['Mileage'].str.replace('km', '').str.replace(',', '')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors = 'coerce').fillna(1)

df['Month Registration'] = df['First Registration'].str.split('/').str[0]
df['Month Registration'] = pd.to_numeric(df['Month Registration'], errors = 'coerce')

df['Year Registration'] = df['First Registration'].str.split('/').str[1]
df['Year Registration'] = pd.to_numeric(df['Year Registration'] )

df['Power [kW]'] = df['Power'].str.split(' ').str[0]
df['Power [kW]'] = pd.to_numeric(df['Power [kW]'], errors = 'coerce')

df['Power [hp]'] = df['Power'].str.replace('(','').str.replace(')','').str.replace(',','').str.split(' ').str[2]
df['Power [hp]'] = pd.to_numeric(df['Power [hp]'])

df['Engine Size [cc]'] = df['Engine Size'].str.replace(',','').str.replace('cc','')
df['Engine Size [cc]'] = pd.to_numeric(df['Engine Size [cc]'])

df['Weight [kg]'] = df['Weight'].str.replace(',','').str.replace('kg','')
df['Weight [kg]'] = pd.to_numeric(df['Weight [kg]'])

df['Fuel Consumption [l/100 km] (comb.)'] = df['Fuel Consumption'].str.replace('(comb.)','').str.replace('(city)','').str.replace('(country)','').str.split('l/100 km').str[0]
df['Fuel Consumption [l/100 km] (comb.)'] = pd.to_numeric(df['Fuel Consumption [l/100 km] (comb.)'])

df['Fuel Consumption [l/100 km] (city)'] = df['Fuel Consumption'].str.replace('(comb.)','').str.replace('(city)','').str.replace('(','').str.replace(')','').str.replace('(country)','').str.replace('(','').str.replace(')','').str.split('l/100 km').str[1]
df['Fuel Consumption [l/100 km] (city)'] = pd.to_numeric(df['Fuel Consumption [l/100 km] (city)'], errors = 'coerce')

df['Fuel Consumption [l/100 km] (country)'] = df['Fuel Consumption'].str.replace('(comb.)','').str.replace('(city)','').str.replace('(','').str.replace(')','').str.replace('(country)','').str.replace('(','').str.replace(')','').str.split('l/100 km').str[2]
df['Fuel Consumption [l/100 km] (country)'] = pd.to_numeric(df['Fuel Consumption [l/100 km] (country)'], errors = 'coerce')

df['CO2 Emissions [g/km]'] = df['CO2 Emissions'].str.split('g/km').str[0].str.replace(',','.')
df['CO2 Emissions [g/km]'] = pd.to_numeric(df['CO2 Emissions [g/km]'])

df['Electric Range'] = df['Electric Range'].str.split(' ').str[0]
df['Electric Range'] = pd.to_numeric(df['Electric Range'])

df['Electric Consumption'] = df['Electric Consumption'].str.split(' ').str[0]
df['Electric Consumption'] = pd.to_numeric(df['Electric Consumption'])

df.drop(['Power','Weight', 'Engine Size', 'First Registration',  'Fuel Consumption', 'CO2 Emissions'], axis = 1, inplace = True)

######### Actualizar colunas nesta linha mas verificar ordem primeiro#############
novas = 'ID Anuncio','Marca','Modelo','Versao','Preco','Quilometros','Tipo de Caixa',\
    'Combustivel','Anunciante','Numero de Mudancas','Numero de Cilindros',\
        'Outros tipos de Combustivel','Classe Emissoes','Site','Portugal','Data','Link',\
            'Autonomia Electrica [km]', 'Consumo [kWh/100km]','Mes de Registo','Ano',\
                'Potencia [kw]','Potencia [cv]','Cilindrada [cm3]','Peso [kg]',\
                    'Consumo Combinado','Consumo Urbano','Consumo Extra Urbano',\
                        'Emissoes CO2 [g/km]'
df.columns # Verificar ordem das colunas aqui
df.columns = novas

# Substituir valores NaN das colunas
df['Consumo Urbano'] = df['Consumo Urbano'].fillna(df['Consumo Urbano'].mean())
df['Consumo Extra Urbano'] = df['Consumo Extra Urbano'].fillna(df['Consumo Extra Urbano'].mean())
df['Consumo Combinado'] = df['Consumo Combinado'].fillna(df[['Consumo Urbano', 'Consumo Extra Urbano']].mean(axis=1)) # substituido pelo valor medio das respectivas colunas de consumo
df['Consumo Urbano'] = df['Consumo Urbano'].fillna(df['Consumo Urbano'].mean())
df['Emissoes CO2 [g/km]'] = df['Emissoes CO2 [g/km]'].fillna(df['Emissoes CO2 [g/km]'].mean())
df.loc[df['Combustivel'].isin(['Eléctrico']), 'Cilindrada [cm3]'] = df.loc[df['Combustivel'].isin(['Eléctrico']), 'Cilindrada [cm3]'].fillna(0)

df.info()
df.describe()
df.nunique()

os.chdir('C:/Users/Rodrigo/Documents/GitHub/ImportedCars/Prepared Data')
wd = os.getcwd()
wd = os.path.join(wd,'AutoScout24_cleaned.csv')
df.to_csv(wd, index = False)
