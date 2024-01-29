# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:10:24 2023

@author: Rodrigo
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

wd = os.getcwd()
wd = os.path.abspath(os.path.join(wd, os.pardir))
path = os.chdir(wd)
path1 = os.path.join(wd,'Updated Data','Merged.csv')

data = pd.read_csv(path1)
# data = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

# Definir função apra calcula da taxa de cilindrada
def calcular_taxa_cilindrada(cilindrada):
    
    if cilindrada == 0:
        return 0
    
    elif 0 < cilindrada <= 1000:
        return cilindrada * 1.04 - 808.60
    
    elif 1000 < cilindrada <= 1250:
        return cilindrada * 1.12 - 810.18
    
    else:
        return cilindrada * 5.34 - 5899.89

# Definir função apra calcula do desconto com base na 'Idade'
def calcular_desconto(idade):
    if idade <= 1:
        return 0.10
    elif 1 < idade <= 2:
        return 0.20
    elif 2 < idade <= 3:
        return 0.28
    elif 3 < idade <= 4:
        return 0.35
    elif 4 < idade <= 5:
        return 0.43
    elif 5 < idade <= 6:
        return 0.52
    elif 6 < idade <= 7:
        return 0.60
    elif 7 < idade <= 8:
        return 0.65
    elif 8 < idade <= 9:
        return 0.70
    elif 9 < idade <= 10:
        return 0.75
    else:
        return 0.80

# Definir função apra calculo da taxa de emissoes de gasolina
def calcular_taxa_gasolina(co2):
    if co2 <= 99:
        return co2 * 4.40 - 406.67
    elif 100 <= co2 <= 115:
        return co2 * 7.70 - 715.23
    elif 116 <= co2 <= 145:
        return co2 * 50.06 - 5622.80
    elif 146 <= co2 <= 175:
        return co2 * 58.32 - 6800.16
    elif 176 <= co2 <= 195:
        return co2 * 148.54 - 22502.16
    else:
        return co2 * 195.86 - 31800.11

# Definir função apra calculo da taxa de emissoes de diesel
def calcular_taxa_diesel(co2):
    if co2 <= 79:
        return co2 * 5.50 - 418.13
    elif 80 <= co2 <= 95:
        return co2 * 22.33 - 1760.55
    elif 96 <= co2 <= 120:
        return co2 * 75.45 - 6852.98
    elif 121 <= co2 <= 140:
        return co2 * 167.36 - 18023.73
    elif 141 <= co2 <= 160:
        return co2 * 186.12 - 20686.59
    else:
        return co2 * 255.64 - 31855.14

# Apply the function to the 'Cilindrada [cm3]' column to create a new 'Tax' column
data['Taxa Cilindrada'] = data['Cilindrada [cm3]'].apply(calcular_taxa_cilindrada)

# Apply the function to the 'Idade' column to create a new 'Discount' column
data['Desconto Idade'] = data['Idade'].apply(calcular_desconto)

# Apply the function to the 'CO2 [g/km]' column to create a new 'Gasoline Tax' column for gasoline cars
data.loc[data['Combustivel'] == 'Gasolina', 'Taxa de Gasolina'] = data.loc[data['Combustivel'] == 'Gasolina', 'Emissoes CO2 [g/km]'].apply(calcular_taxa_gasolina)

# Apply the function to the 'Emissoes CO2 [g/km]' column to create a new 'Taxa de Diesel' column for diesel cars
data.loc[data['Combustivel'] == 'Diesel', 'Taxa de Diesel'] = data.loc[data['Combustivel'] == 'Diesel', 'Emissoes CO2 [g/km]'].apply(calcular_taxa_diesel)

# For electric cars, all taxes are zero
data.loc[data['Combustivel'] == 'Eléctrico', ['Taxa de Gasolina', 'Taxa de Diesel']] = 0

# Apply the function to the 'Emissoes CO2 [g/km]' column to create a new 'Taxa de Diesel' column for hybrid diesel cars
data.loc[data['Combustivel'] == 'Híbrido (Diesel)', 'Taxa de Diesel'] = data.loc[data['Combustivel'] == 'Híbrido (Diesel)', 'Emissoes CO2 [g/km]'].apply(calcular_taxa_diesel)

# Apply the function to the 'Emissoes CO2 [g/km]' column to create a new 'Taxa de Gasolina' column for hybrid gasoline cars
data.loc[data['Combustivel'] == 'Híbrido (Gasolina)', 'Taxa de Gasolina'] = data.loc[data['Combustivel'] == 'Híbrido (Gasolina)', 'Emissoes CO2 [g/km]'].apply(calcular_taxa_gasolina)

# Apply a 75% reduction to the 'Taxa de Diesel' and 'Taxa de Gasolina' for hybrid cars
data.loc[data['Combustivel'].str.startswith('Híbrido'), ['Taxa de Diesel', 'Taxa de Gasolina']] *= 0.25

data['Taxa de Gasolina'] = data['Taxa de Gasolina'].fillna(0)
data['Taxa de Diesel'] = data['Taxa de Diesel'].fillna(0)

data['ISV'] = (data['Taxa Cilindrada'] + data['Taxa de Gasolina'] + data['Taxa de Diesel']) / (1 + data['Desconto Idade'])
data.loc[data['Combustivel'] == 'Eléctrico', 'ISV'] = 0

data['Transporte'] = 1500

data['Legalizacao'] = data['ISV'] + data['Transporte']





'''
Analise por MARCA

'''

# Converte a coluna 'Portugal' para booleano (Verdadeiro ou Falso)
data['Portugal'] = data['Portugal'] == 'Yes'

# Select features to match similar cars
matching_features = ['Marca', 'Modelo', 'Combustivel', 'Ano', 'Potencia [cv]']

# Create a key column by concatenating the matching features
data['Key'] = data[matching_features].astype(str).apply('_'.join, axis=1)

# Identify keys that exist in both Portugal and other countries
common_keys = data.groupby('Key')['Portugal'].nunique()
common_keys = common_keys[common_keys > 1].index

# Filter data to include only cars with common keys
common_cars = data[data['Key'].isin(common_keys)]

# Calculate the average price difference between cars in Portugal and other countries for each key
price_differences = common_cars.groupby(['Key', 'Portugal'])['Preco'].mean().unstack()
price_differences['Diferenca Preco'] = price_differences[False] - price_differences[True]

# Add 'Marca' (Brand) to the price differences DataFrame
price_differences = price_differences.reset_index()
price_differences['Marca'] = price_differences['Key'].str.split('_').str[0]

# Display the keys with the highest average price difference
top_price_differences = price_differences.sort_values('Diferenca Preco', ascending=False).head(20)

# Count the number of matched cars for each key
matched_car_counts = common_cars['Key'].value_counts()

# Add the count of matched cars to the price differences DataFrame
top_price_differences['Contagem de Keys iguais'] = top_price_differences['Key'].apply(lambda x: matched_car_counts[x])

print(top_price_differences)

################################################## Comparar Distribuiçao

# Select top 10 brands with most cars
top_brands = price_differences['Marca'].value_counts().index[:20]

# Filter data to include only top 10 brands
top_brands_data = price_differences[price_differences['Marca'].isin(top_brands)]

# Create a boxplot of price differences by brand
plt.figure(figsize=(40, 20))
sns.boxplot(data=top_brands_data, x="Marca", y="Diferenca Preco")
plt.title('Diferença de preço por Marca')
plt.xlabel('Marca')
plt.ylabel('Diferença de Preço (Preço fora Portugal - Preço em Portugal)')
plt.xticks(rotation=0)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Lucros_Marca.png', dpi=300)
plt.show()

# '''
# The boxplot above shows the distribution of price differences for similar cars sold inside and outside Portugal, 
# grouped by brand. The price difference is calculated as the price outside Portugal minus the price inside Portugal. 
# Therefore, a positive price difference suggests that a car is on average more expensive outside Portugal, 
# while a negative price difference suggests that a car is on average more expensive inside Portugal.
# From the plot, we can see that the median price difference varies among brands. For some brands, like Mercedes-Benz, 
# BMW, and Audi, the median price difference is positive, suggesting that these cars are on average more expensive outside Portugal. 
# For other brands, like Hyundai and Ford, the median price difference is negative, suggesting that these cars are on average more expensive inside Portugal.

# This analysis can help identify which brands might be more profitable to import for resale in Portugal. 
# However, remember that other costs associated with importing cars, such as import taxes, shipping costs, 
# and modifications needed to comply with Portugal's regulations, could significantly affect the profitability of the business.

# '''

############################################################################3





'''

Analise por MODELO
'''

# Convert 'Portugal' column to boolean for easier comparison
#data['Portugal'] = data['Portugal'] == 'Yes'

# Select features to match similar cars
matching_features = ['Marca', 'Modelo', 'Combustivel', 'Ano']

# Create a key column by concatenating the matching features
data['Key'] = data[matching_features].astype(str).apply('_'.join, axis=1)

# Identify keys that exist in both Portugal and other countries
common_keys = data.groupby('Key')['Portugal'].nunique()
common_keys = common_keys[common_keys > 1].index

# Filter data to include only cars with common keys
common_cars = data[data['Key'].isin(common_keys)]

# Calculate the average price difference between cars in Portugal and other countries for each key
price_differences = common_cars.groupby(['Key', 'Portugal'])['Preco'].mean().unstack()
price_differences['Diferenca Preco'] = price_differences[False] - price_differences[True]

# Add 'Marca' (Brand) and 'Modelo' (Model) to the price differences DataFrame
price_differences = price_differences.reset_index()
price_differences['Marca'] = price_differences['Key'].str.split('_').str[0]
price_differences['Modelo'] = price_differences['Key'].str.split('_').str[1]

print(price_differences.head())

########################################### Comparar Distribuiçao

# Select top 10 models with most cars
top_models = price_differences['Modelo'].value_counts().index[:20]

# Filter data to include only top 10 models
top_models_data = price_differences[price_differences['Modelo'].isin(top_models)]

# Create a boxplot of price differences by model
plt.figure(figsize=(15, 10))
ax = sns.boxplot(data=top_models_data, x="Modelo", y="Diferenca Preco")
plt.title('Diferença de preço por Modelo')
plt.xlabel('Modelo')
plt.ylabel('Diferença de preço (Preço fora de Portugal - Preço em Portugal)')
ax.axhline(y=0, color='red', linestyle='-', linewidth=2)
ax.axhline(y=-10000, color='blue', linestyle='--', linewidth=2)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Lucros_Modelo.png', dpi=300)
plt.show()


model_a_data = data[data['Modelo'] == '420']
# Create a boxplot of price differences by model
plt.figure(figsize=(15, 10))
ax = sns.lineplot(data=model_a_data, x="ID Anuncio", y="Preco", hue = data['Portugal'])
plt.title('preço por Modelo')
plt.xlabel('Modelo')
plt.ylabel('Diferença de preço (Preço fora de Portugal - Preço em Portugal)')
# ax.axhline(y=0, color='red', linestyle='-', linewidth=2)
# ax.axhline(y=-10000, color='blue', linestyle='--', linewidth=2)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Lucros_Modelo.png', dpi=300)
plt.show()

import plotly.express as px
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
sorted_data = top_models_data.sort_values(by='Diferenca Preco', ascending=True)
fig = px.box(sorted_data, x="Modelo", y="Diferenca Preco", points = 'all')#,color ='Portugal')
fig.show()

import plotly.express as px
import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
fig = px.box(model_a_data, y="Preco",color = 'Portugal', points = 'all')
fig.update_xaxes(showticklabels=False)
fig.show()

'''
TOp 10 carros para comprar
'''

# Select features to match similar cars
matching_features = ['Marca', 'Modelo', 'Combustivel', 'Ano', 'Potencia [cv]']

# Create a key column by concatenating the matching features
data['Key'] = data[matching_features].astype(str).apply('_'.join, axis=1)

# Identify keys that exist in both Portugal and other countries
common_keys = data.groupby('Key')['Portugal'].nunique()
common_keys = common_keys[common_keys > 1].index

# Filter data to include only cars with common keys
common_cars = data[data['Key'].isin(common_keys)]

# Calculate the average price difference between cars in Portugal and other countries for each key
price_differences = common_cars.groupby(['Key', 'Portugal'])['Preco'].mean().unstack()
price_differences['Diferenca Preco'] = price_differences[False] - price_differences[True]

# Add 'Marca' (Brand) and 'Modelo' (Model) to the price differences DataFrame
price_differences = price_differences.reset_index()
price_differences['Marca'] = price_differences['Key'].str.split('_').str[0]
price_differences['Modelo'] = price_differences['Key'].str.split('_').str[1]
price_differences['Combustivel'] = price_differences['Key'].str.split('_').str[2]
price_differences['Ano'] = price_differences['Key'].str.split('_').str[3]
price_differences['Pontecia [cv]'] = price_differences['Key'].str.split('_').str[4]

# Calculate the average selling price in Portugal for each key
average_price_portugal = common_cars[common_cars['Portugal'] == True].groupby('Key')['Preco'].mean()
min_price_portugal = common_cars[common_cars['Portugal'] == True].groupby('Key')['Preco'].min()

price_differences['Media Preco Venda em Portugal'] = price_differences['Key'].apply(lambda x: average_price_portugal[x] if x in average_price_portugal else np.nan)
price_differences['Minimo Preco Venda em Portugal'] = price_differences['Key'].apply(lambda x: min_price_portugal[x] if x in min_price_portugal else np.nan)

# Filter to include only cars sold outside Portugal
profitable_cars = common_cars[(common_cars['Portugal'] == False) & (common_cars['Key'].isin(price_differences['Key']))]

# Add 'Price Difference' and 'Average Selling Price in Portugal' to the DataFrame
profitable_cars = profitable_cars.merge(price_differences[['Key', 'Media Preco Venda em Portugal', 'Minimo Preco Venda em Portugal']], on='Key')

# Sort cars by price difference
profitable_cars['Lucro'] = profitable_cars['Media Preco Venda em Portugal'] - profitable_cars['Preco'] - profitable_cars['Legalizacao']
profitable_cars = profitable_cars[
                                    (profitable_cars['Lucro'] > 4000) 
                                  & ((profitable_cars['Preco'] + profitable_cars['Legalizacao']) < 20000) 
                                  # & (profitable_cars['Electrico'] == 1)
                                  #& (profitable_cars['Data']) != 0
                                  ]
# Select relevant columns
profitable_cars = profitable_cars[['ID Anuncio', 'Marca', 'Modelo', 'Ano', 'Combustivel', 'Tipo de Caixa',\
                                   'Cilindrada [cm3]', 'Potencia [cv]', 'Preco', 'Media Preco Venda em Portugal',\
                                   'Minimo Preco Venda em Portugal', 'Legalizacao','Lucro','Data','Link']]
profitable_cars = profitable_cars.sort_values('Lucro', ascending=False)
print(profitable_cars)

profitable_cars.to_excel(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Lucros_top10.xlsx')
print(profitable_cars)
