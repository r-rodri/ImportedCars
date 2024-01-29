# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:09:30 2023

@author: Rodrigo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import os

# wd = os.getcwd()
# wd = os.path.abspath(os.path.join(wd, os.pardir))
# path = os.chdir(wd)
# path1 = os.path.join(wd,'Updated Data','Merged.csv')
path2 = r'C:\Users\Rodrigo\Desktop\Projectos\20230701 - Importar Carros\Updated Data\Merged.csv'
data = pd.read_csv(path2)

###################################################################################
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
price_differences['Diferenca Preco'] = price_differences['No'] - price_differences['Yes']

# Add 'Marca' (Brand) and 'Modelo' (Model) to the price differences DataFrame
price_differences = price_differences.reset_index()
price_differences['Marca'] = price_differences['Key'].str.split('_').str[0]
price_differences['Modelo'] = price_differences['Key'].str.split('_').str[1]
price_differences['Combustivel'] = price_differences['Key'].str.split('_').str[2]
price_differences['Ano'] = price_differences['Key'].str.split('_').str[3]
price_differences['Pontecia [cv]'] = price_differences['Key'].str.split('_').str[4]

# Calculate the average selling price in Portugal for each key
average_price_portugal = common_cars[common_cars['Portugal'] == 'Yes'].groupby('Key')['Preco'].mean()
min_price_portugal = common_cars[common_cars['Portugal'] == 'Yes'].groupby('Key')['Preco'].min()

price_differences['Media Preco Venda em Portugal'] = price_differences['Key'].apply(lambda x: average_price_portugal[x] if x in average_price_portugal else np.nan)
price_differences['Minimo Preco Venda em Portugal'] = price_differences['Key'].apply(lambda x: min_price_portugal[x] if x in min_price_portugal else np.nan)

# Filter to include only cars sold outside Portugal
profitable_cars = common_cars[(common_cars['Portugal'] == 'No') & (common_cars['Key'].isin(price_differences['Key']))]

# Add 'Price Difference' and 'Average Selling Price in Portugal' to the DataFrame
profitable_cars = profitable_cars.merge(price_differences[['Key', 'Media Preco Venda em Portugal', 'Minimo Preco Venda em Portugal']], on='Key')

# Sort cars by price difference
profitable_cars['Lucro'] = profitable_cars['Media Preco Venda em Portugal'] - profitable_cars['Preco'] - profitable_cars['Legalizacao']
# profitable_cars = profitable_cars[
#                                     (profitable_cars['Lucro'] > 4000) 
#                                   & ((profitable_cars['Preco'] + profitable_cars['Legalizacao']) < 20000) 
#                                   # & (profitable_cars['Electrico'] == 1)
#                                   #& (profitable_cars['Data']) != 0
#                                   ]
# Select relevant columns
profitable_cars = profitable_cars[['Marca', 'Modelo', 'Ano', 'Combustivel',\
                                   'Cilindrada [cm3]', 'Potencia [cv]', 'Preco', 'Media Preco Venda em Portugal',\
                                   'Minimo Preco Venda em Portugal', 'Legalizacao','Lucro','Data','Link']]
profitable_cars = profitable_cars.sort_values('Lucro', ascending=False)

############################################################################################
# Set the default renderer to 'browser'
pio.renderers.default = 'browser'
st.title('Best Car to Flip')

# Filter options
filter_options = ['Lucro']

# Sidebar - Filter options
st.sidebar.title('Filter Options')

# Input boxes for the user to enter filter values
default_min_lucro = float(4000)
default_max_preco_legalizacao = float(20000)

# min_lucro = st.sidebar.number_input('Enter the minimum Lucro:', value=default_min_lucro, min_value=float(profitable_cars['Lucro'].min()))
# max_preco_legalizacao = st.sidebar.number_input('Enter the maximum Preco+legalizacao:', value=default_max_preco_legalizacao, min_value=float(profitable_cars['Preco'].min()))

# Input boxes for the user to enter filter values
min_lucro_input = st.sidebar.slider('Select the minimum profit:', min_value=float(0), max_value=float(30000), value=default_min_lucro)
max_preco_legalizacao_input = st.sidebar.slider('Select the maximum investment:', min_value=float(0), max_value=float(100000), value= None, step= float(100))#default_max_preco_legalizacao)

# Decide the values based on input or default
min_lucro = min_lucro_input if min_lucro_input is not None else default_min_lucro
max_preco_legalizacao = max_preco_legalizacao_input if max_preco_legalizacao_input is not None else default_max_preco_legalizacao

# Input box for exact price
exact_preco_legalizacao = st.sidebar.number_input('Enter the exact investment value (leave empty to use slider value):', min_value=float(0),max_value=float(100000), value=max_preco_legalizacao_input, step=float(500))

# Use the exact value if provided, otherwise use the slider value
max_preco_legalizacao = exact_preco_legalizacao if exact_preco_legalizacao is not None else max_preco_legalizacao



# Filter the cars based on the input values
filtered_cars2 = profitable_cars[(profitable_cars['Lucro'] >= min_lucro) &
                                 ((profitable_cars['Preco'] + profitable_cars['Legalizacao']) <= max_preco_legalizacao) 
                                 ]
    
# Checkbox for filtering by date
filter_by_date = st.sidebar.checkbox('Filter by Date') 
                                
# Filter by date if the checkbox is checked
if filter_by_date:
    filtered_cars2 = filtered_cars2[filtered_cars2['Data'].notna()]

# Display the filtered table
st.write('Below you can find the best profitable car to buy outside Portugal.')
st.table(filtered_cars2.style.format({'Ano': '{:.0f}',
                                      'Cilindrada [cm3]':'{:.0f}',
                                  'Potencia [cv]': '{:.0f}',
                                  'Preco': '{:.0f}',
                                  'Media Preco Venda em Portugal': '{:.0f}',
                                  'Minimo Preco Venda em Portugal': '{:.0f}',
                                  'Legalizacao': '{:.0f}',
                                  'Lucro': '{:.0f}',
                                  }))
# Dropdown to select the 'modelo'
selected_modelo = st.selectbox('Select model to see all the records:', data['Modelo'].unique())

# Filter data based on selected 'modelo'
filtered_data = data[data['Modelo'] == selected_modelo]

# Create a box plot using Plotly Express for the selected 'modelo'
fig = px.box(filtered_data, y="Preco", color='Portugal', points='all', title=f'Box Plot for {selected_modelo}')
fig.update_xaxes(showticklabels=False)

# Display the plot
st.plotly_chart(fig)

# Display the filtered data in a table
data_table = filtered_data[['Marca','Modelo','Ano','Combustivel','Potencia [cv]','Preco','Data','Link']]
st.write('Filtered Data:')
st.table(data_table.style.format({'Ano': '{:.0f}',
                                  'Potencia [cv]': '{:.0f}',
                                  'Preco': '{:.0f}'}))
