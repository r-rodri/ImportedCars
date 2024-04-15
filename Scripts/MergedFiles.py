# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 23:42:59 2023

@author: Rodrigo
"""

import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
wd = os.getcwd()
wd = os.path.abspath(os.path.join(wd, os.pardir))
path = os.chdir(wd)
path1 = os.path.join(wd,'Prepared Data','StandVirtual_cleaned.csv')
path2 = os.path.join(wd,'Prepared Data','AutoScout24_cleaned.csv')

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df12 = pd.concat([df1, df2], ignore_index=True) #Merged.csv

df = df12.drop(['Autonomia Maxima','Bateria', 'Versao', 'Segmento', 'Numero de Portas', 
                'Origem', 'Condicao', 'Data Anuncio', 
                'Numero de Mudancas', 'IUC', 'Sub-modelo', 'Classico', 'Numero de Cilindros', 
                'Outros tipos de Combustivel', 'Classe Emissoes', 
                'Garantia de stand - incluida no preco [meses]', 'Valor Sem ISV', 'Peso [kg]', 'Mes de Registo'],               
               axis = 1)

new_order = ['ID Anuncio', 'Anunciante', 'Marca', 'Modelo', 'Combustivel', 'Site',
             'Portugal','Cor', 'Tipo de Caixa', 'Preco', 'Ano', 
             'Quilometros', 'Cilindrada [cm3]','Potencia [cv]', 'Potencia [kw]', 
             'Emissoes CO2 [g/km]', 'Consumo Urbano', 'Consumo Extra Urbano', 'Consumo Combinado',
             'Autonomia Electrica [km]', 'Consumo [kWh/100km]','Capacidade da Bateria [kWh]','Data','Link']

df = df.reindex(columns=new_order)

df.info()
df.describe()

# Tratamento dos valores em falta
# Check for missing values again
df_missing_values = df.isnull().sum()
print('\n--- Inicial ---\n',df_missing_values)


'''
Valores em falta na colunas numericas

'''

# Retirar linhas que contenham o combustivel igual a 'GPL'
df = df[~df['Combustivel'].isin(['GPL', 'GNC'])]

# Retirar linhas que não tem ano de registo
df = df.dropna(subset=['Ano'])

# Valores em falta de Cilindrada
df['Cilindrada [cm3]'] = df.groupby('Modelo')['Cilindrada [cm3]'].transform(lambda x: x.fillna(x.median()))
df = df.dropna(subset=['Cilindrada [cm3]'])

# Valores em falta de Potencia
df['Potencia [cv]'] = df.groupby(['Modelo', 'Cilindrada [cm3]'])['Potencia [cv]'].transform(lambda x: x.fillna(x.median()))
df = df.dropna(subset=['Potencia [cv]'])

# Valores em falta de Emissoes CO2
mask = df['Combustivel'].isin(['Eléctrico']) # Colocar a zero para veiculos electricos
df.loc[mask, 'Emissoes CO2 [g/km]'] = df.loc[mask, 'Emissoes CO2 [g/km]'].fillna(0)
df['Emissoes CO2 [g/km]'] = df.groupby(['Modelo', 'Combustivel'])['Emissoes CO2 [g/km]'].transform(lambda x: x.fillna(x.median()))
df['Emissoes CO2 [g/km]']= df['Emissoes CO2 [g/km]'].fillna(df['Emissoes CO2 [g/km]'].median()) # Preencher restantes valor com a mediana = 123

# Valores em falta de Potencia [kw]
## Dividir em dados de treino e nos dados a prever
train_set = df[df['Potencia [kw]'].notna()] # tabela cujo valores kw NAO sao NA
prediction_set = df[df['Potencia [kw]'].isna()] # tabela cujo valores de kw sao NA

X_train = train_set['Potencia [cv]'].values.reshape(-1, 1) ## Dividir e formatar os dados de treino nos valores de X e Y (a prever)
y_train = train_set['Potencia [kw]']

X_pred = prediction_set['Potencia [cv]'].values.reshape(-1, 1) ## Tabela dos dados a prever que sao NA

model = LinearRegression() ## Fit the linear regression model
model.fit(X_train, y_train)

predicted_values = model.predict(X_pred) ## Prever 'Potencia [kw]' nos dados com NA

df.loc[df['Potencia [kw]'].isna(), 'Potencia [kw]'] = predicted_values ##Preencher a tabela original com os valores previstos

slope = round(model.coef_[0],2) ## Coeficientes calculados na expressao do modelo linear
intercept = round(model.intercept_,2)
print(f'\n Potencia(kw) = Potencia (cv) x {slope} {intercept}')

# Preenchimento dos valores NA
mask = df['Combustivel'].isin(['Gasolina', 'Diesel'])
df.loc[mask, ['Autonomia Electrica [km]', 'Consumo [kWh/100km]', 'Capacidade da Bateria [kWh]']] = df.loc[mask, ['Autonomia Electrica [km]', 'Consumo [kWh/100km]', 'Capacidade da Bateria [kWh]']].fillna(0)

# Para 'Híbrido (Diesel)' e 'Híbrido (Gasolina)'
# Foi dados valores aleatorios dentro dos valores reais.
mask = df[df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])]
                              
df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] < 2021), 'Autonomia Electrica [km]'] = \
    df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] < 2021), 'Autonomia Electrica [km]'].fillna(40)

df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] >= 2021), 'Autonomia Electrica [km]'] = \
    df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] >= 2021), 'Autonomia Electrica [km]'].fillna(60)

df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] < 2021), 'Capacidade da Bateria [kWh]'] = \
    df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] < 2021), 'Capacidade da Bateria [kWh]'].fillna(10)

df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] >= 2021), 'Capacidade da Bateria [kWh]'] = \
    df.loc[(df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)'])) & (df['Ano'] >= 2021), 'Capacidade da Bateria [kWh]'].fillna(18)

# Calculo do 'Consumo [kWh/100km]' baseado na 'Capacidade da Bateria [kWh]' e na 'Autonomia Electrica [km]'
df.loc[df['Combustivel'].isin(['Híbrido (Diesel)', 'Híbrido (Gasolina)']), 'Consumo [kWh/100km]'] = \
    (df['Capacidade da Bateria [kWh]'] / df['Autonomia Electrica [km]']) * 100

# Colocar sempre os mesmos valores calculados para o mesmo modelo
df['Autonomia Electrica [km]'] = df.groupby(['Modelo', 'Combustivel'])['Autonomia Electrica [km]'].transform(lambda x: x.fillna(x.median()))
df['Capacidade da Bateria [kWh]'] = df.groupby(['Modelo', 'Combustivel'])['Capacidade da Bateria [kWh]'].transform(lambda x: x.fillna(x.median()))
df['Consumo [kWh/100km]'] = df.groupby(['Modelo', 'Combustivel'])['Consumo [kWh/100km]'].transform(lambda x: x.fillna(x.median()))

# Valores de 'Eléctrico'
# Preenche os valores em falta com base nos valores existentes por 'Modelo' e 'Combustivel' 
df.loc[mask, 'Autonomia Electrica [km]'] = \
    df.loc[mask].groupby(['Modelo', 'Combustivel'])['Autonomia Electrica [km]'].transform(lambda x: x.fillna(x.median()))
df.loc[mask, 'Capacidade da Bateria [kWh]'] = \
    df.loc[mask].groupby(['Modelo', 'Combustivel'])['Capacidade da Bateria [kWh]'].transform(lambda x: x.fillna(x.median()))
df.loc[mask, 'Consumo [kWh/100km]'] = \
    df.loc[mask].groupby(['Modelo', 'Combustivel'])['Consumo [kWh/100km]'].transform(lambda x: x.fillna(x.median()))

# Preenche os valores em falta calculando com os outros valores, se presentes
mask = df['Combustivel'].isin(['Eléctrico'])
df.loc[mask, 'Consumo [kWh/100km]'] = (df['Capacidade da Bateria [kWh]'] / df['Autonomia Electrica [km]'])*100
df.loc[mask, 'Capacidade da Bateria [kWh]'] = (df['Consumo [kWh/100km]'] * df['Autonomia Electrica [km]']) / 100
df.loc[mask, 'Autonomia Electrica [km]'] = (df['Capacidade da Bateria [kWh]'] / df['Consumo [kWh/100km]']) * 100

# Preenche os valores em falta com base na mediana dos valores de 'Combustivel' == 'Eléctrico
df.loc[df['Combustivel'] == 'Eléctrico', 'Autonomia Electrica [km]'] = \
    df.loc[df['Combustivel'] == 'Eléctrico', 'Autonomia Electrica [km]'].fillna(df.loc[df['Combustivel'] == 'Eléctrico', 'Autonomia Electrica [km]'].median())
df.loc[df['Combustivel'] == 'Eléctrico', 'Capacidade da Bateria [kWh]'] = \
    df.loc[df['Combustivel'] == 'Eléctrico', 'Capacidade da Bateria [kWh]'].fillna(df.loc[df['Combustivel'] == 'Eléctrico', 'Capacidade da Bateria [kWh]'].median())
df.loc[df['Combustivel'] == 'Eléctrico', 'Consumo [kWh/100km]'] = \
    df.loc[df['Combustivel'] == 'Eléctrico', 'Consumo [kWh/100km]'].fillna(df.loc[df['Combustivel'] == 'Eléctrico', 'Consumo [kWh/100km]'].median())

# Valores de 'Consumo Urbano', 'Consumo Extra Urbano' e 'Consumo Combinado'
# Preencher os valores em falta de 'Consumo Urbano', 'Consumo Extra Urbano' e 'Consumo Combinado' calculando so valores com base nos outros consumos se presentes
df.loc[df['Consumo Combinado'].isnull(), 'Consumo Combinado'] = (df['Consumo Urbano'] + df['Consumo Extra Urbano']) / 2
df.loc[df['Consumo Urbano'].isnull(), 'Consumo Urbano'] = (df['Consumo Combinado'] * 2) - df['Consumo Extra Urbano']
df.loc[df['Consumo Extra Urbano'].isnull(), 'Consumo Extra Urbano'] = (df['Consumo Combinado'] * 2) - df['Consumo Urbano']

# Preencher os valores em falta de 'Consumo Urbano', 'Consumo Extra Urbano' e 'Consumo Combinado' com a mediana do mesmo 'Modelo', 'Combustivel' and 'Cilindrada [cm3]'
df['Consumo Urbano'] = df.groupby(['Modelo', 'Combustivel', 'Cilindrada [cm3]'])['Consumo Urbano'].transform(lambda x: x.fillna(x.median()))
df['Consumo Extra Urbano'] = df.groupby(['Modelo', 'Combustivel', 'Cilindrada [cm3]'])['Consumo Extra Urbano'].transform(lambda x: x.fillna(x.median()))
df['Consumo Combinado'] = df.groupby(['Modelo', 'Combustivel', 'Cilindrada [cm3]'])['Consumo Combinado'].transform(lambda x: x.fillna(x.median()))

# Preencher os valores em falta de 'Consumo Urbano', 'Consumo Extra Urbano' e 'Consumo Combinado' com a mediana do mesmos, 'Combustivel' and 'Cilindrada [cm3]'
df['Consumo Urbano'] = df.groupby(['Cilindrada [cm3]', 'Combustivel'])['Consumo Urbano'].transform(lambda x: x.fillna(x.median()))
df['Consumo Extra Urbano'] = df.groupby(['Cilindrada [cm3]', 'Combustivel'])['Consumo Extra Urbano'].transform(lambda x: x.fillna(x.median()))
df['Consumo Combinado'] = df.groupby(['Cilindrada [cm3]', 'Combustivel'])['Consumo Combinado'].transform(lambda x: x.fillna(x.median()))

# Preencher os valores em falta de 'Consumo Urbano', 'Consumo Extra Urbano' e 'Consumo Combinado' com a mediana do mesmos 'Cilindrada [cm3]'
df['Consumo Urbano'] = df.groupby('Cilindrada [cm3]')['Consumo Urbano'].transform(lambda x: x.fillna(x.median()))
df['Consumo Extra Urbano'] = df.groupby('Cilindrada [cm3]')['Consumo Extra Urbano'].transform(lambda x: x.fillna(x.median()))
df['Consumo Combinado'] = df.groupby('Cilindrada [cm3]')['Consumo Combinado'].transform(lambda x: x.fillna(x.median()))

# Preencher os valores em falta de 'Consumo Urbano' com a mediana da coluna e as restantes atraves do calculo U=2C−E
df['Consumo Urbano'] = df['Consumo Urbano'].transform(lambda x: x.fillna(x.median()))
df.loc[df['Consumo Combinado'].isnull(), 'Consumo Combinado'] = (df['Consumo Urbano'] + df['Consumo Extra Urbano']) / 2
df.loc[df['Consumo Extra Urbano'].isnull(), 'Consumo Extra Urbano'] = (df['Consumo Combinado'] * 2) - df['Consumo Urbano']

# Preencher os valores em falta de 'Consumo Extra Urbano' com a mediana da coluna e as restantes atraves do calculo E=2C−U
df['Consumo Extra Urbano'] = df['Consumo Extra Urbano'].transform(lambda x: x.fillna(x.median()))
df.loc[df['Consumo Combinado'].isnull(), 'Consumo Combinado'] = (df['Consumo Urbano'] + df['Consumo Extra Urbano']) / 2

# Valores em falta Cor
color_counts = df['Cor'].value_counts(normalize=True)
df['Cor'] = df['Cor'].apply(lambda x: np.random.choice(color_counts.index, p=color_counts.values) if pd.isnull(x) else x)

# Valores em falta Tipo de Caixa
mask = (df['Tipo de Caixa'].isnull()) & (df['Combustivel'].isin(['Eléctrico', 'Híbrido (Gasolina)', 'Híbrido (Diesel)']))
df.loc[mask, 'Tipo de Caixa'] = 'Automática'
# Preencher Tipo de Caixa com o valor mais comum com base no modelo e potencia
df['Tipo de Caixa'] = df.groupby(['Modelo', 'Potencia [cv]'])['Tipo de Caixa'].apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Unknown"))

# Criar coluna com a idade do carro
ano_atual = 2023
df['Idade'] = ano_atual - df['Ano']

# Criar coluna a indicar se o carro é electrico
df['Electrico'] = df['Combustivel'].apply(lambda x: 1 if x in ['Eléctrico', 'Híbrido (Gasolina)', 'Híbrido (Diesel)'] else 0)

# Check for missing values again
df_missing_values = df.isnull().sum()
df.info()
print('\n--- Final ---\n',df_missing_values)

file = os.path.join('Updated Data','Merged.csv')
path = os.getcwd()
full_path = os.path.join(path, file)
df.to_csv(full_path, index = False)
#df.to_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv", index=False)

# Verificar Linhas com valores nulos
nan_rows = df[df.isna().any(axis=1)]
print(nan_rows)


modelos = df['Modelo'].value_counts()
sorted(modelos)
# '''
# Tratamento de outliers

# '''

# # Define a function to calculate IQR and detect outliers
# def detect_outliers_iqr(data, feature):
#     Q1 = data[feature].quantile(0.1)
#     Q3 = data[feature].quantile(0.9)
#     IQR = Q3 - Q1
#     outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
#     return outliers

# # Apply the function to each numerical feature in the dataset
# outliers = {}
# numerical_features = ['Preco', 'Ano', 'Quilometros', 'Cilindrada [cm3]', 'Potencia [cv]', 'Potencia [kw]', 
#                       'Emissoes CO2 [g/km]', 'Consumo Urbano', 'Consumo Extra Urbano', 'Consumo Combinado', 
#                       'Autonomia Electrica [km]', 'Consumo [kWh/100km]', 'Capacidade da Bateria [kWh]']

# for feature in numerical_features:
#     outliers[feature] = detect_outliers_iqr(df, feature)

# # Show the number of outliers detected in each numerical feature
# outliers_counts = {feature: len(outliers_df) for feature, outliers_df in outliers.items()}
# outliers_counts

# outliers_df = pd.concat(outliers.values(), keys=outliers.keys())
