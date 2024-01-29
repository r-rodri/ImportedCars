# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:56:41 2023

@author: Rodrigo
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

#####################################################
# Set the style of the plots
sns.set_style("whitegrid")

# Distribuiçao de preços dos carros
sns.histplot(data=df, x="Preco", kde=True)
plt.title('Distribuição dos Preços')
plt.xticks(rotation=0)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\DistribuicaoPrecos.png', dpi=300)
plt.show()
# Numero de carros por combustivel
sns.countplot(data=df, x="Combustivel", order = df['Combustivel'].value_counts().index)
plt.title('Contagem de Automóveis por Tipo de Combustivel')
plt.ylabel('Contagem')
plt.xticks(rotation=90)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\ContagemCombustivel.png', dpi=300)
plt.show()
# Plot the average price by brand
avg_price_by_brand = df.groupby('Marca')['Preco'].mean().sort_values(ascending=False)
avg_price_by_brand.plot(kind='bar')
plt.title('Preço Médio por Marca')
plt.xticks(rotation=90, fontsize = 6)
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\PreçoMédioMarca.png', dpi=300)
plt.show()
# Contagem de registos em portugal e fora portugal
sns.histplot(data=df, x="Portugal")
plt.title('Contagem de Automóveis em Portugal')
plt.xticks(rotation=0)
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\ContagemAutomoveis.png', dpi=300)
plt.show()

####################

# Grafico dos preços portugal vs outros paises por range de preço

price_ranges = [(5000, 10000), (10001, 20000), (20001, 30000), (30001, 40000), (40001, 50000)]

for i, price_range in enumerate(price_ranges):
    fig, ax = plt.subplots(figsize=(15, 5)) # single subplot

    filtered_data = df[(df['Preco'] > price_range[0]) & (df['Preco'] <= price_range[1])]
    sns.boxplot(data=filtered_data, x="Portugal", y="Preco", ax=ax)
    ax.set_title(f'Preços em Portugal vs Estrangeiro ({price_range[0]} < Preco <= {price_range[1]})')

    plt.tight_layout()
    plt.savefig(fr'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\boxplot_{price_range[0]}_{price_range[1]}.png', dpi=300)
    plt.show()

############

# Anuciante por Pais
ct = pd.crosstab(df['Site'], df['Anunciante'])
ax = ct.plot(kind='bar', stacked=True)
plt.ylabel('Contagem')
ax.set_xticklabels(['Fora de Portugal','Portugal'], rotation=0)
ax.set_xlabel('')
for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Anunciante.png', dpi=300)
plt.show()


# Distribuiçao da 'Marca'
plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='Marca', order=df['Marca'].value_counts().index)
plt.title('Distribuição das Marcas')
plt.xticks(rotation=90)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\DistribuicaoMarca.png', dpi=300)
plt.show()

# Distribuiçao do 'Ano'
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Ano', bins=50, kde=False, color='green')
plt.title('Distribuição dos Anos de Produção')
plt.xlabel('Ano de Produção')
plt.xlim(2000,2023)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\DistribuicaoAnoProducao.png', dpi=300)
plt.show()

# Distribuiçao do 'Idade'
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Idade', bins=20, kde=False, color='green')
plt.title('Distribuição de Idade dos Automóveis')
plt.xlabel('Idade Automoveis')
plt.xlim(0,20)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\DistribuicaoIdade.png', dpi=300)
plt.show()

#  Distribuiçao do 'CO2'
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Emissoes CO2 [g/km]', bins=50, kde=True, color='purple')
plt.title('Distribuição de Emissões de CO2')
plt.xlabel('Emissoes de CO2 (g/km)')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\DistribuicaoCO2.png', dpi=300)
plt.show()

# Plot the distribution of the 'Electrico' column
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Electrico', color = 'green')
plt.title('Distribuicao de Carros Electricos')
plt.xlabel('Electrico (0 = Nao, 1 = Sim)')
plt.xticks([0, 1], ['Não', 'Sim'])
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Distribuicaoelectricos.png', dpi=300)
plt.show()


############ Apriori 

from apyori import apriori
# Select the columns of interest
df_aprio = df[['Modelo', 'Combustivel', 'Cor', 'Tipo de Caixa', 'Ano', 'Portugal', 'Potencia [cv]']].copy()

# Define bins for 'Ano'
bins_ano = [1990, 2000, 2005, 2010, 2015, 2020, df_aprio['Ano'].max()]
labels_ano = ['1990-1999', '2000-2004', '2005-2009', '2010-2014', '2015-2019', '2020-2023']
df_aprio['Ano'] = pd.cut(df_aprio['Ano'], bins=bins_ano, labels=labels_ano)

# Define bins for 'Potencia [cv]'
bins_potencia = [0, 100, 150, 200, df_aprio['Potencia [cv]'].max()]
labels_potencia = ['0-100 cv', '100-150 cv', '150-200 cv', '200-999 cv']
df_aprio['Potencia [cv]'] = pd.cut(df_aprio['Potencia [cv]'], bins=bins_potencia, labels=labels_potencia)

df_aprio = df_aprio.dropna(subset=['Ano'])
df_aprio.head()

# Convert the selected dataset into one-hot encoded DataFrame
df_encoded = pd.get_dummies(df_aprio, prefix='', prefix_sep='')

transactions = df_encoded.values.tolist()
item_names = df_encoded.columns.tolist()

df_encoded.columns = df_encoded.columns.astype(str)
transactions = df_encoded.apply(lambda row: row.index[row == 1].tolist(), axis=1).tolist()

#Apply the Apriori algorithm
# rules = apriori(transactions, min_support=0.2, min_confidence=0.75, min_lift=1, min_length=2)

# # Convert the rules into a list
# rules_list = list(rules)

# # Print the rules
# for rule in rules_list:
#     items = list(rule[0])
#     print("Rule: " + str(items))
#     print("Support: " + str(rule[1]))
#     print("Confidence: " + str(rule[2]))
#     #print("Lift: " + str(rule[3]))
    
size = len(df_aprio)
cols = len(df_aprio.columns)
for i in range(0, size):
    transactions.append([str(df_aprio.values[i, j])
                          for j in range(0, cols)])

rules = apriori(transactions=transactions, min_support=0.2,
                min_confidence=0.75, min_lift=1, min_length=2)
results = list(rules)
print(results)

'''
min_support=0.5, min_confidence=0.5, min_lift=1, min_length=3

Rule: ['Automática']: This rule indicates that there are many transactions with 'Automática'. 
The support is about 0.63, which means that 63% of the transactions in the dataset contain 'Automática'.
Confidence: Since there are no antecedent items (items_base=frozenset()), 
the confidence values are equal to the support values. Confidence measures the likelihood of 
item B (in items_add) being purchased when item A (in items_base) is purchased.
In the final output, you have more complex rules, such as:

Rule: ['Automática', '200-999 cv', '2020-2023']: This rule suggests that transactions often include these three items together.
The support of about 0.13 means that around 13% of transactions contain all of these items.
Confidence: The rule has a confidence of about 0.975 when the items '200-999 cv' and '2020-2023' are present, 
the item 'Automática' is also present in the transaction.
Lift: Lift is the ratio of the observed support to that expected if the two rules were independent. 
The lift of 1.55 indicates that the 'Automática' is 1.55 times more likely to be purchased when '200-999 cv' and 
'2020-2023' are purchased, compared to its general purchase likelihood.
The rules extracted from the Apriori algorithm can provide valuable insights into the purchasing behavior and 
help understand the relationships among the items. However, the interpretation of these rules largely depends on the 
business context and the specific objectives of the analysis.
'''

'''
 min_support=0.2, min_confidence=0.5, min_lift=1, min_length=4
The rules generated by the Apriori algorithm can be interpreted as follows:
Rule 1: If a car has an automatic transmission, it's likely to have 200-999 cv and be from 2020-2023.
This rule has a support of 0.1275, meaning that about 12.75% of all transactions in our dataset meet these criteria.
The confidence of 0.9752 means that in 97.52% of transactions where cars have 200-999 cv and are from 2020-2023, 
the car also has an automatic transmission. The lift of 1.5481 indicates that the presence of a car having 200-999 cv and 
from 2020-2023 raises the likelihood of the car being automatic by a factor of about 1.5481.

Rule 2: If a car is from Portugal (Yes), it's likely to have an automatic transmission and have 200-999 cv. 
This rule has a support of 0.2238, a confidence of 0.9696, and a lift of 1.5392.

Rule 3: If a car is from Portugal (Yes), has 200-999 cv, and is from 2015-2019, 
it's likely to have an automatic transmission. This rule has a support of 0.1086, 
a confidence of 0.9769, and a lift of 1.5509.

These rules might suggest certain trends or relationships in the dataset, 
such as a possible preference for automatic cars with 200-999 cv in Portugal, 
especially for cars from 2015-2019 and 2020-2023. However, these are just possible associations and 
may not imply a causal relationship. Also, the "lift" value of around 1.5 in all cases suggests that 
the relationships are not particularly strong. Further analysis would be needed to better understand 
these relationships and their potential implications.

Regarding the potential business opportunities, it seems like focusing on automatic cars with 200-999 cv from 2015-2019 and 
2020-2023 might be a good idea, especially for the Portuguese market. Also, considering the preference for Diesel and 
Manual cars could be another interesting point to explore.
'''

