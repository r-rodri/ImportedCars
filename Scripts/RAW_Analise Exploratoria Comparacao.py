# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:44:11 2023

@author: Rodrigo
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the data
data = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

data.info()

# Grafico dos preços portugal vs outros paises por range de preço

price_ranges = [(5000, 10000), (10001, 20000), (20001, 30000), (30001, 40000), (40001, 50000)]

for i, price_range in enumerate(price_ranges):
    fig, ax = plt.subplots(figsize=(8, 8)) # single subplot

    filtered_data = data[(data['Preco'] > price_range[0]) & (data['Preco'] <= price_range[1])]
    sns.boxplot(data=filtered_data, x="Portugal", y="Preco", ax=ax)
    ax.set_title(f'Preços em Portugal vs Estrangeiro ({price_range[0]} < Preco <= {price_range[1]})')

    plt.tight_layout()
    plt.savefig(fr'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\boxplot_{price_range[0]}_{price_range[1]}.png', dpi=300)
    plt.show()

#########################################

# Count plot of car brands in Portugal and other countries
top_brands = data['Marca'].value_counts().index[:10]
sns.countplot(data=data[data['Marca'].isin(top_brands)], x="Marca", hue="Portugal", order=top_brands)
plt.title('Contagem de Automóveis por Marca em Portugal e Fora de Portugal')
plt.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\ComparacaoMarcas_PortugalvsOthers.png', dpi=300)
plt.show()

# Count plot of fuel types in Portugal and other countries
sns.countplot(data=data, x="Combustivel", hue="Portugal", order=data['Combustivel'].value_counts().index)
plt.title('Combustivel dos Automóveis em Portugal e Fora de Portugal')
plt.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\ComparacaoCombustivel_PortugalvsOthers.png', dpi=300)
plt.show()

# Boxplot of car age in Portugal and other countries
sns.boxplot(data=data, x="Portugal", y="Idade")
plt.title('Idade dos Automóveis em Portugal e Fora de Portugal')
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\ComparacaoIdade_PortugalvsOthers.png', dpi=300)
plt.show()

# Boxplot of kilometers driven in Portugal and other countries
sns.boxplot(data=data, x="Portugal", y="Quilometros")
plt.title('Quilometragem dos Automóveis em Portugal e Fora de Portugal')
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\ComparacaoQuilometros_PortugalvsOthers.png', dpi=300)
plt.show()


