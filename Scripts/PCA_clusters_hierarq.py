# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:50:19 2023

@author: Rodrigo
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

####################################################
### PCA
def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

# Selecionar as colunas numericas
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Normalizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_columns])

# Aplicar PCA
pca = PCA()
principalComponents = pca.fit_transform(df_scaled)

# Quanta variância cada componente explica
prop_var = pca.explained_variance_ratio_
coeffecients = pd.DataFrame(data=prop_var, index=range(1, 16))
coeffecients = coeffecients.rename(columns={0: 'Coeficientes'})
print(round(coeffecients, 2))

PC_numbers = np.arange(pca.n_components_) + 1

plt.plot(PC_numbers, prop_var, 'ro-')
plt.ylabel('Proporcao de Variancia', fontsize=8)
plt.xlabel('Componentes Principais', fontsize=8)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\PCA_variancia.png', dpi=300)
plt.show()

pca = PCA(n_components=4)
PC = pca.fit_transform(df_scaled)

loadings = pca.components_
loadings_df = pd.DataFrame(loadings.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=numerical_columns)

sns.clustermap(loadings_df, annot=True, cmap='Spectral')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\PCA_loadings.png', dpi=300)


pca_df = pd.DataFrame(data=PC, columns=['PC1', 'PC2', 'PC3', 'PC4'])

plt.figure(figsize=(12, 10))
plt.title('PCA Biplot')
biplot(PC[:,0:2], np.transpose(pca.components_[0:2, :]), labels=df[numerical_columns].columns)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\PCA_Biplot.png', dpi=300)
plt.show()

'''
The loadings are the correlations between the original variables and the principal components. 
For example, the loading of "Preco" on PC1 is 0.102845, which means that "Preco" is 
positively correlated with PC1. If the value of "Preco" increases, the value of PC1 also tends to increase.

From the clustermap, we can see some interesting patterns. For instance, variables such as 
'Autonomia Electrica [km]', 'Consumo [kWh/100km]', 'Capacidade da Bateria [kWh]', and 'Electrico' are 
strongly positively correlated with PC1. This suggests that these features contribute significantly to 
the variance captured by the first principal component. These features are all related to electric cars,
indicating that the type of car (electric vs. non-electric) is a significant factor in 
explaining the variance in the data.

You can use these loadings to interpret the principal components. However, please remember that 
PCA is a dimensionality reduction technique, and while it can help to identify the 
main sources of variance in the data, it does not necessarily provide clear or 
easily interpretable patterns.

'''


###### KMeans
from sklearn.cluster import KMeans

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(pca_df)

# Add cluster labels to the DataFrame
pca_df['Cluster'] = clusters

plt.figure(figsize=(10,10))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette="deep", s=30, edgecolor=None)
plt.title('Clusters after PCA')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\PCA_cluster.png', dpi=300)
plt.show()

# Add the cluster labels to the original scaled DataFrame
df_numeric = df.select_dtypes(include=[np.number])
df_clustered = pd.DataFrame(df_scaled, columns=df_numeric.columns)
df_clustered['Cluster'] = kmeans.labels_

# Calculate the mean values of the original features for each cluster
cluster_characteristics = df_clustered.groupby('Cluster').mean()

# Display the characteristics of each cluster
cluster_characteristics.to_excel(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Cluster_charateristis.xlsx')

'''
Atravez do grafico(sns.scatterplot) e da tabela (cluster_characteristics) podemos observar que:
Here are the characteristics of each cluster:

Cluster 0: Cars in this cluster tend to have below-average prices and power, 
and slightly above-average kilometers. They are slightly older and are not typically electric.

Cluster 1: Cars in this cluster tend to have above-average prices, power, and fuel consumption. 
They also have high CO2 emissions. They are slightly older and are not typically electric.

Cluster 2: Cars in this cluster tend to have slightly above-average prices and power, 
and below-average kilometers and CO2 emissions. They are newer cars and are more likely to be electric.

Cluster 3: Cars in this cluster tend to have slightly above-average prices and power, 
and below-average kilometers and CO2 emissions. They also have low fuel consumption. 
They are newer cars and are very likely to be electric.

This gives us an idea of what types of cars are in each cluster. 
For example, Cluster 0 might represent older, less powerful cars, while Cluster 2 represents newer, 
more eco-friendly cars.

These insights can be very useful for various purposes, such as targeted marketing or 
decision-making about which cars to stock in a dealership. The next steps would depend on the 
specific goals of your analysis.
'''

import matplotlib.colors
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

x = df[['Preco', 'Quilometros', 'Ano', 'Potencia [cv]']]
wcss = []
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
# Plot 'método do cotovelo' para determinar número de clusters
number_clusters = range(1, 10)
plt.plot(number_clusters, wcss)
plt.title('Metodo do cotovelo')
plt.xlabel('Numero de clusters')
plt.ylabel('WCSS')
plt.show()

# Kmeans clustering
k_means = KMeans(n_clusters=3, random_state=42)
k_means.fit(x)
x['KMeans_labels'] = k_means.labels_

identified_clusters = k_means.fit_predict(x)
print('Clusters Identificados: ', identified_clusters)
data_with_clusters = x.copy()
data_with_clusters['Clusters'] = identified_clusters

# Scatter plot com os clusters identificados
colors = ['purple', 'green', 'blue']

 # 2. _____HIERÁRQUICO_____
x = df[['Preco', 'Quilometros', 'Ano', 'Potencia [cv]']]
model = AgglomerativeClustering(n_clusters=4, affinity='euclidean')
model.fit(x)

x['HR_labels'] = model.labels_

# Plotting Preco vs Quilometros
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.scatter(data_with_clusters['Preco'], data_with_clusters['Quilometros'], c=x['HR_labels'],
            cmap=matplotlib.colors.ListedColormap(colors), s=15)
ax1.set_title('Hierarchical Clustering', fontsize=20)
ax1.set_xlabel('Preço', fontsize=14)
ax1.set_ylabel('Quilometros', fontsize=14)

# Create the dendrogram in the second subplot
selected_data = x.iloc[4000:4150, [0,1]]
clusters = shc.linkage(selected_data, method='ward', metric="euclidean")
dend = shc.dendrogram(Z=clusters, ax=ax2)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Hierarquico_Preco_Quilometros.png', dpi=300)
plt.tight_layout()
plt.show()

# Plotting Preco vs Ano
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

ax1.scatter(data_with_clusters['Preco'], data_with_clusters['Ano'], c=x['HR_labels'],
            cmap=matplotlib.colors.ListedColormap(colors), s=15)
ax1.set_title('Hierarchical Clustering', fontsize=20)
ax1.set_xlabel('Preço', fontsize=14)
ax1.set_ylabel('Ano', fontsize=14)

# dendrogram
selected_data = x.iloc[2000:2150, [0,2]]
clusters = shc.linkage(selected_data, method='ward', metric="euclidean")
dend = shc.dendrogram(Z=clusters, ax=ax2)
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Hierarquico_Preco_Ano.png', dpi=300)
plt.tight_layout()
plt.show()

# Plotting Preco vs Potencia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

ax1.scatter(data_with_clusters['Preco'], data_with_clusters['Potencia [cv]'], c=x['HR_labels'],
            cmap=matplotlib.colors.ListedColormap(colors), s=15)
ax1.set_title('Hierarchical Clustering', fontsize=20)
ax1.set_xlabel('Preço', fontsize=14)
ax1.set_ylabel('Potencia [cv]', fontsize=14)

# dendrogram
selected_data = x.iloc[4000:4150, [0,3]]
clusters = shc.linkage(selected_data, method='ward', metric="euclidean")
dend = shc.dendrogram(Z=clusters, ax=ax2)
plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Hierarquico_Preco_Potencia.png', dpi=300)
plt.show()


# plt.figure(figsize=(10, 10))
# plt.scatter(data_with_clusters['Preco'], data_with_clusters['Quilometros'], c=data_with_clusters['Clusters'],
#             cmap=matplotlib.colors.ListedColormap(colors), s=15)
# plt.title('Preco vs Quilometros', fontsize=20)
# plt.xlabel('Preco', fontsize=14)
# plt.ylabel('Quilometros', fontsize=14)
# plt.show()

# # Scatter plot com os clusters identificados
# colors = ['purple', 'green', 'blue', 'black']
# plt.figure(figsize=(10, 10))
# plt.scatter(data_with_clusters['Preco'], data_with_clusters['Ano'], c=data_with_clusters['Clusters'],
#             cmap=matplotlib.colors.ListedColormap(colors), s=15)
# plt.title('Preco vs Ano(idade)', fontsize=20)
# plt.xlabel('Preco', fontsize=14)
# plt.ylabel('Ano', fontsize=14)
# plt.show()

# # Scatter plot com os clusters identificados
# colors = ['purple', 'green', 'blue', 'black']
# plt.figure(figsize=(10, 10))
# plt.scatter(data_with_clusters['Preco'], data_with_clusters['Potencia [cv]'], c=data_with_clusters['Clusters'],
#             cmap=matplotlib.colors.ListedColormap(colors), s=15)
# plt.title('Preco vs Potencia [cv]', fontsize=20)
# plt.xlabel('Preco', fontsize=14)
# plt.ylabel('Potencia [cv]', fontsize=14)
# plt.show()

'''
For example, in the 'Preco vs Quilometros' plot, we can see that cars with lower prices tend 
to have higher mileage (Quilometros). In the 'Preco vs Idade' plot, we can observe that 
older cars (higher Idade) tend to have lower prices. Finally, in the 'Preco vs Potencia [cv]' plot, 
we can see that cars with higher power (Potencia [cv]) tend to have higher prices.
'''
#####################
