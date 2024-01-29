# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:43:56 2023

@author: Rodrigo
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score, roc_curve , roc_auc_score , auc
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


# Load the dataset again
df = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

# Create binary labels: 'Affordable' if price <= 30000, 'Expensive' if price > 30000
df['price_category'] = df['Preco'].apply(lambda x: 'Acessivel' if x <= 30000 else 'Dispendioso')

# Check the distribution of the new binary labels
Contagem = df['price_category'].value_counts()
Contagem.to_excel(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Classificacao_contagem.xlsx')
print(Contagem)

# Define the feature matrix X and the target y
X = df.drop(['ID Anuncio', 'Anunciante', 'Preco', 'price_category'], axis=1)
y = df['price_category']

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the numerical and categorical features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Remove target variable from features
numeric_features = numeric_features.drop('Preco')

# Update the categorical features to only include those that are in X
categorical_features = [i for i in categorical_features if i in X.columns]

# Define preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Fit and transform the training data
X_train = preprocessor.fit_transform(X_train)

# Transform the test data
X_test = preprocessor.transform(X_test)

'''Decision Tree'''
# Train a Decision Tree model
clfDT = DecisionTreeClassifier(random_state=42)
clfDT.fit(X_train, y_train)

# Make predictions on the test set
y_predDT = clfDT.predict(X_test)

# Compute the accuracy of the model
accDT = accuracy_score(y_test, y_predDT)
print('Decision Tree:',accDT)

'''kNN'''
# Train a k-Nearest Neighbors model
clfKNN = KNeighborsClassifier()
clfKNN.fit(X_train, y_train)

# Make predictions on the test set
y_predKNN = clfKNN.predict(X_test)

# Compute the accuracy of the model
accKNN = accuracy_score(y_test, y_predKNN)
print('kNN:',accKNN)

'''Adaboost'''
# Train an AdaBoost model
clfAda = AdaBoostClassifier(random_state=42)
clfAda.fit(X_train, y_train)

# Make predictions on the test set
y_predAda = clfAda.predict(X_test)

# Compute the accuracy of the model
accAda = accuracy_score(y_test, y_predAda)
print('Adaboost:',accAda)

'''Random Forest'''
# Train a Random Forest model
clfRF = RandomForestClassifier(random_state=42)
clfRF.fit(X_train, y_train)

# Make predictions on the test set
y_predRF = clfRF.predict(X_test)

# Compute the accuracy of the model
accRF = accuracy_score(y_test, y_predRF)
print('Random Forest:',accRF)

# Train a Random Forest model
clfRF = RandomForestClassifier(random_state=42)
clfRF.fit(X_train, y_train)



'''Curva ROC'''

# Compute ROC curve and ROC area for each class

# Binarize the output
y_test_bin = label_binarize(y_test, classes=['Acessivel', 'Dispendioso'])
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fprDT = dict()
tprDT = dict()
roc_aucDT = dict()
fprKNN = dict()
tprKNN = dict()
roc_aucKNN = dict()
fprAda = dict()
tprAda = dict()
roc_aucAda = dict()
fprRF = dict()
tprRF = dict()
roc_aucRF = dict()

for i in range(n_classes):
    fprDT[i], tprDT[i], _ = roc_curve(y_test_bin[:, i], clfDT.predict_proba(X_test)[:, i])
    roc_aucDT[i] = auc(fprDT[i], tprDT[i])
    
    fprKNN[i], tprKNN[i], _ = roc_curve(y_test_bin[:, i], clfKNN.predict_proba(X_test)[:, i])
    roc_aucKNN[i] = auc(fprKNN[i], tprKNN[i])
    
    fprAda[i], tprAda[i], _ = roc_curve(y_test_bin[:, i], clfAda.predict_proba(X_test)[:, i])
    roc_aucAda[i] = auc(fprAda[i], tprAda[i])
    
    fprRF[i], tprRF[i], _ = roc_curve(y_test_bin[:, i], clfRF.predict_proba(X_test)[:, i])
    roc_aucRF[i] = auc(fprRF[i], tprRF[i])

# Plot the ROC curves
plt.figure(figsize=(10, 10))
lw = 2

plt.plot(tprDT[0], fprDT[0],  color='blue', lw=lw, label='Curva ROC da Decision Tree (area = %0.2f)' % (1-roc_aucDT[0]))
plt.plot(tprKNN[0], fprKNN[0],  color='green', lw=lw, label='Curva ROC da kNN (area = %0.2f)' % (1-roc_aucKNN[0]))
plt.plot(tprAda[0], fprAda[0],  color='red', lw=lw, label='Curva ROC da AdaBoost (area = %0.2f)' % (1-roc_aucAda[0]))
plt.plot(tprRF[0], fprRF[0],  color='darkorange', lw=lw, label='Curva ROC da Random Forest (area = %0.2f)' % (1-roc_aucRF[0]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Racio de Falsos Positivos')
plt.ylabel('Racio de Positivos Verdadeiros')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\Classificacao_Binaria.png', dpi=300)
plt.show()

'''
Here's the Receiver Operating Characteristic (ROC) curve for our binary classification models:

Decision Tree: Area Under the Curve (AUC) = 0.92
k-Nearest Neighbors (kNN): AUC = 0.91
AdaBoost: AUC = 0.98
Random Forest: AUC = 0.98
The AUC score represents the model's ability to distinguish between the positive and negative classes.
An AUC score of 1.0 means the model has perfect classification ability, while an AUC score of 0.5 means 
the model's ability is no better than random chance.

Both the AdaBoost and Random Forest models have the highest AUC scores (0.98), indicating excellent classification ability. 
The Decision Tree and kNN models also performed well, with AUC scores of 0.92 and 0.91, respectively.
'''
