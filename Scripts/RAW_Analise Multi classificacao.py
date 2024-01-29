# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:12:55 2023

@author: Rodrigo
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Load the dataset again
df = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

# Create a new column 'Price_Category' for the price categories
df['Price_Category'] = pd.cut(df['Preco'], bins=[0, 15000, 30000, 60000, df['Preco'].max()], labels=['Cheap', 'Affordable', 'Expensive', 'Luxury'])

# Separate features and target variable
X = df.drop(['Preco', 'Price_Category'], axis=1)  # features
y = df['Price_Category']  # target variable

# Identify the categorical and numeric columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

# Define preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that does the preprocessing and then trains a Decision Tree
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predict the price categories on the test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Print the classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


'''

The rows of the matrix represent the actual classes 
('Affordable', 'Cheap', 'Expensive', 'Luxury'), while the columns represent the predicted classes. 
The diagonal elements represent the number of points for which the predicted label is equal to the 
true label, while off-diagonal elements are those that are mislabeled by the classifier.
'''

from sklearn.ensemble import RandomForestClassifier

# Create a pipeline for the Random Forest Classifier
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict the price categories on the test set using the Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics for the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Print the metrics for the Random Forest model
print("Random Forest Model Metrics:")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_rf}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_rf))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Create pipelines for kNN and AdaBoost
knn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

ada_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(random_state=42))
])

# Train the kNN model
knn_model.fit(X_train, y_train)

# Train the AdaBoost model
ada_model.fit(X_train, y_train)

# Predict the price categories on the test set using the kNN model
y_pred_knn = knn_model.predict(X_test)

# Predict the price categories on the test set using the AdaBoost model
y_pred_ada = ada_model.predict(X_test)

# Calculate metrics for the kNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

# Calculate metrics for the AdaBoost model
accuracy_ada = accuracy_score(y_test, y_pred_ada)
precision_ada = precision_score(y_test, y_pred_ada, average='weighted')
recall_ada = recall_score(y_test, y_pred_ada, average='weighted')
f1_ada = f1_score(y_test, y_pred_ada, average='weighted')

# Print the metrics for the kNN model
print("kNN Model Metrics:")
print(f"Accuracy: {accuracy_knn}")
print(f"Precision: {precision_knn}")
print(f"Recall: {recall_knn}")
print(f"F1 Score: {f1_knn}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_knn))

# Print the metrics for the AdaBoost model
print("\nAdaBoost Model Metrics:")
print(f"Accuracy: {accuracy_ada}")
print(f"Precision: {precision_ada}")
print(f"Recall: {recall_ada}")
print(f"F1 Score: {f1_ada}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_ada))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_ada))
