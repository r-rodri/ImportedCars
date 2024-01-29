# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:40:49 2023

@author: Rodrigo
"""
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the dataset
data = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")


# Preprocessing steps
numerical_features = ['Ano', 'Quilometros', 'Potencia [cv]']
categorical_features = ['Marca', 'Modelo', 'Tipo de Caixa', 'Combustivel', 'Portugal']

# One-hot encode categorical features and standardize numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Features and target
X = data[categorical_features + numerical_features]
y = data['Preco']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the Random forest regression model
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(random_state=42))])

# Train the Random forest regression model
rf_model.fit(X_train, y_train)

# Make predictions using the Random forest regression model
rf_preds = rf_model.predict(X_test)

# Evaluate the Random forest regression model
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# Print the results
print("Random Forest Regression: \nRMSE = {}\nMAE = {}\nR2 = {}".format(rf_rmse, rf_mae, rf_r2))


# Apply cross-validation with 3 folds
cv_scores = cross_val_score(rf_model, X, y, cv=3)

# Compute the mean of the cross-validation scores
cv_mean_score = cv_scores.mean()
print('Cross Validation: ',cv_mean_score)
'''
Outra Conclusão: 
The Random Forest Regression model performed quite well on the test data. The metrics are:

Root Mean Squared Error (RMSE) = 15571.33: This indicates that the model's predictions are, 
on average, approximately €15571.33 away from the actual values.
Mean Absolute Error (MAE) = 6651.93: On average, the model's predictions are approximately 
€6651.93 away from the actual values.
R2 Score = 0.89: This score is a statistical measure that represents the proportion of the variance 
for a dependent variable that's explained by an independent variable or variables in a regression model.
 It is a statistic used in the context of statistical models whose main purpose is either 
 the prediction of future outcomes or the testing of hypotheses, on the basis of other related 
 information. It provides a measure of how well observed outcomes are replicated by the model,
 based on the proportion of total variation of outcomes explained by the model.
'''

# # Select the columns of interest
# selected_columns = ['Marca', 'Combustivel', 'Portugal', 'Tipo de Caixa', 'Ano', 
#                     'Quilometros', 'Cilindrada [cm3]', 'Potencia [cv]', 'Electrico', 
#                     'Capacidade da Bateria [kWh]', 'Preco']

# # Subset the data
# data = data[selected_columns]

# # Check unique values in each categorical column
# categorical_columns = ['Marca', 'Combustivel', 'Portugal', 'Tipo de Caixa']
    
# # Perform one-hot encoding on the categorical data
# data_encoded = pd.get_dummies(data, columns=categorical_columns)

# # Display the first few rows of the encoded dataframe
# data_encoded.head()

# # Define the features set (X) and the target set (y)
# X = data_encoded.drop(columns=['Preco'])
# y = data_encoded['Preco']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Instantiate the Random Forest Regressor
# rf = RandomForestRegressor(n_estimators=100, random_state=42)

# # Fit the model on the training data
# rf.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = rf.predict(X_test)

# # Compute the evaluation metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(mae, mse, r2)

# # Apply cross-validation with 3 folds
# cv_scores = cross_val_score(rf, X, y, cv=2)

# # Compute the mean of the cross-validation scores
# cv_mean_score = cv_scores.mean()
# print('A Média da cross-validation é: ',cv_mean_score)

'''
The mean R-squared score from 3-fold cross-validation is approximately 0.722. 
This score is lower than the R-squared score we obtained earlier without 
cross-validation (0.896).

Cross-validation provides a more robust measure of model performance by training and 
testing the model on different subsets of the data. The discrepancy between the 
cross-validation score and the previous R-squared score suggests that our model might 
be overfitting to the training data, meaning it performs well on the training data 
but less so on unseen data.
'''
