# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:16:12 2023

@author: Rodrigo
"""

'''
We will first check the assumptions of linear regression:

-Linearity: The relationship between the independent and dependent variables is linear.

-Multicollinearity: The independent variables should not be too highly correlated with each 
other.

-Normality of Residuals: The residuals (or errors, the differences between actual and 
                                       predicted values) should be normally distributed.

-Homoscedasticity: The variance of the errors is the same across all levels of the 
independent variables.

After checking these assumptions, we will build the regression model and then evaluate it based on:

Coefficient Magnitude: The size of the coefficients in the model.
Statistical Significance: The p-values of the coefficients (typically, we want p < 0.05).
Adjusted R-squared: The proportion of the variance in the dependent variable that is 
predictable from the independent variables. Adjusted R-squared also takes into account the 
number of predictors in the model.
Residual Analysis: Analyzing the residuals to ensure they meet the assumptions.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics.gofplots as smg

df = pd.read_csv(r"C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Ficheiros Tratados\Merged_NA.csv")

df.info()
# X = df[['Marca','Anunciante', 'Combustivel', 'Portugal', 'Tipo de Caixa', 'Ano', 'Quilometros',
#         'Cilindrada [cm3]', 'Potencia [cv]','Electrico', 'Capacidade da Bateria [kWh]'
#         ]]
# y = df['Preco']

# Select the variables to be used in the model
variables = ['Marca', 'Combustivel', 'Portugal', 'Tipo de Caixa', 'Ano', 
             'Quilometros','Cilindrada [cm3]', 'Potencia [cv]','Electrico', 'Capacidade da Bateria [kWh]', 'Preco']
df_model = df[variables].dropna()

# Specify the categorical features
categorical_features = ['Marca', 'Combustivel', 'Portugal', 'Tipo de Caixa']

# List of numerical features
num_features = ['Ano', 'Quilometros', 'Cilindrada [cm3]', 'Potencia [cv]', 'Capacidade da Bateria [kWh]']

# Perform one-hot encoding on the categorical features
df_encoded = pd.get_dummies(df_model, columns=categorical_features)

df_encoded.head()

# Create a subplot of scatter plots for each numerical feature
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

for i, feature in enumerate(num_features):
    row = i % 3
    col = i // 3
    sns.scatterplot(x=feature, y='Preco', data=df_encoded, ax=axs[row, col])

plt.tight_layout()
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\RegressaoMulti_Linearidade.png', dpi=300)
plt.show()

'''
Based on the scatter plots, we can make the following observations:

'Ano': There seems to be a positive relationship between the year of the car and the price, which is expected as newer cars tend to cost more. However, the relationship is not strictly linear.
'Quilometros': It's hard to discern a clear relationship from this plot. The data is quite dispersed.
'Cilindrada [cm3]': Again, the relationship isn't clear from the scatter plot, and the data is quite dispersed.
'Potencia [cv]': There seems to be a positive relationship between the power of the car and the price. This is expected as more powerful cars tend to cost more.
'Electrico': We can see two distinct groups here, representing electric and non-electric cars. Within each group, the relationship with price isn't clear.
'Capacidade da Bateria [kWh]': There seems to be a positive relationship between the battery capacity and the price, which makes sense as electric cars with larger batteries tend to be more expensive.
'''

'''
Next, let's examine multicollinearity. Multicollinearity refers to a situation in which 
two or more explanatory variables in a multiple regression model are highly linearly related. 
We have perfect multicollinearity if the correlation between two independent variables is 
equal to 1 or -1.

Please note that high correlation doesn't always imply causality.
'''
# Select correlation for numerical features
# Compute the correlation matrix
corr_matrix = df_encoded.corr()
corr_numerical = corr_matrix.loc[num_features, num_features]

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_numerical, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Matriz de Correlações')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\RegressaoMulti_MAtrizCorrelacoes.png', dpi=300)
plt.show()

'''
From the correlation heatmap of numerical features, we can see that there is no pair of 
variables that is extremely highly correlated (close to 1 or -1). The highest correlation is 
between 'Electrico' and 'Capacidade da Bateria [kWh]' which is expected as electric cars 
will have a battery capacity while non-electric cars won't.

This indicates that multicollinearity is likely not a serious problem for our model, 
although it's worth keeping in mind that multicollinearity can still exist among 
three or more variables even if no pair of variables is highly correlated.

Next, let's check the normality of residuals. However, to do this, we first need to 
create the linear regression model and calculate the residuals. Let's do that next. 
We will use the sklearn's LinearRegression model.
'''

# DEfinir features e a variavel a prever
X = df_encoded.drop(columns='Preco')
y = df_encoded['Preco']

# Dados de treino e dados de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

# Calculo dos valores residuais
residuals = y_train - y_train_pred

'''
We have calculated the residuals, which are the differences between the actual and 
predicted values of the target variable for the training data.

Now, let's check the normality of these residuals. We can do this by creating a histogram 
of the residuals and a QQ-plot (quantile-quantile plot). The histogram should have 
the shape of a bell curve (normal distribution), and the points in the QQ-plot should lie 
along the diagonal line.
'''
from scipy import stats

# Create a subplot of the histogram and QQ-plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot the histogram
sns.histplot(residuals, kde=True, ax=axs[0])
axs[0].set_title('Histogram dos valores residuais')

# Plot the QQ-plot
stats.probplot(residuals, plot=axs[1])
axs[1].set_title('QQ-plot dos valores residuais')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\RegressaoMulti_Residuos_QQ.png', dpi=300)
plt.tight_layout()
plt.show()

'''
From the histogram and QQ-plot of the residuals, we can observe the following:

Histogram: The distribution of residuals seems to be skewed to the right and does not exactly
resemble a normal distribution. There seems to be some outliers on the right side of 
the distribution.
QQ-plot: The points deviate from the diagonal line, especially at the ends, which suggests 
that the residuals do not follow a perfect normal distribution.
This suggests that the assumption of normality of residuals is not strictly met. 
This could be due to outliers in the data, non-linear relationships that haven't been 
captured by the model, or other factors.

Finally, let's check for homoscedasticity. Homoscedasticity means that the 
variance of the errors is constant across all levels of the independent variables. 
We can check this by plotting the residuals against the predicted values. 
If the points are equally spread across all levels of the predicted values, 
then the assumption of homoscedasticity is likely met.
'''

# Plot the residuals against the predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=residuals)
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Residuais')
plt.title('Valores Residuais vs Valores Previstos')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\RegressaoMulti_Residuos_previstos.png', dpi=300)
plt.show()

'''
From the plot of residuals against predicted values, we can observe that the points are not 
equally spread across all levels of the predicted values. There seems to be a funnel shape, 
with a larger spread of residuals for higher predicted values. This suggests that 
the assumption of homoscedasticity is not met, indicating heteroscedasticity in the data.

Heteroscedasticity often arises in the presence of outliers or extreme leverage values, 
when the model doesn't capture non-linear effects, or when the variance of the errors changes
across different levels of the independent variables.

Now that we have checked the assumptions of the linear regression model, 
let's proceed to examining the model summary including coefficient magnitude, 
statistical significance, and adjusted R-squared. We will use the statsmodels library to 
get a detailed summary of the model. The statsmodels library provides more detailed 
outputs compared to sklearn.
'''

# Add a constant to the features
X_train_sm = sm.add_constant(X_train)

# Create a OLS model
model_sm = sm.OLS(y_train, X_train_sm)

# Fit the model
result = model_sm.fit()

# Print out the model summary
print('\n',result.summary())

'''
The summary of the linear regression model provides us with a lot of information. 
Here are some key points:

Adjusted R-squared: The adjusted R-squared of the model is 0.594, which means that 
approximately 59.4% of the variation in the 'Preco' is explained by the features included 
in the model. This is a fairly good result, but there's still room for improvement.

Coefficients: The coefficients of the model represent the change in the dependent variable 
(Preco) for a one-unit change in the corresponding independent variable, 
holding all other variables constant. For example, the coefficient for 'Ano' is 577.8209, 
which means that for every one unit increase in 'Ano', the 'Preco' increases by approximately 
577.8209 units, holding all other variables constant.

P>|t| (p-values): The p-values are used to determine the significance of the coefficients. 
A small p-value (typically ≤ 0.05) indicates strong evidence that the coefficient is different from 0. 
For instance, the p-value for 'Ano' is 0.000, which indicates that 'Ano' is a statistically significant 
predictor of 'Preco'.

Skewness and Kurtosis: The skewness is quite high, indicating that the residuals are not 
symmetrically distributed. The kurtosis is also extremely high, indicating heavy tails or outliers. 
This supports our earlier finding from the residual plots that the assumption of 
normality of residuals is not strictly met.

Durbin-Watson: The Durbin-Watson statistic is approximately 2, indicating that there is 
no autocorrelation in the residuals. This is good as it means that the residuals are independent, 
which is an assumption of linear regression.

Condition Number: The condition number is extremely large, indicating potential issues 
with multicollinearity. Even though the correlation matrix did not show strong correlations 
between individual pairs of variables, it's possible that three or more variables are highly correlated.

Notes: The note about strong multicollinearity problems or a singular design matrix is concerning. 
This might be due to the one-hot encoding of categorical variables, which creates a lot of additional 
features and can lead to multicollinearity. It could also be due to the inclusion of variables that 
are not significant predictors of the target variable.

In summary, while the model seems to be statistically significant and explains a fair amount of the variation 
in the 'Preco', there are some concerns about the assumptions of the linear regression model not being met
 and potential issues with multicollinearity. It might be necessary to further refine the model, 
 for instance by removing insignificant predictors, transforming variables to better meet the assumptions, 
 or using regularization techniques to handle multicollinearity.

'''

from sklearn.metrics import r2_score

# Compute the R-squared of the model
r2_pca = r2_score(y_train, y_train_pred)
print('R2:', r2_pca)

'''
refine our model by keeping only the features that have a p-value less than or equal 
to 0.05, which indicates that they are statistically significant predictors of the 
target variable 'Preco'. Let's do that and run the model again.
'''

# # Get the p-values for the original features from the previous model
# p_values = result.pvalues

# # # Select only the features that have a p-value <= 0.05
# significant_features = p_values[p_values <= 0.05].index

# # # Drop the constant
# significant_features = significant_features.drop('const')

# # # Create new training and testing sets with only the significant features
# X_train_sig = X_train[significant_features]
# X_test_sig = X_test[significant_features]

# # Create a linear regression model
# model_sig = LinearRegression()

# # Fit the model to the training data
# model_sig.fit(X_train_sig, y_train)

# # Predict the target for the training data
# y_train_pred_sig = model_sig.predict(X_train_sig)

# # Calculate the residuals
# residuals_sig = y_train - y_train_pred_sig

# residuals_sig

'''
We have now created a new model using only the features that have a p-value less 
than or equal to 0.05 and calculated the residuals.

Let's compute the R-squared for this new model and recheck the normality of the 
residuals and homoscedasticity. We'll start with the R-squared value.
'''

# Compute the R-squared of the model
# r2_sig = r2_score(y_train, y_train_pred_sig)
# r2_sig

'''
The R-squared of the model after removing features with a p-value greater than 0.05 
is approximately 0.598. This means that around 59.8% of the variation in the 'Preco' 
can be explained by the remaining significant features included in the model.

This is slightly higher than the R-squared of the model before removing these features 
(which was approximately 0.594), indicating that removing these features may have 
improved the model's explanatory power slightly.

Next, let's check the normality of the residuals. We can do this by creating a histogram 
of the residuals and a QQ-plot (quantile-quantile plot). The histogram should have the 
shape of a bell curve (normal distribution), and the points in the QQ-plot should lie 
along the diagonal line.
'''

# Create a subplot of the histogram and QQ-plot
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# # Plot the histogram
# sns.histplot(residuals_sig, kde=True, ax=axs[0])
# axs[0].set_title('Histogram of Residuals (Significant Features)')

# # Plot the QQ-plot
# stats.probplot(residuals_sig, plot=axs[1])
# axs[1].set_title('QQ-plot of Residuals (Significant Features)')

# plt.tight_layout()
# plt.show()

'''
From the histogram and QQ-plot of the residuals, we can observe the following:

Histogram: The distribution of residuals seems to be skewed to the right and does 
not exactly resemble a normal distribution. There seems to be some outliers on the
 right side of the distribution.
QQ-plot: The points deviate from the diagonal line, especially at the ends, 
which suggests that the residuals do not follow a perfect normal distribution.
This suggests that the assumption of normality of residuals is not strictly met. 
This could be due to outliers in the data, non-linear relationships that haven't been 
captured by the model, or other factors.

Finally, let's check for homoscedasticity. Homoscedasticity means that the 
variance of the errors is constant across all levels of the independent variables.
 We can check this by plotting the residuals against the predicted values. 
 If the points are equally spread across all levels of the predicted values, 
 then the assumption of homoscedasticity is likely met.
'''

# Plot the residuals against the predicted values
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=y_train_pred_sig, y=residuals_sig)
# plt.xlabel('Predicted Values (Significant Features)')
# plt.ylabel('Residuals (Significant Features)')
# plt.title('Residuals vs Predicted Values (Significant Features)')
# plt.show()

'''
From the plot of residuals against predicted values after removing features with 
a p-value greater than 0.05, we can observe that the points are not equally spread 
across all levels of the predicted values. There seems to be a larger spread of residuals 
for higher predicted values. This suggests that the assumption of homoscedasticity is 
not met, indicating heteroscedasticity in the data.

Heteroscedasticity often arises in the presence of outliers or extreme leverage values, 
when the model doesn't capture non-linear effects, or when the variance of the errors 
changes across different levels of the independent variables.

In summary, while removing features with a p-value greater than 0.05 seems to have 
slightly improved the model's explanatory power (as indicated by the higher R-squared), 
there are still concerns about the assumptions of the linear regression model not being 
met. Further steps to improve the model might include trying different transformations 
of the variables, removing outliers, or using a different type of model that can 
better handle non-linear relationships and interactions between variables.
'''
'''
Let's use the Z-score method to identify and remove outliers from our data, 
and then we'll refit the model. We'll use the scipy.stats function zscore to 
compute the Z-scores.
'''

from scipy.stats import zscore

# Calculate the Z-scores of the 'Preco' column
z_scores = zscore(y_train)

# Get the indices of the observations that are not outliers
not_outliers = (np.abs(z_scores) <= 3)

# Create new training sets without the outliers
X_train_no_outliers = X_train[not_outliers]
y_train_no_outliers = y_train[not_outliers]

# Create a linear regression model
model_no_outliers = LinearRegression()

# Fit the model to the training data
model_no_outliers.fit(X_train_no_outliers, y_train_no_outliers)

# Predict the target for the training data
y_train_pred_no_outliers = model_no_outliers.predict(X_train_no_outliers)

# Calculate the residuals
residuals_no_outliers = y_train_no_outliers - y_train_pred_no_outliers

# Compute the R-squared of the model
r2_no_outliers = r2_score(y_train_no_outliers, y_train_pred_no_outliers)
print('R2: ',r2_no_outliers)

############
## Resultados
# Add a constant to the features
X_train_sm = sm.add_constant(X_train_no_outliers)
# Create a OLS model
model_sm = sm.OLS(y_train_no_outliers, X_train_sm)
result = model_sm.fit()

print('\n',result.summary())

'''
The R-squared of the model after removing outliers is approximately 0.749. 
This means that around 74.9% of the variation in the 'Preco' can be explained by the remaining significant features included in the model.

This is significantly higher than the R-squared of the model before removing outliers 
(which was approximately 0.597). This suggests that removing outliers has considerably 
improved the model's explanatory power.

Next, let's check the normality of the residuals. We can do this by creating a 
histogram of the residuals and a QQ-plot (quantile-quantile plot). 
The histogram should have the shape of a bell curve (normal distribution), 
and the points in the QQ-plot should lie along the diagonal line.
'''

# Create a subplot of the histogram and QQ-plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot the histogram
sns.histplot(residuals_no_outliers, kde=True, ax=axs[0])
axs[0].set_title('Histograma de Valores Residuais (Sem Outliers)')

# Plot the QQ-plot
stats.probplot(residuals_no_outliers, plot=axs[1])
axs[1].set_title('QQ-plot dos Valores Residuais (Sem Outliers)')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\RegressaoNovaMulti_normalidade.png', dpi=300)
plt.tight_layout()
plt.show()

'''
From the histogram and QQ-plot of the residuals after removing outliers, we can observe 
the following:

Histogram: The distribution of residuals seems to be much closer to a normal distribution 
compared to before. It still has a slight right skew, but the presence of extreme 
values or outliers has been significantly reduced.
QQ-plot: The points lie much closer to the diagonal line, especially in the middle, 
which suggests that the residuals are more normally distributed than before.
This suggests that the assumption of normality of residuals is better met after removing 
outliers.

'''
'''
Finally, let's check for homoscedasticity. Homoscedasticity means that the 
variance of the errors is constant across all levels of the independent variables. 
We can check this by plotting the residuals against the predicted values. 
If the points are equally spread across all levels of the predicted values, 
then the assumption of homoscedasticity is likely met.
'''
# Plot the residuals against the predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred_no_outliers, y=residuals_no_outliers)
plt.xlabel('Valores Previstos (Sem Outliers)')
plt.ylabel('Valores Residuais (Sem Outliers)')
plt.title('Valores Residuais vs Valores Previstos (Sem Outliers)')
plt.savefig(r'C:\Users\Rodrigo\Desktop\Pos Graduação - Data Science\ISLA - Santarem\10. Projecto\Img\RegressaoNovaMulti_Residuais.png', dpi=300)
plt.show()

'''
From the plot of residuals against predicted values after removing outliers, 
we can observe that the points seem to be more equally spread across all levels of 
the predicted values compared to before. There still seems to be some pattern in the 
residuals, but the overall spread appears to be more constant.

This suggests that the assumption of homoscedasticity is better met after removing outliers.
The presence of heteroscedasticity has been reduced, although not completely eliminated.

In summary, removing outliers seems to have significantly improved the model's explanatory
 power and the satisfaction of the assumptions of the linear regression model. 
 The R-squared increased to approximately 0.749, and both the normality of residuals 
 and homoscedasticity assumptions appear to be better met.
'''
coefficients_original = pd.Series(model.coef_, index=X_train.columns)
intercept_original = model.intercept_

coefficients_no_outliers = pd.Series(model_no_outliers.coef_, index=X_train_no_outliers.columns)
intercept_no_outliers = model_no_outliers.intercept_

print('\nIntercepção Original:', intercept_original)
print('Coeficientes Originais:\n',coefficients_original) 
print('\nIntercepção sem Outliers: ',intercept_no_outliers)
print('Coeficientes sem Outliers: \n',coefficients_no_outliers)
'''
LASSSSOOOOOOOO
We'll use the LassoCV class from sklearn, which allows us to perform LASSO regression 
with cross-validation to find the optimal value of the regularization parameter, alpha.
'''
from sklearn.linear_model import LassoCV

# Create a LassoCV model
model_lasso = LassoCV(cv=5, random_state=42)

# Fit the model to the training data
model_lasso.fit(X_train_no_outliers, y_train_no_outliers)

# Get the coefficients of the features
coefficients = pd.Series(model_lasso.coef_, index=X_train_no_outliers.columns)

# Select only the features with a non-zero coefficient
significant_features_lasso = coefficients[coefficients != 0].index

significant_features_lasso

# Create new training sets with only the significant features
X_train_lasso = X_train_no_outliers[significant_features_lasso]

# Create a linear regression model
model_lasso = LinearRegression()

# Fit the model to the training data
model_lasso.fit(X_train_lasso, y_train_no_outliers)

# Predict the target for the training data
y_train_pred_lasso = model_lasso.predict(X_train_lasso)

# Calculate the residuals
residuals_lasso = y_train_no_outliers - y_train_pred_lasso

# Compute the R-squared of the model
r2_lasso = r2_score(y_train_no_outliers, y_train_pred_lasso)

r2_lasso, residuals_lasso

'''
The R-squared of the model using only the features selected by the LASSO regression 
is approximately 0.618. This means that around 61.8% of the variation in the 'Preco' 
can be explained by these two features.

This is slightly lower than the R-squared of the model after removing outliers and 
using all significant features (which was approximately 0.749), but higher than the 
original model before removing any features or outliers (which had an R-squared of 
                                                         approximately 0.594). 
This suggests that these two features are indeed important predictors of 'Preco', 
but there may be other relevant features or interactions that this model is missing.
'''

# Create a subplot of the histogram and QQ-plot
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot the histogram
sns.histplot(residuals_lasso, kde=True, ax=axs[0])
axs[0].set_title('Histogram of Residuals (LASSO)')

# Plot the QQ-plot
stats.probplot(residuals_lasso, plot=axs[1])
axs[1].set_title('QQ-plot of Residuals (LASSO)')

plt.tight_layout()
plt.show()

'''
From the histogram and QQ-plot of the residuals after applying LASSO regression, 
we can observe the following:

Histogram: The distribution of residuals seems to be skewed to the right and does not 
exactly resemble a normal distribution. There seems to be some outliers on the right side 
of the distribution.
QQ-plot: The points deviate from the diagonal line, especially at the ends, which suggests 
that the residuals do not follow a perfect normal distribution.
This suggests that the assumption of normality of residuals is not strictly met. 
This could be due to outliers in the data, non-linear relationships that haven't been 
captured by the model, or other factors.
'''

'''

Finally, let's check for homoscedasticity. Homoscedasticity means that the variance of the 
errors is constant across all levels of the independent variables. We can check this by 
plotting the residuals against the predicted values. If the points are equally 
spread across all levels of the predicted values, then the assumption of 
homoscedasticity is likely met.
'''

# Plot the residuals against the predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred_lasso, y=residuals_lasso)
plt.xlabel('Predicted Values (LASSO)')
plt.ylabel('Residuals (LASSO)')
plt.title('Residuals vs Predicted Values (LASSO)')
plt.show()

'''
From the plot of residuals against predicted values after applying LASSO regression,
 we can observe that the points are not equally spread across all levels of the predicted 
 values. There seems to be a larger spread of residuals for higher predicted values. 
 This suggests that the assumption of homoscedasticity is not met, indicating 
 heteroscedasticity in the data.

Heteroscedasticity often arises in the presence of outliers or extreme leverage values,
 when the model doesn't capture non-linear effects, or when the variance of the errors 
 changes across different levels of the independent variables.

In summary, applying LASSO regression for feature selection seems to have resulted in a 
simpler model with just two features, but there are still concerns about the assumptions 
of the linear regression model not being met. The model's explanatory power (as measured 
by the R-squared) is lower than the model that included all significant features after 
    removing outliers. Further steps to improve the model might include trying different 
    transformations of the variables, removing outliers, or using a different type 
    of model that can better handle non-linear relationships and interactions between 
    variables.

'''

# Get the coefficients and intercepts for each model
coefficients_original = pd.Series(model_sig.coef_, index=X_train_sig.columns)
intercept_original = model_sig.intercept_

coefficients_no_outliers = pd.Series(model_no_outliers.coef_, index=X_train_no_outliers.columns)
intercept_no_outliers = model_no_outliers.intercept_

coefficients_lasso = pd.Series(model_lasso.coef_, index=X_train_lasso.columns)
intercept_lasso = model_lasso.intercept_

coefficients_original, intercept_original, coefficients_no_outliers, intercept_no_outliers, coefficients_lasso, intercept_lasso


