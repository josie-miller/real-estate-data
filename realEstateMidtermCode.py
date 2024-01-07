#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:22:31 2023
Midtern
@author: josephinemiller
"""
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Step a: EDA and Data Preparation
# Load the dataset and remove the 'No' column
file_path = '/Users/josephinemiller/Desktop/real_estate.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['No'])

# Split data into features (X) and target (Y)
X = data.drop(columns=['Y house price of unit area'])
Y = data['Y house price of unit area']

# Descriptive statistics
print(X.describe())

# Histograms
plt.figure(figsize=(12, 10))
for i, column in enumerate(X.columns):
    plt.subplot(3, 2, i+1)
    plt.hist(X[column], bins=20)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter plots
for column in X.columns:
    plt.scatter(X[column], Y)
    plt.xlabel(column)
    plt.ylabel('Y house price of unit area')
    plt.show()
    

# Step b: Preprocessing
# Based on your observations from EDA, decide if any preprocessing is required.

# Example: If you decide to standardize the data
# Log transform the skewed feature
data['X3 distance to the nearest MRT station'] = data['X3 distance to the nearest MRT station'].apply(lambda x: np.log1p(x))

# Standardize all input features
scaler = StandardScaler()
data_scaled = data.copy()  # Create a copy of the original dataframe
input_features = ['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

# Standardize each of the input features
data_scaled[input_features] = scaler.fit_transform(data[input_features])

# Create a new dataframe 'X_scaled' with the updated features
X_scaled = data_scaled[input_features]

# Display the first 5 rows of the scaled features
print(X_scaled.head())

# Step c

# Step 1: Build a Multilinear Regression Model using RFE
# Initialize the Linear Regression model
regression_model = LinearRegression()

# Create an RFE selector with desired number of features 
n_Features_to_select = 4 # Note that this was chosen through testing 1-5, with this resulting in the lowest MSE overall
rfe = RFE(estimator=regression_model, n_features_to_select=n_Features_to_select)

# Fit the RFE selector to the data
rfe.fit(X_scaled, Y)

# Get the selected features from RFE
selected_features = list(X_scaled.columns[rfe.support_])
print("Selected Features from RFE:", selected_features)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled[selected_features], Y, test_size=0.2, random_state=0)

# Fit the multilinear regression model on the selected features
regression_model.fit(X_train, Y_train)
Y_pred = regression_model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Multilinear Regression MSE (RFE selected features):", mse)
'''Notes: mse (mean square error) is the test performance measure I'm using'''


# Step d: Build Regularized Regression Models (Lasso and Ridge)
# Initialize Lasso and Ridge models
lasso_model = Lasso()
ridge_model = Ridge()

# Fit Lasso and Ridge models
lasso_model.fit(X_train, Y_train)
ridge_model.fit(X_train, Y_train)

# Predict with Lasso and Ridge models
Y_pred_lasso = lasso_model.predict(X_test)
Y_pred_ridge = ridge_model.predict(X_test)

# Calculate MSE for Lasso and Ridge models
mse_lasso = mean_squared_error(Y_test, Y_pred_lasso)
mse_ridge = mean_squared_error(Y_test, Y_pred_ridge)

print("Lasso MSE:", mse_lasso)
print("Ridge MSE:", mse_ridge)




# Step e: Cross-Validation (Loop for folds)
for num_folds in range(2, 11):
    cross_val_linear = cross_val_score(regression_model, X_scaled[selected_features], Y, cv=num_folds, scoring='neg_mean_squared_error')
    cross_val_lasso = cross_val_score(lasso_model, X_scaled[selected_features], Y, cv=num_folds, scoring='neg_mean_squared_error')
    cross_val_ridge = cross_val_score(ridge_model, X_scaled[selected_features], Y, cv=num_folds, scoring='neg_mean_squared_error')

    mse_linear_cv = -cross_val_linear.mean()
    mse_lasso_cv = -cross_val_lasso.mean()
    mse_ridge_cv = -cross_val_ridge.mean()

    print(f"\nCross-Validation Results (Folds: {num_folds}):")
    print(f"Multilinear Regression MSE: {mse_linear_cv}")
    print(f"Lasso MSE: {mse_lasso_cv}")
    print(f"Ridge MSE: {mse_ridge_cv}")
