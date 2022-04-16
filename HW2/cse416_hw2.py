from math import sqrt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add any additional imports here
# TODO

np.random.seed(416)

# Import data
sales = pd.read_csv('home_data.csv') 
sales = sales.sample(frac=0.01) 

# All of the features of interest
selected_inputs = [
    'bedrooms', 
    'bathrooms',
    'sqft_living', 
    'sqft_lot', 
    'floors', 
    'waterfront', 
    'view', 
    'condition', 
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 
    'yr_renovated'
]

# Compute the square and sqrt of each feature
all_features = []
for data_input in selected_inputs:
    square_feet = data_input + '_square'
    sqrt_feet = data_input + '_sqrt'
    
    # Q1: Compute the square and square root as two new features
    sales[square_feet] = sales[data_input]**2
    sales[sqrt_feet] = sales[data_input]**0.5

    all_features.extend([data_input, square_feet, sqrt_feet])

price = sales['price']
sales = sales[all_features]

# Train test split
train_and_validation_sales, test_sales, train_and_validation_price, test_price = \
    train_test_split(sales, price, test_size=0.2)
train_sales, validation_sales, train_price, validation_price = \
    train_test_split(train_and_validation_sales, train_and_validation_price, test_size=.125) # .10 (validation) of .80 (train + validation)


# Q2: Standardize data
# TODO
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_sales)
train_sales = scaler.transform(train_sales)
validation_sales = scaler.transform(validation_sales)
test_sales = scaler.transform(test_sales)

# Q3: Train baseline model
# TODO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

test_model = LinearRegression().fit(train_sales, train_price)

test_rmse_unregularized = np.sqrt(mean_squared_error(list(test_price), list(test_model.predict(test_sales))))

# Train Ridge models
l2_lambdas = np.logspace(-5, 5, 11, base = 10)

# Q4: Implement code to evaluate Ridge Regression with various L2 Penalties
# TODO 
from sklearn.linear_model import Ridge
from copy import copy

data = []
for i in l2_lambdas:
    ridge_model = Ridge(alpha = i).fit(train_sales, train_price)
    data.append({
        'l2_penalty': i,
        'model': copy(ridge_model),
        'train_rmse': mean_squared_error(train_price, ridge_model.predict(train_sales), squared=False),
        'validation_rmse': mean_squared_error(validation_price, ridge_model.predict(validation_sales), squared=False)
    })
ridge_data = pd.DataFrame(data)

# Q5: Analyze Ridge data
# TODO
index = ridge_data['validation_rmse'].idxmin()
row = ridge_data.loc[index]

best_l2 = row["l2_penalty"]
test_rmse_ridge = mean_squared_error(test_price, row["model"].predict(test_sales), squared=False)
num_zero_coeffs_ridge = list(row["model"].coef_).count(0)



# Train LASSO models
l1_lambdas = np.logspace(1, 7, 7, base=10)

# Q6: Implement code to evaluate LASSO Regression with various L1 penalties
# TODO
from sklearn.linear_model import Lasso
from copy import copy

data = []
for i in l1_lambdas:
    lasso_model = Lasso(alpha = i).fit(train_sales, train_price)
    data.append({
        'l1_penalty': i,
        'model': copy(lasso_model),
        'train_rmse': mean_squared_error(train_price, lasso_model.predict(train_sales), squared=False),
        'validation_rmse': mean_squared_error(validation_price, lasso_model.predict(validation_sales), squared=False)
    })
lasso_data = pd.DataFrame(data)


# Q7: LASSO Analysis
# TODO
index = lasso_data['validation_rmse'].idxmin()
row = lasso_data.loc[index]

best_l1 = row["l1_penalty"]
test_rmse_lasso = mean_squared_error(test_price, row["model"].predict(test_sales), squared=False)
num_zero_coeffs_lasso = list(row["model"].coef_).count(0)

