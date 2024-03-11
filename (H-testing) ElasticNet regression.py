import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV

# Load dataset
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Define features and target variable
features = ["No.","longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

X_raw = X_raw.drop(columns=['No.'])

# Split data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=0)

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number)
categorical_features = X_raw.select_dtypes(exclude=np.number)

# Impute missing values for numerical features
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features.columns])
X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features.columns])

# Impute missing values for categorical features
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features.columns])
X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features.columns])

# Encode categorical features
encoder = OneHotEncoder()
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()  # Convert to dense array
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed).toarray()  # Convert to dense array

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10],  
              'l1_ratio': [0, 0.1, 0.5, 0.7, 1], 
              'max_iter': [100, 200, 300, 400], 
              'tol': [0.001, 0.01, 0.1]}  

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(ElasticNet(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train_scaled, y_train)

# Best parameters found
print("Best parameters:", grid_search.best_params_)

# ElasticNet model with best hyperparameters
best_model = grid_search.best_estimator_

# Predictions
y_pred_train_best = best_model.predict(X_train_scaled)
y_pred_test_best = best_model.predict(X_test_scaled)

# Metrics with best hyperparameters
mae_train_best = mean_absolute_error(y_train, y_pred_train_best)
rmse_train_best = mean_squared_error(y_train, y_pred_train_best, squared=False)
r2_train_best = r2_score(y_train, y_pred_train_best)
medae_train_best = median_absolute_error(y_train, y_pred_train_best)
mape_train_best = np.mean(np.abs((y_train - y_pred_train_best) / y_train)) * 100
mse_train_best = mean_squared_error(y_train, y_pred_train_best)

mae_test_best = mean_absolute_error(y_test, y_pred_test_best)
rmse_test_best = mean_squared_error(y_test, y_pred_test_best, squared=False)
r2_test_best = r2_score(y_test, y_pred_test_best)
medae_test_best = median_absolute_error(y_test, y_pred_test_best)
mape_test_best = np.mean(np.abs((y_test - y_pred_test_best) / y_test)) * 100
mse_test_best = mean_squared_error(y_test, y_pred_test_best)

# Print metrics with best hyperparameters
print("\nTraining Metrics with best hyperparameters:")
print('Training - MAE: {:.4f}'.format(mae_train_best))
print('Training - RMSE: {:.4f}'.format(rmse_train_best))
print('Training - R2 score: {:.4f}'.format(r2_train_best))
print('Training - Median Absolute Error: {:.4f}'.format(medae_train_best))
print('Training - MAPE: {:.4f}'.format(mape_train_best))
print('Training - MSE: {:.4f}'.format(mse_train_best))

print("\nTesting Metrics with best hyperparameters:")
print('Testing - MAE: {:.4f}'.format(mae_test_best))
print('Testing - RMSE: {:.4f}'.format(rmse_test_best))
print('Testing - R2 score: {:.4f}'.format(r2_test_best))
print('Testing - Median Absolute Error: {:.4f}'.format(medae_test_best))
print('Testing - MAPE: {:.4f}'.format(mape_test_best))
print('Testing - MSE: {:.4f}'.format(mse_test_best))