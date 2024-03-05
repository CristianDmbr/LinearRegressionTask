import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV


# Read the data
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Define features and target variable
features = ["No.","longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

X_raw = X_raw.drop(columns=['No.'])

# Split data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, train_size=0.80, shuffle=True, random_state=0)

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
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed).toarray()

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Create the grid search object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='neg_mean_squared_error',
                           verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found
print("Best parameters found:")
print(grid_search.best_params_)

# Get the best estimator
best_rf = grid_search.best_estimator_

# Predictions
y_pred_train_best = best_rf.predict(X_train_scaled)
y_pred_test_best = best_rf.predict(X_test_scaled)

# Calculate metrics for training set
mae_train_best = mean_absolute_error(y_train, y_pred_train_best)
rmse_train_best = mean_squared_error(y_train, y_pred_train_best, squared=False)
r2_train_best = r2_score(y_train, y_pred_train_best)
mse_train_best = mean_squared_error(y_train, y_pred_train_best)
medae_train_best = median_absolute_error(y_train, y_pred_train_best)
msle_train_best = mean_squared_log_error(y_train, y_pred_train_best)

# Calculate metrics for testing set
mae_test_best = mean_absolute_error(y_test, y_pred_test_best)
rmse_test_best = mean_squared_error(y_test, y_pred_test_best, squared=False)
r2_test_best = r2_score(y_test, y_pred_test_best)
mse_test_best = mean_squared_error(y_test, y_pred_test_best)
medae_test_best = median_absolute_error(y_test, y_pred_test_best)
msle_test_best = mean_squared_log_error(y_test, y_pred_test_best)

# Print metrics for training set
print('\nTraining - MAE: {:.4f}'.format(mae_train_best))
print('Training - RMSE: {:.4f}'.format(rmse_train_best))
print('Training - R2 score: {:.4f}'.format(r2_train_best))
print('Training - MSE: {:.4f}'.format(mse_train_best))
print('Training - Median Absolute Error: {:.4f}'.format(medae_train_best))
print('Training - Mean Squared Log Error: {:.4f}'.format(msle_train_best))

# Print metrics for testing set
print('\nTesting - MAE: {:.4f}'.format(mae_test_best))
print('Testing - RMSE: {:.4f}'.format(rmse_test_best))
print('Testing - R2 score: {:.4f}'.format(r2_test_best))
print('Testing - MSE: {:.4f}'.format(mse_test_best))
print('Testing - Median Absolute Error: {:.4f}'.format(medae_test_best))
print('Testing - Mean Squared Log Error: {:.4f}'.format(msle_test_best))
