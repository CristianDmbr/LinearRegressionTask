import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

# Read data
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Define features and target variable
features = ["No.", "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

X_raw = X_raw.drop(columns=['No.'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=0)

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number)
categorical_features = X_raw.select_dtypes(exclude=np.number)

# Impute missing values for numerical features
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train[numerical_features.columns])
X_test_numerical_imputed = numeric_imputer.transform(X_test[numerical_features.columns])

# Impute missing values for categorical features
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_train_categorical_imputed = categoric_imputer.fit_transform(X_train[categorical_features.columns])
X_test_categorical_imputed = categoric_imputer.transform(X_test[categorical_features.columns])

# Encode categorical features
encoder = OneHotEncoder()
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()  # Convert to dense array
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed).toarray()  # Convert to dense array

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Transform combined features
poly = PolynomialFeatures(degree=3)

# Use polynomial transformation on training data
X_train_Polynomial = poly.fit_transform(X_train_encoded)
X_test_Polynomial = poly.transform(X_test_encoded)

# Scale the polynomial features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_Polynomial)
X_test_scaled = scaler.transform(X_test_Polynomial)

# Define the parameters grid
param_grid = {
    'degree': [2, 3, 4],  # try different degrees of polynomial features
    'interaction_only': [True, False],  # whether to consider only interaction features
    'include_bias': [True, False]  # whether to include a bias column in the polynomial features
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the grid search to find the best hyperparameters
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_prediction_train_best = best_model.predict(X_train_scaled)
y_prediction_test_best = best_model.predict(X_test_scaled)

# Compute evaluation metrics
mae_train_best = mean_absolute_error(y_train, y_prediction_train_best)
mae_test_best = mean_absolute_error(y_test, y_prediction_test_best)
rmse_train_best = mean_squared_error(y_train, y_prediction_train_best, squared=False)
rmse_test_best = mean_squared_error(y_test, y_prediction_test_best, squared=False)
r2_train_best = r2_score(y_train, y_prediction_train_best)
r2_test_best = r2_score(y_test, y_prediction_test_best)
mse_train_best = mean_squared_error(y_train, y_prediction_train_best)
mse_test_best = mean_squared_error(y_test, y_prediction_test_best)
medae_train_best = median_absolute_error(y_train, y_prediction_train_best)
medae_test_best = median_absolute_error(y_test, y_prediction_test_best)
mape_train_best = np.mean(np.abs((y_train - y_prediction_train_best) / y_train)) * 100
mape_test_best = np.mean(np.abs((y_test - y_prediction_test_best) / y_test)) * 100

# Print evaluation metrics
print("\n Training Metrics (Best Model): ")
print("Train MAE:", mae_train_best)
print("Train RMSE:", rmse_train_best)
print("Train R^2:", r2_train_best)
print("Train MSE:", mse_train_best)
print("Train MEDAE:", medae_train_best)
print("Train MAPE:", mape_train_best)

print("\n Testing Metrics (Best Model): ")
print("Test MAE:", mae_test_best)
print("Test RMSE:", rmse_test_best)
print("Test R^2:", r2_test_best)
print("Test MSE:", mse_test_best)
print("Test MEDAE:", medae_test_best)
print("Test MAPE:", mape_test_best)