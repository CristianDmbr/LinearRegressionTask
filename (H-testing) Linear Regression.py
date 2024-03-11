import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV

# Read data
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Define features and target variable
features = ["No.","longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X = testData[features]
y = testData["median_house_value"]

X = X.drop(columns=['No.'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Separate numerical and categorical features
numerical_features = X.select_dtypes(include=np.number)
categorical_features = X.select_dtypes(exclude=np.number)

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
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()  
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed).toarray() 

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Define the parameter grid
param_grid = {
    'fit_intercept': [True, False]
}


# Create the GridSearchCV object
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error')

# Perform the grid search
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train the model with the best parameters
best_model = LinearRegression(**best_params)
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

# Evaluate the model
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
medae_train = median_absolute_error(y_train, y_pred_train)
medae_test = median_absolute_error(y_test, y_pred_test)

print("\n Training Metrics : ")
print("Train MAE:", mae_train)
print("Train RMSE:", rmse_train)
print("Train R^2:", r2_train)
print("Train MSE:", mse_train)
print("Train MEDAE:", medae_train)

print("\n Testing Metrics : ")
print("Test MAE:", mae_test)
print("Test RMSE:", rmse_test)
print("Test R^2:", r2_test)
print("Test MSE:", mse_test)
print("Test MEDAE:", medae_test)