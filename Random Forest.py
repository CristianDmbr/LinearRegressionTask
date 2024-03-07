import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error

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
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train the random forest regressor
randomForest = RandomForestRegressor(n_estimators=150, 
                                    max_depth=15, 
                                    min_samples_split=2,
                                    min_samples_leaf=5,
                                    random_state=0
                                    )
randomForest.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = randomForest.predict(X_train_scaled)
y_pred_test = randomForest.predict(X_test_scaled)

# Calculate metrics for training set
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
r2_train = r2_score(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
medae_train = median_absolute_error(y_train, y_pred_train)
msle_train = mean_squared_log_error(y_train, y_pred_train)

# Calculate metrics for testing set
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
medae_test = median_absolute_error(y_test, y_pred_test)
msle_test = mean_squared_log_error(y_test, y_pred_test)

# Print metrics for training set
print("\n Training Metrics : ")
print('Training - MAE: {:.4f}'.format(mae_train))
print('Training - RMSE: {:.4f}'.format(rmse_train))
print('Training - R2 score: {:.4f}'.format(r2_train))
print('Training - MSE: {:.4f}'.format(mse_train))
print('Training - Median Absolute Error: {:.4f}'.format(medae_train))
print('Training - Mean Squared Log Error: {:.4f}'.format(msle_train))

# Print metrics for testing set
print("\n Testing Metrics : ")
print('Testing - MAE: {:.4f}'.format(mae_test))
print('Testing - RMSE: {:.4f}'.format(rmse_test))
print('Testing - R2 score: {:.4f}'.format(r2_test))
print('Testing - MSE: {:.4f}'.format(mse_test))
print('Testing - Median Absolute Error: {:.4f}'.format(medae_test))
print('Testing - Mean Squared Log Error: {:.4f}'.format(msle_test))

# Plotting actual vs predicted values for training set
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Predicted')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Actual')
plt.title('Actual vs Predicted Values (Training Set)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Plotting actual vs predicted values for testing set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual')
plt.title('Actual vs Predicted Values (Testing Set)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()