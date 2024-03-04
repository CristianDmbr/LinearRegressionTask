import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Define features and target variable
features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

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

# Define hyperparameters
alpha_value = 0.2
l1_ratio_value = 0.3
fit_intercept_value = True
max_iter_value = 100
tol_value = 1e-4

# Create ElasticNet model with specified hyperparameters
model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio_value, fit_intercept=fit_intercept_value,
                   max_iter=max_iter_value, tol=tol_value)

# Fit the model
model.fit(X_train_scaled, y_train)
y_pred_test = model.predict(X_test_scaled)

# Metrics
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)

# Print metrics
print('Test - MAE: {:.4f}'.format(mae_test))
print('Test - RMSE: {:.4f}'.format(rmse_test))
print('Test - R2 score: {:.4f}'.format(r2_test))

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot of data
plt.scatter(y_test, y_pred_test, color='blue', label='Test Data')

# Plotting individual data points
plt.scatter(y_test, y_test, color='green', label='Actual Values', marker='x')

# Plotting the ElasticNet regression line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='ElasticNet Regression Line')

# Labels and title
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Elastic Net Regression')

# Legend
plt.legend()

plt.show()