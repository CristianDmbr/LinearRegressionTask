import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.metrics

# Load dataset
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Feature selection
features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population","households", "median_income", "ocean_proximity"]
features = ["housing_median_age"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

# Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

# Numerical features
X_train_numerical = X_train_raw.select_dtypes(include=np.number)
X_test_numerical = X_test_raw.select_dtypes(include=np.number)

# Impute missing values
numeric_imputer = SimpleImputer(strategy="mean")
numeric_imputer.fit(X_train_numerical)
X_train_numerical_imputed = numeric_imputer.transform(X_train_numerical)
X_test_numerical_imputed = numeric_imputer.transform(X_test_numerical)

# Scale numerical features
scaler = MinMaxScaler()
scaler.fit(X_train_numerical_imputed)
X_train_numerical_scaled = scaler.transform(X_train_numerical_imputed)
X_test_numerical_scaled = scaler.transform(X_test_numerical_imputed)

# Apply polynomial regression
degree = 1
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train_numerical_scaled)
X_test_poly = poly.transform(X_test_numerical_scaled)

# Linear regression on polynomial features
obj = LinearRegression()
obj.fit(X_train_poly, y_train)

# Predictions
y_pred_train = obj.predict(X_train_poly)
y_pred_test = obj.predict(X_test_poly)

# Plotting
plt.scatter(X_test_numerical_scaled[:, 0], y_test, color='black', label='y_test')  # Observed y values
plt.scatter(X_test_numerical_scaled[:, 0], y_pred_test, color='blue', label='y_pred')  # Predicted y values
plt.xlabel('Feature')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

# Evaluate the model
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_train, y_pred_train),sklearn.metrics.r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format( sklearn.metrics.mean_squared_error(y_test, y_pred_test),sklearn.metrics.r2_score(y_test, y_pred_test)))
