import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.options.mode.chained_assignment = None

# Read data
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Define features and target
features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X = testData[features]
y = testData["median_house_value"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Separate numerical and categorical features
numerical_features = X.select_dtypes(include=np.number)
categorical_features = X.select_dtypes(exclude=np.number)

# Impute missing values for numerical features
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train[numerical_features.columns])
X_test_numerical_imputed = numeric_imputer.transform(X_test[numerical_features.columns])

# Impute missing values for categoric features
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_train_categorical_imputed = categoric_imputer.fit_transform(X_train[categorical_features.columns])
X_test_categorical_imputer = categoric_imputer.transform(X_test[categorical_features.columns])

# Encode categorical feature
encoder = OneHotEncoder()
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()  # Convert to dense array
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputer).toarray()  # Convert to dense array

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Transform combined features
poly = PolynomialFeatures(degree=3)
poly.fit(X_train_encoded)
X_train_Polynomial = poly.transform(X_train_encoded)
X_test_Polynomial = poly.transform(X_test_encoded)

# Scale the polynomial features
scaler = MinMaxScaler()
scaler.fit(X_train_Polynomial)
X_train_scaled = scaler.transform(X_train_Polynomial)
X_test_scaled = scaler.transform(X_test_Polynomial)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_prediction = model.predict(X_test_scaled)

# Compute evaluation metrics
mse_Testing = mean_squared_error(y_test, y_prediction)
mae_Testing = mean_absolute_error(y_test, y_prediction)
rmse_Testing = mean_squared_error(y_test, y_prediction, squared=False)
mape_Testing = np.mean(np.abs((y_test - y_prediction) / y_test)) * 100
r2_testing_Testing = r2_score(y_test, y_prediction)

# Print evaluation metrics
print('Test - MAE: {:.4f}'.format(mae_Testing))
print('Test - RMSE: {:.4f}'.format(rmse_Testing))
print('Test - R2 score: {:.4f}'.format(r2_testing_Testing))
print('Test - MAPE: {:.4f}'.format(mape_Testing))
print('Test - MSE: {:.4f}'.format(mse_Testing))

# Visualise : 
# Visualize
plt.figure(figsize=(10, 6))

# Plot actual data
plt.scatter(y_test, y_prediction, color='blue', label='Actual vs Predicted')

# Plotting the polynomial curve
plt.plot(np.linspace(0, np.max(y_test), 100), 
         model.predict(scaler.transform(poly.transform(np.concatenate((
             np.linspace(0, np.max(y_test), 100).reshape(-1, 1),  # Reshape because poly.transform expects 2D array
             np.ones((100, X_train_encoded.shape[1] - 1))  # Fill other features with ones (assuming no impact)
         ), axis=1)))), 
         color='red', label='Polynomial Curve')

plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

