import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import tree

# Read the data
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# Select features and target variable
features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

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

# Train the random forest regressor
randomForest = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=2, min_samples_leaf=10, random_state=0)
randomForest.fit(X_train_scaled, y_train)

# Predictions
y_pred_test = randomForest.predict(X_test_scaled)

# Calculate metrics
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)

# Print metrics
print('Test - MAE: {:.4f}'.format(mae_test))
print('Test - RMSE: {:.4f}'.format(rmse_test))
print('Test - R2 score: {:.4f}'.format(r2_test))
print('Test - MSE: {:.4f}'.format(mse_test))

# Visualize the model's predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Random Forest)')
plt.show()

# Get feature names after one-hot encoding
encoded_feature_names = encoder.get_feature_names_out(input_features=categorical_features.columns)

# Concatenate all feature names
all_feature_names = np.concatenate((numerical_features.columns, encoded_feature_names))

# Visualize one of the trees in the forest
plt.figure(figsize=(20, 10))
tree.plot_tree(randomForest.estimators_[0], feature_names=all_feature_names, filled=True)
plt.show()
