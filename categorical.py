import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.metrics

pd.options.mode.chained_assignment = None  # default='warn'

# Load dataset
testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

# STARTING OF DATAPREPROCESSING

features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
features = ["ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

# Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

# Separate numerical and categorical features
X_train_numerical = X_train_raw.select_dtypes(include=np.number)
X_train_categorical = X_train_raw.select_dtypes(exclude=np.number)

# Impute missing values for categorical features
categorical_imputer = SimpleImputer(strategy="most_frequent")
categorical_imputer.fit(X_train_categorical)
X_train_categorical_imputed = categorical_imputer.transform(X_train_categorical)

# Impute missing values for test set categorical features
X_test_numerical = X_test_raw.select_dtypes(include=np.number)
X_test_categorical = X_test_raw.select_dtypes(exclude=np.number)
X_test_categorical_imputed = categorical_imputer.transform(X_test_categorical)

# Min-max scaling
scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown="ignore")

# Fit encoder on training set
encoder.fit(X_train_categorical_imputed)

# Transform categorical data
X_train_oneHotEncoder = encoder.transform(X_train_categorical_imputed)
X_test_oneHotEncoder = encoder.transform(X_test_categorical_imputed)

# Convert encoder output to dense array
X_train_oneHotEncoder_dense = X_train_oneHotEncoder.toarray()
X_test_oneHotEncoder_dense = X_test_oneHotEncoder.toarray()

# Combine numerical and categorical matrices
X_train = X_train_oneHotEncoder_dense
X_test = X_test_oneHotEncoder_dense

# END OF DATA PREPROCESSING

# Polynomial Regression
degree = 2  # You can adjust the degree of the polynomial
poly = PolynomialFeatures(degree=degree)

# Transform the features to include polynomial terms
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create an instance of the linear regression
obj = sklearn.linear_model.LinearRegression()

# Apply polynomial regression to the training sets
obj.fit(X_train_poly, y_train)

# Make predictions on training data
y_pred_train = obj.predict(X_train_poly)

# Make predictions on testing set
y_pred_test = obj.predict(X_test_poly)

# Plot outputs
X_disp = X_test[:, 0]
plt.scatter(X_disp, y_test, color='black', label='y_test')  # Observed y values
plt.scatter(X_disp, y_pred_test, color='blue', label='y_pred')  # Predicted y values
plt.xlabel('Feature')
plt.ylabel('Final Grade')
plt.legend()
plt.show()

# The mean squared error loss and R2 for the test and train data
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_train, y_pred_train),sklearn.metrics.r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_test, y_pred_test),sklearn.metrics.r2_score(y_test, y_pred_test)))