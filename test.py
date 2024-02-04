import numpy as np # A useful package for dealing with mathematical processes, we will be using it this week for vectors and matrices
import pandas as pd # A common package for viewing tabular data
import sklearn.linear_model, sklearn.datasets # We want to be able to access the sklearn datasets again, also we are using some model evaluation
from sklearn.preprocessing import StandardScaler, MinMaxScaler # We will be using the imbuilt sclaing functions sklearn provides
import matplotlib.pyplot as plt # We will be using Matplotlib for our graphs
from sklearn.preprocessing import PolynomialFeatures # A preprocessing function allowing us to include polynomial features into our model

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # We will be using these to encode categorical features
from sklearn.model_selection import train_test_split # An sklearn library for outomatically splitting our data
from sklearn.impute import SimpleImputer # Performs basic imputations when doing preprocessing
pd.options.mode.chained_assignment = None  # default='warn'

testData = pd.read_csv('housing_coursework_entire_dataset_23-24.csv')

# STARTING OF DATAPREPROCESSING

features = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

# The training data is separated into numerical features and categorical features
X_train_numerical = X_train_raw.select_dtypes(include=np.number)
X_train_categorical = X_train_raw.select_dtypes(exclude=np.number)

# Create imputers for filling in missing records
numeric_imputer = SimpleImputer(strategy="mean")
categorical_imputer = SimpleImputer(strategy="most_frequent")

# Teach the imputers the mean and most frequent of the database
numeric_imputer.fit(X_train_numerical)
categorical_imputer.fit(X_train_categorical)

# The training collumns are now updated using the imputers to fill in the missing values
X_train_numerical_imputated = numeric_imputer.transform(X_train_numerical)
X_train_categorical_imputated = categorical_imputer.transform(X_train_categorical)

# Split the database into testing sets and fill in any missing values with imputers
X_test_numerical = X_test_raw.select_dtypes(include=np.number)
X_test_categorical = X_test_raw.select_dtypes(exclude=np.number)
X_test_numerical_imputated = numeric_imputer.transform(X_test_numerical)
X_test_categorical_imputated = categorical_imputer.transform(X_test_categorical)

# Sets the scaler objective to be between 0 to 1
scaler = MinMaxScaler()

# Fits the scaler to the training data
scaler.fit(X_train_numerical_imputated)

# Training data is scaled
X_train_numerical_scaled = scaler.transform(X_train_numerical_imputated)
X_test_numerical_scaled = scaler.transform(X_test_numerical_imputated)

# Creates the encoder object, the handle_unknown = "ignore" parameters sets any category that has not been seen 
# as a 0
encoder = OneHotEncoder(handle_unknown="ignore")

# The encoder is fited on the training imputated categorical data
encoder.fit(X_train_categorical_imputated)

# Transform the test and training categorical data
X_train_oneHotEncoder = encoder.transform(X_train_categorical_imputated)
X_test_oneHotEncoder = encoder.transform(X_test_categorical_imputated)

# Convert the OneHotEncoder into a sparse matrix
X_train_oneHotEncoder_dense = X_train_oneHotEncoder.toarray()
X_test_oneHotEncoder_dense = X_test_oneHotEncoder.toarray()


# Combine the two numerical and categorical matrix 
X_train = np.concatenate([X_train_numerical_scaled, X_train_oneHotEncoder_dense], axis = 1)
X_test = np.concatenate([X_test_numerical_scaled, X_test_oneHotEncoder_dense], axis = 1)

# END OF DATA PREPROCESSING





# Create an instance of the linear regression 
# The objective is used to perform linear regression operations
obj = sklearn.linear_model.LinearRegression()






#print(X_raw.select_dtypes(exclude=np.number).describe())
#print(X_raw.select_dtypes(include=np.number).describe())