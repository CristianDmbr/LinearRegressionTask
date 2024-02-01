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

testData = pd.read_csv('housing_coursework_entire_dataset_23-24.csv') # Save it to a pandas dataframe

# Set the features for ML to learn from
features = ["No.","longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
target = ["median_house_value"]

X_raw = testData[features]
#Remove the No. because it's a irrelevant data feature.
X_raw = X_raw.drop(columns=['No.'])

y_raw = testData[target]

print(X_raw.select_dtypes(include=np.number).describe())
#print(X_raw.select_dtypes(exclude=np.number).describe())

#print(pd.unique(X_raw['ocean_proximity']))

# Try finding outliers
#print(X_raw[(X_raw['longitude'] > -1000)])

#print(X_raw[(X_raw['longitude'] > -1000) | (X_raw['mock1'] > 100) | (X_raw['mock2'] < 0) | (X_raw['mock2'] > 100) | (X_raw['quiz'] < 0) | (X_raw['quiz'] > 20)])

