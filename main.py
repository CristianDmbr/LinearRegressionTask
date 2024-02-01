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

#print(X_raw.select_dtypes(include=np.number).describe())
#print(X_raw.select_dtypes(exclude=np.number).describe())

# From description of numerical data features total_bedrooms is missing 9 records.
# Filling in the missing values for the total_bedrooms feature by using the encoded mean : 
# Find the mean of the total_bedrooms collumn
mean_total_bedrooms = X_raw["total_bedrooms"].mean()
# Impute missing values with the mean
X_raw["total_bedrooms"].fillna(mean_total_bedrooms, inplace=True)
# Verify that there are no more missing values
#print(X_raw[X_raw["total_bedrooms"].isnull()])





# Select Columns - Remember the full list of features are:
# ['mock1', 'mock2','quiz','studyTime','travelTime','absence','school']
features = ['longitude']
X_raw_features = X_raw[features]

###############
# Preprocessing
###############
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_features, y_raw, train_size=0.80, test_size=0.20, shuffle=True, random_state=0)

# Here we only have numerical data, so could skip this step.
X_train_num = X_train_raw.select_dtypes(include=np.number)

# Create our imputer objects
numeric_imputer = SimpleImputer(strategy='mean')

# Fit the imputers on the training data
numeric_imputer.fit(X_train_num)

# Transform the columns
# Training
X_train_num_imp = numeric_imputer.transform(X_train_num)

#We also need to split and transform our test data
X_test_num = X_test_raw.select_dtypes(include=np.number)
X_test_num_imp = numeric_imputer.transform(X_test_num)

# Scaler Object
scaler = MinMaxScaler()
# Fit on the numeric training data
scaler.fit(X_train_num_imp)
# Transform the training and test data
X_train_num_sca = scaler.transform(X_train_num_imp)
X_test_num_sca = scaler.transform(X_test_num_imp)

X_train = X_train_num_sca
X_test = X_test_num_sca

###################
# End preprocessing
###################

# Create linear regression object
obj = sklearn.linear_model.LinearRegression()

# Train the model using the training sets
obj.fit(X_train, y_train)

# We can make a prediction with the training data
y_pred_train = obj.predict(X_train)
# Remember the predictions with the new data give a better indiction of the true model performance.
# Make predictions using the testing set
y_pred = obj.predict(X_test)

# I decided that for visualisation i wanted to use mock1.
X_disp = X_test[:,0] # We have to choose a single column of the feature matrix so we can plot a 2D scatter plot.

# Plot outputs
plt.scatter(X_disp, y_test,  color='black', label='y_test') # Observed y values
plt.scatter(X_disp, y_pred, color='blue', label='y_pred') # predicted y values
plt.xlabel('Longitude')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

# The mean squared error loss and R2 for the test and train data
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_train, y_pred_train),sklearn.metrics.r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_test, y_pred),sklearn.metrics.r2_score(y_test, y_pred)))