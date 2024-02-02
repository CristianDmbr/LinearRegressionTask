import numpy as np 
import pandas as pd 
import sklearn.linear_model, sklearn.datasets 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error, max_error, median_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer 
pd.options.mode.chained_assignment = None

testData = pd.read_csv('housing_coursework_entire_dataset_23-24.csv') 

# Set the features for ML to learn from.
features = ["No.","longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
# Target of the model is the Median House Price
target = ["median_house_value"]

# Sets the X axis / independent variables to be the features.
X_raw = testData[features]
#Remove the No. of the record because it's a irrelevant data feature training.
X_raw = X_raw.drop(columns=['No.'])

# Sets the Y axis / dependent variable to be the median_house_value
y_raw = testData[target]

# Used to meaasure the stats of the databases
# Numerical features stats
#print(X_raw.select_dtypes(include=np.number).describe())
# Categorical / String feature stats
#print(X_raw.select_dtypes(exclude=np.number).describe())

print(pd.unique(X_raw['ocean_proximity']))

# From description of numerical data features total_bedrooms is missing 9 records.
# Filling in the missing values for the total_bedrooms feature is done by using the encoded mean of the collumn.
mean_total_bedrooms = X_raw["total_bedrooms"].mean()

# Replace the missing values with the mean.
X_raw["total_bedrooms"].fillna(mean_total_bedrooms, inplace=True)
# Verify that there are no more missing values.
#print(X_raw[X_raw["total_bedrooms"].isnull()])

# The varible that the model is testing with the target.
features = ['ocean_proximity']
X_raw_features = X_raw[features]

# Preprocessing.

# Separates the databse into 80% and 20% for training and testing.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_features, y_raw, train_size=0.80, test_size=0.20, shuffle=True, random_state=0)
X_train_num = X_train_raw.select_dtypes(include=np.number)

# Create our imputer objects.
numeric_imputer = SimpleImputer(strategy='mean')

# Fit the imputers on the training data.
numeric_imputer.fit(X_train_num)

# Training.
X_train_num_imp = numeric_imputer.transform(X_train_num)

# Split and transform our test data.
X_test_num = X_test_raw.select_dtypes(include=np.number)
X_test_num_imp = numeric_imputer.transform(X_test_num)

# Feature scalling
# Scaler Object.
scaler = MinMaxScaler()
# Fit on the numeric training data.
scaler.fit(X_train_num_imp)
# Transform the training and test data.
X_train_num_sca = scaler.transform(X_train_num_imp)
X_test_num_sca = scaler.transform(X_test_num_imp)

X_train = X_train_num_sca
X_test = X_test_num_sca

# Create linear regression object.
obj = sklearn.linear_model.LinearRegression()

# Train the model using the training sets.
obj.fit(X_train, y_train)

# We can make a prediction with the training data
y_pred_train = obj.predict(X_train)
# Remember the predictions with the new data give a better indiction of the true model performance.
# Make predictions using the testing set.
y_pred = obj.predict(X_test)

# Choosen a single column of the feature matrix so we can plot a 2D scatter plot.
X_disp = X_test[:,0] 

# Plot outputs
plt.scatter(X_disp, y_test,  color='black', label='y_test')
plt.scatter(X_disp, y_pred, color='blue', label='y_pred') 
plt.xlabel('Ocean Proximity')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

# Regression metrics for both training and testing sets
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
max_err_train = max_error(y_train, y_pred_train)
medae_train = median_absolute_error(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
max_err_test = max_error(y_test, y_pred)
medae_test = median_absolute_error(y_test, y_pred)

# Prints the regression metrics for both training and testing sets
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))
print('Train - MAE: {:.4f}'.format(mae_train))
print('Test - MAE: {:.4f}'.format(mae_test))
print('Train - RMSE: {:.4f}'.format(rmse_train))
print('Test - RMSE: {:.4f}'.format(rmse_test))
print('Train - Max Error: {:.4f}'.format(max_err_train))
print('Test - Max Error: {:.4f}'.format(max_err_test))
print('Train - Median Absolute Error: {:.4f}'.format(medae_train))
print('Test - Median Absolute Error: {:.4f}'.format(medae_test))