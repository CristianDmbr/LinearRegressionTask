import numpy as np # A useful package for dealing with mathematical processes, we will be using it this week for vectors and matrices
import pandas as pd # A common package for viewing tabular data
import sklearn.linear_model, sklearn.datasets # We want to be able to access the sklearn datasets again, also we are using some model evaluation
from sklearn.preprocessing import StandardScaler, MinMaxScaler # We will be using the imbuilt sclaing functions sklearn provides
import matplotlib.pyplot as plt # We will be using Matplotlib for our graphs
from sklearn.preprocessing import PolynomialFeatures # A preprocessing function allowing us to include polynomial features into our model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # We will be using these to encode categorical features
from sklearn.model_selection import train_test_split # An sklearn library for outomatically splitting our data
from sklearn.impute import SimpleImputer # Performs basic imputations when doing preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.options.mode.chained_assignment = None  # default='warn'

testData = pd.read_csv('Databases/housing_coursework_entire_dataset_23-24.csv')

features = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
features = ["housing_median_age"]
X_raw = testData[features]
y_raw = testData["median_house_value"]

# Before Imputation : 
#print("Before Imputation : ")
# print(testData.select_dtypes(include=np.number).describe())
# print(testData.select_dtypes(exclude=np.number).describe())

# Spliting the database into 80:20 for training:testing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, train_size=0.80, shuffle=True, random_state=0)

# Seperate numerical and cetegorical features 
X_train_Number = X_train_raw.select_dtypes(include=np.number)
X_train_Categorical = X_train_raw.select_dtypes(exclude=np.number)

# Mean and most frequent imputers
numeric_Imputer = SimpleImputer(strategy="mean")
categorical_Imputer = SimpleImputer(strategy="most_frequent")

# Fit the imputers on the training data
numeric_Imputer.fit(X_train_Number)
# categorical_Imputer.fit(X_train_Categorical)

# Transform the new training variables
X_TrainingNumericalImputed = numeric_Imputer.transform(X_train_Number)
#    X_TrainingCategoricalImputed = categorical_Imputer.transform(X_train_Categorical)

# Convert the transformed numpy array back to a pandas DataFrame
#X_TrainingNumericalImputed = pd.DataFrame(X_TrainingNumericalImputed, columns=X_train_Number.columns)

# Create testing variables
X_test_numerical = X_test_raw.select_dtypes(include=np.number)
# X_test_categorical = X_test_raw.select_dtypes(exclude=np.number)

# Fill in the missing values :
X_test_numericalImputed = numeric_Imputer.transform(X_test_numerical)
# X_test_categoricalImputed = categorical_Imputer.transform(X_test_categorical)


###########################################################################
# Check database after imputation : 
#print("After Imputation : ")
# Concatenate training and testing numerical data along rows
#all_numerical_data = pd.concat([X_train_raw.select_dtypes(include=np.number), X_test_raw.select_dtypes(include=np.number)], axis=0)

# Impute missing values for all numerical data
#all_numerical_data_imputed = numeric_Imputer.transform(all_numerical_data)

# Convert the transformed numpy array back to a pandas DataFrame
#all_numerical_data_imputed_df = pd.DataFrame(all_numerical_data_imputed, columns=all_numerical_data.columns)

# Print the description of the whole dataset
#print(all_numerical_data_imputed_df.describe())
###########################################################################

# Initiate preprocessing scaler and model objectives
scaler = MinMaxScaler()
model = sklearn.linear_model.LinearRegression()

scaler.fit(X_TrainingNumericalImputed)

# Transform testing and training sets : 
X_train_NumericalScaled = scaler.transform(X_TrainingNumericalImputed)
X_test_NumericalScaled = scaler.transform(X_test_numericalImputed)

X_train = X_train_NumericalScaled
X_test = X_test_NumericalScaled

# Fit the X and Y training : 
model.fit(X_train, y_train)

# Make predictions of training
y_predictionsTrain = model.predict(X_train)

# Make predictions of testing
y_predictionTesting = model.predict(X_test)

###########################################################################
# Metrics : 

# Compute Mean Absolute Error (MAE)
mae_Testing = mean_absolute_error(y_test, y_predictionTesting)

# Compute Root Mean Squared Error (RMSE)
rmse_Testing = mean_squared_error(y_test, y_predictionTesting, squared=False)

# Define a function to compute Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Compute Mean Absolute Percentage Error (MAPE)
mape_Testing = mean_absolute_percentage_error(y_test, y_predictionTesting)

# Compute R-squared (R2) for testing data
r2_testing_Testing = r2_score(y_test, y_predictionTesting)

# Mean Absolute Error
print('Test - MAE: {:.4f}'.format(mae_Testing))

# Root Mean Squared Error 
print('Test - RMSE: {:.4f}'.format(rmse_Testing))

# R2 score
print('Test - R2 score: {:.4f}'.format(r2_testing_Testing))

# Mean Absolute Percentage Error 
print('Test - MAPE: {:.4f}'.format(mape_Testing))

# Mean Square Error 
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_train, y_predictionsTrain ),sklearn.metrics.r2_score(y_train, y_predictionsTrain)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_test, y_predictionTesting),sklearn.metrics.r2_score(y_test, y_predictionTesting)))

###########################################################################]
# Visualising : 

# A single column is used for plotting a 2D graph
X_disp = X_test[:,0]

# Plot outputs
plt.scatter(X_disp, y_test,  color='black', label='y_test') # Observed y values
plt.scatter(X_disp, y_predictionTesting, color='blue', label='y_pred') # predicted y values
plt.xlabel('Feature')
plt.ylabel('Final Grade')
plt.legend()
plt.show()