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

# Your file is now in the Colab filesystem on the left
testData = pd.read_csv('housing_coursework_entire_dataset_23-24.csv') # Save it to a pandas dataframe

features = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
#features = ["longitude"]
X_raw = testData[features]
y_raw = testData['median_house_value']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

# To apply the different imputers, we first have to split our data into separate numerical and categorical data
X_train_num = X_train_raw.select_dtypes(include=np.number)
X_train_cat = X_train_raw.select_dtypes(exclude=np.number)

# Create our imputer objects
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Fit the imputers on the training data
numeric_imputer.fit(X_train_num)
categorical_imputer.fit(X_train_cat)

# Transform the columns
# Training
X_train_num_imp = numeric_imputer.transform(X_train_num)
X_train_cat_imp = categorical_imputer.transform(X_train_cat)

#We also need to split and transform our test data
X_test_num = X_test_raw.select_dtypes(include=np.number)
X_test_cat = X_test_raw.select_dtypes(exclude=np.number)
X_test_num_imp = numeric_imputer.transform(X_test_num)
X_test_cat_imp = categorical_imputer.transform(X_test_cat)

# Scaler Object
scaler = MinMaxScaler()
# Fit on the numeric training data
scaler.fit(X_train_num_imp)
# Transform the training and test data
X_train_num_sca = scaler.transform(X_train_num_imp)
X_test_num_sca = scaler.transform(X_test_num_imp)

# create the encoder object
encoder = OneHotEncoder(handle_unknown='ignore')
# Fit encoder on teh training data
encoder.fit(X_train_cat_imp)
# Transform the test and train data
X_train_onehot = encoder.transform(X_train_cat_imp).toarray()
X_test_onehot = encoder.transform(X_test_cat_imp).toarray()

print("X_train_num_sca shape:", X_train_num_sca.shape)
print("X_train_onehot shape:", X_train_onehot.shape)

X_train = np.concatenate([X_train_num_sca, X_train_onehot], axis=1)
X_test = np.concatenate([X_test_num_sca, X_test_onehot], axis=1)

# We can see the scaled test results and the OHE category columns now.
print(X_train)


# Keep only the 'ocean_proximity' feature for analysis
ocean_proximity_feature = ["ocean_proximity"]
X_raw_ocean_proximity = X_raw[ocean_proximity_feature]

# Split the data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_ocean_proximity, y_raw, test_size=0.20, shuffle=True, random_state=0)

# Create an imputer object for categorical data
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Fit and transform the columns
X_train_cat_imp = categorical_imputer.fit_transform(X_train_raw)
X_test_cat_imp = categorical_imputer.transform(X_test_raw)

# Encoder Object
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit encoder on the training data
encoder.fit(X_train_cat_imp)

# Transform the test and train data
X_train_onehot = encoder.transform(X_train_cat_imp).toarray()
X_test_onehot = encoder.transform(X_test_cat_imp).toarray()

# Concatenate the one-hot encoded features with numerical features
X_train = np.concatenate([X_train_onehot], axis=1)
X_test = np.concatenate([X_test_onehot], axis=1)

# Create linear regression object
obj = sklearn.linear_model.LinearRegression()

# Train the model using the training sets
obj.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = obj.predict(X_test)

# Plot outputs
plt.scatter(X_test_onehot[:, 0], y_test, color='black', label='y_test')  # Observed y values
plt.scatter(X_test_onehot[:, 0], y_pred, color='blue', label='y_pred')  # Predicted y values
plt.xlabel('Ocean Proximity')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

# The mean squared error loss and R2 for the test data
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_test, y_pred), sklearn.metrics.r2_score(y_test, y_pred)))
