import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler ,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV

testData = pd.read_csv('Databases/Titanic_coursework_entire_dataset_23-24.cvs.csv')

features = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Embarked"]
X_raw = testData[features]
y_raw = testData["Survival"]

X_raw = X_raw.drop(columns=["PassengerId"])
X_raw = X_raw.drop(columns=["Name"])

# Split data into 650 training sets and 240 testing sets
X_train_raw = X_raw[:650]
y_train = y_raw[:650]

X_test_raw = X_raw[650:]
y_test = y_raw[650:]

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number)
categorical_features = X_raw.select_dtypes(exclude=np.number)

# Numerical imputation that uses the mean of the database to fill in missing values
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features.columns])
X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features.columns])

# Impute missing values for categorical features with the use of the most frequent used string
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features.columns])
X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features.columns])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')  # Ignore unknown categories
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()
# For test data, use the categories learned from the training data
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed).toarray()

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# SVM Classifier
svm_classifier = SVC(kernel='rbf', random_state=0)
svm_classifier.fit(X_train_scaled, y_train)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']  # Kernel type
}

# Instantiate the SVM classifier
svm_classifier = SVC(random_state=0)

# Perform grid search
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Predictions with the best model
best_svm_classifier = grid_search.best_estimator_
y_pred_train = best_svm_classifier.predict(X_train_scaled)
y_pred_test = best_svm_classifier.predict(X_test_scaled)

# Metrics
print("\nTraining Metrics:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Precision:", precision_score(y_train, y_pred_train))
print("Recall:", recall_score(y_train, y_pred_train))
print("F1-Score:", f1_score(y_train, y_pred_train))

print("\nTesting Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1-Score:", f1_score(y_test, y_pred_test))