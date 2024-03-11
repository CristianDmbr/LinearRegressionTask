import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, make_scorer
from sklearn.metrics import precision_recall_fscore_support

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

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4, 5],
    'ccp_alpha': [0, 0.1, 0.01, 0.001]
}

# Create the grid search object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                           param_grid=param_grid,
                           cv=5,
                           scoring={'accuracy': make_scorer(accuracy_score)},
                           refit='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Predictions
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

# Metrics
print("\nTraining Metrics:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))

print("\nTesting Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
