import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, make_scorer
from sklearn.metrics import precision_recall_fscore_support

# Load dataset
testData = pd.read_csv('Databases/Titanic_coursework_entire_dataset_23-24.cvs.csv')

# Define features and target variable
features = ["PassengerId","Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

# Slice the dataset
X_raw = testData[features]
y_raw = testData["Survival"]

X_raw = X_raw.drop(columns=["PassengerId"])

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number)
categorical_features = X_raw.select_dtypes(exclude=np.number)

# Impute missing values for numerical features
numeric_imputer = SimpleImputer(strategy="mean")
X_numerical_imputed = numeric_imputer.fit_transform(X_raw[numerical_features.columns])

# Impute missing values for categorical features
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_categorical_imputed = categoric_imputer.fit_transform(X_raw[categorical_features.columns])

# Encode categorical features
encoder = OneHotEncoder()
X_categorical_encoded = encoder.fit_transform(X_categorical_imputed).toarray()

# Concatenate numerical and encoded categorical features
X_encoded = np.concatenate((X_numerical_imputed, X_categorical_encoded), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split data into training and testing sets
X_train = X_scaled[:650]
y_train = y_raw[:650]

X_test = X_scaled[650:]
y_test = y_raw[650:]

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
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Metrics
print("\nTraining Metrics:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))

print("\nTesting Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
