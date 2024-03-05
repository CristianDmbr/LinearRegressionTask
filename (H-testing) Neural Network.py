import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler ,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier  

# Load dataset
testData = pd.read_csv('Databases/Titanic_coursework_entire_dataset_23-24.cvs.csv')

# Define features and target variable
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

# Slice the dataset
X_raw = testData[features]
y_raw = testData["Survival"]

# Split data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number)
categorical_features = X_raw.select_dtypes(exclude=np.number)

# Impute missing values for numerical features
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features.columns])
X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features.columns])

# Impute missing values for categorical features
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features.columns])
X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features.columns])

# Encode categorical features
encoder = OneHotEncoder()
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed).toarray()
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed).toarray()

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'logistic', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'max_iter': [10000] 
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train_scaled, y_train)

# Get best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Use the best parameters to train the model
best_mlp = MLPClassifier(**best_params)
best_mlp.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = best_mlp.predict(X_train_scaled)
y_pred_test = best_mlp.predict(X_test_scaled)


# Metrics
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred_test)
precision_train, recall_train, f1_score_train, support_train = precision_recall_fscore_support(y_train, y_pred_train)
precision_test_avg = sum(precision)/len(precision)
recall_test_avg = sum(recall)/len(recall)
f1_score_test_avg = sum(f1_score)/len(f1_score)
precision_train_avg = sum(precision_train)/len(precision_train)
recall_train_avg = sum(recall_train)/len(recall_train)
f1_score_train_avg = sum(f1_score_train)/len(f1_score_train)
precision_test, recall_test, f1_score_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average='binary')
precision_train, recall_train, f1_score_train, _ = precision_recall_fscore_support(y_train, y_pred_train, average='binary')

print("\nTraining Metrics:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Precision :", precision_train_avg)
print("Recall :", recall_train_avg)
print("F1-Score :", f1_score_train_avg)
print("Precision :", precision_train)
print("Recall :", recall_train)
print("F1-Score :", f1_score_train)

print("\nTesting Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Precision :", precision_test_avg)
print("Recall :", recall_test_avg)
print("F1-Score :", f1_score_test_avg)
print("Precision :", precision_test)
print("Recall :", recall_test)
print("F1-Score :", f1_score_test)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)
