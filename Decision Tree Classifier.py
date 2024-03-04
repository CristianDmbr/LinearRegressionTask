import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Load dataset
testData = pd.read_csv('Databases/Titanic_coursework_entire_dataset_23-24.cvs.csv')

# Define features and target variable
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
X_raw = testData[features]
y_raw = testData["Survival"]

# Split data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, train_size=0.80, shuffle=True, random_state=0)

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

# Decision tree classifier
clf = DecisionTreeClassifier(random_state=0, max_depth=3)  # Limiting the depth for better visualization
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

# Metrics
print("\nTraining Metrics:")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))

print("\nTesting Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Get feature names after one-hot encoding
encoded_feature_names = encoder.get_feature_names_out(input_features=categorical_features.columns)

# Concatenate numerical and encoded categorical feature names
all_feature_names = numerical_features.columns.tolist() + encoded_feature_names.tolist()

# Visualize decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf, feature_names=all_feature_names, class_names=["Not Survived", "Survived"], filled=True)
plt.show()

