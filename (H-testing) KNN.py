import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, precision_recall_fscore_support

testData = pd.read_csv('Databases/Titanic_coursework_entire_dataset_23-24.cvs.csv')

features = ["PassengerId","Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
X_raw = testData[features]
y_raw = testData["Survival"]

X_raw = X_raw.drop(columns=["PassengerId"])

# Split data into 650 training sets and 240 testing sets
X_train_raw = X_raw[:650]
y_train = y_raw[:650]

X_test_raw = X_raw[650:]
y_test = y_raw[650:]

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

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute nearest neighbors
    'leaf_size': [20, 30, 40],  # Leaf size for tree-based algorithms
    'p': [1, 2],  # Power parameter for the Minkowski metric
    'metric': ['euclidean', 'manhattan']  # Distance metric
}

# Instantiate KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_knn = grid_search.best_estimator_

# Predictions
y_pred_train = best_knn.predict(X_train_scaled)
y_pred_test = best_knn.predict(X_test_scaled)

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

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# K Nearest Neighbors Classifier with PCA-transformed features
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

# Plotting decision boundaries using PCA-transformed features
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure()
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.4)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=20, edgecolor='k')
plt.title('K Nearest Neighbors Decision Boundaries (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
