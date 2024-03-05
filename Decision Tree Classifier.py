import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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

# Numerical imputation that uses the mean of the database to fill in missing values
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features.columns])
X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features.columns])

# Impute missing values for categorical features with the use of the most frequent used string
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
# Best Hyperparameters: {'ccp_alpha': 0, 'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 2}
clf = DecisionTreeClassifier(random_state=0, 
                            max_depth=7, 
                            min_samples_split=2, 
                            min_samples_leaf=4, 
                            ccp_alpha=0)  
clf.fit(X_train_scaled, y_train)

clf.fit(X_train_scaled, y_train)

# Prediction variables
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

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

#Visualisation
encoded_feature_names = encoder.get_feature_names_out(input_features=categorical_features.columns)
all_feature_names = numerical_features.columns.tolist() + encoded_feature_names.tolist()

plt.figure(figsize=(15,10))
tree.plot_tree(clf, feature_names=all_feature_names, class_names=["Not Survived", "Survived"], filled=True)
plt.show()