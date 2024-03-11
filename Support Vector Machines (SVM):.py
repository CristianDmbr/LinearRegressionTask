import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler ,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

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
svm_classifier = SVC(kernel='linear', random_state=0, C = 0.1, gamma = 0.001)
svm_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = svm_classifier.predict(X_train_scaled)
y_pred_test = svm_classifier.predict(X_test_scaled)

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

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()