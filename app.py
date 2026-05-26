from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset using scikit-learn
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target
print(df.head())

# Splitting dataDrame as features and target
X = df.drop(columns=['target'])
y = df['target']

# Removing redundant columns
corr_matrix = X.corr()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > 0.90)]
X_reduced = X.drop(columns=columns_to_drop)
print(X_reduced.head())

# Split into training and testing sets to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Applying standardization to the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using Linear SVM
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)

accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Linear SVM Accuracy: {accuracy_linear * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_linear, target_names=cancer_data.target_names))

# Using rbf SVM
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"rbf SVM Accuracy: {accuracy_rbf * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_rbf, target_names=cancer_data.target_names))

