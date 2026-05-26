# Breast Cancer Diagnosis System Using Support Vector Machines

A robust Python machine learning project designed to ingest cellular nuclei measurements, dynamically filter highly correlated redundant features, and predict the presence of malignant or benign breast tumors using a Linear Support Vector Machine (SVM). This project emphasizes industry best practices in distance-based algorithm scaling and dimensional reduction.

## 💡 Overview

Predictive analytics in oncology requires mathematically sound preprocessing and feature selection. This project automates a complete workflow for an SVM classifier. It loads the Breast Cancer Wisconsin dataset, automatically calculates a correlation matrix to drop redundant dimensions, strictly scales the continuous data to prevent distance bias, and outputs highly reliable diagnostic predictions. By eliminating multicollinearity, the model is simplified, optimized for speed, and less prone to overfitting.

## ✨ Features

* **Automated Data Acquisition:** Seamlessly loads the verified Breast Cancer Wisconsin dataset directly from `scikit-learn`'s native datasets, requiring no external file management.
* **Algorithmic Feature Selection:** Dynamically calculates a correlation matrix and automatically drops features that share a correlation higher than 0.90 (e.g., dropping `mean perimeter` if `mean radius` is already present).
* **Distance-Bias Prevention:** Utilizes `scikit-learn`'s `StandardScaler` to normalize the vast ranges in cellular measurements (e.g., area vs. smoothness) so the SVM does not mathematically favor larger numbers.
* **Leak-Free Architecture:** Enforces a strict separation of the dataset into training and testing sets *before* any statistical scaling is applied, preventing future data from "leaking" into the model's training phase.
* **Objective Evaluation:** Generates a comprehensive clinical assessment of the model, outputting the exact Accuracy Score and a detailed Classification Report mapping Precision, Recall, and F1-Scores for both malignant and benign classes.

## 🛠️ Prerequisites

* Python 3.8 or higher
* A standard Python IDE (VS Code, PyCharm) or Jupyter Notebook
* Core Scientific Libraries: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

## 🚀 Usage

1. Clone this repository to your local machine.
2. Open your terminal or command prompt and install the necessary dependencies:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```
3. Launch your Python environment or execute the script directly:
   ```bash
   python app.py
   ```

## 📊 Expected Output

Upon successful execution, the script will process the raw cellular data in memory and output fact-grounded terminal metrics, including:

1. **Dimensionality Diagnostics:** Terminal output showing the number of features automatically dropped due to multicollinearity, alongside the `head()` of the finalized, reduced DataFrame.

2. **Model Evaluation Metrics:** Terminal output stating the exact calculated Accuracy Score (which consistently achieves 95%+ with this optimized setup).

3. **Classification Report:** A structured breakdown displaying True Positives/Negatives via Recall and Precision metrics, demonstrating the algorithm's clinical viability in identifying both class 0 (malignant) and class 1 (benign) tumors accurately.

## 🧩 How It Works (Under the Hood)

This script serves as a practical application of standard machine learning pipelines tailored for Support Vector Machines:

1. **Data Ingestion:** The script loads the dataset utilizing `load_breast_cancer()`. It converts the internal `Bunch` object into a workable `pandas` DataFrame for manipulation.

2. **Multicollinearity Filtering:** The algorithm generates a pairwise correlation matrix of all 30 cellular features. Looking at the upper triangle of the matrix, it isolates and drops any secondary feature that correlates at 0.90 or higher with a primary feature. 

3. **Train/Test Splitting:** The algorithm securely splits the data into training (80%) and testing (20%) subsets before any scaling occurs to preserve the integrity of the unseen test environment.

4. **Standardization:** Continuous clinical features are routed through a `StandardScaler` using `fit_transform` on the training data and `transform` on the testing data. This brings all features to a mean of 0 and standard deviation of 1.

5. **Linear SVM Classification:** The mathematically normalized data flows into an `SVC` (Support Vector Classifier) with `kernel='linear'`. The model draws a multi-dimensional hyperplane to maximize the margin between benign and malignant instances.

6. **Non-Linear Pattern Recognition:** While the baseline model utilizes a linear hyperplane, the architecture seamlessly supports the Radial Basis Function (RBF) kernel. This advanced configuration mathematically projects the cellular features into a higher-dimensional space, allowing the algorithm to draw complex, non-linear boundaries and capture intricate relationships.