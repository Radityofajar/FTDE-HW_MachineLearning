# Machine Learning Project for Insurance Prediction

This repository contains the implementation of a machine learning pipeline for predicting insurance responses using various machine learning models. The project consists of two main steps:
1. Data Preprocessing
2. Model Training

## 1. Data Preprocessing

The first notebook (`1_dataPreprocessing.ipynb`) is focused on exploring and preparing the data before it is used for model training. The main steps include:

- **Data Exploration:**
    - No missing data found.
    - There is a wide range in the data values (e.g., Annual Premium ranges from 2640 to 540165 with an average of 30564).
    - The target variable (`Response`) is highly imbalanced with only 12.25% of the responses being positive (1).
    - Categorical columns such as `Gender`, `Vehicle_Age`, and `Vehicle_Damage` need to be encoded.
    - No clear correlations found with the target variable.
  
- **Imbalanced Data Handling:**
    - Given the significant imbalance in the dataset, techniques such as SMOTE are applied to balance the dataset.

## 2. Model Training

The second notebook (`2_ModelTraining.ipynb`) focuses on training and evaluating various machine learning models, including:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Support Vector Machine (SVM)**
- **Sklearn MLP (Multilayer Perceptron)**

### Results & Conclusions

- **XGBoost** showed the best performance across different metrics, and it was chosen as the final model for prediction.
- While the accuracy of the model remained high even after applying SMOTE to handle imbalance, accuracy alone is not a reliable metric in imbalanced datasets.
  
### Key Observations:

1. **Accuracy** is still high on the test data, meaning the model predicts most examples correctly. However, due to class imbalance, accuracy can be misleading.
  
2. **Recall** remains high, indicating that the model successfully identifies most of the true positive cases (i.e., the model focuses on reducing false negatives).
  
3. **Precision** drops significantly, meaning that many of the model’s positive predictions are false positives. This reduces trust in the positive predictions.

### Issues:

- **Overfitting**: The model might be overfitting to the training data, leading to a mismatch in performance between the training and test datasets.
- **SMOTE Effect**: The use of SMOTE for generating synthetic data can alter the data distribution, causing the model to perform poorly when exposed to real-world (test) data.

## How to Run

To run this project locally, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/Radityofajar/FTDE-HW_MachineLearning.git
