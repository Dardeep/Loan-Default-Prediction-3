# Loan Default Prediction Project

This project focuses on building a robust predictive model to identify loan defaults. Using Gradient Boosting as the final model, the goal was to maximize the recall for defaults (Class 1) while maintaining a balance with precision and overall model performance. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, and hyperparameter tuning.

---

## **Project Structure**

- **Data Preprocessing**: 
  - Handled missing values.
  - Transformed skewed numerical features using log and square root transformations.
  - Scaled features to standardize values.

- **Exploratory Data Analysis (EDA)**: 
  - Performed univariate and bivariate analysis to understand feature distributions and relationships with the target variable.
  - Visualized correlations and feature distributions to guide feature engineering.

- **Feature Engineering and Challenges**:

### **Challenges Faced**:
1. **Class Imbalance**:
   - The dataset had a significant imbalance between default (Class 1) and non-default (Class 0) loans, making it difficult for models to learn effectively about defaults.
   - This imbalance led to high accuracy for non-default predictions but poor recall for defaults, which was the primary business objective.

2. **Skewed Numerical Features**:
   - Features like `person_income` and `loan_amnt` were highly skewed, which could bias the model and reduce its predictive power.

3. **Redundant and Correlated Features**:
   - Some features, such as `person_income` and its log-transformed version, were highly correlated and needed careful selection to avoid redundancy.

4. **Categorical Data**:
   - Features like `loan_intent` and `loan_grade` required encoding for compatibility with machine learning algorithms.

### **Steps Taken to Address These Challenges**:
1. **Class Imbalance**:
   - Applied **Synthetic Minority Oversampling Technique (SMOTE)** to generate synthetic examples for the minority class.
   - Experimented with **undersampling** the majority class and using **class weights** to penalize misclassification of defaults.

2. **Skewness Reduction**:
   - Applied **log transformations** to highly skewed features such as `person_income` and `loan_amnt`.
   - Used **square root transformations** for other skewed features like `person_emp_length` to stabilize variance.

3. **Feature Selection**:
   - Retained only the most informative features, removing redundant ones (e.g., keeping either scaled or log-transformed versions).
   - Created new features like `debt_to_income` to better capture the financial burden on borrowers.

4. **Encoding Categorical Data**:
   - One-hot encoded features such as `loan_intent` and `loan_grade` for seamless integration into machine learning models.

### **Feature Engineering**
- Log and square root transformations were applied to reduce skewness in numerical features like `person_income` and `loan_amnt`.
- New features such as `debt_to_income` (ratio of loan amount to income) were created to improve predictive power.
- Categorical variables like `loan_intent` and `loan_grade` were one-hot encoded for compatibility with machine learning models.

- **Modeling**:
  - Built and evaluated Logistic Regression and Gradient Boosting models.
  - Addressed class imbalance using SMOTE, undersampling, and class weights.
  - Fine-tuned hyperparameters for both models using GridSearchCV.
  - Created **two datasets**:
    - **Logistic Regression Dataset**: Contained scaled features suitable for linear models.
    - **Gradient Boosting Dataset**: Contained log-transformed features to handle non-linear relationships.

- **Threshold Adjustment**:
  - Adjusted the decision threshold for Gradient Boosting to maximize recall while maintaining precision.

---

## **Final Model**

### **Gradient Boosting Model with Threshold 0.3**

#### **Performance Metrics**:
- **Recall (Class 1)**: 
  - Improved to **81%**, ensuring a higher proportion of actual defaults are correctly identified, aligning with the business goal.

- **Precision (Class 1)**:
  - At **81%**, providing a good balance between identifying defaults and minimizing false positives.

- **AUC-ROC**:
  - **0.945**, confirming excellent model capability in distinguishing between defaults and non-defaults.

#### **Best Hyperparameters**:
- `n_estimators`: **300**
- `learning_rate`: **0.1**
- `max_depth`: **7**
- `subsample`: **0.8**

---

## **Dataset Information**

### **Shape and Columns**
The dataset contains **29,459 rows** and **12 columns** after preprocessing and cleaning. The original features included:

1. **person_age**: Age of the individual.
2. **person_income**: Annual income of the individual (USD).
3. **person_home_ownership**: Type of home ownership (RENT, OWN, MORTGAGE, OTHER).
4. **person_emp_length**: Employment length in years.
5. **loan_intent**: Purpose of the loan (EDUCATION, MEDICAL, VENTURE, etc.).
6. **loan_grade**: Grade assigned to the loan (A to G).
7. **loan_amnt**: Loan amount requested (USD).
8. **loan_int_rate**: Interest rate on the loan.
9. **loan_status**: Target variable indicating default status (1 = Default, 0 = Non-default).
10. **loan_percent_income**: Loan amount as a percentage of income.
11. **cb_person_default_on_file**: Credit bureau default flag (Y/N).
12. **cb_person_cred_hist_length**: Credit history length in years.

---

## **Files**
- **`log_features_dataset.csv`**: Dataset with log-transformed features for Gradient Boosting.
- **`scaled_features_dataset.csv`**: Dataset with scaled features for Logistic Regression.
- **`final_gradient_boosting_model.pkl`**: Serialized Gradient Boosting model.
- **`final_threshold.txt`**: Final decision threshold (0.3) for predictions.

---

## **Installation and Usage**

### **Requirements**
- Python 3.7+
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imblearn`
  - `matplotlib`
  - `joblib`

### **Setup**
1. Clone this repository.
   ```bash
   git clone <repository_url>

