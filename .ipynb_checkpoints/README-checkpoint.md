# Student Dropout Prediction

## Project Overview

This project develops a machine learning system to predict student dropout risk using a higher-education student dataset.

The project demonstrates an end-to-end machine learning lifecycle covering:

- Problem understanding and framing
- Data collection and understanding
- Data preprocessing and exploratory data analysis
- Data leakage identification and removal
- Feature engineering
- Feature selection
- Model development and comparison
- Hyperparameter tuning
- Model explainability using SHAP
- Bias and fairness auditing
- Ethical AI considerations
- Reproducible model artefacts and prediction code

---

## Business Problem

Student dropout can have academic, financial, and operational consequences for educational institutions.

The objective of this project is to identify students who may be at higher risk of dropping out so that institutions can potentially provide appropriate early intervention and support.

The problem is framed as a **binary classification task**:

- `0` = Non-dropout
- `1` = Dropout

The modelling objective prioritises identifying students who are actually at risk of dropout while maintaining useful precision.

---

## Dataset

The project uses a higher-education student dataset containing:

- **4,424 student records**
- **36 original predictor variables**
- **1 original target variable**

The original target contains three outcomes:

- Dropout
- Enrolled
- Graduate

For this project, the target was converted into a binary classification problem:

- Dropout → `1`
- Enrolled or Graduate → `0`

The resulting class distribution was:

- Non-dropout: 3,003 students (67.88%)
- Dropout: 1,421 students (32.12%)

The dataset contains demographic, academic, socioeconomic, and enrolment-related information.

Examples include:

- Age at enrollment
- Gender
- Course
- Application mode
- Previous qualification
- Admission grade
- Tuition fees up to date
- Debtor status
- Scholarship holder
- Parental qualification
- Parental occupation

The dataset is stored in:

`Data/data.csv`

---

## Data Leakage Prevention

A key part of the preprocessing stage was identifying variables that would not be available at the time an early intervention decision is made.

The dataset contains academic-performance variables relating to student performance after enrolment, including curricular-unit results from the first and second semesters.

These post-enrolment variables were removed because using them for an early dropout-risk prediction system could introduce **data leakage**.

This resulted in a modelling dataset based on information that is more appropriate for an early intervention setting.

---

## Feature Engineering

Several domain-based features were created during preprocessing, including:

- `Age_Group`
- `Parental_Qualification_Same`
- `Parental_Occupation_Same`
- `Financial_Risk_Flag`

Categorical variables were encoded and numerical variables were processed through the modelling pipeline.

Feature selection was subsequently applied to identify the most informative modelling features.

The final modelling dataset contained **49 selected encoded features**.

---

## Models

Five supervised learning models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Support Vector Machine (SVM)

### Model Comparison

| Model | ROC-AUC | Dropout Recall | F1-score | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.826 | 0.500 | 0.606 | 0.768 | 0.791 |
| Logistic Regression | 0.818 | 0.486 | 0.603 | 0.793 | 0.794 |
| Random Forest | 0.816 | 0.697 | 0.648 | 0.606 | 0.757 |
| SVM | 0.814 | 0.669 | 0.637 | 0.607 | 0.755 |
| Decision Tree | 0.793 | 0.514 | 0.598 | 0.716 | 0.779 |

---

## Final Model

The **baseline Random Forest classifier** was retained as the final model.

Random Forest achieved the highest dropout recall and highest F1-score among the five evaluated models while maintaining competitive ROC-AUC.

### Final Test Performance

- ROC-AUC: **0.816**
- Dropout Recall: **69.7%**
- F1-score: **64.8%**
- Precision: **60.6%**
- Accuracy: **75.7%**

The held-out test set contained **885 students**, including **284 actual dropout students**.

At the selected classification threshold of 0.50:

- 198 of 284 actual dropout students were correctly identified.
- 327 students were predicted as potential dropouts.
- Dropout recall was 69.72%.

---

## Hyperparameter Tuning

Random Forest hyperparameter tuning was performed using `RandomizedSearchCV`.

The search evaluated parameters including:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `max_features`

The tuning process used 5-fold cross-validation with F1-score as the optimisation metric.

The selected tuned configuration was:

- `n_estimators = 300`
- `max_depth = 10`
- `min_samples_leaf = 10`
- `max_features = log2`

The tuned model achieved a cross-validation F1-score of approximately **0.623**.

On the held-out test set, the tuned model achieved:

- ROC-AUC: approximately 0.811
- Dropout Recall: approximately 0.697
- F1-score: approximately 0.644
- Precision: approximately 0.598
- Accuracy: approximately 0.753

The baseline Random Forest achieved a slightly higher held-out F1-score of approximately **0.648** and was therefore retained as the final model.

---

## Model Explainability

SHAP TreeExplainer was used to explain the final Random Forest model's predictions for the Dropout class.

The strongest predictors identified through mean absolute SHAP values included:

- Age at enrollment
- Tuition fees up to date
- Financial_Risk_Flag
- Scholarship holder
- Gender
- Application mode
- Course
- Age group
- Admission grade
- Previous qualification grade
- Debtor status

SHAP results describe patterns learned by the model. They should not be interpreted as evidence that these variables causally determine student dropout.

The SHAP feature importance results are stored in:

`Artifacts/shap_feature_importance.csv`

---

## Bias and Fairness Audit

Model outputs were evaluated across gender and age groups because these attributes may represent sensitive or potentially protected characteristics.

### Gender

The test set contained:

- Gender_1: 321 students
- Gender_0: 564 students

| Group | N | Actual Dropout Rate | Predicted Dropout Rate | Dropout Recall | False Positive Rate |
|---|---:|---:|---:|---:|---:|
| Gender_1 | 321 | 45.5% | 57.3% | 78.1% | 40.0% |
| Gender_0 | 564 | 24.5% | 25.4% | 60.9% | 13.9% |

Fairness metrics:

- Demographic Parity Difference: **0.320**
- Disparate Impact: **2.26**
- Equal Opportunity Difference: **0.172**

These results show substantial differences in model outcomes between the two gender groups.

However, the groups also have substantially different observed dropout rates. Therefore, these metrics demonstrate outcome disparity but do not, by themselves, establish that the model is introducing discriminatory bias.

---

## Age-Group Fairness Audit

The test set contained:

- Under 20: 412 students
- Middle/reference group: 323 students
- 30+: 150 students

| Age Group | N | Actual Dropout Rate | Predicted Dropout Rate | Dropout Recall | False Positive Rate |
|---|---:|---:|---:|---:|---:|
| Under 20 | 412 | 20.4% | 14.3% | 38.1% | 8.2% |
| Middle/reference | 323 | 35.0% | 44.6% | 74.3% | 28.6% |
| 30+ | 150 | 58.0% | 82.7% | 94.3% | 66.7% |

Relative to the middle/reference group:

| Comparison Group | Demographic Parity Difference | Disparate Impact | Equal Opportunity Difference |
|---|---:|---:|---:|
| Under 20 | -0.303 | 0.321 | -0.362 |
| 30+ | +0.381 | 1.854 | +0.199 |

The age-group audit shows substantial differences in prediction rates and performance.

The 30+ group, for example, has high dropout recall but also a high false-positive rate.

These results should be interpreted alongside the underlying differences in observed dropout rates.

The disparate-impact ratios provide useful fairness warning signals, but they should not be treated as a standalone determination of unfairness.

---

## Ethical AI and Limitations

Several limitations were identified during the project.

### Class Imbalance

The dropout class represents approximately **32.1%** of the dataset.

Potential mitigation strategies include:

- Class weighting
- Resampling
- Classification threshold analysis

### Fairness Disparities

Model outcomes differ across gender and age groups.

Potential mitigation strategies include:

- Regular subgroup monitoring
- Group-aware threshold analysis
- Reweighting
- Comparing alternative modelling approaches

### Data Leakage

Features used for prediction must be available at the time an intervention decision is made.

Potential mitigation strategies include:

- Temporal validation
- Documentation of feature availability
- Periodic review of the prediction pipeline

### Overfitting

Cross-validation performance differed from held-out test performance.

The cross-validation F1-score for the tuned Random Forest was approximately **0.623**, while the baseline Random Forest achieved approximately **0.648 F1** on the held-out test set.

Potential mitigation strategies include:

- Cross-validation
- Hold-out testing
- Ongoing monitoring
- Testing on future cohorts

### Sensitive Features

Gender and age-related variables contribute to model predictions.

Potential mitigation strategies include:

- Comparing performance with and without sensitive features
- Comparing fairness metrics across model versions
- Monitoring subgroup performance
- Reviewing how predictions are used in intervention decisions

---

## Repository Structure

```text
Student-Dropout-Prediction/
│
├── README.md
├── requirements.txt
│
├── Data/
│   └── data.csv
│
├── Notebooks/
│   ├── Capstone Step 2 (Coding File).ipynb
│   ├── Capstone Step 3 (Coding File).ipynb
│   ├── Capstone Step 4 (Coding File).ipynb
│   └── Capstone Step 5 (Coding File).ipynb
│
├── Models/
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   └── xgboost.joblib
│
├── Artifacts/
│   ├── age_group_fairness_metrics.csv
│   ├── age_group_fairness_results.csv
│   ├── final_model_config.json
│   ├── final_predictions.joblib
│   ├── gender_fairness_metrics.csv
│   ├── gender_fairness_results.csv
│   ├── model_comparison.csv
│   ├── preprocessor.joblib
│   ├── random_forest_feature_importance.csv
│   ├── random_forest_tuning.json
│   ├── selected_features.joblib
│   ├── shap_feature_importance.csv
│   ├── step3_config.json
│   ├── step4_results.json
│   ├── step5_limitations_mitigations.csv
│   ├── X_test_selected.joblib
│   ├── X_train_selected.joblib
│   ├── y_test.joblib
│   └── y_train.joblib
│
└── SRC/
    └── predict.py
```

---

## Reproducibility

The repository contains the main artefacts required to reproduce and inspect the project workflow, including:

- Jupyter notebooks documenting the analysis
- Preprocessing artefacts
- Selected feature information
- Trained model files
- Model comparison results
- Hyperparameter tuning results
- SHAP feature importance
- Fairness audit results
- Final prediction artefacts
- A reusable prediction script
- `requirements.txt` containing the main Python dependencies

---

## Project Workflow

```text
Problem Definition
        ↓
Data Understanding
        ↓
Data Leakage Assessment
        ↓
Preprocessing & EDA
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Model Development
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Model Selection
        ↓
SHAP Explainability
        ↓
Fairness & Bias Audit
        ↓
Reproducible Repository
```

---

## Project Status

Steps 1–5 of the machine learning lifecycle have been completed.

Step 6 covers final presentation and communication.

Step 7 focuses on packaging the project into a reproducible GitHub repository.

Optional deployment and MLOps components can be developed separately.
