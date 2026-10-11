# Customer Churn Prediction

Machine Learning project for customer segmentation and churn prediction using the Telco Customer Churn dataset.

The project covers:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Customer clustering
- Churn classification
- Class imbalance handling
- Cross-validation
- Hyperparameter tuning
- MLflow experiment tracking
- Streamlit application
- Docker deployment

---

## Project Objective

The objective is to analyze customer behavior and build a machine learning system capable of:

1. Segmenting customers into meaningful profiles using clustering.
2. Predicting whether a customer is likely to churn.
3. Estimating the probability of churn.
4. Providing predictions through a Streamlit web application.

---

## Project Structure

```text
customer-churn-ml/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── churn_model.joblib
│   └── cluster_model.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_cluster_feature.ipynb
│   ├── 05_model_optimization.ipynb
│   ├── 06_mlflow_tracking.ipynb
│   └── 07_deployment.ipynb
├── src/
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 1. Exploratory Data Analysis

The dataset contains 7043 customers.

The analysis included:

- Missing value detection
- Duplicate detection
- Outlier analysis
- Numerical distributions
- Categorical distributions
- Correlation analysis
- Churn distribution analysis

The target variable is imbalanced:

```text
No Churn  ≈ 73.5%
Churn     ≈ 26.5%
```

`TotalCharges` contained 11 empty values. These customers had a tenure of 0 months, so their `TotalCharges` values were replaced with 0.

---

## 2. Feature Engineering

A new feature called `NumServices` was created.

It represents the number of services used by a customer, including:

- Phone Service
- Multiple Lines
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

Adding `NumServices` produced a small improvement in classification performance.

---

## 3. Customer Clustering

K-Means clustering was used to segment customers.

The optimal number of clusters was selected using:

- Elbow Method
- Silhouette Score

The selected value was:

```text
K = 3
```

The three customer profiles were approximately:

| Cluster | Profile |
|---|---|
| 0 | Long-term / High-value customer |
| 1 | Basic / Low-cost customer |
| 2 | Newer / Higher-risk customer |

The clustering solution was also compared with Agglomerative Clustering.

K-Means produced better clustering metrics.

PCA was used to visualize the customer segments in two dimensions.

---

## 4. Churn Classification

Several classification algorithms were tested:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Gradient Boosting
- XGBoost

The models were evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Class imbalance was handled using:

- `class_weight="balanced"`
- SMOTE

SMOTE was applied only to the training data through an `imblearn` pipeline to prevent data leakage.

---

## Model Comparison

Some of the main baseline results were:

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.657 | 0.559 | 0.604 | 0.842 |
| Decision Tree | 0.485 | 0.511 | 0.497 | 0.657 |
| Random Forest | 0.615 | 0.492 | 0.547 | 0.816 |
| SVM | 0.640 | 0.484 | 0.551 | 0.793 |
| Gradient Boosting | 0.678 | 0.529 | 0.595 | 0.845 |
| XGBoost | 0.667 | 0.519 | 0.583 | 0.844 |
| Logistic + SMOTE | 0.508 | 0.797 | 0.620 | 0.840 |

---

## 5. Clustering + Classification

The K-Means cluster was added as an additional classification feature.

Performance was compared with and without the cluster feature.

For baseline Logistic Regression:

```text
Without Cluster:
F1 = 0.604
Recall = 0.559

With Cluster:
F1 = 0.608
Recall = 0.564
```

The improvement was very small.

For Logistic Regression + SMOTE, adding the cluster did not improve performance.

Therefore, clustering provides useful customer segmentation, but does not consistently improve churn prediction.

---

## 6. Model Optimization

5-fold stratified cross-validation was used.

Hyperparameters were optimized using `GridSearchCV`.

### Logistic Regression + SMOTE

Best parameters:

```text
C = 1
solver = lbfgs
```

Cross-validation results:

```text
Precision ≈ 0.528
Recall    ≈ 0.789
F1-score  ≈ 0.633
ROC-AUC   ≈ 0.845
```

Hyperparameter tuning did not significantly improve the model because the default configuration was already the best configuration tested.

### Gradient Boosting

Best parameters:

```text
learning_rate = 0.1
max_depth = 4
n_estimators = 100
```

The optimization produced only a very small improvement during cross-validation.

---

## Final Model

The final churn model selected for deployment is:

```text
Logistic Regression + SMOTE
```

Final test performance:

```text
Precision = 0.508
Recall    = 0.797
F1-score  = 0.620
ROC-AUC   = 0.840
```

This model was selected because identifying customers likely to churn is the main objective, making Recall particularly important.

The model detects approximately 80% of actual churners.

---

## 7. MLflow

MLflow was used to track machine learning experiments.

For each experiment, the following information was recorded:

- Model hyperparameters
- Preprocessing configuration
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Trained model

The main experiments compared were:

```text
Logistic Regression + SMOTE
Tuned Gradient Boosting
```

Logistic Regression + SMOTE was retained as the final model because of its higher Recall and F1-score.

---

## 8. Streamlit Application

The Streamlit application allows the user to enter customer information and displays:

- Customer cluster/profile
- Churn prediction
- Churn probability

Run the application locally:

```bash
streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

---

## 9. Docker

The application is containerized using Docker.

Build the image:

```bash
docker build -t customer-churn-app .
```

Run the container:

```bash
docker run --rm -p 8501:8501 customer-churn-app
```

Then open:

```text
http://localhost:8501
```

If port `8501` is already being used:

```bash
docker run --rm -p 8502:8501 customer-churn-app
```

Then open:

```text
http://localhost:8502
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:YounessAitJamaa/customer-churn-ml.git
cd customer-churn-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Notebook Execution Order

To reproduce the complete project, run the notebooks in this order:

```text
01_eda.ipynb
02_clustering.ipynb
03_classification.ipynb
04_cluster_feature.ipynb
05_model_optimization.ipynb
06_mlflow_tracking.ipynb
07_deployment.ipynb
```

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- imbalanced-learn
- MLflow
- Streamlit
- Docker
- Jupyter

---

## Conclusion

The project demonstrates a complete machine learning workflow from data exploration to deployment.

Customer segmentation identified three meaningful profiles, while several classification models were evaluated for churn prediction.

Logistic Regression combined with SMOTE provided the best balance for churn detection, reaching approximately 80% Recall.

The final solution is exposed through a Streamlit interface and can be deployed using Docker.