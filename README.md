# Machine-Learning-Case-Study

## Contents

- [Regression](#regression-10-algorithm-notebooks)
- [Classification](#classification-10-algorithms)
- [Clustering](#clustering-2-algorithms)
- [Web App](#web-app)

---

## REGRESSION: 10 ALGORITHM NOTEBOOKS

- **Dataset:** `airfoil_with_missing.csv`
- **Target:** `Scaled_Sound_Pressure_Level_dB`

### 10 notebooks

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor
10. K-Nearest Neighbors Regressor

### Common pipeline

- 80:20 train/test split, `random_state=42`
- Missing-value audit and median imputation
- Duplicate check
- IQR-based outlier treatment using training data only
- Engineered `Velocity_AOA_Interaction` feature
- StandardScaler fitted on training data only
- R2, RMSE, MAE
- 5-fold CV R2
- Predicted-vs-actual plot
- Residual plot

> **Important:**
> The course guideline says the team's analysis/interpretation and feature-engineering decisions must be original. Replace the observation placeholders with your own findings before submission.

---

## CLASSIFICATION: 10 ALGORITHMS

- **Dataset:** `faults_with_missing(1).csv`
- **Target:** `Fault_Type`

### 9 notebooks

1. Dataset Audit
2. EDA and Visualisations
3. Preprocessing and Feature Engineering
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes
7. Decision Tree
8. Support Vector Classifier (SVC)
9. Final Classification Comparison

### 5 more algorithms added (in `ml.py` and the app)

1. Random Forest
2. AdaBoost
3. Gradient Boosting
4. Bagging (Decision Tree)
5. MLP Neural Network

### Common pipeline

- Dataset audit
- Target distribution
- Feature distribution plots
- Correlation heatmap
- At least 2 feature-target scatter plots
- Missing-value audit and median imputation
- Duplicate check
- IQR-based outlier check and treatment using training data only
- Engineered geometric/shape features
- Stratified 80:20 train/test split, `random_state=42`
- Encoding where required
- StandardScaler fitted on training data only
- Accuracy
- Precision
- Recall
- Weighted F1
- ROC-AUC
- Confusion matrix
- Classification report
- Decision-tree visualisation
- Decision-tree feature importance
- One final table comparing all 10 algorithms

### Final comparison table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6478 | 0.6972 | 0.6478 | 0.6403 | 0.8861 |
| KNN | 0.6967 | 0.7036 | 0.6967 | 0.6930 | 0.8858 |
| Gaussian Naive Bayes | 0.5913 | 0.6517 | 0.5913 | 0.5668 | 0.8775 |
| Decision Tree | 0.7044 | 0.7101 | 0.7044 | 0.7066 | 0.8029 |
| SVC | 0.7121 | 0.7437 | 0.7121 | 0.7110 | 0.9156 |
| Random Forest | 0.7584 | 0.7736 | 0.7584 | 0.7613 | 0.9329 |
| AdaBoost | 0.5501 | 0.6514 | 0.5501 | 0.5105 | 0.8147 |
| Gradient Boosting | 0.7661 | 0.7735 | 0.7661 | 0.7671 | 0.9318 |
| Bagging (Decision Tree) | 0.7995 | 0.8081 | 0.7995 | 0.8009 | 0.9355 |
| MLP Neural Network | 0.7249 | 0.7365 | 0.7249 | 0.7278 | 0.9024 |

### Plots for the 10 algorithms

- Bar chart and heatmap of all metrics
- Radar chart
- ROC curves
- Confusion matrix for each model

> **Important:**
> The course guideline says the team's analysis/interpretation and feature-engineering decisions must be original. Replace or modify the observation placeholders with your own findings after running the notebooks before submission.

---

## CLUSTERING: 2 ALGORITHMS

- **Dataset:** `marketing_campaign.csv`
- **Goal:** Group customers into segments

### 2 algorithms

1. K-Means
2. Hierarchical clustering (Ward)

### Common pipeline

- Missing-value handling (median for numbers, most frequent for categories)
- Outlier removal (Age above 100, Income above 200000)
- Engineered `Age`, `TotalSpend`, `TotalPurchases`, `HouseholdChildren` features
- One-hot encoding for categorical columns
- StandardScaler
- k from 2 to 10
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow curve for K-Means
- Dendrogram for Hierarchical clustering
- PCA 2D scatter plot with cluster colours for both algorithms
- t-SNE plot
- Segment summary table

> **Important:**
> The course guideline says the team's analysis/interpretation and feature-engineering decisions must be original. Replace or modify the observation placeholders with your own findings before submission.

---

## WEB APP

A Streamlit app for the classification and clustering parts.

### Files

| File | What it does |
|---|---|
| `app.py` | The screens |
| `ml.py` | Training and prediction |
| `data/` | The two csv files |

### Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

### What it does

- Predicts the fault of a steel plate with any of the 10 models
- Compares all 10 models in one table
- Groups customers with K-Means or Hierarchical clustering
- Places a new customer into a segment

Deployment steps are in `DEPLOY.md`.
