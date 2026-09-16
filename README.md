# Machine-Learning-Case-Study

# REGRESSION: 10 ALGORITHM NOTEBOOKS

Dataset: airfoil_with_missing.csv
Target: Scaled_Sound_Pressure_Level_dB

10 notebooks:
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

Common pipeline:
- 80:20 train/test split, random_state=42
- missing-value audit and median imputation
- duplicate check
- IQR-based outlier treatment using training data only
- engineered Velocity_AOA_Interaction feature
- StandardScaler fitted on training data only
- R2, RMSE, MAE
- 5-fold CV R2
- predicted-vs-actual plot
- residual plot

Important:
The course guideline says the team's analysis/interpretation and feature-engineering decisions must be original. Replace the observation placeholders with your own findings before submission.


# CLASSIFICATION: 9 ALGORITHM NOTEBOOKS

Dataset: faults_with_missing(1).csv  
Target: Fault_Type

9 notebooks:
1. Dataset Audit
2. EDA and Visualisations
3. Preprocessing and Feature Engineering
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes
7. Decision Tree
8. Support Vector Classifier (SVC)
9. Final Classification Comparison

Common pipeline:
- Dataset audit
- Target distribution
- Feature distribution plots
- Correlation heatmap
- At least 2 feature-target scatter plots
- Missing-value audit and median imputation
- Duplicate check
- IQR-based outlier check and treatment using training data only
- Engineered geometric/shape features
- Stratified 80:20 train/test split, random_state=42
- Encoding where required
- StandardScaler fitted on training data only
- Accuracy
- Precision
- Recall
- Weighted F1
- Confusion matrix
- Classification report
- Decision-tree visualisation
- Decision-tree feature importance
- Final comparison table

Important:
The course guideline says the team's analysis/interpretation and feature-engineering decisions must be original. Replace or modify the observation placeholders with your own findings after running the notebooks before submission.
