# Credit-Risk-Scoring-Model
This pipeline develops a credit risk scoring model to predict the likelihood of loan default. It uses historical loan data and incorporates techniques to handle imbalanced data and interpretability for financial applications.


Pipeline Steps:

Data Collection: Use a dataset like LendingClub or UCI’s German Credit Data.
Data Preprocessing: Clean missing values, encode categorical variables, and normalize numerical features.
EDA: Analyze correlations between credit score, income, debt-to-income ratio, and default rates.
Feature Engineering: Create features like debt-to-income ratio, payment history, and credit utilization.
Model Development: Train a gradient boosting model (LightGBM) for high performance and interpretability.
Model Evaluation: Use AUC-ROC, precision-recall, and confusion matrix for evaluation.
Model Deployment: Deploy as a Flask API with SHAP for feature importance explanations.
Monitoring: Log predictions and monitor feature drift.


Key Notes:

Dataset: LendingClub or UCI datasets provide rich features like credit score, loan amount, and repayment history. Synthetic data can be used for testing.
Interpretability: SHAP values explain feature contributions, critical for regulatory compliance in credit scoring.
Evaluation: AUC-ROC is suitable for binary classification, with confusion matrix providing insights into false positives/negatives.
Deployment: Flask API allows integration with loan processing systems, with SHAP for transparency.
