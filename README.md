Anomaly Detection in Energy Plant Sensors
📌 Overview

This project focuses on detecting anomalous behavior in energy plant sensor data using machine learning.
The goal is to classify whether a set of sensor readings corresponds to normal operation or an anomaly, based on historical multivariate time-series signals.

The solution applies extensive feature engineering, temporal analysis, and tree-based ensemble models such as LightGBM to achieve strong predictive performance on a highly imbalanced dataset.

🚀 Key Features

Temporal feature extraction (hour, day, weekday, month)

Rolling statistics for sensors (mean, std)

Missing-value handling and preprocessing

Stratified train-validation split

Gradient boosting models (LightGBM, CatBoost)

Hyperparameter tuning

Probability threshold optimization

Kaggle submission pipeline

Performance evaluation using Precision, Recall, and F1-score

🧠 Models Used

Logistic Regression (baseline)

Random Forest

KNN (tested, removed for scalability)

LightGBM (primary model)

CatBoost (secondary model)

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

LightGBM

CatBoost

Matplotlib / Seaborn

📊 Evaluation Metric

F1-Score (primary metric)

Accuracy

Precision & Recall

The dataset was heavily imbalanced, so emphasis was placed on recall-precision tradeoffs and threshold tuning.

📈 Workflow

Data loading & cleaning

Feature engineering

Temporal signal extraction

Rolling statistics

Train/validation split

Model training

Hyperparameter tuning

Threshold optimization

Test prediction generation

Submission file creation

📂 Repository Structure
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── anomaly_detection.ipynb
├── submission/
│   └── submission_v2.csv
├── README.md

🏆 Results

Achieved competitive leaderboard ranking in Kaggle competition.

Built scalable models suitable for large tabular sensor datasets.

Demonstrated full end-to-end ML pipeline from feature engineering to deployment-ready inference.

🔮 Future Improvements

Time-aware cross-validation

Leakage-safe rolling features

Ensemble blending

Advanced imbalance handling (SMOTE / focal loss)

SHAP-based interpretability
