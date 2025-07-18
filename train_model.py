import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import pandas as pd

# Define correct column names
col_names = [
    'Status_checking_account', 'Duration', 'Credit_history', 'Purpose',
    'Credit_amount', 'Savings_account', 'Employment', 'Installment_rate',
    'Personal_status', 'Other_debtors', 'Residence_since', 'Property',
    'Age', 'Other_installment_plans', 'Housing', 'Number_existing_credits',
    'Job', 'Number_dependents', 'Telephone', 'Foreign_worker', 'target'
]

# Read with column names
df = pd.read_csv("german.data", sep=" ", header=None, names=col_names)

# Target: 1 = good, 2 = bad → map to binary
df['target'] = df['target'].map({1:0, 2:1})


# 2. Preprocessing: One-hot encode categoricals
X = pd.get_dummies(df.drop('target', axis=1), drop_first=True)
y = df['target']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Train baseline model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Baseline Results:")
print(classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

# 5. Hyperparameter Tuning - GridSearchCV
param_grid = {'n_estimators':[100,200], 'max_depth':[None,10,20], 'min_samples_split':[2,5]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("GridSearch Best:", grid.best_params_)
best = grid.best_estimator_
y_pred_best = best.predict(X_test)

print("Tuned Model Results:")
print(classification_report(y_test, y_pred_best, target_names=['Good', 'Bad']))

# 6. (Optional) RandomizedSearchCV
param_dist = {'n_estimators': range(100,301,50), 'max_depth':[None,10,20,30], 'min_samples_split':[2,5,10]}
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=5, scoring='f1', random_state=42, n_jobs=-1)
rand.fit(X_train, y_train)
print("RandomSearch Best:", rand.best_params_)

# 7. Save best model
joblib.dump(best, "credit_rf_model.pkl")
print("Model saved.")

# Save model
joblib.dump(best, "credit_rf_model.pkl")

# Save feature names used in training
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("Model & features saved.")
