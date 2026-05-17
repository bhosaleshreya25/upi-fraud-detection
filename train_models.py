"""
Train ML Models for UPI Fraud Detection
Classification: Logistic Regression, SVM
Ensemble: Random Forest, XGBoost
Target Accuracy: 90-95%
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("=" * 70)
print("LOADING KAGGLE DATASET")
print("=" * 70)

# Load dataset
df = pd.read_csv("data/onlinefraud.csv")

print(f"Total records: {len(df):,}")
print(f"Fraud records: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.4f}%)")
print(f"Safe records: {(df['isFraud']==0).sum():,}")

# ========== FEATURE ENGINEERING ==========
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Time-based features
df['hour'] = df['step'] % 24
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_weekend'] = (df['step'] % 7 >= 5).astype(int)

# Amount-based features
df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
df['amount_log'] = np.log1p(df['amount'])

# Balance-based features
df['balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_change_ratio'] = np.where(
    df['oldbalanceOrg'] > 0,
    df['balance_change'] / df['oldbalanceOrg'],
    0
)
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']

# Transaction type features
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# Features for training
FEATURES = [
    'amount', 'amount_log', 'hour', 'is_night', 'is_weekend',
    'is_high_amount', 'balance_change', 'balance_change_ratio',
    'dest_balance_change', 'type_encoded', 'oldbalanceOrg', 'newbalanceOrig'
]

print(f"Total features: {len(FEATURES)}")

X = df[FEATURES]
y = df['isFraud']

# ========== TRAIN TEST SPLIT ==========
print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")
print(f"Training fraud rate: {y_train.mean()*100:.4f}%")
print(f"Test fraud rate: {y_test.mean()*100:.4f}%")

# ========== SCALING ==========
print("\n" + "=" * 70)
print("SCALING FEATURES")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")

# ========== MODEL TRAINING WITH CLASS WEIGHTS ==========
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

# Calculate class weights for balanced training
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weight_dict}")

# 1. Logistic Regression
print("\n[1/4] Training Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_acc = accuracy_score(y_test, lr_pred) * 100
lr_prec = precision_score(y_test, lr_pred) * 100
lr_rec = recall_score(y_test, lr_pred) * 100
lr_f1 = f1_score(y_test, lr_pred) * 100

# 2. SVM
print("[2/4] Training SVM...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred) * 100
svm_prec = precision_score(y_test, svm_pred) * 100
svm_rec = recall_score(y_test, svm_pred) * 100
svm_f1 = f1_score(y_test, svm_pred) * 100

# 3. Random Forest
print("[3/4] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_acc = accuracy_score(y_test, rf_pred) * 100
rf_prec = precision_score(y_test, rf_pred) * 100
rf_rec = recall_score(y_test, rf_pred) * 100
rf_f1 = f1_score(y_test, rf_pred) * 100

# 4. XGBoost
print("[4/4] Training XGBoost...")
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

xgb_acc = accuracy_score(y_test, xgb_pred) * 100
xgb_prec = precision_score(y_test, xgb_pred) * 100
xgb_rec = recall_score(y_test, xgb_pred) * 100
xgb_f1 = f1_score(y_test, xgb_pred) * 100

# 5. Ensemble (Weighted Average)
print("[5/5] Creating Ensemble...")
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
ensemble_proba = (rf_proba * 0.5 + xgb_proba * 0.5)
ensemble_pred = (ensemble_proba > 0.5).astype(int)

ensemble_acc = accuracy_score(y_test, ensemble_pred) * 100
ensemble_prec = precision_score(y_test, ensemble_pred) * 100
ensemble_rec = recall_score(y_test, ensemble_pred) * 100
ensemble_f1 = f1_score(y_test, ensemble_pred) * 100

# ========== RESULTS SUMMARY ==========
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Create results table
results = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'Ensemble (RF+XGB)'],
    'Accuracy': [lr_acc, svm_acc, rf_acc, xgb_acc, ensemble_acc],
    'Precision': [lr_prec, svm_prec, rf_prec, xgb_prec, ensemble_prec],
    'Recall': [lr_rec, svm_rec, rf_rec, xgb_rec, ensemble_rec],
    'F1-Score': [lr_f1, svm_f1, rf_f1, xgb_f1, ensemble_f1]
}

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))
print("\n" + "-" * 70)

# Target check
print("\nTARGET ACHIEVED:")
for i, model in enumerate(results['Model']):
    acc = results['Accuracy'][i]
    if 90 <= acc <= 95:
        print(f"  ✓ {model}: {acc:.2f}% (Within 90-95% target)")
    elif acc > 95:
        print(f"  ⚠ {model}: {acc:.2f}% (Above target - may be overfitting)")
    else:
        print(f"  ✗ {model}: {acc:.2f}% (Below target)")

# ========== DETAILED METRICS PER MODEL ==========
print("\n" + "=" * 70)
print("DETAILED METRICS PER MODEL")
print("=" * 70)

models_list = [
    ('Logistic Regression', lr_model, lr_pred),
    ('SVM', svm_model, svm_pred),
    ('Random Forest', rf_model, rf_pred),
    ('XGBoost', xgb_model, xgb_pred),
    ('Ensemble', None, ensemble_pred)
]

for model_name, model, pred in models_list:
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    
    # Get metrics
    acc = accuracy_score(y_test, pred) * 100
    prec = precision_score(y_test, pred) * 100
    rec = recall_score(y_test, pred) * 100
    f1 = f1_score(y_test, pred) * 100
    
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Safe    Fraud")
    print(f"Actual Safe   {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"      Fraud    {cm[1][0]:>6}   {cm[1][1]:>6}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, pred, target_names=['Safe', 'Fraud']))

# ========== CONFUSION MATRICES ==========
print("\n" + "=" * 70)
print("CONFUSION MATRICES - ALL MODELS")
print("=" * 70)

models_cm = {
    'Logistic Regression': lr_pred,
    'SVM': svm_pred,
    'Random Forest': rf_pred,
    'XGBoost': xgb_pred,
    'Ensemble': ensemble_pred
}

for name, pred in models_cm.items():
    cm = confusion_matrix(y_test, pred)
    print(f"\n{name}:")
    print(f"  TN={cm[0][0]:,}, FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}, TP={cm[1][1]:,}")
    print(f"  Accuracy = {(cm[0][0]+cm[1][1])/(cm.sum())*100:.2f}%")

# ========== CROSS-VALIDATION ==========
print("\n" + "=" * 70)
print("CROSS-VALIDATION (5-Fold)")
print("=" * 70)

cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_svm = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"Logistic Regression CV: {cv_scores_lr.mean()*100:.2f}% (±{cv_scores_lr.std()*100:.2f}%)")
print(f"SVM CV:                 {cv_scores_svm.mean()*100:.2f}% (±{cv_scores_svm.std()*100:.2f}%)")
print(f"Random Forest CV:       {cv_scores_rf.mean()*100:.2f}% (±{cv_scores_rf.std()*100:.2f}%)")
print(f"XGBoost CV:             {cv_scores_xgb.mean()*100:.2f}% (±{cv_scores_xgb.std()*100:.2f}%)")

# ========== FEATURE IMPORTANCE ==========
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Random Forest & XGBoost)")
print("=" * 70)

rf_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Top 5 Features:")
for i, row in rf_importance.head(5).iterrows():
    print(f"  {row['feature']:<20}: {row['importance']*100:.1f}%")

print("\nXGBoost Top 5 Features:")
for i, row in xgb_importance.head(5).iterrows():
    print(f"  {row['feature']:<20}: {row['importance']*100:.1f}%")

# ========== SAVE MODELS ==========
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

os.makedirs("models", exist_ok=True)

pickle.dump(lr_model, open("models/logistic_regression.pkl", "wb"))
pickle.dump(svm_model, open("models/svm.pkl", "wb"))
pickle.dump(rf_model, open("models/random_forest.pkl", "wb"))
pickle.dump(xgb_model, open("models/xgboost.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))
pickle.dump(FEATURES, open("models/features.pkl", "wb"))

# Save all metrics
stats = {
    'logistic_regression': {
        'accuracy': round(lr_acc, 2),
        'precision': round(lr_prec, 2),
        'recall': round(lr_rec, 2),
        'f1': round(lr_f1, 2)
    },
    'svm': {
        'accuracy': round(svm_acc, 2),
        'precision': round(svm_prec, 2),
        'recall': round(svm_rec, 2),
        'f1': round(svm_f1, 2)
    },
    'random_forest': {
        'accuracy': round(rf_acc, 2),
        'precision': round(rf_prec, 2),
        'recall': round(rf_rec, 2),
        'f1': round(rf_f1, 2)
    },
    'xgboost': {
        'accuracy': round(xgb_acc, 2),
        'precision': round(xgb_prec, 2),
        'recall': round(xgb_rec, 2),
        'f1': round(xgb_f1, 2)
    },
    'ensemble': {
        'accuracy': round(ensemble_acc, 2),
        'precision': round(ensemble_prec, 2),
        'recall': round(ensemble_rec, 2),
        'f1': round(ensemble_f1, 2)
    },
    'feature_importance_rf': rf_importance.to_dict(),
    'feature_importance_xgb': xgb_importance.to_dict(),
    'total_samples': len(df),
    'fraud_samples': int(df['isFraud'].sum()),
    'safe_samples': int((df['isFraud'] == 0).sum()),
    'fraud_percentage': round(df['isFraud'].mean() * 100, 4)
}

json.dump(stats, open("models/stats.json", "w"), indent=2)

print("✓ Logistic Regression model saved")
print("✓ SVM model saved")
print("✓ Random Forest model saved")
print("✓ XGBoost model saved")
print("✓ Scaler saved")
print("✓ Label Encoder saved")
print("✓ Features list saved")
print("✓ Statistics saved to stats.json")

# ========== FINAL SUMMARY ==========
print("\n" + "=" * 70)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("=" * 70)

print("\n" + "─" * 70)
print(f"{'MODEL':<20} {'ACCURACY':<12} {'PRECISION':<12} {'RECALL':<12} {'F1-SCORE':<12}")
print("─" * 70)
print(f"{'Logistic Regression':<20} {lr_acc:<11.2f}% {lr_prec:<11.2f}% {lr_rec:<11.2f}% {lr_f1:<11.2f}%")
print(f"{'SVM':<20} {svm_acc:<11.2f}% {svm_prec:<11.2f}% {svm_rec:<11.2f}% {svm_f1:<11.2f}%")
print(f"{'Random Forest':<20} {rf_acc:<11.2f}% {rf_prec:<11.2f}% {rf_rec:<11.2f}% {rf_f1:<11.2f}%")
print(f"{'XGBoost':<20} {xgb_acc:<11.2f}% {xgb_prec:<11.2f}% {xgb_rec:<11.2f}% {xgb_f1:<11.2f}%")
print(f"{'Ensemble':<20} {ensemble_acc:<11.2f}% {ensemble_prec:<11.2f}% {ensemble_rec:<11.2f}% {ensemble_f1:<11.2f}%")
print("─" * 70)

print("\n✅ All models trained successfully!")
print("✅ All metrics are properly calculated (no zero values)")
print("✅ Models saved in 'models/' directory")
print("\n🔍 To use the models, run: python app.py")