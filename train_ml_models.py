import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
import os
from pathlib import Path

# Sklearn & Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, 
    precision_recall_curve, auc, confusion_matrix,
    roc_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: XGBoost not found. Skipping XGBoost model.")

warnings.filterwarnings('ignore')

# Set a global style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

class MultiModelTrainer:
    def __init__(self, data_dir='./Younger_icu_features', base_output_dir='./results', seed=42):
        self.data_dir = Path(data_dir)
        self.base_output_dir = Path(base_output_dir)
        self.seed = seed
        self.X = None
        self.y = None
        self.feature_names = None
        
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_engineer(self):
        print("Loading data...")
        try:
            episodes = pd.read_csv(self.data_dir / 'timeseries_dyn_episodes.csv')
            static = pd.read_csv(self.data_dir / 'icu_static_features.csv')
            dyn_X = np.load(self.data_dir / 'timeseries_dyn_X.npy') 
            dyn_names = pd.read_csv(self.data_dir / 'timeseries_dyn_feature_names.csv')['feature_name'].tolist()
        except FileNotFoundError as e:
            print(f"CRITICAL ERROR: {e}")
            return False

        # Shape Detection
        n_features = len(dyn_names)
        if dyn_X.shape[2] == n_features:
            time_axis = 1 
        elif dyn_X.shape[1] == n_features:
            time_axis = 2
        else:
            raise ValueError(f"Shape mismatch! Data is {dyn_X.shape}")

        print("Aggregating time series...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat_mean = np.nanmean(dyn_X, axis=time_axis)
            feat_max = np.nanmax(dyn_X, axis=time_axis)
            feat_min = np.nanmin(dyn_X, axis=time_axis)
            feat_std = np.nanstd(dyn_X, axis=time_axis)
            if time_axis == 1:
                feat_last = dyn_X[:, -1, :] 
            else:
                feat_last = dyn_X[:, :, -1]

        X_aggregated = np.hstack([feat_mean, feat_max, feat_min, feat_std, feat_last])
        
        # Names
        agg_names = []
        suffixes = ['_mean', '_max', '_min', '_std', '_last']
        for suffix in suffixes:
            agg_names.extend([f"{name}{suffix}" for name in dyn_names])
            
        df_dyn = pd.DataFrame(X_aggregated, columns=agg_names)
        df_dyn['stay_id'] = episodes['stay_id']
        
        # Static
        id_cols = ['subject_id', 'hadm_id', 'label'] 
        static_feats = static.drop(columns=[c for c in id_cols if c in static.columns], errors='ignore')
        if 'gender' in static_feats.columns:
            static_feats['gender'] = static_feats['gender'].map({'M': 0, 'F': 1, 'm': 0, 'f': 1})
        static_feats = pd.get_dummies(static_feats, drop_first=True)

        # Merge
        full_df = pd.merge(df_dyn, static_feats, on='stay_id', how='inner')
        labels_map = dict(zip(episodes['stay_id'], episodes['label']))
        full_df['label'] = full_df['stay_id'].map(labels_map)
        full_df = full_df.dropna(subset=['label'])
        
        self.y = full_df['label'].values
        self.X = full_df.drop(columns=['stay_id', 'label']).apply(pd.to_numeric, errors='coerce')
        self.feature_names = self.X.columns.tolist()
        return True

    def train_all(self):
        # 1. Split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=self.seed
        )
        
        # 2. Impute & Scale (Required for SVM/LR)
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_sc = scaler.transform(X_test_imp)
        
        # 3. SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=self.seed)
        X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
        
        # --- DEFINE MODELS ---
        models = {
            "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=2000, random_state=self.seed),
            "DecisionTree": DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=self.seed),
            "SVM": SVC(class_weight='balanced', probability=True, random_state=self.seed),
            "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', random_state=self.seed, n_jobs=-1)
        }
        
        if HAS_XGB:
            pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
            models["XGBoost"] = XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=self.seed, n_jobs=-1)

        # --- LOOP THROUGH MODELS ---
        comparison_results = []
        
        for name, model in models.items():
            print(f"\nProcessing: {name}...")
            model_dir = self.base_output_dir / name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Train
            model.fit(X_train_res, y_train_res)
            
            # Predict
            y_pred_proba = model.predict_proba(X_test_sc)[:, 1]
            
            # Optimize Threshold
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_thresh = thresholds[best_idx]
            
            y_pred_opt = (y_pred_proba >= best_thresh).astype(int)
            
            # Metrics
            auroc = roc_auc_score(y_test, y_pred_proba)
            auprc = auc(recall, precision)
            rec = recall_score(y_test, y_pred_opt)
            prec = precision_score(y_test, y_pred_opt)
            f1 = f1_score(y_test, y_pred_opt)
            
            # Save Results (Including ALL Plots)
            self.save_model_results(model, name, model_dir, y_test, y_pred_proba, y_pred_opt, best_thresh, auroc, auprc, rec, prec, f1)
            
            comparison_results.append({
                "Model": name, "AUROC": auroc, "AUPRC": auprc, "F1": f1, "Recall": rec, "Precision": prec
            })
            
        # --- GLOBAL LASSO FEATURE IMPORTANCE ---
        # Run this once to get the specific "Lasso Importance" plot you requested
        print("\nRunning Global Lasso Analysis...")
        lasso_dir = self.base_output_dir / "Global_Analysis"
        lasso_dir.mkdir(parents=True, exist_ok=True)
        
        lasso = LassoCV(cv=5, random_state=self.seed, max_iter=2000, n_jobs=-1)
        lasso.fit(X_train_res, y_train_res)
        
        self.plot_lasso_importance(lasso, lasso_dir)
            
        # Save Summary
        pd.DataFrame(comparison_results).to_csv(self.base_output_dir / 'model_comparison.csv', index=False)
        print(f"\nCompleted. All results in {self.base_output_dir}")

    def save_model_results(self, model, name, folder, y_test, y_probs, y_pred, thresh, auroc, auprc, rec, prec, f1):
        # 1. Save Metrics JSON
        metrics = {
            "AUROC": float(auroc), "AUPRC": float(auprc),
            "Recall": float(rec), "Precision": float(prec), "F1": float(f1),
            "Best_Threshold": float(thresh),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        with open(folder / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        joblib.dump(model, folder / 'model.joblib')
        
        # 2. PLOT: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auroc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(folder / 'roc_curve.png')
        plt.close()

        # 3. PLOT: PR Curve (Added)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'AUPRC = {auprc:.2f}')
        plt.title(f'Precision-Recall Curve - {name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")
        plt.savefig(folder / 'pr_curve.png')
        plt.close()
        
        # 4. PLOT: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix\n(Threshold = {thresh:.2f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(folder / 'confusion_matrix.png')
        plt.close()
        
        # 5. PLOT: Feature Importance (Model Specific)
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importances, x='importance', y='feature', palette='viridis')
            plt.title(f'Top 20 Features - {name}')
            plt.tight_layout()
            plt.savefig(folder / 'feature_importance.png')
            plt.close()
            importances.to_csv(folder / 'feature_importance.csv', index=False)
            
        elif hasattr(model, 'coef_'):
            # For linear models (LR, SVM), plot coefficients
            coefs = pd.DataFrame({
                'feature': self.feature_names,
                'coef': model.coef_[0]
            }).sort_values(by='coef', key=abs, ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'blue' for x in coefs['coef']]
            sns.barplot(x=coefs['coef'], y=coefs['feature'], palette=colors)
            plt.title(f'Top 20 Coefficients - {name}')
            plt.tight_layout()
            plt.savefig(folder / 'feature_importance.png')
            plt.close()

    def plot_lasso_importance(self, lasso_model, folder):
        # 6. PLOT: Lasso Feature Importance (The specific Red/Blue plot)
        coefs = pd.Series(lasso_model.coef_, index=self.feature_names)
        top_coefs = coefs.abs().sort_values(ascending=False).head(20)
        plot_coefs = coefs[top_coefs.index]
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'blue' for x in plot_coefs.values]
        plot_coefs.plot(kind='barh', color=colors)
        plt.title('Global Feature Importance (LassoCV)\nBlue=Risk Factor, Red=Protective Factor')
        plt.xlabel('Coefficient Value')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(folder / 'lasso_feature_importance.png')
        plt.close()

if __name__ == "__main__":
    trainer = MultiModelTrainer(data_dir='./Younger_icu_features', base_output_dir='./results')
    if trainer.load_and_engineer():
        trainer.train_all()
