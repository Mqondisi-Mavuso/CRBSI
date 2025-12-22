import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve, f1_score, accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import sys
import copy

# Import the vanilla model
try:
    from vanilla_transformer_model import VanillaTransformer
except ImportError:
    print("Error: vanilla_transformer_model.py not found.")
    sys.exit(1)

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CONFIG = {
    'batch_size': 64,
    'epochs': 50,           
    'lr': 1e-4,              
    'patience': 10,          
    'weight_decay': 1e-4,    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'n_splits': 5  # 5-Fold CV
}

class CRBSIDataset(Dataset):
    def __init__(self, static, dynamic, labels):
        self.static = torch.FloatTensor(static)
        self.dynamic = torch.FloatTensor(dynamic)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'static': self.static[idx],
            'dynamic': self.dynamic[idx],
            'label': self.labels[idx]
        }

def load_and_process_data(data_dir='/staging/biology/u800011783/AI_Intelligent_Medicine/final_project/Public_Mimi_iv_dataset/Younger_icu_features'):
    print("Loading and processing data...")
    path = Path(data_dir)
    
    try:
        episodes = pd.read_csv(path / 'timeseries_dyn_episodes.csv')
        static = pd.read_csv(path / 'icu_static_features.csv')
        dyn_X = np.load(path / 'timeseries_dyn_X.npy') 
        dyn_names = pd.read_csv(path / 'timeseries_dyn_feature_names.csv')['feature_name'].tolist()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    n_features = len(dyn_names)
    if dyn_X.shape[2] == n_features:
        pass 
    elif dyn_X.shape[1] == n_features:
        dyn_X = dyn_X.transpose(0, 2, 1)
    
    # --- Forward Fill Imputation ---
    print("Applying Forward Fill Imputation...")
    dyn_X = np.nan_to_num(dyn_X, nan=np.nan, posinf=np.nan, neginf=np.nan)
    N, T, F = dyn_X.shape
    for i in range(N):
        patient_df = pd.DataFrame(dyn_X[i])
        patient_df = patient_df.ffill().bfill() 
        dyn_X[i] = patient_df.values
    dyn_X = np.nan_to_num(dyn_X, nan=0.0)

    # --- Static Features ---
    print("Processing static features...")
    id_cols = ['subject_id', 'hadm_id', 'label'] 
    static_feats = static.drop(columns=[c for c in id_cols if c in static.columns], errors='ignore')
    if 'gender' in static_feats.columns:
        static_feats['gender'] = static_feats['gender'].map({'M': 0, 'F': 1, 'm': 0, 'f': 1})
    static_feats = pd.get_dummies(static_feats, drop_first=True)
    
    df_dyn_map = pd.DataFrame({'stay_id': episodes['stay_id'], 'original_idx': range(len(episodes))})
    merged = pd.merge(df_dyn_map, static_feats, on='stay_id', how='inner')
    
    labels_map = dict(zip(episodes['stay_id'], episodes['label']))
    merged['label'] = merged['stay_id'].map(labels_map)
    merged = merged.dropna(subset=['label'])
    
    valid_indices = merged['original_idx'].values
    final_static = merged.drop(columns=['stay_id', 'original_idx', 'label']).values.astype(float)
    final_dynamic = dyn_X[valid_indices]
    final_labels = merged['label'].values.astype(float)
    
    # --- Scaling ---
    imputer = SimpleImputer(strategy='mean')
    final_static = imputer.fit_transform(final_static)
    scaler_s = StandardScaler()
    final_static = scaler_s.fit_transform(final_static)
    
    scaler_d = StandardScaler()
    final_dynamic = scaler_d.fit_transform(final_dynamic.reshape(-1, F)).reshape(N, T, F)
    
    return final_static, final_dynamic, final_labels

def calculate_accuracy(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = labels.detach().cpu().numpy()
    return accuracy_score(y_true, preds)

def train_one_fold(fold_idx, train_idx, val_idx, static, dyn, y):
    print(f"\n--- Training Fold {fold_idx+1}/{CONFIG['n_splits']} ---")
    
    train_ds = CRBSIDataset(static[train_idx], dyn[train_idx], y[train_idx])
    val_ds = CRBSIDataset(static[val_idx], dyn[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    feat_dim = dyn.shape[2]
    static_dim = static.shape[1]
    
    model = VanillaTransformer(
        static_dim=static_dim, 
        dynamic_dim=feat_dim,
        d_model=64, nhead=4, num_layers=2, dropout=0.3
    ).to(CONFIG['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    pos_count = np.sum(y[train_idx])
    neg_count = len(train_idx) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(CONFIG['device'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 0
    
    # History for this fold
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_acc_sum = 0
        
        for batch in train_loader:
            s = batch['static'].to(CONFIG['device'])
            d = batch['dynamic'].to(CONFIG['device'])
            label = batch['label'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits = model(s, d).squeeze()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc_sum += calculate_accuracy(logits, label)
            
        # Validation
        model.eval()
        val_loss = 0
        val_acc_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                s = batch['static'].to(CONFIG['device'])
                d = batch['dynamic'].to(CONFIG['device'])
                label = batch['label'].to(CONFIG['device'])
                logits = model(s, d).squeeze()
                loss = criterion(logits, label)
                val_loss += loss.item()
                val_acc_sum += calculate_accuracy(logits, label)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_acc = train_acc_sum / len(train_loader)
        avg_val_acc = val_acc_sum / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG['patience']:
                break
                
    # Evaluate best model
    model.load_state_dict(best_model_wts)
    model.eval()
    y_true, y_probs = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            s = batch['static'].to(CONFIG['device'])
            d = batch['dynamic'].to(CONFIG['device'])
            logits = model(s, d).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(batch['label'].cpu().numpy())
            if np.ndim(probs) == 0: y_probs.append(probs.item())
            else: y_probs.extend(probs)
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    try: auroc = roc_auc_score(y_true, y_probs)
    except: auroc = 0.5
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auprc = auc(recall, precision)
    
    return auroc, auprc, y_true, y_probs, history

def plot_cv_history(all_histories):
    """
    Robust plotting that handles unequal epoch counts across folds.
    """
    output_dir = Path('./results/VanillaTransformer')
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. FIX: Find the minimum length across all folds to allow averaging
    min_len = min([len(h['train_loss']) for h in all_histories])
    
    # Truncate all histories to min_len
    clean_train_loss = [h['train_loss'][:min_len] for h in all_histories]
    clean_val_loss = [h['val_loss'][:min_len] for h in all_histories]
    clean_train_acc = [h['train_acc'][:min_len] for h in all_histories]
    clean_val_acc = [h['val_acc'][:min_len] for h in all_histories]
    
    # --- Plot Loss ---
    plt.figure(figsize=(10, 6))
    
    # Plot individual folds (full length) in light color
    for h in all_histories:
        plt.plot(h['train_loss'], color='blue', alpha=0.1)
        plt.plot(h['val_loss'], color='orange', alpha=0.1)
        
    # Plot Mean (truncated)
    mean_train_loss = np.mean(clean_train_loss, axis=0)
    mean_val_loss = np.mean(clean_val_loss, axis=0)
    
    plt.plot(mean_train_loss, color='blue', linewidth=2, label='Mean Train Loss')
    plt.plot(mean_val_loss, color='orange', linewidth=2, label='Mean Val Loss')
    
    plt.title('CV Training & Validation Loss (5 Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'cv_loss_curves.png')
    plt.close()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 6))
    
    for h in all_histories:
        plt.plot(h['train_acc'], color='green', alpha=0.1)
        plt.plot(h['val_acc'], color='red', alpha=0.1)

    mean_train_acc = np.mean(clean_train_acc, axis=0)
    mean_val_acc = np.mean(clean_val_acc, axis=0)

    plt.plot(mean_train_acc, color='green', linewidth=2, label='Mean Train Acc')
    plt.plot(mean_val_acc, color='red', linewidth=2, label='Mean Val Acc')
    
    plt.title('CV Training & Validation Accuracy (5 Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(output_dir / 'cv_accuracy_curves.png')
    plt.close()

def run_kfold_training():
    static, dyn, labels = load_and_process_data()
    
    skf = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=CONFIG['seed'])
    
    results = {'auroc': [], 'auprc': [], 'all_y_true': [], 'all_y_probs': []}
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(static, labels)):
        auroc, auprc, y_true, y_probs, history = train_one_fold(fold, train_idx, val_idx, static, dyn, labels)
        
        results['auroc'].append(auroc)
        results['auprc'].append(auprc)
        results['all_y_true'].extend(y_true)
        results['all_y_probs'].extend(y_probs)
        all_histories.append(history)
        
        print(f"Fold {fold+1} Result: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
        
    # --- Aggregate Results ---
    mean_auroc = np.mean(results['auroc'])
    mean_auprc = np.mean(results['auprc'])
    
    print(f"\n{'='*30}\n5-FOLD CV RESULTS\n{'='*30}")
    print(f"Mean AUROC: {mean_auroc:.4f}")
    print(f"Mean AUPRC: {mean_auprc:.4f}")
    
    # Save Plots
    plot_cv_history(all_histories)
    
    # Save Final Metrics & CM
    save_final_results(results['all_y_true'], results['all_y_probs'], mean_auroc, mean_auprc)

def save_final_results(y_true, y_probs, mean_auroc, mean_auprc):
    output_dir = Path('./results/VanillaTransformer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    y_pred = (np.array(y_probs) >= best_thresh).astype(int)
    
    metrics = {
        "Mean_AUROC": mean_auroc,
        "Mean_AUPRC": mean_auprc,
        "Overall_F1": f1_score(y_true, y_pred),
        "Best_Threshold": float(best_thresh),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    with open(output_dir / 'cv_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    sns.set_style("whitegrid")
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Mean AUC = {mean_auroc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('ROC Curve - Vanilla Transformer (5-Fold CV)')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Infection', 'Infection'],
                yticklabels=['No Infection', 'Infection'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Aggregated Confusion Matrix\n(Threshold = {best_thresh:.2f})')
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    run_kfold_training()