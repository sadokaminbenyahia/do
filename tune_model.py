import optuna
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from solution import preprocess
import numpy as np

def main():
    print("1. Loading raw data...")
    # Load Data Outside Objective
    df_train = pd.read_csv('train1.csv')
    df_val = pd.read_csv('test1.csv')

    # Separate features and targets
    y_train = df_train['Purchased_Coverage_Bundle']
    X_train_raw = df_train.drop(columns=['Purchased_Coverage_Bundle'])

    y_val = df_val['Purchased_Coverage_Bundle']
    X_val_raw = df_val.drop(columns=['Purchased_Coverage_Bundle'])

    print("2. Applying preprocess()...")
    # Apply Preprocessing
    X_train_pre = preprocess(X_train_raw)
    X_val_pre = preprocess(X_val_raw)

    print("3. Dropping User_ID & Formatting data for model...")
    # Drop User_ID from both
    X_train = X_train_pre.drop(columns=['User_ID'])
    X_val = X_val_pre.drop(columns=['User_ID'])

    # Encode Categories
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    if len(cat_cols) > 0:
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
        
        # Handle any missing categorical columns in validation gracefully
        missing_cat_cols = set(cat_cols) - set(X_val.columns)
        for c in missing_cat_cols:
            X_val[c] = 'Missing'
            
        X_val[cat_cols] = encoder.transform(X_val[cat_cols].astype(str))

    # Also make sure all features in X_train exist in X_val (in case a numeric column was dropped)
    features = X_train.columns.tolist()
    missing_features = set(features) - set(X_val.columns)
    for f in missing_features:
        X_val[f] = np.nan
        
    X_val = X_val[features]  # Guarantee identical order
    
    # ------------------ Optuna Setup ------------------
    print("4. Starting Optuna Hyperparameter Study...")

    def objective(trial):
        # Define the hyperparameter search space
        params = {
            'objective': 'multiclass',
            'num_class': 10,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            # Cap estimators to prevent the 26-second latency blowout
            'n_estimators': trial.suggest_int('n_estimators', 200, 350), 
            # Allow slightly higher learning rate to compensate for fewer trees
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
            # Keep trees relatively shallow to prevent the 97MB size blowout
            'max_depth': trial.suggest_int('max_depth', 6, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 90),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 60)
        }

        # Model Training & Evaluation
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        macro_f1 = f1_score(y_val, preds, average='macro')
        
        return macro_f1

    # Run Study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Output optimal results
    print("\n----------------------------------------------------")
    print("OPTUNA STUDY COMPLETED!")
    print(f"BEST MACRO F1-SCORE: {study.best_value:.4f}")
    print("----------------------------------------------------")
    print("BEST HYPERPARAMETERS TO COPY INTO train_pipeline.py:")
    print("----------------------------------------------------")
    for key, value in study.best_params.items():
        print(f"        {key}={value},")
    print("----------------------------------------------------")

if __name__ == "__main__":
    main()
