import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

def preprocess(df):
    df_out = df.copy()
    
    # ------------------ Demographics ------------------
    dep_cols = ['Adult_Dependents', 'Child_Dependents', 'Infant_Dependents']
    if all(c in df_out.columns for c in dep_cols):
        df_out['Total_Dependents'] = df_out['Adult_Dependents'] + df_out['Child_Dependents'] + df_out['Infant_Dependents']
    else:
        df_out['Total_Dependents'] = 0

    df_out['Is_Family'] = (df_out['Total_Dependents'] > 0).astype(int)

    if 'Estimated_Annual_Income' in df_out.columns:
        df_out['Revenu_par_tete'] = df_out['Estimated_Annual_Income'] / (df_out['Total_Dependents'] + 1)
        
    # ------------------ Risk Profile ------------------
    if 'Previous_Claims_Filed' in df_out.columns and 'Previous_Policy_Duration_Months' in df_out.columns:
        denom = df_out['Previous_Policy_Duration_Months'].clip(lower=1)
        df_out['Claims_Per_Month'] = df_out['Previous_Claims_Filed'] / denom
        
    if 'Previous_Claims_Filed' in df_out.columns:
        df_out['Has_Previous_Claims'] = (df_out['Previous_Claims_Filed'] > 0).astype(int)
        
    # ------------------ Operational Friction ------------------
    friction_cols = ['Underwriting_Processing_Days', 'Days_Since_Quote']
    if all(c in df_out.columns for c in friction_cols):
        df_out['Total_Friction_Days'] = df_out['Underwriting_Processing_Days'] + df_out['Days_Since_Quote']
    
    # ------------------ Identify Columns ------------------
    cat_cols = df_out.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'User_ID' in cat_cols:
        cat_cols.remove('User_ID')
        
    num_cols = df_out.select_dtypes(exclude=['object', 'category']).columns.tolist()
    if 'User_ID' in num_cols:
        num_cols.remove('User_ID')
    if 'Purchased_Coverage_Bundle' in num_cols:
        num_cols.remove('Purchased_Coverage_Bundle')
        
    # ------------------ Missing Values & Types ------------------
    for c in cat_cols:
        df_out[c] = df_out[c].fillna('Missing').astype(str)
        
    for c in num_cols:
        # Do not fill missing numerical values with -1. Leave them as NaN. 
        # LightGBM is optimized to handle missing values natively.
        df_out[c] = df_out[c].astype(float)

    return df_out

def main():
    print("Mise en place du Pipeline pour la Phase 1...")
    
    print("1. Chargement de train1.csv...")
    df = pd.read_csv('train1.csv')
    
    # Séparer X et y en gardant User_ID dans X pour le preprocessing (même s'il ne faut pas le drop dans l'énoncé de process)
    y = df['Purchased_Coverage_Bundle']
    X = df.drop(columns=['Purchased_Coverage_Bundle'])
    
    print("2. Preprocessing & Feature Engineering...")
    X_preprocessed = preprocess(X)
    
    # On isole les variables pour l'entraînement (sans User_ID)
    X_train = X_preprocessed.drop(columns=['User_ID'])
    
    print("3. Encodage des catégories textuelles...")
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # On utilise handle_unknown pour gérer d'éventuelles strings inconnues dans le test set
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        
    print("4. Entraînement du modèle LGBMClassifier...")
    model = LGBMClassifier(
        class_weight='balanced',
        objective='multiclass',
        num_class=10,
        random_state=42,
        n_jobs=-1,
        n_estimators=335,
        learning_rate=0.06922234836478142,
        max_depth=10,
        num_leaves=81,
        subsample=0.8247062763577886,
        colsample_bytree=0.8260390616334965,
        min_child_samples=55
    )
    model.fit(X_train, y)
    
    print("5. Sauvegarde en model.joblib...")
    model_artifacts = {
        'model': model,
        'encoder': encoder,
        'cat_cols': cat_cols,
        'features': X_train.columns.tolist()  # On sauvegarde l'ordre et le nom des features
    }
    joblib.dump(model_artifacts, 'model.joblib')
    
    print("Entraînement terminé et modèle sauvegardé avec succès.")

if __name__ == "__main__":
    main()
