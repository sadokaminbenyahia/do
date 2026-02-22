# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation will be computed for the predict function only.
#
# Good luck!
# ----------------------------------------------------------------


# Import necessary libraries here
import pandas as pd
import joblib
import numpy as np

def preprocess(df):
    # Implement any preprocessing steps required for your model here.
    # Return a Pandas DataFrame of the data
    #
    # Note: Don't drop the 'User_ID' column here.
    # It will be used in the predict function to return the final predictions.
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


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------

    # Inside this block, load your trained model.
    # --- Example ---
    # import joblib
    # model = joblib.load('model.pkl')
    model = joblib.load('model.joblib')

    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    predictions = None
    # ------------------ PREDICTION LOGIC ------------------

    # Inside this block, generate predictions using your model.
    # This function should only contain prediction logic.
    # It must be efficient and run within the time limits.
    #
    # You must return a Pandas DataFrame with exactly two columns:
    #
    #   User_ID,Purchased_Coverage_Bundle
    #   USR_060868,7
    #   USR_060869,2
    #   USR_060870,4
    #   ...
    #
    # --- Example ---
    # import pandas as pd
    # preds = model.predict(df.drop(columns=['User_ID']))
    # predictions = pd.DataFrame({
    #     'User_ID': df['User_ID'],
    #     'Purchased_Coverage_Bundle': preds
    # })
    
    # Récupération des objets du modèle
    model_artifacts = model
    clf = model_artifacts['model']
    encoder = model_artifacts['encoder']
    cat_cols = model_artifacts['cat_cols']
    features = model_artifacts['features']
    
    # Extraction de l'User_ID - requis en sortie
    if 'User_ID' not in df.columns:
        raise ValueError("La colonne 'User_ID' est manquante dans les données.")
    user_ids = df['User_ID']
    
    # Assurer que les colonnes catégorielles existaient à l'entraînement
    missing_cat_cols = set(cat_cols) - set(df.columns)
    for c in missing_cat_cols:
        df[c] = 'Missing'
            
    # Application de l'encodeur
    if len(cat_cols) > 0:
        df[cat_cols] = encoder.transform(df[cat_cols].astype(str))
        
    # Gérer les potentielles colonnes qui existeraient lors de l'entraînement 
    # mais qui seraient absentes dans ce dataset de test
    missing_features = set(features) - set(df.columns)
    for f in missing_features:
        df[f] = np.nan
            
    # Inférence
    preds = clf.predict(df[features])
    
    # Création du format de soumission final
    predictions = pd.DataFrame({
        'User_ID': user_ids,
        'Purchased_Coverage_Bundle': preds.astype(int)
    })

    # ------------------ END PREDICTION LOGIC ------------------
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    from time import time

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions and time taken:
    start = time.perf_counter()
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
