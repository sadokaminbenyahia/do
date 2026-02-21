import pandas as pd
from sklearn.metrics import f1_score
from solution import preprocess, load_model, predict
import time

def main():
    print("1. Chargement du fichier de test (test1.csv)...")
    df_test = pd.read_csv('test1.csv')
    
    # On isole les vraies étiquettes (labels) pour l'évaluation finale
    y_true = df_test['Purchased_Coverage_Bundle']
    
    # Pour simuler les conditions d'inférence, on retire la colonne cible des données
    df_inference = df_test.drop(columns=['Purchased_Coverage_Bundle'])
    
    print(f"Taille du jeu de test : {df_inference.shape[0]} lignes.")
    
    # Mesure du temps d'exécution
    start_time = time.time()
    
    print("\n2. Exécution du pipeline de la solution...")
    # Étape 1 : Preprocess
    df_preprocessed = preprocess(df_inference)
    
    # Étape 2 : Chargement du modèle
    artifacts = load_model()
    
    # Étape 3 : Prédiction (doit retourner User_ID et Purchased_Coverage_Bundle)
    predictions_df = predict(df_preprocessed, artifacts)
    
    execution_time = time.time() - start_time
    
    print("\n3. Format du DataFrame retourné par predict() :")
    print(predictions_df.head())
    print(f"Colonnes présentes : {list(predictions_df.columns)}")
    print(f"\nTemps d'inférence (Preprocess + Pred) : {execution_time:.2f} secondes.")
    
    # 4. Évaluation avec le Macro F1-Score
    print("\n4. Évaluation des performances (Macro F1-Score)...")
    
    # On s'assure d'aligner les User_ID si jamais l'ordre avait changé (bien que non supposé)
    results_merged = df_test[['User_ID', 'Purchased_Coverage_Bundle']].rename(columns={'Purchased_Coverage_Bundle': 'y_true'})
    results_merged = results_merged.merge(predictions_df, on='User_ID', suffixes=('', '_pred'))
    
    macro_f1 = f1_score(results_merged['y_true'], results_merged['Purchased_Coverage_Bundle'], average='macro')
    
    print(f"==> Macro F1-Score sur test1.csv : {macro_f1:.4f} <==")
    
    if execution_time < 10:
        print("[OK] Condition respectee : L'inference prend bien moins de 10 secondes.")
    else:
        print("[KO] Attention : L'inference est trop lente (> 10 sec).")

if __name__ == "__main__":
    main()
