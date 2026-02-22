import pandas as pd
from sklearn.metrics import f1_score
from solution import preprocess, load_model, predict
import time
import os

def main():
    print("1. Chargement du fichier de test (test1.csv)...")
    df_test = pd.read_csv('test1.csv')
    
    # On isole les vraies étiquettes (labels) pour l'évaluation finale
    y_true = df_test['Purchased_Coverage_Bundle']
    
    # Pour simuler les conditions d'inférence, on retire la colonne cible des données
    df_inference = df_test.drop(columns=['Purchased_Coverage_Bundle'])
    
    print(f"Taille du jeu de test : {df_inference.shape[0]} lignes.")
    
    print("\n2. Exécution du pipeline de la solution...")
    # Étape 1 : Preprocess
    df_preprocessed = preprocess(df_inference)
    
    # Étape 2 : Chargement du modèle
    artifacts = load_model()
    
    # Mesure du temps d'exécution (seulement sur predict() selon les règles)
    start_time = time.time()
    
    # Étape 3 : Prédiction (doit retourner User_ID et Purchased_Coverage_Bundle)
    predictions_df = predict(df_preprocessed, artifacts)
    
    latency_s = time.time() - start_time
    
    print("\n3. Format du DataFrame retourné par predict() :")
    print(predictions_df.head())
    print(f"Colonnes présentes : {list(predictions_df.columns)}")
    
    # Check Model Size
    try:
        size_mb = os.path.getsize('model.joblib') / (1024 * 1024)
    except FileNotFoundError:
        print("\nERREUR: model.joblib introuvable. Taille définie à 0 MB.")
        size_mb = 0
    
    # 4. Évaluation & Calcul du Score Final
    print("\n4. Évaluation des performances...")
    
    # On s'assure d'aligner les User_ID si jamais l'ordre avait changé
    results_merged = df_test[['User_ID', 'Purchased_Coverage_Bundle']].rename(columns={'Purchased_Coverage_Bundle': 'y_true'})
    results_merged = results_merged.merge(predictions_df, on='User_ID', suffixes=('', '_pred'))
    
    macro_f1 = f1_score(results_merged['y_true'], results_merged['Purchased_Coverage_Bundle'], average='macro')
    
    size_penalty = max(0.5, 1 - (size_mb / 200))
    latency_penalty = max(0.5, 1 - (latency_s / 10))
    
    final_score = macro_f1 * size_penalty * latency_penalty
    
    print(f"==> Base Macro F1-Score : {macro_f1:.4f}")
    print(f"==> Model Size          : {size_mb:.2f} MB (Multiplier: {size_penalty:.4f})")
    print(f"==> Inference Latency   : {latency_s:.4f} sec (Multiplier: {latency_penalty:.4f})")
    print("--------------------------------------------------")
    print(f"==> FINAL ADJUSTED SCORE: {final_score:.4f} <==")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
