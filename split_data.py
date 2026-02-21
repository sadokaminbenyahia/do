import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1. Lire le fichier train.csv
    print("Lecture de train.csv...")
    df = pd.read_csv('train.csv')
    
    # 2. Séparer le dataset en 80% train / 20% test avec un split stratifié statifié
    # La colonne cible est 'Purchased_Coverage_Bundle'
    print("Séparation des données (80% train / 20% test) avec stratification...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['Purchased_Coverage_Bundle']
    )
    
    # 3. Sauvegarder les deux datasets dans de nouveaux fichiers sans index
    print("Sauvegarde de train1.csv...")
    train_df.to_csv('train1.csv', index=False)
    
    print("Sauvegarde de test1.csv...")
    test_df.to_csv('test1.csv', index=False)
    
    print("Terminé ! Les fichiers train1.csv et test1.csv ont été créés avec succès.")

if __name__ == "__main__":
    main()
