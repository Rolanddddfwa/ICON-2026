import joblib
import pandas as pd
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

# --- pulizia colonna ingredients ---
def clean_ingredients_format(text):
    if pd.isna(text): return ""  # noqa: E701
    text = re.sub(r'[\[\]"]', '', text)
    text = re.sub(r'\d+/\d+|\d+\s*(pound|cup|with|liquid|green|tablespoon|teaspoon|package|large|ounce|g|ml|lb|oz)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\b', '', text)
    return text.lower().strip()

def balance_dataset(df, target_count):

    target_classes = ['sweet', 'savory', 'spicy'] #i gusti principali definiti
    balanced_chunks = []

    #creiamo sun sabset delt dataset che ha in se solo i piatti: 'sweet', 'savory', 'spicy'
    for taste in target_classes:
        subset = df[df['primary_taste'] == taste]
        if len(subset) >= target_count:
            subset_sampled = subset.sample(n=target_count, random_state=42)
            balanced_chunks.append(subset_sampled)
        else:
            print(f"Attenzione: la classe {taste} ha solo {len(subset)} campioni.")
            balanced_chunks.append(subset)

    df_balanced = pd.concat(balanced_chunks).sample(frac=1).reset_index(drop=True)
    return df_balanced


#funzione che esegue una Randomized Search degli hyperparameters del modello scelto
#si usa il k fold cross validation
def RandomizedSearch(hyperparameters, X_train, y_train):

    dtc = RandomForestClassifier(random_state=42)
    cvFold = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
    
    randomSearch = RandomizedSearchCV(
        estimator=dtc, 
        cv=cvFold, 
        param_distributions=hyperparameters,
        n_iter=3, # Numero di combinazioni da provare
        n_jobs=-1
    )
    
    best_model = randomSearch.fit(X_train, y_train)
    return best_model

#funzione che valuta  le metriche: Accuracy, Classification Report e ROC Score
def modelEvaluation(y_test, y_pred, pred_prob):
    print('Classification report: \n', classification_report(y_test, y_pred, zero_division=0))
    roc_score = roc_auc_score(y_test, pred_prob, multi_class='ovr')
    print('ROC score: ', roc_score)
    return roc_score

#funzione che cerca i migliori hyperparameters , ripetendo più volte la funzione che effettua la Randomized Search
#restituisce un dizionario contenente hyperparameters e ROC score
def HyperparametersSearch(X_train, X_test, y_train, y_test):
    result = {}
    
    # Definiamo gli iperparametri per il Decision Tree
    hyperparameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(5, 30, 5)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }

    i = 0
    while i < 3: 
        print(f"Iterazione ricerca iperparametri: {i+1}/3")
        best_model = RandomizedSearch(hyperparameters, X_train, y_train)
        
        # Estraiamo i parametri migliori
        bestParams = best_model.best_estimator_.get_params()
        
        # Ricreiamo il modello per testarlo
        dtc = RandomForestClassifier(**bestParams)
        dtc.fit(X_train, y_train)
        
        pred_prob = dtc.predict_proba(X_test)
        roc_score = roc_auc_score(y_test, pred_prob, multi_class='ovr')

        result[i] = {
            'params': bestParams,
            'roc_score': roc_score
        }
        i += 1

    # Ordiniamo per ROC score migliore
    result = dict(sorted(result.items(), key=lambda x: x[1]['roc_score'], reverse=True))
    first_el = list(result.keys())[0]
    return result[first_el]

#funzione che valuta mano a mano i modelli con i diversi iperparametri scelti
def SearchingBestModelStats(X_train, X_test, y_train, y_test):
    
    print('\nIniziale composizione del modello con iperparametri basici...')
    dtc = RandomForestClassifier(max_depth=5, random_state=42)
    dtc.fit(X_train, y_train)

    y_pred = dtc.predict(X_test)
    pred_prob = dtc.predict_proba(X_test)

    print('\nValutazione del modello base...')
    modelEvaluation(y_test, y_pred, pred_prob)

    print('\nRicerca iperparametri ottimali in corso...')
    best_res = HyperparametersSearch(X_train, X_test, y_train, y_test)

    print('\n--- MIGLIORI IPERPARAMETRI TROVATI ---')
    best_params = best_res['params']
    for p, v in best_params.items():
        if v is not None and p in ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            print(f"{p}: {v}")

    print('\nRicomponiamo il modello utilizzando i nuovi iperparametri...')
    dtc_final = RandomForestClassifier(**best_params)
    dtc_final.fit(X_train, y_train)

    y_pred_final = dtc_final.predict(X_test)
    pred_prob_final = dtc_final.predict_proba(X_test)

    modelEvaluation(y_test, y_pred_final, pred_prob_final)
    return dtc_final

def main():
    # caricamento dataset
    df = pd.read_csv('recipes_extended.csv')
    
    # Pulizia righe con valori mancanti nelle colonne fondamentali
    cols_to_check = ['ingredient_text', 'primary_taste']
    df = df.dropna(subset=cols_to_check)

    # bilanciamento dataset
    print("Bilanciamento delle classi in corso...")
    df = balance_dataset(df, 9500)
    
    print("\nNuova distribuzione delle classi:")
    print(df['primary_taste'].value_counts())

    # pulizia del testo
    print("\nPulizia ingredienti...")
    df['clean_ingredients'] = df['ingredient_text'].apply(clean_ingredients_format)

    # vettorizzazione
    # aumentiamo a 500 feature per catturare meglio le differenze tra savory e spicy
    vectorizer = CountVectorizer(stop_words='english', max_features=500, binary=True)
    ing_matrix = vectorizer.fit_transform(df['clean_ingredients']).toarray()
    X = pd.DataFrame(ing_matrix, columns=vectorizer.get_feature_names_out())
    y = df['primary_taste']

    # split (Manteniamo la stratificazione per sicurezza)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ottimizzazione e valutazione
    best_dtc = SearchingBestModelStats(X_train, X_test, y_train, y_test)

    print('\nFase di apprendimento completata con successo.')
    print('Il modello è ora addestrato su un dataset bilanciato (n=~28500 totali).')

    # salvataggio del modello
    model_data = {
        'model': best_dtc,
        'vectorizer': vectorizer
    }
    joblib.dump(model_data, 'modello_gusti_ricette.pkl')
    print("\nModello e vettorizzatore salvati con successo in 'modello_gusti_ricette.pkl'!")

if __name__ == "__main__":
    main()