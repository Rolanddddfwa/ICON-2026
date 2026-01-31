import joblib
import pandas as pd
from collections import Counter
import KB
import CSP
import re
# --- FUNZIONE DI PREDIZIONE ---
def predict_user_taste(model, vectorizer, user_recipes):
    if not user_recipes:
        return "neutral"
        
    predicted_tastes = []
    feature_names = vectorizer.get_feature_names_out()
    
    for recipe in user_recipes:
        
        raw_text = recipe['ingredients'].lower()
        cleaned_ing = re.sub(r'[^a-zA-Z\s]', ' ', raw_text) 

        ing_matrix = vectorizer.transform([cleaned_ing]).toarray()
        ing_df = pd.DataFrame(ing_matrix, columns=feature_names)
        
        taste = model.predict(ing_df)[0]
        predicted_tastes.append(taste)
    
    return Counter(predicted_tastes).most_common(1)[0][0]

# --- FUNZIONE STAMPA MENU ---
def stampa_menu_completo(soluzione, preferred_taste):
    if not soluzione:
        print("\n[!] Impossibile generare un menù con questi vincoli.")
        return
        
    giorni = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    print("\n" + "="*60)
    print(f"{'IL TUO MENÙ SETTIMANALE DETTAGLIATO':^60}")
    print(f"{'Gusto di riferimento: ' + preferred_taste.upper():^60}")
    print("="*60)

    for g in giorni:
        print(f"\n{g.upper()}:")
        pasti_del_giorno = []
        
        for pasto in ["Colazione", "Pranzo", "Cena"]:
            ricetta = soluzione[g][pasto]
            titolo = ricetta['recipe_title'].replace('_', ' ').title()
            print(f"  {pasto:10}: {titolo} [{ricetta['primary_taste']}]")
            pasti_del_giorno.append(ricetta)
        
        print(f"\n  --- LISTA INGREDIENTI {g.upper()} ---")
        for r in pasti_del_giorno:
            titolo = r['recipe_title'].replace('_', ' ').title()
            # Mostra gli ingredienti separati da virgola per leggibilità
            ingredienti_raw = r['ingredients_raw'].replace('_', ', ')
            print(f"  * Per {titolo}:")
            print(f"    {ingredienti_raw}")
        
        print("-" * 40)
    
    print("="*60)

# --- MAIN ---
def main():
    try:
        print("Caricamento modello...")
        data = joblib.load('modello_gusti_ricette.pkl')
        # Estrazione modello e vettorizzatore
        model = data['model']
        vectorizer = data['vectorizer']
        
    except FileNotFoundError:
        print("\n[ERRORE] Modello 'modello_gusti_ricette.pkl' non trovato!")
        print("Assicurati di aver addestrato il modello prima di avviare il Main.")
        return
    except (KeyError, Exception) as e:
        print(f"\n[ERRORE] Caricamento modello fallito: {e}")
        return

    # Costruzione del dataframe ricette dal modulo KB
    print("Caricamento KB...")
    df = KB.build_dataframe()
    if df is None:
        return
        
    my_kb = KB.load_kb_from_pickle()
    if my_kb is None:
        print("KB non trovata, creazione in corso...")
        my_kb = KB.populate_kb(df)
        print("\n--- SISTEMA DI RACCOMANDAZIONE NUTRIZIONALE ---")
    
    # --- CONFIGURAZIONE PROFILO UTENTE ---
    user_recipes = []
    print("\nInserisci alcuni piatti che ti piacciono (almeno uno).")
    while True:
        nome = input("\nNome piatto preferito (o 'fine' per terminare): ")
        if nome.lower() == 'fine': 
            break
        ingred = input("Ingredienti principali (es: tomato, pasta, chili): ")
        user_recipes.append({'title': nome, 'ingredients': ingred})

    if not user_recipes:
        print("Nessun dato inserito. Chiusura.")
        return
    # Predizione del gusto preferito
    preferred_taste = predict_user_taste(model, vectorizer, user_recipes)
    print(f"\n>>> GUSTO RILEVATO: {preferred_taste.upper()} <<<")

    try:
        peso = float(input("\nPeso (kg): "))
        altezza = float(input("Altezza (cm): "))
    except ValueError:
        print("Inserire valori numerici validi per peso e altezza.")
        return
    
    user_data = {
        "bmi": peso / ((altezza/100)**2),
        "sport": input("Fai sport regolarmente? (s/n): ").lower() == 's',
        "intolleranze": {
            "lattosio": input("Intollerante al lattosio? (s/n): ").lower() == 's',
            "noci": input("Allergia alle noci/frutta a guscio? (s/n): ").lower() == 's',
            "glutine": input("Celiachia o sensibilità al glutine? (s/n): ").lower() == 's'
        }
    }

    
    # --- CICLO DI GENERAZIONE MENU ---
    while True:
        print("\nGenerazione menù personalizzato in corso...")
        # Il CSP userà il gusto predetto per filtrare o dare priorità ai piatti
        menu_finale = CSP.solve_menu_csp(my_kb, user_data, preferred_taste,df)
        stampa_menu_completo(menu_finale, preferred_taste)

        ancora = input("\nDesideri generare un'altra versione di questo menù? (s/n): ").lower()
        if ancora != 's':
            print("\nUscita in corso. Buona dieta e buon appetito!")
            break

if __name__ == "__main__":
    main()