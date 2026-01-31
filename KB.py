import pytholog as pl
import pandas as pd
import re
import pickle
import os

# --- Funzione pulizia dati + sintassi Prolog ---
def clean_pl(text):
    if pd.isna(text): 
        return ""
    # Rimuove caratteri speciali che rompono la sintassi Prolog
    text = re.sub(r'[\[\]"\'\(\),]', '', text)
    # Rimuove pesi e misure
    text = re.sub(r'\d+/\d+|\d+\s*(pound|cup|tablespoon|teaspoon|ounce|g|ml|lb|oz)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\b', '', text)
    # Sostituisce spazi con underscore per nomi atomici Prolog
    text = text.strip().replace(" ", "_")
    return text.lower()

# --- Costruzione dataframe ridotto ---
def build_dataframe(limit=30000):
    try:
        df = pd.read_csv("recipes_extended.csv")
    except FileNotFoundError:
        print("Errore: File 'recipes_extended.csv' non trovato.")
        return None
    
    df['ingredients_raw'] = df['ingredients']
    
    # Pulizia iniziale nomi e categorie
    cols_to_clean = ['recipe_title', 'cuisine_list', 'difficulty', 'primary_taste', 'ingredients']
    for col in cols_to_clean:
        df[col] = df[col].apply(clean_pl)
    
    # Conversione booleani
    bool_cols = ['is_vegetarian', 'is_dairy_free', 'is_gluten_free', 'is_nut_free']
    for col in bool_cols:
        df[col] = df[col].apply(lambda x: 'yes' if x == 1 or x is True else 'no')

    # CAMPIONAMENTO
    if len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    return df[['recipe_title', 'cuisine_list', 'difficulty', 'est_prep_time_min',
               'primary_taste', 'is_vegetarian', 'is_dairy_free',
               'is_gluten_free', 'is_nut_free', 'ingredients','ingredients_raw']]

# --- KNOWLEDGE BASE ---
def populate_kb(dataframe):
    recipe_kb = pl.KnowledgeBase('Ricette_Logiche_KB')
    kb = []

    # --- FATTI ---
    for _, row in dataframe.iterrows():
        title = row['recipe_title']
        kb.append(f"is_dairy_free({title},{row['is_dairy_free']})")
        kb.append(f"taste({title},{row['primary_taste']})")
        kb.append(f"is_veg({title},{row['is_vegetarian']})")
        kb.append(f"no_gluten({title},{row['is_gluten_free']})")
        kb.append(f"nut_free({title},{row['is_nut_free']})")

        ing_list = row['ingredients'].split('_')
        for ing in ing_list:
            if len(ing) > 2:
                kb.append(f"contains({title},{ing})")

    # --- CONOSCENZA DEL DOMINIO ---
    taxonomy = {
        'is_protein_source': ['meat', 'beef', 'chicken', 'egg', 'fish', 'tofu', 'tempeh'],
        'is_fat_source': ['oil', 'butter', 'avocado', 'lard', 'bacon', 'cheese', 'walnut'],
        'is_fiber_source': ['lentils', 'beans', 'broccoli', 'oats', 'apple'],
        'is_carb_source': ['rice', 'pasta', 'potato', 'bread', 'quinoa', 'flour'],
        'is_vegetable_source': ['spinach', 'carrot', 'broccoli', 'tomato', 'zucchini', 'kale', 'pepper'],
        'is_fish_source': ['salmon', 'tuna', 'cod', 'shrimp', 'mackerel', 'sardines'],
        'is_high_calorie': ['peanut_butter', 'pasta', 'rice', 'olive_oil', 'honey', 'walnuts', 'beef', 'whole_milk']
    }
    
    kb.append("high_fat(R) :- has_fat(R)")
    for category, items in taxonomy.items():
        for item in items:
            kb.append(f"{category}({item})")

    # Fonti di Potassio ed Elettroliti
    for item in ['banana', 'spinach', 'potato', 'coconut_water', 'yogurt', 'avocado', 'salmon']: 
        kb.append(f"is_electrolyte_source({item})")
        kb.append(f"is_potassium_source({item})")

    # --- REGOLE  ---
    kb.append("has_protein(R) :- contains(R, I), is_protein_source(I)")
    kb.append("has_fat(R) :- contains(R, I), is_fat_source(I)")
    kb.append("has_carb(R) :- contains(R, I), is_carb_source(I)")
    kb.append("has_veggies(R) :- contains(R, I), is_vegetable_source(I)")
    kb.append("has_fish(R) :- contains(R, I), is_fish_source(I)")
    kb.append("has_fiber(R) :- contains(R, I), is_fiber_source(I)")
    kb.append("is_safe_nut_allergy(Recipe) :- nut_free(Recipe, yes)")
    kb.append("has_electrolytes(R) :- contains(R, I), is_electrolyte_source(I)")
    kb.append("has_potassium(R) :- contains(R, I), is_potassium_source(I)")
    kb.append("has_high_energy(R) :- contains(R, I), is_high_calorie(I)")

    # --- REGOLE COMPLESSE ---
    kb.append("is_muscle_recovery(R) :- has_protein(R), has_electrolytes(R), has_potassium(R)")
    kb.append("is_weight_gainer(R) :- has_high_energy(R), has_carb(R), has_fat(R)")
    kb.append("is_peak_performance(R) :- is_muscle_recovery(R), has_carb(R)")
    kb.append("is_complete(R) :- has_protein(R), has_carb(R), has_fat(R)")
    kb.append("is_vitamin_full(R) :- has_veggies(R), has_fish(R)")

    # Modificata Dieta Mediterranea: non richiede più il rank di salute
    kb.append("is_mediterranean(R) :- has_fish(R), has_veggies(R), has_carb(R)")

    kb.append("is_athlete_diet(R) :- has_protein(R), has_carb(R), has_fiber(R)")
    kb.append("is_keto_style(R) :- has_protein(R), has_fat(R)")
    kb.append("is_super_veggie(R) :- is_veg(R, yes), has_fiber(R), has_veggies(R)")

    save_kb_to_pickle(kb)
    recipe_kb(kb)
    return recipe_kb

# --- FUNZIONI UTILI ---
def print_query_results(kb, query_str, description, limit=3):
    print(f"\n{'='*50}")
    print(f" {description.upper()}")
    print(f"{'='*50}")
    results = kb.query(pl.Expr(query_str))
    if results and isinstance(results, list) and results[0] != 'No':
        visti = set()
        count = 0
        for r in results:
            if isinstance(r, dict):
                for key in r:
                    valore = r[key]
                    if valore not in visti:
                        print(f" > {valore.replace('_', ' ').title()}")
                        visti.add(valore)
                        count += 1
            if count >= limit: break
    else:
        print(" ! Nessun risultato trovato per questa categoria.")

def save_kb_to_pickle(kb_list, filename="kb_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(kb_list, f)
    print(f"KB serializzata in {filename}")

def load_kb_from_pickle(filename="kb_data.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            kb_list = pickle.load(f)
        new_kb = pl.KnowledgeBase('Ricette_Logiche_KB')
        new_kb(kb_list)
        print(f"KB caricata da {filename}")
        return new_kb
    return None

if __name__ == "__main__":
    df = build_dataframe()
    if df is not None:
        my_kb = populate_kb(df)
        
        # Test Profilo Gustativo
        #for gusto in ['spicy', 'savory', 'sweet']:
         #   print_query_results(my_kb, f"taste(X, {gusto})", f"Profilo Gustativo: {gusto}")

        # Test Nutrizione Sportiva
        #print_query_results(my_kb, "is_muscle_recovery(Recipe)", "Recupero Muscolare")
        #print_query_results(my_kb, "is_weight_gainer(Recipe)", "Weight Gainer")
        #print_query_results(my_kb, "is_peak_performance(Recipe)", "Peak Performance")

        # Test Diete (Semplificate)
        #print_query_results(my_kb, "is_mediterranean(Recipe)", "Dieta Mediterranea (Ingredienti Base)")
        #print_query_results(my_kb, "is_super_veggie(Recipe)", "Super Veggie")
        
        #print("\n" + "="*50)
        #print(" ELABORAZIONE KB COMPLETATA SENZA HEALTH RANK ")
        #print("="*50)