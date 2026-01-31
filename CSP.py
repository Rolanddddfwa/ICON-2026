from constraint import Problem, AllDifferentConstraint
import pytholog as pl
import random

def solve_menu_csp(kb, user_data, preferred_taste, df):
    problem = Problem()

    # --- HELPER PER INTERROGARE LA KB ---
    def check_kb(query_str):
        try:
            res = kb.query(pl.Expr(query_str))
            if res is None: return False
            return res and res != 'No' and len(res) > 0
        except Exception:
            return False

    # --- RECUPERO DOMINI DALLA KB ---
    tutte_le_ricette = df.to_dict('records')
    
    colazione_titles = []
    pranzo_cena_titles = []

    for res in tutte_le_ricette:
        t = res['recipe_title']
        taste = res['primary_taste']
        
        # --- FILTRI INTOLLERANZE ---
        if user_data['intolleranze']['lattosio'] and not check_kb(f"is_dairy_free({t}, yes)"): continue
        if user_data['intolleranze']['noci'] and not check_kb(f"nut_free({t}, yes)"): continue
        if user_data['intolleranze']['glutine'] and not check_kb(f"no_gluten({t}, yes)"): continue

        # --- SUDDIVISIONE PASTI ---
        if taste in ['sweet', 'neutral']:
            colazione_titles.append(t)
        
        # Pranzi e cene: includiamo il gusto preferito o tutto ciò che non è prettamente da colazione
        if taste not in ['sweet', 'neutral'] or taste == preferred_taste:
            pranzo_cena_titles.append(t)

    # --- DEFINIZIONE VARIABILI ---
    giorni = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    pasti_tipo = ["Colazione", "Pranzo", "Cena"]
    variabili_create = []

    for g in giorni:
        for pt in pasti_tipo:
            var_name = f"{g}_{pt}"
            domain = list(colazione_titles if pt == "Colazione" else pranzo_cena_titles)
            
            if not domain:
                print(f"ERRORE: Dominio vuoto per {var_name}. Controlla i filtri sanitari o il dataset.")
                return None 
            
            random.shuffle(domain) # Aggiunge varietà ad ogni generazione
            problem.addVariable(var_name, domain)
            variabili_create.append(var_name)

    # --- VINCOLI ---
    
    # Varietà settimanale: non mangiare la stessa cosa a pranzo (o cena) in giorni diversi
    vars_pranzo = [f"{g}_Pranzo" for g in giorni if f"{g}_Pranzo" in variabili_create]
    vars_cena = [f"{g}_Cena" for g in giorni if f"{g}_Cena" in variabili_create]
    
    if len(vars_pranzo) > 1:
        problem.addConstraint(AllDifferentConstraint(), vars_pranzo)
    if len(vars_cena) > 1:
        problem.addConstraint(AllDifferentConstraint(), vars_cena)

    for g in giorni:
        p_var = f"{g}_Pranzo"
        c_var = f"{g}_Cena"
        
        if p_var in variabili_create and c_var in variabili_create:
            # Pranzo diverso da Cena nello stesso giorno
            problem.addConstraint(lambda p, c: p != c, (p_var, c_var))

            # Vincolo Grassi: Non due pasti "High Fat" nello stesso giorno
            def fat_limit_constraint(p, c):
                return not (check_kb(f"high_fat({p})") and check_kb(f"high_fat({c})"))
            
            problem.addConstraint(fat_limit_constraint, (p_var, c_var))


    pasti_principali = [f"{g}_{pt}" for g in giorni for pt in ["Pranzo", "Cena"]]
    pasti_principali_effettivi = [v for v in pasti_principali if v in variabili_create]
    
    if pasti_principali_effettivi:
        # Almeno un piatto Mediterraneo a settimana
        def general_mediterranean_constraint(*pasti):
            return any(check_kb(f"is_mediterranean({p})") for p in pasti)
        problem.addConstraint(general_mediterranean_constraint, pasti_principali_effettivi)

        # Vincoli Sportivi (Recupero o Performance)
        if user_data.get('sport'):
            def sport_requirements_constraint(*pasti):
                has_recovery = any(check_kb(f"is_muscle_recovery({p})") for p in pasti)
                has_peak = any(check_kb(f"is_peak_performance({p})") for p in pasti)
                has_athlete = any(check_kb(f"is_athlete_diet({p})") for p in pasti)
                return has_recovery or has_peak or has_athlete
            problem.addConstraint(sport_requirements_constraint, pasti_principali_effettivi)

        # BMI Basso (Weight Gainer)
        if user_data.get('bmi', 20) < 18.5:
            def weight_gainer_constraint(*pasti):
                return any(check_kb(f"is_weight_gainer({p})") for p in pasti)
            problem.addConstraint(weight_gainer_constraint, pasti_principali_effettivi)
        
        # Super Veggie (Solo se l'utente è vegetariano)
        if user_data.get('is_vegetarian'):
            def super_veggie_constraint(*pasti):
                return any(check_kb(f"is_super_veggie({p})") for p in pasti)
            problem.addConstraint(super_veggie_constraint, pasti_principali_effettivi)

    # --- RISOLUZIONE ---
    soluzione = problem.getSolution()
    
    if not soluzione:
        print(" :( Nessun menu trovato. Prova a ridurre i piatti preferiti o cambiare i dati.")
        return None

    # Formattazione output
    ricette_dict = {row['recipe_title']: row for row in tutte_le_ricette}
    
    menu_finale = {}
    for g in giorni:
        menu_finale[g] = {}
        for pt in pasti_tipo:
            titolo_scelto = soluzione[f"{g}_{pt}"]
            # Recuperiamo l'oggetto ricetta completo dal nostro lookup
            menu_finale[g][pt] = ricette_dict[titolo_scelto]
            
    return menu_finale