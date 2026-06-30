"""
=============================================================
  ÉTAPE 1 — SOURCES : Génération Faker Python
  Données simulées contextualisées Cameroun
  PROJET : Data Warehouse - Maladies Chroniques
  ALIGNEMENT : Features exactes de train_models.py (vue_consultations_completes)
=============================================================

PRÉREQUIS :
  pip install faker pandas numpy

GÉNÈRE :
  data/raw/faker/hopitaux.csv         (~80 lignes)
  data/raw/faker/medecins.csv         (~200 lignes)
  data/raw/faker/patients.csv         (~5 000 lignes)
  data/raw/faker/maladies.csv         (référentiel fixe)
  data/raw/faker/medicaments.csv      (référentiel fixe)
  data/raw/faker/consultations.csv    (~15 000 lignes, dénormalisé pour ML)

UTILISATION :
  python faker_cameroun.py
=============================================================
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from faker import Faker

fake = Faker("fr_FR")
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data/raw/faker"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DONNÉES DE RÉFÉRENCE CAMEROUN
# ─────────────────────────────────────────────

REGIONS = [
    "Adamaoua", "Centre", "Est", "Extrême-Nord",
    "Littoral", "Nord", "Nord-Ouest", "Ouest",
    "Sud", "Sud-Ouest"
]

VILLES_PAR_REGION = {
    "Adamaoua"     : ["Ngaoundéré", "Meiganga", "Tibati", "Banyo"],
    "Centre"       : ["Yaoundé", "Bafia", "Mbalmayo", "Obala", "Nanga Eboko"],
    "Est"          : ["Bertoua", "Abong-Mbang", "Batouri", "Yokadouma"],
    "Extrême-Nord" : ["Maroua", "Kousseri", "Mora", "Yagoua", "Kaélé"],
    "Littoral"     : ["Douala", "Edéa", "Nkongsamba", "Loum"],
    "Nord"         : ["Garoua", "Guider", "Figuil", "Poli"],
    "Nord-Ouest"   : ["Bamenda", "Kumbo", "Wum", "Nkambe"],
    "Ouest"        : ["Bafoussam", "Dschang", "Foumban", "Bangangté"],
    "Sud"          : ["Ebolowa", "Kribi", "Sangmélima", "Ambam"],
    "Sud-Ouest"    : ["Buea", "Limbe", "Kumba", "Mamfe"],
}

PRENOMS_M = [
    "Jean-Pierre", "Paul", "Emmanuel", "François", "Joseph", "Samuel",
    "Alain", "Hervé", "Patrick", "David", "Rodrigue", "Fabrice",
    "Mohamadou", "Hamadou", "Abdoulaye", "Ibrahim", "Oumarou", "Saidou"
]
PRENOMS_F = [
    "Marie", "Célestine", "Agnès", "Marguerite", "Véronique", "Christine",
    "Fatima", "Aïcha", "Mariama", "Hawa", "Aminatou", "Ramatou",
    "Yvonne", "Claudine", "Solange", "Georgette", "Pauline", "Thérèse"
]
NOMS_FAMILLE = [
    "Mvondo", "Essomba", "Nkomo", "Biya", "Fotso", "Kamdem", "Nganou",
    "Tchoukoua", "Mbassi", "Ondoa", "Abega", "Nlend", "Owono", "Ayissi",
    "Alhadji", "Moussa", "Oumarou", "Saidou", "Bouba", "Maigari",
    "Djibrilla", "Hamidou", "Yerima", "Mbida", "Zogo", "Ateba",
    "Nkoulou", "Eyenga", "Belinga", "Onana", "Ngono", "Bikié"
]

# Maladies chroniques ciblées (Alignées sur generate_synthetic_dataset.sql)
MALADIES = [
    {"id_maladie": "MAL001", "code_cim10": "E11",   "libelle": "Diabète de type 2", "categorie": "Endocrinologie"},
    {"id_maladie": "MAL002", "code_cim10": "I10",   "libelle": "Hypertension artérielle", "categorie": "Cardiologie"},
    {"id_maladie": "MAL003", "code_cim10": "N18",   "libelle": "Insuffisance rénale chronique", "categorie": "Néphrologie"},
    {"id_maladie": "MAL004", "code_cim10": "J45",   "libelle": "Asthme", "categorie": "Pneumologie"},
    {"id_maladie": "MAL005", "code_cim10": "J44",   "libelle": "BPCO", "categorie": "Pneumologie"},
]

MEDICAMENTS = [
    {"id_medicament": "MED001", "dci": "Metformine",       "nom_commercial": "Glucophage",   "indication": "Diabète type 2"},
    {"id_medicament": "MED002", "dci": "Glibenclamide",    "nom_commercial": "Daonil",       "indication": "Diabète type 2"},
    {"id_medicament": "MED003", "dci": "Amlodipine",       "nom_commercial": "Amlor",        "indication": "Hypertension"},
    {"id_medicament": "MED004", "dci": "Énalapril",        "nom_commercial": "Renitec",      "indication": "Hypertension"},
    {"id_medicament": "MED005", "dci": "Salbutamol",       "nom_commercial": "Ventoline",    "indication": "Asthme / BPCO"},
    {"id_medicament": "MED006", "dci": "Furosémide",       "nom_commercial": "Lasilix",      "indication": "Insuffisance rénale/cardiaque"},
]

SPECIALITES = ["Médecine générale", "Cardiologie", "Endocrinologie", "Néphrologie", "Pneumologie"]
TYPES_HOPITAL = ["Hôpital Général", "Hôpital de District", "Centre Médical d'Arrondissement", "Clinique Privée"]
NIVEAUX_SOIN = ["Niveau 1 (Soins primaires)", "Niveau 2 (Soins secondaires)", "Niveau 3 (Soins tertiaires)"]

# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────

def rand_region(): return random.choice(REGIONS)
def rand_ville(region): return random.choice(VILLES_PAR_REGION[region])

def rand_telephone():
    p = random.choice(["6", "2"])
    sec = random.choice(["50","51","70","71","22","23"]) if p == "6" else f"{random.randint(22,29)}"
    return f"+237 {p}{sec} {random.randint(10,99)} {random.randint(10,99)} {random.randint(10,99)}"

def rand_date(start_year=1940, end_year=2005):
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def rand_date_consultation():
    start = date(2020, 1, 1)
    end   = date(2024, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def get_categorie_imc(imc):
    if imc < 18.5: return "Maigreur"
    elif imc < 25: return "Normal"
    elif imc < 30: return "Surpoids"
    else: return "Obésité"

def get_saison_cameroun(mois):
    return "Saison des pluies" if 5 <= mois <= 10 else "Saison sèche"

def get_categorie_tension(sys, dia):
    if sys >= 140 or dia >= 90: return "HTA"
    elif sys >= 120 or dia >= 80: return "Élevée"
    else: return "Normale"

def get_categorie_glycemie(gly):
    if gly >= 7.0: return "Diabète"
    elif gly >= 5.6: return "Prédiabète"
    else: return "Normale"

def get_niveau_experience(annee_diplome):
    age_exp = 2024 - annee_diplome
    if age_exp < 5: return "Junior"
    elif age_exp < 15: return "Confirmé"
    else: return "Senior"

# ─────────────────────────────────────────────
# GÉNÉRATION : HÔPITAUX
# ─────────────────────────────────────────────
def generate_hopitaux(n=80):
    print(f"[1/6] Génération de {n} hôpitaux...")
    rows = []
    hop_id = 1
    for region in REGIONS:
        nb = max(4, n // len(REGIONS))
        for _ in range(nb):
            ville = rand_ville(region)
            type_h = random.choice(TYPES_HOPITAL)
            rows.append({
                "id_hopital"   : f"HOP{hop_id:04d}",
                "nom_hopital"  : f"{type_h} de {ville}",
                "type_hopital" : type_h,
                "ville"        : ville,
                "region"       : region,
                "niveau_soin"  : random.choice(NIVEAUX_SOIN),
                "secteur"      : random.choice(["Public", "Public", "Public", "Privé"]),
                "capacite_lits": random.choice([20, 50, 100, 200, 300]),
            })
            hop_id += 1
    df = pd.DataFrame(rows[:n])
    path = os.path.join(OUTPUT_DIR, "hopitaux.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✓ {len(df)} hôpitaux → {path}")
    return df

# ─────────────────────────────────────────────
# GÉNÉRATION : MÉDECINS
# ─────────────────────────────────────────────
def generate_medecins(hopitaux_df, n=200):
    print(f"[2/6] Génération de {n} médecins...")
    rows = []
    hop_ids = hopitaux_df["id_hopital"].tolist()
    for i in range(1, n + 1):
        sexe = random.choice(["M", "F"])
        prenom = random.choice(PRENOMS_M if sexe == "M" else PRENOMS_F)
        nom    = random.choice(NOMS_FAMILLE)
        annee  = random.randint(1990, 2022)
        rows.append({
            "id_medecin"    : f"MEDECIN{i:04d}",
            "nom"           : nom,
            "prenom"        : prenom,
            "sexe"          : sexe,
            "specialite"    : random.choice(SPECIALITES),
            "id_hopital"    : random.choice(hop_ids),
            "annee_diplome" : annee,
            "niveau_experience": get_niveau_experience(annee),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "medecins.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✓ {len(df)} médecins → {path}")
    return df

# ─────────────────────────────────────────────
# GÉNÉRATION : PATIENTS
# ─────────────────────────────────────────────
def generate_patients(n=5000):
    print(f"[3/6] Génération de {n} patients...")
    rows = []
    for i in range(1, n + 1):
        sexe   = random.choice(["M", "F"])
        prenom = random.choice(PRENOMS_M if sexe == "M" else PRENOMS_F)
        nom    = random.choice(NOMS_FAMILLE)
        region = rand_region()
        ville  = rand_ville(region)
        ddn    = rand_date(1940, 2005)
        age    = (date.today() - ddn).days // 365
        
        imc = round(random.gauss(26.5, 4.5), 1)
        imc = max(16.0, min(imc, 45.0))

        rows.append({
            "id_patient"      : f"PAT{i:06d}",
            "nom"             : nom,
            "prenom"          : prenom,
            "sexe"            : sexe,
            "date_naissance"  : ddn.strftime("%Y-%m-%d"),
            "age"             : age,
            "patient_region"  : region, # Aligné avec train_models.py
            "ville"           : ville,
            "imc"             : imc,
            "categorie_imc"   : get_categorie_imc(imc),
            "fumeur"          : random.random() < (0.25 if sexe == "M" else 0.05),
            "couverture_maladie": random.choice(["CNPS", "Mutuelle", "Aucune", "Aucune", "Privée"]),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "patients.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✓ {len(df)} patients → {path}")
    return df

# ─────────────────────────────────────────────
# GÉNÉRATION : CONSULTATIONS (Dénormalisé pour ML)
# ─────────────────────────────────────────────
def generate_consultations(patients_df, medecins_df, hopitaux_df, maladies_df, medicaments_df, n=15000):
    print(f"[6/6] Génération de {n} consultations (Features ML alignées)...")

    # Index pour jointure rapide
    pat_dict = patients_df.set_index("id_patient").to_dict("index")
    med_dict = medecins_df.set_index("id_medecin").to_dict("index")
    hop_dict = hopitaux_df.set_index("id_hopital").to_dict("index")
    
    mal_ids = maladies_df["id_maladie"].tolist()
    med_diab = medicaments_df[medicaments_df["indication"].str.contains("Diabète")]["id_medicament"].tolist()
    med_hta  = medicaments_df[medicaments_df["indication"].str.contains("Hypertension")]["id_medicament"].tolist()
    med_all  = medicaments_df["id_medicament"].tolist()
    med_map = {"MAL001": med_diab, "MAL002": med_hta}

    rows = []
    for i in range(1, n + 1):
        pat_id  = random.choice(patients_df["id_patient"].tolist())
        med_id  = random.choice(medecins_df["id_medecin"].tolist())
        hop_id  = random.choice(hopitaux_df["id_hopital"].tolist())
        mal_id  = random.choice(mal_ids)
        
        pat = pat_dict[pat_id]
        med = med_dict[med_id]
        hop = hop_dict[hop_id]
        
        date_c  = rand_date_consultation()
        mois = date_c.month
        
        # Signes vitaux cohérents avec la pathologie
        is_hta = mal_id == "MAL002"
        is_diab = mal_id == "MAL001"

        tension_sys = int(random.gauss(145 if is_hta else 120, 15))
        tension_dia = int(random.gauss(92  if is_hta else 78, 10))
        glycemie    = round(random.gauss(7.8 if is_diab else 5.2, 1.5), 1)
        poids       = round(random.gauss(72, 12), 1)
        temperature = round(random.gauss(37.1, 0.4), 1)
        fc          = int(random.gauss(82, 12))
        
        # Logique de rechute (Cible ML)
        prob_rechute = 0.1
        if tension_sys > 140: prob_rechute += 0.2
        if glycemie > 7.0: prob_rechute += 0.3
        if pat["age"] > 60: prob_rechute += 0.1
        rechute = random.random() < prob_rechute
        
        # Hospitalisation et coûts
        hospitalisation = random.random() < (0.3 if rechute else 0.05)
        duree_sejour = random.randint(2, 15) if hospitalisation else 0
        
        cout_base = random.choice([5000, 10000, 15000])
        if hospitalisation: cout_base += duree_sejour * 25000
        if hop["secteur"] == "Privé": cout_base *= 1.5
        cout_fcfa = int(cout_base)

        rows.append({
            # Identifiants
            "id_consultation" : f"CONS{i:07d}",
            "id_patient"      : pat_id,
            "id_medecin"      : med_id,
            "id_hopital"      : hop_id,
            "id_maladie"      : mal_id,
            "date_consultation": date_c.strftime("%Y-%m-%d"),
            
            # Features Patient (Dénormalisées)
            "age"             : pat["age"],
            "sexe"            : pat["sexe"],
            "patient_region"  : pat["patient_region"],
            "categorie_imc"   : pat["categorie_imc"],
            "fumeur"          : pat["fumeur"],
            "couverture_maladie": pat["couverture_maladie"],
            
            # Features Médecin/Hôpital (Dénormalisées)
            "medecin_specialite": med["specialite"],
            "niveau_experience" : med["niveau_experience"],
            "type_hopital"    : hop["type_hopital"],
            "niveau_soin"     : hop["niveau_soin"],
            "secteur_hopital" : hop["secteur"],
            
            # Signes vitaux
            "tension_systolique": max(80, min(tension_sys, 220)),
            "tension_diastolique": max(50, min(tension_dia, 140)),
            "glycemie_g_l"    : round(max(3.0, min(glycemie, 25.0)), 1),
            "poids_kg"        : round(max(30.0, min(poids, 130.0)), 1),
            "temperature_c"   : round(max(35.5, min(temperature, 41.0)), 1),
            "frequence_cardiaque": max(40, min(fc, 160)),
            "fievre"          : temperature >= 38.0,
            
            # Catégories dérivées (pour ML)
            "categorie_tension": get_categorie_tension(tension_sys, tension_dia),
            "categorie_glycemie": get_categorie_glycemie(glycemie),
            
            # Contexte temporel
            "saison_cameroun" : get_saison_cameroun(mois),
            "est_weekend"     : date_c.weekday() >= 5,
            "est_ferie"       : date_c.month == 1 and date_c.day == 1, # Simplifié
            
            # Cibles et Coûts
            "cout_fcfa"       : cout_fcfa,
            "hospitalisation" : hospitalisation,
            "duree_sejour_j"  : duree_sejour,
            "rechute"         : rechute,
            "observance_estime_pct": random.randint(40, 100) if not rechute else random.randint(20, 70),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "consultations.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   ✓ {len(df)} consultations → {path}")
    return df

# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  FAKER CAMEROUN — Génération alignée avec train_models.py")
    print("=" * 70 + "\n")

    hopitaux    = generate_hopitaux(n=80)
    medecins    = generate_medecins(hopitaux, n=200)
    patients    = generate_patients(n=5000)
    maladies    = pd.DataFrame(MALADIES)
    medicaments = pd.DataFrame(MEDICAMENTS)
    
    consultations = generate_consultations(
        patients, medecins, hopitaux,
        maladies, medicaments, n=15000
    )

    print(f"\n{'='*70}")
    print("  ✅ GÉNÉRATION TERMINÉE")
    print(f"{'='*70}")
    print(f"  Les fichiers CSV sont prêts à être chargés dans le DWH.")
    print(f"  La table de faits 'consultations' contient déjà les features")
    print(f"  dénormalisées attendues par la vue 'vue_consultations_completes'.")
    print(f"{'='*70}")