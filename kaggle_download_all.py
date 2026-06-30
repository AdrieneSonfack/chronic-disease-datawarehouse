"""
Telechargement des 6 datasets Kaggle
Projet : Data Warehouse - Maladies Chroniques (Cameroun)
Aligné avec les features du pipeline ML (train_models.py)

UTILISATION :
  $env:KAGGLE_API_TOKEN = "KGAT_6bec3a136540db3c563d2b045388edad"
  python kaggle_download_all.py
"""

import os
import subprocess
import pandas as pd

# Configuration
OUTPUT_DIR = "data/raw/kaggle"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Liste des 6 datasets (Alignés sur Diabète, HTA, IRC, Asthme + Coûts)
DATASETS = [
    {
        "slug"    : "mathchi/diabetes-data-set",
        "name"    : "Pima Indians Diabetes (NIDDK)",
        "maladie" : "Diabète type 2 (E11)",
        "licence" : "CC0",
        "lignes"  : "768",
        "colonnes_cles": ["Glucose (glycemie_g_l)", "BloodPressure (tension)", "BMI (imc)", "Age", "Outcome (rechute/diabete)"],
        "dim_cible": "DIM_PATIENT, DIM_MALADIE, FACT_CONSULTATIONS (Features: glycemie, tension, imc)"
    },
    {
        "slug"    : "nigoraxonnasimova/synthetic-diabetes-2-type-prediction-dataset",
        "name"    : "Synthetic Diabetes Type 2 Prediction",
        "maladie" : "Diabète type 2 (E11)",
        "licence" : "CC BY",
        "lignes"  : "172302",
        "colonnes_cles": ["Age", "Gender (sexe)", "BMI (imc)", "HbA1c_level", "blood_glucose_level (glycemie_g_l)"],
        "dim_cible": "DIM_PATIENT, FACT_CONSULTATIONS (Gros volume pour entraînement)"
    },
    {
        "slug"    : "miadul/hypertension-risk-prediction-dataset",
        "name"    : "Hypertension Risk Prediction Dataset",
        "maladie" : "Hypertension artérielle (I10)",
        "licence" : "A verifier",
        "lignes"  : "~10000",
        "colonnes_cles": ["Age", "BMI (imc)", "SystolicBP (tension_systolique)", "DiastolicBP (tension_diastolique)", "Cholesterol"],
        "dim_cible": "DIM_PATIENT, FACT_CONSULTATIONS (Features: tension_sys, tension_dia)"
    },
    {
        "slug"    : "mansoordaku/ckdisease",
        "name"    : "Chronic Kidney Disease Dataset (UCI)",
        "maladie" : "Insuffisance rénale chronique (N18)",
        "licence" : "CC BY 4.0",
        "lignes"  : "400",
        "colonnes_cles": ["age", "bp (tension)", "sc (créatinine)", "hemo (hémoglobine)", "bgr (glycemie)", "htn (htn_bool)"],
        "dim_cible": "DIM_PATIENT, DIM_MALADIE, FACT_CONSULTATIONS (Features biologiques)"
    },
    {
        "slug"    : "rabieelkharoua/asthma-disease-dataset",
        "name"    : "Asthma Disease Dataset",
        "maladie" : "Asthme / BPCO (J45/J44)",
        "licence" : "CC BY 4.0",
        "lignes"  : "~2500",
        "colonnes_cles": ["Age", "Gender (sexe)", "BMI (imc)", "Smoking (fumeur)", "LungFunctionFEV1", "Diagnosis"],
        "dim_cible": "DIM_PATIENT, DIM_MALADIE, FACT_CONSULTATIONS (Features: fumeur, imc)"
    },
    {
        "slug"    : "yashvikedia/health-insurance-cost-prediction",
        "name"    : "Health Insurance Cost Prediction",
        "maladie" : "Transverse (Coûts & Hospitalisations)",
        "licence" : "CC0",
        "lignes"  : "~1300",
        "colonnes_cles": ["age", "sex (sexe)", "bmi (imc)", "smoker (fumeur)", "charges (cout_fcfa)", "region"],
        "dim_cible": "FACT_CONSULTATIONS (Cible: cout_fcfa, Features: imc, fumeur, age, region)"
    },
]


def download_all():
    print("=" * 70)
    print("  TELECHARGEMENT DES DATASETS KAGGLE (Aligné ML & DWH)")
    print("=" * 70)

    results = []

    for i, ds in enumerate(DATASETS, 1):
        folder_name = ds["slug"].split("/")[1]
        dest = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(dest, exist_ok=True)

        print(f"\n[{i}/{len(DATASETS)}] {ds['name']}")
        print(f"  Maladie  : {ds['maladie']}")
        print(f"  Licence  : {ds['licence']}")
        print(f"  Slug     : {ds['slug']}")

        try:
            cmd = [
                "python", "-m", "kaggle",
                "datasets", "download",
                "-d", ds["slug"],
                "-p", dest,
                "--unzip"
            ]
            subprocess.run(cmd, check=True)
            status = "OK"
            print(f"  Statut   : OK - Telecharge dans {dest}")
        except subprocess.CalledProcessError as e:
            status = "ERREUR"
            print(f"  Statut   : ERREUR - {e}")

        results.append({
            "Dataset"       : ds["name"],
            "Maladie"       : ds["maladie"],
            "Licence"       : ds["licence"],
            "Lignes"        : ds["lignes"],
            "Colonnes cles" : ", ".join(ds["colonnes_cles"]),
            "Tables DWH"    : ds["dim_cible"],
            "Dossier local" : dest,
            "Statut"        : status
        })

    return results


def generate_inventory(results):
    inv_path = os.path.join(OUTPUT_DIR, "inventaire_sources_kaggle.csv")
    df = pd.DataFrame(results)
    df.to_csv(inv_path, index=False, encoding="utf-8-sig")
    print(f"\n{'='*70}")
    print(f"  Inventaire genere : {inv_path}")
    print(f"{'='*70}")
    print(df[["Dataset", "Maladie", "Licence", "Statut"]].to_string(index=False))


def quick_profile():
    print(f"\n{'='*70}")
    print("  PROFILING RAPIDE DES CSV TELECHARGES")
    print(f"{'='*70}")

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith(".csv") and "inventaire" not in f:
                path = os.path.join(root, f)
                try:
                    full = pd.read_csv(path)
                    size = os.path.getsize(path) / 1024
                    nulls = full.isnull().sum().sum()
                    print(f"\n  Fichier  : {f}")
                    print(f"  Lignes   : {len(full):,}  |  Colonnes : {len(full.columns)}")
                    print(f"  Taille   : {size:.1f} Ko  |  Valeurs nulles : {nulls}")
                    print(f"  Colonnes : {list(full.columns)}")
                except Exception as e:
                    print(f"  Impossible de lire {f} : {e}")


if __name__ == "__main__":
    results = download_all()
    generate_inventory(results)
    quick_profile()
    print("\nProchaine etape : charger ces CSV dans PostgreSQL (staging area)")
    print("Puis alimenter la vue vue_consultations_completes pour le pipeline ML.")