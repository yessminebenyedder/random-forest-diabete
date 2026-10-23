"""
===============================================================================
 Donnees : base du DIABETE (Pima Indians Diabetes, UCI / National Institute of
 Diabetes and Digestive and Kidney Diseases)
===============================================================================

 768 patientes (femmes d'au moins 21 ans, d'origine Pima) ; on veut predire
 si la patiente est DIABETIQUE (1) ou NON (0) a partir de 8 mesures medicales.

   data/diabete_train.csv : 614 patientes pour l'entrainement (80 %)
   data/diabete_test.csv  : 154 patientes pour le test (20 %)

 Colonnes : Label | Patient | 8 variables
   Label   : 1 = diabetique, 0 = non diabetique
   Patient : identifiant (texte, PAS une variable)

 Attention : dans la base d'origine, un 0 dans Glucose, Pression, EpaisseurPeau,
 Insuline ou IMC signifie en realite "mesure non disponible". Les arbres
 gerent cela naturellement (ils peuvent couper a "<= 0").
===============================================================================
"""

import csv
from pathlib import Path

DOSSIER = Path(__file__).parent / "data"
FICHIER_TRAIN = DOSSIER / "diabete_train.csv"
FICHIER_TEST = DOSSIER / "diabete_test.csv"

# Signification des variables (nom -> description)
DESCRIPTIONS = {
    "Grossesses":    "nombre de grossesses",
    "Glucose":       "taux de glucose dans le sang 2 h apres un test de charge (mg/dL)",
    "Pression":      "pression arterielle diastolique (mm Hg)",
    "EpaisseurPeau": "epaisseur du pli cutane du triceps (mm)",
    "Insuline":      "insuline dans le sang 2 h apres le test (mu U/mL)",
    "IMC":           "indice de masse corporelle (poids / taille^2)",
    "Heredite":      "score d'antecedents familiaux de diabete (plus grand = plus de risque)",
    "Age":           "age (annees)",
}
CLASSES = {0: "0 : non diabetique", 1: "1 : diabetique"}


def charger_csv(chemin):
    """
    Lit un CSV et retourne (X, y, ids, noms_features).
      X   : liste de lignes de nombres (les 8 variables)
      y   : liste de labels (0 ou 1)
      ids : identifiants des patientes
    """
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        lecteur = csv.reader(f)
        entete = next(lecteur)
        i_label, i_id = entete.index("Label"), entete.index("Patient")
        i_var = [i for i in range(len(entete)) if i not in (i_label, i_id)]
        noms = [entete[i] for i in i_var]
        X, y, ids = [], [], []
        for ligne in lecteur:
            if ligne:
                y.append(int(float(ligne[i_label])))
                ids.append(ligne[i_id])
                X.append([float(ligne[i]) for i in i_var])
    return X, y, ids, noms
