"""
===============================================================================
 Random Forest interactif - prediction du DIABETE
===============================================================================

 Base utilisee (dossier data/) : 768 patientes, 8 mesures medicales chacune
   Grossesses, Glucose, Pression, EpaisseurPeau, Insuline, IMC, Heredite, Age
   Label : 1 = diabetique, 0 = non diabetique  (details dans donnees.py)

 Le programme demande :
   1. le NOMBRE D'ARBRES      (choisi par l'utilisateur)
   2. la profondeur maximale  (Entree = 3)

 Puis il :
   - construit la foret en AFFICHANT chaque arbre pas a pas
     (bootstrap, variables tirees, Gini de chaque coupure, arbre final) ;
   - DESSINE les arbres dans une page web (foret_resultat.html, ouverte
     automatiquement dans le navigateur) ;
   - demande les mesures d'une NOUVELLE patiente (saisies au clavier) et PREDIT
     son cas : chemin suivi dans chaque arbre + vote final ; on peut recommencer.

 Execution : python foret_interactive.py
===============================================================================
"""

from collections import Counter

from donnees import CLASSES, DESCRIPTIONS, DOSSIER, FICHIER_TEST, FICHIER_TRAIN, charger_csv
from random_forest_treelib import RandomForest
from visualisation import generer_html

FICHIER_HTML = DOSSIER.parent / "foret_resultat.html"   # page de visualisation
PROFONDEUR_DEFAUT = 3


# =============================================================================
# Saisies avec controle (on redemande tant que la reponse n'est pas valide)
# =============================================================================

def demander_entier(question, minimum, maximum=None, defaut=None):
    """Demande un entier ; Entree seule -> valeur par defaut (si fournie)."""
    while True:
        texte = input(question).strip()
        if texte == "" and defaut is not None:
            return defaut
        try:
            valeur = int(texte)
            if valeur >= minimum and (maximum is None or valeur <= maximum):
                return valeur
        except ValueError:
            pass
        borne = f"entre {minimum} et {maximum}" if maximum is not None else f">= {minimum}"
        print(f"  Reponse invalide : tapez un nombre entier {borne}.")


def demander_nombre(question):
    """Demande un nombre (entier ou decimal, virgule acceptee)."""
    while True:
        try:
            return float(input(question).strip().replace(",", "."))
        except ValueError:
            print("  Reponse invalide : tapez un nombre (ex. 28 ou 33.6).")


def demander_oui_non(question):
    while True:
        r = input(question + " (o/n) : ").strip().lower()
        if r in ("o", "oui", "y", "yes"):
            return True
        if r in ("n", "non", "no"):
            return False


def saisir_patiente(noms, X):
    """Demande les 8 valeurs d'une nouvelle patiente."""
    print("\nEntrez les mesures de la nouvelle patiente :")
    x = []
    for j, nom in enumerate(noms):
        mini, maxi = min(l[j] for l in X), max(l[j] for l in X)
        x.append(demander_nombre(f"  {nom} - {DESCRIPTIONS[nom]}\n    (base : de {mini:g} a {maxi:g}) : "))
    return x


# =============================================================================
# Prediction d'une patiente, avec explication
# =============================================================================

def predire_et_expliquer(foret, x):
    """Fait passer x dans chaque arbre, affiche les chemins puis le vote."""
    print("\n" + "=" * 78)
    print(" PREDICTION : chemin suivi dans chaque arbre")
    print("=" * 78)
    votes = []
    for i, arbre in enumerate(foret.arbres, start=1):
        classe, etapes = arbre.chemin(x)
        votes.append(classe)
        print(f"Arbre {i} :")
        for e in etapes:
            print(f"    {e}")
        print(f"    => vote : {classe}   ({CLASSES[classe]})")
    compte = Counter(votes)
    gagnant = compte.most_common(1)[0][0]
    print("\nETAPE 5 - Vote majoritaire :")
    for classe, nb in compte.most_common():
        print(f"    {CLASSES[classe]} : {nb} arbre(s) sur {len(votes)}  ({100 * nb / len(votes):.0f} %)")
    print(f"\n>>> PREDICTION FINALE : {CLASSES[gagnant]}")
    return gagnant, votes


def predire_et_visualiser(foret, noms, x, vraie_classe=None):
    """Predit x (explication dans le terminal) puis ecrit et ouvre la page web."""
    gagnant, votes = predire_et_expliquer(foret, x)
    prediction = {"x": x, "votes": votes, "final": gagnant}
    if vraie_classe is not None:
        verdict = "CORRECTE" if gagnant == vraie_classe else "FAUSSE"
        print(f"Vraie classe (connue dans la base) : {CLASSES[vraie_classe]}  -> prediction {verdict}")
        prediction["vraie_classe"] = vraie_classe
    chemin = generer_html(foret, noms, prediction, fichier=FICHIER_HTML,
                          titre="Random Forest - prediction du diabete", noms_classes=CLASSES)
    print(f"\nVisualisation des arbres enregistree et ouverte dans le navigateur :\n  {chemin}")


# =============================================================================
# Programme principal
# =============================================================================

def main():
    print("=== Random Forest interactif - prediction du diabete ===\n")
    X, y, _, noms = charger_csv(FICHIER_TRAIN)
    X_test, y_test, ids_test, _ = charger_csv(FICHIER_TEST)
    print(f"Entrainement : {len(X)} patientes, classes {dict(Counter(y))}")
    print(f"Test         : {len(X_test)} patientes, classes {dict(Counter(y_test))}")
    print("Label 1 = diabetique, 0 = non diabetique.")
    print("\nVariables utilisees :")
    for nom in noms:
        print(f"  {nom:<14} {DESCRIPTIONS[nom]}")

    n_arbres = demander_entier("\nNombre d'arbres de la foret : ", 1)
    prof = demander_entier(f"Profondeur maximale (Entree = {PROFONDEUR_DEFAUT}) : ", 1,
                           defaut=PROFONDEUR_DEFAUT)

    print(f"\nConstruction de {n_arbres} arbre(s), profondeur max = {prof}, "
          f"k = {max(1, int(len(noms) ** 0.5))} variable(s) tiree(s) par noeud.")
    foret = RandomForest(n_arbres=n_arbres, max_depth=prof, verbose=True)
    foret.fit(X, y, noms_features=noms)
    print("\n" + "=" * 78)
    print(f" Foret terminee : {n_arbres} arbre(s) construit(s)")
    print("=" * 78)

    # ----- Prediction d'une NOUVELLE patiente saisie par l'utilisateur ---------------
    print("\nSaisissez maintenant les mesures d'une NOUVELLE patiente pour predire son cas.")
    while True:
        predire_et_visualiser(foret, noms, saisir_patiente(noms, X))
        if not demander_oui_non("\nPredire une autre nouvelle patiente ?"):
            break

    print("\nFin du programme.")


if __name__ == "__main__":
    main()