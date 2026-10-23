"""
===============================================================================
 Evaluation du Random Forest (from scratch) sur la base du diabete
===============================================================================
 Entraine sur data/diabete_train.csv, teste sur data/diabete_test.csv et affiche
 la matrice de confusion et les metriques.

 Execution : python evaluation.py
             python evaluation.py --arbres 50 --profondeur 6 --seed 1
===============================================================================
"""

import argparse
import time
from collections import Counter

from donnees import DESCRIPTIONS, FICHIER_TEST, FICHIER_TRAIN, charger_csv
from random_forest_treelib import RandomForest


def rapport(y_vrai, y_pred, titre):
    """Matrice de confusion + accuracy, sensibilite, specificite, precision, F1."""
    vp = sum(1 for v, p in zip(y_vrai, y_pred) if v == 1 and p == 1)
    fn = sum(1 for v, p in zip(y_vrai, y_pred) if v == 1 and p == 0)
    fp = sum(1 for v, p in zip(y_vrai, y_pred) if v == 0 and p == 1)
    vn = sum(1 for v, p in zip(y_vrai, y_pred) if v == 0 and p == 0)
    div = lambda a, b: a / b if b else 0.0
    sens, spec = div(vp, vp + fn), div(vn, vn + fp)
    prec = div(vp, vp + fp)
    print(f"\n--- {titre} ({len(y_vrai)} patientes) ---")
    print("                      predit 0   predit 1")
    print(f"  reel 0 (sain)       {vn:8d}   {fp:8d}")
    print(f"  reel 1 (diabete)    {fn:8d}   {vp:8d}")
    print(f"  Accuracy            : {div(vp + vn, len(y_vrai)):.3f}")
    print(f"  Sensibilite         : {sens:.3f}   (diabetiques bien detectees)")
    print(f"  Specificite         : {spec:.3f}   (non diabetiques bien ecartees)")
    print(f"  Precision           : {prec:.3f}")
    print(f"  F1                  : {div(2 * prec * sens, prec + sens):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arbres", type=int, default=25)
    parser.add_argument("--profondeur", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, _, noms = charger_csv(FICHIER_TRAIN)
    X_t, y_t, _, _ = charger_csv(FICHIER_TEST)
    print(f"Entrainement : {len(X)} patientes, classes {dict(Counter(y))}")
    print(f"Test         : {len(X_t)} patientes, classes {dict(Counter(y_t))}")

    debut = time.time()
    foret = RandomForest(n_arbres=args.arbres, max_depth=args.profondeur, seed=args.seed)
    foret.fit(X, y, noms_features=noms)
    print(f"\nForet de {args.arbres} arbres entrainee en {time.time() - debut:.1f} s")

    rapport(y, foret.predict(X), "ENTRAINEMENT")
    rapport(y_t, foret.predict(X_t), "TEST (resultat qui compte)")

    compteur = Counter()
    for arbre in foret.arbres:
        for noeud in arbre.arbre.all_nodes_itr():
            if noeud.data["valeur"] is None:
                compteur[noms[noeud.data["feature"]]] += 1
    print("\n=== Variables les plus utilisees comme coupure ===")
    for nom, nb in compteur.most_common():
        print(f"  {nom:<14} {nb:4d} coupures   {DESCRIPTIONS.get(nom, '')}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        sk = RandomForestClassifier(n_estimators=args.arbres, random_state=args.seed).fit(X, y)
        rapport(y_t, list(sk.predict(X_t)), "TEST - scikit-learn (reference)")
    except ImportError:
        print("\n(scikit-learn non installe : comparaison ignoree)")


if __name__ == "__main__":
    main()
