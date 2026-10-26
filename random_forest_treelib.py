"""
===============================================================================
 Random Forest "from scratch" avec treelib
===============================================================================

 Objectif
 --------
 Implementer l'algorithme Random Forest (Breiman, 2001) sans utiliser
 RandomForestClassifier de scikit-learn. Tout l'apprentissage est ecrit a la
 main ; seule la STRUCTURE d'arbre (noeuds relies entre eux + affichage)
 vient de la bibliotheque treelib.

 Etapes de l'algorithme et fonctions correspondantes
 ---------------------------------------------------
   1. Bootstrap : tirer n lignes AVEC remise pour chaque arbre
        -> RandomForest.fit()
   2. Tirer k variables au hasard a CHAQUE noeud (k = racine de p par defaut)
        -> ArbreDecision._construire()
   3. Chercher la meilleure coupure parmi ces k variables (critere de Gini)
        -> gini(), gini_pondere(), meilleur_split()
   4. Faire pousser l'arbre jusqu'a des feuilles pures (pas d'elagage)
        -> ArbreDecision._construire()  (recursif)
   5. Predire par vote majoritaire des arbres
        -> RandomForest.predict()

 Notations utilisees dans les commentaires de complexite
 -------------------------------------------------------
   n = nombre d'exemples (lignes)      p = nombre de variables (colonnes)
   k = variables tirees par noeud      B = nombre d'arbres (n_arbres)
   h = profondeur d'un arbre           (h ~ log2(n) si l'arbre est equilibre,
                                        h ~ n dans le pire cas)

 Installation : pip install -r requirements.txt   (numpy, treelib)
 Execution    : python random_forest_treelib.py
 Optionnel    : scikit-learn (comparaison) et matplotlib (graphique) ;
                le programme fonctionne sans eux.
===============================================================================
"""

from collections import Counter   # dictionnaire de comptage (bibliotheque standard)

import numpy as np                 # generateur aleatoire (bootstrap, tirage des variables)
from treelib import Tree           # structure d'arbre : noeuds + liens parent/enfants


# =============================================================================
# 1. FONCTIONS DE CALCUL (critere CART)
# =============================================================================

def gini(y):
    """
    Impurete de Gini d'un noeud :  G = 1 - somme_k (p_k)^2
    ou p_k est la proportion de la classe k dans le noeud.

    G = 0   -> noeud pur (une seule classe)
    G = 0.5 -> melange 50/50 de deux classes (impurete maximale a 2 classes)

    Exemple : y = ["Oui", "Oui", "Oui", "Non"]
              G = 1 - (0.75^2 + 0.25^2) = 0.375

    Complexite : temps O(n) (un passage pour compter), espace O(c)
                 avec c = nombre de classes distinctes.
    """
    n = len(y)
    if n == 0:                      # noeud vide : impurete nulle par convention
        return 0.0
    # Counter(y) compte les occurrences : {"Oui": 3, "Non": 1}
    return 1.0 - sum((c / n) ** 2 for c in Counter(y).values())


def gini_pondere(y_gauche, y_droite):
    """
    Impurete d'une coupure = moyenne des Gini des deux enfants,
    ponderee par leur taille :

        G_split = (n_g / n) * G(gauche) + (n_d / n) * G(droite)

    C'est cette valeur que l'on cherche a MINIMISER.
    Complexite : temps O(n), espace O(c).
    """
    n = len(y_gauche) + len(y_droite)
    return (len(y_gauche) / n) * gini(y_gauche) + (len(y_droite) / n) * gini(y_droite)


def meilleur_split(X, y, indices_features):
    """
    Cherche la meilleure coupure "X[j] <= seuil" parmi les variables
    d'indices `indices_features` (les k variables tirees au hasard).

    Methode (version rapide, par balayage) :
      - pour chaque variable j, on TRIE les lignes selon X[.][j] (une seule fois) ;
      - on deplace ensuite le seuil de gauche a droite en transferant une ligne
        a la fois du cote droit vers le cote gauche, en mettant a jour les
        COMPTEURS de classes -> le Gini de chaque coupure se calcule en O(c)
        au lieu de O(n).
      - on ne teste un seuil qu'entre deux valeurs DISTINCTES consecutives
        (le seuil est la plus petite des deux, comme avant : "<= seuil").

    Le resultat est identique a la version naive (memes coupures, memes scores),
    seule la vitesse change.

    Retourne (j, seuil, score) ; j vaut None si aucune coupure n'est possible
    (toutes les valeurs identiques -> un des deux cotes serait vide).

    Complexite (n lignes dans le noeud, c classes) :
      temps O(k * (n log n + n * c))   (tri + balayage)   au lieu de O(k * n^2)
      espace O(n)
    """
    n = len(y)
    total = Counter(y)                       # effectifs de chaque classe dans le noeud
    meilleur_j, meilleur_seuil, meilleur_score = None, None, float("inf")
    for j in indices_features:
        ordre = sorted(range(n), key=lambda i: X[i][j])   # lignes triees selon la variable j
        gauche = Counter()                                # effectifs a gauche du seuil
        droite = Counter(total)                           # effectifs a droite du seuil
        for pos in range(n - 1):
            i = ordre[pos]
            gauche[y[i]] += 1                # la ligne i passe de droite a gauche
            droite[y[i]] -= 1
            v, v_suivant = X[i][j], X[ordre[pos + 1]][j]
            if v == v_suivant:               # meme valeur : pas de seuil possible ici
                continue
            n_g, n_d = pos + 1, n - pos - 1
            g_g = 1.0 - sum((c / n_g) ** 2 for c in gauche.values())
            g_d = 1.0 - sum((c / n_d) ** 2 for c in droite.values())
            score = (n_g / n) * g_g + (n_d / n) * g_d
            if score < meilleur_score:       # strictement inferieur : 1er trouve gagne
                meilleur_j, meilleur_seuil, meilleur_score = j, v, score
    return meilleur_j, meilleur_seuil, meilleur_score


# =============================================================================
# 2. ARBRE DE DECISION stocke dans un treelib.Tree
# =============================================================================

class ArbreDecision:
    """
    Arbre de decision CART construit recursivement.

    Il est stocke dans un objet treelib.Tree. Chaque noeud treelib contient :
      - identifier : un entier unique (1, 2, 3, ... dans l'ordre de creation)
      - tag        : le texte affiche, ex. "Age <= 25 ?" ou "=> Oui {'Oui': 3}"
      - data       : un dictionnaire Python
            noeud interne : {"feature": j, "seuil": s, "valeur": None,
                             "gauche": id_enfant_gauche, "droite": id_enfant_droit}
            feuille       : {"valeur": classe_predite}
        La convention est : valeur != None  <=>  le noeud est une feuille.
    """

    def __init__(self, n_features=None, rng=None, noms_features=None, verbose=False):
        self.verbose = verbose              # True : explique la construction pas a pas
        self.n_features = n_features        # k : variables tirees par noeud (None = toutes)
        # Generateur aleatoire : fourni par la foret (partage) ou cree ici
        self.rng = rng if rng is not None else np.random.default_rng()
        self.noms_features = noms_features  # noms des colonnes, pour l'affichage
        self.arbre = Tree()                 # la structure treelib qui contient les noeuds
        self._compteur = 0                  # dernier identifiant de noeud attribue

    # ------------------------------------------------------------------ #
    #  Entrainement                                                       #
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        Construit l'arbre a partir des donnees (X, y).

        Complexite : temps O(k * n^2) si l'arbre est equilibre,
                     O(k * n^3) dans le pire cas (arbre en "peigne").
                     espace : O(n) noeuds dans l'arbre final.
        """
        # Copie en listes Python (accepte aussi bien des listes que des tableaux numpy)
        X = [list(x) for x in X]
        y = list(y)
        if self.noms_features is None:               # noms par defaut : X0, X1, ...
            self.noms_features = [f"X{j}" for j in range(len(X[0]))]
        self.arbre = Tree()                          # on repart d'un arbre vide
        self._compteur = 0
        self._construire(X, y, profondeur=0, parent=None, cote="")
        return self                                  # permet foret.fit(...).predict(...)

    def _nouvel_id(self):
        """Retourne un nouvel identifiant de noeud unique (1, 2, 3, ...). O(1)."""
        self._compteur += 1
        return self._compteur

    def _construire(self, X, y, profondeur, parent, cote):
        """
        Construction RECURSIVE (parcours en profondeur, gauche d'abord).

        1. Si le noeud est PUR (Gini = 0 : une seule classe) -> on cree une FEUILLE.
        2. Sinon on tire k variables au hasard et on cherche la meilleure coupure.
        3. Si aucune coupure n'existe -> FEUILLE.
        4. Sinon on cree un NOEUD INTERNE, on separe les lignes et on
           rappelle _construire() sur chaque moitie.

        Retourne l'identifiant du noeud cree.
        Espace : la pile d'appels contient au plus h appels ; chacun garde
        ses listes X, y -> O(n * p) si equilibre, O(n^2 * p) dans le pire cas.
        """
        nid = self._nouvel_id()
        prefixe = f"[{cote}] " if cote else ""         # "[oui] " ou "[non] " pour l'affichage
        majoritaire = Counter(y).most_common(1)[0][0]  # classe la plus frequente du noeud

        # ---- Criteres d'arret ------------------------------------------------
        pur = len(set(y)) == 1
        feature, seuil = None, None
        marge = "    " * profondeur                     # indentation des messages (mode detaille)
        if self.verbose:
            print(f"{marge}Noeud #{nid} {prefixe}(profondeur {profondeur}) : {len(y)} lignes, "
                  f"classes {dict(Counter(y))}, Gini = {gini(y):.3f}")
            if pur:
                print(f"{marge}  -> noeud PUR : on s'arrete")
        if not pur:
            p = len(X[0])
            k = p if self.n_features is None else min(self.n_features, p)
            # ETAPE 2 : k indices de variables tires SANS remise, a chaque noeud
            indices = self.rng.choice(p, size=k, replace=False)
            if self.verbose:
                print(f"{marge}  ETAPE 2 - {k} variable(s) tiree(s) au hasard parmi {p} :")
                for j in indices:                       # meilleure coupure de CHAQUE variable tiree
                    fj, sj, scj = meilleur_split(X, y, [j])
                    if fj is None:
                        print(f"{marge}    - {self.noms_features[j]:<24} aucune coupure possible (valeurs identiques)")
                    else:
                        print(f"{marge}    - {self.noms_features[j]:<24} meilleure coupure : <= {sj:<10.4g} Gini pondere = {scj:.4f}")
            # ETAPE 3 : meilleure coupure parmi ces k variables
            feature, seuil, score = meilleur_split(X, y, indices)
            if self.verbose and feature is not None:
                print(f"{marge}  ETAPE 3 - coupure retenue (Gini pondere minimal = {score:.4f}) : "
                      f"{self.noms_features[feature]} <= {seuil:.4g}")
            elif self.verbose:
                print(f"{marge}  ETAPE 3 - aucune coupure possible parmi ces variables -> feuille")

        # ---- Cas 1 : FEUILLE ---------------------------------------------------
        if feature is None:
            if self.verbose:
                print(f"{marge}  => FEUILLE : classe predite = {majoritaire}")
            self.arbre.create_node(
                tag=f"{prefixe}=> {majoritaire}  {dict(Counter(y))}",  # ex. "[oui] => Non  {'Non': 2}"
                identifier=nid, parent=parent,
                data={"valeur": majoritaire, "n": len(y), "classes": dict(Counter(y)), "gini": gini(y)},
            )
            return nid

        # ---- Cas 2 : NOEUD INTERNE ---------------------------------------------
        data = {"feature": feature, "seuil": seuil, "valeur": None,
                "n": len(y), "classes": dict(Counter(y)), "gini": gini(y)}
        self.arbre.create_node(
            tag=f"{prefixe}{self.noms_features[feature]} <= {seuil} ?",
            identifier=nid, parent=parent, data=data,
        )
        # Separation des lignes selon la coupure trouvee
        idx_g = [i for i in range(len(X)) if X[i][feature] <= seuil]
        idx_d = [i for i in range(len(X)) if X[i][feature] > seuil]
        if self.verbose:
            print(f"{marge}  => NOEUD INTERNE : {len(idx_g)} lignes a gauche [oui], "
                  f"{len(idx_d)} lignes a droite [non]")
        # Appels recursifs ; on memorise l'identifiant de chaque enfant dans data
        # pour pouvoir redescendre l'arbre lors de la prediction.
        data["gauche"] = self._construire([X[i] for i in idx_g], [y[i] for i in idx_g],
                                          profondeur + 1, nid, "oui")
        data["droite"] = self._construire([X[i] for i in idx_d], [y[i] for i in idx_d],
                                          profondeur + 1, nid, "non")
        return nid

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        """Predit la classe de chaque ligne de X. Temps O(m * h) pour m lignes."""
        return [self._predire_un(x) for x in X]

    def _predire_un(self, x):
        """
        Descend l'arbre de la racine jusqu'a une feuille :
        a chaque noeud interne, on va a gauche si x[feature] <= seuil, sinon a droite.

        get_node() est une recherche dans un dictionnaire -> O(1) en moyenne,
        donc temps total O(h). Espace O(1).
        """
        noeud = self.arbre.get_node(self.arbre.root)
        while noeud.data["valeur"] is None:          # tant qu'on n'est pas sur une feuille
            d = noeud.data
            suivant = d["gauche"] if x[d["feature"]] <= d["seuil"] else d["droite"]
            noeud = self.arbre.get_node(suivant)
        return noeud.data["valeur"]

    def chemin(self, x):
        """
        Comme _predire_un, mais retourne aussi la liste des tests effectues :
        (classe_predite, ["Age = 28 <= 25 ? -> non", ...]). Sert a expliquer une prediction.
        """
        etapes = []
        noeud = self.arbre.get_node(self.arbre.root)
        while noeud.data["valeur"] is None:
            d = noeud.data
            valeur = x[d["feature"]]
            va_a_gauche = valeur <= d["seuil"]
            etapes.append(f"{self.noms_features[d['feature']]} = {valeur:.4g} <= {d['seuil']:.4g} ?"
                          f"  -> {'oui' if va_a_gauche else 'non'}")
            noeud = self.arbre.get_node(d["gauche"] if va_a_gauche else d["droite"])
        return noeud.data["valeur"], etapes

    # ------------------------------------------------------------------ #
    #  Affichage                                                         #
    # ------------------------------------------------------------------ #
    def afficher(self):
        """
        Affiche l'arbre dans le terminal grace a treelib.
        sorting=False : garde l'ordre de creation des enfants ([oui] puis [non]) ;
        par defaut treelib trie les enfants par ordre alphabetique de leur tag,
        ce qui affichait [non] avant [oui].
        """
        self.arbre.show(sorting=False)


# =============================================================================
# 3. RANDOM FOREST
# =============================================================================

class RandomForest:
    """
    Foret de n_arbres arbres de decision.

    Chaque arbre est entraine sur un echantillon bootstrap different et tire
    ses variables au hasard a chaque noeud : les arbres sont donc differents
    et leurs erreurs se compensent lors du vote.
    """

    def __init__(self, n_arbres=10, n_features=None, seed=None, verbose=False):
        self.verbose = verbose          # True : affiche la construction de chaque arbre
        self.n_arbres = n_arbres        # B : nombre d'arbres
        self.n_features = n_features    # k (None -> racine de p, calcule dans fit)
        self.seed = seed                # graine : resultats reproductibles
        self.arbres = []                # liste des ArbreDecision entraines

    def fit(self, X, y, noms_features=None):
        """
        Entraine les B arbres.

        Complexite : temps O(B * k * n^2) (arbres equilibres),
                     espace O(B * n) pour stocker la foret.
        """
        X = [list(x) for x in X]
        y = list(y)
        # UN SEUL generateur pour toute la foret, partage par tous les arbres
        rng = np.random.default_rng(self.seed)
        n, p = len(X), len(X[0])
        # Nombre de variables tirees par noeud : racine de p par defaut (classification)
        k = self.n_features or max(1, int(np.sqrt(p)))
        self.arbres = []
        for b in range(self.n_arbres):
            # ETAPE 1 : bootstrap = n indices tires AVEC remise entre 0 et n-1
            # (certaines lignes apparaissent plusieurs fois, ~37 % n'apparaissent pas)
            idx = rng.integers(0, n, n)
            if self.verbose:
                print("\n" + "=" * 78)
                print(f" ARBRE {b + 1} / {self.n_arbres}")
                print("=" * 78)
                distinctes = sorted(set(int(i) for i in idx))
                absentes = [i for i in range(n) if i not in set(distinctes)]
                print(f"ETAPE 1 - Bootstrap : {n} lignes tirees AVEC remise -> "
                      f"{len(distinctes)} lignes distinctes, {len(absentes)} jamais tirees")
                if n <= 60:
                    print(f"  lignes tirees (numero de ligne, 1er = 0) : {[int(i) for i in idx]}")
                    print(f"  lignes jamais tirees                     : {absentes}")
                print()
            arbre = ArbreDecision(k, rng, noms_features, verbose=self.verbose)
            arbre.fit([X[i] for i in idx], [y[i] for i in idx])
            if self.verbose:
                print(f"\nArbre {b + 1} obtenu :")
                arbre.afficher()
            self.arbres.append(arbre)
        return self

    def predict(self, X):
        """
        ETAPE 5 : vote majoritaire.

        preds est une liste de B listes (une par arbre) ; zip(*preds) la
        "transpose" pour obtenir, pour chaque exemple, les B votes.
        Complexite : temps O(B * m * h) pour m exemples, espace O(B * m).
        """
        preds = [a.predict(X) for a in self.arbres]       # B listes de m predictions
        return [Counter(col).most_common(1)[0][0] for col in zip(*preds)]


# =============================================================================
# 4. VISUALISATION (optionnelle, necessite matplotlib)
# =============================================================================

def visualiser_foret(foret, X, y, noms, fichier="frontiere_foret.png"):
    """
    Dessine, pour 2 variables, la zone ou la foret predit "Oui" ou "Non"
    (couleur = proportion d'arbres qui votent "Oui"), avec les clients
    d'entrainement par-dessus. Enregistre l'image dans `fichier`.
    """
    import matplotlib
    matplotlib.use("Agg")                            # pas de fenetre : ecriture dans un fichier
    import matplotlib.pyplot as plt

    xs = np.linspace(15, 55, 161)                    # grille d'ages
    ys = np.linspace(10, 60, 201)                    # grille de salaires
    grille = [[a, s] for s in ys for a in xs]
    votes = np.array([a.predict(grille) for a in foret.arbres])   # (B, nb points)
    part_oui = (votes == "Oui").mean(axis=0).reshape(len(ys), len(xs))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(xs, ys, part_oui, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, label='Part des arbres qui votent "Oui"')
    for classe, marque in (("Oui", "o"), ("Non", "X")):
        pts = [x for x, c in zip(X, y) if c == classe]
        ax.scatter([a for a, _ in pts], [s for _, s in pts], marker=marque, s=90,
                   c="black", label=f"Client {classe}")
    ax.set_xlabel(noms[0])
    ax.set_ylabel(noms[1] + " (k)")
    ax.set_title(f"Random Forest ({len(foret.arbres)} arbres) : decision par vote")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    print(f"Graphique enregistre dans {fichier}")


# =============================================================================
# 5. PROGRAMME PRINCIPAL : exemple des clients A a F
# =============================================================================

if __name__ == "__main__":
    noms = ["Age", "Salaire"]
    # 6 clients : [age, salaire en milliers] et la reponse "Oui"/"Non"
    X = [[20, 15], [25, 20], [30, 25], [35, 40], [45, 50], [50, 55]]
    y = ["Non", "Non", "Oui", "Oui", "Oui", "Non"]

    # --- Verification des calculs de Gini (valeurs attendues en commentaire) ---
    print("=== Gini du noeud de depart (3 Oui, 3 Non) ===")
    print(gini(y))                                     # 0.5

    print("\n=== Test du split 'Age <= 25' ===")
    y_g, y_d = ["Non", "Non"], ["Oui", "Oui", "Oui", "Non"]
    print("Gini gauche  :", gini(y_g))                 # 0
    print("Gini droite  :", gini(y_d))                 # 0.375
    print("Gini pondere :", gini_pondere(y_g, y_d))    # 0.25

    # --- Un arbre seul, avec toutes les variables a chaque noeud ---
    # Graine fixee : "Age <= 25" et "Salaire <= 20" ont le meme Gini (0.25) ;
    # la premiere variable testee gagne, et l'ordre des variables est tire au
    # hasard. Sans graine, la racine changerait d'une execution a l'autre.
    print("\n=== Un seul arbre (toutes les features) ===")
    arbre = ArbreDecision(noms_features=noms, rng=np.random.default_rng(0)).fit(X, y)
    arbre.afficher()
    print("Predictions :", arbre.predict(X))

    # --- La foret : 25 arbres, graine fixee pour des resultats reproductibles ---
    print("\n=== Random Forest (25 arbres) ===")
    foret = RandomForest(n_arbres=25, seed=42).fit(X, y, noms_features=noms)
    print("Premier arbre de la foret :")
    foret.arbres[0].afficher()
    print("Predictions :", foret.predict(X))
    nouveau = [[28, 30]]
    votes = Counter(a.predict(nouveau)[0] for a in foret.arbres)
    print("Nouveau client (28 ans, 30k) :", foret.predict(nouveau), "- votes :", dict(votes))

    # --- Comparaison facultative avec l'arbre de scikit-learn ---
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        sk = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
        print("\n=== Meme arbre avec scikit-learn (pour comparer) ===")
        print(export_text(sk, feature_names=noms))
    except ImportError:
        print("\n(scikit-learn non installe : comparaison ignoree)")

    # --- Graphique facultatif ---
    try:
        visualiser_foret(foret, X, y, noms)
    except ImportError:
        print("(matplotlib non installe : graphique ignore)")