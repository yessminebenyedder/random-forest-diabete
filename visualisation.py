"""
===============================================================================
 Visualisation de la foret : page web (HTML + SVG), sans bibliotheque externe
===============================================================================

 generer_html(foret, noms, prediction=...) ecrit un fichier .html qui contient :
   - la donnee predite et le vote des arbres (si une prediction est fournie) ;
   - le dessin de CHAQUE arbre :
        * noeud gris  = test "variable <= seuil ?"   (gauche = oui, droite = non)
        * noeud colore = feuille (couleur = classe predite)
        * trait epais noir = chemin suivi par la donnee predite
   Le fichier s'ouvre dans le navigateur (webbrowser.open).
===============================================================================
"""

import html
import webbrowser
from collections import Counter
from pathlib import Path

PALETTE = ["#2a6fbb", "#e07b00", "#2e9d5b", "#8e44ad", "#c0392b", "#7f8c8d"]
ESPACE_Y, HAUT_NOEUD, MARGE = 108, 48, 30


# -----------------------------------------------------------------------------
# Outils de dessin
# -----------------------------------------------------------------------------

def _texte_noeud(d, noms):
    """Retourne (ligne 1, ligne 2) du texte d'un noeud."""
    if d["valeur"] is None:
        l1 = f"{noms[d['feature']]} <= {d['seuil']:.4g} ?"
        l2 = f"n={d['n']}   Gini={d['gini']:.2f}"
    else:
        l1 = f"=> {d['valeur']}"
        l2 = "  ".join(f"{c}: {nb}" for c, nb in sorted(d["classes"].items(), key=lambda t: str(t[0])))
    return l1, l2


def _mise_en_page(arbre):
    """Position (x en 'cases', profondeur) de chaque noeud : feuilles alignees, parents centres."""
    pos, compteur = {}, [0]

    def rec(nid, prof):
        d = arbre.get_node(nid).data
        if d["valeur"] is not None:                 # feuille : prochaine case libre
            x = compteur[0]
            compteur[0] += 1
        else:                                       # noeud interne : centre au-dessus de ses enfants
            x = (rec(d["gauche"], prof + 1) + rec(d["droite"], prof + 1)) / 2
        pos[nid] = (x, prof)
        return x

    rec(arbre.root, 0)
    return pos, compteur[0]


def _ids_chemin(arbre, x):
    """Identifiants des noeuds traverses par la donnee x (de la racine a la feuille)."""
    ids = [arbre.root]
    noeud = arbre.get_node(arbre.root)
    while noeud.data["valeur"] is None:
        d = noeud.data
        suivant = d["gauche"] if x[d["feature"]] <= d["seuil"] else d["droite"]
        ids.append(suivant)
        noeud = arbre.get_node(suivant)
    return ids


def _svg_arbre(arbre, noms, couleurs, largeur, chemin=()):
    """Dessine un arbre en SVG. `chemin` = ids des noeuds a surligner."""
    pos, n_feuilles = _mise_en_page(arbre.arbre)
    espace_x = largeur + 18
    prof_max = max(p for _, p in pos.values())
    W = n_feuilles * espace_x + 2 * MARGE
    H = (prof_max + 1) * ESPACE_Y + MARGE
    centre = lambda nid: (MARGE + pos[nid][0] * espace_x + espace_x / 2,
                          MARGE + pos[nid][1] * ESPACE_Y + HAUT_NOEUD / 2)
    sur_chemin = set(chemin)
    liens_chemin = set(zip(chemin, chemin[1:]))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.0f} {H:.0f}" font-family="Segoe UI, Arial, sans-serif">']

    # Liens d'abord (sous les noeuds)
    for nid in pos:
        d = arbre.arbre.get_node(nid).data
        if d["valeur"] is not None:
            continue
        x0, y0 = centre(nid)
        for cle, etiquette in (("gauche", "oui"), ("droite", "non")):
            enfant = d[cle]
            x1, y1 = centre(enfant)
            actif = (nid, enfant) in liens_chemin
            out.append(f'<line x1="{x0:.1f}" y1="{y0 + HAUT_NOEUD / 2:.1f}" x2="{x1:.1f}" y2="{y1 - HAUT_NOEUD / 2:.1f}" '
                       f'stroke="{"#111" if actif else "#a8a8a8"}" stroke-width="{3.5 if actif else 1.5}"/>')
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 2
            decalage = -10 if cle == "gauche" else 10
            out.append(f'<text x="{mx + decalage:.1f}" y="{my:.1f}" font-size="12" text-anchor="middle" '
                       f'fill="{"#111" if actif else "#666"}" font-weight="{700 if actif else 400}">{etiquette}</text>')

    # Noeuds
    for nid in pos:
        d = arbre.arbre.get_node(nid).data
        cx, cy = centre(nid)
        actif = nid in sur_chemin
        if d["valeur"] is None:
            fond, bord, op = "#eeeeea", "#8a8a85", 1
        else:
            fond = bord = couleurs[d["valeur"]]
            op = 0.22
        out.append(f'<rect x="{cx - largeur / 2:.1f}" y="{cy - HAUT_NOEUD / 2:.1f}" width="{largeur}" height="{HAUT_NOEUD}" '
                   f'rx="8" fill="{fond}" fill-opacity="{op}" stroke="{"#111" if actif else bord}" '
                   f'stroke-width="{3.5 if actif else 1.5}"/>')
        l1, l2 = _texte_noeud(d, noms)
        out.append(f'<text x="{cx:.1f}" y="{cy - 4:.1f}" font-size="12.5" font-weight="600" text-anchor="middle" '
                   f'fill="#1c1c1c">{html.escape(l1)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{cy + 13:.1f}" font-size="11" text-anchor="middle" '
                   f'fill="#444">{html.escape(l2)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# -----------------------------------------------------------------------------
# Page complete
# -----------------------------------------------------------------------------

def generer_html(foret, noms, prediction=None, fichier="foret_resultat.html",
                 titre="Random Forest", noms_classes=None, ouvrir=True):
    """
    foret      : RandomForest entrainee
    noms       : noms des variables
    prediction : None ou dict {"x": donnee, "votes": [classe de chaque arbre],
                 "final": classe gagnante, "vraie_classe": (facultatif)}
    noms_classes : dict facultatif {classe: "texte de la legende"}
    Retourne le chemin du fichier ecrit.
    """
    noms_classes = noms_classes or {}
    classes = sorted({n.data["valeur"] for a in foret.arbres for n in a.arbre.all_nodes_itr()
                      if n.data["valeur"] is not None}, key=str)
    couleurs = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    e = html.escape

    # largeur des noeuds : adaptee au texte le plus long
    plus_long = max(len(t) for a in foret.arbres for n in a.arbre.all_nodes_itr()
                    for t in _texte_noeud(n.data, noms))
    largeur = max(130, min(250, int(plus_long * 6.9) + 18))

    legende = "".join(
        f'<span class="puce"><i style="background:{couleurs[c]}"></i>{e(str(noms_classes.get(c, c)))}</span>'
        for c in classes)

    bloc_prediction = ""
    chemins = [()] * len(foret.arbres)
    if prediction is not None:
        x, votes, final = prediction["x"], prediction["votes"], prediction["final"]
        chemins = [_ids_chemin(a.arbre, x) for a in foret.arbres]
        # variables effectivement testees le long des chemins
        utilisees = []
        for a, ids in zip(foret.arbres, chemins):
            for nid in ids:
                d = a.arbre.get_node(nid).data
                if d["valeur"] is None and d["feature"] not in utilisees:
                    utilisees.append(d["feature"])
        valeurs = "".join(f"<tr><td>{e(noms[j])}</td><td>{x[j]:.4g}</td></tr>" for j in utilisees)
        compte = Counter(votes)
        barre = "".join(
            f'<div style="width:{100 * compte[c] / len(votes):.1f}%;background:{couleurs[c]}">'
            f'{e(str(c))} : {compte[c]}</div>' for c in classes if compte[c])
        puces = "".join(
            f'<span class="vote" style="border-color:{couleurs[v]}">Arbre {i + 1} &rarr; <b>{e(str(v))}</b></span>'
            for i, v in enumerate(votes))
        vraie = ""
        if "vraie_classe" in prediction:
            ok = prediction["vraie_classe"] == final
            vraie = (f'<p>Vraie classe connue : <b>{e(str(prediction["vraie_classe"]))}</b> '
                     f'&mdash; {"prediction correcte" if ok else "prediction fausse"}</p>')
        bloc_prediction = f"""
<section class="carte">
  <h2>Prediction d'une nouvelle donnee</h2>
  <div class="deux">
    <div><p class="petit">Variables testees dans les arbres :</p>
      <table>{valeurs}</table></div>
    <div><p class="petit">Vote des {len(votes)} arbre(s) :</p>
      <div class="votes">{puces}</div>
      <div class="barre">{barre}</div>
      <p class="final">Prediction finale : <span style="color:{couleurs[final]}">{e(str(final))}</span></p>
      {vraie}</div>
  </div>
</section>"""

    arbres_html = ""
    for i, (a, ch) in enumerate(zip(foret.arbres, chemins), start=1):
        sous_titre = ""
        if prediction is not None:
            v = prediction["votes"][i - 1]
            sous_titre = f' &mdash; vote : <b style="color:{couleurs[v]}">{e(str(v))}</b>'
        arbres_html += (f'<section class="carte"><h2>Arbre {i}{sous_titre}</h2>'
                        f'<div class="defile">{_svg_arbre(a, noms, couleurs, largeur, chemin=ch)}</div></section>\n')

    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(titre)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; padding:24px 16px 60px; background:#f6f5f1; color:#1c1c1c;
         font-family:"Segoe UI",Arial,sans-serif; }}
  main {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  h2 {{ font-size:18px; margin:0 0 12px; }}
  .sous {{ color:#555; margin:0 0 16px; }}
  .legende {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:0 0 20px; font-size:14px; }}
  .puce i {{ display:inline-block; width:14px; height:14px; border-radius:4px; margin-right:6px; vertical-align:-2px; }}
  .carte {{ background:#fff; border:1px solid #dcdad2; border-radius:12px; padding:18px 20px; margin:0 0 18px; }}
  .defile {{ overflow-x:auto; }}
  .deux {{ display:grid; grid-template-columns:minmax(220px,1fr) 2fr; gap:28px; }}
  @media (max-width:700px) {{ .deux {{ grid-template-columns:1fr; }} }}
  .petit {{ color:#555; font-size:13px; margin:0 0 6px; }}
  table {{ border-collapse:collapse; font-size:14px; }}
  td {{ padding:3px 14px 3px 0; border-bottom:1px solid #eee; }}
  td:last-child {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  .votes {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }}
  .vote {{ border:2px solid; border-radius:8px; padding:3px 10px; font-size:14px; background:#fff; }}
  .barre {{ display:flex; height:30px; border-radius:6px; overflow:hidden; color:#fff; font-size:13px; font-weight:600; }}
  .barre div {{ display:flex; align-items:center; justify-content:center; white-space:nowrap; }}
  .final {{ font-size:20px; font-weight:700; margin:14px 0 4px; }}
</style></head><body><main>
<h1>{e(titre)}</h1>
<p class="sous">{len(foret.arbres)} arbre(s) &middot; chaque noeud gris pose une question (&laquo; oui &raquo; = branche de gauche) &middot; chaque noeud colore est une feuille (classe predite).</p>
<div class="legende">{legende}<span>Trait epais noir = chemin de la donnee predite</span></div>
{bloc_prediction}
{arbres_html}
</main></body></html>"""

    chemin_fichier = Path(fichier).resolve()
    chemin_fichier.write_text(page, encoding="utf-8")
    if ouvrir:
        webbrowser.open(chemin_fichier.as_uri())
    return chemin_fichier
