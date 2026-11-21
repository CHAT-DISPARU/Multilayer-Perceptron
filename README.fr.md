[🇬🇧 English](README.md) | 🇫🇷 Français

# Multilayer Perceptron : détection du cancer du sein

Un réseau de neurones **écrit from scratch avec NumPy** (sans TensorFlow, PyTorch ou scikit-learn) qui apprend à classer une tumeur du sein comme **maligne (M)** ou **bénigne (B)** à partir de mesures faites sur des images de cellules.

Ce README a deux objectifs : expliquer comment utiliser le projet, et surtout expliquer **pourquoi ça marche** : les maths, les algorithmes et les choix faits.

---

## Sommaire

1. [Le sujet](#1-le-sujet)
2. [Les données](#2-les-données)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation et utilisation](#4-installation-et-utilisation)
5. [Les bases d'un réseau de neurones](#5-les-bases-dun-réseau-de-neurones)
6. [Le passage avant (forward)](#6-le-passage-avant-forward)
7. [Les fonctions d'activation](#7-les-fonctions-dactivation)
8. [Mesurer l'erreur : la fonction de coût](#8-mesurer-lerreur--la-fonction-de-coût)
9. [La rétropropagation (backpropagation)](#9-la-rétropropagation-backpropagation)
10. [Les optimiseurs : SGD et Adam](#10-les-optimiseurs--sgd-et-adam)
11. [Initialisation des poids](#11-initialisation-des-poids)
12. [Standardisation des données](#12-standardisation-des-données)
13. [La boucle d'entraînement](#13-la-boucle-dentraînement)
14. [Early stopping](#14-early-stopping)
15. [Les métriques d'évaluation](#15-les-métriques-dévaluation)
16. [Lire la sortie de l'entraînement](#16-lire-la-sortie-de-lentraînement)
17. [Glossaire](#17-glossaire)

---

## 1. Le sujet

Ce projet est un exercice classique d'apprentissage automatique : **implémenter soi-même un perceptron multicouche** (*multilayer perceptron*, MLP) et l'utiliser sur un vrai problème médical.

### Le contexte médical

Le cancer du sein est l'un des cancers les plus fréquents. Quand une anomalie est détectée (palpation, mammographie), le médecin réalise souvent une **ponction à l'aiguille fine** : on prélève quelques cellules de la masse suspecte. Ces cellules sont observées au microscope, et une image numérisée est analysée pour mesurer la forme des noyaux cellulaires : leur taille, leur régularité, la netteté de leur contour, etc.

Le but est de savoir si la tumeur est :

- **bénigne (B)** : elle ne se propage pas, elle est généralement peu dangereuse ;
- **maligne (M)** : c'est un cancer, qui peut envahir les tissus voisins.

Un réseau de neurones peut apprendre, à partir de centaines de cas déjà diagnostiqués, quelles combinaisons de mesures sont associées à une tumeur maligne. C'est un problème de **classification binaire supervisée** :

- *classification* : la réponse est une catégorie (M ou B), pas un nombre ;
- *binaire* : il y a deux catégories ;
- *supervisée* : on apprend à partir d'exemples dont on connaît la bonne réponse.


### Ce que fait le programme

1. `split.py` découpe le jeu de données en un jeu d'**entraînement** et un jeu de **validation/prédiction**.
2. `training.py` entraîne le réseau, affiche les courbes d'apprentissage et sauvegarde le modèle.
3. `predict.py` recharge le modèle et prédit le diagnostic de nouvelles données.

---

## 2. Les données

Le jeu de données est le **Wisconsin Diagnostic Breast Cancer (WDBC)** : 569 patientes, dont environ 357 cas bénins et 212 cas malins.

Chaque ligne du CSV contient 32 colonnes :

| Colonne | Contenu |
|---|---|
| 1 | `ID` : identifiant du patient (non utilisé pour apprendre) |
| 2 | `Diagnosis` : `M` (malin) ou `B` (bénin) : c'est la **cible** |
| 3 à 32 | **30 caractéristiques** numériques (les *features*) |

Les 30 caractéristiques viennent de **10 mesures** faites sur chaque noyau cellulaire :

| # | Mesure | Signification |
|---|---|---|
| a | radius | moyenne des distances du centre aux points du contour |
| b | texture | écart-type des niveaux de gris |
| c | perimeter | périmètre |
| d | area | aire |
| e | smoothness | variation locale de la longueur des rayons |
| f | compactness | périmètre² / aire − 1 |
| g | concavity | gravité des creux du contour |
| h | concave points | nombre de creux du contour |
| i | symmetry | symétrie |
| j | fractal dimension | « approximation de la côte » − 1 |

Pour chacune de ces 10 mesures, on a trois versions : la **moyenne** (`_mean`), l'**erreur standard** (`_se`) et la **pire valeur** (`_worst`, moyenne des trois plus grandes valeurs). Ce qui fait 10 × 3 = **30 features**.

Intuitivement, les cellules cancéreuses ont tendance à être plus grandes, plus irrégulières, avec un contour plus déchiqueté. Le réseau doit découvrir lui-même quelle combinaison de ces signaux est la plus discriminante.

### Les données ne sont pas équilibrées

Environ 63 % des cas sont bénins. Un « modèle » qui répondrait toujours *bénin* aurait donc déjà ~63 % de bonnes réponses sans rien apprendre. C'est pour cela qu'on ne regarde pas seulement l'accuracy, mais aussi la précision, le rappel et le F1 (voir [section 15](#15-les-métriques-dévaluation)).

---

## 3. Structure du projet

```
.
├── split.py              # découpe data.csv en train / validation
├── training.py           # entraîne le réseau, trace les courbes, sauvegarde
├── predict.py            # charge un modèle et prédit sur un CSV
├── mlp/
│   ├── __init__.py
│   ├── csv_reader.py     # lecture et validation stricte du CSV
│   ├── activations.py    # relu, sigmoid, tanh, softmax + dérivées
│   ├── layer.py          # une couche dense : forward / backward
│   ├── network.py        # empile les couches, fit, save, load
│   ├── losses.py         # binary cross-entropy, accuracy, précision, rappel, F1
│   └── optimiseur.py     # SGD et Adam
└── datasets/             # créé par split.py
    ├── data_train.csv
    └── data_predict.csv
```

Le projet est **modulaire** : l'architecture du réseau (nombre et taille des couches), l'optimiseur, la fonction d'activation et les hyperparamètres se règlent en ligne de commande, sans toucher au code.

---

## 4. Installation et utilisation

### Dépendances

```bash
pip install numpy matplotlib
```

### Étape 1 : découper les données

```bash
python split.py data.csv 0.8            # 80 % train, 20 % validation, découpage reproductible
python split.py data.csv 0.8 --rand     # découpage aléatoire différent à chaque lancement
```

Ça crée `datasets/data_train.csv` et `datasets/data_predict.csv`. Sans `--rand`, la graine est fixée à 42 : on obtient toujours le même découpage.

### Étape 2 : entraîner

```bash
python training.py --train datasets/data_train.csv --val datasets/data_predict.csv
```

Exemple complet :

```bash
python training.py --train datasets/data_train.csv --val datasets/data_predict.csv \
    --layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
```

| Option | Défaut | Rôle |
|---|---|---|
| `--train` | (obligatoire) | CSV d'entraînement |
| `--val` | (obligatoire) | CSV de validation |
| `--layer` | `24 24` | tailles des couches cachées (au moins 2) |
| `--epochs` | `70` | nombre de passages complets sur les données |
| `--batch_size` | `16` | taille des mini-batchs |
| `--lr` / `--learning_rate` | `0.001` (Adam), `0.01` (SGD) | pas d'apprentissage |
| `--optimizer` | `adam` | `adam` ou `sgd` |
| `--patience` | `0` (désactivé) | early stopping : arrêt après N epochs sans amélioration |
| `--save_path` | `saved_model` | préfixe des fichiers sauvegardés |
| `--all` | non | compare ReLU / Sigmoid / Tanh avec SGD et Adam |

Fichiers produits, pour chaque combinaison optimiseur/activation :

- `saved_model_<optimiseur>_<activation>.npy` : le modèle (poids, biais, **et** paramètres de standardisation) ;
- `saved_model_<optimiseur>_<activation>_history.json` : l'historique complet des métriques.

### Étape 3 : prédire

```bash
python predict.py --dataset datasets/data_predict.csv --model saved_model_adam_relu.npy
```

Le script affiche la binary cross-entropy, l'accuracy, la matrice de confusion, la précision, le rappel et le F1, puis écrit `predictions.csv` (colonnes `ID, Prediction, P_M, P_B`).

---

## 5. Les bases d'un réseau de neurones

### 5.1 Le neurone

Un neurone artificiel est une très petite fonction. Il reçoit des nombres $x_1, \dots, x_n$, les multiplie chacun par un **poids** $w_i$, additionne le tout avec un **biais** $b$, puis applique une **fonction d'activation** $f$ :

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}\cdot\mathbf{x} + b$$

$$a = f(z)$$

- les **poids** disent à quel point chaque entrée compte (et dans quel sens) ;
- le **biais** décale le seuil de déclenchement ;
- l'**activation** introduit de la non-linéarité (section 7).

### 5.2 La couche

Une couche regroupe plusieurs neurones qui regardent les **mêmes entrées**. Avec $n_{in}$ entrées et $n_{out}$ neurones, tous les poids se rangent dans une matrice $W$ de taille $n_{out} \times n_{in}$ (une ligne par neurone) et les biais dans un vecteur $\mathbf{b}$ de taille $n_{out}$ :

$$\mathbf{z} = W\mathbf{x} + \mathbf{b} \qquad\qquad \mathbf{a} = f(\mathbf{z})$$

### 5.3 Le réseau

Un réseau empile des couches : la sortie d'une couche est l'entrée de la suivante.

```
 entrée          couche cachée 1     couche cachée 2        sortie
 (30 features)   (24 neurones)       (24 neurones)          (2 neurones)

   x1 ─┐
   x2 ─┤            ○                    ○                    ○  → P(maligne)
   x3 ─┼─────────►  ○  ─────────────►    ○  ──────────────►   ○  → P(bénigne)
   …   ┤            ⋮                    ⋮
  x30 ─┘            ○                    ○
                  ReLU                  ReLU                softmax
```

Par défaut : `30 → 24 → 24 → 2`. L'architecture est réglable avec `--layer`.

- Les **couches cachées** transforment peu à peu les 30 mesures brutes en représentations plus abstraites, plus utiles pour décider.
- La **couche de sortie** a 2 neurones, un par classe, avec une activation softmax : elle produit deux probabilités qui somment à 1.

### 5.4 Pourquoi la non-linéarité est indispensable

Sans fonction d'activation, enchaîner deux couches revient à une seule :

$$W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2)$$

C'est encore une transformation linéaire, peu importe le nombre de couches : le réseau ne saurait tracer que des frontières de décision en ligne droite. Les activations non linéaires permettent de courber ces frontières et d'apprendre des relations complexes entre les mesures.

### 5.5 Apprendre = trouver les bons poids

Au départ, les poids sont aléatoires : le réseau répond n'importe quoi. **Apprendre**, c'est ajuster les $W$ et $\mathbf{b}$ de toutes les couches pour que les prédictions collent aux vrais diagnostics. Le réseau de `30 → 24 → 24 → 2` possède :

$$(30\times 24 + 24) + (24\times 24 + 24) + (24\times 2 + 2) = 744 + 600 + 50 = 1394 \text{ paramètres.}$$

Le cycle d'apprentissage se répète des centaines de fois :

1. **Forward** : on calcule les prédictions avec les poids actuels.
2. **Coût** : on mesure à quel point elles sont fausses.
3. **Backward** : on calcule dans quelle direction changer chaque poids pour diminuer l'erreur (rétropropagation).
4. **Mise à jour** : on déplace légèrement chaque poids dans cette direction (optimiseur).

---

## 6. Le passage avant (forward)

Notons $\mathbf{a}^{(0)} = \mathbf{x}$ l'entrée (les 30 features standardisées) et $L$ le nombre de couches. Pour chaque couche $l = 1, \dots, L$ :

$$\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \qquad\qquad \mathbf{a}^{(l)} = f_l\big(\mathbf{z}^{(l)}\big)$$

La sortie du réseau est $\hat{\mathbf{y}} = \mathbf{a}^{(L)} = (\hat p_M,\ \hat p_B)$.

### Version batch, comme dans le code

Pour traiter $N$ échantillons d'un coup, on range les entrées dans une matrice $A$ de taille $N \times n_{in}$ (un échantillon par ligne). Le code calcule alors :

$$Z = A_{prev}\, W^\top + \mathbf{b}$$

avec les dimensions $(N \times n_{in})\cdot(n_{in}\times n_{out}) = N \times n_{out}$ (`a_prev @ self.W.T + self.b` dans `layer.py`). Le biais $\mathbf{b}$ est ajouté à chaque ligne (*broadcasting*). Utiliser des matrices au lieu de boucles Python rend le calcul des dizaines de fois plus rapide.

### Mini exemple chiffré

Un neurone, deux entrées, $\mathbf{w} = (0{,}5;\ -1)$, $b = 0{,}2$, entrée $\mathbf{x} = (2;\ 1)$, activation ReLU :

$$z = 0{,}5\times 2 + (-1)\times 1 + 0{,}2 = 0{,}2 \qquad a = \max(0,\ 0{,}2) = 0{,}2$$

---

## 7. Les fonctions d'activation

Implémentées dans `mlp/activations.py`. Chacune a besoin de sa **dérivée**, qui servira à la rétropropagation.

### ReLU (*Rectified Linear Unit*)

$$\text{relu}(z) = \max(0, z) \qquad\qquad \text{relu}'(z) = \begin{cases}1 & z > 0\\ 0 & z \le 0\end{cases}$$

Très simple et très rapide. Elle ne sature pas pour les valeurs positives (le gradient reste 1), ce qui évite que l'apprentissage ralentisse dans les réseaux profonds. Défaut : un neurone dont $z$ reste négatif a un gradient nul et ne s'améliore plus (*dying ReLU*).

### Sigmoïde

$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad\qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

Elle écrase tout nombre dans $]0, 1[$. Démonstration de la dérivée (aussi en commentaire dans `activations.py`) :

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \underbrace{\frac{1}{1+e^{-z}}}_{\sigma(z)} \cdot \underbrace{\frac{e^{-z}}{1+e^{-z}}}_{1-\sigma(z)}$$

Défaut : pour $|z|$ grand, la courbe est presque plate donc $\sigma' \approx 0$. Le gradient devient minuscule : c'est le **vanishing gradient** (gradient qui s'évanouit), qui ralentit l'apprentissage des couches profondes. Le code limite $z$ à $[-500, 500]$ avant l'exponentielle pour éviter un dépassement numérique.

### Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad\qquad \tanh'(z) = 1 - \tanh^2(z)$$

Comme la sigmoïde mais dans $]-1, 1[$, et **centrée sur 0**, ce qui aide souvent l'optimisation. Elle sature aussi pour $|z|$ grand.

### Softmax (couche de sortie)

La softmax transforme un vecteur de scores quelconques (appelés *logits*) en **distribution de probabilités** :

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Chaque sortie est dans $]0,1[$ et leur somme vaut exactement 1. Ici, $\hat p_M + \hat p_B = 1$. La classe prédite est celle dont la probabilité est la plus grande.

**Stabilité numérique** : $e^{1000}$ déborde. On utilise l'invariance de la softmax par translation : soustraire le maximum ne change pas le résultat mais garde les exponentielles dans une plage sûre :

$$\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - \max(\mathbf z)}}{\sum_j e^{z_j - \max(\mathbf z)}}$$

C'est ce que fait `softmax()` dans le code (`z - np.max(z, axis=1, keepdims=True)`).

---

## 8. Mesurer l'erreur : la fonction de coût

Pour apprendre, il faut un **nombre** qui dit à quel point le réseau se trompe. Plus il est petit, mieux c'est : c'est la **fonction de coût** (*loss*).

### Représentation des étiquettes (one-hot)

Les diagnostics sont encodés ainsi :

| Diagnostic | Vecteur cible $\mathbf{y}$ |
|---|---|
| Malin (M) | `[1, 0]` |
| Bénin (B) | `[0, 1]` |

### La cross-entropy binaire

Pour un échantillon, avec $p$ la probabilité de la classe bénigne prédite par le réseau et $y \in \{0, 1\}$ la vraie valeur (1 = bénin) :

$$\mathcal{L}(p, y) = -\Big[\,y \ln p + (1-y)\ln(1-p)\,\Big]$$

Sur tout le jeu de données, on prend la moyenne :

$$J = \frac{1}{N}\sum_{k=1}^{N}\mathcal{L}(p_k, y_k)$$

### Pourquoi cette formule ?

Regardons les deux cas possibles :

- Si la tumeur est bénigne ($y = 1$) : $\mathcal{L} = -\ln p$. Si le réseau est sûr à raison ($p \to 1$), $-\ln p \to 0$ : pas de pénalité. S'il est sûr à tort ($p \to 0$), $-\ln p \to +\infty$ : pénalité énorme.
- Si la tumeur est maligne ($y = 0$) : $\mathcal{L} = -\ln(1-p)$, qui fonctionne de façon symétrique.

| Prédiction du réseau | Vérité | Coût |
|---|---|---|
| 99 % bénin | bénin | 0,010 |
| 70 % bénin | bénin | 0,357 |
| 50 % bénin | bénin | 0,693 |
| 10 % bénin | bénin | 2,303 |
| 1 % bénin | bénin | 4,605 |

La cross-entropy ne punit pas seulement les erreurs, elle punit surtout les **erreurs faites avec confiance**. C'est ce qu'on veut en médecine : un modèle qui dit « 99 % sûr » et se trompe est bien plus dangereux qu'un modèle qui hésite.

C'est aussi, pour la classification, une meilleure fonction de coût que l'erreur quadratique : combinée à la softmax, elle donne un gradient très simple (section 9) et ne s'écrase pas quand le réseau est très faux.

### Détail d'implémentation

- `np.clip(y_pred, 1e-15, 1 - 1e-15)` évite $\ln(0) = -\infty$.
- Avec deux sorties softmax, le code utilise la colonne 1 (probabilité de bénin) : comme $\hat p_M = 1 - \hat p_B$, cette seule colonne suffit.

### Lien avec la cross-entropy catégorielle

Pour deux classes, la cross-entropy catégorielle $-\sum_i y_i \ln \hat p_i$ est **identique** à la cross-entropy binaire, car $y_M = 1 - y_B$ et $\hat p_M = 1 - \hat p_B$. C'est pour cela que la loss affichée (binaire) et le gradient utilisé (section 9, écrit pour la softmax) sont cohérents.

---

## 9. La rétropropagation (backpropagation)

C'est le cœur de l'apprentissage. Le problème : on veut savoir, pour **chacun des 1394 paramètres**, comment le coût $J$ change si on le modifie un peu. Mathématiquement, on veut le **gradient** $\dfrac{\partial J}{\partial W^{(l)}}$ et $\dfrac{\partial J}{\partial \mathbf b^{(l)}}$ pour chaque couche.

### 9.1 L'intuition

Imagine une montagne dont l'altitude est le coût $J$ et dont les coordonnées sont tous les poids. Le gradient indique **la direction de la pente la plus raide vers le haut**. Pour descendre (diminuer l'erreur), on avance dans la direction opposée. C'est la **descente de gradient**.

La rétropropagation est simplement un moyen efficace de calculer ce gradient, en appliquant la **règle de dérivation en chaîne** de la dernière couche vers la première, en réutilisant les calculs intermédiaires.

### 9.2 La règle de la chaîne

Si $J$ dépend de $a$, qui dépend de $z$, qui dépend de $w$ :

$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w}$$

Le réseau est une grande composition de fonctions, donc le gradient est un produit de petits morceaux, un par couche traversée.

### 9.3 L'erreur de la couche de sortie : softmax + cross-entropy

On définit l'**erreur** (*delta*) d'une couche comme le gradient du coût par rapport à ses pré-activations :

$$\boldsymbol\delta^{(l)} = \frac{\partial J}{\partial \mathbf z^{(l)}}$$

Pour la couche de sortie, avec softmax et cross-entropy, il y a une simplification remarquable. Pour un échantillon, avec $\mathcal{L} = -\sum_k y_k \ln p_k$ et $p_k = \text{softmax}(\mathbf z)_k$ :

La dérivée de la softmax est :

$$\frac{\partial p_k}{\partial z_i} = p_k(\delta_{ki} - p_i) \qquad (\delta_{ki} = 1 \text{ si } k=i, \text{ sinon } 0)$$

Donc :

$$\frac{\partial \mathcal L}{\partial z_i} = -\sum_k \frac{y_k}{p_k}\, p_k(\delta_{ki} - p_i) = -\sum_k y_k(\delta_{ki} - p_i) = -y_i + p_i\underbrace{\sum_k y_k}_{=1}$$

$$\boxed{\boldsymbol\delta^{(L)} = \hat{\mathbf y} - \mathbf y}$$

Le gradient à la sortie est simplement **la prédiction moins la vérité**. C'est la première ligne de `Network.backward` :

```python
delta = y_pred - y_true
```

Si le réseau prédit $(0{,}9;\ 0{,}1)$ pour une tumeur maligne $(1;\ 0)$, l'erreur est $(-0{,}1;\ +0{,}1)$ : il faut augmenter le score « malin » et diminuer le score « bénin ».

### 9.4 Les gradients des poids d'une couche

Une fois $\boldsymbol\delta^{(l)}$ connu, les gradients de la couche $l$ s'obtiennent directement, car $\mathbf z^{(l)} = W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}$ :

$$\frac{\partial J}{\partial W^{(l)}} = \boldsymbol\delta^{(l)}\,\big(\mathbf a^{(l-1)}\big)^\top \qquad\qquad \frac{\partial J}{\partial \mathbf b^{(l)}} = \boldsymbol\delta^{(l)}$$

Interprétation : le gradient du poids $w_{ij}$ est (l'erreur du neurone $i$) × (l'activité de l'entrée $j$). Un poids est blâmé en proportion de l'erreur du neurone **et** de l'importance du signal qui passait par lui.

Sur un batch de $N$ échantillons, on **moyenne** (matrices avec un échantillon par ligne) :

$$dW^{(l)} = \frac{1}{N}\,\Delta^{(l)\top} A^{(l-1)} \qquad\qquad d\mathbf b^{(l)} = \frac{1}{N}\sum_{k=1}^{N}\boldsymbol\delta^{(l)}_k$$

C'est exactement `Layer.backward` :

```python
self.dW = (delta.T @ self.a_prev) / N
self.db = np.mean(delta, axis=0)
```

### 9.5 Propager l'erreur vers l'arrière

Pour passer à la couche précédente, on « renvoie » l'erreur à travers les poids, puis on la module par la dérivée de l'activation de cette couche :

$$\boldsymbol\delta^{(l-1)} = \Big(\big(W^{(l)}\big)^\top\boldsymbol\delta^{(l)}\Big)\ \odot\ f_{l-1}'\big(\mathbf z^{(l-1)}\big)$$

où $\odot$ est le produit élément par élément.

- $\big(W^{(l)}\big)^\top\boldsymbol\delta^{(l)}$ : chaque neurone de la couche $l-1$ reçoit la somme des erreurs des neurones qu'il alimente, pondérées par les poids de ces connexions ;
- $f'(\mathbf z)$ : si l'activation était saturée (dérivée ≈ 0), l'erreur est atténuée ; si le neurone ReLU était éteint, elle est coupée net.

Dans le code (`network.py`), pour la couche $i$ :

```python
delta = self.layers[i].backward(delta)      # calcule dW, db et renvoie delta @ W
if i > 0:
    delta *= activation_derivative(self.layers[i - 1].z, self.layers[i - 1].activation)
```

(En notation « un échantillon par ligne », $\boldsymbol\delta^\top W$ s'écrit `delta @ W`.)

### 9.6 Résumé de l'algorithme

```
FORWARD
    a⁰ = x
    pour l = 1 … L :
        zˡ = Wˡ·aˡ⁻¹ + bˡ
        aˡ = fₗ(zˡ)

BACKWARD
    δᴸ = ŷ − y                                   (softmax + cross-entropy)
    pour l = L … 1 :
        dWˡ = δˡ · (aˡ⁻¹)ᵀ / N
        dbˡ = moyenne(δˡ)
        si l > 1 :
            δˡ⁻¹ = (Wˡ)ᵀ·δˡ ⊙ f'ₗ₋₁(zˡ⁻¹)
```

Le coût du calcul est du même ordre qu'un passage avant : c'est ce qui rend l'apprentissage de réseaux avec des millions de paramètres praticable.

### 9.7 Remarque sur `softmax_derivative`

La dérivée de la softmax est en réalité une matrice (jacobienne) $\text{diag}(\mathbf s) - \mathbf s \mathbf s^\top$. Le code la fournit pour être complet, mais elle n'est **jamais appelée** pendant l'entraînement : grâce à la simplification de la section 9.3, le delta de la couche de sortie est calculé directement comme $\hat{\mathbf y} - \mathbf y$, et la softmax n'est utilisée que sur la dernière couche.

---

## 10. Les optimiseurs : SGD et Adam

Le gradient donne une direction. L'**optimiseur** décide de **comment avancer** dans cette direction. Les deux sont dans `mlp/optimiseur.py`.

### 10.1 Descente de gradient stochastique (SGD)

Règle de mise à jour, pour chaque paramètre $\theta$ (poids ou biais) :

$$\theta \leftarrow \theta - \eta\,\frac{\partial J}{\partial \theta}$$

$\eta$ (*eta*) est le **learning rate** (taux d'apprentissage) : la taille du pas.

| Learning rate | Effet |
|---|---|
| trop petit | apprentissage très lent, risque de rester coincé |
| bien choisi | descente régulière vers un minimum |
| trop grand | le pas dépasse le creux : la loss oscille, voire diverge |

Défaut : un seul $\eta$ pour tous les paramètres, et aucune « mémoire » des pas précédents.

> On parle de gradient **stochastique** parce qu'il est calculé sur un mini-batch tiré au hasard et non sur toutes les données : c'est une estimation bruitée du vrai gradient, mais bien moins chère à calculer, et le bruit aide même parfois à sortir des mauvais minimums.

### 10.2 Adam (*Adaptive Moment Estimation*)

Adam améliore SGD avec deux idées :

1. **Momentum** : garder une moyenne mobile des gradients passés (comme une balle qui prend de l'élan) pour lisser le bruit et accélérer dans une direction constante.
2. **Taux adaptatif** : donner à chaque paramètre son propre pas, réduit quand ses gradients sont souvent grands et augmenté quand ils sont petits.

À chaque pas $t$, avec $g_t$ le gradient :

**Premier moment** (moyenne des gradients, l'« élan ») :

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t$$

**Second moment** (moyenne des gradients au carré, l'« échelle ») :

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$

**Correction du biais** (car $m$ et $v$ démarrent à 0, ils sont sous-estimés au début) :

$$\hat m_t = \frac{m_t}{1-\beta_1^t} \qquad\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

**Mise à jour** :

$$\theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$$

Valeurs par défaut utilisées (celles de l'article original) : $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\varepsilon = 10^{-8}$ (évite la division par zéro), $\eta = 0{,}001$.

La division par $\sqrt{\hat v_t}$ normalise le pas : un paramètre aux gradients toujours énormes fait des pas modérés, un paramètre aux gradients minuscules fait des pas plus grands. Résultat : Adam est généralement **moins sensible au choix du learning rate** et converge plus vite que SGD, ce qu'on peut vérifier avec `--all`.

> Dans ce projet, l'optimiseur gère lui-même sa mémoire ($m$, $v$, $t$) par couche. La classe `Layer` ne fait que calculer `dW` et `db`.

---

## 11. Initialisation des poids

Si tous les poids démarraient à 0, tous les neurones d'une couche calculeraient la même chose, recevraient les mêmes gradients et resteraient identiques à jamais (le problème de **symétrie**). On initialise donc au hasard, mais pas n'importe comment : la taille des poids doit garder le signal d'une amplitude stable d'une couche à l'autre. Trop grand, il explose ; trop petit, il s'éteint.

`mlp/layer.py` tire les poids dans une loi uniforme $[-\text{limite}, +\text{limite}]$ :

| Activation | Méthode | Limite |
|---|---|---|
| ReLU | **He** (Kaiming) uniforme | $\sqrt{\dfrac{6}{n_{in}}}$ |
| sigmoïde, tanh, softmax | **Glorot / Xavier** uniforme | $\sqrt{\dfrac{6}{n_{in}+n_{out}}}$ |

Les biais démarrent à 0.

- **Xavier** garde la variance des activations à peu près constante pour des activations symétriques comme tanh.
- **He** double la variance par rapport à Xavier pour compenser que ReLU met à zéro la moitié des valeurs.

---

## 12. Standardisation des données

Les 30 features n'ont pas du tout la même échelle : l'aire (`Area`) vaut plusieurs centaines à plusieurs milliers, alors que la finesse du contour (`Smoothness`) vaut environ 0,1. On centre et réduit chaque feature, en utilisant la moyenne $\mu_j$ et l'écart-type $\sigma_j$ **du jeu d'entraînement** :

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

Chaque feature a alors une moyenne 0 et un écart-type 1.

### Pourquoi c'est indispensable

En théorie, les poids d'une couche pourraient absorber un décalage ou un changement d'échelle : la standardisation **n'ajoute pas de capacité** au réseau. Mais elle est essentielle pour **l'optimisation** :

1. **Les gradients dépendent de l'échelle des entrées.** Le gradient d'un poids est proportionnel à la valeur de la feature qui le traverse (section 9.4). Avec un même learning rate pour tous, une feature de l'ordre de 1000 reçoit des gradients des milliers de fois plus grands qu'une feature de l'ordre de 0,1 : le pas qui convient à l'une fait diverger l'autre.
2. **L'initialisation suppose des entrées d'amplitude ≈ 1.** Avec une entrée à 1000, un poids de 0,3 donne un score de 300 et la softmax sature dès le départ (probabilités 0 ou 1, gradients nuls).
3. Le paysage de la loss est plus régulier (moins de vallées étroites et allongées), donc la descente est plus rapide et plus stable.

### Ce que la standardisation n'est pas

Elle n'indique pas au réseau « ceci est la valeur moyenne » : c'est un simple recadrage fixe qui **conserve les écarts** (une valeur plus grande reste plus grande).

### Ne jamais restandardiser avec les nouvelles données

À la prédiction, on réutilise **la moyenne et l'écart-type du train**, pas ceux des nouvelles données. Si on standardisait des nouvelles données avec leurs propres statistiques, une valeur « anormalement haute » deviendrait 0 et serait vue comme une valeur normale : le réseau serait aveugle à la dérive des données. C'est pourquoi `mean` et `std` sont **enregistrés dans le fichier du modèle** : le modèle est autonome, et un autre entraînement ne peut pas écraser les paramètres de normalisation par erreur.

---

## 13. La boucle d'entraînement

Implémentée dans `Network.fit`. Vocabulaire :

- **epoch** : un passage complet sur tout le jeu d'entraînement ;
- **batch** : un petit paquet d'échantillons (`--batch_size`, 16 par défaut) ;
- **itération** : une mise à jour des poids, sur un batch.

Avec ~455 échantillons d'entraînement et des batchs de 16, une epoch = 29 mises à jour.

```
pour chaque epoch :
    mélanger aléatoirement les échantillons
    pour chaque mini-batch :
        ŷ = forward(batch)
        backward(ŷ, y_batch)        # calcule dW, db
        optimiseur.step()           # met à jour W et b
    évaluer sur tout le train et toute la validation
    enregistrer loss, accuracy, précision, rappel, F1
    early stopping ?
```

### Pourquoi des mini-batchs ?

| Méthode | Gradient calculé sur | Avantages / inconvénients |
|---|---|---|
| Batch complet | tout le jeu | gradient exact, mais 1 seule mise à jour par epoch : lent |
| Un échantillon à la fois | 1 échantillon | mises à jour fréquentes mais très bruitées |
| **Mini-batch** | 8 à 64 échantillons | bon compromis : bruit modéré, calcul vectorisé efficace |

### Pourquoi mélanger à chaque epoch ?

Si les données sont toujours présentées dans le même ordre, le réseau peut apprendre des artefacts de cet ordre, et les batchs sont toujours les mêmes. Le mélange (`np.random.permutation`) casse ces corrélations.

### Entraînement vs validation

On sépare les données en deux ensembles :

- **train** : le réseau apprend dessus (c'est là que les poids sont modifiés) ;
- **validation** : le réseau **ne s'entraîne jamais dessus**. Il sert à mesurer s'il **généralise** à des cas qu'il n'a jamais vus.

Un réseau suffisamment grand peut simplement **mémoriser** le train (**surapprentissage**, *overfitting*) : excellent score sur le train, mauvais sur de nouveaux cas. Le surapprentissage se voit quand `loss` continue de baisser alors que `val_loss` remonte.

---

## 14. Early stopping

Principe : arrêter l'entraînement quand la validation cesse de s'améliorer, plutôt que de continuer jusqu'au surapprentissage.

Avec `--patience N` :

1. à chaque epoch, on compare `val_loss` à la meilleure valeur vue jusqu'ici ;
2. si elle s'améliore : on **copie** les poids actuels (`W.copy()`, `b.copy()`) et on remet le compteur d'attente à 0 ;
3. sinon, on incrémente le compteur ; quand il atteint `N`, on arrête ;
4. à la fin, on **restaure les poids de la meilleure epoch**.

La restauration est importante : sans elle, le modèle sauvegardé serait celui de la dernière epoch, alors que les `N` dernières epochs étaient en train de se dégrader. Les copies sont nécessaires, sinon on conserverait de simples références vers des tableaux que l'optimiseur continue de modifier.

```bash
python training.py --train datasets/data_train.csv --val datasets/data_predict.csv \
    --epochs 200 --patience 10
```

Sortie typique :

```
early stopping at epoch 47 (best epoch: 37, val_loss: 0.0612)
> best weights restored (epoch 37)
```

L'epoch retenue est enregistrée dans le JSON (`best_epoch`), et la légende des graphiques affiche les valeurs de cette epoch.

---

## 15. Les métriques d'évaluation

### La matrice de confusion

On choisit la classe **maligne (M) comme classe « positive »**, car c'est elle qu'on cherche à détecter.

|  | Prédit M | Prédit B |
|---|---|---|
| **Vrai M** | **TP** (vrai positif) : cancer détecté | **FN** (faux négatif) : cancer raté |
| **Vrai B** | **FP** (faux positif) : fausse alarme | **TN** (vrai négatif) : bénin reconnu |

### Les métriques

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{précision} = \frac{TP}{TP + FP} \qquad\qquad \text{rappel} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2\cdot\text{précision}\cdot\text{rappel}}{\text{précision}+\text{rappel}}$$

| Métrique | Question à laquelle elle répond |
|---|---|
| **Accuracy** | Quelle part de toutes les réponses est correcte ? |
| **Précision** | Quand le modèle dit « malin », a-t-il raison ? (pénalise les fausses alarmes) |
| **Rappel** (*recall*, sensibilité) | Parmi les vrais cancers, combien en détecte-t-il ? (pénalise les cancers ratés) |
| **F1** | Un compromis entre précision et rappel (moyenne harmonique, tirée vers le bas dès que l'un des deux est faible) |

### Pourquoi l'accuracy ne suffit pas

Avec 63 % de cas bénins, répondre « bénin » partout donne 63 % d'accuracy mais un rappel de 0 : aucun cancer détecté. De plus, les deux types d'erreur ne se valent pas :

- un **faux négatif** (cancer raté) peut retarder un traitement vital ;
- un **faux positif** (fausse alarme) mène à des examens supplémentaires, stressants mais rarement graves.

En contexte médical, c'est donc le **rappel** sur la classe maligne qui est la métrique la plus parlante.

### Cas particuliers

Si $TP + FP = 0$ (le modèle ne prédit jamais « malin »), la précision est une division 0/0 : le code renvoie 0 pour éviter l'erreur. Idem pour le rappel et le F1.

---

## 16. Lire la sortie de l'entraînement

Exemple de ligne affichée à chaque epoch :

```
epoch 16/70 - loss: 0.0352 - val_loss: 0.0930 - acc: 0.9956 - val_acc: 0.9825 - val_precision: 1.0000 - val_recall: 0.9608 - val_f1: 0.9800
```

| Champ | Signification |
|---|---|
| `epoch 16/70` | 16ᵉ passage complet sur les données, sur 70 demandés |
| `loss` | cross-entropy sur le jeu d'**entraînement** |
| `val_loss` | cross-entropy sur le jeu de **validation** (surveillée par l'early stopping) |
| `acc` / `val_acc` | accuracy sur le train / la validation |
| `val_precision` | part des « malin » annoncés qui le sont vraiment |
| `val_recall` | part des vrais cancers détectés |
| `val_f1` | compromis précision / rappel |

Toutes ces valeurs sont calculées sur l'ensemble des données à la fin de l'epoch, pas sur un seul batch.

### Comment interpréter

- `loss` < `val_loss` est normal : le modèle est meilleur sur ce qu'il a appris. Un écart modeste est sain.
- Si `val_loss` **monte** pendant que `loss` **baisse** : surapprentissage.
- Si `loss` et `val_loss` sont toutes deux élevées : sous-apprentissage (modèle trop petit, learning rate inadapté, trop peu d'epochs).
- Le jeu de validation est petit (~114 cas) : un seul cas de plus ou de moins déplace les métriques de 1 à 2 points, donc il ne faut pas sur-interpréter les variations d'une epoch à l'autre.

### Courbes d'apprentissage

Avec `--all`, le programme trace pour chaque optimiseur (SGD, Adam) la loss et l'accuracy des trois activations (ReLU, sigmoïde, tanh), en trait plein pour le train et en pointillés pour la validation. Cela permet de comparer visuellement les vitesses de convergence.

### Historique enregistré

Chaque entraînement écrit un `*_history.json` contenant l'optimiseur, l'activation, les hyperparamètres, l'epoch retenue et toutes les courbes (loss, accuracy, précision, rappel, F1, pour le train et la validation). Cela permet de comparer plusieurs runs sans les relancer.

---

## 17. Glossaire

| Terme | Définition |
|---|---|
| **Feature** | une variable d'entrée (ici, une des 30 mesures) |
| **Label / cible** | la bonne réponse (M ou B) |
| **Poids / biais** | paramètres appris par le réseau |
| **Activation** | fonction non linéaire appliquée à la sortie d'un neurone |
| **Logit** | score brut avant la softmax |
| **Forward** | calcul de la sortie à partir de l'entrée |
| **Loss / coût** | nombre mesurant l'erreur du modèle |
| **Gradient** | vecteur des dérivées de la loss par rapport aux paramètres |
| **Backpropagation** | calcul efficace du gradient par la règle de la chaîne |
| **Learning rate** | taille du pas de mise à jour |
| **Epoch** | un passage complet sur le jeu d'entraînement |
| **Batch** | sous-ensemble d'échantillons traités ensemble |
| **Surapprentissage** | le modèle mémorise le train au lieu de généraliser |
| **Généralisation** | bon comportement sur des données jamais vues |
| **Hyperparamètre** | réglage fixé avant l'entraînement (learning rate, tailles de couches…) |
| **Standardisation** | centrer à 0 et réduire à un écart-type de 1 |
| **Early stopping** | arrêt de l'entraînement quand la validation ne s'améliore plus |

---

## Références

- Données : *Wisconsin Diagnostic Breast Cancer (WDBC)*, W. H. Wolberg, W. N. Street, O. L. Mangasarian (UCI Machine Learning Repository).
- Adam : D. P. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*, 2014.
- Initialisation Glorot : X. Glorot, Y. Bengio, *Understanding the difficulty of training deep feedforward neural networks*, 2010.
- Initialisation He : K. He et al., *Delving Deep into Rectifiers*, 2015.
- Rétropropagation : D. E. Rumelhart, G. E. Hinton, R. J. Williams, *Learning representations by back-propagating errors*, 1986.

### Wikipedia

(pages en anglais)

**Réseaux de neurones et apprentissage automatique**

- Multilayer perceptron : [https://en.wikipedia.org/wiki/Multilayer_perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- Neural network : [https://en.wikipedia.org/wiki/Neural_network_(machine_learning)](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
- Supervised learning : [https://en.wikipedia.org/wiki/Supervised_learning](https://en.wikipedia.org/wiki/Supervised_learning)
- Binary classification : [https://en.wikipedia.org/wiki/Binary_classification](https://en.wikipedia.org/wiki/Binary_classification)
- Feature : [https://en.wikipedia.org/wiki/Feature_(machine_learning)](https://en.wikipedia.org/wiki/Feature_(machine_learning))
- One-hot : [https://en.wikipedia.org/wiki/One-hot](https://en.wikipedia.org/wiki/One-hot)
- Overfitting : [https://en.wikipedia.org/wiki/Overfitting](https://en.wikipedia.org/wiki/Overfitting)
- Early stopping : [https://en.wikipedia.org/wiki/Early_stopping](https://en.wikipedia.org/wiki/Early_stopping)
- Regularization : [https://en.wikipedia.org/wiki/Regularization_(mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))
- Dropout : [https://en.wikipedia.org/wiki/Dropout_(neural_networks)](https://en.wikipedia.org/wiki/Dropout_(neural_networks))
- Cross-validation : [https://en.wikipedia.org/wiki/Cross-validation_(statistics)](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- Feature scaling : [https://en.wikipedia.org/wiki/Feature_scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- NumPy : [https://en.wikipedia.org/wiki/NumPy](https://en.wikipedia.org/wiki/NumPy)

**Fonctions d'activation**

- Activation function : [https://en.wikipedia.org/wiki/Activation_function](https://en.wikipedia.org/wiki/Activation_function)
- ReLU (and dying ReLU) : [https://en.wikipedia.org/wiki/Rectifier_(neural_networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- Sigmoid function : [https://en.wikipedia.org/wiki/Sigmoid_function](https://en.wikipedia.org/wiki/Sigmoid_function)
- Hyperbolic functions (tanh) : [https://en.wikipedia.org/wiki/Hyperbolic_functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
- Softmax : [https://en.wikipedia.org/wiki/Softmax_function](https://en.wikipedia.org/wiki/Softmax_function)
- Logit : [https://en.wikipedia.org/wiki/Logit](https://en.wikipedia.org/wiki/Logit)
- Vanishing gradient problem : [https://en.wikipedia.org/wiki/Vanishing_gradient_problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

**Coût et optimisation**

- Loss function : [https://en.wikipedia.org/wiki/Loss_function](https://en.wikipedia.org/wiki/Loss_function)
- Cross-entropy : [https://en.wikipedia.org/wiki/Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy)
- Backpropagation : [https://en.wikipedia.org/wiki/Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- Gradient descent : [https://en.wikipedia.org/wiki/Gradient_descent](https://en.wikipedia.org/wiki/Gradient_descent)
- Stochastic gradient descent (mini-batch, momentum, Adam) : [https://en.wikipedia.org/wiki/Stochastic_gradient_descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- Learning rate : [https://en.wikipedia.org/wiki/Learning_rate](https://en.wikipedia.org/wiki/Learning_rate)
- Moving average : [https://en.wikipedia.org/wiki/Moving_average](https://en.wikipedia.org/wiki/Moving_average)

**Métriques d'évaluation**

- Confusion matrix : [https://en.wikipedia.org/wiki/Confusion_matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- Accuracy and precision : [https://en.wikipedia.org/wiki/Accuracy_and_precision](https://en.wikipedia.org/wiki/Accuracy_and_precision)
- Precision and recall : [https://en.wikipedia.org/wiki/Precision_and_recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- Sensitivity and specificity : [https://en.wikipedia.org/wiki/Sensitivity_and_specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- F-score : [https://en.wikipedia.org/wiki/F-score](https://en.wikipedia.org/wiki/F-score)
- Harmonic mean : [https://en.wikipedia.org/wiki/Harmonic_mean](https://en.wikipedia.org/wiki/Harmonic_mean)

**Mathématiques**

- Derivative : [https://en.wikipedia.org/wiki/Derivative](https://en.wikipedia.org/wiki/Derivative)
- Partial derivative : [https://en.wikipedia.org/wiki/Partial_derivative](https://en.wikipedia.org/wiki/Partial_derivative)
- Chain rule : [https://en.wikipedia.org/wiki/Chain_rule](https://en.wikipedia.org/wiki/Chain_rule)
- Gradient : [https://en.wikipedia.org/wiki/Gradient](https://en.wikipedia.org/wiki/Gradient)
- Jacobian matrix and determinant : [https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
- Kronecker delta : [https://en.wikipedia.org/wiki/Kronecker_delta](https://en.wikipedia.org/wiki/Kronecker_delta)
- Linear map : [https://en.wikipedia.org/wiki/Linear_map](https://en.wikipedia.org/wiki/Linear_map)
- Matrix : [https://en.wikipedia.org/wiki/Matrix_(mathematics)](https://en.wikipedia.org/wiki/Matrix_(mathematics))
- Matrix multiplication : [https://en.wikipedia.org/wiki/Matrix_multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)
- Transpose : [https://en.wikipedia.org/wiki/Transpose](https://en.wikipedia.org/wiki/Transpose)
- Hadamard product : [https://en.wikipedia.org/wiki/Hadamard_product_(matrices)](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
- Exponential function : [https://en.wikipedia.org/wiki/Exponential_function](https://en.wikipedia.org/wiki/Exponential_function)
- Natural logarithm : [https://en.wikipedia.org/wiki/Natural_logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
- Uniform distribution : [https://en.wikipedia.org/wiki/Continuous_uniform_distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
- Variance : [https://en.wikipedia.org/wiki/Variance](https://en.wikipedia.org/wiki/Variance)
- Standard deviation : [https://en.wikipedia.org/wiki/Standard_deviation](https://en.wikipedia.org/wiki/Standard_deviation)
- Standard score (standardization) : [https://en.wikipedia.org/wiki/Standard_score](https://en.wikipedia.org/wiki/Standard_score)
