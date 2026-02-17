# Atelier NLP : Le Détecteur d'Émotions

### Objectif pédagogique

L'objectif de cet atelier est de manipuler la chaîne de traitement complète du Langage Naturel (NLP). Nous allons transformer des textes bruts (tweets) en vecteurs mathématiques pour comparer deux approches fondamentales de l'IA :

1. **Le Clustering** : Laisser l'IA regrouper les textes par affinités (sans connaître les réponses).
2. **La Classification** : Apprendre à l'IA à reconnaître des émotions précises à partir d'exemples étiquetés.

### Le Jeu de Données

Nous utiliserons le dataset **"Emotion"** disponible sur Hugging Face. Il contient des tweets classés en 6 catégories : *sadness, joy, love, anger, fear, surprise*.
**Lien du dataset :** [Hugging Face - Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)

```python
from datasets import load_dataset
import pandas as pd

# 1. Téléchargement du dataset
# Le nom sur le hub est "dair-ai/emotion"
dataset = load_dataset("dair-ai/emotion")

# 2. Aperçu de la structure
print(dataset)
```

---

## Partie 1 : Prétraitement & Vectorisation (Le Pipeline)

Avant qu'une machine puisse "lire", il faut nettoyer le texte. La qualité de vos résultats dépendra de la rigueur de cette étape.

* **Nettoyage avec NLTK** :
* Charger les données via la bibliothèque `datasets`.
* Créer une fonction `clean_text` qui réalise les opérations suivantes :
  * Passage en minuscules.
  * Suppression des caractères spéciaux et de la ponctuation (Regex).
  * Suppression des *stopwords* (mots vides sans valeur sémantique).
  * **Lemmatisation** : ramener les mots à leur racine (ex: *running* -> *run*).
  * **Vectorisation TF-IDF** :
    * Transformer votre texte nettoyé en une matrice numérique à l'aide de `TfidfVectorizer`.

---

## Partie 2 : Clustering (Apprentissage Non-Supervisé)

Ici, on oublie les labels (les émotions réelles). On demande à l'algorithme : "Regroupe ces messages en 6 tas cohérents".

* **Modèle** : Utiliser l'algorithme **K-Means** avec `n_clusters=6`.
* **Analyse des résultats** :
  * Pour chaque cluster, identifiez les mots les plus représentatifs (les mots proches du **centroïde**).
* **Défi** : Arrivez-vous à nommer une émotion pour chaque groupe créé par l'IA ? Les groupes correspondent-ils aux émotions réelles du dataset ?

---

## Partie 3 : Classification (Apprentissage Supervisé)

Maintenant, nous allons apprendre à l'IA à prédire l'émotion en lui montrant les bonnes réponses.

* **Entraînement** : Utiliser un modèle classique (**Naive Bayes** ou **Random Forest**) sur vos données vectorisées.
* **Évaluation** : Affichez un rapport de classification (`classification_report`) pour voir la précision par émotion.
* **Le test du monde réel** : Testez votre modèle avec des phrases contenant des négations :
  * *"I am not happy"*
  * *"This is not bad"*
* **Analyse critique** : Le modèle comprend-il l'inversion de sens apportée par le mot "not" ? Pourquoi ?

Respectez les bonne pratiques et affichez la matrice de confusion et le rapport de classification.