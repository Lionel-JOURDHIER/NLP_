# Détection d'Émotions avec RNN & Embedding

### Objectif

L'objectif est de construire un classifieur capable de comprendre la nuance des émotions dans des tweets (dataset **Emotion** de Hugging Face). Contrairement au ML classique, nous allons utiliser une mémoire séquentielle pour capter le contexte.

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

## Étape 1 : Préparation & Padding (Réflexion)

Les réseaux de neurones ont besoin de données de taille identique (tenseurs fixes). Comme les tweets ont des longueurs variées, nous utilisons le **Padding**.

**À faire :** 
1. Analysez la longueur maximale des tweets dans le dataset.
2. Appliquez un padding de taille **80** (suffisant pour ce dataset).
3. **Réflexion :** Testez les deux options `padding='pre'` et `padding='post'`.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# On uniformise à 80 mots
# Option 1 : padding='pre' (les zéros au début)
# Option 2 : padding='post' (les zéros à la fin)
X_train_padded = pad_sequences(X_train_sequences, maxlen=80, padding='pre')

```

---

## Étape 2 : La couche Embedding & le Masking

La première couche de votre modèle est la couche **Embedding**. C'est ici que les mots (indexés) deviennent des vecteurs de sens dans un espace géométrique.

**Le Masking :** Puisque nous avons ajouté beaucoup de "vide" (les zéros du padding), nous ne voulons pas que le modèle s'épuise à calculer du néant.

En ajoutant `mask_zero=True`, vous dites au modèle : *"Ignore les index 0, ils ne servent qu'à boucher les trous, ne les utilise pas pour ta mémoire."*

```python
# vocab_size : nb de mots du dictionnaire, embedding_dim : taille du vecteur (ex: 128), input_length : taille du padding (80)
model.add(layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=80, mask_zero=True))

```
---

## Étape 3 : Construction du Modèle RNN

Créez un modèle `Sequential` comprenant :

1. Une couche `Embedding` (taille de sortie : 128).
2. Une couche `LSTM` (64 unités) pour la mémoire séquentielle.
3. Une couche `Dense` de sortie pour la classification.

**Attention aux réglages :**

* Nous avons **6 émotions** à prédire. Quelle fonction d'activation choisir ?
* Les labels sont des entiers (0, 1, 2...). Quelle fonction de `loss` est la plus adaptée ?



---

## Étape 4 : Classification & Analyse

* Entraînez le modèle sur 5 à 10 époques.
* Comparez vos résultats avec ceux obtenus lundi en Machine Learning classique.
* **Test :** Essayez de prédire l'émotion de phrases complexes comme :
  * *"I am not that happy about the results"*
  * *"Surprisingly, it was not as bad as I thought"*

Respectez les bonne pratiques et affichez la matrice de confusion et le rapport de classification.

---

## Étape 5 : Transfer Learning
Au lieu de laisser TensorFlow initialiser les vecteurs au hasard, on lui injecte une matrice de poids fixe. On règle ensuite le paramètre `trainable=False`. Cela signifie que pendant l'entraînement, le modèle va ajuster ses neurones de décision (Dense) et sa mémoire (LSTM), mais il ne touchera pas aux définitions des mots. C'est un gain de temps énorme et cela permet de travailler avec de petits jeux de données.

```python
# --- 1. CHARGEMENT DES DONNÉES ---
# On récupère les tweets du dataset "Emotion"
dataset = load_dataset("dair-ai/emotion")
df_train = pd.DataFrame(dataset['train'])
texts = df_train['text'].tolist()
```

On doit tokeniser puis on devra les padder

```python
# --- 2. TOKENIZATION (Le passage aux nombres) ---
# On crée le dictionnaire (ex: 'happy' -> 12)
max_words = 10000 
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# On transforme les phrases en listes de chiffres
sequences = tokenizer.texts_to_sequences(texts)

# Padding : On égalise la longueur à 80 mots (Pre-padding conseillé)
X_train = pad_sequences(sequences, maxlen=80, padding='pre')
```

Création de la matrice d'embedding en la réduisant (la matrice complète est énorme)

```python
# --- 3. CHARGEMENT DU SAVOIR DE GOOGLE (Word2Vec) ---
# Téléchargement du modèle pré-entraîné
print("Chargement de Word2Vec Google News...")
google_model = api.load("word2vec-google-news-300")

# --- 4. CRÉATION DE LA MATRICE DE CORRESPONDANCE (Mapping) ---
# On crée une matrice vide (lignes = mots du tokenizer, colonnes = 300 dim de Google)
vocab_size = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in word_index.items():
    if i >= max_words:
        continue
    if word in google_model:
        # On injecte le vecteur de Google à la ligne correspondante
        embedding_matrix[i] = google_model[word]
```

La couche d'embedding utilisera la `embedding_matrix` 

```python
# --- 5. CRÉATION DE LA COUCHE D'EMBEDDING TENSORFLOW ---
# Cette couche sera la première de notre modèle Sequential
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=300,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    input_length=80,
    trainable=False,  # On "gèle" pour ne pas perdre le savoir de Google
    mask_zero=True    # On ignore les zéros du padding
)

print("Pipeline terminé : La couche d'embedding est prête à être intégrée !")
```

---

### Aide-mémoire pour la couche de sortie :

| Nombre de classes | Activation | Loss Function |
| --- | --- | --- |
| 2 (Binaire) | `sigmoid` | `binary_crossentropy` |
| **6 (Multi-classe)** | **`softmax`** | **`sparse_categorical_crossentropy`** |

