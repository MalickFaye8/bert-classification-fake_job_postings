# ============================================================
# dataset.py — Dataset PyTorch personnalisé
# Devoir 3 : Fine-tuning BERT - Fake Job Postings
# ============================================================
# Contenu :
#   - Exploration et statistiques du dataset CSV
#   - Classe JobPostingDataset (torch.utils.data.Dataset)
#   - Tokenization BERT avec padding et attention mask
#   - Split train/validation 80/20 stratifié
#   - Création des DataLoaders
# ============================================================

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


# ============================================================
# 1. EXPLORATION DU DATASET
# ============================================================

def explore_dataset(csv_path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV et affiche les statistiques clés du dataset.

    Cette étape est obligatoire selon l'énoncé : il faut inspecter
    le dataset AVANT d'écrire la moindre ligne d'entraînement.

    Statistiques affichées :
      - Nombre total d'exemples et de classes
      - Distribution des classes (détection du déséquilibre)
      - Longueur des textes en mots (min, max, moyenne)
      - 5 exemples de textes avec leurs labels

    Args:
        csv_path: Chemin vers le fichier fake_job_postings.csv.

    Returns:
        DataFrame pandas avec les données brutes.
    """
    print("=" * 60)
    print("EXPLORATION DU DATASET — Fake Job Postings")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    # --- Dimensions ---
    print(f"\n[dataset] Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"[dataset] Colonnes   : {df.columns.tolist()}")

    # --- Distribution des classes ---
    print("\n[dataset] Distribution des classes (label 'fraudulent') :")
    counts = df["fraudulent"].value_counts()
    total = len(df)
    for label, count in counts.items():
        name = "Réel (0)" if label == 0 else "Frauduleux (1)"
        print(f"  {name} : {count} exemples ({count/total*100:.1f}%)")

    ratio = counts[0] / counts[1]
    print(f"\n  ⚠ Ratio de déséquilibre : {ratio:.1f}:1 (seuil énoncé = 2:1)")
    print("  → Stratégie adoptée : class_weight dans CrossEntropyLoss")
    print("    + F1-score weighted comme métrique principale")

    # --- Longueur des textes ---
    # On combine title + description car ce sont les champs les plus informatifs
    df["text"] = (
        df["title"].fillna("") + " [SEP] " + df["description"].fillna("")
    )
    df["text_length"] = df["text"].apply(lambda x: len(x.split()))

    print("\n[dataset] Longueur des textes combinés (title + description) en mots :")
    print(f"  Min    : {df['text_length'].min()}")
    print(f"  Max    : {df['text_length'].max()}")
    print(f"  Moyenne: {df['text_length'].mean():.1f}")
    print(f"  Médiane: {df['text_length'].median():.1f}")
    print(f"  95e pct: {df['text_length'].quantile(0.95):.1f}")
    print("\n  → max_length=256 tokens choisi pour couvrir la majorité des textes")

    # --- 5 exemples ---
    print("\n[dataset] 5 exemples de textes avec leurs labels :")
    samples = df.sample(5, random_state=42)[["title", "description", "fraudulent"]]
    for i, (_, row) in enumerate(samples.iterrows()):
        label_name = "FRAUDULEUX" if row["fraudulent"] == 1 else "RÉEL"
        desc_preview = str(row["description"])[:120].replace("\n", " ")
        print(f"\n  [{i+1}] Label : {label_name}")
        print(f"       Titre : {row['title']}")
        print(f"       Desc  : {desc_preview}...")

    print("\n" + "=" * 60)
    return df


# ============================================================
# 2. CLASSE DATASET PYTORCH
# ============================================================

class JobPostingDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour la classification d'offres d'emploi.

    Chaque exemple est une offre d'emploi représentée par la concaténation
    de son titre et de sa description, tokenisée par BERT.

    Le tokenizer BERT produit 3 tenseurs pour chaque exemple :
      - input_ids      : indices des tokens dans le vocabulaire BERT
      - attention_mask : 1 pour les vrais tokens, 0 pour le padding
      - label          : 0 (réel) ou 1 (frauduleux)

    Args:
        texts:     Liste de chaînes de caractères (offres d'emploi).
        labels:    Liste d'entiers (0 ou 1).
        tokenizer: Tokenizer BERT HuggingFace.
        max_length: Longueur maximale de la séquence en tokens.
                    256 est justifié par les statistiques du dataset
                    (le 95e percentile est ~200 mots).
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: BertTokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Retourne le nombre d'exemples dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenise et retourne un exemple sous forme de dictionnaire de tenseurs.

        La tokenization BERT ajoute automatiquement :
          - [CLS] en début de séquence (utilisé pour la classification)
          - [SEP] en fin de séquence
          - Padding jusqu'à max_length si le texte est plus court
          - Troncature si le texte dépasse max_length

        Args:
            idx: Index de l'exemple dans le dataset.

        Returns:
            Dictionnaire avec 'input_ids', 'attention_mask', 'label'.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenisation avec padding et troncature automatiques
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",   # Complète avec [PAD] jusqu'à max_length
            truncation=True,        # Tronque si le texte dépasse max_length
            return_tensors="pt",    # Retourne des tenseurs PyTorch
        )

        return {
            # [1, max_length] → squeeze pour enlever la dimension batch
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # Masque d'attention : CRUCIAL pour BERT
            # 1 = token réel à prendre en compte
            # 0 = token de padding à ignorer
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 3. CHARGEMENT ET SPLIT DU DATASET
# ============================================================

def build_texts(df: pd.DataFrame) -> list:
    """
    Construit la liste de textes en combinant title et description.

    On concatène ces deux champs car :
      - Le titre donne le type de poste (signal fort)
      - La description contient les détails suspects (salaire irréaliste,
        fautes, demandes de données personnelles...)
    Le séparateur [SEP] est un token spécial BERT qui délimite les segments.

    Args:
        df: DataFrame avec les colonnes 'title' et 'description'.

    Returns:
        Liste de chaînes "titre [SEP] description".
    """
    texts = (
        df["title"].fillna("") + " [SEP] " + df["description"].fillna("")
    ).tolist()
    return texts


def load_and_split_data(
    csv_path: str,
    tokenizer: BertTokenizer,
    max_length: int = 256,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Charge le CSV, prépare les textes et crée les datasets train/val.

    Le split est STRATIFIÉ : même proportion de frauduleux (1) dans
    train et val. C'est essentiel avec un déséquilibre de 20:1 car
    un split aléatoire pourrait mettre trop peu de frauduleux en val.

    Args:
        csv_path:   Chemin vers le fichier CSV.
        tokenizer:  Tokenizer BERT HuggingFace.
        max_length: Longueur max des séquences en tokens.
        test_size:  Proportion du jeu de validation (défaut : 0.2 = 20%).
        seed:       Graine pour la reproductibilité du split.

    Returns:
        Tuple (train_dataset, val_dataset, train_labels) où
        train_labels est nécessaire pour calculer les poids de classes.
    """
    df = pd.read_csv(csv_path)

    # Construction des textes combinés
    texts = build_texts(df)
    labels = df["fraudulent"].tolist()

    # Split stratifié 80/20 (stratify garantit la même proportion de classes)
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,   # Préserve la distribution des classes dans chaque split
    )

    print(f"\n[dataset] Split stratifié 80/20 :")
    print(f"  Train : {len(X_train)} exemples "
          f"({sum(y_train)} frauduleux, {len(y_train)-sum(y_train)} réels)")
    print(f"  Val   : {len(X_val)} exemples "
          f"({sum(y_val)} frauduleux, {len(y_val)-sum(y_val)} réels)")

    # Création des datasets PyTorch
    train_dataset = JobPostingDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = JobPostingDataset(X_val, y_val, tokenizer, max_length)

    # On retourne y_train pour calculer les class_weights dans train.py
    return train_dataset, val_dataset, y_train


# ============================================================
# 4. CRÉATION DES DATALOADERS
# ============================================================

def get_dataloaders(
    train_dataset: JobPostingDataset,
    val_dataset: JobPostingDataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple:
    """
    Crée les DataLoaders PyTorch pour l'entraînement et la validation.

    Le DataLoader gère automatiquement :
      - Le découpage en mini-batches
      - Le mélange aléatoire des données (shuffle=True en train uniquement)
      - Le chargement parallèle (num_workers)

    Args:
        train_dataset: Dataset d'entraînement.
        val_dataset:   Dataset de validation.
        batch_size:    Taille des mini-batches (16 recommandé pour BERT).
        num_workers:   Processus parallèles pour le chargement des données.
                       0 = chargement dans le processus principal (Windows safe).

    Returns:
        Tuple (train_loader, val_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Mélange à chaque epoch → meilleure généralisation
        num_workers=num_workers,
        pin_memory=True,       # Accélère le transfert CPU → GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # Pas de mélange en validation (ordre non important)
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n[dataset] DataLoaders créés :")
    print(f"  Train : {len(train_loader)} batches de {batch_size}")
    print(f"  Val   : {len(val_loader)} batches de {batch_size}")

    return train_loader, val_loader


# ============================================================
# 5. TEST RAPIDE DU MODULE (exécution directe)
# ============================================================

if __name__ == "__main__":
    """
    Test rapide du module : vérifie que le dataset et les DataLoaders
    fonctionnent correctement avant d'assembler le pipeline complet.
    """
    from transformers import BertTokenizer

    CSV_PATH = os.path.join("data", "fake_job_postings 2.csv")

    # Exploration du dataset
    df = explore_dataset(CSV_PATH)

    # Chargement du tokenizer
    print("\n[dataset] Chargement du tokenizer bert-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Création du split et des datasets
    train_ds, val_ds, y_train = load_and_split_data(
        CSV_PATH, tokenizer, max_length=256
    )

    # Vérification d'un exemple
    sample = train_ds[0]
    print(f"\n[dataset] Vérification d'un exemple :")
    print(f"  input_ids      : shape={sample['input_ids'].shape}, "
          f"dtype={sample['input_ids'].dtype}")
    print(f"  attention_mask : shape={sample['attention_mask'].shape}, "
          f"dtype={sample['attention_mask'].dtype}")
    print(f"  label          : {sample['label'].item()}")

    # Création des DataLoaders
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=16)

    # Vérification d'un batch
    batch = next(iter(train_loader))
    print(f"\n[dataset] Vérification d'un batch :")
    print(f"  input_ids      : {batch['input_ids'].shape}")
    print(f"  attention_mask : {batch['attention_mask'].shape}")
    print(f"  labels         : {batch['label'].shape}")
    print("\n[dataset] Module dataset.py OK !")
