# ============================================================
# dataset.py — Dataset PyTorch personnalisé
# Devoir 3 : Fine-tuning BERT - Fake Job Postings
# ============================================================
# Tâche : Classification multi-classes de la colonne 'description'
#         selon la colonne 'required_experience' (7 classes)
#
# Contenu :
#   - Exploration et statistiques réelles du dataset
#   - Encodage des labels string → entier (LabelEncoder)
#   - Sauvegarde du mapping dans best_model/label_encoder.json
#   - Classe JobPostingDataset (torch.utils.data.Dataset)
#   - Tokenization BERT avec padding et attention mask
#   - Split train/validation 80/20 stratifié
#   - Création des DataLoaders
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. EXPLORATION DU DATASET
# ============================================================

def explore_dataset(csv_path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV et affiche les statistiques réelles du dataset.

    Stats affichées :
      - Dimensions brutes et valeurs manquantes
      - Distribution complète de required_experience (avec NaN)
      - Distribution après suppression des NaN
      - Longueur des descriptions par classe
      - 1 exemple par classe

    Args:
        csv_path: Chemin vers le fichier CSV.

    Returns:
        DataFrame pandas BRUT (avant tout filtrage).
    """
    print("=" * 60)
    print("EXPLORATION DU DATASET — Fake Job Postings")
    print("Tâche : description → required_experience (7 classes)")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    # --- Dimensions brutes ---
    print(f"\n[dataset] Dimensions brutes : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"[dataset] NaN dans 'required_experience' : {df['required_experience'].isna().sum()} "
          f"({df['required_experience'].isna().sum()/len(df)*100:.1f}%)")
    print(f"[dataset] NaN dans 'description'         : {df['description'].isna().sum()}")

    # --- Distribution complète avec NaN ---
    print("\n[dataset] Distribution required_experience (dataset brut, avec NaN) :")
    vc_full = df['required_experience'].value_counts(dropna=False)
    for label, count in vc_full.items():
        label_str = str(label) if pd.notna(label) else "NaN (supprimés)"
        print(f"  {label_str:<25} : {count:>5} ({count/len(df)*100:.1f}%)")

    # --- Après suppression NaN ---
    df_clean = df.dropna(subset=['required_experience']).copy()
    print(f"\n[dataset] Après suppression des NaN : {len(df_clean)} exemples restants")

    # --- Distribution finale ---
    print("\n[dataset] Distribution finale (7 classes) :")
    vc_clean = df_clean['required_experience'].value_counts()
    total = len(df_clean)
    for label, count in vc_clean.items():
        print(f"  {label:<25} : {count:>5} ({count/total*100:.1f}%)")

    ratio = vc_clean.max() / vc_clean.min()
    print(f"\n  ⚠ Ratio déséquilibre : {ratio:.1f}:1 "
          f"({vc_clean.idxmax()} vs {vc_clean.idxmin()})")
    print("  → Stratégie : class_weight dans CrossEntropyLoss + F1 macro")

    # --- Longueur des descriptions ---
    df_clean['desc_len'] = df_clean['description'].apply(
        lambda x: len(str(x).split())
    )
    print("\n[dataset] Longueur des descriptions (en mots) :")
    print(f"  Min    : {df_clean['desc_len'].min()}")
    print(f"  Max    : {df_clean['desc_len'].max()}")
    print(f"  Moyenne: {df_clean['desc_len'].mean():.1f}")
    print(f"  Médiane: {df_clean['desc_len'].median():.1f}")
    print(f"  90e pct: {df_clean['desc_len'].quantile(0.90):.1f}")
    print(f"  95e pct: {df_clean['desc_len'].quantile(0.95):.1f}")
    print(f"\n  → max_length=128 tokens (vitesse CPU) ; 256 recommandé sur GPU")

    # --- Longueur par classe ---
    print("\n[dataset] Longueur description par classe (moyenne / médiane) :")
    for cls in vc_clean.index:
        sub = df_clean[df_clean['required_experience'] == cls]['desc_len']
        print(f"  {cls:<25} : moy={sub.mean():.0f}  med={sub.median():.0f}")

    # --- 1 exemple par classe ---
    print("\n[dataset] 1 exemple par classe :")
    for cls in vc_clean.index:
        sub = df_clean[df_clean['required_experience'] == cls]
        sample = sub.sample(1, random_state=42).iloc[0]
        preview = str(sample['description'])[:100].replace('\n', ' ')
        print(f"\n  [{cls}]")
        print(f"  {preview}...")

    print("\n" + "=" * 60)
    return df


# ============================================================
# 2. ENCODAGE DES LABELS
# ============================================================

def build_label_encoder(labels: list) -> LabelEncoder:
    """
    Crée et ajuste un LabelEncoder sur les labels string.

    LabelEncoder transforme les classes string en entiers :
      Associate        → 0
      Director         → 1
      Entry level      → 2
      Executive        → 3
      Internship       → 4
      Mid-Senior level → 5
      Not Applicable   → 6
    (ordre alphabétique, déterministe)

    Args:
        labels: Liste des labels string du dataset complet.

    Returns:
        LabelEncoder ajusté.
    """
    le = LabelEncoder()
    le.fit(labels)
    print(f"\n[dataset] Classes encodées ({len(le.classes_)}) :")
    for i, cls in enumerate(le.classes_):
        print(f"  {i} → {cls}")
    return le


def save_label_encoder(le: LabelEncoder, save_dir: str = "best_model") -> None:
    """
    Sauvegarde le mapping LabelEncoder dans un fichier JSON.

    Ce fichier est nécessaire pour demo.py : il permet de décoder
    les prédictions entières (0-6) en noms de classes lisibles,
    sans avoir à réimporter sklearn ou relire le CSV.

    Args:
        le:       LabelEncoder ajusté.
        save_dir: Dossier de sauvegarde (même que le modèle BERT).
    """
    os.makedirs(save_dir, exist_ok=True)
    mapping = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
    path = os.path.join(save_dir, "label_encoder.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[dataset] LabelEncoder sauvegardé → {path}")


def load_label_encoder(save_dir: str = "best_model") -> dict:
    """
    Charge le mapping LabelEncoder depuis le fichier JSON.

    Args:
        save_dir: Dossier contenant label_encoder.json.

    Returns:
        Dictionnaire {entier: nom_classe}.
    """
    path = os.path.join(save_dir, "label_encoder.json")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # JSON charge les clés en string → conversion en int
    return {int(k): v for k, v in mapping.items()}


# ============================================================
# 3. CLASSE DATASET PYTORCH
# ============================================================

class JobPostingDataset(Dataset):
    """
    Dataset PyTorch pour la classification multi-classes de descriptions.

    Entrée  : colonne 'description' (texte brut de l'offre d'emploi)
    Sortie  : classe de 'required_experience' encodée en entier (0-6)

    Le tokenizer BERT produit 3 tenseurs par exemple :
      - input_ids      : indices des tokens dans le vocabulaire BERT
      - attention_mask : 1 pour les vrais tokens, 0 pour le padding
      - label          : entier de 0 à 6 (classe d'expérience)

    Args:
        texts:      Liste de descriptions (strings).
        labels:     Liste d'entiers encodés (0 à num_classes-1).
        tokenizer:  Tokenizer BERT HuggingFace.
        max_length: Longueur max en tokens (128 sur CPU, 256 sur GPU).
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: BertTokenizer,
        max_length: int = 128,
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
        Tokenise et retourne un exemple sous forme de dictionnaire.

        La tokenization BERT ajoute [CLS] en début et [SEP] en fin.
        Le padding complète jusqu'à max_length, la troncature coupe
        les textes trop longs.

        Args:
            idx: Index de l'exemple.

        Returns:
            Dict avec 'input_ids', 'attention_mask', 'label'.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",  # Complète avec [PAD] jusqu'à max_length
            truncation=True,       # Tronque si le texte dépasse max_length
            return_tensors="pt",   # Retourne des tenseurs PyTorch
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # attention_mask : 1 = token réel, 0 = padding (CRUCIAL pour BERT)
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 4. CHARGEMENT ET SPLIT DU DATASET
# ============================================================

def load_and_split_data(
    csv_path: str,
    tokenizer: BertTokenizer,
    max_length: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
    save_dir: str = "best_model",
) -> tuple:
    """
    Charge le CSV, prépare les données et crée les datasets train/val.

    Pipeline :
      1. Lecture du CSV
      2. Suppression des lignes NaN sur 'required_experience'
         (7 050 lignes supprimées, 10 830 restantes)
      3. Suppression des descriptions vides
      4. Encodage LabelEncoder (string → 0..6)
      5. Sauvegarde du LabelEncoder en JSON
      6. Split stratifié 80/20

    Le split STRATIFIÉ est critique ici : Executive n'a que 141 exemples
    (112 train / 29 val). Sans stratification, certaines classes pourraient
    être sous-représentées en validation.

    Args:
        csv_path:   Chemin vers le CSV.
        tokenizer:  Tokenizer BERT.
        max_length: Longueur max des séquences.
        test_size:  Proportion validation (0.2 = 20%).
        seed:       Graine pour reproductibilité.
        save_dir:   Dossier où sauvegarder label_encoder.json.

    Returns:
        Tuple (train_dataset, val_dataset, y_train_encoded, label_encoder).
    """
    df = pd.read_csv(csv_path)

    # --- Suppression des NaN sur le label ---
    n_before = len(df)
    df = df.dropna(subset=['required_experience']).copy()
    n_after = len(df)
    print(f"\n[dataset] Suppression NaN : {n_before} → {n_after} exemples "
          f"({n_before - n_after} supprimés)")

    # --- Suppression des descriptions vides ---
    df = df.dropna(subset=['description']).copy()
    print(f"[dataset] Après suppression desc vides : {len(df)} exemples")

    # --- Textes et labels ---
    # On utilise uniquement 'description' (pas title) car la consigne
    # demande de classifier la colonne description
    texts = df['description'].tolist()
    labels_str = df['required_experience'].tolist()

    # --- Encodage LabelEncoder ---
    le = build_label_encoder(labels_str)
    labels_encoded = le.transform(labels_str).tolist()

    # Sauvegarde du mapping pour demo.py
    save_label_encoder(le, save_dir)

    # --- Split stratifié 80/20 ---
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=labels_encoded,  # Préserve la proportion de chaque classe
    )

    print(f"\n[dataset] Split stratifié 80/20 :")
    print(f"  Train : {len(X_train)} exemples")
    print(f"  Val   : {len(X_val)} exemples")

    # Vérification de la distribution dans chaque split
    class_names = list(le.classes_)
    print(f"\n[dataset] Distribution par split :")
    print(f"  {'Classe':<25} {'Train':>6} {'Val':>6}")
    print(f"  {'-'*40}")
    for i, cls in enumerate(class_names):
        n_train = y_train.count(i)
        n_val = y_val.count(i)
        print(f"  {cls:<25} {n_train:>6} {n_val:>6}")

    # Création des datasets PyTorch
    train_dataset = JobPostingDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = JobPostingDataset(X_val, y_val, tokenizer, max_length)

    return train_dataset, val_dataset, y_train, le


# ============================================================
# 5. CRÉATION DES DATALOADERS
# ============================================================

def get_dataloaders(
    train_dataset: JobPostingDataset,
    val_dataset: JobPostingDataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple:
    """
    Crée les DataLoaders PyTorch pour l'entraînement et la validation.

    Args:
        train_dataset: Dataset d'entraînement.
        val_dataset:   Dataset de validation.
        batch_size:    Taille des mini-batches (16 pour BERT).
        num_workers:   Processus parallèles (0 = safe sur Windows).

    Returns:
        Tuple (train_loader, val_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Mélange à chaque epoch → meilleure généralisation
        num_workers=num_workers,
        pin_memory=True,    # Accélère le transfert CPU → GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # Pas de mélange en validation
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n[dataset] DataLoaders créés :")
    print(f"  Train : {len(train_loader)} batches de {batch_size}")
    print(f"  Val   : {len(val_loader)} batches de {batch_size}")

    return train_loader, val_loader


# ============================================================
# 6. TEST RAPIDE DU MODULE
# ============================================================

if __name__ == "__main__":
    """
    Test rapide : vérifie que le dataset et les DataLoaders fonctionnent.
    """
    from transformers import BertTokenizer

    CSV_PATH = os.path.join("data", "fake_job_postings 2.csv")

    # Exploration
    df = explore_dataset(CSV_PATH)

    # Chargement tokenizer
    print("\n[dataset] Chargement du tokenizer bert-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Split et datasets
    train_ds, val_ds, y_train, le = load_and_split_data(
        CSV_PATH, tokenizer, max_length=128
    )

    # Vérification d'un exemple
    sample = train_ds[0]
    print(f"\n[dataset] Vérification d'un exemple :")
    print(f"  input_ids      : shape={sample['input_ids'].shape}")
    print(f"  attention_mask : shape={sample['attention_mask'].shape}")
    print(f"  label          : {sample['label'].item()} "
          f"({le.classes_[sample['label'].item()]})")

    # Vérification d'un batch
    train_loader, val_loader = get_dataloaders(train_ds, val_ds, batch_size=16)
    batch = next(iter(train_loader))
    print(f"\n[dataset] Vérification d'un batch :")
    print(f"  input_ids : {batch['input_ids'].shape}")
    print(f"  labels    : {batch['label'].shape}")
    print("\n[dataset] Module dataset.py OK !")
