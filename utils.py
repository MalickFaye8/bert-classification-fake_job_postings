# ============================================================
# utils.py — Fonctions utilitaires du projet
# Devoir 3 : Fine-tuning BERT - Fake Job Postings
# ============================================================
# Contenu :
#   - Fixation des seeds pour la reproductibilité
#   - Calcul des métriques (accuracy, F1-score)
#   - Calcul des poids de classes (déséquilibre 20:1)
#   - Visualisation des courbes d'apprentissage
#   - Visualisation de la matrice de confusion
# ============================================================

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ------------------------------------------------------------
# 1. REPRODUCTIBILITÉ
# ------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les sources d'aléatoire pour garantir la reproductibilité
    des expériences entre les deux membres du binôme.

    Args:
        seed: Valeur de la graine aléatoire (défaut : 42).
    """
    random.seed(seed)                          # Module random de Python
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed_all(seed)           # PyTorch GPU (tous les GPUs)

    # Rend les opérations CUDA déterministes (légèrement plus lent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[utils] Seed fixée à {seed} pour la reproductibilité.")


# ------------------------------------------------------------
# 2. MÉTRIQUES
# ------------------------------------------------------------

def compute_metrics(preds: list, labels: list) -> dict:
    """
    Calcule les métriques d'évaluation pour la classification multi-classes.

    Deux F1-scores sont calculés :
      - F1 macro    : moyenne non pondérée sur toutes les classes.
                      Traite chaque classe ÉGALEMENT, y compris Executive (141 ex.)
                      → métrique principale pour évaluer l'équité entre classes.
      - F1 weighted : moyenne pondérée par le support de chaque classe.
                      Reflet de la performance globale, favorise les classes maj.

    Déséquilibre : 27:1 (Mid-Senior 3809 vs Executive 141)
    → F1 macro est la métrique principale pour ce dataset multi-classes.

    Args:
        preds:  Liste des prédictions du modèle (entiers 0 à 6).
        labels: Liste des vraies étiquettes (entiers 0 à 6).

    Returns:
        Dictionnaire avec 'accuracy', 'f1_macro' et 'f1_weighted'.
    """
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),       # Métrique principale (multi-classes)
        "f1_weighted": round(f1_weighted, 4),  # Métrique secondaire
    }


def print_classification_report(preds: list, labels: list,
                                 class_names: list = None) -> None:
    """
    Affiche le rapport de classification complet (precision, recall, F1
    par classe). Utile pour diagnostiquer le comportement sur la classe
    minoritaire (frauduleux).

    Args:
        preds:        Liste des prédictions du modèle.
        labels:       Liste des vraies étiquettes.
        class_names:  Noms des classes (ex. ["Real", "Fraudulent"]).
    """
    if class_names is None:
        class_names = ["Real", "Fraudulent"]

    report = classification_report(labels, preds, target_names=class_names,
                                   zero_division=0)
    print("\n[utils] Rapport de classification :")
    print(report)


# ------------------------------------------------------------
# 3. GESTION DU DÉSÉQUILIBRE DES CLASSES
# ------------------------------------------------------------

def get_class_weights(labels: list, device: torch.device) -> torch.Tensor:
    """
    Calcule les poids inverses des classes pour la CrossEntropyLoss.

    Le dataset fake_job_postings est très déséquilibré (~20:1).
    En pondérant la loss, on pénalise davantage les erreurs sur la
    classe minoritaire (frauduleux), ce qui améliore le rappel sur
    les offres frauduleuses.

    Formule : weight[c] = total_samples / (nb_classes * count[c])

    Args:
        labels: Liste de toutes les étiquettes du dataset d'entraînement.
        device: Device PyTorch (cpu ou cuda) pour placer le tenseur.

    Returns:
        Tenseur de poids [weight_class_0, weight_class_1] sur le device.
    """
    labels_array = np.array(labels)
    classes, counts = np.unique(labels_array, return_counts=True)
    nb_classes = len(classes)
    total = len(labels_array)

    # Poids inversement proportionnels à la fréquence de chaque classe
    weights = total / (nb_classes * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"[utils] Poids des classes : { {int(c): round(w, 4) for c, w in zip(classes, weights)} }")
    return weights_tensor


# ------------------------------------------------------------
# 4. VISUALISATION DES COURBES D'APPRENTISSAGE
# ------------------------------------------------------------

def plot_curves(history: dict, save_dir: str = "figures") -> None:
    """
    Trace et sauvegarde les courbes de loss et d'accuracy (train vs val).

    Args:
        history:  Dictionnaire contenant les listes de métriques par epoch :
                  {
                      'train_loss': [...],
                      'val_loss':   [...],
                      'train_acc':  [...],
                      'val_acc':    [...],
                  }
        save_dir: Dossier de sauvegarde des figures (créé si inexistant).
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'apprentissage — BERT Fine-tuning", fontsize=14)

    # --- Courbe de Loss ---
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title("Loss par epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Courbe d'Accuracy ---
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Accuracy")
    axes[1].set_title("Accuracy par epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Courbes sauvegardées → {save_path}")


# ------------------------------------------------------------
# 5. MATRICE DE CONFUSION
# ------------------------------------------------------------

def plot_confusion_matrix(preds: list, labels: list,
                          class_names: list = None,
                          save_dir: str = "figures") -> None:
    """
    Trace et sauvegarde la matrice de confusion normalisée et brute.

    Args:
        preds:       Liste des prédictions du modèle.
        labels:      Liste des vraies étiquettes.
        class_names: Noms des classes (défaut : ["Real", "Fraudulent"]).
        save_dir:    Dossier de sauvegarde (créé si inexistant).
    """
    if class_names is None:
        class_names = ["Real", "Fraudulent"]

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    # Normalisation par ligne pour avoir des pourcentages par classe réelle
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Matrice de Confusion — BERT Fake Job Postings", fontsize=13)

    # --- Matrice brute (nombre d'exemples) ---
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Valeurs brutes")
    axes[0].set_ylabel("Vraie classe")
    axes[0].set_xlabel("Classe prédite")

    # --- Matrice normalisée (pourcentages) ---
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Normalisée (% par classe réelle)")
    axes[1].set_ylabel("Vraie classe")
    axes[1].set_xlabel("Classe prédite")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Matrice de confusion sauvegardée → {save_path}")


# ------------------------------------------------------------
# 6. UTILITAIRES DIVERS
# ------------------------------------------------------------

def get_device() -> torch.device:
    """
    Retourne le device disponible : GPU (CUDA) si disponible, sinon CPU.

    Returns:
        torch.device : 'cuda' ou 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Device utilisé : {device}")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Compte le nombre de paramètres entraînables du modèle.
    Utile pour vérifier que la tête de classification est bien connectée.

    Args:
        model: Modèle PyTorch.

    Returns:
        Nombre total de paramètres entraînables.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[utils] Paramètres entraînables : {total:,}")
    return total
