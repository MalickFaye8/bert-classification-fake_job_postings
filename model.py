# ============================================================
# model.py — Chargement et gestion du modèle BERT
# Devoir 3 : Fine-tuning BERT - Fake Job Postings
# ============================================================
# Contenu :
#   - Chargement du tokenizer et du modèle pré-entraîné BERT
#   - Tête de classification (fournie par BertForSequenceClassification)
#   - Sauvegarde du meilleur modèle (best val_loss)
#   - Chargement du modèle sauvegardé pour la démo Gradio
# ============================================================

import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification


# ============================================================
# 1. CHARGEMENT DU MODÈLE ET DU TOKENIZER
# ============================================================

def get_model_and_tokenizer(
    num_labels: int = 2,
    model_name: str = "bert-base-uncased",
) -> tuple:
    """
    Charge le tokenizer et le modèle BERT pré-entraîné depuis HuggingFace.

    On utilise BertForSequenceClassification qui ajoute automatiquement
    une tête de classification au-dessus de BERT :
      - Couche Dropout (régularisation, évite l'overfitting)
      - Couche Linéaire : hidden_size (768) → num_labels (2)
    La représentation [CLS] du dernier état caché est utilisée comme
    entrée de la tête de classification (convention BERT pour la classification).

    Modèle choisi : bert-base-uncased
      - 12 couches Transformer, 768 dimensions cachées, 110M paramètres
      - Pré-entraîné sur BooksCorpus + Wikipedia anglais
      - "uncased" = texte converti en minuscules avant tokenisation
        → adapté car les offres d'emploi varient en casse

    Args:
        num_labels: Nombre de classes (2 : réel / frauduleux).
        model_name: Identifiant HuggingFace du modèle pré-entraîné.

    Returns:
        Tuple (model, tokenizer) prêts à l'emploi.
    """
    print(f"[model] Chargement du tokenizer '{model_name}'...")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    print(f"[model] Chargement du modèle '{model_name}' "
          f"avec {num_labels} classes...")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        # Retourne les probabilités (softmax) en plus des logits
        # désactivé ici car on applique softmax manuellement dans demo.py
        output_attentions=False,
        output_hidden_states=False,
    )

    # Affichage de l'architecture de la tête de classification
    print(f"[model] Architecture de la tête :")
    print(f"  Dropout    : p={model.config.hidden_dropout_prob}")
    print(f"  Classifier : Linear({model.config.hidden_size} → {num_labels})")

    return model, tokenizer


# ============================================================
# 2. SAUVEGARDE DU MEILLEUR MODÈLE
# ============================================================

def save_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    save_dir: str = "best_model",
) -> None:
    """
    Sauvegarde le modèle et le tokenizer dans un dossier dédié.

    On sauvegarde à la fois le modèle ET le tokenizer pour pouvoir
    recharger l'ensemble sans dépendance externe lors de la démo Gradio.

    La sauvegarde HuggingFace crée plusieurs fichiers :
      - pytorch_model.bin  : poids du modèle
      - config.json        : hyperparamètres de l'architecture
      - vocab.txt          : vocabulaire du tokenizer
      - tokenizer_config.json, special_tokens_map.json

    Note : best_model/ est dans .gitignore car trop lourd pour GitHub.

    Args:
        model:    Modèle fine-tuné à sauvegarder.
        tokenizer: Tokenizer associé au modèle.
        save_dir: Dossier de destination.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[model] Meilleur modèle sauvegardé → {save_dir}/")


# ============================================================
# 3. CHARGEMENT DU MODÈLE SAUVEGARDÉ
# ============================================================

def load_model(
    save_dir: str = "best_model",
    num_labels: int = 2,
    device: torch.device = None,
) -> tuple:
    """
    Charge le modèle fine-tuné et le tokenizer depuis un dossier local.

    Utilisé dans demo.py pour charger le meilleur modèle sauvegardé
    pendant l'entraînement et l'exposer via l'interface Gradio.

    Args:
        save_dir:   Dossier contenant les fichiers sauvegardés.
        num_labels: Nombre de classes (doit correspondre au modèle sauvegardé).
        device:     Device sur lequel charger le modèle (cpu ou cuda).
                    Si None, détection automatique.

    Returns:
        Tuple (model, tokenizer) chargés et prêts pour l'inférence.

    Raises:
        FileNotFoundError: Si le dossier save_dir n'existe pas.
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(
            f"[model] Dossier '{save_dir}' introuvable. "
            f"Lancez d'abord train.py pour entraîner et sauvegarder le modèle."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[model] Chargement du modèle depuis '{save_dir}'...")
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    model = BertForSequenceClassification.from_pretrained(
        save_dir,
        num_labels=num_labels,
    )

    # Déplace le modèle sur le bon device
    model.to(device)

    # Mode évaluation : désactive Dropout et BatchNorm
    # OBLIGATOIRE pour l'inférence (résultats reproductibles)
    model.eval()

    print(f"[model] Modèle chargé sur {device} en mode évaluation.")
    return model, tokenizer


# ============================================================
# 4. TEST RAPIDE DU MODULE (exécution directe)
# ============================================================

if __name__ == "__main__":
    """
    Test rapide : charge BERT et vérifie un forward pass sur un batch fictif.
    """
    from utils import get_device, count_parameters

    device = get_device()

    # Chargement du modèle
    model, tokenizer = get_model_and_tokenizer(num_labels=2)
    model.to(device)

    # Nombre de paramètres entraînables
    count_parameters(model)

    # Forward pass sur un batch fictif (batch_size=2, max_length=32)
    dummy_input_ids = torch.randint(0, 1000, (2, 32)).to(device)
    dummy_attention_mask = torch.ones(2, 32, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )

    print(f"\n[model] Forward pass OK :")
    print(f"  Logits shape : {outputs.logits.shape}")  # [2, 2]
    print("\n[model] Module model.py OK !")
