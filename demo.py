# ============================================================
# demo.py — Interface de démonstration Gradio
# Devoir 3 : Fine-tuning BERT - Fake Job Postings
# ============================================================
# Tâche : Prédire le niveau d'expérience requis (required_experience)
#         à partir de la description d'une offre d'emploi (7 classes)
#
# Classes :
#   0 → Associate       | 1 → Director        | 2 → Entry level
#   3 → Executive       | 4 → Internship       | 5 → Mid-Senior level
#   6 → Not Applicable
#
# Lancement : python demo.py
# Accès     : http://127.0.0.1:7860
# ============================================================

import os

import torch
import torch.nn.functional as F
import gradio as gr

from model import load_model
from dataset import load_label_encoder
from utils import get_device


# ============================================================
# 1. CONFIGURATION
# ============================================================

SAVE_DIR = "best_model"     # Dossier contenant le modèle + label_encoder.json
MAX_LENGTH = 128             # Doit correspondre à max_length utilisé en entraînement
NUM_LABELS = 7               # 7 niveaux d'expérience


# ============================================================
# 2. CHARGEMENT DU MODÈLE ET DU LABEL ENCODER (au démarrage)
# ============================================================

print("[demo] Initialisation de la démo Gradio...")
device = get_device()

# Chargement du modèle fine-tuné (placé en mode .eval() automatiquement)
model, tokenizer = load_model(
    save_dir=SAVE_DIR,
    num_labels=NUM_LABELS,
    device=device,
)

# Chargement du mapping entier → nom de classe depuis label_encoder.json
# Exemple : {0: "Associate", 1: "Director", ..., 6: "Not Applicable"}
idx_to_class = load_label_encoder(save_dir=SAVE_DIR)

print(f"[demo] Classes chargées : {list(idx_to_class.values())}")
print("[demo] Modèle chargé. Démarrage de l'interface Gradio...\n")


# ============================================================
# 3. FONCTION DE PRÉDICTION
# ============================================================

def predict(description: str) -> dict:
    """
    Prédit le niveau d'expérience requis à partir d'une description de poste.

    Pipeline :
      1. Validation de l'entrée
      2. Tokenisation BERT (même format qu'à l'entraînement)
      3. Forward pass en mode évaluation (torch.no_grad)
      4. Softmax → probabilités sur les 7 classes
      5. Retour au format Gradio {nom_classe: probabilité}

    Args:
        description: Texte brut de la description du poste.

    Returns:
        Dictionnaire {nom_classe: probabilité} pour l'affichage Gradio.
    """
    # Validation : description non vide
    if not description or not description.strip():
        return {idx_to_class[i]: 0.0 for i in range(NUM_LABELS)}

    # --- Tokenisation ---
    # Même paramètres que dans dataset.py : max_length=128, padding, truncation
    encoding = tokenizer(
        description.strip(),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # --- Inférence sans gradient ---
    # model.eval() est déjà activé depuis load_model()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # Shape : [1, 7]

    # --- Conversion logits → probabilités ---
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # --- Format Gradio : {nom_classe: probabilité} ---
    return {idx_to_class[i]: round(probabilities[i], 4) for i in range(NUM_LABELS)}


# ============================================================
# 4. EXEMPLES PRÉ-REMPLIS (tirés du dataset réel)
# ============================================================

# 2 exemples réels du dataset fake_job_postings
# Choisis pour illustrer des niveaux d'expérience contrastés

EXAMPLES = [
    # Exemple 1 : Stage (Internship) — agence PR, offre junior
    [
        "With offices in San Francisco, Orlando and New York, Vantage PR is an "
        "award-winning public relations and social media agency servicing the tech "
        "industry. Celebrating 24 years this year, Vantage brings both passion and "
        "senior-level technology experience to help companies succeed. Vantage is known "
        "for delivering maximum exposure through top-tier media outlets for its clients "
        "by leveraging established media connections. In 2013 alone, we took home 13 "
        "major awards, including The Agency Post's Top Press-worthy PR Campaign for "
        "mobile game discovery app, Hooked, and the Stevie Gold Award for PR Campaign."
    ],
    # Exemple 2 : Dirigeant (Executive) — VP Marketing, rôle de direction globale
    [
        "Help Drive the Growth of an Industry Leading Research Firm on the Cutting Edge "
        "of Changes in the Nature of Work and the Workforce. As a member of the executive "
        "team, the Vice President of Marketing will report to the President and lead "
        "Staffing Industry Analysts marketing efforts on a global basis. This is a key "
        "role in a company that relies on marketing to drive interest in its core research "
        "membership business and attendance at its conferences and events. Specific duties "
        "include working collaboratively as a key executive on the SIA leadership team to "
        "develop and implement the company marketing strategy."
    ],
]


# ============================================================
# 5. INTERFACE GRADIO
# ============================================================

demo = gr.Interface(
    fn=predict,

    # --- Input : description uniquement ---
    inputs=gr.Textbox(
        label="Description du poste (Job Description)",
        placeholder=(
            "Collez ici la description complète de l'offre d'emploi...\n\n"
            "Exemple : We are looking for a software engineer with 5+ years of experience..."
        ),
        lines=12,
    ),

    # --- Output : probabilités des 7 classes ---
    # gr.Label affiche les classes triées par probabilité décroissante
    outputs=gr.Label(
        num_top_classes=7,
        label="Niveau d'expérience prédit (required_experience)",
    ),

    # --- Métadonnées ---
    title="Prédicteur de Niveau d'Expérience Requis",
    description=(
        "## BERT Fine-tuned — Job Experience Level Classifier\n\n"
        "Ce modèle utilise **BERT (`bert-base-uncased`)** fine-tuné sur le dataset "
        "**Fake Job Postings** pour prédire le niveau d'expérience requis "
        "(`required_experience`) à partir de la description d'un poste.\n\n"
        "**7 classes :** Associate | Director | Entry level | Executive | "
        "Internship | Mid-Senior level | Not Applicable\n\n"
        "**Dataset :** 10 830 offres d'emploi (après suppression des NaN) | "
        "Déséquilibre 27:1 (Mid-Senior vs Executive)\n\n"
        "**Comment utiliser :**\n"
        "1. Collez la description complète de l'offre d'emploi\n"
        "2. Cliquez sur **Submit**\n"
        "3. Les probabilités de chaque niveau d'expérience s'affichent\n\n"
        "> Ce modèle est un prototype académique — Master DIT Deep Learning 2"
    ),

    # Exemples tirés du dataset réel (Internship et Executive)
    examples=EXAMPLES,

    # Texte du pied de page
    article=(
        "**Devoir 3 — Master DIT Deep Learning 2**\n\n"
        "Fine-tuning BERT avec PyTorch | Classification multi-classes (7 classes) | "
        "Suivi WandB\n\n"
        "Modèle : `bert-base-uncased` | Optimiseur : AdamW | "
        "Loss : CrossEntropyLoss + class_weights"
    ),

    # Gradio 6.0 : flagging_mode remplace allow_flagging
    flagging_mode="never",
)


# ============================================================
# 6. LANCEMENT
# ============================================================

if __name__ == "__main__":
    """
    Lancement : python demo.py
    Accès     : http://127.0.0.1:7860
    """
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,         # True pour générer un lien public temporaire Gradio
        show_error=True,
        theme=gr.themes.Soft(),
    )
