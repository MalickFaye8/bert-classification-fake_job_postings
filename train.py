# ============================================================
# train.py — Boucle d'entraînement manuelle PyTorch + WandB
# Devoir 3 : Fine-tuning BERT - Fake Job Postings
# ============================================================
# Contenu :
#   - train_epoch() : une epoch d'entraînement (boucle PyTorch pure)
#   - eval_epoch()  : une epoch d'évaluation (mode .eval() + no_grad)
#   - main()        : pipeline complet avec WandB logging
#
# CONTRAINTE ÉNONCÉ : Pas d'utilisation du Trainer HuggingFace.
# La boucle d'entraînement est entièrement manuelle, comme au Devoir 2.
# ============================================================

import os
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

from dataset import explore_dataset, load_and_split_data, get_dataloaders
from model import get_model_and_tokenizer, save_model
from utils import (
    set_seed,
    get_device,
    count_parameters,
    compute_metrics,
    get_class_weights,
    plot_curves,
    plot_confusion_matrix,
    print_classification_report,
)

# Chargement des variables d'environnement (.env pour WANDB_API_KEY)
load_dotenv()


# ============================================================
# 1. HYPERPARAMÈTRES PAR DÉFAUT
# ============================================================

# Regroupés ici pour être facilement modifiables et loggés dans WandB
DEFAULT_CONFIG = {
    # Modèle
    "model_name": "bert-base-uncased",
    "num_labels": 7,           # 7 classes : Associate, Director, Entry level,
                               # Executive, Internship, Mid-Senior level, Not Applicable
    "max_length": 128,         # 128 sur CPU ; repasser à 256 sur GPU (Colab)
                               # La médiane des descriptions est 157 mots

    # Entraînement
    "epochs": 4,               # BERT converge vite, 3-5 epochs suffisent
    "batch_size": 16,          # 16 recommandé pour BERT (réduire à 8 si RAM faible)
    "learning_rate": 2e-5,     # Typique pour fine-tuning BERT (2e-5 à 5e-5)
    "weight_decay": 0.01,      # Régularisation L2 dans AdamW
    "warmup_ratio": 0.1,       # 10% des steps en warmup (montée progressive du LR)
    "seed": 42,

    # Chemins
    "data_path": os.path.join("data", "fake_job_postings 2.csv"),
    "save_dir": "best_model",
    "figures_dir": "figures",

    # WandB
    "wandb_project": "bert-fake-job-postings",
    "wandb_run_name": "bert-required-experience-run1",
}


# ============================================================
# 2. BOUCLE D'ENTRAÎNEMENT — UNE EPOCH
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Exécute une epoch complète d'entraînement sur le DataLoader fourni.

    Étapes pour chaque batch :
      1. Déplacer les données sur le device (CPU/GPU)
      2. Remettre les gradients à zéro (évite l'accumulation)
      3. Forward pass : calculer les logits
      4. Calculer la loss (CrossEntropy avec class weights)
      5. Backward pass : calculer les gradients
      6. Clipper les gradients (évite l'explosion des gradients avec BERT)
      7. Mettre à jour les poids (optimizer.step)
      8. Mettre à jour le scheduler de learning rate

    Args:
        model:     Modèle BERT en mode entraînement.
        loader:    DataLoader du jeu d'entraînement.
        optimizer: Optimiseur AdamW.
        scheduler: Scheduler linéaire avec warmup.
        loss_fn:   CrossEntropyLoss avec poids de classes.
        device:    Device PyTorch (cpu ou cuda).
        epoch:     Numéro d'epoch courant (pour l'affichage).

    Returns:
        Dictionnaire {'loss': float, 'accuracy': float, 'f1_weighted': float}.
    """
    # Mode entraînement : active Dropout et BatchNorm
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    # Barre de progression avec tqdm
    progress_bar = tqdm(
        loader,
        desc=f"  Epoch {epoch} [Train]",
        leave=False,
        ncols=100,
    )

    for batch in progress_bar:
        # --- 1. Déplacement des données sur le device ---
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # --- 2. Remise à zéro des gradients ---
        # IMPORTANT : sans cela, les gradients s'accumulent entre les batches
        optimizer.zero_grad()

        # --- 3. Forward pass ---
        # BERT retourne un objet SequenceClassifierOutput
        # attention_mask est CRUCIAL : indique quels tokens sont du padding
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # Shape : [batch_size, num_labels]

        # --- 4. Calcul de la loss ---
        # labels doit être un tenseur d'entiers (pas one-hot)
        loss = loss_fn(logits, labels)

        # --- 5. Backward pass : calcul des gradients ---
        loss.backward()

        # --- 6. Gradient clipping (norme max = 1.0) ---
        # Stabilise l'entraînement de BERT qui a tendance à avoir
        # des gradients explosifs lors du fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- 7. Mise à jour des poids ---
        optimizer.step()

        # --- 8. Mise à jour du scheduler ---
        scheduler.step()

        # --- Collecte des métriques ---
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        # Mise à jour de la barre de progression
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],       # Métrique principale multi-classes
        "f1_weighted": metrics["f1_weighted"],
    }


# ============================================================
# 3. BOUCLE D'ÉVALUATION — UNE EPOCH
# ============================================================

def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Exécute une epoch complète d'évaluation sur le DataLoader de validation.

    DIFFÉRENCES IMPORTANTES par rapport à train_epoch :
      - model.eval()        : désactive Dropout → prédictions déterministes
      - torch.no_grad()     : désactive le calcul des gradients → économise
                              la mémoire et accélère l'évaluation
      - Pas d'optimizer.step() ni de scheduler.step()

    Args:
        model:   Modèle BERT en mode évaluation.
        loader:  DataLoader du jeu de validation.
        loss_fn: CrossEntropyLoss (mêmes poids que l'entraînement).
        device:  Device PyTorch (cpu ou cuda).
        epoch:   Numéro d'epoch courant (pour l'affichage).

    Returns:
        Dictionnaire {'loss', 'accuracy', 'f1_weighted', 'preds', 'labels'}.
    """
    # Mode évaluation : désactive Dropout et BatchNorm
    # OBLIGATOIRE — oublier cela dégrade les performances de validation
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(
        loader,
        desc=f"  Epoch {epoch} [Val]  ",
        leave=False,
        ncols=100,
    )

    # Désactivation du calcul des gradients pendant la validation
    # Réduit la consommation mémoire de ~50% et accélère l'inférence
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass sans gradient
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)

    return {
        "loss": round(avg_loss, 4),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],       # Métrique principale multi-classes
        "f1_weighted": metrics["f1_weighted"],
        "preds": all_preds,    # Retournés pour la matrice de confusion finale
        "labels": all_labels,
    }


# ============================================================
# 4. PIPELINE PRINCIPAL
# ============================================================

def main(config: dict = None) -> None:
    """
    Pipeline complet d'entraînement BERT avec logging WandB.

    Étapes :
      1. Initialisation (seed, device, WandB)
      2. Exploration du dataset (stats obligatoires selon l'énoncé)
      3. Chargement modèle + tokenizer
      4. Création datasets et DataLoaders
      5. Configuration optimizer, scheduler et loss
      6. Boucle d'entraînement (N epochs)
      7. Logging WandB à chaque epoch
      8. Sauvegarde du meilleur modèle (best val_loss)
      9. Visualisations finales + rapport WandB

    Args:
        config: Dictionnaire de configuration (hyperparamètres).
                Si None, utilise DEFAULT_CONFIG.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # ----------------------------------------------------------
    # 4.1 Initialisation
    # ----------------------------------------------------------
    set_seed(config["seed"])
    device = get_device()

    # --- Initialisation WandB ---
    # WandB va tracker toutes les métriques et hyperparamètres.
    # La clé API doit être dans .env : WANDB_API_KEY=votre_cle
    # ou configurée via : wandb login
    #
    # Fix Windows : start_method="thread" évite le bug asyncio/ConnectionResetError
    # qui se produit avec le mode "spawn" (défaut sur Windows). Le mode "thread"
    # est légèrement moins isolé mais fonctionne parfaitement pour notre usage.
    print("\n[train] Initialisation WandB...")
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config={
            # Log de tous les hyperparamètres pour reproductibilité
            "model_name": config["model_name"],
            "num_labels": config["num_labels"],
            "max_length": config["max_length"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "warmup_ratio": config["warmup_ratio"],
            "seed": config["seed"],
            "dataset": "fake_job_postings",
            "task": "multi-class: description → required_experience (7 classes)",
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss + class_weights (27:1 imbalance)",
            "scheduler": "linear_with_warmup",
            "metric_principal": "f1_macro",
        },
        settings=wandb.Settings(start_method="thread"),  # Fix Windows asyncio bug
    )

    # ----------------------------------------------------------
    # 4.2 Exploration du dataset
    # ----------------------------------------------------------
    print("\n[train] Exploration du dataset...")
    explore_dataset(config["data_path"])

    # ----------------------------------------------------------
    # 4.3 Chargement modèle + tokenizer
    # ----------------------------------------------------------
    model, tokenizer = get_model_and_tokenizer(
        num_labels=config["num_labels"],
        model_name=config["model_name"],
    )
    model.to(device)
    count_parameters(model)

    # ----------------------------------------------------------
    # 4.4 Création des datasets et DataLoaders
    # ----------------------------------------------------------
    # load_and_split_data retourne aussi le LabelEncoder pour récupérer
    # les noms des classes et sauvegarder le mapping (utilisé par demo.py)
    train_dataset, val_dataset, y_train, label_encoder = load_and_split_data(
        csv_path=config["data_path"],
        tokenizer=tokenizer,
        max_length=config["max_length"],
        seed=config["seed"],
        save_dir=config["save_dir"],
    )

    # Noms des classes dans l'ordre du LabelEncoder (alphabétique)
    class_names = list(label_encoder.classes_)

    train_loader, val_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config["batch_size"],
    )

    # ----------------------------------------------------------
    # 4.5 Optimiseur, Scheduler et Loss
    # ----------------------------------------------------------

    # AdamW : variante d'Adam avec weight decay décorrélé
    # C'est l'optimiseur standard pour le fine-tuning de BERT
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eps=1e-8,  # Terme epsilon pour la stabilité numérique
    )

    # Scheduler linéaire avec warmup :
    # Le LR monte progressivement pendant warmup_steps, puis redescend
    # Évite le "catastrophic forgetting" en début d'entraînement
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n[train] Steps totaux : {total_steps} | Warmup : {warmup_steps}")

    # CrossEntropyLoss avec poids de classes
    # Déséquilibre 27:1 (Mid-Senior 3809 vs Executive 141) → poids inversement
    # proportionnels à la fréquence pour équilibrer l'apprentissage sur les 7 classes
    class_weights = get_class_weights(y_train, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ----------------------------------------------------------
    # 4.6 Boucle d'entraînement
    # ----------------------------------------------------------

    # Historique des métriques pour les courbes (sauvegardées en local)
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\n[train] Début de l'entraînement sur {config['epochs']} epochs...\n")

    for epoch in range(1, config["epochs"] + 1):
        print(f"{'='*60}")
        print(f"  EPOCH {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        # --- Entraînement ---
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )

        # --- Évaluation ---
        val_metrics = eval_epoch(
            model, val_loader, loss_fn, device, epoch
        )

        # Récupération du learning rate courant (pour le logger dans WandB)
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Affichage des métriques ---
        print(f"\n  Train → Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1-macro: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   → Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  LR    → {current_lr:.2e}")

        # --- Logging WandB ---
        # F1-macro est la métrique principale (traite toutes les classes également)
        # F1-weighted est loggé en secondaire pour comparaison
        wandb.log({
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "train/f1_macro": train_metrics["f1_macro"],
            "train/f1_weighted": train_metrics["f1_weighted"],
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "val/f1_macro": val_metrics["f1_macro"],
            "val/f1_weighted": val_metrics["f1_weighted"],
            "learning_rate": current_lr,
        })

        # --- Mise à jour de l'historique local ---
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])  # F1-macro en historique

        # --- Sauvegarde du meilleur modèle ---
        # On sauvegarde le modèle avec la plus faible val_loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_model(model, tokenizer, config["save_dir"])
            print(f"  ✓ Meilleur modèle sauvegardé (val_loss={best_val_loss:.4f})")

            # Marqueur WandB pour identifier les meilleurs checkpoints
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_f1_macro"] = val_metrics["f1_macro"]
            wandb.run.summary["best_epoch"] = best_epoch

    # ----------------------------------------------------------
    # 4.7 Rapport final et visualisations
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT TERMINÉ")
    print(f"  Meilleur modèle : epoch {best_epoch} | val_loss={best_val_loss:.4f}")
    print(f"{'='*60}\n")

    # Rapport de classification détaillé par classe (precision, recall, F1)
    # Les noms des classes viennent du LabelEncoder → cohérence garantie
    print_classification_report(
        val_metrics["preds"],
        val_metrics["labels"],
        class_names=class_names,
    )

    # Courbes d'apprentissage (sauvegardées localement)
    plot_curves(history, save_dir=config["figures_dir"])

    # Matrice de confusion avec les 7 noms de classes réels
    plot_confusion_matrix(
        val_metrics["preds"],
        val_metrics["labels"],
        class_names=class_names,
        save_dir=config["figures_dir"],
    )

    # Upload des images de courbes vers WandB pour le rapport
    wandb.log({
        "curves/learning_curves": wandb.Image(
            os.path.join(config["figures_dir"], "learning_curves.png")
        ),
        "curves/confusion_matrix": wandb.Image(
            os.path.join(config["figures_dir"], "confusion_matrix.png")
        ),
    })

    # Clôture de la session WandB
    wandb.finish()
    print("\n[train] Session WandB fermée. Entraînement complet !")


# ============================================================
# 5. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    """
    Lancement de l'entraînement :
        python train.py
        python train.py --epochs 3 --batch_size 32 --lr 3e-5
    """
    parser = argparse.ArgumentParser(
        description="Fine-tuning BERT pour la classification description → required_experience (7 classes)"
    )

    # Arguments en ligne de commande (permettent de modifier les hyperparamètres
    # sans éditer le code — utile pour les expériences avec WandB)
    parser.add_argument("--epochs", type=int,
                        default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int,
                        default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float,
                        default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max_length", type=int,
                        default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--wandb_run_name", type=str,
                        default=DEFAULT_CONFIG["wandb_run_name"])

    args = parser.parse_args()

    # Construction du config à partir des arguments CLI
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["max_length"] = args.max_length
    config["seed"] = args.seed
    config["wandb_run_name"] = args.wandb_run_name

    main(config)
