import time
import torch
import torch.nn as nn
from evaluate import evaluate_model
from utils import save_checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle pendant une époque complète sur le jeu d'entraînement.

    Returns:
        tuple: (avg_train_loss, train_accuracy)
    """
    # Active le mode entraînement (mise à jour des statistiques BatchNorm et activation du Dropout)
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in dataloader:
        # 1. Transfert explicite des tenseurs sur le périphérique cible (GPU/CPU)
        images = images.to(device)  # Forme : (B, 3, 32, 32)
        labels = labels.to(device)  # Forme : (B,)

        # 2. Réinitialisation des gradients accumulés lors de l'itération précédente
        optimizer.zero_grad()

        # 3. Passe avant (Forward pass) -> logits de forme (B, 10)
        outputs = model(images)

        # 4. Calcul de la fonction de perte (Cross-Entropy)
        loss = criterion(outputs, labels)

        # 5. Rétropropagation (calcul des gradients dLoss/dW pour tous les poids)
        loss.backward()

        # 6. Mise à jour des paramètres du modèle selon la règle SGD
        optimizer.step()

        # Suivi des métriques d'entraînement
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, epochs, lr, device, save_path):
    """
    Entraîne un modèle sur plusieurs époques avec validation à chaque époque
    et sauvegarde automatique du meilleur checkpoint selon la Validation Accuracy.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimiseur SGD avec momentum et régularisation L2 (weight_decay) standard pour ResNet/CIFAR-10
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Scheduler décroissant en cosinus sur la durée totale de l'entraînement
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # Mise à jour du learning rate en fin d'époque
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Sauvegarde du meilleur modèle uniquement si la validation progresse
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, save_path)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)"
        )

    training_time = time.time() - start_time
    history["best_val_acc"] = best_val_acc
    history["training_time"] = training_time
    print(f"\nEntraînement terminé en {training_time:.1f}s | Meilleure Val Acc : {best_val_acc:.2f}%\n")

    return history