import torch
import numpy as np


def evaluate_model(model, dataloader, criterion, device, return_preds=False):
    """
    Évalue un modèle PyTorch sur un DataLoader donné (validation ou test).

    Args:
        model (nn.Module): Modèle à évaluer.
        dataloader (DataLoader): DataLoader de validation ou de test.
        criterion (nn.Module): Fonction de perte (ex: nn.CrossEntropyLoss()).
        device (torch.device): Périphérique d'exécution ('cuda' ou 'cpu').
        return_preds (bool): Si True, retourne également les prédictions et vrais labels
                             (utile pour tracer une matrice de confusion).

    Returns:
        tuple: (avg_loss, accuracy) ou (avg_loss, accuracy, all_preds, all_labels)
    """
    # Passe le modèle en mode évaluation : fige les statistiques des couches BatchNorm
    # et désactive les couches Dropout
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    # Désactive le suivi des gradients 
    with torch.no_grad():
        for images, labels in dataloader:
            
            images = images.to(device)  # Forme : (B, 3, 32, 32)
            labels = labels.to(device)  # Forme : (B,)

            # Passe avant (Forward pass) -> logits bruts de forme (B, 10)
            outputs = model(images)

            # Calcul de la perte moyenne sur le batch courant
            loss = criterion(outputs, labels)

            # Pondération par la taille exacte du batch 
            total_loss += loss.item() * images.size(0)

            # Récupération de l'indice de la classe ayant le logit maximal : (B, 10) -> (B,)
            preds = torch.argmax(outputs, dim=1)

            # Accumulation du nombre de prédictions correctes sur l'ensemble du dataset
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            if return_preds:
                # Rapatriement sur CPU et conversion en NumPy pour l'analyse finale
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    if return_preds:
        # Fusionne la liste de tableaux (un par batch) en un seul vecteur 1D de taille N
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return avg_loss, accuracy, all_preds, all_labels

    return avg_loss, accuracy