import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Fixe toutes les graines aléatoires pour garantir la reproductibilité des expériences."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """
    Calcule le nombre total de paramètres et le nombre de paramètres entraînables.
    Retourne : (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """
    Estime la taille mémoire des paramètres et buffers du modèle en Mégaoctets (Mo).
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_all_mb


def save_checkpoint(model, path):
    """Sauvegarde proprement les poids (state_dict) du modèle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)