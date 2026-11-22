import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate_model
from utils import save_checkpoint


class DistillationLoss(nn.Module):
    """
    Fonction de perte complète pour la Knowledge Distillation :
    - Stratégie 1 (Scores uniquement) : (1 - alpha) * L_CE + alpha * L_scores(T)
    - Stratégie 2 (Scores + Feature Maps) : Stratégie 1 + beta * L_feat
    Supporte les métriques : 'mse', 'kl' (Kullback-Leibler), et 'cosine'.
    """
    def __init__(self, alpha=0.5, beta=0.5, temperature=4.0, metric="mse", student_channels=512, teacher_channels=2048):
        
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.metric = metric.lower()

        # Couche d'adaptation (Convolution 1x1) pour projeter les Feature Maps du Student
        # (B, student_channels, H, W) vers l'espace du Teacher (B, teacher_channels, H, W)
        # sans modifier la résolution spatiale (H, W).
        if student_channels != teacher_channels:
            self.regressor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        else:
            self.regressor = nn.Identity()

    def compute_score_loss(self, student_logits, teacher_logits):
        """Calcule la divergence entre les distributions adoucies (température T) du Student et du Teacher."""
        T = self.temperature
        soft_student = F.softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)

        if self.metric == "mse":
            # Distance quadratique moyenne entre les probabilités adoucies
            return F.mse_loss(soft_student, soft_teacher)

        elif self.metric == "kl":
            # Divergence KL : F.kl_div attend log(q^S) en 1er argument et q^T en 2ème argument.
            # Multiplication par T^2 indispensable pour compenser le facteur 1/T^2 des gradients.
            log_soft_student = F.log_softmax(student_logits / T, dim=1)
            return F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

        elif self.metric == "cosine":
            # Alignement angulaire (1 - cos(theta)) entre les vecteurs de logits/scores
            return (1.0 - F.cosine_similarity(soft_student, soft_teacher, dim=1)).mean()

        else:
            raise ValueError(f"Métrique non supportée : {self.metric}. Choisir parmi 'mse', 'kl', 'cosine'.")

    def compute_feature_loss(self, student_feat, teacher_feat):
        """Calcule la distance entre les cartes de caractéristiques intermédiaires alignées."""
        # Projection du Student : (B, C_S, H, W) -> (B, C_T, H, W)
        projected_student_feat = self.regressor(student_feat)

        if self.metric == "cosine":
            # Aplatissement spatial (B, C_T * H * W) pour mesurer l'angle global des activations
            s_flat = projected_student_feat.flatten(1)
            t_flat = teacher_feat.flatten(1)
            return (1.0 - F.cosine_similarity(s_flat, t_flat, dim=1)).mean()
        else:
            # MSE standard sur les activations 4D (utilisée aussi avec 'kl' car les features ne sont pas des probabilités)
            return F.mse_loss(projected_student_feat, teacher_feat)

    def forward(self, student_logits, teacher_logits, labels, student_feat=None, teacher_feat=None):
        # 1. Perte supervisée classique sur les vrais labels (température T = 1)
        ce_loss = F.cross_entropy(student_logits, labels)

        # 2. Perte de distillation au niveau des scores (température T)
        score_loss = self.compute_score_loss(student_logits, teacher_logits)

        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * score_loss

        # 3. Perte de distillation au niveau des Feature Maps (Stratégie 2 uniquement)
        if student_feat is not None and teacher_feat is not None:
            feat_loss = self.compute_feature_loss(student_feat, teacher_feat)
            total_loss = total_loss + self.beta * feat_loss

        return total_loss


def train_kd_epoch(student_model, teacher_model, dataloader, criterion_kd, optimizer, device, use_features=False):
    """
    Exécute une époque complète d'entraînement du Student sous la supervision du Teacher figé.
    """
    student_model.train()
    teacher_model.eval()  # Garantit que les BatchNorm du Teacher restent figés

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)  # (B, 3, 32, 32)
        labels = labels.to(device)  # (B,)

        optimizer.zero_grad()

        # 1. Passe avant du Teacher SANS calcul de gradient (économie mémoire et temps)
        with torch.no_grad():
            if use_features:
                teacher_logits, teacher_feat = teacher_model(images, return_features=True)
            else:
                teacher_logits = teacher_model(images, return_features=False)
                teacher_feat = None

        # 2. Passe avant du Student AVEC suivi des gradients
        if use_features:
            student_logits, student_feat = student_model(images, return_features=True)
        else:
            student_logits = student_model(images, return_features=False)
            student_feat = None

        # 3. Calcul de la perte de distillation totale
        loss = criterion_kd(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_feat=student_feat,
            teacher_feat=teacher_feat,
        )

        # 4. Rétropropagation et mise à jour (Student + Regressor si Stratégie 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(student_logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


def train_kd_model(
    student_model,
    teacher_model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    save_path,
    alpha=0.5,
    beta=0.5,
    temperature=4.0,
    metric="mse",
    use_features=False,
):
    """
    Orchestre l'entraînement complet par Knowledge Distillation (Stratégie 1 ou Stratégie 2).
    """
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    # Verrouillage explicite des paramètres du Teacher
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    criterion_kd = DistillationLoss(
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        metric=metric,
        student_channels=student_model.feature_dim,
        teacher_channels=teacher_model.feature_dim,
    ).to(device)

    # Important : inclure les paramètres du projecteur `criterion_kd` dans l'optimiseur !
    params_to_optimize = list(student_model.parameters()) + list(criterion_kd.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Critère standard pour mesurer la validation loss à T = 1 de façon comparable aux autres modèles
    val_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_kd_epoch(
            student_model, teacher_model, train_loader, criterion_kd, optimizer, device, use_features=use_features
        )
        val_loss, val_acc = evaluate_model(student_model, val_loader, val_criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(student_model, save_path)

        mode_str = "Scores + FeatMaps" if use_features else "Scores Only"
        print(
            f"[KD - {mode_str} | {metric.upper()}] Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)"
        )

    training_time = time.time() - start_time
    history["best_val_acc"] = best_val_acc
    history["training_time"] = training_time
    print(f"\nDistillation terminée en {training_time:.1f}s | Meilleure Val Acc : {best_val_acc:.2f}%\n")

    return history