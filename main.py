import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from dataset import get_cifar10_dataloaders
from models import CIFARResNet, CustomCNN
from train import train_model
from distillation import train_kd_model
from evaluate import evaluate_model
from utils import set_seed, count_parameters, get_model_size_mb


def log_and_test_model(exp_name, model, checkpoint_path, test_loader, history, device, results_list):
    """
    Recharge le meilleur checkpoint sauvegardé, évalue le modèle sur le jeu de test officiel,
    calcule la matrice de confusion et enregistre toutes les métriques de l'expérience.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate_model(
        model, test_loader, criterion, device, return_preds=True
    )
    cm = confusion_matrix(labels, preds)

    total_params, trainable_params = count_parameters(model)
    size_mb = get_model_size_mb(model)

    record = {
        "Experience": exp_name,
        "Total_Params": total_params,
        "Trainable_Params": trainable_params,
        "Size_MB": round(size_mb, 2),
        "Train_Time_s": round(history["training_time"], 1),
        "Best_Val_Acc": round(history["best_val_acc"], 2),
        "Test_Loss": round(test_loss, 4),
        "Test_Acc": round(test_acc, 2),
    }
    results_list.append(record)

    print(f"=== RÉSULTAT TEST [{exp_name}] ===")
    print(
        f"Params: {total_params:,} | Taille: {size_mb:.2f} Mo | "
        f"Temps: {history['training_time']:.1f}s | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n"
    )
    return cm


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Exécution sur : {device}")

    # Hyperparamètres globaux (ajuste epochs=20 ou 30 sur GPU pour de meilleurs scores)
    batch_size = 128
    epochs = 20
    lr_pretrained = 0.01   # Learning rate plus doux pour le fine-tuning de poids ImageNet
    lr_scratch = 0.05      # Learning rate plus élevé pour l'entraînement depuis zéro
    alpha = 0.5
    beta = 0.5
    temperature = 4.0

    os.makedirs("checkpoints", exist_ok=True)
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    results = []

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 1 : Teacher ResNet50 pré-entraîné sur ImageNet -> CIFAR-10
    # -------------------------------------------------------------------------
    teacher_ckpt = "checkpoints/exp1_teacher_resnet50.pth"
    teacher = CIFARResNet(arch="resnet50", pretrained=True, num_classes=10)
    hist1 = train_model(teacher, train_loader, val_loader, epochs, lr_pretrained, device, teacher_ckpt)
    log_and_test_model("Exp1: Teacher ResNet50 (Pretrained)", teacher, teacher_ckpt, test_loader, hist1, device, results)

    # On recharge les meilleurs poids du Teacher pour superviser toutes les distillations suivantes
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 2 : Student ResNet18 pré-entraîné (Baseline sans distillation)
    # -------------------------------------------------------------------------
    student2_ckpt = "checkpoints/exp2_student_resnet18_pre.pth"
    student2 = CIFARResNet(arch="resnet18", pretrained=True, num_classes=10)
    hist2 = train_model(student2, train_loader, val_loader, epochs, lr_pretrained, device, student2_ckpt)
    log_and_test_model("Exp2: Student ResNet18 (Pretrained)", student2, student2_ckpt, test_loader, hist2, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 3 : KD Scores uniquement (ResNet50 -> ResNet18 Pretrained, MSE)
    # -------------------------------------------------------------------------
    student3_ckpt = "checkpoints/exp3_kd_scores_pre_mse.pth"
    student3 = CIFARResNet(arch="resnet18", pretrained=True, num_classes=10)
    hist3 = train_kd_model(
        student3, teacher, train_loader, val_loader, epochs, lr_pretrained, device, student3_ckpt,
        alpha=alpha, temperature=temperature, metric="mse", use_features=False
    )
    log_and_test_model("Exp3: KD Scores (Pretrained, MSE)", student3, student3_ckpt, test_loader, hist3, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 4 : KD Scores + Feature Maps (ResNet50 -> ResNet18 Pretrained, MSE)
    # -------------------------------------------------------------------------
    student4_ckpt = "checkpoints/exp4_kd_feat_pre_mse.pth"
    student4 = CIFARResNet(arch="resnet18", pretrained=True, num_classes=10)
    hist4 = train_kd_model(
        student4, teacher, train_loader, val_loader, epochs, lr_pretrained, device, student4_ckpt,
        alpha=alpha, beta=beta, temperature=temperature, metric="mse", use_features=True
    )
    log_and_test_model("Exp4: KD Scores+Feat (Pretrained, MSE)", student4, student4_ckpt, test_loader, hist4, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 5 : Student ResNet18 From Scratch (Baseline sans distillation)
    # -------------------------------------------------------------------------
    student5_ckpt = "checkpoints/exp5_student_resnet18_scratch.pth"
    student5 = CIFARResNet(arch="resnet18", pretrained=False, num_classes=10)
    hist5 = train_model(student5, train_loader, val_loader, epochs, lr_scratch, device, student5_ckpt)
    log_and_test_model("Exp5: Student ResNet18 (Scratch)", student5, student5_ckpt, test_loader, hist5, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 6 : KD Scores uniquement (ResNet50 -> ResNet18 Scratch, MSE)
    # -------------------------------------------------------------------------
    student6_ckpt = "checkpoints/exp6_kd_scores_scratch_mse.pth"
    student6 = CIFARResNet(arch="resnet18", pretrained=False, num_classes=10)
    hist6 = train_kd_model(
        student6, teacher, train_loader, val_loader, epochs, lr_scratch, device, student6_ckpt,
        alpha=alpha, temperature=temperature, metric="mse", use_features=False
    )
    log_and_test_model("Exp6: KD Scores (Scratch, MSE)", student6, student6_ckpt, test_loader, hist6, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 7 & 8 : KD Scores + Feature Maps (Scratch, MSE) -> Comparaison Stratégies 1 vs 2
    # -------------------------------------------------------------------------
    student7_ckpt = "checkpoints/exp7_kd_feat_scratch_mse.pth"
    student7 = CIFARResNet(arch="resnet18", pretrained=False, num_classes=10)
    hist7 = train_kd_model(
        student7, teacher, train_loader, val_loader, epochs, lr_scratch, device, student7_ckpt,
        alpha=alpha, beta=beta, temperature=temperature, metric="mse", use_features=True
    )
    log_and_test_model("Exp7: KD Scores+Feat (Scratch, MSE)", student7, student7_ckpt, test_loader, hist7, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 9 : Comparaison d'autres métriques de distance (KL & Cosine)
    # -------------------------------------------------------------------------
    for metric in ["kl", "cosine"]:
        ckpt_m = f"checkpoints/exp9_kd_feat_scratch_{metric}.pth"
        student_m = CIFARResNet(arch="resnet18", pretrained=False, num_classes=10)
        hist_m = train_kd_model(
            student_m, teacher, train_loader, val_loader, epochs, lr_scratch, device, ckpt_m,
            alpha=alpha, beta=beta, temperature=temperature, metric=metric, use_features=True
        )
        log_and_test_model(f"Exp9: KD Scores+Feat ({metric.upper()})", student_m, ckpt_m, test_loader, hist_m, device, results)

    # -------------------------------------------------------------------------
    # EXPÉRIENCE 10 : Custom Teacher CNN -> Custom Students (/2, /3, /4)
    # -------------------------------------------------------------------------
    custom_teacher_ckpt = "checkpoints/exp10_custom_teacher.pth"
    custom_teacher = CustomCNN(divisor=1, num_classes=10)
    hist_ct = train_model(custom_teacher, train_loader, val_loader, epochs, lr_scratch, device, custom_teacher_ckpt)
    log_and_test_model("Exp10: Custom Teacher (/1)", custom_teacher, custom_teacher_ckpt, test_loader, hist_ct, device, results)
    custom_teacher.load_state_dict(torch.load(custom_teacher_ckpt, map_location=device))

    for div in [2, 3, 4]:
        # Baseline sans distillation pour le Student /div
        base_ckpt = f"checkpoints/exp10_custom_student_div{div}_base.pth"
        s_base = CustomCNN(divisor=div, num_classes=10)
        hist_sb = train_model(s_base, train_loader, val_loader, epochs, lr_scratch, device, base_ckpt)
        log_and_test_model(f"Exp10: Custom Student /{div} (Baseline)", s_base, base_ckpt, test_loader, hist_sb, device, results)

        # Distillation Scores + Feature Maps depuis Custom Teacher
        kd_ckpt = f"checkpoints/exp10_custom_student_div{div}_kd.pth"
        s_kd = CustomCNN(divisor=div, num_classes=10)
        hist_skd = train_kd_model(
            s_kd, custom_teacher, train_loader, val_loader, epochs, lr_scratch, device, kd_ckpt,
            alpha=alpha, beta=beta, temperature=temperature, metric="kl", use_features=True
        )
        log_and_test_model(f"Exp10: Custom Student /{div} (KD)", s_kd, kd_ckpt, test_loader, hist_skd, device, results)

    # Sauvegarde JSON de tous les résultats pour le rapport de TP
    with open("results_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n================ TABLEAU RÉCAPITULATIF FINAL ================")
    for r in results:
        print(f"{r['Experience']:<38} | Params: {r['Total_Params']:>10,} | Size: {r['Size_MB']:>6.2f} MB | Test Acc: {r['Test_Acc']:>6.2f}%")


if __name__ == "__main__":
    main()