import torch
import torch.nn as nn
import torchvision.models as models


class CIFARResNet(nn.Module):
    """
    Wrapper autour de ResNet-18 / ResNet-50 adapté aux images 32x32 de CIFAR-10.
    Permet d'extraire à la fois :
    - les logits finaux : (B, num_classes)
    - la feature map de layer4 : (B, C, 4, 4)
    """
    def __init__(self, arch="resnet50", pretrained=True, num_classes=10):
        super().__init__()
        if arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.feature_dim = 2048
        elif arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_dim = 512
        else:
            raise ValueError(f"Architecture non supportée : {arch}")

        # 1. Adaptation à la résolution 32x32 de CIFAR-10 :
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        self.maxpool = nn.Identity()
        
        self.layer1 = base_model.layer1  # Sortie : (B, 64 ou 256, 32, 32)
        self.layer2 = base_model.layer2  # Sortie : (B, 128 ou 512, 16, 16)
        self.layer3 = base_model.layer3  # Sortie : (B, 256 ou 1024, 8, 8)
        self.layer4 = base_model.layer4  # Sortie : (B, 512 ou 2048, 4, 4)

        self.avgpool = base_model.avgpool  # Sortie : (B, C, 1, 1)

        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features=False):
        # Entrée x : (B, 3, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat4 = self.layer4(x)  # Tenseur 4D : (B, feature_dim, 4, 4)

        out = self.avgpool(feat4)          # (B, feature_dim, 1, 1)
        out = torch.flatten(out, 1)        # (B, feature_dim)
        logits = self.fc(out)              # (B, num_classes)

        if return_features:
            return logits, feat4
        return logits


class CustomCNN(nn.Module):
    """
    Architecture CNN personnalisée pour l'Expérience 10.
    - divisor = 1 : Teacher Network (canaux : 64 -> 128 -> 256)
    - divisor = 2, 3, 4 : Student Networks compressés (/2, /3, /4)
    Conserve exactement la même interface (feature_dim + return_features) que CIFARResNet.
    """
    def __init__(self, divisor=1, num_classes=10):
        super().__init__()
        c1 = max(1, 64 // divisor)
        c2 = max(1, 128 // divisor)
        c3 = max(1, 256 // divisor)
        self.feature_dim = c3

        # Bloc 1 : (B, 3, 32, 32) -> (B, c1, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Bloc 2 : (B, c1, 16, 16) -> (B, c2, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Bloc 3 : (B, c2, 8, 8) -> (B, c3, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x, return_features=False):
        x = self.block1(x)
        x = self.block2(x)
        feat3 = self.block3(x)  # Tenseur 4D : (B, c3, 4, 4)

        out = self.avgpool(feat3)    # (B, c3, 1, 1)
        out = torch.flatten(out, 1)  # (B, c3)
        logits = self.fc(out)        # (B, num_classes)

        if return_features:
            return logits, feat3
        return logits


if __name__ == "__main__":
    from utils import count_parameters, get_model_size_mb

    teacher = CIFARResNet(arch="resnet50", pretrained=True, num_classes=10)
    student = CIFARResNet(arch="resnet18", pretrained=True, num_classes=10)

    dummy_x = torch.randn(4, 3, 32, 32)
    t_logits, t_feat = teacher(dummy_x, return_features=True)
    s_logits, s_feat = student(dummy_x, return_features=True)

    print(f"Teacher ResNet-50 | Params: {count_parameters(teacher)[0]:,} | Taille: {get_model_size_mb(teacher):.2f} Mo")
    print(f"  -> Logits: {t_logits.shape} | Feat4: {t_feat.shape}")
    print(f"Student ResNet-18 | Params: {count_parameters(student)[0]:,} | Taille: {get_model_size_mb(student):.2f} Mo")
    print(f"  -> Logits: {s_logits.shape} | Feat4: {s_feat.shape}")