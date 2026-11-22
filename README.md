# Knowledge Distillation for Images Classification (CIFAR-10)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-ResNet%20%7C%20CIFAR--10-EE4C2C.svg)](https://pytorch.org/vision/stable/index.html)
[![NumPy](https://img.shields.io/badge/NumPy-1.x%20%7C%202.x-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)


Implémentation modulaire en **PyTorch** et étude expérimentale de la **Knowledge Distillation (KD)** appliquée à la classification d'images sur le jeu de données **CIFAR-10** ($32 \times 32$).


Ce projet évalue le transfert de connaissances d'un réseau *Teacher* volumineux vers un réseau *Student* compact selon deux stratégies (**Scores uniquement** vs **Scores + Feature Maps**), trois métriques de distance (**MSE**, **KL Divergence**, **Cosine Similarity**) et plusieurs niveaux de compression architecturale. Il est réalisé dans le cadre du **Master 2 MVI de l'Université Paris Cité**.

---

## Architecture du Projet

```text
Knowledge-Distillation-For-Images-Classification/
├── dataset.py                      # Chargement de CIFAR-10, Data Augmentation et split Train/Val/Test (45k/5k/10k)
├── models.py                       # Architectures CIFARResNet (ResNet-18/50 adaptées 32x32) et CustomCNN (/1, /2, /3, /4)
├── train.py                        # Boucle d'entraînement supervisé classique (SGD + CosineAnnealingLR)
├── distillation.py                 # Classe DistillationLoss (MSE, KL, Cosine + projecteur 1x1) et boucle KD
├── evaluate.py                     # Évaluation des modèles (Loss, Accuracy, prédictions)
├── utils.py                        # Reproductibilité (seed), comptage des paramètres, taille mémoire et checkpoints
├── main.py                         # Orchestration des 16 expériences et export dans results_summary.json
├── plot_results.py                 # Génération des graphiques comparatifs haute résolution
├── results_summary.json            # Résultats quantitatifs complets des expériences
├── fig1_resnet_kd_comparison.png   # Comparaison des stratégies sur ResNet-18
├── fig2_custom_cnn_compression.png # Impact de la compression sur CustomCNN
├── fig3_pareto_efficiency.png      # Front de Pareto (Précision vs Paramètres)
├── requirements.txt                # Dépendances Python du projet
└── .gitignore                      # Exclusion des dossiers lourds (.venv/, data/, checkpoints/)
```

---

## Installation et Exécution

### 1. Création de l'environnement virtuel
```bash
git clone https://github.com/Fallou-Diouf/Knowledge-Distillation-For-Images-Classification.git
cd Knowledge-Distillation-For-Images-Classification

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Lancer l'ensemble des expériences
```bash
python main.py
```

### 3. Générer les figures à partir des résultats sauvegardés
```bash
python plot_results.py
```

---

## Synthèse des Résultats Expérimentaux (CIFAR-10)

### 1. Transfert ResNet-50 $\rightarrow$ ResNet-18 (20 époques, $T=4.0$, $\alpha=0.5$, $\beta=0.5$)

| Expérience | Modèle & Stratégie | Paramètres | Taille | Test Loss | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp 1** | Teacher ResNet-50 (Pretrained) | 23 520 842 | 89,93 Mo | 0,1227 | **97,06 %** |
| **Exp 2** | Student ResNet-18 (Pretrained, Baseline) | 11 173 962 | 42,66 Mo | 0,1731 | **95,45 %** |
| **Exp 3** | KD Scores uniquement (Pretrained, MSE) | 11 173 962 | 42,66 Mo | 0,1766 | 94,87 % |
| **Exp 4** | KD Scores + Feature Maps (Pretrained, MSE) | 11 173 962 | 42,66 Mo | 0,1635 | 95,29 % |
| **Exp 5** | Student ResNet-18 (Scratch, Baseline) | 11 173 962 | 42,66 Mo | 0,2651 | 91,57 % |
| **Exp 6** | KD Scores uniquement (Scratch, MSE) | 11 173 962 | 42,66 Mo | 0,2778 | 91,29 % |
| **Exp 7** | KD Scores + Feature Maps (Scratch, MSE) | 11 173 962 | 42,66 Mo | 0,2713 | 91,90 % |
| **Exp 9a** | KD Scores + Feature Maps (Scratch, **KL**) | 11 173 962 | 42,66 Mo | 0,3916 | **92,23 %** |
| **Exp 9b** | KD Scores + Feature Maps (Scratch, **Cosine**) | 11 173 962 | 42,66 Mo | 0,2803 | 92,00 % |

### 2. Compression architecturale sur `CustomCNN` (Stratégie 2, KL Divergence)

| Modèle | Paramètres | Taille | Test Acc (Baseline) | Test Acc (Avec KD) | Gain KD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom Teacher ($/1$)** | 1 148 874 | 4,39 Mo | 89,60 % | — | — |
| **Custom Student $/2$** | 288 746 (-74,9 %) | 1,10 Mo | 87,69 % | **88,41 %** | **+0,72 %** |
| **Custom Student $/3$** | 126 957 (-88,9 %) | 0,49 Mo | 85,45 % | **86,44 %** | **+0,99 %** |
| **Custom Student $/4$** | 72 954 (-93,6 %) | 0,28 Mo | 83,87 % | **85,01 %** | **+1,14 %** |

---

## Principaux Enseignements
* **Supériorité des Feature Maps :** L'alignement spatial via convolution $1 \times 1$ améliore systématiquement la distillation des scores seuls ($+0{,}42\,\%$ en pré-entraîné, $+0{,}61\,\%$ *from scratch*).
* **Choix de la métrique :** La divergence KL surpasse la similarité cosinus et la MSE ($92{,}23\,\%$ vs $92{,}00\,\%$ et $91{,}90\,\%$).
* **Amplification du gain sous forte compression :** Plus le *Student* est réduit ($/2 \rightarrow /3 \rightarrow /4$), plus le guidage du *Teacher* compense la perte de capacité ($+0{,}72\,\% \rightarrow +0{,}99\,\% \rightarrow +1{,}14\,\%$).