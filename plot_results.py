import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration esthétique niveau publication scientifique
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 300,
})


def load_results(json_path="results_summary.json"):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_resnet_comparison(results):
    """
    Graphique 1 : Comparaison des stratégies et métriques sur ResNet-18 (Exp. 1 à 9).
    """
    # Filtrer uniquement les expériences ResNet (Exp 1 à Exp 9)
    resnet_exps = [r for r in results if not r["Experience"].startswith("Exp10")]

    labels = [r["Experience"].replace("Exp", "E").replace(": ", "\n") for r in resnet_exps]
    accs = [r["Test_Acc"] for r in resnet_exps]

    # Code couleur par catégorie d'expérience
    colors = []
    for r in resnet_exps:
        name = r["Experience"]
        if "Teacher" in name:
            colors.append("#2c3e50")      # Bleu nuit (Teacher)
        elif "Pretrained" in name:
            colors.append("#3498db")      # Bleu clair (Pretrained)
        elif "Scratch)" in name:
            colors.append("#95a5a6")      # Gris (Baseline Scratch)
        else:
            colors.append("#27ae60")      # Vert (KD Scratch)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.7, width=0.6)

    ax.set_ylim(89.0, 98.5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Comparaison des stratégies de Knowledge Distillation (ResNet-50 → ResNet-18)")
    plt.xticks(rotation=25, ha="right")

    # Annotation exacte au-dessus de chaque barre
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("fig1_resnet_kd_comparison.png")
    plt.close()
    print("Graphique sauvegardé : fig1_resnet_kd_comparison.png")


def plot_custom_cnn_compression(results):
    """
    Graphique 2 : Impact du facteur de compression (/1, /2, /3, /4) sur CustomCNN (Exp. 10)
    — Baseline vs Knowledge Distillation.
    """
    teacher = [r for r in results if "Custom Teacher (/1)" in r["Experience"]][0]
    divisors = [2, 3, 4]

    base_accs = []
    kd_accs = []
    params_k = []

    for d in divisors:
        b = [r for r in results if f"Custom Student /{d} (Baseline)" in r["Experience"]][0]
        k = [r for r in results if f"Custom Student /{d} (KD)" in r["Experience"]][0]
        base_accs.append(b["Test_Acc"])
        kd_accs.append(k["Test_Acc"])
        params_k.append(b["Total_Params"] / 1000.0)

    x = np.arange(len(divisors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars1 = ax.bar(x - width / 2, base_accs, width, label="Baseline (Sans KD)", color="#e67e22", edgecolor="black")
    bars2 = ax.bar(x + width / 2, kd_accs, width, label="Avec Knowledge Distillation (KL + Feat)", color="#16a085", edgecolor="black")

    # Ligne horizontale de référence du Custom Teacher (/1)
    ax.axhline(
        teacher["Test_Acc"],
        color="#2c3e50",
        linestyle="--",
        linewidth=1.5,
        label=f"Custom Teacher (/1) : {teacher['Test_Acc']:.2f}% ({teacher['Total_Params']/1000:.0f}k params)",
    )

    x_labels = [f"Student /{d}\n({p:.1f}k params)" for d, p in zip(divisors, params_k)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(81.0, 91.5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Expérience 10 : Effet de la compression et gain de la KD sur CustomCNN")
    ax.legend(loc="lower left", frameon=True)

    # Affichage des scores et du gain (+Δ%)
    for b1, b2 in zip(bars1, bars2):
        h1, h2 = b1.get_height(), b2.get_height()
        gain = h2 - h1
        ax.text(b1.get_x() + b1.get_width() / 2, h1 + 0.15, f"{h1:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.text(
            b2.get_x() + b2.get_width() / 2,
            h2 + 0.15,
            f"{h2:.2f}%\n(+{gain:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#0e6655",
        )

    plt.tight_layout()
    plt.savefig("fig2_custom_cnn_compression.png")
    plt.close()
    print("Graphique sauvegardé : fig2_custom_cnn_compression.png")


def plot_pareto_efficiency(results):
    """
    Graphique 3 : Compromis Précision vs Taille du modèle (Échelle logarithmique des paramètres).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for r in results:
        params = r["Total_Params"]
        acc = r["Test_Acc"]
        name = r["Experience"]

        if "Teacher" in name:
            marker, color, size = "*", "#2c3e50", 220
        elif "(KD)" in name or "KD " in name:
            marker, color, size = "o", "#27ae60", 110
        else:
            marker, color, size = "s", "#e74c3c", 100

        ax.scatter(params, acc, c=color, marker=marker, s=size, edgecolors="black", zorder=3)

    # Annotations ciblées des points clés pour garder le graphique lisible
    highlights = {
        "Exp1: Teacher ResNet50 (Pretrained)": ("ResNet-50 Teacher", (-85, -14)),
        "Exp2: Student ResNet18 (Pretrained)": ("ResNet-18 Pretrained", (-95, 8)),
        "Exp5: Student ResNet18 (Scratch)": ("ResNet-18 Scratch", (-95, -15)),
        "Exp9: KD Scores+Feat (KL)": ("ResNet-18 KD (KL)", (-85, 8)),
        "Exp10: Custom Teacher (/1)": ("Custom Teacher (/1)", (10, -4)),
        "Exp10: Custom Student /2 (KD)": ("Student /2 (KD)", (10, 4)),
        "Exp10: Custom Student /4 (KD)": ("Student /4 (KD)", (10, 4)),
        "Exp10: Custom Student /4 (Baseline)": ("Student /4 (Base)", (10, -10)),
    }

    for r in results:
        if r["Experience"] in highlights:
            label, offset = highlights[r["Experience"]]
            ax.annotate(
                label,
                (r["Total_Params"], r["Test_Acc"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Nombre de paramètres (échelle logarithmique)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Front de Pareto : Compromis Précision vs Complexité du Modèle")

    # Légende personnalisée
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", label="Teacher", markerfacecolor="#2c3e50", markeredgecolor="k", markersize=13),
        Line2D([0], [0], marker="o", color="w", label="Student avec KD", markerfacecolor="#27ae60", markeredgecolor="k", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="Student Baseline (sans KD)", markerfacecolor="#e74c3c", markeredgecolor="k", markersize=9),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig("fig3_pareto_efficiency.png")
    plt.close()
    print("Graphique sauvegardé : fig3_pareto_efficiency.png")


if __name__ == "__main__":
    results = load_results("results_summary.json")
    plot_resnet_comparison(results)
    plot_custom_cnn_compression(results)
    plot_pareto_efficiency(results)