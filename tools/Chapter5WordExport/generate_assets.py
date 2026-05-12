from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(r"E:\IoT_FedProto")
OUT_DIR = ROOT / ".tmp" / "chapter5_word_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_accuracy_f1():
    labels = [
        "MLP-Local",
        "MLP-FedAvg",
        "MLP-FedProto",
        "CNN1D-Local",
        "CNN1D-FedAvg",
        "CNN1D-FedProto",
        "Trans-Local",
        "Trans-FedAvg",
        "Trans-FedProto",
    ]
    acc = [0.9392, 0.8684, 0.9384, 0.9416, 0.8976, 0.9484, 0.9292, 0.8856, 0.9300]
    f1 = [0.9384, 0.5557, 0.9377, 0.9412, 0.6087, 0.9481, 0.9280, 0.5891, 0.9288]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(11.5, 4.8), dpi=220)
    plt.bar(x - width / 2, acc, width, label="Accuracy", color="#4C78A8")
    plt.bar(x + width / 2, f1, width, label="F1", color="#F58518")
    plt.ylim(0.5, 1.0)
    plt.ylabel("Score")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=False, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5-1-accuracy-f1.png", bbox_inches="tight")
    plt.close()


def save_comm_latency():
    labels = [
        "MLP\nLocal",
        "MLP\nFedAvg",
        "MLP\nFedProto",
        "CNN1D\nLocal",
        "CNN1D\nFedAvg",
        "CNN1D\nFedProto",
        "Trans\nLocal",
        "Trans\nFedAvg",
        "Trans\nFedProto",
    ]
    latency = np.array([0.0362, 0.0360, 0.0356, 0.0691, 0.0722, 0.0808, 0.1413, 0.1500, 0.1507])
    comm = np.array([0, 1828440, 51200, 0, 3220440, 51200, 0, 3076440, 51200], dtype=float)

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11.8, 5), dpi=220)
    bars = ax1.bar(x, comm, color="#54A24B", width=0.6, label="Communication parameters")
    ax1.set_ylabel("Communication parameters")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, latency, color="#E45756", marker="o", linewidth=2, label="Latency (ms/sample)")
    ax2.set_ylabel("Latency (ms/sample)")

    handles = [bars, ax2.lines[0]]
    ax1.legend(handles, ["Communication parameters", "Latency (ms/sample)"], frameon=False, loc="upper left")
    fig.tight_layout()
    plt.savefig(OUT_DIR / "fig5-3-comm-latency.png", bbox_inches="tight")
    plt.close()


def add_label(image: Image.Image, text: str) -> Image.Image:
    labeled = Image.new("RGB", (image.width, image.height + 44), "white")
    labeled.paste(image, (0, 0))
    draw = ImageDraw.Draw(labeled)
    draw.text((image.width // 2 - len(text) * 4, image.height + 12), text, fill="black")
    return labeled


def save_tsne_triptych():
    paths = [
        (ROOT / "results" / "CNN1D" / "Local" / "figures" / "IoT_Local_IoT_CNN1D_test_0_feature_tsne.png", "Local"),
        (ROOT / "results" / "CNN1D" / "FedAvg" / "figures" / "IoT_FedAvg_IoT_CNN1D_test_0_feature_tsne.png", "FedAvg"),
        (ROOT / "results" / "CNN1D" / "FedProto" / "figures" / "IoT_FedProto_IoT_CNN1D_test_0_feature_tsne.png", "FedProto"),
    ]

    images = []
    for path, label in paths:
        img = Image.open(path).convert("RGB")
        img = ImageOps.contain(img, (1050, 800))
        images.append(add_label(img, label))

    width = sum(img.width for img in images) + 40
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), "white")

    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + 20

    canvas.save(OUT_DIR / "fig5-2-cnn1d-tsne-triptych.png")


def save_heterogeneous_tsne():
    src = ROOT / "results" / "heterogeneous_models" / "FedProto" / "figures" / "IoT_FedProto_IoT_MIX_MLP_CNN1D_test_0_feature_tsne.png"
    img = Image.open(src).convert("RGB")
    img = ImageOps.contain(img, (1600, 1000))
    img.save(OUT_DIR / "fig5-4-heterogeneous-tsne.png")


if __name__ == "__main__":
    save_accuracy_f1()
    save_comm_latency()
    save_tsne_triptych()
    save_heterogeneous_tsne()
    print(OUT_DIR)
