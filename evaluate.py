"""Stage 4 - evaluate: test the model, save confusion_matrix.png and metrics.json."""
import json

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = keras.models.load_model("models/model.h5")
    test = np.load("data/processed/test.npz")

    loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
    preds = model.predict(test["x"], verbose=0).argmax(axis=1)

    cm = confusion_matrix(test["y"], preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Fashion-MNIST ANN - test accuracy {acc:.4f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)

    metrics = {"test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
