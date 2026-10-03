"""Stage 3 - train: build the ANN from params.yaml, train it, save models/model.h5 + history."""
import csv
from pathlib import Path

import numpy as np
import yaml
from tensorflow import keras

MODEL_DIR = Path("models")


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    keras.utils.set_random_seed(p["seed"])
    train = np.load("data/processed/train.npz")
    val = np.load("data/processed/val.npz")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train["x"], train["y"],
        validation_data=(val["x"], val["y"]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        verbose=2,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "model.h5")

    h = history.history
    with open(MODEL_DIR / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *h.keys()])
        for i in range(len(h["loss"])):
            writer.writerow([i + 1, *(round(h[k][i], 5) for k in h)])
    print("Saved models/model.h5 and models/history.csv")


if __name__ == "__main__":
    main()
