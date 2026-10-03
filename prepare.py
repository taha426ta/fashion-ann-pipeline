"""Stage 1 - prepare: download Fashion-MNIST and save raw arrays to data/raw/."""
from pathlib import Path

import numpy as np
from tensorflow import keras

OUT_DIR = Path("data/raw")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    np.savez_compressed(OUT_DIR / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(OUT_DIR / "test.npz", x=x_test, y=y_test)
    print(f"Saved raw data -> train {x_train.shape}, test {x_test.shape}")


if __name__ == "__main__":
    main()
