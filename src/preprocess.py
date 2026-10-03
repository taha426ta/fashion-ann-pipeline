"""Stage 2 - preprocess: normalize to [0, 1], split train/val, save to data/processed/."""
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


# Z-score standardization with the Fashion-MNIST training mean (0.2860) and std (0.3530)
def normalize(x):
    return (x.astype("float32") / 255.0 - 0.2860) / 0.3530


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = np.load(RAW_DIR / "train.npz")
    test = np.load(RAW_DIR / "test.npz")

    x_train, x_val, y_train, y_val = train_test_split(
        normalize(train["x"]),
        train["y"],
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )

    np.savez_compressed(OUT_DIR / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(OUT_DIR / "val.npz", x=x_val, y=y_val)
    np.savez_compressed(OUT_DIR / "test.npz", x=normalize(test["x"]), y=test["y"])
    print(f"Processed -> train {x_train.shape}, val {x_val.shape}, test {test['x'].shape}")


if __name__ == "__main__":
    main()
