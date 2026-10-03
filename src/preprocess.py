"""Stage 2 - preprocess: normalize (method from params.yaml), split train/val, save to data/processed/."""
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


# Fashion-MNIST training-set pixel mean and std on the [0, 1] scale (used by "standard")
MEAN, STD = 0.2860, 0.3530


def normalize(x, method="minmax"):
    """Scale uint8 pixels: "standard" -> z-score, "symmetric" -> [-1, 1], default "minmax" -> [0, 1]."""
    x = x.astype("float32")
    if method == "standard":
        return (x / 255.0 - MEAN) / STD
    if method == "symmetric":
        return x / 127.5 - 1.0
    return x / 255.0


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]
    method = params.get("normalization", "minmax")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = np.load(RAW_DIR / "train.npz")
    test = np.load(RAW_DIR / "test.npz")

    x_train, x_val, y_train, y_val = train_test_split(
        normalize(train["x"], method),
        train["y"],
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )

    np.savez_compressed(OUT_DIR / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(OUT_DIR / "val.npz", x=x_val, y=y_val)
    np.savez_compressed(OUT_DIR / "test.npz", x=normalize(test["x"], method), y=test["y"])
    print(f"Processed -> train {x_train.shape}, val {x_val.shape}, test {test['x'].shape}")


if __name__ == "__main__":
    main()
