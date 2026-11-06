# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Recreate the fixed train/test split in data/splits/ (930 products: 802 train, 128 test).

Usage:
    python scripts/make_split.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from negotiation_rl.products import load_products

SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"

SEED = 42
TEST_SIZE = 128


def main() -> None:
    """Shuffle all product indices with a fixed seed; the first 128 are the test set."""
    indices = list(range(len(load_products())))
    random.Random(SEED).shuffle(indices)
    splits = {"test_indices.json": indices[:TEST_SIZE], "train_indices.json": indices[TEST_SIZE:]}

    # Write each split as {"n": <count>, "indices": [...]}.
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        with open(SPLITS_DIR / name, "w", encoding="utf-8") as f:
            json.dump({"n": len(split), "indices": split}, f, indent=2)
            f.write("\n")
        print(f"wrote {SPLITS_DIR / name} ({len(split)} products)")


if __name__ == "__main__":
    main()
