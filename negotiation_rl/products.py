# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# Derived from the AmazonHistoryPrice bargaining benchmark
# (https://github.com/TianXiaSJTU/AmazonPriceHistory, Apache License 2.0), file product.py.
"""Loading the AmazonHistoryPrice products and formatting their listings.

Each product becomes a one-item inventory. Its list price is the larger of the listing's
``list_price`` and ``highest_price``; its seller cost is the ``lowest_price`` it has sold for.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "AmazonHistoryPrice"


@dataclass
class Product:
    """One product: its listing, list price, the seller's private cost, and a codename."""

    title: str
    description: str
    price: float
    cost: float
    codename: str

    def seller_text(self) -> str:
        """Return the listing as the seller sees it (with the cost)."""
        return (
            f'Title: "{self.title}"\n'
            f'Description: "{self.description}"\n'
            f"List Price: ${self.price}\n"
            f"Cost: ${self.cost}\n\n"
        )

    def buyer_text(self) -> str:
        """Return the listing as the buyer sees it (without the cost)."""
        return (
            f'Title: "{self.title}"\n'
            f'Description: "{self.description}"\n'
            f"List Price: ${self.price}\n\n"
        )


class Inventory:
    """The products on sale in one negotiation (one product in this dataset)."""

    def __init__(self, products: list[Product]) -> None:
        self.products = products

    def buyer_list(self) -> str:
        """Return the inventory list shown to the buyer.

        Example::

            Inventory List:
            Product1
            Codename: beauty_29
            Title: "..."
            Description: "..."
            List Price: $70.0
        """
        entries = [
            f"Product{i + 1}\nCodename: {p.codename}\n{p.buyer_text()}" for i, p in enumerate(self.products)
        ]
        return ("Inventory List:\n" + "".join(entries)).strip()

    def seller_list(self) -> str:
        """Return the inventory list shown to the seller (the buyer's list plus a Cost line)."""
        entries = [
            f"Product{i + 1}\nCodename: {p.codename}\n{p.seller_text()}" for i, p in enumerate(self.products)
        ]
        return ("Inventory List:\n" + "".join(entries)).strip()


def _parse_price(text) -> float:
    """Read a price such as "$1,299.99"; a missing or unreadable price becomes $0.01.

    before: "$1,299.99"  after: 1299.99
    before: None         after: 0.01
    """
    if not isinstance(text, str):
        return 0.01
    try:
        return max(0.01, float(text.removeprefix("$").replace(",", "").strip()))
    except ValueError:
        return 0.01


def _make_product(record: dict, number: int) -> Product:
    """Turn one dataset record into a Product with codename "<category>_<number>".

    before: {"title": "National Tree ...", "category": "home-kitchen", "list_price": "$469.99",
             "highest_price": "$549.99", "lowest_price": "$174.99", "description": "...", ...}
    after:  Product(title="National Tree ...", description="...", price=549.99, cost=174.99,
                    codename="home-kitchen_<number>")
    """
    # Text: fall back to the feature list when the description is nearly empty.
    title = record.get("title", "").strip() or record.get("amazon_title", "Unknown")
    features = record.get("features", "") or ""
    description = record.get("description", "") or ""
    if len(str(description)) < 5 and len(str(features)) > len(str(description)):
        description = features
    if not description:
        description = title[:200]

    # Prices: list price is the higher of list and highest price; cost is the lowest price.
    price = max(_parse_price(record.get("list_price")), _parse_price(record.get("highest_price")), 0.01)
    cost = _parse_price(record.get("lowest_price"))
    if price < cost:
        price = cost * 1.1

    category = record.get("category", "other").replace(" ", "-")
    return Product(title, description, price, cost, f"{category}_{number}")


def load_products(data_dir: str | os.PathLike | None = None) -> list[Inventory]:
    """Load every product, one Inventory each, in a fixed order (files sorted by name).

    The position in the returned list is the product index used by the train/test split files.
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No dataset files in {data_dir}; install the repository with pip install -e .")
    inventories: list[Inventory] = []
    for path in paths:
        # Each file holds a list of records (or a single record).
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = [records]
        # Codenames are numbered from 1 across all files, in loading order.
        for record in records:
            inventories.append(Inventory([_make_product(record, len(inventories) + 1)]))
    return inventories
