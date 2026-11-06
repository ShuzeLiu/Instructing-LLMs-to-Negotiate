# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Training and test datasets: batches of products, each played by a group of buyer rollouts."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

import chz
import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder

from negotiation_rl.env import NegotiationGroupBuilder
from negotiation_rl.products import DEFAULT_DATA_DIR, load_products
from negotiation_rl.seller import create_seller_completer
from negotiation_rl.spec import NegotiationSpec

SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"


def _epoch_rng(seed: int, epoch: int) -> random.Random:
    """Return the generator that orders the products of one epoch.

    Each epoch has its own generator, seeded from a hash of "<seed>::epoch_order::<epoch>", so the
    order of an epoch does not depend on how many epochs come before it.
    """
    key = f"{seed}::epoch_order::{epoch}"
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


class NegotiationDataset(RLDataset):
    """A sequence of batches; each batch is ``batch_size`` products, each a group of rollouts.

    Training data is reshuffled every epoch; each epoch has ``len(specs) // batch_size`` batches,
    so the products left over are skipped in that epoch. Test data is kept in file order.
    """

    def __init__(
        self,
        specs: list[NegotiationSpec],
        *,
        batch_size: int,
        group_size: int,
        num_epochs: int,
        shuffle_seed: int | None,
        renderer_name: str,
        buyer_model: str,
        max_trajectory_tokens: int,
        seller_completer: TinkerMessageCompleter,
    ) -> None:
        self._specs = specs
        self._batch_size = batch_size
        self._group_size = group_size
        self._renderer_name = renderer_name
        self._buyer_model = buyer_model
        self._max_trajectory_tokens = max_trajectory_tokens
        self._seller_completer = seller_completer

        # Test data: every product once, in file order (the last batch may be smaller).
        # Training data: one shuffled order per epoch, cut into full batches.
        if shuffle_seed is None:
            self._batches = [list(range(i, min(i + batch_size, len(specs)))) for i in range(0, len(specs), batch_size)]
        else:
            self._batches = []
            for epoch in range(num_epochs):
                order = list(range(len(specs)))
                _epoch_rng(shuffle_seed, epoch).shuffle(order)
                self._batches += [order[i:i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]

    def __len__(self) -> int:
        """Return the number of batches (training iterations)."""
        return len(self._batches)

    def get_batch(self, index: int) -> Sequence[NegotiationGroupBuilder]:
        """Return the rollout groups of batch ``index``, one per product."""
        return [
            NegotiationGroupBuilder(
                self._specs[i],
                group_size=self._group_size,
                renderer_name=self._renderer_name,
                buyer_model=self._buyer_model,
                max_trajectory_tokens=self._max_trajectory_tokens,
                seller_completer=self._seller_completer,
            )
            for i in self._batches[index]
        ]


def load_specs(indices_path: str | Path, data_dir: str | Path, budget_ratio: float, max_turns: int) -> list[NegotiationSpec]:
    """Build one NegotiationSpec per product index listed in a split file.

    A split file looks like {"n": 128, "indices": [28, 450, 302, ...]}.
    """
    with open(indices_path) as f:
        indices = json.load(f)["indices"]
    inventories = load_products(data_dir)
    return [NegotiationSpec.from_inventory(inventories[i], i, budget_ratio, max_turns) for i in indices]


@chz.chz
class NegotiationDatasetBuilder(RLDatasetBuilder):
    """Builds the training and test datasets and the shared seller sampler."""

    buyer_model: str
    seller_model: str
    renderer_name: str
    max_turns: int
    max_tokens: int
    batch_size: int
    group_size: int
    num_epochs: int
    seed: int
    budget_ratio: float
    seller_temperature: float
    max_trajectory_tokens: int
    train_indices_path: str = str(SPLITS_DIR / "train_indices.json")
    test_indices_path: str = str(SPLITS_DIR / "test_indices.json")
    data_dir: str = str(DEFAULT_DATA_DIR)

    async def __call__(self) -> tuple[NegotiationDataset, NegotiationDataset]:
        """Return (training dataset, test dataset)."""
        # One seller sampler, shared by every episode.
        seller_completer = create_seller_completer(
            tinker.ServiceClient(), self.seller_model, self.max_tokens, self.seller_temperature
        )

        # The two splits differ only in their products and in shuffling.
        common = dict(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer_name=self.renderer_name,
            buyer_model=self.buyer_model,
            max_trajectory_tokens=self.max_trajectory_tokens,
            seller_completer=seller_completer,
        )
        train = NegotiationDataset(
            load_specs(self.train_indices_path, self.data_dir, self.budget_ratio, self.max_turns),
            num_epochs=self.num_epochs,
            shuffle_seed=self.seed,
            **common,
        )
        test = NegotiationDataset(
            load_specs(self.test_indices_path, self.data_dir, self.budget_ratio, self.max_turns),
            num_epochs=1,
            shuffle_seed=None,
            **common,
        )
        return train, test
