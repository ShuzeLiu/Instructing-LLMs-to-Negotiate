# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The fixed settings of one negotiation: the product, the buyer's budget and the seller's cost."""

from __future__ import annotations

from dataclasses import dataclass

from negotiation_rl.products import Inventory


@dataclass(frozen=True)
class NegotiationSpec:
    """Everything one negotiation episode needs; shared by all rollouts of a group.

    The budget is ``budget_ratio`` times the list price. Only the buyer sees the budget and only
    the seller sees the cost; the reward uses both.
    """

    inventory: Inventory
    product_index: int
    codename: str
    list_price: float
    budget: float
    cost: float
    max_turns: int

    @classmethod
    def from_inventory(
        cls, inventory: Inventory, product_index: int, budget_ratio: float, max_turns: int
    ) -> "NegotiationSpec":
        """Build the spec of a one-product inventory."""
        product = inventory.products[0]
        return cls(
            inventory=inventory,
            product_index=product_index,
            codename=product.codename,
            list_price=product.price,
            budget=product.price * budget_ratio,
            cost=product.cost,
            max_turns=max_turns,
        )

    @property
    def stated_budget(self) -> float:
        """Return the budget as the buyer is told it, rounded to cents."""
        return round(self.budget, 2)

    @property
    def shopping_list(self) -> str:
        """Return the buyer's shopping list, which states its budget.

        Example::

            codename: beauty_29
            quantity: 1
            budget: $56.00
        """
        return f"codename: {self.codename}\nquantity: 1\nbudget: ${self.stated_budget:.2f}"

    @property
    def is_conflict_of_interest(self) -> bool:
        """Return True when the cost is above the budget, so no deal can benefit both sides."""
        return self.cost > self.budget

    @property
    def category(self) -> str:
        """Return the product category, the codename without its number ("beauty_29" -> "beauty")."""
        return self.codename.rsplit("_", 1)[0]
