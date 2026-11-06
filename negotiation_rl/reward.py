# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The verifiable reward of a negotiation episode.

Let B be the buyer's budget and C the seller's cost. When B > C, a deal at price P earns

    R = (B - P) / (B - C), clipped to [0, 1],

so R = 1 when the buyer pays the seller's cost and R = 0 when it pays its whole budget. When
B < C (a conflict of interest), every deal costs more than the budget and earns -1, so the
buyer's best move is to walk away.

Ending without a deal (either side quits, or the turn limit is reached) earns 0. A buyer error
earns -1: a reply without a readable action, a [SELL], a [BUY] or [DEAL] above the budget, or a
[DEAL] that does not copy the seller's latest offer. A seller error ends the episode with 0, since
the buyer did nothing wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

DEAL_OUTCOMES = ("buyer_deal", "seller_deal")
NO_DEAL_OUTCOMES = ("buyer_quit", "seller_quit", "deadlock")
BUYER_ERROR_OUTCOMES = (
    "buyer_format_error__no_action",  # no "Action:" line
    "buyer_format_error__outside_parse_failed",  # the action cannot be read
    "buyer_format_error__sell_action",  # the buyer used the seller's [SELL]
    "buyer_price_error",  # a [BUY] or [DEAL] above the budget
    "buyer_deal_error__no_seller_offer",  # [DEAL] before any seller offer
    "buyer_deal_error__offer_price_mismatch",  # [DEAL] at a price the seller did not offer
    "buyer_deal_error__offer_item_mismatch",  # [DEAL] on items the seller did not offer
)
SELLER_ERROR_OUTCOMES = (
    "seller_generation_error",  # the seller model could not be sampled
    "seller_format_error__no_action",
    "seller_format_error__outside_parse_failed",
    "seller_format_error__buy_action",  # the seller used the buyer's [BUY]
    "seller_price_error",  # a [SELL] or [DEAL] below the cost
    "seller_deal_error__no_buyer_offer",
    "seller_deal_error__offer_price_mismatch",
    "seller_deal_error__offer_item_mismatch",
)
OUTCOMES = DEAL_OUTCOMES + NO_DEAL_OUTCOMES + BUYER_ERROR_OUTCOMES + SELLER_ERROR_OUTCOMES


@dataclass(frozen=True)
class RewardResult:
    """The reward of an episode, its outcome label, and the buyer's share of the surplus."""

    reward: float
    outcome: str
    buyer_bargained_ratio: float | None = None
    """(B - P) / |B - C| for a deal, before clipping; None when there was no deal."""


def compute_reward(budget: float, cost: float, deal_price: float | None, outcome: str) -> RewardResult:
    """Return the reward of an episode that ended with ``outcome`` (one of ``OUTCOMES``).

    ``deal_price`` is set exactly for the deal outcomes.
    """
    # A deal: the buyer's share of the bargaining range, clipped.
    if outcome in DEAL_OUTCOMES:
        if deal_price is None:
            raise ValueError(f"deal_price must be set for outcome {outcome!r}")
        gap = abs(budget - cost)
        if gap == 0:
            return RewardResult(reward=1.0, outcome=outcome, buyer_bargained_ratio=1.0)
        ratio = (budget - deal_price) / gap
        # Conflict of interest: the seller never sells below its cost, so the deal price is at
        # least the cost, the ratio is at most -1, and the reward is -1.
        if cost > budget:
            return RewardResult(reward=max(-1.0, min(0.0, ratio)), outcome=outcome, buyer_bargained_ratio=ratio)
        return RewardResult(reward=max(0.0, min(1.0, ratio)), outcome=outcome, buyer_bargained_ratio=ratio)

    # No deal: zero. Buyer errors: -1. Seller errors: zero.
    if outcome in NO_DEAL_OUTCOMES:
        return RewardResult(reward=0.0, outcome=outcome)
    if outcome in BUYER_ERROR_OUTCOMES:
        return RewardResult(reward=-1.0, outcome=outcome)
    if outcome in SELLER_ERROR_OUTCOMES:
        return RewardResult(reward=0.0, outcome=outcome)
    raise ValueError(f"Unknown outcome: {outcome!r}")
