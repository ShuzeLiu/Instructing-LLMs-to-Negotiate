# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The negotiation environment: the trained buyer bargains with a fixed LLM seller.

One episode is one negotiation over one product. Each buyer turn is a reply in the
Thought / Talk / Action format (see ``prompts.py``). After a [BUY] or [REJECT] the seller
answers; the other side only ever sees the Talk and Action lines of a reply, never its
Thought. The episode ends on a deal, on a [QUIT] by either side, on an error, or after
``max_turns`` buyer turns. The reward is given at the end (see ``reward.py``).

The seller is held to its cost and to the rules: a seller reply that offers or accepts below the
cost, breaks the reply format, uses [BUY], or uses [DEAL] on anything but the buyer's latest offer
is sampled again, as is a failed request (up to ``SELLER_MAX_ATTEMPTS`` samples in all), before
the episode ends with a seller error.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, get_text_content
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

from negotiation_rl.parsing import Action, ActionParser, parse_reply, visible_message
from negotiation_rl.prompts import build_buyer_messages
from negotiation_rl.reward import DEAL_OUTCOMES, OUTCOMES, RewardResult, compute_reward
from negotiation_rl.seller import Seller
from negotiation_rl.spec import NegotiationSpec

SELLER_MAX_ATTEMPTS = 6

_PARSER = ActionParser()
logger = logging.getLogger(__name__)


class NegotiationEnv(MessageEnv):
    """One negotiation episode, seen from the buyer, with the seller inside the environment."""

    def __init__(self, spec: NegotiationSpec, seller_completer: TinkerMessageCompleter) -> None:
        self._spec = spec
        self._seller_completer = seller_completer
        # Episode state, reset in initial_observation().
        self._messages: list[Message] = []  # the buyer's conversation
        self._seller: Seller | None = None
        self._buyer_offers: list[Action] = []  # every buyer [BUY]
        self._seller_offers: list[Action] = []  # every seller [SELL], and its accepted [DEAL]
        self._turn = 0  # buyer turns so far
        self._turn_log: list[dict] = []  # one entry per action, for the episode summary

    async def initial_observation(self) -> list[Message]:
        """Start a new episode and return the buyer's opening messages."""
        spec = self._spec
        self._messages = build_buyer_messages(spec.inventory.buyer_list(), spec.shopping_list, spec.max_turns)
        self._seller = Seller(self._seller_completer, spec.inventory.seller_list(), spec.max_turns)
        self._buyer_offers, self._seller_offers, self._turn, self._turn_log = [], [], 0, []
        return list(self._messages)

    async def step(self, message: Message) -> MessageStepResult:
        """Apply one buyer reply: check it, let the seller answer, and end the episode when due."""
        spec = self._spec
        self._turn += 1
        reply = get_text_content(message)
        self._messages.append({"role": "assistant", "content": reply})

        # Read the buyer's action; a missing, unreadable or seller-only action ends the episode.
        action_text = parse_reply(reply)[2] if reply else ""
        if not action_text:
            return self._end("buyer_format_error__no_action")
        try:
            action = _PARSER(action_text)
        except RuntimeError:
            return self._end("buyer_format_error__outside_parse_failed")
        self._log("buyer", action)
        if action.is_sell():
            return self._end("buyer_format_error__sell_action")

        # Offering or accepting a price above the budget (as stated to the buyer, in cents) breaks
        # the buyer's own constraint.
        if (action.is_buy() or action.is_deal()) and action.money > spec.stated_budget:
            return self._end("buyer_price_error", price_overshoot=action.money - spec.stated_budget)

        # [DEAL] must copy the seller's latest offer; [QUIT] walks away.
        if action.is_deal():
            return self._buyer_deal(action)
        if action.is_quit():
            return self._end("buyer_quit")

        # [BUY] or [REJECT]: the seller answers.
        if action.is_buy():
            self._buyer_offers.append(action)
        seller_reply, outcome, deal_price, attempts = await self._seller_turn(visible_message(reply))
        if seller_reply is not None:
            self._messages.append({"role": "user", "content": visible_message(seller_reply)})
        if outcome is not None:
            return self._end(outcome, deal_price=deal_price, seller_attempts=attempts)

        # The turn limit is checked after the seller, so the last buyer offer still gets an answer.
        if self._turn >= spec.max_turns:
            return self._end("deadlock", seller_attempts=attempts)
        return MessageStepResult(reward=0.0, episode_done=False, next_messages=list(self._messages))

    def _buyer_deal(self, action: Action) -> MessageStepResult:
        """End the episode on a buyer [DEAL], which must match the seller's latest offer."""
        if not self._seller_offers:
            return self._end("buyer_deal_error__no_seller_offer")
        latest = self._seller_offers[-1]
        if ActionParser.same_offer(latest, action):
            return self._end("buyer_deal", deal_price=action.money)
        if ActionParser.same_items(latest, action):
            return self._end("buyer_deal_error__offer_price_mismatch")
        return self._end("buyer_deal_error__offer_item_mismatch")

    async def _seller_turn(self, buyer_message: str) -> tuple[str | None, str | None, float | None, int]:
        """Get a valid seller answer; return (reply, ending outcome or None, deal price, attempts).

        An invalid reply (an error outcome) is sampled again, up to ``SELLER_MAX_ATTEMPTS`` in all;
        when every attempt is invalid, the last error ends the episode.
        """
        assert self._seller is not None
        for attempt in range(1, SELLER_MAX_ATTEMPTS + 1):
            # Sample; a failed request leaves the seller's history unchanged and counts as an
            # invalid answer.
            try:
                reply = await self._seller.respond(buyer_message)
            except Exception as error:
                logger.warning("Seller request failed (%s, attempt %d): %s", self._spec.codename, attempt, error)
                reply = None
            outcome, deal_price, action = self._judge_seller_reply(reply)

            # A valid answer: the episode continues, or ends with a seller [DEAL] or [QUIT].
            if outcome is None or outcome in ("seller_deal", "seller_quit"):
                self._log("seller", action)
                return reply, outcome, deal_price, attempt

            # An invalid answer: forget it and ask again, unless this was the last attempt.
            if attempt < SELLER_MAX_ATTEMPTS and reply is not None:
                self._seller.undo_last_exchange()
        self._log("seller", action)
        return reply, outcome, None, SELLER_MAX_ATTEMPTS

    def _judge_seller_reply(self, reply: str | None) -> tuple[str | None, float | None, Action | None]:
        """Classify a seller reply; return (outcome or None to continue, deal price, action)."""
        # The request failed, or the reply has no readable action.
        if reply is None:
            return "seller_generation_error", None, None
        action_text = parse_reply(reply)[2] if reply else ""
        if not action_text:
            return "seller_format_error__no_action", None, None
        try:
            action = _PARSER(action_text)
        except RuntimeError:
            return "seller_format_error__outside_parse_failed", None, None
        if action.is_buy():
            return "seller_format_error__buy_action", None, action

        # The seller may never offer or accept below its cost.
        if (action.is_sell() or action.is_deal()) and action.money < self._spec.cost:
            return "seller_price_error", None, action

        # [QUIT] ends the episode; [REJECT] and [SELL] continue it.
        if action.is_quit():
            return "seller_quit", None, action
        if action.is_reject():
            return None, None, action
        if action.is_sell():
            self._seller_offers.append(action)
            return None, None, action

        # [DEAL] must copy the buyer's latest [BUY].
        if not self._buyer_offers:
            return "seller_deal_error__no_buyer_offer", None, action
        latest = self._buyer_offers[-1]
        if ActionParser.same_offer(latest, action):
            self._seller_offers.append(action)
            return "seller_deal", action.money, action
        if ActionParser.same_items(latest, action):
            return "seller_deal_error__offer_price_mismatch", None, action
        return "seller_deal_error__offer_item_mismatch", None, action

    def _log(self, role: str, action: Action | None) -> None:
        """Record one action in the turn log, e.g. {"turn": 1, "role": "buyer", "action": "BUY", "offer": 15.0}."""
        self._turn_log.append({
            "turn": self._turn,
            "role": role,
            "action": action.action_type if action is not None else None,
            "offer": action.money if action is not None else None,
        })

    def _end(
        self,
        outcome: str,
        *,
        deal_price: float | None = None,
        price_overshoot: float | None = None,
        seller_attempts: int | None = None,
    ) -> MessageStepResult:
        """Finish the episode: compute the reward, its metrics and summary, and log both conversations."""
        spec = self._spec
        result: RewardResult = compute_reward(spec.budget, spec.cost, deal_price, outcome)
        first_offer = next((e["offer"] for e in self._turn_log if e["role"] == "buyer" and e["action"] == "BUY"), None)

        # Metrics are averaged over the episodes that report them, so a metric that does not apply
        # to this episode (e.g. the deal-only ratio without a deal) is left out.
        is_deal = outcome in DEAL_OUTCOMES
        metrics: dict[str, float | None] = {f"outcome_{name}": float(name == outcome) for name in OUTCOMES}
        metrics.update({
            "deal": float(is_deal),
            "is_conflict_of_interest": float(spec.is_conflict_of_interest),
            # Buyer's share of the bargaining range, (B - P) / (B - C), on deals with B > C.
            "buyer_bargained_ratio_mutual_interest_deals": (
                result.buyer_bargained_ratio if is_deal and spec.budget > spec.cost else None
            ),
            # The buyer's first [BUY] price divided by its budget.
            "first_offer_ratio": first_offer / spec.budget if first_offer is not None else None,
            # Dollars above the budget, on buyer price errors.
            "price_overshoot": price_overshoot,
            # Seller samples needed for its last answer (1 = the first was valid).
            "seller_attempts": float(seller_attempts) if seller_attempts is not None else None,
        })
        metrics = {name: value for name, value in metrics.items() if value is not None}

        # One summary per episode, written to the rollout summary files.
        summary = {
            "codename": spec.codename,
            "category": spec.category,
            "product_index": spec.product_index,
            "list_price": spec.list_price,
            "budget": spec.budget,
            "cost": spec.cost,
            "is_conflict_of_interest": spec.is_conflict_of_interest,
            "max_turns": spec.max_turns,
            "buyer_turns": self._turn,
            "outcome": outcome,
            "deal_price": deal_price,
            "reward": result.reward,
            "buyer_bargained_ratio": result.buyer_bargained_ratio,
            "first_offer": first_offer,
            "turn_log": self._turn_log,
        }

        # The full conversations, as each side saw them, go to the HTML log.
        with logtree.scope_header(f"{spec.codename}: {outcome}, reward {result.reward:.3f}"):
            with logtree.scope_header("Buyer's view"):
                logtree.log_formatter(ConversationFormatter(messages=self._messages))
            with logtree.scope_header("Seller's view"):
                logtree.log_formatter(ConversationFormatter(messages=self._seller.messages))

        return MessageStepResult(
            reward=result.reward,
            episode_done=True,
            next_messages=list(self._messages),
            metrics=metrics,
            logs={"episode_summary": json.dumps(summary)},
        )


class _TokenEnv(EnvFromMessageEnv):
    """Token-level wrapper of a NegotiationEnv; a buyer reply cut off at max_tokens counts as a format error."""

    async def step(self, action: types.Action, *, extra: types.ActionExtra | None = None) -> types.StepResult:
        """End the episode with the format-error reward when the reply hit the token limit."""
        if (extra or {}).get("stop_reason") == "length":
            return types.StepResult(
                reward=self.failed_parse_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._base_stop_condition,
                metrics={"max_tokens_reached": 1.0},
            )
        return await super().step(action, extra=extra)


class NegotiationGroupBuilder(EnvGroupBuilder):
    """Builds ``group_size`` parallel episodes on one product; their rewards are compared within the group."""

    def __init__(
        self,
        spec: NegotiationSpec,
        group_size: int,
        renderer_name: str,
        buyer_model: str,
        max_trajectory_tokens: int,
        seller_completer: TinkerMessageCompleter,
    ) -> None:
        self._spec = spec
        self._group_size = group_size
        self._renderer_name = renderer_name
        self._buyer_model = buyer_model
        self._max_trajectory_tokens = max_trajectory_tokens
        self._seller_completer = seller_completer

    async def make_envs(self) -> Sequence[types.Env]:
        """Return the group's token-level environments, each with its own seller."""
        # A buyer reply the chat template cannot parse ends the episode with -1; a conversation
        # longer than max_trajectory_tokens ends it with 0.
        renderer = get_renderer(self._renderer_name, get_tokenizer(self._buyer_model))
        return [
            _TokenEnv(
                renderer=renderer,
                message_env=NegotiationEnv(self._spec, self._seller_completer),
                failed_parse_reward=-1.0,
                terminate_on_parse_error=True,
                max_trajectory_tokens=self._max_trajectory_tokens,
                context_overflow_reward=0.0,
            )
            for _ in range(self._group_size)
        ]

    def logging_tags(self) -> list[str]:
        """Tag the group with its product category, so metrics are also reported per category."""
        return [self._spec.category]
