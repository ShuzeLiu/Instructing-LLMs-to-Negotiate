# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# Derived from the AmazonHistoryPrice bargaining benchmark
# (https://github.com/TianXiaSJTU/AmazonPriceHistory, Apache License 2.0), files session.py
# (parseReply) and utils/Action.py (Action, ActionParser).
"""Parsing of agent replies and of the actions inside them.

A reply has three parts, each starting on its own line::

    Thought: The list price is $70 and my budget is $56. I will open low.
    Talk: Can you do it for $15?
    Action: [BUY] $15 (1x beauty_29)

The action is one of::

    [BUY] $M (N codename)     buyer offers $M for N items
    [SELL] $M (N codename)    seller offers N items for $M
    [DEAL] $M (N codename)    accept the other side's latest offer, copied exactly
    [REJECT]                  decline and wait for a new offer
    [QUIT]                    walk away
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Field headers count only at the start of a line, so "... their Action: they ..." inside a
# sentence is not mistaken for the Action field.
_THOUGHT_HEADER = re.compile(r"(?:^|\n)Thought:\s*", re.MULTILINE)
_TALK_HEADER = re.compile(r"(?:^|\n)Talk:\s*", re.MULTILINE)
_ACTION_HEADER = re.compile(r"(?:^|\n)Action:\s*", re.MULTILINE)

ACTION_TYPES_WITH_MONEY = ("BUY", "SELL", "DEAL")
ACTION_TYPES_WITHOUT_MONEY = ("REJECT", "QUIT")


def parse_reply(reply: str) -> tuple[str, str, str, str]:
    """Split a reply into (thought, talk, action, message); message is what the other side sees.

    The last ``Talk:`` / ``Action:`` pair is used, so drafts written earlier in the same reply
    are ignored. The action is the rest of the line after the last ``Action:``. The thought runs
    from the first ``Thought:`` to the start of that last ``Talk:``.

    before: "Thought: my budget is $56\\nTalk: Can you do $15?\\nAction: [BUY] $15 (1x beauty_29)"
    after:  ("my budget is $56", "Can you do $15?", "[BUY] $15 (1x beauty_29)",
             "Talk: Can you do $15?\\nAction: [BUY] $15 (1x beauty_29)")
    """
    if not reply:
        raise ValueError("parse_reply got an empty reply")

    # Locate every header; the last Talk/Action pair is the reply's final answer.
    thought_matches = list(_THOUGHT_HEADER.finditer(reply))
    talk_matches = list(_TALK_HEADER.finditer(reply))
    action_matches = list(_ACTION_HEADER.finditer(reply))

    # Action: the rest of the line after the last "Action:".
    if action_matches:
        action = reply[action_matches[-1].end():].split("\n", 1)[0].strip()
    else:
        action = ""

    # Talk: the text between the last "Talk:" and the last "Action:".
    if talk_matches and action_matches and talk_matches[-1].start() < action_matches[-1].start():
        talk = reply[talk_matches[-1].end():action_matches[-1].start()].strip()
    else:
        talk = ""

    # Thought: from the first "Thought:" up to the final Talk (or Action, or the end).
    if thought_matches:
        start = thought_matches[0].end()
        if talk_matches:
            thought = reply[start:talk_matches[-1].start()].strip()
        elif action_matches:
            thought = reply[start:action_matches[-1].start()].strip()
        else:
            thought = reply[start:].strip()
    else:
        thought = ""

    message = f"Talk: {talk}\nAction: {action}"
    return thought, talk, action, message


def visible_message(reply: str) -> str:
    """Return the part of a reply the other side sees: its Talk and Action, without the Thought.

    before: "Thought: my cost is $23.24\\nTalk: I can't go below $25.\\nAction: [REJECT]"
    after:  "Talk: I can't go below $25.\\nAction: [REJECT]"
    """
    # An empty or blank reply has no fields to forward.
    if not reply.strip():
        return "Talk: \nAction: "
    return parse_reply(reply)[3]


@dataclass
class Action:
    """One parsed action: its type and, for BUY/SELL/DEAL, the total price and the items."""

    action_type: str
    money: float | None = None
    objects: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Check that the fields fit the action type, and round the price to cents."""
        # BUY / SELL / DEAL need a non-negative price and at least one positive item count.
        if self.action_type in ACTION_TYPES_WITH_MONEY:
            if self.money is None:
                raise ValueError("Action: field money is None")
            if self.money < 0:
                raise ValueError("Action: field money is negative")
            self.money = round(self.money, 2)
            if not self.objects:
                raise ValueError("Action: field objects is None or empty")
            for name, count in self.objects.items():
                if count <= 0:
                    raise ValueError(f"Action: item count is not positive: {count} {name}")
        # REJECT / QUIT carry no price and no items.
        elif self.action_type in ACTION_TYPES_WITHOUT_MONEY:
            if self.money is not None or self.objects is not None:
                raise ValueError(f"Action: [{self.action_type}] takes no price or items")
        else:
            raise ValueError(f"Action: unknown type {self.action_type}")

    def is_buy(self) -> bool:
        """Return True for [BUY]."""
        return self.action_type == "BUY"

    def is_sell(self) -> bool:
        """Return True for [SELL]."""
        return self.action_type == "SELL"

    def is_deal(self) -> bool:
        """Return True for [DEAL]."""
        return self.action_type == "DEAL"

    def is_reject(self) -> bool:
        """Return True for [REJECT]."""
        return self.action_type == "REJECT"

    def is_quit(self) -> bool:
        """Return True for [QUIT]."""
        return self.action_type == "QUIT"

    def has_money_and_objects(self) -> bool:
        """Return True when the action carries both a price and items."""
        return self.money is not None and self.objects is not None


class ActionParser:
    """Parse the action string of a reply; when it holds several actions, the last one is used.

    before: "[BUY] $1,250.50 (1x electronics_42)"
    after:  Action(action_type="BUY", money=1250.5, objects={"electronics_42": 1})
    """

    _PATTERN = r"\[([A-Z]+)\](?: \$([\d\,]+(?:\.[\d\,]+)?) \(((?:\d+x? .+, )*\d+x? .+?)\))?"
    _ITEM_PATTERN = r"(\d+)x? (.+)"

    def __call__(self, text: str) -> Action:
        """Return the last action in ``text``; raise RuntimeError when none can be read."""
        try:
            matches = re.findall(self._PATTERN, text)
            if not matches:
                raise RuntimeError("no action in text")
            return Action(*self._convert(matches[-1]))
        except Exception as error:
            raise RuntimeError(f"ActionParser: {error}: {text}") from error

    def _convert(self, fields: tuple[str, str, str]) -> tuple:
        """Turn the regex groups of one action into Action arguments.

        before: ("BUY", "1,250.50", "1x electronics_42")
        after:  ("BUY", 1250.5, {"electronics_42": 1})
        """
        action_type, money_text, objects_text = fields
        if action_type not in ACTION_TYPES_WITH_MONEY + ACTION_TYPES_WITHOUT_MONEY:
            raise ValueError(f"[{action_type}] is not an action type")

        # An action written without a price and items, e.g. "[REJECT]".
        if money_text == "" or objects_text == "":
            return (action_type,)

        # Price: drop thousands separators.
        money = float(money_text.replace(",", ""))

        # Items: "2x apple_1, 1x pear_3" -> {"apple_1": 2, "pear_3": 1}.
        objects: dict[str, int] = {}
        for item in objects_text.split(", "):
            match = re.search(self._ITEM_PATTERN, item)
            count, name = match.group(1), match.group(2)
            objects[name] = objects.get(name, 0) + int(count)
        return (action_type, money, objects)

    @staticmethod
    def same_items(first: Action, second: Action) -> bool:
        """Return True when both actions name the same items in the same counts."""
        if first.has_money_and_objects() and second.has_money_and_objects():
            return first.objects == second.objects
        return False

    @staticmethod
    def same_offer(first: Action, second: Action) -> bool:
        """Return True when both actions name the same items at the same price (within one cent)."""
        if not ActionParser.same_items(first, second):
            return False
        return abs(first.money - second.money) <= 0.01
