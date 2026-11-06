# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The seller: a fixed (not trained) LLM sampled through Tinker, holding its own conversation."""

from __future__ import annotations

import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from negotiation_rl.prompts import build_seller_messages


def create_seller_completer(
    service_client: tinker.ServiceClient, seller_model: str, max_tokens: int, temperature: float
) -> TinkerMessageCompleter:
    """Build the sampler every seller shares (one Tinker sampling client per process)."""
    # The seller uses the chat template recommended for its model.
    renderer = get_renderer(get_recommended_renderer_name(seller_model), get_tokenizer(seller_model))
    return TinkerMessageCompleter(
        sampling_client=service_client.create_sampling_client(base_model=seller_model),
        renderer=renderer,
        max_tokens=max_tokens,
        temperature=temperature,
    )


class Seller:
    """One seller for one episode; it sees its cost and keeps the whole conversation."""

    def __init__(self, completer: TinkerMessageCompleter, seller_inventory: str, max_turns: int) -> None:
        self._completer = completer
        self.messages: list[Message] = build_seller_messages(seller_inventory, max_turns)

    async def respond(self, buyer_message: str) -> str:
        """Add the buyer's message, sample the seller's reply, add it, and return its text."""
        self.messages.append({"role": "user", "content": buyer_message})
        try:
            reply = await self._completer(self.messages)
        except Exception:
            # Keep the history consistent so the same buyer message can be sent again.
            self.messages.pop()
            raise
        text = get_text_content(reply)
        self.messages.append({"role": "assistant", "content": text})
        return text

    def undo_last_exchange(self) -> None:
        """Remove the last buyer message and seller reply, before asking the seller again."""
        assert self.messages[-1]["role"] == "assistant" and self.messages[-2]["role"] == "user"
        del self.messages[-2:]
