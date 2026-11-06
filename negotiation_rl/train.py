# Copyright 2026 Shuze Daniel Liu
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Train the buyer with reinforcement learning against the fixed seller, through Tinker.

Usage (defaults are the paper's settings; any field of TrainingConfig can be set as key=value):

    python -m negotiation_rl.train
    python -m negotiation_rl.train batch_size=8 group_size=4 max_steps=3 eval_every=0

Every ``eval_every`` iterations the buyer plays all 128 test products. Running the same command
again with the same ``run_name`` resumes from the run's last checkpoint.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

import chz
from dotenv import find_dotenv, load_dotenv
from tinker_cookbook.rl import train

from negotiation_rl.dataset import NegotiationDatasetBuilder


@chz.chz
class TrainingConfig:
    """The settings of one training run."""

    # Models: the trained buyer, and the seller it negotiates with (not trained).
    buyer_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    seller_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str = "qwen3"  # chat template of the buyer
    buyer_temperature: float = 1.0  # training rollouts; the test evaluation samples at 1.0
    seller_temperature: float = 0.7

    # Episodes: buyer turns per negotiation, tokens per reply (buyer and seller), the budget as
    # a fraction of the list price, and the token limit of a whole conversation.
    max_turns: int = 6
    max_tokens: int = 300
    budget_ratio: float = 0.8
    max_trajectory_tokens: int = 4096

    # Batches: products per iteration, rollouts per product, and passes over the training set.
    # 802 training products give 12 iterations per epoch, so 5 epochs are 60 iterations.
    batch_size: int = 64
    group_size: int = 8
    num_epochs: int = 5
    seed: int = 0  # orders the products in each epoch
    max_steps: int | None = None  # stop after this many iterations (None: all epochs)

    # Optimization (no KL penalty).
    learning_rate: float = 3e-5
    lora_rank: int = 32

    # Test-set evaluation (all 128 test products, group_size rollouts each) and checkpoints,
    # every this many iterations; 0 turns either off.
    eval_every: int = 2
    save_every: int = 10

    # Output: <log_dir>/<run_name>; the run name defaults to the start time.
    log_dir: str = "outputs"
    run_name: str | None = None
    wandb_project: str | None = None


def build_config(cfg: TrainingConfig) -> train.Config:
    """Turn the run settings into the training-loop configuration of tinker_cookbook."""
    run_name = cfg.run_name or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(cfg.log_dir, run_name)
    dataset_builder = NegotiationDatasetBuilder(
        buyer_model=cfg.buyer_model,
        seller_model=cfg.seller_model,
        renderer_name=cfg.renderer_name,
        max_turns=cfg.max_turns,
        max_tokens=cfg.max_tokens,
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
        budget_ratio=cfg.budget_ratio,
        seller_temperature=cfg.seller_temperature,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
    )
    # The dataset already holds every epoch, so the training loop makes a single pass over it.
    return train.Config(
        model_name=cfg.buyer_model,
        recipe_name="instructing_llms_to_negotiate",
        dataset_builder=dataset_builder,
        renderer_name=cfg.renderer_name,
        learning_rate=cfg.learning_rate,
        lora_rank=cfg.lora_rank,
        loss_fn="importance_sampling",
        kl_penalty_coef=0.0,
        max_tokens=cfg.max_tokens,
        temperature=cfg.buyer_temperature,
        max_steps=cfg.max_steps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        log_path=log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=run_name if cfg.wandb_project else None,
    )


def main() -> None:
    """Read the settings from the command line and train."""
    # TINKER_API_KEY may be set in the environment or in a .env file in the working directory.
    load_dotenv(find_dotenv(usecwd=True))
    cfg = chz.entrypoint(TrainingConfig)
    asyncio.run(train.main(build_config(cfg)))


if __name__ == "__main__":
    main()
