# Instructing LLMs to Negotiate using Reinforcement Learning with Verifiable Rewards

This repository trains an LLM **buyer** to bargain over the price of a product with an LLM
**seller**, using reinforcement learning with a reward computed from the deal itself. The buyer
(`Qwen3-30B-A3B-Instruct-2507`) is trained with group-relative policy updates through the
[Tinker](https://thinkingmachines.ai/tinker/) API and
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). The seller is the same
base model, not trained, and is held to its private cost. Products come from the
AmazonHistoryPrice dataset (930 real listings in 18 categories; 802 for training, 128 for testing).

## Setup

Python 3.11 or newer.

```bash
git clone https://github.com/ShuzeLiu/Instructing-LLMs-to-Negotiate.git
cd Instructing-LLMs-to-Negotiate
pip install -e .
cp .env.example .env    # then put your Tinker API key in .env
```

Install in editable mode (`-e`): the dataset in `data/` is read from the repository. The key is
read from the `TINKER_API_KEY` environment variable, or from a `.env` file in the directory you
run from.

## Training

```bash
python -m negotiation_rl.train
```

This runs the paper's setting: 60 iterations of 64 products × 8 buyer rollouts (512 negotiations
per iteration). Every 2 iterations the buyer is also evaluated on all 128 test products, 8 rollouts
each (1,024 negotiations), with the training limit of 300 tokens per reply; this evaluation gives
the paper's training curves. The paper's benchmark tables come from a separate evaluation of the
trained buyer, with 4 rollouts per test product and up to 4,000 tokens per reply, which is not
part of this repository. Every field of `TrainingConfig` in `negotiation_rl/train.py` can be set
on the command line as `key=value`; for example, a three-iteration test without evaluation:

```bash
python -m negotiation_rl.train batch_size=8 group_size=4 max_steps=3 eval_every=0
```

Outputs go to `outputs/<run_name>/` (`run_name` defaults to the start time). Running the same
command again with `run_name=<that name>` resumes from the run's last checkpoint.

| Setting (`command-line field`) | Default (the paper's training run) |
|---|---|
| **Models** | |
| buyer, trained (`buyer_model`) | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| seller, not trained (`seller_model`) | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| buyer chat template (`renderer_name`) | `qwen3` |
| seller chat template | the one tinker-cookbook recommends for the seller model |
| buyer sampling temperature in training (`buyer_temperature`) | 1.0 (test evaluation: 1.0) |
| seller sampling temperature (`seller_temperature`) | 0.7 |
| **Negotiation** | |
| buyer turns per negotiation (`max_turns`) | 6 |
| tokens per reply, buyer and seller (`max_tokens`) | 300 |
| buyer budget as a fraction of the list price (`budget_ratio`) | 0.8 |
| tokens per whole conversation (`max_trajectory_tokens`) | 4096 |
| seller samples per turn before a seller error | 6 |
| **Training** | |
| products per iteration (`batch_size`) | 64 |
| buyer rollouts per product (`group_size`) | 8 |
| passes over the 802 training products (`num_epochs`) | 5 (12 iterations each, 60 in all) |
| order of the products in each pass (`seed`) | 0 |
| stop early after this many iterations (`max_steps`) | none (all 60) |
| learning rate (`learning_rate`) | 3e-5 |
| LoRA rank (`lora_rank`) | 32 |
| KL penalty | 0 |
| loss | importance-sampled policy gradient (tinker-cookbook `importance_sampling`) |
| advantage | reward minus the mean reward of the product's group |
| **Evaluation and output** | |
| test-set evaluation every (`eval_every`; 0 turns it off) | 2 iterations |
| rollouts per test product, tokens per reply in that evaluation | 8 and 300 (the training `group_size` and `max_tokens`) |
| checkpoint every (`save_every`; 0 turns it off) | 10 iterations |
| output folder (`log_dir`, `run_name`) | `outputs/<start time>` |
| Weights & Biases project (`wandb_project`) | none (off) |

## The negotiation

Each reply of either side has three parts, each starting on its own line:

```
Thought: The list price is $70 and my budget is $56. I will open low to test the seller.
Talk: Can you do it for $15? That's all I can afford.
Action: [BUY] $15 (1x beauty_29)
```

The other side sees only the `Talk` and `Action` lines; the `Thought` line stays private. The
actions are:

| Action | Who | Meaning |
|---|---|---|
| `[BUY] $M (N codename)` | buyer | offer M dollars for N items |
| `[SELL] $M (N codename)` | seller | offer N items for M dollars |
| `[DEAL] $M (N codename)` | both | accept the other side's latest offer, copied exactly; ends the negotiation |
| `[REJECT]` | both | decline and wait for a new offer |
| `[QUIT]` | both | walk away; ends the negotiation |

The buyer sees the product's title, description and list price, and its own budget $B$ (stated
to the cent). The seller sees the same listing and its cost $C$ (the lowest price the product has
sold for). The seller is held to its cost and to the rules: a seller reply that offers or accepts
below $C$, breaks the format, uses `[BUY]`, or uses `[DEAL]` on anything but the buyer's latest
offer is sampled again, as is a failed request, up to 6 samples in all.

A negotiation ends on a deal, on a `[QUIT]`, on an error, or after 6 buyer turns. A deal at price
$P$ earns the buyer

$$R = \dfrac{B - P}{|B - C|}$$

clipped to $[0, 1]$ when $B \gt C$: 1 when the buyer pays the seller's cost, 0 when it pays its
whole budget. When $B \lt C$ (a conflict of interest) no deal can benefit both sides, and any deal
earns $-1$. A negotiation without a deal earns 0. A buyer error earns $-1$: a reply with no
readable action, a `[SELL]`, a `[BUY]` or `[DEAL]` above the budget, a `[DEAL]` that does not
copy the seller's latest offer, or a reply cut off at the token limit. A seller error, or a
conversation longer than 4096 tokens, ends the negotiation with 0.

## Outputs and metrics

`outputs/<run_name>/` holds `metrics.jsonl` (one line per iteration), `checkpoints.jsonl` (the
Tinker checkpoint paths), and one folder per iteration, `iteration_NNNNNN/`, with the rollout
summaries (JSONL; each finished negotiation carries an `episode_summary` with its product,
budget, cost, outcome, deal price, reward and every action) and HTML logs of sampled
conversations from both sides.

Test-set metrics are logged under `test/env/all/` and training metrics under `env/all/`; the same
metrics per product category are under `env/<category>/` (and `test/env/<category>/`). The paper's metrics are:

| Paper metric | Logged as | Averaged over |
|---|---|---|
| reward | `reward/total` | all negotiations |
| buyer bargained ratio | `buyer_bargained_ratio_mutual_interest_deals` | deals where the budget is above the cost |
| first-turn offer ratio | `first_offer_ratio` | negotiations with a `[BUY]` |
| deal rate | `deal` | negotiations ended by the environment |
| price overshoot rate | `outcome_buyer_price_error` | negotiations ended by the environment |

The buyer bargained ratio is $\dfrac{B - P}{B - C}$, the buyer's share of the bargaining range on a
deal. The first-turn offer ratio is the buyer's first `[BUY]` price divided by its budget $B$
(above 1 when that offer exceeds the budget).
Every outcome has its own `outcome_<name>` fraction (for example `outcome_seller_deal`,
`outcome_buyer_quit`, `outcome_deadlock`); the names are listed in `negotiation_rl/reward.py`.
A negotiation that the training loop ends itself (a reply the chat template cannot parse, a reply
cut off at the token limit, or a conversation longer than 4096 tokens) reports only its reward.

## Code

| File | Contents |
|---|---|
| `negotiation_rl/train.py` | settings and entry point |
| `negotiation_rl/dataset.py` | training and test batches |
| `negotiation_rl/env.py` | the negotiation environment |
| `negotiation_rl/spec.py` | one negotiation's product, budget and cost |
| `negotiation_rl/reward.py` | the reward |
| `negotiation_rl/seller.py` | the seller, sampled through Tinker |
| `negotiation_rl/prompts.py` | buyer and seller prompts |
| `negotiation_rl/parsing.py` | reading replies and actions |
| `negotiation_rl/products.py` | loading the dataset |
| `scripts/make_split.py` | recreates the train/test split in `data/splits/` |

## Data

`data/AmazonHistoryPrice/` is the AmazonHistoryPrice dataset of Xia et al. (2024), unchanged. Each
product's list price is the larger of its listed and highest price, and its cost is its lowest
price. `data/splits/` holds the fixed split (product indices in loading order; seed 42).

## Citation

```bibtex
@misc{liu2026instructing,
  title  = {Instructing {LLMs} to Negotiate using Reinforcement Learning with Verifiable Rewards},
  author = {Liu, Shuze Daniel and Chen, Claire and Xiao, Jiabao Sean and Lei, Lei and
            Zhang, Yuheng and Yue, Yisong and Simchi-Levi, David},
  year   = {2026}
}
```

The dataset:

```bibtex
@inproceedings{xia2024measuring,
  title     = {Measuring Bargaining Abilities of {LLMs}: A Benchmark and A Buyer-Enhancement Method},
  author    = {Xia, Tian and He, Zhiwei and Ren, Tong and Miao, Yibo and Zhang, Zhuosheng and
               Yang, Yang and Wang, Rui},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages     = {3579--3602},
  year      = {2024}
}
```

## License

Apache License 2.0 (see `LICENSE`). The dataset and parts of `prompts.py`, `parsing.py` and
`products.py` come from the [AmazonHistoryPrice benchmark](https://github.com/TianXiaSJTU/AmazonPriceHistory),
also under the Apache License 2.0; see `NOTICE`.
