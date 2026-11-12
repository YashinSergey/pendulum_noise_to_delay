import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC

from src import config
from src.envs import make_env


def evaluate(model, seeds, delay=0):
    rows = []
    env = make_env(delay=delay)
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            rng = np.random.default_rng([config.RANDOM_POLICY_SEED, seed])
            total_reward = 0.0
            length = 0
            while True:
                if model is None:
                    action = rng.uniform(env.action_space.low, env.action_space.high)
                    action = action.astype(env.action_space.dtype)
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    break
            rows.append({"episode_seed": seed, "delay": delay, "reward": total_reward,
                         "length": length, "terminated": terminated, "truncated": truncated})
    finally:
        env.close()
    return pd.DataFrame(rows)


def summarize(episodes):
    if episodes.duplicated(["condition", "train_seed", "delay", "episode_seed"]).any():
        raise ValueError("В таблице повторяются эпизоды одного запуска")
    runs = episodes.groupby(["condition", "train_seed", "delay"], as_index=False).agg(
        reward=("reward", "mean"), episodes=("reward", "size")
    )
    baseline = runs[runs.delay == 0][["condition", "train_seed", "reward"]]
    if len(baseline) != len(runs[["condition", "train_seed"]].drop_duplicates()):
        raise ValueError("Для каждого запуска нужна оценка без задержки")
    runs = runs.merge(baseline.rename(columns={"reward": "baseline_reward"}),
                      on=["condition", "train_seed"], validate="many_to_one")
    runs["loss"] = runs.baseline_reward - runs.reward
    summary = runs.groupby(["condition", "delay"], as_index=False).agg(
        reward_mean=("reward", "mean"), reward_std=("reward", "std"),
        loss_mean=("loss", "mean"), loss_std=("loss", "std"), seeds=("train_seed", "size")
    )
    effects = runs[runs.delay == config.PRIMARY_DELAY].pivot(
        index="train_seed", columns="condition", values="loss"
    )
    if {"clean", "noisy"}.issubset(effects.columns):
        effects["effect"] = effects.clean - effects.noisy
    return runs, summary, effects.reset_index()


def main():
    torch.set_num_threads(config.TORCH_THREADS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    args = parser.parse_args()
    seeds = config.TEST_SEEDS if args.split == "test" else config.PILOT_EVAL_SEEDS
    args.output.mkdir(parents=True, exist_ok=False)
    tables = []
    for run in args.runs:
        metadata = json.loads((run / "metadata.json").read_text())
        model = SAC.load(run / "model.zip", device=config.DEVICE)
        for delay in config.DELAYS:
            table = evaluate(model, seeds, delay)
            table["condition"] = metadata["condition"]
            table["train_seed"] = metadata["seed"]
            tables.append(table)
    episodes = pd.concat(tables, ignore_index=True)
    episodes.to_csv(args.output / "episodes.csv", index=False)
    runs, summary, effects = summarize(episodes)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    effects.to_csv(args.output / "effects.csv", index=False)
    evaluate(None, seeds).to_csv(args.output / "random.csv", index=False)
    (args.output / "evaluation.json").write_text(json.dumps({
        "split": args.split, "episode_seeds": seeds,
        "runs": [str(run.resolve()) for run in args.runs],
        "delays": config.DELAYS, "deterministic": True,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
