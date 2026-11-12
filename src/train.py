import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src import config
from src.envs import make_env
from src.evaluate import evaluate


class EvaluationCallback(BaseCallback):
    def __init__(self, frequency, seeds, output):
        super().__init__()
        self.frequency = frequency
        self.seeds = seeds
        self.output = output
        self.tables = []
        self.last_step = None

    def record(self):
        table = evaluate(self.model, self.seeds)
        table["timesteps"] = self.num_timesteps
        self.tables.append(table)
        pd.concat(self.tables, ignore_index=True).to_csv(self.output, index=False)
        self.last_step = self.num_timesteps

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        if self.num_timesteps % self.frequency == 0:
            self.record()
        return True

    def _on_training_end(self):
        if self.last_step != self.num_timesteps:
            self.record()


def train(condition, seed, steps, output, eval_frequency, eval_seeds):
    output = Path(output)
    if steps <= 0 or eval_frequency <= 0:
        raise ValueError("Число шагов и частота оценки должны быть положительными")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.TORCH_THREADS)
    env = Monitor(make_env(condition), str(output / "train.monitor.csv"))
    settings = dict(config.SAC_SETTINGS)
    started = time.perf_counter()
    try:
        model = SAC("MlpPolicy", env, seed=seed, device=config.DEVICE,
                    policy_kwargs={"net_arch": list(config.NET_ARCH)}, **settings)
        callback = EvaluationCallback(eval_frequency, eval_seeds, output / "evaluation.csv")
        model.learn(total_timesteps=steps, callback=callback)
        elapsed = time.perf_counter() - started
        model.save(output / "model.zip")
        metadata = {
            "condition": condition, "seed": seed, "timesteps": model.num_timesteps,
            "elapsed_seconds": elapsed, "device": config.DEVICE,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(), "net_arch": config.NET_ARCH,
            "sac_settings": settings, "eval_frequency": eval_frequency,
            "eval_seeds": eval_seeds, "environment": config.ENV_ID,
            "angle_noise_std": config.ANGLE_NOISE_STD if condition == "noisy" else 0,
            "velocity_noise_std": config.VELOCITY_NOISE_STD if condition == "noisy" else 0,
            "versions": {name: version(name) for name in
                         ("gymnasium", "stable-baselines3", "torch", "numpy", "pandas")},
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{condition}, seed={seed}: {model.num_timesteps} шагов за {elapsed:.2f} с")
        return model
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=config.CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-frequency", type=int, default=5000)
    parser.add_argument("--split", choices=("pilot", "main"), default="pilot")
    args = parser.parse_args()
    seeds = config.PILOT_EVAL_SEEDS if args.split == "pilot" else config.INTERMEDIATE_EVAL_SEEDS
    allowed = config.PILOT_TRAIN_SEEDS if args.split == "pilot" else config.TRAIN_SEEDS
    if args.seed not in allowed:
        parser.error("Seed не входит в выбранный набор обучения")
    train(args.condition, args.seed, args.steps, args.output, args.eval_frequency, seeds)


if __name__ == "__main__":
    main()
