import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"clean": "#2463a0", "noisy": "#d46a25"}


def band(ax, table, x, y):
    for condition, group in table.groupby("condition"):
        stats = group.groupby(x)[y].agg(["mean", "std"])
        ax.plot(stats.index, stats['mean'], label=condition, color=COLORS[condition])
        ax.fill_between(stats.index, stats['mean'] - stats['std'],
                        stats['mean'] + stats['std'], alpha=0.18, color=COLORS[condition])
    ax.grid(alpha=0.25)
    ax.legend()


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    plt.switch_backend('Agg')
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "results/main")
    parser.add_argument("--output", type=Path, default=root / "results/figures")
    args = parser.parse_args()
    data = args.data
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(data / 'training.csv')
    training['smoothed'] = training.groupby(['condition', 'train_seed']).reward.transform(
        lambda values: values.rolling(10, min_periods=1).mean()
    )
    intermediate = pd.read_csv(data / 'intermediate.csv').groupby(
        ['condition', 'train_seed', 'timesteps'], as_index=False
    ).reward.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    band(axes[0], training, 'timesteps', 'smoothed')
    band(axes[1], intermediate, 'timesteps', 'reward')
    axes[0].set(title='Обучение: среднее последних 10 эпизодов')
    axes[1].set(title='Промежуточная оценка: чистая среда')
    for ax in axes:
        ax.set(xlabel='Взаимодействия со средой, шаги', ylabel='Награда за эпизод')
    save(fig, output / 'learning.png')

    runs = pd.read_csv(data / 'runs.csv')
    random_reward = pd.read_csv(data / 'random.csv').reward.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    band(axes[0], runs, 'delay', 'reward')
    band(axes[1], runs, 'delay', 'loss')
    axes[0].axhline(random_reward, color='gray', linestyle=':', label='Случайная политика, d=0')
    axes[0].legend()
    axes[0].set(title='Итоговая награда', ylabel='Награда за эпизод')
    axes[1].set(title='Потеря относительно отсутствия задержки', ylabel='Разница наград за эпизод')
    for ax in axes:
        ax.set(xlabel='Задержка наблюдения, шаги', xticks=[0, 1, 2, 4])
    save(fig, output / 'delay.png')

    effects = pd.read_csv(data / 'effects.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(effects.train_seed.astype(str), effects.effect, color='#2463a0', label='E для каждого seed')
    ax.axhline(0, color='black', linewidth=1)
    ax.set(title='Основное сравнение при задержке 2 шага', xlabel='Seed обучения',
           ylabel='E: разница потерь награды')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    save(fig, output / 'effects.png')


if __name__ == '__main__':
    main()
