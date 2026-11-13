from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

from src import config
from src.envs import make_env
from src.plots import COLORS, save


def main():
    plt.switch_backend('Agg')
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/animations'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.TORCH_THREADS)
    rows = []
    for condition in config.CONDITIONS:
        model = SAC.load(root / f'artifacts/main_{condition}_{config.VIDEO_TRAIN_SEED}/model.zip', device='cpu')
        for delay in (0, config.PRIMARY_DELAY):
            env = make_env(delay=delay, render_mode='rgb_array')
            frames = []
            try:
                observation, _ = env.reset(seed=config.VIDEO_EPISODE_SEED)
                for step in range(200):
                    angle, velocity = env.unwrapped.state
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    rows.append(dict(condition=condition, delay=delay, step=step,
                                     angle=(angle + np.pi) % (2 * np.pi) - np.pi,
                                     velocity=velocity, action=float(action[0]), reward=float(reward)))
                    frames.append(Image.fromarray(env.render()).resize((300, 300)))
                    if terminated or truncated:
                        break
                frames[0].save(output / f'{condition}_delay_{delay}.gif', save_all=True,
                               append_images=frames[1:], duration=round(env.unwrapped.dt * 1000), loop=0)
            finally:
                env.close()
    table = pd.DataFrame(rows)
    table.to_csv(output / 'trajectories.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for column, delay in enumerate((0, config.PRIMARY_DELAY)):
        for condition in config.CONDITIONS:
            part = table[(table.condition == condition) & (table.delay == delay)]
            for row, value in enumerate(('angle', 'action')):
                axes[row, column].plot(part.step * 0.05, part[value],
                                       color=COLORS[condition], label=condition)
                axes[row, column].grid(alpha=0.25)
                axes[row, column].legend()
        axes[0, column].set(title=f'Задержка {delay} шага', ylabel='Угол, рад')
        axes[1, column].set(xlabel='Время, с', ylabel='Момент силы, Н·м')
    save(fig, output / 'behavior.png')
    print(table.groupby(['condition', 'delay']).reward.sum().to_string())


if __name__ == '__main__':
    main()
