from collections import deque

import gymnasium as gym
import numpy as np

from src import config


class ObservationNoise(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        angle_std=config.ANGLE_NOISE_STD,
        velocity_std=config.VELOCITY_NOISE_STD,
    ):
        super().__init__(env)
        if angle_std < 0 or velocity_std < 0:
            raise ValueError("Стандартные отклонения должны быть неотрицательными")
        self.angle_std = angle_std
        self.velocity_std = velocity_std
        self.noise_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Отдельный поток не расходует случайные числа базовой среды
            self.noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        return super().reset(seed=seed, options=options)

    def observation(self, observation):
        noisy = observation.copy()
        if self.angle_std > 0:
            angle = np.arctan2(observation[1], observation[0])
            angle += self.noise_rng.normal(0, self.angle_std)
            noisy[:2] = (np.cos(angle), np.sin(angle))
        if self.velocity_std > 0:
            noisy[2] = np.clip(
                observation[2] + self.noise_rng.normal(0, self.velocity_std),
                self.observation_space.low[2],
                self.observation_space.high[2],
            )
        return noisy


class ObservationDelay(gym.ObservationWrapper):
    def __init__(self, env, delay):
        super().__init__(env)
        if not isinstance(delay, int) or delay < 0:
            raise ValueError("Задержка должна быть целым неотрицательным числом")
        self.delay = delay
        self.history = deque(maxlen=delay + 1)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.history.clear()
        self.history.extend(observation.copy() for _ in range(self.delay + 1))
        return observation.copy(), info

    def observation(self, observation):
        self.history.append(observation.copy())
        return self.history[0].copy()


def make_env(condition="clean", delay=0, render_mode=None):
    if condition not in config.CONDITIONS:
        raise ValueError(f"Неизвестная конфигурация: {condition}")
    if not isinstance(delay, int) or delay < 0:
        raise ValueError("Задержка должна быть целым неотрицательным числом")
    if condition == "noisy" and delay > 0:
        raise ValueError("В эксперименте шум и задержка применяются раздельно")

    env = gym.make(config.ENV_ID, render_mode=render_mode)
    if condition == "noisy":
        env = ObservationNoise(env)
    if delay > 0:
        env = ObservationDelay(env, delay)
    return env
