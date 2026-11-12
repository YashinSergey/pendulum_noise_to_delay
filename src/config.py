ENV_ID = "Pendulum-v1"
DEVICE = "cpu"
CONDITIONS = ("clean", "noisy")

ANGLE_NOISE_STD = 0.05
VELOCITY_NOISE_STD = 0.1
DELAYS = (0, 1, 2, 4)
PRIMARY_DELAY = 2

PILOT_TRAIN_SEEDS = (0, 1, 2)
PILOT_EVAL_SEEDS = tuple(range(100, 105))
TRAIN_SEEDS = (42, 43, 44)
INTERMEDIATE_EVAL_SEEDS = tuple(range(200, 205))
TEST_SEEDS = tuple(range(1000, 1020))
VIDEO_EPISODE_SEED = 2000
VIDEO_TRAIN_SEED = 42
RANDOM_POLICY_SEED = 3000

MAX_TRAIN_SECONDS = 1800
TOTAL_TIMESTEPS = 30000
EVAL_FREQUENCY = 5000

# Общие настройки зафиксированы после пилота, до основных запусков
TORCH_THREADS = 1
NET_ARCH = (64, 64)
SAC_SETTINGS = {
    "learning_rate": 0.0003,
    "buffer_size": 100000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
}
