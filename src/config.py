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

# Бюджет и параметры SAC выбираются после замера скорости в пилоте
MAX_TRAIN_SECONDS = 1800
