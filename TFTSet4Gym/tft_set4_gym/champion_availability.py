import numpy as np
from .stats import COST

NUM_CHAMPIONS = 58

BOARD_CHAMPIONS_ELEMS = 58 * 4 * 7
BOARD_STARS_ELEMS = 1 * 4 * 7
BOARD_CHOSEN_ELEMS = 1 * 4 * 7
BENCH_CHAMPIONS_ELEMS = 58 * 4 * 7

BOARD_CHAMPIONS_START = 0
BOARD_STARS_START = BOARD_CHAMPIONS_ELEMS
BOARD_CHOSEN_START = BOARD_STARS_START + BOARD_STARS_ELEMS
BENCH_CHAMPIONS_START = BOARD_CHOSEN_START + BOARD_CHOSEN_ELEMS
FIRST_3304_SIZE = BENCH_CHAMPIONS_START + BENCH_CHAMPIONS_ELEMS


def encode_champion_availability(observation: np.ndarray) -> np.ndarray:
    obs = observation.flatten()

    board_champions = obs[BOARD_CHAMPIONS_START:BOARD_CHAMPIONS_START + BOARD_CHAMPIONS_ELEMS].reshape(58, 4, 7)
    board_stars = obs[BOARD_STARS_START:BOARD_STARS_START + BOARD_STARS_ELEMS].reshape(1, 4, 7)
    board_chosen = obs[BOARD_CHOSEN_START:BOARD_CHOSEN_START + BOARD_CHOSEN_ELEMS].reshape(1, 4, 7)
    bench_champions = obs[BENCH_CHAMPIONS_START:BENCH_CHAMPIONS_START + BENCH_CHAMPIONS_ELEMS].reshape(58, 4, 7)

    level = np.zeros(58)
    chosen = np.zeros(58)

    for n in range(58):
        ys, xs = np.where(board_champions[n] > 0.5)
        if len(ys) > 0:
            level[n] = max(board_stars[0, y, x] for y, x in zip(ys, xs))
            chosen[n] = 1.0 if any(board_chosen[0, y, x] > 0.5 for y, x in zip(ys, xs)) else 0.0
        elif np.any(bench_champions[n] > 0.5):
            level[n] = 1.0

    result = np.zeros(2 * 58)
    result[0:58] = level / 3.0
    result[58:116] = chosen
    return result
