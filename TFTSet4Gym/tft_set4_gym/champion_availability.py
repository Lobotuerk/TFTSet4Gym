import numpy as np
from .observation_schema import get_observation_schema

NUM_CHAMPIONS = 58

_PER_SLOT_SIZE = 122
_STAR_SLICE = 120
_CHOSEN_SLICE = 121


def encode_champion_availability(observation: np.ndarray) -> np.ndarray:
    obs = observation.flatten()
    schema = get_observation_schema("current_player")

    board_slice = schema.get_field_slice("board")
    bench_slice = schema.get_field_slice("bench_champions")

    board_data = obs[board_slice].reshape(-1, _PER_SLOT_SIZE)
    bench_data = obs[bench_slice].reshape(-1, _PER_SLOT_SIZE)

    level = np.zeros(58)
    chosen = np.zeros(58)

    for slot_idx in range(board_data.shape[0]):
        slot = board_data[slot_idx]
        star_level = slot[_STAR_SLICE]
        if star_level > 0:
            champ_idx = int(round(slot[0]))
            if 0 <= champ_idx < 58:
                if star_level > level[champ_idx]:
                    level[champ_idx] = star_level
                if slot[_CHOSEN_SLICE] > 0.5:
                    chosen[champ_idx] = 1.0

    for slot_idx in range(bench_data.shape[0]):
        slot = bench_data[slot_idx]
        star_level = slot[_STAR_SLICE]
        if star_level > 0:
            champ_idx = int(round(slot[0]))
            if 0 <= champ_idx < 58:
                if star_level > level[champ_idx]:
                    level[champ_idx] = star_level
                if slot[_CHOSEN_SLICE] > 0.5:
                    chosen[champ_idx] = 1.0

    result = np.zeros(2 * 58)
    result[0:58] = level / 3.0
    result[58:116] = chosen
    return result
