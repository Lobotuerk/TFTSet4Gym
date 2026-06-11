from . import config as config
import numpy as np
from functools import wraps
from time import time
from .stats import COST


def _get_player_state_value(observation, index):
    """Extract a value from the player_state field in a flat observation."""
    from .observation_builder import get_field_value_from_obs
    player_state = get_field_value_from_obs(observation, 'player_state')
    return float(player_state.flat[index])


def _get_champion_name(champion_index):
    idx = int(round(champion_index))
    if 0 <= idx + 1 < len(COST.keys()):
        return list(COST.keys())[idx + 1]
    return None


def get_field_indices_safe(field_name, default_start=0, default_end=1):
    """Get field indices with fallback for backward compatibility."""
    try:
        from .observation_schema import get_field_indices
        return get_field_indices(field_name)
    except (ImportError, KeyError):
        hardcoded_map = {
            'health': (60 + 58, 60 + 58 + 1),
            'round': (60 + 58 + 1 + 1 + 1, 60 + 58 + 1 + 1 + 1 + 1),
            'turns_for_combat': (60 + 58 + 1, 60 + 58 + 1 + 1),
            'level': (60 + 58 + 1 + 1, 60 + 58 + 1 + 1 + 1),
            'exp_to_level': (60 + 58 + 1 + 1 + 1 + 1 + 59, 60 + 58 + 1 + 1 + 1 + 1 + 60),
            'gold': (60 + 58 + 1 + 1 + 1 + 1 + 60, 60 + 58 + 1 + 1 + 1 + 1 + 61),
            'streak': (60 + 58 + 1 + 1 + 1 + 1 + 61, 60 + 58 + 1 + 1 + 1 + 1 + 62),
            'shop_champions': (60 + 58 + 1 + 1 + 1 + 1, 60 + 58 + 1 + 1 + 1 + 1 + 58),
            'shop_chosen': (60 + 58 + 1 + 1 + 1 + 1 + 58, 60 + 58 + 1 + 1 + 1 + 1 + 59)
        }
        return hardcoded_map.get(field_name, (default_start, default_end))


def champ_binary_encode(n):
    return list(np.unpackbits(np.array([n], np.uint8))[2:8])

def champ_binary_decode(array):
    temp = list(array.copy().astype(int))
    temp.insert(0, 0)
    temp.insert(0, 0)
    return np.packbits(temp, axis=-1)[0]

def item_binary_encode(n):
    return list(np.unpackbits(np.array([n], np.uint8))[2:8])

def champ_one_hot_encode(n):
    return np.eye(config.MAX_CHAMPION_IN_SET)[n]

def item_one_hot_encode(n):
    return np.eye(9)[n]

def one_hot_encode_number(number, depth):
    return np.eye(depth)[number]


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        print(f'{f.__name__} took {elapsed} seconds to finish')
        return result
    return wrapper


def decode_action(str_actions):
    actions = []
    for str_action in str_actions:
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0, 0, 0, 0]
        for i in range(num_items + 1):
            element_list[i] = int(split_action[i])
        actions.append(np.asarray(element_list))
    return np.asarray(actions)


def x_y_to_1d_coord(x1, y1):
    if y1 == -1:
        return x1 + 28
    else:
        return 7 * y1 + x1


def player_map_from_obs(observation):
    player_map = {}
    player_map["gold"] = gold_from_obs(observation)
    player_map["shop"], player_map["chosen_shop"] = units_in_shop_from_obs(observation)
    player_map["board"] = board_from_obs(observation)
    player_map["bench"] = bench_from_obs(observation)
    player_map["level"] = level_from_obs(observation)
    player_map['hp'] = hp_from_obs(observation)
    player_map['round'] = round_from_obs(observation)
    player_map['turns_for_combat'] = t_f_c_from_obs(observation)
    player_map['exp_to_level'] = exp_to_level_from_obs(observation)
    player_map['streak'] = streak_from_obs(observation)
    return player_map


def streak_from_obs(observation):
    return _get_player_state_value(observation, 5)


def gold_from_obs(observation):
    return _get_player_state_value(observation, 1)


def exp_to_level_from_obs(observation):
    return _get_player_state_value(observation, 4)


def hp_from_obs(observation):
    return _get_player_state_value(observation, 0)


def round_from_obs(observation):
    return _get_player_state_value(observation, 3)


def t_f_c_from_obs(observation):
    return _get_player_state_value(observation, 6)


def units_in_shop_from_obs(observation):
    from .observation_builder import get_field_value_from_obs
    shop = get_field_value_from_obs(observation, 'shop')
    chosen_field = get_field_value_from_obs(observation, 'shop_chosen')

    chosen_idx = int(round(chosen_field.flat[0])) if chosen_field.size > 0 else 0

    parsed_units = []
    chosen = ""
    for slot in range(min(5, shop.shape[0])):
        champion_index = shop[slot, 0]
        if champion_index <= 0:
            continue
        champion_name = _get_champion_name(champion_index)
        if champion_name is None:
            continue
        if chosen_idx == slot + 1:
            chosen = champion_name + "_chosen"
        parsed_units.append(champion_name)
    return parsed_units, chosen


def board_from_obs(observation):
    from .observation_builder import get_field_value_from_obs
    board = get_field_value_from_obs(observation, 'board')
    champs = []
    if board.ndim != 2 or board.shape[0] != 28 or board.shape[1] != 122:
        return champs

    for slot_idx in range(28):
        champion_index = board[slot_idx, 0]
        if champion_index <= 0:
            continue
        champion_name = _get_champion_name(champion_index)
        if champion_name is None:
            continue
        champ = {
            "name": champion_name,
            "id": int(round(champion_index)),
            "pos_y": slot_idx % 4,
            "pos_x": slot_idx // 4,
            "stars": int(round(board[slot_idx, 120])),
            "chosen": board[slot_idx, 121] > 0.5
        }
        champs.append(champ)
    return champs


def bench_from_obs(observation):
    from .observation_builder import get_field_value_from_obs
    bench = get_field_value_from_obs(observation, 'bench_champions')
    bench_list = []
    if bench.ndim != 2 or bench.shape[1] != 122:
        return bench_list
    for slot_idx in range(bench.shape[0]):
        champion_index = bench[slot_idx, 0]
        if champion_index <= 0:
            continue
        champion_name = _get_champion_name(champion_index)
        if champion_name is None:
            continue
        count = max(1, int(round(bench[slot_idx, 120])))
        for _ in range(count):
            bench_list.append(champion_name)
    return bench_list


def champ_id_from_name(champ_name):
    return (list(COST.keys()).index(champ_name)) - 1


def level_from_obs(observation):
    return _get_player_state_value(observation, 2)
