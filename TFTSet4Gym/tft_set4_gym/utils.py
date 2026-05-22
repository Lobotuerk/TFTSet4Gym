from . import config as config
import numpy as np
from functools import wraps
from time import time
from .stats import COST

# Schema-based field access functions
def get_field_indices_safe(field_name, default_start=0, default_end=1):
    """Get field indices with fallback for backward compatibility."""
    try:
        from .observation_schema import get_field_indices
        return get_field_indices(field_name)
    except (ImportError, KeyError):
        # Fallback to hardcoded values if schema not available
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
    start, end = get_field_indices_safe('streak', 60 + 58 + 1 + 1 + 1 + 1 + 61, 60 + 58 + 1 + 1 + 1 + 1 + 62)
    return observation[start][0][0]
    
def gold_from_obs(observation):
    start, end = get_field_indices_safe('gold', 60 + 58 + 1 + 1 + 1 + 1 + 60, 60 + 58 + 1 + 1 + 1 + 1 + 61)
    return observation[start][0][0]

def exp_to_level_from_obs(observation):
    start, end = get_field_indices_safe('exp_to_level', 60 + 58 + 1 + 1 + 1 + 1 + 59, 60 + 58 + 1 + 1 + 1 + 1 + 60)
    return observation[start][0][0]

def hp_from_obs(observation):
    start, end = get_field_indices_safe('health', 60 + 58, 60 + 58 + 1)
    return observation[start][0][0]

def round_from_obs(observation):
    start, end = get_field_indices_safe('round', 60 + 58 + 1 + 1 + 1, 60 + 58 + 1 + 1 + 1 + 1)
    return observation[start][0][0]

def t_f_c_from_obs(observation):
    start, end = get_field_indices_safe('turns_for_combat', 60 + 58 + 1, 60 + 58 + 1 + 1)
    return observation[start][0][0]

def units_in_shop_from_obs(observation):
    shop_start, shop_end = get_field_indices_safe('shop_champions', 60 + 58 + 1 + 1 + 1 + 1, 60 + 58 + 1 + 1 + 1 + 1 + 58)
    chosen_start, chosen_end = get_field_indices_safe('shop_chosen', 60 + 58 + 1 + 1 + 1 + 1 + 58, 60 + 58 + 1 + 1 + 1 + 1 + 59)
    
    units = observation[shop_start:shop_end, 0, 0]
    chosen = observation[chosen_start][0][0]
    if int(chosen) > 0:
        chosen = list(COST.keys())[int(chosen)+1] + "_chosen"
    else:
        chosen = ""
    parsed_units = []
    for i, count in enumerate(units):
        if count > 0:
            for _ in range(int(count)):
                parsed_units.append(list(COST.keys())[i+1])
    return parsed_units, chosen

def board_from_obs(observation):
    board = observation[0:58]
    stars = observation[58]
    chosen = observation[59]
    champs = []
    for i, unit_board in enumerate(board):
        indexes = np.where(unit_board == 1.0)
        if not len(indexes[0]) == 0:
            # print(list(COST.keys())[i+1], indexes[0], indexes[1], stars[indexes[0], indexes[1]], chosen[indexes[0], indexes[1]])
            champ = {"name":list(COST.keys())[i+1],
                     "id": i,
                     "pos_y": indexes[0][0],
                     "pos_x": indexes[1][0],
                     "stars": stars[indexes[0], indexes[1]][0],
                     "chosen": chosen[indexes[0], indexes[1]][0] > 0.}
            # print(champ)
            champs.append(champ)
    return champs

def bench_from_obs(observation):
    bench_list = []
    bench = observation[60:60+58, 0, 0]
    for i,n in enumerate(bench):
        if n > 0:
            for _ in range(int(n)):
                bench_list.append(list(COST.keys())[i+1])
    return bench_list

def champ_id_from_name(champ_name):
    return (list(COST.keys()).index(champ_name)) - 1

def level_from_obs(observation):
    start, end = get_field_indices_safe('level', 60 + 58 + 1 + 1, 60 + 58 + 1 + 1 + 1)
    return observation[start][0][0]