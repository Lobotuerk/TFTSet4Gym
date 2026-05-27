import pytest
import numpy as np
from TFTSet4Gym.tft_set4_gym.champion_availability import (
    encode_champion_availability,
    NUM_CHAMPIONS,
    BOARD_CHAMPIONS_ELEMS, BOARD_STARS_ELEMS, BOARD_CHOSEN_ELEMS, BENCH_CHAMPIONS_ELEMS,
    BOARD_CHAMPIONS_START, BOARD_STARS_START, BOARD_CHOSEN_START, BENCH_CHAMPIONS_START, FIRST_3304_SIZE,
)
from TFTSet4Gym.tft_set4_gym.stats import COST
from TFTSet4Gym.tft_set4_gym.observation_builder import ObservationBuilder
from TFTSet4Gym.tft_set4_gym.player import Player
from TFTSet4Gym.tft_set4_gym import config
from TFTSet4Gym.tft_set4_gym.observation_schema import update_observation_size_in_config


update_observation_size_in_config()


class MockPool:
    def __init__(self):
        pass
    def update_pool(self, name, amount):
        pass


class MockChampion:
    def __init__(self, name, stars=1, chosen=False):
        self.name = name
        self.stars = stars
        self.chosen = chosen


def get_champion_index(champ_name: str) -> int:
    return list(COST.keys()).index(champ_name) - 1


def test_output_shape():
    obs = np.zeros(config.OBSERVATION_SIZE)
    result = encode_champion_availability(obs)
    assert result.shape == (2 * NUM_CHAMPIONS,)
    assert result.shape == (116,)


def test_empty_observation():
    obs = np.zeros(config.OBSERVATION_SIZE)
    result = encode_champion_availability(obs)
    assert np.all(result == 0.0)


def test_single_champion_on_board():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    aatrox = MockChampion('aatrox', stars=2, chosen=False)
    player.board[0][0] = aatrox

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    aatrox_idx = get_champion_index('aatrox')
    assert result[aatrox_idx] == pytest.approx(2.0 / 3.0)
    assert result[NUM_CHAMPIONS + aatrox_idx] == 0.0
    for n in range(58):
        if n != aatrox_idx:
            assert result[n] == 0.0
            assert result[NUM_CHAMPIONS + n] == 0.0


def test_chosen_champion_on_board():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    ahri = MockChampion('ahri', stars=1, chosen=True)
    player.board[0][0] = ahri

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    ahri_idx = get_champion_index('ahri')
    assert result[ahri_idx] == pytest.approx(1.0 / 3.0)
    assert result[NUM_CHAMPIONS + ahri_idx] == 1.0


def test_three_star_champion():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    diana = MockChampion('diana', stars=3, chosen=False)
    player.board[2][1] = diana

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    diana_idx = get_champion_index('diana')
    assert result[diana_idx] == pytest.approx(1.0)
    assert result[NUM_CHAMPIONS + diana_idx] == 0.0


def test_champion_on_bench():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    zed = MockChampion('zed', stars=1, chosen=False)
    player.bench[0] = zed

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    zed_idx = get_champion_index('zed')
    assert result[zed_idx] == pytest.approx(1.0 / 3.0)
    assert result[NUM_CHAMPIONS + zed_idx] == 0.0


def test_multiple_champions_on_board_and_bench():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    vayne = MockChampion('vayne', stars=3, chosen=True)
    player.board[0][0] = vayne

    garen = MockChampion('garen', stars=1, chosen=False)
    player.board[1][1] = garen

    lissandra = MockChampion('lissandra', stars=1, chosen=False)
    player.bench[0] = lissandra

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    vayne_idx = get_champion_index('vayne')
    garen_idx = get_champion_index('garen')
    lissandra_idx = get_champion_index('lissandra')

    assert result[vayne_idx] == pytest.approx(1.0)
    assert result[NUM_CHAMPIONS + vayne_idx] == 1.0

    assert result[garen_idx] == pytest.approx(1.0 / 3.0)
    assert result[NUM_CHAMPIONS + garen_idx] == 0.0

    assert result[lissandra_idx] == pytest.approx(1.0 / 3.0)
    assert result[NUM_CHAMPIONS + lissandra_idx] == 0.0

    for n in range(58):
        if n not in (vayne_idx, garen_idx, lissandra_idx):
            assert result[n] == 0.0
            assert result[NUM_CHAMPIONS + n] == 0.0


def test_same_champion_on_board_and_bench():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    yasuo = MockChampion('yasuo', stars=2, chosen=True)
    player.board[0][0] = yasuo

    yasuo2 = MockChampion('yasuo', stars=1, chosen=False)
    player.bench[0] = yasuo2

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    yasuo_idx = get_champion_index('yasuo')
    assert result[yasuo_idx] == pytest.approx(2.0 / 3.0)
    assert result[NUM_CHAMPIONS + yasuo_idx] == 1.0


def test_full_board_takes_max_level():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    diana_1 = MockChampion('diana', stars=1, chosen=False)
    player.board[0][0] = diana_1
    diana_2 = MockChampion('diana', stars=2, chosen=False)
    player.board[0][1] = diana_2

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])

    diana_idx = get_champion_index('diana')
    assert result[diana_idx] == pytest.approx(2.0 / 3.0)


def test_accepts_first_3304_features_directly():
    obs_3304 = np.zeros(FIRST_3304_SIZE)
    aatrox_idx = get_champion_index('aatrox')

    board_champions = obs_3304[BOARD_CHAMPIONS_START:BOARD_CHAMPIONS_START + BOARD_CHAMPIONS_ELEMS].reshape(58, 4, 7)
    board_champions[aatrox_idx, 0, 0] = 1.0

    board_stars = obs_3304[BOARD_STARS_START:BOARD_STARS_START + BOARD_STARS_ELEMS].reshape(1, 4, 7)
    board_stars[0, 0, 0] = 3.0

    result = encode_champion_availability(obs_3304)
    assert result[aatrox_idx] == pytest.approx(1.0)


def test_chosen_flag_on_board():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    for i in range(7):
        for j in range(4):
            if player.board[i][j] is None:
                champ_name = list(COST.keys())[i * 4 + j + 1]
                chosen = (i == 3 and j == 2)
                player.board[i][j] = MockChampion(champ_name, stars=1, chosen=chosen)
                break
        else:
            continue
        break

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])
    assert result.shape == (116,)


def test_full_board_stress():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    champions = list(COST.keys())[1:29]
    for i in range(7):
        for j in range(4):
            idx = i * 4 + j
            player.board[i][j] = MockChampion(champions[idx], stars=(idx % 3) + 1)

    bench_champs = list(COST.keys())[29:38]
    for i in range(9):
        player.bench[i] = MockChampion(bench_champs[i])

    obs_dict = builder.build_observation("player_0", player)
    result = encode_champion_availability(obs_dict["tensor"])
    assert result.shape == (116,)

    for idx, champ_name in enumerate(champions):
        champ_idx = get_champion_index(champ_name)
        expected_level = ((idx % 3) + 1) / 3.0
        assert result[champ_idx] == pytest.approx(expected_level), f"Board {champ_name}: expected {expected_level}"

    for champ_name in bench_champs:
        champ_idx = get_champion_index(champ_name)
        assert result[champ_idx] == pytest.approx(1.0 / 3.0), f"Bench {champ_name} should be 1/3"
