"""
Test module for player functionality in TFT Set 4 Gym.
"""

import pytest
import itertools
import sys
import os

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tft_set4_gym.observation import Observation
from tft_set4_gym.player import Player
from tft_set4_gym.pool import pool
from tft_set4_gym.champion import champion
from tft_set4_gym import champion as c_object
from tft_set4_gym.item_stats import trait_items, starting_items
from tft_set4_gym.utils import champ_id_from_name, player_map_from_obs


def setup(player_num=0) -> Player:
    """Creates fresh player and pool"""
    # Ensure player_num is an integer (pytest fixture issue workaround)
    if not isinstance(player_num, int):
        player_num = 0
    base_pool = pool()
    player1 = Player(base_pool, player_num)
    return player1


@pytest.mark.player
def test_level2_champion():
    """Creates 3 Zileans, there should be 1 2* Zilean on bench"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 10
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 2, "champion should be 2*"
    for x in range(1, 9):
        assert p1.bench[x] is None, "these slot should be empty"
    for x in p1.board:
        for y in x:
            assert y is None, "the board should be empty"


@pytest.mark.player
def test_level3_champion():
    """Creates 9 Zileans, there should be 1 3* Zilean on bench"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 1000
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 2
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[1].stars == 2
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 3, "champion should be 3*"
    for x in range(1, 9):
        assert p1.bench[x] is None, "this slot should be empty"
    for x in p1.board:
        for y in x:
            assert y is None, "the board should be empty"


@pytest.mark.player
def test_level_champ_from_field():
    """buy third copy while 1 copy on field"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 1000
    p1.buy_champion(champion("zilean"))
    p1.buy_champion(champion("zilean"))
    p1.move_bench_to_board(1, 0, 0)
    p1.buy_champion(champion("zilean"))
    for x in p1.bench:
        assert x is None, "bench should be empty"
    assert p1.board[0][0].stars == 2, "the unit placed on the field should be 2*"


@pytest.mark.player
def test_buy_exp():
    """Test experience buying and leveling"""
    p1 = setup()
    p1.level_up()
    lvl = p1.level
    while p1.level < p1.max_level:
        p1.exp = p1.level_costs[p1.level + 1]
        p1.level_up()
        lvl += 1
        assert lvl == p1.level


@pytest.mark.player
def test_spam_exp():
    """buys tons of experience"""
    p1 = setup()
    p1.gold = 100000
    for _ in range(1000):
        p1.buy_exp()
    assert p1.level == p1.max_level, "I should be max level"
    assert p1.exp == 0, "I should not have been able to buy experience after hitting max lvl"


@pytest.mark.player
def test_income_basic():
    """first test for gold income"""
    p1 = setup()
    p1.gold = 15
    p1.gold_income(5)
    assert p1.gold == 21, f"Interest calculation is messy, gold should be 21, it is {p1.gold}"


@pytest.mark.player
def test_income_cap():
    """Check for income cap"""
    p1 = setup()
    p1.gold = 1000
    p1.gold_income(5)
    assert p1.gold == 1010, f"Interest calculation is messy, gold should be 1010, it is {p1.gold}"


@pytest.mark.player
def test_win_streak_gold():
    """Checks win streak gold"""
    p1 = setup()
    test_cases = [
        (0, 5, 0), (1, 5, 1), (2, 6, 2), (3, 6, 3), (4, 7, 4), (5, 8, 5), (500, 8, 500)
    ]
    
    for win_streak, expected_gold, streak_val in test_cases:
        p1.gold = 0
        p1.win_streak = win_streak
        p1.gold_income(5)
        assert p1.gold == expected_gold, f"Win streak {streak_val}: expected {expected_gold}, got {p1.gold}"


@pytest.mark.player
def test_loss_streak_gold():
    """Checks loss streak gold"""
    p1 = setup()
    test_cases = [
        (0, 5, 0), (-1, 5, -1), (-2, 6, -2), (-3, 6, -3), (-4, 7, -4), (-5, 8, -5), (-500, 8, -500)
    ]
    
    for loss_streak, expected_gold, streak_val in test_cases:
        p1.gold = 0
        p1.loss_streak = loss_streak
        p1.gold_income(5)
        assert p1.gold == expected_gold, f"Loss streak {streak_val}: expected {expected_gold}, got {p1.gold}"


# Legacy compatibility function
def list_of_tests():
    """Legacy function for backward compatibility."""
    test_level2_champion()
    test_level3_champion()
    test_level_champ_from_field()
    test_buy_exp()
    test_spam_exp()
    test_income_basic()
    test_income_cap()
    test_win_streak_gold()
    test_loss_streak_gold()


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("Running player tests...")
    list_of_tests()
    print("All player tests passed!")