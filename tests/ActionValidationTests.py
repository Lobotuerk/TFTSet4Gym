"""
Test module for verifying that illegal actions are handled gracefully.
"""

import pytest
import sys
import os
import numpy as np

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.player import Player
from TFTSet4Gym.tft_set4_gym.pool import pool
from TFTSet4Gym.tft_set4_gym.champion import champion
from TFTSet4Gym.tft_set4_gym import config

def setup_player(player_num=0) -> Player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = Player(base_pool, player_num)
    return player1

@pytest.mark.validation
def test_buy_champion_no_gold():
    """Verify that buying a champion with 0 gold fails gracefully."""
    p = setup_player()
    p.gold = 0
    
    # Try to buy a 1-cost champion
    success = p.buy_champion(champion("nidalee"))
    
    assert success is False
    assert p.gold == 0
    assert all(slot is None for slot in p.bench)

@pytest.mark.validation
def test_buy_exp_no_gold():
    """Verify that buying EXP with insufficient gold fails gracefully."""
    p = setup_player()
    p.gold = 2 # Need 4
    p.exp = 0
    
    success = p.buy_exp()
    
    assert success is False
    assert p.gold == 2
    assert p.exp == 0

@pytest.mark.validation
def test_refresh_no_gold():
    """Verify that refreshing the shop with insufficient gold fails gracefully."""
    p = setup_player()
    p.gold = 1 # Need 2
    
    success = p.refresh()
    
    assert success is False
    assert p.gold == 1

@pytest.mark.validation
def test_move_bench_to_board_at_max_capacity():
    """
    Verify that moving a unit from bench to an empty board slot fails 
    if the board is already at max capacity.
    """
    p = setup_player()
    p.gold = 100
    p.max_units = 1
    
    # Put a unit on board [0, 0]
    p.buy_champion(champion("nidalee"))
    p.move_bench_to_board(0, 0, 0)
    assert p.num_units_in_play == 1
    
    # Put another unit on bench
    p.buy_champion(champion("fiora"))
    
    # Try to move fiora to board [1, 1] (empty slot)
    # Should fail because max_units is 1 and board already has 1.
    success = p.move_bench_to_board(0, 1, 1)
    
    assert success is False
    assert p.board[1][1] is None
    assert p.num_units_in_play == 1

@pytest.mark.validation
def test_sell_non_existent_unit_bench():
    """Verify that selling from an empty bench slot handles gracefully."""
    p = setup_player()
    p.gold = 10
    
    # Sell from empty bench slot 0
    result = p.sell_from_bench(0)
    
    assert result is False
    assert p.gold == 10

@pytest.mark.validation
def test_sell_non_existent_unit_board():
    """Verify that selling from an empty board slot handles gracefully."""
    p = setup_player()
    p.gold = 10
    
    # Sell from empty board slot [0, 0]
    # Note: sell_champion doesn't check if slot is empty, but batch_2d_controller does
    # Let's test how player handles it directly if possible
    # In player.py, sell_champion takes the champion object, so we can't call it with None easily.
    # But move_board_to_bench handles selling if bench is full.
    
    # If we call sell_champion with None it might crash, but the controller prevents this.
    # Let's check move_board_to_bench with empty slot.
    success = p.move_board_to_bench(0, 0)
    assert success is False
    assert p.gold == 10

@pytest.mark.validation
def test_move_to_invalid_index():
    """Verify that moving to/from invalid indices fails gracefully."""
    p = setup_player()
    p.buy_champion(champion("nidalee"))
    
    # Invalid bench index
    assert p.move_bench_to_board(10, 0, 0) is False
    assert p.move_bench_to_board(-1, 0, 0) is False
    
    # Invalid board index
    assert p.move_bench_to_board(0, 7, 0) is False
    assert p.move_bench_to_board(0, 0, 4) is False
    
    # Board to bench invalid
    assert p.move_board_to_bench(7, 0) is False
    assert p.move_board_to_bench(0, 4) is False
    
    # Board to board invalid
    assert p.move_board_to_board(0, 0, 7, 0) is False
    assert p.move_board_to_board(7, 0, 0, 0) is False

@pytest.mark.validation
def test_buy_champion_full_bench():
    """
    Verify that buying a champion with a full bench auto-sells it.
    This is the intended behavior in the current simulator implementation.
    """
    p = setup_player()
    p.gold = 100
    
    # Fill bench with UNIQUE champions to avoid tripling
    champions_to_buy = ["fiora", "nidalee", "garen", "vayne", "maokai", "elise", "twistedfate", "diana", "yasuo"]
    for champ_name in champions_to_buy:
        p.buy_champion(champion(champ_name))
    
    assert all(slot is not None for slot in p.bench)
    
    gold_before = p.gold
    # Buy another one (e.g., Tahm Kench)
    success = p.buy_champion(champion("tahmkench"))
    
    assert success is False
    # Gold should be same (bought for 1, sold for 1)
    assert p.gold == gold_before
    # Bench should still have the same champions as before (no tahmkench)
    bench_names = [slot.name for slot in p.bench if slot is not None]
    assert "tahmkench" not in bench_names
    for name in champions_to_buy:
        assert name in bench_names

