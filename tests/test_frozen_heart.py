"""
Unit test for frozen_heart list modification fix.
"""

import pytest
import sys
import os

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.combat_state import get_state
from TFTSet4Gym.tft_set4_gym.champion import champion, reset_global_variables
from TFTSet4Gym.tft_set4_gym import items


def test_frozen_heart_remove_dead_units_no_list_corruption():
    """
    Test that when a frozen_heart holder dies, its entry in frozen_heart_list
    is removed without causing list corruption or ValueError.
    """
    reset_global_variables()
    state = get_state()

    # Create champions
    holder = champion("zilean", team="blue", y=0, x=0, stars=1, itemlist=["frozen_heart"])
    victim = champion("zilean", team="red", y=0, x=1, stars=1, itemlist=[])

    state.blue.append(holder)
    state.red.append(victim)

    # Trigger frozen_heart to apply debuff and populate frozen_heart_list
    items.frozen_heart(holder)

    # Verify frozen_heart_list is populated
    assert len(items.frozen_heart_list) > 0
    assert items.frozen_heart_list[0][0] == holder
    assert victim in items.frozen_heart_list[0][1]

    # Save initial AS before cleanup
    victim_as_after_debuff = victim.AS

    # Simulate holder dying: remove holder from team units list
    state.blue.remove(holder)

    # Call frozen_heart again on any unit to trigger the dead units cleanup loop
    items.frozen_heart(victim)

    # Verify that frozen_heart_list is now cleared for the dead holder
    assert len(items.frozen_heart_list) == 0

    # Verify AS was restored
    assert victim.AS > victim_as_after_debuff


def test_frozen_heart_multiple_dead_units():
    """
    Test that having multiple dead units in frozen_heart_list does not
    cause IndexError or list.remove ValueError.
    """
    reset_global_variables()
    state = get_state()

    holder1 = champion("zilean", team="blue", y=0, x=0, stars=1, itemlist=["frozen_heart"])
    holder2 = champion("zilean", team="blue", y=0, x=2, stars=1, itemlist=["frozen_heart"])
    victim = champion("zilean", team="red", y=0, x=1, stars=1, itemlist=[])

    state.blue.append(holder1)
    state.blue.append(holder2)
    state.red.append(victim)

    # Apply frozen_hearts
    items.frozen_heart(holder1)
    items.frozen_heart(holder2)

    assert len(items.frozen_heart_list) >= 2

    # Both holders die
    state.blue.remove(holder1)
    state.blue.remove(holder2)

    # This should run without any list mutation exceptions (the bug being tested)
    items.frozen_heart(victim)

    # Ensure everything is cleaned up
    assert len(items.frozen_heart_list) == 0
