"""
Unit test for moonlight trait IndexError bug in origin_class.py.
"""

import pytest
import sys
import os

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.combat_state import get_state
from TFTSet4Gym.tft_set4_gym.champion import champion, reset_global_variables
from TFTSet4Gym.tft_set4_gym import origin_class

def test_moonlight_fewer_champions_than_tier_does_not_crash():
    """
    Test that when the moonlight trait tier is active (e.g. tier 1 or 2 requires upgrading champions),
    but there are fewer moonlight champions on the board than the tier demands,
    the code does not crash with IndexError.
    """
    reset_global_variables()
    state = get_state()

    # Create 1 moonlight champion (Diana)
    diana = champion("diana", team="blue", y=0, x=0, stars=1, itemlist=[])
    state.blue.append(diana)

    # Manually set the moonlight amount to trigger tier 2 (amount=5, tier=2)
    # but we only have 1 moonlight champion on the board.
    origin_class.amounts['moonlight']['blue'] = 5

    # Run the moonlight trait logic
    # This should not crash with IndexError because of min(tier, len(c_level))
    try:
        origin_class.moonlight(state.blue, state.red)
    except IndexError as e:
        pytest.fail(f"moonlight() raised IndexError: {e}")

    # Verify Diana gets upgraded to 2 stars (golden() was called on her)
    assert diana.stars == 2
