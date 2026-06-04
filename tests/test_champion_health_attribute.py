"""
Unit test to verify that construct, galio, and aphelios_turret have their health,
max_health, and AD attributes initialized immediately upon creation,
and that field.find_target handles champions without a health attribute gracefully.
"""

import pytest
import sys
import os

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.combat_state import get_state
from TFTSet4Gym.tft_set4_gym.champion import champion, reset_global_variables
from TFTSet4Gym.tft_set4_gym import field

def test_champion_health_initialization_on_creation():
    """
    Test that construct, galio, and aphelios_turret have health, max_health,
    and AD initialized during champion.__init__, before or as soon as they
    are constructed.
    """
    reset_global_variables()
    state = get_state()

    # 1. Create a parent champion (e.g., Zilean for Aphelios turret/construct tests)
    parent = champion("zilean", team="blue", y=0, x=0, stars=1)
    
    # 2. Create construct
    c = champion("construct", team="blue", y=1, x=1, stars=1)
    assert hasattr(c, "health")
    assert hasattr(c, "max_health")
    assert hasattr(c, "AD")
    assert c.health == 1500  # Based on HEALTH['construct'][1]
    
    # 3. Create galio
    g = champion("galio", team="blue", y=2, x=2, stars=1)
    assert hasattr(g, "health")
    assert hasattr(g, "max_health")
    assert hasattr(g, "AD")

    # 4. Create aphelios_turret
    t = champion("aphelios_turret", team="blue", y=3, x=3, stars=1, overlord=parent)
    assert hasattr(t, "health")
    assert hasattr(t, "max_health")
    assert hasattr(t, "AD")
    assert t.health == 1
    assert t.max_health == 1


def test_find_target_handles_missing_health_gracefully():
    """
    Test that field.find_target handles cases where a grid cell contains a
    champion that is missing a health attribute (for example, during a transient state).
    """
    reset_global_variables()
    state = get_state()

    # Create seeker on blue team
    seeker = champion("zilean", team="blue", y=0, x=0, stars=1)
    
    # Create target on red team
    target = champion("zilean", team="red", y=0, x=1, stars=1)
    
    # Delete the health attribute to simulate the transient bug condition
    if hasattr(target, "health"):
        del target.health
        
    # Calling find_target should not raise AttributeError
    try:
        field.find_target(seeker)
    except AttributeError as e:
        pytest.fail(f"field.find_target raised AttributeError: {e}")

    # The seeker should not target the unit with missing health
    assert seeker.target is None
