"""
Unit test for trap_claw UnboundLocalError and logical flow bug in champion.py.
"""

import pytest
import sys
import os

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.combat_state import get_state
from TFTSet4Gym.tft_set4_gym.champion import champion, reset_global_variables


def test_trap_claw_blocks_spell_without_unbound_local_error():
    """
    Test that when a target has trap_claw, casting a spell on them:
    1. Does not crash with UnboundLocalError (for damage or team_list).
    2. Correctly triggers trap_claw item effect on the target (removing it and stunning the caster).
    3. Does not apply spell damage or other on-hit spell effects to the target.
    """
    reset_global_variables()
    state = get_state()

    caster = champion("zilean", team="blue", y=0, x=0, stars=1, itemlist=[])
    target = champion("zilean", team="red", y=0, x=1, stars=1, itemlist=["trap_claw"])

    state.blue.append(caster)
    state.red.append(target)

    # Initially target has trap_claw, caster is not stunned, target health is full (600 for Zilean 1*)
    assert "trap_claw" in target.items
    assert not caster.stunned
    initial_health = target.health

    # Cast a spell
    caster.spell(target, dmg=100)

    # After the spell is blocked by trap_claw:
    # 1. trap_claw should be consumed/removed from target items
    assert "trap_claw" not in target.items

    # 2. Caster should be stunned by the trap_claw effect
    assert caster.stunned

    # 3. Target should have taken 0 damage (health unchanged)
    assert target.health == initial_health
