"""
Unit tests for shield depletion handling in champion attacks and spells.
Verifies that IndexError is not raised when shields are fully consumed.
"""

import pytest
import sys
import os

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.combat_state import get_state
from TFTSet4Gym.tft_set4_gym.champion import champion, reset_global_variables
from TFTSet4Gym.tft_set4_gym.champion_functions import attack


def test_attack_shield_depletion_no_index_error():
    """
    Test that when an attack deals damage exceeding the target's shield,
    the shield is fully consumed and removed from shields list without raising IndexError.
    """
    reset_global_variables()
    state = get_state()

    attacker = champion("zilean", team="blue", y=0, x=0, stars=1)
    target = champion("zilean", team="red", y=0, x=1, stars=1)

    state.blue.append(attacker)
    state.red.append(target)

    # Disable random elements (crit/dodge) to make tests deterministic
    attacker.crit_chance = 0
    target.dodge = 0

    # Give target a small shield
    target.shields = [{'amount': 10, 'original_amount': 10, 'applier': attacker, 'identifier': 'test_shield'}]
    
    # Set target health high to prevent death
    target.health = 500
    target.max_health = 500

    # Ensure attacker has enough AD to break the shield
    attacker.AD = 100

    # Execute attack
    attacker.attack(target=target)

    # Verify shield was completely depleted and removed
    assert len(target.shields) == 0
    assert target.shield_amount() == 0


def test_spell_shield_depletion_no_index_error():
    """
    Test that when a spell deals damage exceeding the target's shield,
    the shield is fully consumed and removed from shields list without raising IndexError.
    """
    reset_global_variables()
    state = get_state()

    caster = champion("zilean", team="blue", y=0, x=0, stars=1)
    target = champion("zilean", team="red", y=0, x=1, stars=1)

    state.blue.append(caster)
    state.red.append(target)

    # Disable random elements to make tests deterministic
    caster.crit_chance = 0
    target.dodge = 0

    # Give target a shield
    target.shields = [{'amount': 50, 'original_amount': 50, 'applier': caster, 'identifier': 'test_shield'}]
    
    target.health = 500
    target.max_health = 500

    # Trigger caster spell dealing magic/true damage exceeding the shield
    # spell(self, target, dmg, true_dmg=0, item_damage=False, burn_damage=False, trait_damage=False)
    caster.spell(target, dmg=200)

    # Verify shield is gone
    assert len(target.shields) == 0
    assert target.shield_amount() == 0


def test_multiple_shields_cascade():
    """
    Test that damage cascades across multiple shields and removes them appropriately.
    """
    reset_global_variables()
    state = get_state()

    attacker = champion("zilean", team="blue", y=0, x=0, stars=1)
    target = champion("zilean", team="red", y=0, x=1, stars=1)

    state.blue.append(attacker)
    state.red.append(target)

    # Disable random elements to make tests deterministic
    attacker.crit_chance = 0
    target.dodge = 0

    # Give target multiple shields
    target.shields = [
        {'amount': 30, 'original_amount': 30, 'applier': attacker, 'identifier': 'shield_1'},
        {'amount': 40, 'original_amount': 40, 'applier': attacker, 'identifier': 'shield_2'},
    ]
    
    target.health = 500
    target.max_health = 500

    # Attacker AD set to deal exactly 50 damage
    target.armor = 0
    attacker.AD = 50

    attacker.attack(target=target)

    # 50 damage should:
    # 1. deplete shield_1 (30 amount), removing it.
    # 2. deplete remaining 20 from shield_2, leaving shield_2 with 20 amount.
    assert len(target.shields) == 1
    assert target.shields[0]['identifier'] == 'shield_2'
    assert target.shields[0]['amount'] == 20
    assert target.shield_amount() == 20
