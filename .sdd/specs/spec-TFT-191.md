# Technical Specification: TFT-191 Bug Fix

## 1. Overview

This document outlines the technical design for fixing a critical `IndexError` bug identified in issue TFT-191. The bug occurs in the shield-handling logic within the TFT Set 4 Gym environment, causing a crash during combat simulation.

## 2. Root Cause Analysis

The traceback and subsequent analysis have identified an `IndexError: list index out of range` when accessing `target.shields[0]`. This happens because the `target.shields` list can be emptied by damage calculation, but subsequent code attempts to access the first element without re-validating the list's state.

This race condition exists in two locations:
1.  `TFTSet4Gym/tft_set4_gym/champion_functions.py` in the `attack` function.
2.  `TFTSet4Gym/tft_set4_gym/champion.py` in the `spell` function.

## 3. Proposed Solution

To fix this bug, we will implement "Approach A" as discussed in the issue comments. This approach is the cleanest and most performant, as it avoids repeated list lookups and potential race conditions.

The core of the change is to get a reference to the shield dictionary *once* at the beginning of the shield damage loop and use that reference for all subsequent operations within that iteration.

### 3.1. `champion_functions.py` Modification

In `TFTSet4Gym/tft_set4_gym/champion_functions.py`, inside the `attack` function, the `while` loop that processes shield damage will be updated.

**Current (Problematic) Logic:**
```python
while target.shields and damage > 0:
    try:
        top_shield = target.shields[0]['amount']
        if top_shield > damage:
            target.shields[0]['amount'] -= damage
            damage = 0
        else:
            damage -= top_shield
            target.shields.pop(0)
    except IndexError:
        # This only protects the first access
        break
```
*Note: The actual code has a more complex structure, but this captures the essence of the bug.*

**Proposed (Corrected) Logic:**
```python
while target.shields and damage > 0:
    try:
        # Get a stable reference to the shield object
        top_shield_ref = target.shields[0]
        shield_amount = top_shield_ref['amount']

        if shield_amount > damage:
            top_shield_ref['amount'] -= damage
            damage = 0
        else:
            damage -= shield_amount
            target.shields.pop(0)
    except IndexError:
        # This guard is now more robust, though less likely to be hit
        # with the new logic if the list is empty on entry.
        break
```

### 3.2. `champion.py` Modification

A similar change will be applied to the `spell` function in `TFTSet4Gym/tft_set4_gym/champion.py`, which contains a similar vulnerable shield-damaging loop. The logic will be updated to use a stable reference to the shield object, identical to the fix in `champion_functions.py`.

## 4. Testing and Verification

The `SDD-Implementer` will be responsible for:
1.  Implementing the code changes as described above in both files.
2.  Adding a new test case to `TFTSet4Gym/tests/test_shield_handling.py`. This test should specifically trigger the scenario where a shield is completely depleted, and the `shields` list becomes empty, ensuring the `IndexError` is no longer raised. A champion with an item like Randuin's Omen should be used to reliably create this scenario.
3.  Running the full existing test suite to ensure no regressions have been introduced.

## 5. Git Implementation

The changes will be committed to the `sdd/feature-TFT-191` branch. The commit message should reference issue TFT-191.
