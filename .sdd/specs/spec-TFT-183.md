# Technical Specification: Observation System Refactor (TFT-183)

## 1. Overview

This document outlines the technical plan to refactor the observation generation system in `TFTSet4Gym`. The goal is to improve efficiency by removing redundant computations, eliminating dead code, and optimizing data lookups, as approved in the issue discussion. The refactor will fully deprecate the legacy observation components in favor of the new schema-based system (`ObservationBuilder`).

## 2. Background

The current implementation contains a mix of legacy and new observation logic.
- **Legacy:** `observation.py` contains methods (`generate_shop_vector`, `generate_game_comps_vector`, `generate_other_player_vectors`) that perform unnecessary calculations or are no longer used.
- **New:** `observation_builder.py` implements a more efficient, schema-driven approach.

The `SDD-Discovery` agent identified several key inefficiencies:
- A redundant `shop_counts` array is built but never used by the new system.
- `generate_other_player_vectors` references a non-existent `player_public_vector` attribute.
- `generate_game_comps_vector` is dead code.
- Several hot paths use O(n) list index searches instead of O(1) dictionary lookups.

This specification details the plan to address these issues.

## 3. Proposed Changes

### 3.1. Decommission Legacy `Observation` Class Methods

The following methods in `tft_set4_gym/observation.py` will be removed:

1.  **`generate_shop_vector(self, players)`**: This method calculates a 58-dimension `shop_counts` vector that is ignored by the new `ObservationBuilder._build_shop()`.
    - **Action:** Delete the entire method.
    - **Impact:** The call to this method in `tft_set4_gym/step_function.py` at `line 125` must also be removed.

2.  **`generate_game_comps_vector(self)`**: This method is commented out in its only call site and its output `game_comp_vector` is not consumed by the new observation schema.
    - **Action:** Delete the entire method. The corresponding attribute `self.game_comp_vector` in `Observation.__init__` will also be removed.

3.  **`generate_other_player_vectors(self, cur_player, players)`**: This method is not called anywhere and contains a reference to a non-existent attribute (`other_player.player_public_vector`), which would cause a runtime `AttributeError`.
    - **Action:** Delete the entire method.

### 3.2. Optimize Data Lookups

The following files contain inefficient O(n) list-based index lookups that will be converted to O(1) dictionary-based lookups.

1.  **`tft_set4_gym/observation.py`**:
    - **Location:** `generate_shop_vector` method, line ~167.
    - **Current:** `list(COST.keys()).index(champ_cost)`
    - **Proposed:** Create a reverse mapping `COST_TO_INDEX = {cost: i for i, cost in enumerate(COST.keys())}` at the module level and use `COST_TO_INDEX[champ_cost]`.
    - **Note:** Since `generate_shop_vector` is being removed, this specific change is now part of the deletion. If any similar pattern exists elsewhere, it should be refactored.

2.  **`tft_set4_gym/observation_builder.py`**:
    - **Location:** `_get_item_index` method, lines ~32-34.
    - **Current:** `list(item_builds.keys()).index(item_name)`
    - **Proposed:** Create a memoized or module-level reverse mapping for `item_builds` keys.
      ```python
      # At module level or cached within the class
      ITEM_TO_INDEX = {name: i for i, name in enumerate(item_builds.keys())}

      # In _get_item_index
      return ITEM_TO_INDEX.get(item_name, -1) # Use .get for safety
      ```

### 3.3. Clean Up `Observation` Class Attributes

The `Observation` class `__init__` method in `tft_set4_gym/observation.py` will be cleaned up to remove attributes related to the decommissioned legacy methods.

- **Action:**
    - Remove `self.shop_vector = None`.
    - Remove `self.shop_mask = np.ones(5, dtype=np.int8)`.
    - Remove `self.game_comp_vector = np.zeros(208)`.

The `build_full_observation` method will be updated to no longer accept a `shop_vector` argument, as it's no longer generated or needed.

### 3.4. Remove Unused Call Sites

The primary call site for the legacy observation generation is in `tft_set4_gym/step_function.py`.

- **Location:** `Step_Function.generate_shop_vectors`.
- **Action:** This method will be removed entirely, as its only purpose was to orchestrate the creation of the legacy `shop_vector`. All calls to `generate_shop_vectors` (e.g., in `tft_simulator.py`) will also be removed.

## 4. Implementation Plan

1.  **Modify `tft_set4_gym/observation_builder.py`**:
    - Implement the O(1) lookup for `_get_item_index`.

2.  **Modify `tft_set4_gym/observation.py`**:
    - Remove the methods: `generate_shop_vector`, `generate_game_comps_vector`, `generate_other_player_vectors`.
    - Remove the legacy attributes from `__init__`.
    - Update `build_full_observation` method signature.

3.  **Modify `tft_set4_gym/step_function.py`**:
    - Remove the `generate_shop_vectors` method.

4.  **Modify `tft_set4_gym/tft_simulator.py`**:
    - Remove all calls to `step_function.generate_shop_vectors`.

## 5. Validation

- All existing unit tests must continue to pass.
- New tests may be added to verify the O(1) lookup optimizations if not already covered.
- A full simulation run should be executed to ensure the environment operates correctly with the refactored observation system.
