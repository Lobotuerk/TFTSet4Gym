# Technical Specification: Observation System Refactor

**Issue:** [TFT-183](mention://issue/76846d96-e26a-430c-96e6-a606566acbcf)

## 1. Overview

This document outlines the technical plan to refactor the observation generation system in `TFTSet4Gym`. The primary goal is to improve efficiency and maintainability by removing legacy code, eliminating redundant computations, and optimizing data structures. This aligns with the direction to fully adopt the new schema-based observation system provided by `ObservationBuilder`.

## 2. Background

The current implementation contains a mix of a new, schema-based system (`observation_builder.py`) and a legacy system (`observation.py`). The legacy system performs several calculations that are no longer used by the new system, leading to wasted CPU cycles. It also contains dead code and inefficient data lookups.

The user has approved a full migration to the new system, decommissioning all unused legacy components.

## 3. Proposed Changes

### 3.1. Decommission Legacy Observation Methods

The following methods in `tft_set4_gym/observation.py` will be removed entirely, as they are either dead code or produce data that is no longer consumed by the `ObservationBuilder`.

1.  **`generate_shop_vector(self, players)`**: This method calculates a 59-dimensional vector for the shop. The new `_build_shop` in `ObservationBuilder` reads champion data directly from `player.shop_elems`, making this calculation redundant.
2.  **`generate_game_comps_vector(self)`**: This method generates a 208-dimensional vector for game compositions. The new observation schema uses separate, more direct fields (`team_traits`, `team_origins`), rendering this vector obsolete.
3.  **`generate_other_player_vectors(self, cur_player, players)`**: This method is currently broken, referencing a non-existent `player_public_vector` attribute. It is also unused.

Corresponding calls to these methods, such as `step_function.generate_shop_vectors()` in `tft_simulator.py`, will also be removed.

### 3.2. Remove Legacy Attributes from `Observation` Class

The `Observation` class in `tft_set4_gym/observation.py` will be simplified by removing attributes that were only used by the decommissioned legacy methods.

-   Remove `self.shop_vector`
-   Remove `self.shop_mask`
-   Remove `self.game_comp_vector`

The `__init__` method will be updated accordingly.

### 3.3. Optimize Data Lookups

Several parts of the codebase use `list(dict.keys()).index(key)` to find the index of an item. This is an O(n) operation performed in hot paths of the observation loop.

These lookups will be optimized to O(1) by pre-computing a mapping from the key to its index.

**Locations to be refactored:**

1.  **`tft_set4_gym/observation_builder.py`**:
    -   In `_get_item_index()`: The lookup for item indices will be converted to use a pre-computed dictionary.
    -   Similar patterns for champion and trait lookups will be identified and optimized.
2.  **`tft_set4_gym/game_data/`**: The constants like `COST` will be accompanied by pre-computed index mappings. For example:
    ```python
    # Before
    COST = {"1": 29, "2": 22, ...}

    # After
    COST = {"1": 29, "2": 22, ...}
    COST_INDEX = {name: i for i, name in enumerate(COST.keys())}
    ```

### 3.4. Update Call Sites

The primary call site in `tft_simulator.py` will be updated to reflect the removal of the legacy methods.

-   In `TFT_Simulator.__init__` and `TFT_Simulator.reset`, the call to `self.step_function.generate_shop_vectors(self.PLAYERS)` will be removed.

## 4. Validation

After implementation, the following steps will be taken to ensure the changes are correct and have not introduced regressions:

1.  **Run existing tests:** Execute the test suite to confirm that all existing tests still pass.
2.  **Manual simulation:** Run a short simulation to visually inspect that the `shop` and other related observation fields are being populated correctly.
3.  **Profiler comparison (Optional but recommended):** Run a profiler before and after the changes to quantify the performance improvement.

By completing these changes, the observation system will be cleaner, more efficient, and easier to maintain for future development.
