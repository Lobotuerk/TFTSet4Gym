# Technical Specification: Observation System Refactor (TFT-183)

## 1. Objective

This document outlines the technical plan to refactor the observation generation system in `TFTSet4Gym`. The primary goals are to improve performance, remove dead and redundant code, and fully transition to the new schema-based observation system introduced in `observation_builder.py`, decommissioning legacy components.

## 2. Analysis & Background

A prior discovery phase ([TFT-183](mention://issue/76846d96-e26a-430c-96e6-a606566acbcf)) confirmed several inefficiencies and legacy artifacts that need to be addressed. The user has approved the complete removal of all legacy code paths in favor of the new `ObservationBuilder` system.

### 2.1. Redundant Shop Vector Calculation

- **Location:** `observation.py`, `generate_shop_vector()`
- **Issue:** This function iterates through the player's shop to count champion occurrences, creating a 58-element `shop_counts` array. This array is prepended to the `shop_vector`. However, the new system's `_build_shop()` method in `observation_builder.py` completely ignores this count, instead reading directly from `player.shop_elems` to generate the required 5x32 champion embedding matrix. The entire champion counting loop is unnecessary work.

### 2.2. Dead Code and Unused Attributes

- **`generate_other_player_vectors()`:**
    - **Location:** `observation.py`
    - **Issue:** This method attempts to access `other_player.player_public_vector`, an attribute that does not exist on the `Player` object, which would cause an `AttributeError` if ever called. This method is unused and broken.
- **`generate_game_comps_vector()`:**
    - **Location:** `observation.py`
    - **Issue:** The call to this method is commented out in `step_function.py`. The 208-dim vector it produces is not consumed by the new observation schema, which uses separate fields for team traits and origins. This method is dead code.
- **Legacy `Observation` Attributes:**
    - **Location:** `observation.py`, `Observation.__init__()`
    - **Issue:** The attributes `self.shop_vector` and `self.game_comp_vector` are relics of the old system and are no longer used by the `ObservationBuilder`. They serve no purpose and should be removed.

### 2.3. Performance Bottlenecks (O(n) Lookups)

- **Locations:**
    - `observation_builder.py`: `_get_item_index()`
    - `observation.py`: `generate_shop_vector()` (legacy)
- **Issue:** These functions perform lookups like `list(some_dict.keys()).index(key)`. This is an O(n) operation that is inefficient and runs on every observation step. These should be replaced with pre-computed dictionaries for O(1) lookups.

## 3. Proposed Changes

The following changes will be implemented to address the issues identified above.

### 3.1. `observation.py` Refactoring

1.  **Remove `generate_other_player_vectors`:** Delete the entire `generate_other_player_vectors` method from the `Observation` class.
2.  **Remove `generate_game_comps_vector`:** Delete the entire `generate_game_comps_vector` method from the `Observation` class.
3.  **Clean up `Observation.__init__`:**
    -   Remove the initialization of `self.shop_vector = None`.
    -   Remove the initialization of `self.game_comp_vector = np.zeros(208)`.
4.  **Simplify `generate_shop_vector`:**
    -   The champion counting loop (iterating `player.shop` to produce `shop_counts`) will be completely removed.
    -   The function's responsibility will be reduced. Since the new `ObservationBuilder` handles everything, this function and its call site in `step_function.py` will be removed. The `build_observation` function should be called directly with the `player` object.

### 3.2. `observation_builder.py` Optimizations

1.  **Create Item-to-Index Mappings:**
    -   Two new module-level dictionaries will be created to serve as O(1) lookup tables for item indices.
    ```python
    _ITEM_BUILDS_INDEX = {name: i for i, name in enumerate(item_builds.keys())}
    _UNCRAFTABLE_ITEMS_INDEX = {name: i for i, name in enumerate(uncraftable_items)}
    ```
2.  **Optimize `_get_item_index()`:**
    -   The implementation of `_get_item_index` will be updated to use these new dictionaries for efficient lookups.
    ```python
    # Proposed new implementation
    def _get_item_index(item_name: str) -> int:
        """Get the embedding index for an item name."""
        if item_name in _ITEM_BUILDS_INDEX:
            return _ITEM_BUILDS_INDEX[item_name] + len(uncraftable_items)
        elif item_name in _UNCRAFTABLE_ITEMS_INDEX:
            return _UNCRAFTABLE_ITEMS_INDEX[item_name]
        return 0
    ```

### 3.3. `step_function.py` Updates

1.  **Remove Legacy Calls:** The line `obs.generate_shop_vector(player)` will be removed.
2.  **Adopt New Observation Call:** The main observation generation will be simplified to a direct call to the builder's main entry point, likely `obs.build_full_observation(...)` or `build_observation(...)`, ensuring the `shop_vector` argument is no longer passed as it's internally derived.

## 4. Validation

After implementation, the following must be verified:
1.  All unit and integration tests must pass.
2.  A manual inspection of the observation tensor generated by the refactored code should confirm that the `shop`, `opponent_boards`, and `opponent_info` fields are correctly populated without the legacy methods.
3.  Profiling the `step` function should show a measurable performance improvement, particularly in the observation generation phase.
