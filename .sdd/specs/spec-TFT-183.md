# Technical Specification: Observation System Refactor (TFT-183)

## 1. Overview

This document outlines the technical plan to refactor and streamline the observation generation system. The current implementation contains redundant calculations, dead code, and inefficient data lookups. This refactor will remove legacy code paths, optimize performance-critical sections, and align the entire observation generation process with the new schema-based system introduced in [TFT-182](mention://issue/33c36125-cd22-496e-b107-9837e61f6cc2).

## 2. Background

The `Observation` class in `observation.py` contains legacy code that is no longer used by the new `ObservationBuilder` in `observation_builder.py`. The user has confirmed that all legacy code should be removed, and the system should rely exclusively on the new observation schema.

## 3. Proposed Changes

### 3.1. Removal of Redundant Shop Vector Calculation

-   **File:** `TFTSet4Gym/tft_gym/envs/observation.py`
-   **Action:** The `shop_counts` array calculation within `generate_shop_vector()` (lines 149-169) will be removed. The new `_build_shop()` method in `observation_builder.py` reads `player.shop_elems` directly, making this calculation obsolete.

### 3.2. Removal of Dead and Unused Code

-   **File:** `TFTSet4Gym/tft_gym/envs/observation.py`
-   **Action:**
    -   The `generate_other_player_vectors()` method will be removed. It references a non-existent attribute (`player_public_vector`) and is currently dead code.
    -   The `generate_game_comps_vector()` method will be removed. It is commented out in `step_function.py` and its functionality is replaced by the new schema's `team_traits` and `team_origins` fields.
    -   The corresponding attributes on the `Observation` class (`self.shop_vector`, `self.shop_mask`, `self.game_comp_vector`) will be removed.

### 3.3. Optimization of Data Lookups

-   **Files:**
    -   `TFTSet4Gym/tft_gym/envs/observation.py`
    -   `TFTSet4Gym/tft_gym/envs/observation_builder.py`
-   **Action:** All `list(dict.keys()).index()` calls, which are O(n), will be replaced with direct dictionary lookups, which are O(1). This will be done by creating inverted index dictionaries for `COST` and `item_builds` at initialization.

    -   In `observation.py`, the `COST` lookup will be optimized.
    -   In `observation_builder.py`, the `_get_item_index` method will be updated to use a pre-computed index for `item_builds`.

### 3.4. General Code Cleanup

-   **Action:** Any remaining legacy code paths, try-catch blocks for handling old data shapes, or other unused variables related to the removed methods will be deleted to simplify the codebase.

## 4. Implementation Plan

1.  **Create inverted indexes:** Create and store inverted indexes for `COST` and `item_builds` for O(1) lookups.
2.  **Refactor `observation.py`:**
    -   Remove the methods `generate_other_player_vectors` and `generate_game_comps_vector`.
    -   Remove the redundant computation from `generate_shop_vector`.
    -   Remove the class attributes `self.shop_vector`, `self.shop_mask`, and `self.game_comp_vector`.
3.  **Refactor `observation_builder.py`:**
    -   Update `_get_item_index` to use the new inverted index for `item_builds`.
4.  **Testing:** Run existing tests and, if necessary, add new tests to verify that the observation generation remains correct after the refactor.

## 5. Risks and Mitigation

-   **Risk:** Removing legacy code could have unforeseen side effects if any part of the system still relies on it.
-   **Mitigation:** The user has explicitly approved the removal of all legacy code paths. A thorough code search will be conducted to ensure no other components are using the legacy observation fields.
