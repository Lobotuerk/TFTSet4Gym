# Technical Specification: Observation System Refactor (TFT-183)

## 1. Overview

This document outlines the technical plan to refactor the observation generation system in `TFTSet4Gym`. The goal is to improve performance, increase code clarity, and remove all legacy code paths, aligning the system with the modern, schema-based approach in `ObservationBuilder` for version 1.0.0.

The core of this refactor involves removing redundant calculations, eliminating dead code, and optimizing data lookups from O(n) to O(1).

## 2. Files to be Modified

-   `TFTSet4Gym/tft_set4_gym/observation.py`
-   `TFTSet4Gym/tft_set4_gym/observation_builder.py`
-   `TFTSet4Gym/tft_set4_gym/step_function.py`
-   `TFTSet4Gym/tft_set4_gym/player.py`
-   `TFTSet4Gym/tft_set4_gym/tft_simulator.py`

## 3. Detailed Implementation Plan

### 3.1. Data Flow Simplification

The current data flow for the shop is inefficient. `step_function.py` generates a shop, passes it to `observation.py` to be converted into a vector (`shop_vector`), which is then passed to `observation_builder.py`, which largely ignores the pre-computed vector.

We will refactor this to a direct flow: `step_function.py` will place the generated shop directly onto the `player` object, and `observation_builder.py` will read it from there.

1.  **`TFTSet4Gym/tft_set4_gym/player.py`**
    -   In the `player_class.__init__` method, add a new attribute to store the raw shop data:
        ```python
        self.shop = []
        ```

2.  **`TFTSet4Gym/tft_set4_gym/step_function.py`**
    -   In the `generate_shop` method, after sampling a new shop, assign it directly to the player object. Remove the call to the legacy `generate_shop_vector`.
        ```python
        # In generate_shop(self, key, player):
        self.shops[key] = self.pool_obj.sample(player, 5)
        player.shop = self.shops[key]  # Add this line
        # self.observation_objs[key].generate_shop_vector(self.shops[key], player) # Remove this line
        ```
    -   In `generate_shops`, do the same for all players.
        ```python
        # In generate_shops(self, players):
        for player_id, player in players.items():
            if player:
                self.shops[player_id] = self.pool_obj.sample(player, 5)
                player.shop = self.shops[player_id] # Add this line
        # self.generate_shop_vectors(players) # Remove this line
        ```
    -   Delete the entire `generate_shop_vectors` method.

### 3.2. Legacy Code Removal

The `Observation` class and `tft_simulator` still contain calls and attributes related to the old observation system. These will be removed.

1.  **`TFTSet4Gym/tft_set4_gym/observation.py`**
    -   Delete the following methods entirely:
        -   `generate_shop_vector`
        -   `generate_other_player_vectors`
        -   `generate_game_comps_vector`
    -   In the `__init__` method, remove the initialization of legacy attributes:
        -   `self.shop_vector`
        -   `self.shop_mask`
        -   `self.game_comp_vector`
    -   In the `observation` method, update the call to `self.builder.build_observation` to no longer pass the shop vector.
        ```python
        # In observation(self, player_id, player, ...):
        # obs_dict = self.builder.build_observation(player_id, player, self.shop_vector) # Old
        obs_dict = self.builder.build_observation(player_id, player) # New
        ```

2.  **`TFTSet4Gym/tft_set4_gym/tft_simulator.py`**
    -   In both the `__init__` and `reset` methods, remove the call to `self.step_function.generate_shop_vectors(self.PLAYERS)`.

### 3.3. Performance Optimizations

The observation builder performs several O(n) list-to-index lookups in hot paths. These will be converted to O(1) dictionary lookups.

1.  **`TFTSet4Gym/tft_set4_gym/observation_builder.py`**
    -   At the module level, create pre-computed index mapping dictionaries for champions, items, traits, and origins.
        ```python
        # Add these at the top of the file
        COST_INDEX = {name: i - 1 for i, name in enumerate(COST.keys())}
        ITEM_BUILDS_INDEX = {name: i + len(uncraftable_items) for i, name in enumerate(item_builds.keys())}
        UNCRAFTABLE_INDEX = {name: i for i, name in enumerate(uncraftable_items)}
        TRAIT_INDEX = {name: i for i, name in enumerate(TRAIT_LIST)}
        ORIGIN_INDEX = {name: i for i, name in enumerate(ORIGIN_LIST)}
        ```
    -   Update the helper methods to use these dictionaries for O(1) lookups.
        ```python
        def _get_item_index(item_name: str) -> int:
            if item_name in ITEM_BUILDS_INDEX:
                return ITEM_BUILDS_INDEX[item_name]
            if item_name in UNCRAFTABLE_INDEX:
                return UNCRAFTABLE_INDEX[item_name]
            return 0

        def _get_trait_index(trait_name: str) -> int:
            return TRAIT_INDEX.get(trait_name, 0)

        def _get_origin_index(origin_name: str) -> int:
            return ORIGIN_INDEX.get(origin_name, 0)

        # In ObservationBuilder class:
        def _get_champion_index(self, champ: Any) -> int:
            name = getattr(champ, 'name', None)
            return COST_INDEX.get(name, -1)
        ```
    -   Refactor the `_build_shop` method to remove the `shop_vector` parameter and logic. It should read shop champions directly from `player.shop`.
        ```python
        # In ObservationBuilder class:
        def _build_shop(self, player: Any) -> tuple:
            shop_champs = np.zeros((5, 32), dtype=np.float64)
            chosen_idx = 0.0

            shop_items = getattr(player, 'shop', [])
            for i, champ_name in enumerate(shop_items):
                # Assuming shop_items contains champion names or objects with a 'name' attribute
                champ_obj = # logic to get champ object if needed
                champ_index = self._get_champion_index(champ_obj)
                if champ_index != -1:
                    shop_champs[i, 0:32] = champ_index
                if getattr(champ_obj, 'chosen', False):
                    chosen_idx = i + 1 # or appropriate index
            
            return shop_champs, chosen_idx
        ```
        *(Note to implementer: The exact logic for getting champion objects/indices from `player.shop` will depend on what `pool.sample()` returns. The goal is to derive the shop embeddings directly from `player.shop` without relying on any intermediate vectors.)*

## 4. Validation

-   All existing unit and integration tests must pass.
-   The implementation must ensure that the shape and content of the final observation tensor remain compatible with the network models that consume it.
-   Manual inspection of the generated observation tensor for a few sample states is recommended to verify correctness.
