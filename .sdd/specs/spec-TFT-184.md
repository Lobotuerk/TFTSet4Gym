# Technical Specification: Add Item Assignment Back as Action

**Issue:** [TFT-184](mention://issue/16958819-2ddb-4b27-97d6-6932fb9eb706)

## 1. Summary

This specification outlines the re-introduction of the item assignment action, which was previously disabled for testing purposes. This change enables the agent to move items from the item bench to champions on the board or on the player's bench. This involves modifications to the observation schema, observation builder, and the step function to handle the new action.

## 2. Changes

### 2.1. `TFTSet4Gym/TFTSet4Gym/tft_set4_gym/observation_schema.py`

The observation schema will be updated to include the `item_bench`.

-   **New Field:** `item_bench`
    -   **Shape:** `(10, 24)`
    -   **Type:** `np.float64`
    -   **Description:** Represents the 10 slots of the item bench. Each slot is represented by a 24-dimensional embedding vector corresponding to the item in that slot.

### 2.2. `TFTSet4Gym/TFTSet4Gym/tft_set4_gym/observation_builder.py`

The `ObservationBuilder` will be modified to encode the `item_bench` into the observation tensor.

-   A new private method `_build_bench_items` will be created to handle the encoding of the `item_bench`.
-   This method will iterate through the player's `item_bench`, get the index for each item, and populate a `(10, 24)` numpy array with the corresponding item embeddings.
-   The `build_observation` method will call `_build_bench_items` and place the resulting array into the correct slice of the observation tensor.

### 2.3. `TFTSet4Gym/TFTSet4Gym/tft_set4_gym/step_function.py`

The `batch_2d_controller` method in the `Step_Function` class will be updated to handle the item assignment action.

-   The code block for `action_selector == 6` will be uncommented and restored.
-   This action will be parsed as `[6, item_selector, move_loc]`.
    -   `item_selector`: The index of the item on the item bench (0-9). This is derived from `param1 % 10`.
    -   `move_loc`: The target location for the item (0-36). This is derived from `param2 % 37`.
        -   `0-27`: A slot on the board.
        -   `28-36`: A slot on the bench.
-   The corresponding player methods (`move_item_to_board` or `move_item_to_bench`) will be called based on the `move_loc`.

### 2.4. `TFTSet4Gym/TFTSet4Gym/tft_set4_gym/champion_availability.py`

This file will be updated to use the new observation schema to access observation fields instead of using hardcoded indices. This makes the code more robust to future changes in the observation schema.

-   The `encode_champion_availability` function will be updated to get the `board` and `bench_champions` slices from the `ObservationSchema` registry.

## 3. Action Space

The action space will be updated to fully support 7 actions. The shape of the action space will be `[7, 37, 10]`.

-   `0`: Pass
-   `1`: Swap/Move champion
-   `2`: Buy from shop
-   `3`: Sell champion
-   `4`: Refresh shop
-   `5`: Buy EXP
-   `6`: Place item on champion

## 4. Testing

The existing test suite for `TFTSet4Gym` should be run to ensure that the changes do not introduce any regressions. It is expected that all existing tests will continue to pass. Any failures should be investigated, but pre-existing failures are out of scope for this change.
