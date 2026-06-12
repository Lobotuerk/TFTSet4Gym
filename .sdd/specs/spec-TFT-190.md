# Technical Specification: TFT-190 - Action Dimension and Masking Rework

## 1. Overview

This document outlines the necessary changes to fix the action space representation in the `TFTSet4Gym` environment and `TFTMuZeroAgent`. The current `ACTION_DIM` of `[8, 37, 10]` incorrectly limits unit movement and other actions. This will be corrected to `[7, 37, 37]`, and a robust, state-aware action masking mechanism will be implemented to ensure the agent only considers valid actions.

## 2. Background

The agent's action space is defined by a `MultiDiscrete` distribution with dimensions `[8, 37, 10]`. This leads to several issues:
-   **Limited Movement**: The third dimension, intended for destination positions, is capped at 10, making most of the board and bench unreachable for move/swap actions.
-   **No Action Masking**: The existing action mask is a placeholder and does not prevent the agent from selecting invalid actions, which harms learning efficiency.
-   **Unused Action Selector**: The first dimension has a size of 8, but only 7 action selectors are implemented.

The agreed-upon solution is to:
1.  Change `ACTION_DIM` to `[7, 37, 37]`.
2.  Implement a comprehensive, state-aware action masking system.
3.  Update the MCTS implementation to utilize this new mask.

## 3. Detailed Design

### 3.1. `TFTSet4Gym` Environment Changes

#### 3.1.1. `config.py`

-   Modify the `ACTION_DIM` constant:
    -   **From**: `ACTION_DIM = [8, 37, 10]`
    -   **To**: `ACTION_DIM = [7, 37, 37]`

#### 3.1.2. `observation_schema.py`

-   Update the size of the `valid_actions` space to match the new `ACTION_DIM`.
    -   The size will change from `54` (`8 + 37 + 10`) to `81` (`7 + 37 + 37`).
    -   The `valid_actions` definition in `ObservationSchema` should be updated accordingly.

#### 3.1.3. `observation_builder.py`

-   This is the core of the masking logic implementation. The `build_observation` function will be updated to generate a detailed, state-aware action mask.
-   The mask will be a flat boolean array of size 81.
-   The logic for populating the mask will be as follows:
    -   **Action Type Mask (7 bits):**
        -   Enable/disable action types based on the current game phase (e.g., disable combat-related actions during the shopping phase).
    -   **Parameter 1 Mask (37 bits):**
        -   For "sell" actions, only enable bits corresponding to occupied board/bench slots.
        -   For "move" actions, only enable bits corresponding to occupied board/bench slots.
        -   For "buy" actions, only enable bits corresponding to available champions in the shop.
    -   **Parameter 2 Mask (37 bits):**
        -   For "move" actions, only enable bits corresponding to valid destination slots (empty slots or swappable units).

#### 3.1.4. `step_function.py`

-   Remove any workarounds related to the old action space, such as `param2 % 37`. The `param2` value from the agent will now be in the correct `0-36` range.
-   Ensure the handlers for each action selector correctly use `param1` and `param2`.

### 3.2. `TFTMuZeroAgent` Changes

#### 3.2.1. `tft_mcts.py`

-   The `get_action_probabilities` function will be modified to apply the action mask received from the environment.
-   The `masked_softmax` and `masked_distribution` utility functions, which are currently unused, will be integrated into the action selection process.
-   The policy logits from the neural network will be masked *before* the softmax operation to prevent the agent from assigning probabilities to invalid actions.
-   The `get_precreated_moves` function will be updated. Instead of a hardcoded list of moves, it will generate moves based on the provided action mask, making it dynamic and state-aware.

## 4. Testing and Verification

-   **Unit Tests**:
    -   Add unit tests for the action mask generation in `observation_builder.py` to verify that the mask is correctly generated for various game states.
    -   Add unit tests for the MCTS action selection to ensure the mask is correctly applied.
-   **Integration Tests**:
    -   Run the agent in the environment for several steps and inspect the selected actions and corresponding masks to ensure end-to-end correctness.

## 5. Rollout Plan

1.  Implement the changes in the `TFTSet4Gym` repository on the `sdd/feature-TFT-190` branch.
2.  Implement the corresponding changes in the `TFTMuZeroAgent` repository (on a new branch).
3.  Thoroughly test the changes in both repositories.
4.  Once verified, the changes can be merged into the main branches.
