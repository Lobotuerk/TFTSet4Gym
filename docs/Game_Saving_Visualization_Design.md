# Game Saving & Visualization Design

## Overview

This document describes the format used to save and replay game trajectories from the TFT Set 4 Gym environment. Trajectory files capture the full sequence of actions taken during a game, together with the random seed that initialised the environment, so that the game can be deterministically replayed.

## File Format

Trajectories are stored as newline-delimited JSON (NDJSON) or single JSON objects with the following schema:

### Top-Level Structure

```jsonc
{
  "metadata": {
    "game_id": "tft_trajectory_20260607_123456",
    "timestamp": "2026-06-07T12:34:56.789Z",
    "num_agents": 8,
    "num_steps": 45,
    "actions_per_turn": 10
  },
  "seed": <int>,
  "steps": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `metadata` | object | Bookkeeping information (see below). |
| `seed` | int | The random seed passed to `env.reset(seed=<value>)`. |
| `steps` | array[Step] | Ordered list of steps taken in the game. |

### Step Object

```jsonc
{
  "step_index": 0,
  "actions": {
    "player_0": [1, 2, 3],
    "player_1": [0, 5, 1],
    ...
  }
}
```

| Field | Type | Description |
|---|---|---|
| `step_index` | int | 0-based step counter. |
| `actions` | dict[str, [int, int, int]] | Mapping from agent id (`"player_0"` … `"player_7"`) to the action tuple `[action_type, target_item_id, position]` that was executed at this step. |

### Replay Guarantee

Given the same `seed`, replaying the sequence of `steps` (i.e. calling `env.step()` with the exact same actions dictionaries in the same order) MUST produce identical observations at every step and an identical final state. This guarantee relies on the environment being deterministic when the Python `random` and `numpy.random` modules are seeded with the same value at reset time.

## Usage

### Recording

```python
from TFTSet4Gym.tft_set4_gym import parallel_env
from TFTSet4Gym.tft_set4_gym.game_recorder import GameRecorder

env = parallel_env()
env = GameRecorder(env)

observations, infos = env.reset(seed=42)
while env.env.agents:
    actions = {agent: env.env.action_space(agent).sample() for agent in env.env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.save_trajectory("trajectory.json")
```

### Replaying

```python
from TFTSet4Gym.tft_set4_gym import parallel_env
from TFTSet4Gym.tft_set4_gym.game_recorder import replay_trajectory

observations, infos, actions_taken = replay_trajectory("trajectory.json")
# observations contain the final observations for comparison
```

## Visualization (Future Work)

The saved trajectory format is designed to be consumed by external visualisation tools. A planned visualiser will:

1. Load a `.json` trajectory file.
2. Step through the environment using the recorded actions.
3. Render each game state (board, bench, shop, items, champion positions) as an HTML/CSS overlay.
4. Provide play/pause/step controls and a timeline scrubber.
