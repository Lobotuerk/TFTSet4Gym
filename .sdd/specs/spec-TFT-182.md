# Specification: Observation Design (TFT-182)

## Overview

Redesign the observation system for the TFT Set 4 Gym environment to use learnable embedding tables for champions, items, traits, and origins. Replace the current one-hot encoding with dense embeddings and add comprehensive game state information including items, trait/origin activations, and opponent board visibility.

## Current State

The current observation schema (`observation_schema.py`) uses one-hot encoding with 1,804 total dimensions:
- Board champions: (58, 4, 7) = 1,624 dims — one-hot champion ID on 4x7 grid
- Board stars: (1, 4, 7) = 28 dims
- Board chosen: (1, 4, 7) = 28 dims
- Bench champions: (58,) = 58 dims — flat champion counts
- Player state: 7 scalars (health, turns_for_combat, level, round, exp_to_level, gold, streak)
- Shop: (58,) + 1 = 59 dims — champion counts + chosen index
- Action mask: (54,) = 54 dims

## Design Decisions

### Embedding Dimensions

| Table | Dimension | Rationale |
|-------|-----------|-----------|
| Champions | 32 | Dense representation of 58 champions |
| Items | 24 | Dense representation of ~37 item types |
| Traits | 8 | Per-champion trait identity |
| Origins | 8 | Per-champion origin identity |

### Trait/Origin Encoding

Traits and origins use a **single normalized float per type** for team-level encoding:
- Value = `active_tier / max_tier` in range [0.0, 1.0]
- 0.0 = trait/origin not active
- 1.0 = trait/origin at maximum tier
- Preserves ordinal relationship (higher float = stronger tier)
- Allows the model to learn "close to next tier" signals

### Per-Champion Item Slots

Each champion (board and bench) carries up to **3 item embedding slots**. Empty slots use zero vectors.

### Opponent Visibility

All 7 opponent boards are included in the observation. Each opponent also exposes 4 public scalars.

## New Observation Schema

### Embedding Tables

| Table | Size | Parameters |
|-------|------|------------|
| Champions | 58 x 32 | 1,856 |
| Items | 37 x 24 | 888 |
| Traits | 20 x 8 | 160 |
| Origins | 10 x 8 | 80 |
| **Total** | | **2,984** |

### Observation Fields

| Field | Shape | Size | Description |
|-------|-------|------|-------------|
| **board** | (28, 122) | 3,416 | 28 board slots (4x7 grid), each with per-champion encoding |
| **bench_champions** | (9, 122) | 1,098 | 9 bench champion slots, same per-champion encoding |
| **bench_items** | (10, 24) | 240 | 10 item bench slots, item embeddings |
| **shop** | (5, 32) + (1,) | 161 | 5 shop champion slots (32-dim embeddings) + chosen index scalar |
| **team_traits** | (20,) | 20 | 1 normalized float per trait (tier/max_tier) |
| **team_origins** | (10,) | 10 | 1 normalized float per origin (tier/max_tier) |
| **player_state** | (7,) | 7 | [health, gold, level, round, exp_to_level, streak, turns_for_combat] |
| **opponent_boards** | (7, 28, 122) | 23,912 | 7 opponent boards, same structure as own board |
| **opponent_info** | (7, 4) | 28 | 7 opponents x [health, gold, level, streak] |
| **action_mask** | (54,) | 54 | Valid actions (unchanged) |
| **TOTAL** | | **28,946** | |

### Per-Champion Slot Encoding (122 dims)

Each board and bench champion slot contains:

| Component | Dim | Description |
|-----------|-----|-------------|
| champion_embedding | 32 | Learnable embedding from champion ID |
| item1_embedding | 24 | First item on champion (zero if empty) |
| item2_embedding | 24 | Second item on champion (zero if empty) |
| item3_embedding | 24 | Third item on champion (zero if empty) |
| trait_embedding | 8 | Learnable trait identity embedding |
| origin_embedding | 8 | Learnable origin identity embedding |
| star_level | 1 | Champion star level (1, 2, or 3) |
| chosen_flag | 1 | Binary: 1 if chosen unit, 0 otherwise |
| **Total** | **122** | |

### Trait/Origin Lists

**Traits (20):** cultist, divine, dusk, elderwood, enlightened, exile, ninja, spirit, the_boss, warlord, adept, assassin, brawler, dazzler, duelist, emperor, hunter, keeper, mage, mystic, shade, sharpshooter, fortune, moonlight, tormented

Note: The `amounts` dict in `origin_class.py` defines 26 trait keys. The schema uses the 20 traits that have tier definitions in `origin_class_stats.tiers`. Traits without tier data (fortune, moonlight, tormented) are included with max_tier=1 (value = active count).

**Origins (10):** The codebase uses traits and origins interchangeably for the `origin_class` system. Each champion has 1-3 traits from the above list. The `team_origins` field mirrors `team_traits` — they encode the same activation data.

### Opponent Info Fields

For each of the 7 opponents:
1. **health** — Current player health
2. **gold** — Current gold amount
3. **level** — Current player level
4. **streak** — Current win/loss streak (signed integer)

**Note:** Round is excluded from opponent info per design review. Only the current player's round is included in `player_state`.

### Empty Slot Handling

- Board slots without a champion: all 122 dims are zero
- Bench champion slots without a champion: all 122 dims are zero
- Item slots without an item: 24-dim zero vector
- Item slots on champions without items: 3 x 24-dim zero vectors

## Implementation Requirements

### 1. Schema Registry Update (`observation_schema.py`)

Replace the `current_player` schema fields with the new field definitions. Update `ObservationSchemaRegistry._setup_default_schemas()` to register the new fields with correct shapes and dtypes.

### 2. Observation Builder Update (`observation_builder.py`)

Rewrite `ObservationBuilder._build_tensor_observation()` to:
- Build per-champion slot vectors with champion embedding lookup, item embeddings, trait/origin embeddings, star level, and chosen flag
- Construct the board field from `player.board` (7x4 grid, flattened to 28 slots)
- Construct the bench_champions field from `player.bench` (9 slots)
- Construct the bench_items field from `player.item_bench` (10 slots)
- Construct the shop field from the shop vector (5 champion embeddings + chosen index)
- Compute team_traits as normalized float vector (20 dims)
- Compute team_origins as normalized float vector (10 dims)
- Encode player_state scalars
- Build opponent_boards from all other players' board states
- Build opponent_info from all other players' public state

### 3. Embedding Layer Registration

The observation builder must maintain or reference learnable embedding layers:
- `champion_embeddings`: nn.Embedding(58, 32)
- `item_embeddings`: nn.Embedding(37, 24)
- `trait_embeddings`: nn.Embedding(20, 8)
- `origin_embeddings`: nn.Embedding(10, 8)

These are consumed by the neural network, not the observation builder itself (which produces flat tensors). The embedding layer registration is a contract with the model definition.

### 4. Config Update (`config.py`)

`config.OBSERVATION_SIZE` will be updated dynamically by `update_observation_size_in_config()` to reflect the new total of 28,946 dimensions.

### 5. Backward Compatibility

The `Observation` class (`observation.py`) must maintain its existing interface:
- `observation(player_id, player, action_vector)` returns `{"tensor": ..., "action_mask": ...}`
- Legacy attributes (`shop_vector`, `game_comp_vector`, `other_player_observations`) are preserved for compatibility
- Time-stepping behavior (`cur_player_observations` deque) is unchanged

## Dimension Verification

```
board:              28 x 122   = 3,416
bench_champions:     9 x 122   = 1,098
bench_items:        10 x  24   =   240
shop:                5 x  32   =   160 + 1   = 161
team_traits:                    20           =    20
team_origins:                     10           =    10
player_state:                      7           =     7
opponent_boards:    7 x 28 x 122 = 23,912
opponent_info:      7 x   4    =    28
action_mask:                54           =    54
                                            -------
Total:                             28,946
```

## Migration Notes

- The existing `encode_champ_object` and `encode_item_object` functions in `player.py` produce numpy arrays used by the old schema. These are replaced by the embedding-based builder.
- The `encoded_list` class in `player.py` is used for board/bench/item storage and should be preserved. The encoding functions are only used by the legacy observation path.
- Champion names are indexed via `list(COST.keys())` — champion IDs are 0-based indices into this list (minus 1 for 1-based indexing in the old system). The new system uses the same 0-based indexing for embedding table lookups.
- Item names map to embedding indices: basic items (9) + combined items (37 total in `item_builds.keys()`) = the item set. The exact item-to-index mapping should match `item_builds.keys()` for combined items plus `basic_items` for components.
