# SDD Specification: Observation Design (TFT-182)

## Overview

Redesign the observation system for the TFT Set 4 Gym environment to replace one-hot champion encoding with learnable embeddings and add comprehensive game state information including items, traits, origins, and opponent visibility.

## Current State

The current observation schema uses one-hot encoding for champions with 13 fields totaling 1,804 dimensions. Key limitations:
- No item information (on champions or item bench)
- No trait/origin composition data
- No champion combat stats
- No opponent board state
- No mana system
- Minimal time-stepping (single frame only)
- Flat bench encoding loses positioning

## Design Decisions

### Encoding Strategy: Learnable Embeddings

Replace one-hot encoding with learnable embedding tables for:
- **Champions**: 58 x 32 (learnable semantic relationships between champions)
- **Items**: 37 x 24 (learnable item interactions and synergies)
- **Traits**: 20 x 8 (per-champion trait embeddings)
- **Origins**: 10 x 8 (per-champion origin embeddings)

Total learnable parameters: 2,984 — manageable overhead.

### Trait/Origin Encoding: Single Float Approach

Team-level traits and origins use a single normalized float per trait/origin:
- Value = `active_tier / max_tier` in range [0, 1]
- Example: Tier 2 of max 3 = 0.67
- Preserves ordinal relationship (higher = stronger)
- Compact: 20 floats for traits, 10 for origins (vs 150 for one-hot per tier)

### Champion Per-Slot Feature Vector

Each board/bench slot (28 board positions, 9 bench positions) encodes:
- Champion embedding: 32 dims
- Item 1 embedding: 24 dims (empty = zero vector)
- Item 2 embedding: 24 dims (empty = zero vector)
- Item 3 embedding: 24 dims (empty = zero vector)
- Trait embedding: 8 dims (summed if multiple traits)
- Origin embedding: 8 dims (summed if multiple origins)
- Star level: 1 dim (normalized to [0, 1])
- Chosen flag: 1 dim (binary)

**Total per slot: 122 dims**

### Design Choices Made

1. **No raw champion stats** — Only star level and chosen flag as scalars. The embedding captures archetype; the model learns strength from star level.
2. **Per-champion items** — Each champion slot has 3 item embedding slots (max 3 items per champion in the game).
3. **Bench capacity** — 9 champion slots (BENCH_SIZE), 10 item slots (MAX_BENCH_SPACE).
4. **7 opponent boards** — Full visibility into all opponent layouts (public information in TFT).
5. **Opponent info** — health, gold, level, streak (NOT round — all players share the same round).
6. **Spatial structure preserved** — Board is (28, 122) on a 4x7 grid; position matters for TFT strategy.

## New Observation Schema

### Embedding Tables

| Table | Dimensions | Parameters |
|-------|-----------|------------|
| Champions | 58 x 32 | 1,856 |
| Items | 37 x 24 | 888 |
| Traits | 20 x 8 | 160 |
| Origins | 10 x 8 | 80 |
| **Total** | | **2,984** |

### Observation Fields

| Field | Shape | Size | Details |
|-------|-------|------|---------|
| `board` | (28, 122) | 3,416 | Per slot: champ(32) + item1(24) + item2(24) + item3(24) + traits(8) + origins(8) + star(1) + chosen(1) |
| `bench_champions` | (9, 122) | 1,098 | Same per-champion encoding as board |
| `bench_items` | (10, 24) | 240 | Item embedding per slot |
| `shop` | (5, 32) | 160 | 5 champion slots, champion embeddings only |
| `shop_chosen` | (1,) | 1 | Index of Chosen unit in shop |
| `team_traits` | (20,) | 20 | 1 normalized float per trait (tier / max_tier) |
| `team_origins` | (10,) | 10 | 1 normalized float per origin (tier / max_tier) |
| `player_state` | (7,) | 7 | health, gold, level, round, exp_to_level, streak, turns_for_combat |
| `opponent_boards` | (7, 28, 122) | 23,912 | 7 opponent boards, same structure as own board |
| `opponent_info` | (7, 4) | 28 | 7 opponents x [health, gold, level, streak] |
| `valid_actions` | (54,) | 54 | Action mask (unchanged) |
| **TOTAL** | | **28,946** | |

### Trait/Origin Float Encoding

Each trait/origin gets 1 float: `active_tier / max_tier` in [0, 1].

- Inactive: 0.0
- Tier 1 of max 3: 0.33
- Tier 2 of max 3: 0.67
- Tier 3 of max 3: 1.0

### Empty Slot Handling

Slots with no champion or item use zero vectors for all embedding fields.

## Implementation Requirements

### Files to Modify

1. **`tft_set4_gym/observation_schema.py`** — Define new observation fields and shapes
2. **`tft_set4_gym/observation_builder.py`** — Implement embedding-based observation construction
3. **`tft_set4_gym/observation.py`** — Update observation class, add opponent board generation
4. **`tft_set4_gym/config.py`** — Update OBSERVATION_SIZE constant
5. **`tft_set4_gym/champion.py`** — Expose traits/origins for embedding lookup
6. **`tft_set4_gym/player.py`** — Expose team_tiers, team_composition, item_bench for observation

### Files to Add

1. **Embedding registry/module** — Manage learnable embedding tables (can be integrated into observation_builder.py or observation_schema.py)

### Backward Compatibility

- Action mask shape and semantics remain unchanged (54 actions)
- Player state scalar order remains: health, gold, level, round, exp_to_level, streak, turns_for_combat
- The Observation class should maintain legacy attribute accessors where possible

### Key Implementation Notes

1. **Embedding initialization** — Embedding tables should be initialized with small random values (e.g., uniform[-0.1, 0.1])
2. **Zero-vector for empty slots** — When a board/bench slot is empty, all embedding fields are zeros
3. **Item zero vectors** — When a champion has fewer than 3 items, unused item slots are zero vectors
4. **Trait/origin summation** — If a champion has multiple traits/origins, their embeddings are summed before concatenation
5. **Opponent boards** — Use the same per-slot encoding as the player's own board; opponent champion IDs map through the same embedding table
