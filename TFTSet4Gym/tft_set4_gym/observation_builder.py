"""
Centralized observation builder for TFT Set 4 Gym.
This module handles the construction of observations using the centralized schema.
Implements the embedding-based observation system per TFT-182 spec.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from . import config
from .stats import COST
from .item_stats import item_builds, uncraftable_items, basic_items
from .origin_class_stats import tiers as trait_tiers

# Trait and origin lists in the order used by the schema
TRAIT_LIST = [
    'cultist', 'divine', 'dusk', 'elderwood', 'enlightened',
    'exile', 'ninja', 'spirit', 'the_boss', 'warlord',
    'adept', 'assassin', 'brawler', 'dazzler', 'duelist',
    'emperor', 'hunter', 'keeper', 'mage', 'mystic',
]

ORIGIN_LIST = [
    'shade', 'sharpshooter', 'vanguard', 'fortune', 'moonlight',
    'tormented', 'cultist', 'divine', 'dusk', 'elderwood',
]

# Pre-computed index lookup dictionaries for O(1) access
COST_INDEX = {name: i - 1 for i, name in enumerate(COST.keys())}
ITEM_BUILDS_INDEX = {name: i + len(uncraftable_items) for i, name in enumerate(item_builds.keys())}
UNCRAFTABLE_INDEX = {name: i for i, name in enumerate(uncraftable_items)}
TRAIT_INDEX = {name: i for i, name in enumerate(TRAIT_LIST)}
ORIGIN_INDEX = {name: i for i, name in enumerate(ORIGIN_LIST)}


def _get_item_index(item_name: str) -> int:
    """Get the embedding index for an item name using O(1) dict lookup."""
    if item_name in ITEM_BUILDS_INDEX:
        return ITEM_BUILDS_INDEX[item_name]
    if item_name in UNCRAFTABLE_INDEX:
        return UNCRAFTABLE_INDEX[item_name]
    return 0


def _get_trait_index(trait_name: str) -> int:
    """Get the embedding index for a trait name using O(1) dict lookup."""
    return TRAIT_INDEX.get(trait_name, 0)


def _get_origin_index(origin_name: str) -> int:
    """Get the embedding index for an origin name using O(1) dict lookup."""
    return ORIGIN_INDEX.get(origin_name, 0)


class ObservationBuilder:
    """Centralized builder for creating observations from player state."""

    def __init__(self):
        self.current_player_schema = get_observation_schema("current_player")
        self.action_mask_schema = get_observation_schema("action_mask")

        # Embedding table sizes (for model registration, not used by builder itself)
        self.embedding_dims = {
            'champion_embeddings': (58, 32),
            'item_embeddings': (37, 24),
            'trait_embeddings': (20, 8),
            'origin_embeddings': (10, 8),
        }

    def build_observation(self, player_id: str, player: Any) -> Dict[str, np.ndarray]:
        """
        Build a complete observation for a player.

        Args:
            player_id: The player identifier
            player: The player object containing state

        Returns:
            Dictionary containing 'tensor' and 'action_mask' keys
        """
        tensor_obs = self._build_tensor_observation(player)
        action_mask = self._build_action_mask(player)

        return {
            "tensor": tensor_obs,
            "action_mask": action_mask
        }

    def _build_per_champion_slot(self, champ: Any) -> np.ndarray:
        """Build a 122-dim per-champion slot vector.

        Per slot: champion(32) + item1(24) + item2(24) + item3(24) + traits(8) + origins(8) + star(1) + chosen(1)
        """
        if champ is None:
            return np.zeros(122, dtype=np.float64)

        champ_idx = self._get_champion_index(champ)
        if champ_idx < 0 or champ_idx >= 58:
            return np.zeros(122, dtype=np.float64)

        slot = np.zeros(122, dtype=np.float64)

        # Champion embedding: indices 0:32
        slot[0:32] = champ_idx

        # Item embeddings: indices 32:104 (3 items x 24)
        items = getattr(champ, 'items', [])
        for i in range(3):
            if i < len(items) and items[i]:
                item_idx = _get_item_index(items[i])
                offset = 32 + (i * 24)
                slot[offset:offset+24] = item_idx

        # Trait embedding: indices 104:112
        traits = getattr(champ, 'origin', [])
        if traits:
            trait_idx = _get_trait_index(traits[0])
            slot[104:112] = trait_idx

        # Origin embedding: indices 112:120
        if len(traits) > 1:
            origin_idx = _get_origin_index(traits[1])
            slot[112:120] = origin_idx

        # Star level: index 120
        slot[120] = float(getattr(champ, 'stars', 1))

        # Chosen flag: index 121
        slot[121] = 1.0 if getattr(champ, 'chosen', False) else 0.0

        return slot

    def _get_champion_index(self, champ: Any) -> int:
        """Get 0-based champion index for embedding lookup (champion object)."""
        try:
            name = getattr(champ, 'name', None)
            if name is None:
                return -1
            return COST_INDEX.get(name, -1)
        except (ValueError, AttributeError):
            return -1

    def _build_board(self, player: Any) -> np.ndarray:
        """Build the (28, 122) board field from player.board.

        player.board is a list of 7 encoded_list(4, ...), flattened to 28 slots.
        """
        board = np.zeros((28, 122), dtype=np.float64)

        try:
            for x in range(7):
                for y in range(4):
                    slot_idx = x * 4 + y
                    champ = player.board[x][y]
                    board[slot_idx] = self._build_per_champion_slot(champ)
        except Exception as e:
            print(f"Warning: Error encoding board for observation - {e}")

        return board

    def _build_bench_champions(self, player: Any) -> np.ndarray:
        """Build the (9, 122) bench_champions field."""
        bench = np.zeros((9, 122), dtype=np.float64)

        try:
            for i in range(min(len(player.bench), 9)):
                champ = player.bench[i]
                bench[i] = self._build_per_champion_slot(champ)
        except Exception as e:
            print(f"Warning: Error encoding bench for observation - {e}")

        return bench

    def _build_bench_items(self, player: Any) -> np.ndarray:
        """Build the (10, 24) bench_items field from player.item_bench."""
        bench_items = np.zeros((10, 24), dtype=np.float64)

        try:
            for i in range(min(len(player.item_bench), 10)):
                item_name = player.item_bench[i]
                if item_name:
                    item_idx = _get_item_index(item_name)
                    bench_items[i] = item_idx
        except Exception as e:
            print(f"Warning: Error encoding bench items for observation - {e}")

        return bench_items

    def _build_shop(self, player: Any) -> tuple:
        """Build the shop field: (5, 32) + (1,) chosen index.

        Shop champion embeddings are derived directly from player.shop.
        """
        shop = np.zeros((5, 32), dtype=np.float64)
        chosen_idx = 0.0

        shop_items = getattr(player, 'shop', [])
        for i in range(min(len(shop_items), 5)):
            champ_name = shop_items[i]
            if champ_name == " ":
                continue

            chosen = False
            if champ_name.endswith("_c"):
                chosen = True
                parts = champ_name.split('_')
                if len(parts) >= 3 and parts[-1] == 'c':
                    champ_name = parts[0]
                else:
                    champ_name = champ_name[:-2]

            champ_idx = COST_INDEX.get(champ_name, -1)
            if 0 <= champ_idx < 58:
                shop[i, 0:32] = champ_idx
            if chosen:
                chosen_idx = float(i + 1)

        return shop, chosen_idx

    def _build_team_traits(self, player: Any) -> np.ndarray:
        """Build the (20,) team_traits field.

        Each trait gets 1 float: active_tier / max_tier in range [0, 1].
        """
        team_traits = np.zeros(20, dtype=np.float64)

        try:
            team_tiers = getattr(player, 'team_tiers', {})
            team_composition = getattr(player, 'team_composition', {})

            for i, trait_name in enumerate(TRAIT_LIST):
                tier = team_tiers.get(trait_name, 0)
                if tier > 0 and trait_name in trait_tiers:
                    max_tier = len(trait_tiers[trait_name])
                    team_traits[i] = tier / max_tier
                elif trait_name not in trait_tiers:
                    # Traits without tier data (fortune, moonlight, tormented)
                    # Use count as normalized value (max expected ~8)
                    count = team_composition.get(trait_name, 0)
                    team_traits[i] = min(count / 8.0, 1.0)
        except Exception as e:
            print(f"Warning: Error encoding team traits - {e}")

        return team_traits

    def _build_team_origins(self, player: Any) -> np.ndarray:
        """Build the (10,) team_origins field.

        Same as team_traits per spec (they encode the same activation data).
        """
        team_origins = np.zeros(10, dtype=np.float64)

        try:
            team_tiers = getattr(player, 'team_tiers', {})

            for i, origin_name in enumerate(ORIGIN_LIST):
                tier = team_tiers.get(origin_name, 0)
                if tier > 0 and origin_name in trait_tiers:
                    max_tier = len(trait_tiers[origin_name])
                    team_origins[i] = tier / max_tier
                elif origin_name not in trait_tiers:
                    count = getattr(player, 'team_composition', {}).get(origin_name, 0)
                    team_origins[i] = min(count / 8.0, 1.0)
        except Exception as e:
            print(f"Warning: Error encoding team origins - {e}")

        return team_origins

    def _build_player_state(self, player: Any) -> np.ndarray:
        """Build the (7,) player_state field.

        [health, gold, level, round, exp_to_level, streak, turns_for_combat]
        """
        state = np.zeros(7, dtype=np.float64)

        state[0] = float(getattr(player, 'health', 0))
        state[1] = float(getattr(player, 'gold', 0))
        state[2] = float(getattr(player, 'level', 1))
        state[3] = float(getattr(player, 'round', 0))

        try:
            exp_to_level = player.level_costs[player.level] - player.exp
        except (IndexError, AttributeError):
            exp_to_level = 0
        state[4] = float(exp_to_level)

        win_streak = getattr(player, 'win_streak', 0)
        loss_streak = getattr(player, 'loss_streak', 0)
        state[5] = float(loss_streak if abs(loss_streak) > win_streak else win_streak)

        state[6] = float(getattr(player, 'turns_for_combat', 0))

        return state

    def _build_opponent_boards(self, players: List[Any], cur_player_id: str) -> np.ndarray:
        """Build the (7, 28, 122) opponent_boards field.

        For each opponent (excluding current player), build their board state.
        """
        opponent_boards = np.zeros((7, 28, 122), dtype=np.float64)

        try:
            opp_idx = 0
            for pid, other_player in players.items():
                if pid == cur_player_id:
                    continue
                if opp_idx >= 7:
                    break

                # Build opponent board the same way as own board
                opp_board = np.zeros((28, 122), dtype=np.float64)
                try:
                    for x in range(7):
                        for y in range(4):
                            slot_idx = x * 4 + y
                            champ = other_player.board[x][y]
                            opp_board[slot_idx] = self._build_per_champion_slot(champ)
                except Exception:
                    pass

                opponent_boards[opp_idx] = opp_board
                opp_idx += 1
        except Exception as e:
            print(f"Warning: Error encoding opponent boards - {e}")

        return opponent_boards

    def _build_opponent_info(self, players: List[Any], cur_player_id: str) -> np.ndarray:
        """Build the (7, 4) opponent_info field.

        For each opponent: [health, gold, level, streak]
        """
        opponent_info = np.zeros((7, 4), dtype=np.float64)

        try:
            opp_idx = 0
            for pid, other_player in players.items():
                if pid == cur_player_id:
                    continue
                if opp_idx >= 7:
                    break

                opponent_info[opp_idx, 0] = float(getattr(other_player, 'health', 0))
                opponent_info[opp_idx, 1] = float(getattr(other_player, 'gold', 0))
                opponent_info[opp_idx, 2] = float(getattr(other_player, 'level', 1))

                win_streak = getattr(other_player, 'win_streak', 0)
                loss_streak = getattr(other_player, 'loss_streak', 0)
                opponent_info[opp_idx, 3] = float(loss_streak if abs(loss_streak) > win_streak else win_streak)

                opp_idx += 1
        except Exception as e:
            print(f"Warning: Error encoding opponent info - {e}")

        return opponent_info

    def _build_tensor_observation(self, player: Any) -> np.ndarray:
        """Build the main tensor observation by reading player properties directly."""
        schema = self.current_player_schema
        observation = np.zeros(schema.total_size, dtype=np.float64)

        def set_field(name: str, value: Any):
            """Helper to set a field in the observation tensor."""
            try:
                field_slice = schema.get_field_slice(name)
                field_def = schema.get_field(name)

                if isinstance(value, (int, float, np.number)):
                    val_arr = np.ones(field_def.shape) * value
                elif isinstance(value, np.ndarray):
                    if value.shape != field_def.shape:
                        val_arr = np.zeros(field_def.shape)
                        slices = tuple(slice(0, min(s_src, s_dst)) for s_src, s_dst in zip(value.shape, field_def.shape))
                        val_arr[slices] = value[slices]
                    else:
                        val_arr = value
                else:
                    val_arr = np.zeros(field_def.shape)

                observation[field_slice] = val_arr.flatten()
            except KeyError:
                pass

        # 1. Board state
        set_field("board", self._build_board(player))

        # 2. Bench champions
        set_field("bench_champions", self._build_bench_champions(player))

        # 3. Bench items
        set_field("bench_items", self._build_bench_items(player))

        # 4. Shop
        shop, chosen_idx = self._build_shop(player)
        set_field("shop", shop)
        set_field("shop_chosen", chosen_idx)

        # 5. Team traits
        set_field("team_traits", self._build_team_traits(player))

        # 6. Team origins
        set_field("team_origins", self._build_team_origins(player))

        # 7. Player state
        set_field("player_state", self._build_player_state(player))

        # 8-9. Opponent data - requires access to all players
        # These are set to zero by default; the caller should fill them via
        # set_field_value_in_obs after build_observation returns.
        # The observation tensor already has zero-filled opponent_boards and opponent_info.

        return observation

    def build_full_observation(self, player_id: str, player: Any, players: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Build a complete observation including opponent data.

        Args:
            player_id: The current player identifier
            player: The current player object
            players: Dict of all player objects keyed by player_id
        """
        tensor_obs = self._build_tensor_observation(player)

        # Fill in opponent data
        schema = self.current_player_schema
        observation = tensor_obs.copy()

        opp_board = self._build_opponent_boards(players, player_id)
        opp_board_slice = schema.get_field_slice("opponent_boards")
        observation[opp_board_slice] = opp_board.flatten()

        opp_info = self._build_opponent_info(players, player_id)
        opp_info_slice = schema.get_field_slice("opponent_info")
        observation[opp_info_slice] = opp_info.flatten()

        action_mask = self._build_action_mask(player)

        return {
            "tensor": observation,
            "action_mask": action_mask
        }

    def _build_action_mask(self, player: Any) -> np.ndarray:
        """Build action mask based on player state."""
        mask_size = self.action_mask_schema.total_size
        action_mask = np.ones(mask_size, dtype=np.int8)

        if hasattr(player, 'shop_mask') and player.shop_mask is not None:
            shop_size = min(len(player.shop_mask), mask_size)
            action_mask[:shop_size] = player.shop_mask[:shop_size]

        return action_mask

    def get_field_from_observation(self, observation: np.ndarray, field_name: str) -> np.ndarray:
        """Extract a specific field from a flat observation."""
        field_slice = self.current_player_schema.get_field_slice(field_name)
        field = self.current_player_schema.get_field(field_name)

        flat_data = observation[field_slice]
        return flat_data.reshape(field.shape)

    def set_field_in_observation(self, observation: np.ndarray, field_name: str, value: np.ndarray) -> np.ndarray:
        """Set a specific field in a flat observation."""
        field_slice = self.current_player_schema.get_field_slice(field_name)
        observation = observation.copy()
        observation[field_slice] = value.flatten()
        return observation


class ObservationManagerMixin:
    """Mixin class to add observation management capabilities to existing classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_builder = ObservationBuilder()

    def update_observation_field(self, field_name: str, value: Any):
        """Update a specific field in the player's observation vectors."""
        pass

    def get_observation_for_field(self, field_name: str) -> np.ndarray:
        """Get the current value for a specific observation field."""
        raise NotImplementedError("Subclass must implement get_observation_for_field")


def update_config_observation_size():
    """Update config.OBSERVATION_SIZE to match current schema."""
    schema = get_observation_schema("current_player")
    config.OBSERVATION_SIZE = schema.total_size
    return config.OBSERVATION_SIZE


# Convenience functions for backward compatibility
def build_observation(player_id: str, player: Any) -> Dict[str, np.ndarray]:
    """Convenience function to build observation."""
    builder = ObservationBuilder()
    return builder.build_observation(player_id, player)


def get_field_slice(field_name: str) -> slice:
    """Convenience function to get field slice."""
    schema = get_observation_schema("current_player")
    return schema.get_field_slice(field_name)


def get_field_value_from_obs(observation: np.ndarray, field_name: str) -> np.ndarray:
    """Extract a specific field from any observation tensor."""
    builder = ObservationBuilder()
    return builder.get_field_from_observation(observation, field_name)


def set_field_value_in_obs(observation: np.ndarray, field_name: str, value: np.ndarray) -> np.ndarray:
    """Set a specific field in any observation tensor."""
    builder = ObservationBuilder()
    return builder.set_field_in_observation(observation, field_name, value)


def create_minimal_schema_for_research():
    """Create a minimal observation schema for fast training experiments."""
    from .observation_schema import ObservationField, ObservationSchema, OBSERVATION_REGISTRY

    minimal_fields = [
        ObservationField("health", (1,), np.dtype('float64'), "Player health"),
        ObservationField("gold", (1,), np.dtype('float64'), "Current gold"),
        ObservationField("level", (1,), np.dtype('float64'), "Player level"),
        ObservationField("board_units", (28,), np.dtype('float64'), "Units on board"),
        ObservationField("shop_units", (5,), np.dtype('float64'), "Units in shop"),
        ObservationField("round_info", (2,), np.dtype('float64'), "Round and phase"),
    ]

    minimal_schema = ObservationSchema(minimal_fields)
    OBSERVATION_REGISTRY.register_schema("minimal", minimal_schema)

    return minimal_schema
