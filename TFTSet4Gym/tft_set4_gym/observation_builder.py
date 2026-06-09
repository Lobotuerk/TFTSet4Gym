"""
Centralized observation builder for TFT Set 4 Gym.
This module handles the construction of observations using the centralized schema.

TFT-182: Embedding-based observation schema with learnable embeddings for
champions, items, traits, and origins. The environment provides indices and
scalars; the model's representation network applies embedding lookups.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from . import config
from .stats import COST
from .origin_class_stats import origin_class


class ObservationBuilder:
    """Centralized builder for creating observations from player state."""

    # Champion and item vocab sizes for normalization
    NUM_CHAMPIONS = 58
    NUM_ITEMS = 37
    NUM_TRAITS = 20
    NUM_ORIGINS = 10

    def __init__(self):
        self.current_player_schema = get_observation_schema("current_player")
        self.action_mask_schema = get_observation_schema("action_mask")

    def build_observation(self, player_id: str, player: Any,
                          shop_vector: Optional[np.ndarray] = None,
                          all_players: Optional[List] = None) -> Dict[str, np.ndarray]:
        """
        Build a complete observation for a player.

        Args:
            player_id: The player identifier
            player: The player object containing state
            shop_vector: Optional shop vector override
            all_players: Optional list of all players for opponent observation

        Returns:
            Dictionary containing 'tensor' and 'action_mask' keys
        """
        tensor_obs = self._build_tensor_observation(player, shop_vector, all_players)
        action_mask = self._build_action_mask(player)

        return {
            "tensor": tensor_obs,
            "action_mask": action_mask
        }

    def _build_tensor_observation(self, player: Any,
                                   shop_vector: Optional[np.ndarray] = None,
                                   all_players: Optional[List] = None) -> np.ndarray:
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
                        slices = tuple(
                            slice(0, min(s_src, s_dst))
                            for s_src, s_dst in zip(value.shape, field_def.shape)
                        )
                        val_arr[slices] = value[slices]
                    else:
                        val_arr = value
                else:
                    val_arr = np.zeros(field_def.shape)

                observation[field_slice] = val_arr.flatten()
            except KeyError:
                pass

        # 1. Board state
        try:
            board = self._encode_board(player.board)
            set_field("board", board)
        except Exception as e:
            print(f"Warning: Error encoding board for observation - {e}")

        # 2. Bench champions
        try:
            bench = self._encode_bench_champions(player.bench)
            set_field("bench_champions", bench)
        except Exception as e:
            print(f"Warning: Error encoding bench champions for observation - {e}")

        # 3. Bench items
        try:
            bench_items = self._encode_bench_items(player.item_bench)
            set_field("bench_items", bench_items)
        except Exception as e:
            print(f"Warning: Error encoding bench items for observation - {e}")

        # 4. Shop
        if shop_vector is not None:
            try:
                shop_champs = self._encode_shop_champions(shop_vector)
                set_field("shop_champions", shop_champs)
                set_field("shop_chosen", shop_vector[58])
            except Exception as e:
                print(f"Warning: Error encoding shop for observation - {e}")

        # 5. Team traits and origins
        try:
            team_traits = self._encode_team_traits(player.team_tiers)
            set_field("team_traits", team_traits)
            team_origins = self._encode_team_origins(player.team_tiers)
            set_field("team_origins", team_origins)
        except Exception as e:
            print(f"Warning: Error encoding team traits/origins - {e}")

        # 6. Player state
        try:
            player_state = self._encode_player_state(player)
            set_field("player_state", player_state)
        except Exception as e:
            print(f"Warning: Error encoding player state - {e}")

        # 7. Opponent boards and info
        if all_players is not None:
            try:
                opp_boards, opp_info = self._encode_opponents(player, all_players)
                set_field("opponent_boards", opp_boards)
                set_field("opponent_info", opp_info)
            except Exception as e:
                print(f"Warning: Error encoding opponents - {e}")

        return observation

    def _champ_index(self, champ_name: str) -> int:
        """Get the index of a champion in the champion vocabulary."""
        try:
            return list(COST.keys()).index(champ_name)
        except ValueError:
            return 0

    def _item_index(self, item_name: str) -> int:
        """Get the index of an item in the item vocabulary."""
        from .item_stats import items
        if item_name is None:
            return 0
        try:
            return list(items.keys()).index(item_name)
        except ValueError:
            return 0

    def _trait_index(self, trait_name: str) -> int:
        """Get the index of a trait in the trait vocabulary."""
        from .origin_class import team_traits
        trait_names = list(team_traits.keys())
        if trait_name not in trait_names:
            return 0
        return trait_names.index(trait_name)

    def _origin_index(self, origin_name: str) -> int:
        """Get the index of an origin in the origin vocabulary."""
        from .origin_class import team_traits
        origin_names = list(team_traits.keys())
        if origin_name not in origin_names:
            return 0
        return origin_names.index(origin_name)

    def _encode_slot(self, unit: Any) -> np.ndarray:
        """Encode a single board/bench slot as a feature vector.

        The model's representation network will apply embedding lookups to
        the champion ID, item IDs, trait IDs, and origin IDs contained here.
        """
        slot_dim = self.current_player_schema.get_field("board").shape[1]
        vector = np.zeros(slot_dim, dtype=np.float64)

        if unit is None:
            return vector

        # Champion ID (normalized to [0, 1])
        champ_idx = self._champ_index(unit.name)
        vector[0] = champ_idx / (self.NUM_CHAMPIONS - 1)

        # Item IDs (normalized to [0, 1])
        items = getattr(unit, 'items', []) or []
        for i in range(3):
            if i < len(items):
                item_idx = self._item_index(items[i])
                vector[1 + i] = item_idx / (self.NUM_ITEMS - 1)

        # Trait IDs (normalized to [0, 1])
        traits = getattr(unit, 'origin', [])
        for i, trait in enumerate(traits[:2]):
            if i < 2:
                t_idx = self._trait_index(trait)
                vector[4 + i] = t_idx / (self.NUM_TRAITS - 1)

        # Origin IDs (normalized to [0, 1])
        origins = traits[2:] if len(traits) > 2 else []
        for i, origin in enumerate(origins[:2]):
            if i < 2:
                o_idx = self._origin_index(origin)
                vector[6 + i] = o_idx / (self.NUM_ORIGINS - 1)

        # Star level (normalized to [0, 1])
        stars = getattr(unit, 'stars', 1)
        vector[8] = (stars - 1) / 2.0  # 1->0.0, 2->0.5, 3->1.0

        # Chosen flag
        chosen = getattr(unit, 'chosen', False)
        vector[9] = 1.0 if chosen else 0.0

        return vector

    def _encode_board(self, board) -> np.ndarray:
        """Encode the 4x7 board as (28, slot_dim)."""
        slot_dim = self.current_player_schema.get_field("board").shape[1]
        board_encoded = np.zeros((28, slot_dim), dtype=np.float64)

        for x in range(7):
            for y in range(4):
                unit = board[x][y] if board[x][y] is not None else None
                slot_idx = x * 4 + y
                board_encoded[slot_idx] = self._encode_slot(unit)

        return board_encoded

    def _encode_bench_champions(self, bench) -> np.ndarray:
        """Encode bench champions as (9, slot_dim)."""
        slot_dim = self.current_player_schema.get_field("bench_champions").shape[1]
        bench_encoded = np.zeros((9, slot_dim), dtype=np.float64)

        for i in range(9):
            unit = bench[i] if i < len(bench) else None
            bench_encoded[i] = self._encode_slot(unit)

        return bench_encoded

    def _encode_bench_items(self, item_bench) -> np.ndarray:
        """Encode item bench as (10, 24)."""
        bench_item_dim = self.current_player_schema.get_field("bench_items").shape[1]
        bench_items_encoded = np.zeros((10, bench_item_dim), dtype=np.float64)

        for i in range(10):
            if i < len(item_bench) and item_bench[i] is not None:
                item_idx = self._item_index(item_bench[i])
                # Store normalized item index in first dimension
                bench_items_encoded[i, 0] = item_idx / (self.NUM_ITEMS - 1)

        return bench_items_encoded

    def _encode_shop_champions(self, shop_vector: np.ndarray) -> np.ndarray:
        """Encode shop champions as (5, 32)."""
        shop_dim = self.current_player_schema.get_field("shop_champions").shape[1]
        shop_encoded = np.zeros((5, shop_dim), dtype=np.float64)

        for i in range(5):
            champ_idx = int(shop_vector[i])
            shop_encoded[i, 0] = champ_idx / (self.NUM_CHAMPIONS - 1)

        return shop_encoded

    def _encode_team_traits(self, team_tiers: Dict) -> np.ndarray:
        """Encode team trait tiers as normalized floats (20,)."""
        trait_dim = self.current_player_schema.get_field("team_traits").shape[0]
        traits_encoded = np.zeros(trait_dim, dtype=np.float64)

        from .origin_class import team_traits
        trait_names = list(team_traits.keys())

        for i, trait_name in enumerate(trait_names[:trait_dim]):
            if trait_name in team_tiers:
                tier = team_tiers[trait_name]
                # Normalize tier by max possible tier (typically 3-6)
                max_tier = 6
                traits_encoded[i] = min(tier, max_tier) / max_tier

        return traits_encoded

    def _encode_team_origins(self, team_tiers: Dict) -> np.ndarray:
        """Encode team origin tiers as normalized floats (10,)."""
        origin_dim = self.current_player_schema.get_field("team_origins").shape[0]
        origins_encoded = np.zeros(origin_dim, dtype=np.float64)

        from .origin_class import team_traits
        origin_names = list(team_traits.keys())

        for i, origin_name in enumerate(origin_names[:origin_dim]):
            if origin_name in team_tiers:
                tier = team_tiers[origin_name]
                max_tier = 6
                origins_encoded[i] = min(tier, max_tier) / max_tier

        return origins_encoded

    def _encode_player_state(self, player: Any) -> np.ndarray:
        """Encode player state as 7 scalars.

        Order: health, gold, level, round, exp_to_level, streak, turns_for_combat
        """
        state = np.zeros(7, dtype=np.float64)

        state[0] = float(getattr(player, 'health', 0))
        state[1] = float(getattr(player, 'gold', 0))
        state[2] = float(getattr(player, 'level', 1))
        state[3] = float(getattr(player, 'round', 0))

        try:
            exp_to_level = player.level_costs[player.level] - player.exp
            state[4] = float(exp_to_level)
        except (IndexError, AttributeError):
            state[4] = 0.0

        win_streak = getattr(player, 'win_streak', 0)
        loss_streak = getattr(player, 'loss_streak', 0)
        state[5] = float(max(abs(win_streak), abs(loss_streak)))

        state[6] = float(getattr(player, 'turns_for_combat', 0))

        return state

    def _encode_opponents(self, cur_player: Any,
                           all_players: List) -> Tuple[np.ndarray, np.ndarray]:
        """Encode opponent boards and info.

        Returns:
            opp_boards: (7, 28, 122) opponent board representations
            opp_info: (7, 4) opponent scalars (health, gold, level, streak)
        """
        slot_dim = self.current_player_schema.get_field("board").shape[1]
        opp_boards = np.zeros((7, 28, slot_dim), dtype=np.float64)
        opp_info = np.zeros((7, 4), dtype=np.float64)

        player_list = list(all_players.values()) if isinstance(all_players, dict) else all_players

        opp_idx = 0
        for p in player_list:
            if p is None or p is cur_player:
                continue
            if opp_idx >= 7:
                break

            # Encode opponent board
            try:
                opp_board = self._encode_board(p.board)
                opp_boards[opp_idx] = opp_board
            except Exception:
                pass

            # Encode opponent info: health, gold, level, streak
            opp_info[opp_idx, 0] = float(getattr(p, 'health', 0))
            opp_info[opp_idx, 1] = float(getattr(p, 'gold', 0))
            opp_info[opp_idx, 2] = float(getattr(p, 'level', 1))

            win_streak = getattr(p, 'win_streak', 0)
            loss_streak = getattr(p, 'loss_streak', 0)
            opp_info[opp_idx, 3] = float(max(abs(win_streak), abs(loss_streak)))

            opp_idx += 1

        return opp_boards, opp_info

    def _build_action_mask(self, player: Any) -> np.ndarray:
        """Build action mask based on player state."""
        mask_size = self.action_mask_schema.total_size
        action_mask = np.ones(mask_size, dtype=np.int8)

        if hasattr(player, 'shop_mask') and player.shop_mask is not None:
            shop_size = min(len(player.shop_mask), mask_size)
            action_mask[:shop_size] = player.shop_mask[:shop_size]

        return action_mask

    def get_field_from_observation(self, observation: np.ndarray,
                                    field_name: str) -> np.ndarray:
        """Extract a specific field from a flat observation."""
        field_slice = self.current_player_schema.get_field_slice(field_name)
        field = self.current_player_schema.get_field(field_name)

        flat_data = observation[field_slice]
        return flat_data.reshape(field.shape)

    def set_field_in_observation(self, observation: np.ndarray,
                                  field_name: str,
                                  value: np.ndarray) -> np.ndarray:
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
def build_observation(player_id: str, player: Any,
                      shop_vector: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Convenience function to build observation."""
    builder = ObservationBuilder()
    return builder.build_observation(player_id, player, shop_vector)


def get_field_slice(field_name: str) -> slice:
    """Convenience function to get field slice."""
    schema = get_observation_schema("current_player")
    return schema.get_field_slice(field_name)


def get_field_value_from_obs(observation: np.ndarray,
                              field_name: str) -> np.ndarray:
    """Extract a specific field from any observation tensor."""
    builder = ObservationBuilder()
    return builder.get_field_from_observation(observation, field_name)


def set_field_value_in_obs(observation: np.ndarray, field_name: str,
                           value: np.ndarray) -> np.ndarray:
    """Set a specific field in any observation tensor."""
    builder = ObservationBuilder()
    return builder.set_field_in_observation(observation, field_name, value)


def create_minimal_schema_for_research():
    """Create a minimal observation schema for fast training experiments."""
    from .observation_schema import ObservationField, ObservationSchema, \
        OBSERVATION_REGISTRY

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
