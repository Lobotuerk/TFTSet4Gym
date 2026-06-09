"""
Centralized observation builder for TFT Set 4 Gym.
This module handles the construction of observations using the centralized schema.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from . import config
from .stats import COST
from .item_stats import uncraftable_items, item_builds


def encode_item_id(item_name: str) -> float:
    if item_name in uncraftable_items:
        return float(list(uncraftable_items).index(item_name) + 1)
    elif item_name in item_builds.keys():
        return float(list(item_builds.keys()).index(item_name) + 1 + len(uncraftable_items))
    return 0.0


class ObservationBuilder:
    """Centralized builder for creating observations from player state."""
    
    def __init__(self):
        self.current_player_schema = get_observation_schema("current_player")
        self.action_mask_schema = get_observation_schema("action_mask")
    
    def build_observation(self, player_id: str, player: Any, shop_vector: Optional[np.ndarray] = None, all_players: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Build a complete observation for a player.
        
        Args:
            player_id: The player identifier
            player: The player object containing state
            shop_vector: Optional shop vector override
            all_players: Optional dict of all players for opponent context
            
        Returns:
            Dictionary containing 'tensor' and 'action_mask' keys
        """
        # Create the tensor observation
        tensor_obs = self._build_tensor_observation(player, shop_vector, all_players=all_players)
        
        # Create action mask (simplified for now)
        action_mask = self._build_action_mask(player)
        
        return {
            "tensor": tensor_obs,
            "action_mask": action_mask
        }
    
    def _build_tensor_observation(self, player: Any, shop_vector: Optional[np.ndarray] = None, all_players: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Build the main tensor observation by reading player properties directly."""
        schema = self.current_player_schema
        observation = np.zeros(schema.total_size, dtype=np.float64)
        
        def set_field(name: str, value: Any):
            """Helper to set a field in the observation tensor."""
            try:
                field_slice = schema.get_field_slice(name)
                field_def = schema.get_field(name)
                
                if isinstance(value, (int, float, np.number)):
                    # For scalar values, broadcast to the field shape (usually (1, 4, 7))
                    val_arr = np.ones(field_def.shape) * value
                elif isinstance(value, np.ndarray):
                    # For arrays, ensure they match the expected shape
                    if value.shape != field_def.shape:
                        # Try to broadcast or reshape if possible, or just take what fits
                        val_arr = np.zeros(field_def.shape)
                        # Truncate or pad as needed
                        slices = tuple(slice(0, min(s_src, s_dst)) for s_src, s_dst in zip(value.shape, field_def.shape))
                        val_arr[slices] = value[slices]
                    else:
                        val_arr = value
                else:
                    val_arr = np.zeros(field_def.shape)
                    
                observation[field_slice] = val_arr.flatten()
            except KeyError:
                pass

        # 1. Board state - Read directly from player.board encoded_list
        try:
            board_encodings = []
            for x in range(7):
                enc = player.board[x].get_encoding()
                board_encodings.append(enc.reshape((60, 4, 1)))
            
            board_tensor = np.concatenate(board_encodings, axis=2) # (60, 4, 7)
            
            set_field("board_champions", board_tensor[0:58])
            set_field("board_stars", board_tensor[58:59])
            set_field("board_chosen", board_tensor[59:60])

            # Board items (extracted directly from champion objects)
            board_items = np.zeros((3, 4, 7))
            for x in range(7):
                for y in range(4):
                    champ = player.board[x][y]
                    if champ is not None:
                        for item_idx, item in enumerate(getattr(champ, 'items', [])[:3]):
                            board_items[item_idx, x, y] = encode_item_id(item)
            set_field("board_items", board_items)
        except Exception as e:
            print(f"Warning: Error encoding board for observation - {e}")

        # 2. Bench state (slot-wise, 9 slots)
        try:
            bench_champions = np.zeros((58, 9))
            bench_stars = np.zeros((1, 9))
            bench_items = np.zeros((3, 9))

            for slot_idx in range(9):
                unit = player.bench[slot_idx]
                if unit:
                    c_index = list(COST.keys()).index(unit.name)
                    bench_champions[c_index - 1, slot_idx] = 1.0
                    bench_stars[0, slot_idx] = getattr(unit, 'stars', 1)
                    for item_idx, item in enumerate(getattr(unit, 'items', [])[:3]):
                        bench_items[item_idx, slot_idx] = encode_item_id(item)

            set_field("bench_champions", bench_champions)
            set_field("bench_stars", bench_stars)
            set_field("bench_items", bench_items)
        except Exception as e:
            print(f"Warning: Error encoding bench for observation - {e}")

        # 3. Item bench / inventory
        try:
            item_bench = np.zeros(10)
            for i in range(len(player.item_bench)):
                item = player.item_bench[i]
                if item:
                    item_bench[i] = encode_item_id(item)
            set_field("item_bench", item_bench)
        except Exception as e:
            print(f"Warning: Error encoding item bench for observation - {e}")

        # 4. Player state (Public)
        set_field("health", player.health)
        set_field("turns_for_combat", player.turns_for_combat)
        set_field("level", player.level)
        set_field("round", player.round)
        
        # 5. Player state (Private)
        try:
            exp_to_level = player.level_costs[player.level] - player.exp
        except (IndexError, AttributeError):
            exp_to_level = 0
            
        set_field("exp_to_level", exp_to_level)
        set_field("gold", player.gold)
        
        streak = player.loss_streak if abs(player.loss_streak) > player.win_streak else player.win_streak
        set_field("streak", streak)
        
        # 6. Shop state
        if shop_vector is not None:
            set_field("shop_champions", shop_vector[0:58])
            set_field("shop_chosen", shop_vector[58:59])
        set_field("shop_locked", float(getattr(player, 'shop_locked', False)))

        # 7. Opponents (public context)
        if all_players and len(all_players) > 1:
            try:
                opponents_health = np.zeros(7)
                opponents_level = np.zeros(7)
                opponents_gold = np.zeros(7)
                opp_idx = 0
                for pid, opp in all_players.items():
                    if pid == player.player_num:
                        continue
                    if opp_idx >= 7:
                        break
                    opponents_health[opp_idx] = getattr(opp, 'health', 0)
                    opponents_level[opp_idx] = getattr(opp, 'level', 0)
                    opponents_gold[opp_idx] = getattr(opp, 'gold', 0)
                    opp_idx += 1
                set_field("opponents_health", opponents_health)
                set_field("opponents_level", opponents_level)
                set_field("opponents_gold", opponents_gold)
            except Exception as e:
                print(f"Warning: Error encoding opponents for observation - {e}")
        
        return observation
    
    def _build_action_mask(self, player: Any) -> np.ndarray:
        """Build action mask based on player state."""
        mask_size = self.action_mask_schema.total_size
        action_mask = np.ones(mask_size, dtype=np.int8)
        
        # Use player's existing shop mask if available
        if hasattr(player, 'shop_mask') and player.shop_mask is not None:
            # Map shop mask to action mask (first 5 actions typically)
            shop_size = min(len(player.shop_mask), mask_size)
            action_mask[:shop_size] = player.shop_mask[:shop_size]
        
        return action_mask
    
    def get_field_from_observation(self, observation: np.ndarray, field_name: str) -> np.ndarray:
        """Extract a specific field from a flat observation."""
        field_slice = self.current_player_schema.get_field_slice(field_name)
        field = self.current_player_schema.get_field(field_name)
        
        # Extract and reshape
        flat_data = observation[field_slice]
        return flat_data.reshape(field.shape)
    
    def set_field_in_observation(self, observation: np.ndarray, field_name: str, value: np.ndarray) -> np.ndarray:
        """Set a specific field in a flat observation."""
        field_slice = self.current_player_schema.get_field_slice(field_name)
        observation = observation.copy()  # Don't modify original
        observation[field_slice] = value.flatten()
        return observation


class ObservationManagerMixin:
    """Mixin class to add observation management capabilities to existing classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_builder = ObservationBuilder()
    
    def update_observation_field(self, field_name: str, value: Any):
        """Update a specific field in the player's observation vectors."""
        # This method can be overridden by classes that use this mixin
        # to update their internal vectors when specific fields change
        pass
    
    def get_observation_for_field(self, field_name: str) -> np.ndarray:
        """Get the current value for a specific observation field."""
        # This would need to be implemented by the mixing class
        # to return the current value from their internal state
        raise NotImplementedError("Subclass must implement get_observation_for_field")


def update_config_observation_size():
    """Update config.OBSERVATION_SIZE to match current schema."""
    schema = get_observation_schema("current_player")
    config.OBSERVATION_SIZE = schema.total_size
    return config.OBSERVATION_SIZE


# Convenience functions for backward compatibility
def build_observation(player_id: str, player: Any, shop_vector: Optional[np.ndarray] = None, all_players: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Convenience function to build observation."""
    builder = ObservationBuilder()
    return builder.build_observation(player_id, player, shop_vector, all_players=all_players)


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