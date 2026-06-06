"""
Centralized observation builder for TFT Set 4 Gym.
This module handles the construction of observations using the centralized schema.
"""

from typing import Dict, Any, Optional
import numpy as np
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from . import config
from .stats import COST


class ObservationBuilder:
    """Centralized builder for creating observations from player state."""
    
    def __init__(self):
        self.current_player_schema = get_observation_schema("current_player")
        self.action_mask_schema = get_observation_schema("action_mask")
    
    def build_observation(self, player_id: str, player: Any, shop_vector: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Build a complete observation for a player.
        
        Args:
            player_id: The player identifier
            player: The player object containing state
            shop_vector: Optional shop vector override
            
        Returns:
            Dictionary containing 'tensor' and 'action_mask' keys
        """
        # Create the tensor observation
        tensor_obs = self._build_tensor_observation(player, shop_vector)
        
        # Create action mask (simplified for now)
        action_mask = self._build_action_mask(player)
        
        return {
            "tensor": tensor_obs,
            "action_mask": action_mask
        }
    
    def _build_tensor_observation(self, player: Any, shop_vector: Optional[np.ndarray] = None) -> np.ndarray:
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
                # Field not in schema, skip it
                pass

        # 1. Board state - Read directly from player.board encoded_list
        try:
            # board[x] is an encoded_list(4, encode_champ_object) -> get_encoding() is (60, 1, 4)
            # We want (60, 4, 7)
            board_encodings = []
            for x in range(7):
                # Reshape (60, 1, 4) to (60, 4, 1)
                enc = player.board[x].get_encoding()
                board_encodings.append(enc.reshape((60, 4, 1)))
            
            board_tensor = np.concatenate(board_encodings, axis=2) # (60, 4, 7)
            
            set_field("board_champions", board_tensor[0:58])
            set_field("board_stars", board_tensor[58:59])
            set_field("board_chosen", board_tensor[59:60])
        except Exception as e:
            print(f"Warning: Error encoding board for observation - {e}")

        # 2. Bench state (flat 1D vector, one entry per champion type)
        try:
            bench_champions = np.zeros(58)
            for unit in player.bench:
                if unit:
                    # CHAMPION_NAMES index mapping (consistent with encode_champ_object)
                    c_index = list(COST.keys()).index(unit.name)
                    bench_champions[c_index-1] += 1
            set_field("bench_champions", bench_champions)
        except Exception as e:
            print(f"Warning: Error encoding bench for observation - {e}")

        # 3. Player state (Public)
        set_field("health", player.health)
        set_field("turns_for_combat", player.turns_for_combat)
        set_field("level", player.level)
        set_field("round", player.round)
        
        # 4. Player state (Private)
        try:
            exp_to_level = player.level_costs[player.level] - player.exp
        except (IndexError, AttributeError):
            exp_to_level = 0
            
        set_field("exp_to_level", exp_to_level)
        set_field("gold", player.gold)
        
        streak = player.loss_streak if abs(player.loss_streak) > player.win_streak else player.win_streak
        set_field("streak", streak)
        
        # 5. Shop state
        if shop_vector is not None:
            # shop_vector is expected to be (62, 4, 7)
            set_field("shop_champions", shop_vector[0:58])
            set_field("shop_chosen", shop_vector[58:59])
        
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
def build_observation(player_id: str, player: Any, shop_vector: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Convenience function to build observation."""
    builder = ObservationBuilder()
    return builder.build_observation(player_id, player, shop_vector)


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