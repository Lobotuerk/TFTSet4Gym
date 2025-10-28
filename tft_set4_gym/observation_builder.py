"""
Centralized observation builder for TFT Set 4 Gym.
This module handles the construction of observations using the centralized schema.
"""

from typing import Dict, Any, Optional
import numpy as np
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from . import config


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
        """Build the main tensor observation."""
        schema = self.current_player_schema
        observation = np.zeros(schema.total_size, dtype=np.float64)
        
        # Use the player's existing vectors and map them to schema fields
        try:
            # Player public vector contains board, health, etc.
            if hasattr(player, '_player_public_vector') and player._player_public_vector is not None:
                public_flat = player._player_public_vector.flatten()
                public_size = min(len(public_flat), schema.total_size)
                observation[:public_size] = public_flat[:public_size]
            
            # Player private vector contains gold, exp, etc.
            if hasattr(player, 'player_private_vector') and player.player_private_vector is not None:
                private_flat = player.player_private_vector.flatten()
                # Append private data after public data
                start_idx = len(player._player_public_vector.flatten()) if hasattr(player, '_player_public_vector') else 0
                end_idx = min(start_idx + len(private_flat), schema.total_size)
                if start_idx < schema.total_size:
                    observation[start_idx:end_idx] = private_flat[:end_idx-start_idx]
            
            # Add shop vector if provided
            if shop_vector is not None:
                shop_flat = shop_vector.flatten()
                # Find where to place shop data (after player data)
                player_data_size = 0
                if hasattr(player, '_player_public_vector'):
                    player_data_size += player._player_public_vector.size
                if hasattr(player, 'player_private_vector'):
                    player_data_size += player.player_private_vector.size
                
                shop_start = player_data_size
                shop_end = min(shop_start + len(shop_flat), schema.total_size)
                if shop_start < schema.total_size:
                    observation[shop_start:shop_end] = shop_flat[:shop_end-shop_start]
                    
        except Exception as e:
            print(f"Warning: Error building observation - {e}")
            # Return zeros if there's an issue
            pass
        
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