"""
Centralized observation schema registry for TFT Set 4 Gym.
This module provides a single source of truth for all observation formats,
making it easy to modify observation structure without touching multiple files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from . import config


@dataclass
class ObservationField:
    """Defines a single field in the observation vector."""
    name: str
    shape: Tuple[int, ...]
    dtype: Union[type, np.dtype]
    description: str
    start_idx: Optional[int] = field(default=None)
    end_idx: Optional[int] = field(default=None)


@dataclass 
class ObservationSchema:
    """Defines the complete observation schema with field mappings."""
    fields: List[ObservationField] = field(default_factory=list)
    _field_map: Dict[str, ObservationField] = field(default_factory=dict, init=False)
    _total_size: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Calculate indices and build field map."""
        current_idx = 0
        
        for field_def in self.fields:
            field_def.start_idx = int(current_idx)
            field_size = int(np.prod(field_def.shape))
            field_def.end_idx = int(current_idx + field_size)
            current_idx += field_size
            
            self._field_map[field_def.name] = field_def
        
        self._total_size = int(current_idx)
    
    @property
    def total_size(self) -> int:
        """Get total observation size."""
        return self._total_size
    
    def get_field(self, name: str) -> ObservationField:
        """Get field definition by name."""
        if name not in self._field_map:
            raise KeyError(f"Field '{name}' not found in observation schema")
        return self._field_map[name]
    
    def get_field_slice(self, name: str) -> slice:
        """Get numpy slice for a field."""
        field = self.get_field(name)
        return slice(field.start_idx, field.end_idx)
    
    def get_field_indices(self, name: str) -> Tuple[int, int]:
        """Get start and end indices for a field."""
        field = self.get_field(name)
        if field.start_idx is None or field.end_idx is None:
            raise ValueError(f"Field '{name}' indices not calculated")
        return field.start_idx, field.end_idx
    
    def validate_observation(self, observation: np.ndarray) -> bool:
        """Validate that observation matches schema."""
        if observation.size != self._total_size:
            return False
        return True


class ObservationSchemaRegistry:
    """Central registry for all observation schemas."""
    
    def __init__(self):
        self._schemas = {}
        self._setup_default_schemas()
    
    def _setup_default_schemas(self):
        """Setup the default TFT observation schemas."""
        
        # New observation schema with learnable embeddings.
        # Total dimensions: 28,946
        #
        # Embedding tables:
        #   Champions: 58 x 32  = 1,856 params
        #   Items:     37 x 24  =   888 params
        #   Traits:    20 x 8   =   160 params
        #   Origins:   10 x 8   =    80 params
        #   Total:                 2,984 params
        #
        # Per-slot encoding (board + bench champions):
        #   champion(32) + item1(24) + item2(24) + item3(24) +
        #   traits(8) + origins(8) + star(1) + chosen(1) = 122 dims
        #
        # Trait/Origin encoding: single normalized float per trait/origin
        #   value = active_tier / max_tier in [0, 1]

        current_player_fields = [
            # Board representation: 28 slots x 122-dim per-slot encoding
            ObservationField("board", (28, 122), np.dtype('float64'),
                             "Board slots: champ(32) + items(3x24) + traits(8) + origins(8) + star(1) + chosen(1)"),

            # Bench champions: 9 slots x 122-dim per-slot encoding
            ObservationField("bench_champions", (9, 122), np.dtype('float64'),
                             "Bench champions: same per-slot encoding as board"),

            # Bench items: 10 slots x 24-dim item embedding
            ObservationField("bench_items", (10, 24), np.dtype('float64'),
                             "Item bench: item embedding per slot"),

            # Shop: 5 champion slots x 32-dim champion embedding
            ObservationField("shop", (5, 32), np.dtype('float64'),
                             "Shop champions: champion embedding per slot"),

            # Shop chosen index
            ObservationField("shop_chosen", (1,), np.dtype('float64'),
                             "Index of Chosen unit in shop"),

            # Team traits: 1 normalized float per trait (tier / max_tier)
            ObservationField("team_traits", (20,), np.dtype('float64'),
                             "Team trait tiers: normalized float per trait"),

            # Team origins: 1 normalized float per origin (tier / max_tier)
            ObservationField("team_origins", (10,), np.dtype('float64'),
                             "Team origin tiers: normalized float per origin"),

            # Player state (scalars, flattened)
            ObservationField("player_state", (7,), np.dtype('float64'),
                             "health, gold, level, round, exp_to_level, streak, turns_for_combat"),

            # Opponent boards: 7 opponents x 28 slots x 122 dims
            ObservationField("opponent_boards", (7, 28, 122), np.dtype('float64'),
                             "Opponent board states: same per-slot encoding as own board"),

            # Opponent info: 7 opponents x 4 scalars
            ObservationField("opponent_info", (7, 4), np.dtype('float64'),
                             "Opponent state: health, gold, level, streak"),
        ]

        self.register_schema("current_player", ObservationSchema(current_player_fields))
        
        # Action mask schema
        action_mask_fields = [
            ObservationField("valid_actions", (54,), np.dtype('int8'), "Mask for valid actions")
        ]
        
        self.register_schema("action_mask", ObservationSchema(action_mask_fields))
    
    def register_schema(self, name: str, schema: ObservationSchema):
        """Register a new observation schema."""
        self._schemas[name] = schema
    
    def get_schema(self, name: str) -> ObservationSchema:
        """Get a registered schema by name."""
        if name not in self._schemas:
            raise KeyError(f"Schema '{name}' not found")
        return self._schemas[name]
    
    def get_gymnasium_space_config(self, schema_name: str) -> Dict[str, Any]:
        """Get Gymnasium space configuration for a schema."""
        schema = self.get_schema(schema_name)
        
        if schema_name == "current_player":
            return {
                "low": -1.0,
                "high": 1.0,
                "shape": (schema.total_size,),
                "dtype": np.float64
            }
        elif schema_name == "action_mask":
            return {
                "low": 0,
                "high": 1,
                "shape": (schema.total_size,),
                "dtype": np.int8
            }
        else:
            raise ValueError(f"No Gymnasium config defined for schema '{schema_name}'")
    
    def get_combined_schema_config(self) -> Dict[str, Dict[str, Any]]:
        """Get combined schema config for Dict observation space."""
        return {
            "tensor": self.get_gymnasium_space_config("current_player"),
            "action_mask": self.get_gymnasium_space_config("action_mask")
        }


# Global registry instance
OBSERVATION_REGISTRY = ObservationSchemaRegistry()


def get_observation_schema(name: str = "current_player") -> ObservationSchema:
    """Convenience function to get observation schema."""
    return OBSERVATION_REGISTRY.get_schema(name)


def get_field_slice(field_name: str, schema_name: str = "current_player") -> slice:
    """Convenience function to get field slice."""
    return OBSERVATION_REGISTRY.get_schema(schema_name).get_field_slice(field_name)


def get_field_indices(field_name: str, schema_name: str = "current_player") -> Tuple[int, int]:
    """Convenience function to get field indices."""
    return OBSERVATION_REGISTRY.get_schema(schema_name).get_field_indices(field_name)


def update_observation_size_in_config():
    """Update the config.OBSERVATION_SIZE based on current schema."""
    schema = get_observation_schema("current_player")
    config.OBSERVATION_SIZE = schema.total_size
    return config.OBSERVATION_SIZE