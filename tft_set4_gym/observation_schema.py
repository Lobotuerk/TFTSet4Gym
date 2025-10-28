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
        
        # Current player observation schema
        current_player_fields = [
            # Board representation (58 champions, 4x7 grid)
            ObservationField("board_champions", (58, 4, 7), np.dtype('float64'), "Champion positions on board"),
            ObservationField("board_stars", (1, 4, 7), np.dtype('float64'), "Star levels of champions on board"),
            ObservationField("board_chosen", (1, 4, 7), np.dtype('float64'), "Chosen status of champions on board"),
            
            # Bench representation (58 possible champions)
            ObservationField("bench_champions", (58, 4, 7), np.dtype('float64'), "Champions on bench by type"),
            
            # Player state
            ObservationField("health", (1, 4, 7), np.dtype('float64'), "Player health"),
            ObservationField("turns_for_combat", (1, 4, 7), np.dtype('float64'), "Remaining action turns"),
            ObservationField("level", (1, 4, 7), np.dtype('float64'), "Player level"),
            ObservationField("round", (1, 4, 7), np.dtype('float64'), "Current round number"),
            
            # Private information (flattened for concatenation)
            ObservationField("exp_to_level", (1, 4, 7), np.dtype('float64'), "Experience needed to level"),
            ObservationField("gold", (1, 4, 7), np.dtype('float64'), "Current gold amount"),
            ObservationField("streak", (1, 4, 7), np.dtype('float64'), "Win/loss streak"),
            
            # Shop information
            ObservationField("shop_champions", (58, 4, 7), np.dtype('float64'), "Available champions in shop"),
            ObservationField("shop_chosen", (1, 4, 7), np.dtype('float64'), "Chosen champion in shop"),
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
                "low": 0.0,
                "high": 55.0,
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