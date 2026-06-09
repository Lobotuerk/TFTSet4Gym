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
        """Setup the default TFT observation schemas.

        New embedding-based schema (TFT-182):
        - Learnable embeddings for champions (58x32), items (37x24), traits (20x8), origins (10x8)
        - Per-slot feature vectors: champion(32) + items(3x24) + traits(8) + origins(8) + star(1) + chosen(1) = 122
        - Team-level trait/origin tiers as normalized floats
        - Full opponent board visibility for all 7 opponents
        """
        # Per-slot feature vector components
        CHAMP_EMBED = 32
        ITEM_EMBED = 24
        NUM_ITEM_SLOTS = 3
        TRAIT_EMBED = 8
        ORIGIN_EMBED = 8
        STAR_DIM = 1
        CHOSEN_DIM = 1
        PER_SLOT_DIM = (CHAMP_EMBED +
                        NUM_ITEM_SLOTS * ITEM_EMBED +
                        TRAIT_EMBED + ORIGIN_EMBED +
                        STAR_DIM + CHOSEN_DIM)  # 32 + 72 + 8 + 8 + 1 + 1 = 122

        # Board: 28 slots (4x7 grid), each slot is PER_SLOT_DIM
        BOARD_DIM = 28
        BOARD_SHAPE = (BOARD_DIM, PER_SLOT_DIM)

        # Bench champions: 9 slots
        BENCH_CHAMP_DIM = 9
        BENCH_CHAMP_SHAPE = (BENCH_CHAMP_DIM, PER_SLOT_DIM)

        # Bench items: 10 slots, each ITEM_EMBED
        BENCH_ITEM_DIM = 10
        BENCH_ITEM_SHAPE = (BENCH_ITEM_DIM, ITEM_EMBED)

        # Shop: 5 champion slots x CHAMP_EMBED + 1 chosen index
        SHOP_CHAMP_DIM = 5
        SHOP_SHAPE = (SHOP_CHAMP_DIM, CHAMP_EMBED)
        SHOP_CHOSEN_DIM = 1

        # Team traits: 1 normalized float per trait (~20 traits)
        NUM_TRAITS = 20
        TEAM_TRAITS_SHAPE = (NUM_TRAITS,)

        # Team origins: 1 normalized float per origin (~10 origins)
        NUM_ORIGINS = 10
        TEAM_ORIGINS_SHAPE = (NUM_ORIGINS,)

        # Player state: 7 scalars
        PLAYER_STATE_DIM = 7
        PLAYER_STATE_SHAPE = (PLAYER_STATE_DIM,)

        # Opponent boards: 7 opponents x 28 slots x PER_SLOT_DIM
        NUM_OPPONENTS = 7
        OPPONENT_BOARDS_SHAPE = (NUM_OPPONENTS, BOARD_DIM, PER_SLOT_DIM)

        # Opponent info: 7 opponents x 4 scalars (health, gold, level, streak)
        OPPONENT_INFO_DIM = 4
        OPPONENT_INFO_SHAPE = (NUM_OPPONENTS, OPPONENT_INFO_DIM)

        current_player_fields = [
            ObservationField("board", BOARD_SHAPE, np.dtype('float64'),
                             f"Board representation: {BOARD_DIM} slots x {PER_SLOT_DIM} features (champ embed + items + traits + origins + star + chosen)"),
            ObservationField("bench_champions", BENCH_CHAMP_SHAPE, np.dtype('float64'),
                             f"Bench champions: {BENCH_CHAMP_DIM} slots x {PER_SLOT_DIM} features"),
            ObservationField("bench_items", BENCH_ITEM_SHAPE, np.dtype('float64'),
                             f"Bench items: {BENCH_ITEM_DIM} slots x {ITEM_EMBED} item embeddings"),
            ObservationField("shop_champions", SHOP_SHAPE, np.dtype('float64'),
                             f"Shop champions: {SHOP_CHAMP_DIM} slots x {CHAMP_EMBED} champion embeddings"),
            ObservationField("shop_chosen", (SHOP_CHOSEN_DIM,), np.dtype('float64'),
                             "Chosen champion index in shop (scalar)"),
            ObservationField("team_traits", TEAM_TRAITS_SHAPE, np.dtype('float64'),
                             f"Team trait tiers: {NUM_TRAITS} normalized floats (active_tier / max_tier)"),
            ObservationField("team_origins", TEAM_ORIGINS_SHAPE, np.dtype('float64'),
                             f"Team origin tiers: {NUM_ORIGINS} normalized floats (active_tier / max_tier)"),
            ObservationField("player_state", PLAYER_STATE_SHAPE, np.dtype('float64'),
                             f"Player state: {PLAYER_STATE_DIM} scalars (health, gold, level, round, exp_to_level, streak, turns_for_combat)"),
            ObservationField("opponent_boards", OPPONENT_BOARDS_SHAPE, np.dtype('float64'),
                             f"Opponent boards: {NUM_OPPONENTS} opponents x {BOARD_DIM} slots x {PER_SLOT_DIM} features"),
            ObservationField("opponent_info", OPPONENT_INFO_SHAPE, np.dtype('float64'),
                             f"Opponent info: {NUM_OPPONENTS} opponents x {OPPONENT_INFO_DIM} scalars (health, gold, level, streak)"),
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
                "low": -100.0,
                "high": 100.0,
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
    
    def get_per_slot_dim(self) -> int:
        """Get the per-slot feature dimension for board/bench champion slots."""
        schema = self.get_schema("current_player")
        board_field = schema.get_field("board")
        return board_field.shape[1]  # Second dimension is per-slot dim
    
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