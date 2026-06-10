"""
Centralized observation builder for TFT Set 4 Gym.
This module handles the construction of observations using the centralized schema,
with learnable embedding tables for champions, items, traits, and origins.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from . import config
from .stats import COST
from .origin_class import team_traits, amounts
from .origin_class_stats import origin_class as champion_origins, tiers as origin_tiers


# Embedding table dimensions
CHAMPION_EMBED_DIM = 32
ITEM_EMBED_DIM = 24
TRAIT_EMBED_DIM = 8
ORIGIN_EMBED_DIM = 8

# Per-slot encoding dimensions
PER_SLOT_DIM = (CHAMPION_EMBED_DIM +        # champion embedding
                3 * ITEM_EMBED_DIM +         # 3 item embeddings
                TRAIT_EMBED_DIM +            # trait embedding (summed)
                ORIGIN_EMBED_DIM +           # origin embedding (summed)
                1 +                          # star level (normalized)
                1)                           # chosen flag


class EmbeddingTable:
    """Manages a learnable embedding table."""

    def __init__(self, num_embeddings: int, embed_dim: int):
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.weights = np.random.uniform(-0.1, 0.1, (num_embeddings, embed_dim)).astype(np.float64)

    def lookup(self, indices: np.ndarray) -> np.ndarray:
        """Lookup embeddings for given indices. Returns zero vectors for -1 indices."""
        indices = np.asarray(indices, dtype=np.int32)
        clamped = np.clip(indices, 0, self.num_embeddings - 1)
        result = self.weights[clamped]
        result[indices == -1] = 0.0
        return result


class ObservationEmbeddingRegistry:
    """Registry of all embedding tables used in the observation system."""

    def __init__(self):
        self.champion_embeddings = EmbeddingTable(config.MAX_CHAMPION_IN_SET, CHAMPION_EMBED_DIM)
        self.item_embeddings = EmbeddingTable(37, ITEM_EMBED_DIM)
        self.trait_embeddings = EmbeddingTable(20, TRAIT_EMBED_DIM)
        self.origin_embeddings = EmbeddingTable(10, ORIGIN_EMBED_DIM)

    def get_trait_index(self, trait_name: str) -> int:
        """Get the index for a trait name (0-19)."""
        trait_names = list(team_traits.keys())
        if trait_name in trait_names:
            return trait_names.index(trait_name)
        return 0

    def get_origin_index(self, origin_name: str) -> int:
        """Get the index for an origin name (0-9)."""
        origin_names = _get_origin_names()
        if origin_name in origin_names:
            return origin_names.index(origin_name)
        return 0


def _get_trait_names() -> List[str]:
    return list(team_traits.keys())


def _get_origin_names() -> List[str]:
    return ['cultist', 'divine', 'dusk', 'elderwood', 'enlightened',
            'exile', 'ninja', 'spirit', 'the_boss', 'warlord']


def _get_champion_index(champ_name: str) -> int:
    if champ_name in COST:
        return list(COST.keys()).index(champ_name)
    return -1


def _get_item_index(item_name: str) -> int:
    all_items = sorted([
        'bf_sword', 'chain_vest', 'giants_belt', 'needlessly_large_rod',
        'negatron_cloak', 'recurve_bow', 'sparring_gloves', 'spatula',
        'tear_of_the_goddess',
        'bloodthirster', 'blue_buff', 'bramble_vest', 'chalice_of_power',
        'deathblade', 'dragons_claw', 'duelists_zeal', 'elderwood_heirloom',
        'force_of_nature', 'frozen_heart', 'gargoyle_stoneplate', 'giant_slayer',
        'guardian_angel', 'guinsoos_rageblade', 'hand_of_justice',
        'hextech_gunblade', 'infinity_edge', 'ionic_spark', 'jeweled_gauntlet',
        'last_whisper', 'locket_of_the_iron_solari', 'ludens_echo', 'mages_cap',
        'mantle_of_dusk', 'morellonomicon', 'quicksilver', 'rabadons_deathcap',
        'rapid_firecannon', 'redemption', 'runaans_hurricane', 'shroud_of_stillness',
        'spear_of_shojin', 'statikk_shiv', 'sunfire_cape', 'sword_of_the_divine',
        'thieves_gloves', 'titans_resolve', 'trap_claw', 'vanguards_cuirass',
        'warlords_banner', 'warmogs_armor', 'youmuus_ghostblade', 'zekes_herald',
        'zephyr', 'zzrot_portal',
    ])
    if item_name in all_items:
        return all_items.index(item_name)
    return -1


def _build_per_slot_encoding(champ, trait_embeddings, origin_embeddings) -> np.ndarray:
    """Build a single per-slot encoding vector (122 dims)."""
    if champ is None:
        return np.zeros(PER_SLOT_DIM, dtype=np.float64)

    # Champion embedding
    champ_idx = _get_champion_index(champ.name)
    champ_embed = EMBEDDING_REGISTRY.champion_embeddings.lookup(np.array([champ_idx]))[0]

    # Item embeddings (up to 3)
    item_embeds = np.zeros((3, ITEM_EMBED_DIM), dtype=np.float64)
    if hasattr(champ, 'items'):
        for i, item_name in enumerate(champ.items[:3]):
            if item_name:
                item_idx = _get_item_index(item_name)
                if item_idx >= 0:
                    item_embeds[i] = EMBEDDING_REGISTRY.item_embeddings.lookup(np.array([item_idx]))[0]

    # Trait embedding (summed then averaged if multiple)
    trait_embed = np.zeros(TRAIT_EMBED_DIM, dtype=np.float64)
    num_traits = 0
    if hasattr(champ, 'origin'):
        for trait_name in champ.origin:
            trait_idx = EMBEDDING_REGISTRY.get_trait_index(trait_name)
            trait_embed += trait_embeddings.lookup(np.array([trait_idx]))[0]
            num_traits += 1
    if num_traits > 1:
        trait_embed /= num_traits

    # Origin embedding (summed then averaged if multiple)
    origin_embed = np.zeros(ORIGIN_EMBED_DIM, dtype=np.float64)
    num_origins = 0
    origin_names = _get_origin_names()
    if hasattr(champ, 'origin'):
        for origin_name in champ.origin:
            if origin_name in origin_names:
                origin_idx = origin_names.index(origin_name)
                origin_embed += origin_embeddings.lookup(np.array([origin_idx]))[0]
                num_origins += 1
    if num_origins > 1:
        origin_embed /= num_origins

    # Star level (normalized to [0, 1])
    star_norm = (float(getattr(champ, 'stars', 1)) - 1.0) / 2.0

    # Chosen flag
    chosen_flag = 1.0 if getattr(champ, 'chosen', False) else 0.0

    return np.concatenate([
        champ_embed,
        item_embeds.flatten(),
        trait_embed,
        origin_embed,
        np.array([star_norm]),
        np.array([chosen_flag]),
    ])


# Global embedding registry
EMBEDDING_REGISTRY = ObservationEmbeddingRegistry()


class ObservationBuilder:
    """Centralized builder for creating observations from player state."""

    def __init__(self):
        self.current_player_schema = get_observation_schema("current_player")
        self.action_mask_schema = get_observation_schema("action_mask")

    def build_observation(self, player_id: str, player: Any, shop_vector: Optional[np.ndarray] = None,
                          players: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        tensor_obs = self._build_tensor_observation(player, shop_vector, players)
        action_mask = self._build_action_mask(player)
        return {"tensor": tensor_obs, "action_mask": action_mask}

    def _build_tensor_observation(self, player: Any, shop_vector: Optional[np.ndarray] = None,
                                   players: Optional[Dict] = None) -> np.ndarray:
        schema = self.current_player_schema
        observation = np.zeros(schema.total_size, dtype=np.float64)

        def set_field(name: str, value: Any):
            try:
                field_slice = schema.get_field_slice(name)
                field_def = schema.get_field(name)
                if isinstance(value, (int, float, np.number)):
                    val_arr = np.ones(field_def.shape) * float(value)
                elif isinstance(value, np.ndarray):
                    if value.shape != field_def.shape:
                        val_arr = np.zeros(field_def.shape)
                        slices = tuple(slice(0, min(s_src, s_dst))
                                       for s_src, s_dst in zip(value.shape, field_def.shape))
                        val_arr[slices] = value[slices]
                    else:
                        val_arr = value
                else:
                    val_arr = np.zeros(field_def.shape)
                observation[field_slice] = val_arr.flatten()
            except KeyError:
                pass

        # 1. Board state - one encoding per board hex (28 hexes)
        try:
            board_encoding = np.zeros((28, PER_SLOT_DIM), dtype=np.float64)
            hex_index = 0
            for x in range(7):
                for y in range(4):
                    unit = player.board[x][y]
                    if unit is not None:
                        board_encoding[hex_index] = _build_per_slot_encoding(
                            unit, EMBEDDING_REGISTRY.trait_embeddings,
                            EMBEDDING_REGISTRY.origin_embeddings)
                    hex_index += 1
            set_field("board", board_encoding)
        except Exception as e:
            pass

        # 2. Bench champions (9 slots)
        try:
            bench_encoding = np.zeros((9, PER_SLOT_DIM), dtype=np.float64)
            for i in range(9):
                unit = player.bench[i]
                if unit is not None:
                    bench_encoding[i] = _build_per_slot_encoding(
                        unit, EMBEDDING_REGISTRY.trait_embeddings,
                        EMBEDDING_REGISTRY.origin_embeddings)
            set_field("bench_champions", bench_encoding)
        except Exception:
            pass

        # 3. Bench items (10 slots)
        try:
            bench_items_encoding = np.zeros((10, ITEM_EMBED_DIM), dtype=np.float64)
            for i in range(10):
                item_name = player.item_bench[i]
                if item_name:
                    item_idx = _get_item_index(str(item_name))
                    if item_idx >= 0:
                        bench_items_encoding[i] = EMBEDDING_REGISTRY.item_embeddings.lookup(
                            np.array([item_idx]))[0]
            set_field("bench_items", bench_items_encoding)
        except Exception:
            pass

        # 4. Shop - champion embeddings (5 slots)
        try:
            shop_encoding = np.zeros((5, CHAMPION_EMBED_DIM), dtype=np.float64)
            if hasattr(player, 'shop_elems') and player.shop_elems is not None:
                for slot in range(5):
                    elem_idx = int(player.shop_elems[slot]) if player.shop_elems[slot] >= 0 else -1
                    if 0 <= elem_idx < config.MAX_CHAMPION_IN_SET:
                        shop_encoding[slot] = EMBEDDING_REGISTRY.champion_embeddings.lookup(
                            np.array([elem_idx]))[0]
            set_field("shop", shop_encoding)
        except Exception:
            pass

        # Shop chosen index
        try:
            shop_chosen = 0.0
            if shop_vector is not None:
                chosen = shop_vector[58]
                if hasattr(chosen, 'item'):
                    shop_chosen = float(chosen.item())
                else:
                    shop_chosen = float(chosen)
            set_field("shop_chosen", shop_chosen)
        except Exception:
            pass

        # 5. Team traits - normalized float per trait (20 traits)
        try:
            team_traits_encoding = np.zeros(20, dtype=np.float64)
            trait_names = _get_trait_names()
            for i, trait_name in enumerate(trait_names):
                tier = player.team_tiers.get(trait_name, 0)
                max_tier_list = origin_tiers.get(trait_name, [0])
                max_tier = max(max_tier_list) if max_tier_list else 1
                if max_tier > 0:
                    team_traits_encoding[i] = tier / max_tier
            set_field("team_traits", team_traits_encoding)
        except Exception:
            pass

        # 6. Team origins - normalized float per origin (10 origins)
        try:
            team_origins_encoding = np.zeros(10, dtype=np.float64)
            origin_names = _get_origin_names()
            for i, origin_name in enumerate(origin_names):
                tier = player.team_tiers.get(origin_name, 0)
                max_tier_list = origin_tiers.get(origin_name, [0])
                max_tier = max(max_tier_list) if max_tier_list else 1
                if max_tier > 0:
                    team_origins_encoding[i] = tier / max_tier
            set_field("team_origins", team_origins_encoding)
        except Exception:
            pass

        # 7. Player state (7 scalars)
        try:
            player_state = np.zeros(7, dtype=np.float64)
            player_state[0] = float(player.health)
            player_state[1] = float(player.gold)
            player_state[2] = float(player.level)
            player_state[3] = float(player.round)
            try:
                player_state[4] = float(player.level_costs[player.level] - player.exp)
            except (IndexError, AttributeError):
                player_state[4] = 0.0
            streak = player.loss_streak if abs(player.loss_streak) > player.win_streak else player.win_streak
            player_state[5] = float(streak)
            player_state[6] = float(player.turns_for_combat)
            set_field("player_state", player_state)
        except Exception:
            pass

        # 8. Opponent boards (7 opponents x 28 hexes x 122 dims)
        try:
            opponent_boards_encoding = np.zeros((7, 28, PER_SLOT_DIM), dtype=np.float64)
            if players is not None:
                player_num = getattr(player, 'player_num', 0)
                opp_idx = 0
                for p_id in sorted(players.keys()):
                    if p_id != player_num:
                        opp_player = players[p_id]
                        if opp_player is not None:
                            hex_idx = 0
                            for x in range(7):
                                for y in range(4):
                                    unit = opp_player.board[x][y]
                                    if unit is not None:
                                        opponent_boards_encoding[opp_idx, hex_idx] = (
                                            _build_per_slot_encoding(
                                                unit,
                                                EMBEDDING_REGISTRY.trait_embeddings,
                                                EMBEDDING_REGISTRY.origin_embeddings
                                            )
                                        )
                                    hex_idx += 1
                        opp_idx += 1
                        if opp_idx >= 7:
                            break
            set_field("opponent_boards", opponent_boards_encoding)
        except Exception:
            pass

        # 9. Opponent info (7 opponents x 4 scalars: health, gold, level, streak)
        try:
            opponent_info_encoding = np.zeros((7, 4), dtype=np.float64)
            if players is not None:
                player_num = getattr(player, 'player_num', 0)
                opp_idx = 0
                for p_id in sorted(players.keys()):
                    if p_id != player_num:
                        opp_player = players[p_id]
                        if opp_player is not None:
                            opponent_info_encoding[opp_idx, 0] = float(opp_player.health)
                            opponent_info_encoding[opp_idx, 1] = float(opp_player.gold)
                            opponent_info_encoding[opp_idx, 2] = float(opp_player.level)
                            opp_streak = (opp_player.loss_streak
                                          if abs(opp_player.loss_streak) > opp_player.win_streak
                                          else opp_player.win_streak)
                            opponent_info_encoding[opp_idx, 3] = float(opp_streak)
                        opp_idx += 1
                        if opp_idx >= 7:
                            break
            set_field("opponent_info", opponent_info_encoding)
        except Exception:
            pass

        return observation

    def _build_action_mask(self, player: Any) -> np.ndarray:
        mask_size = self.action_mask_schema.total_size
        action_mask = np.ones(mask_size, dtype=np.int8)
        if hasattr(player, 'shop_mask') and player.shop_mask is not None:
            shop_size = min(len(player.shop_mask), mask_size)
            action_mask[:shop_size] = player.shop_mask[:shop_size]
        return action_mask

    def get_field_from_observation(self, observation: np.ndarray, field_name: str) -> np.ndarray:
        field_slice = self.current_player_schema.get_field_slice(field_name)
        field = self.current_player_schema.get_field(field_name)
        return observation[field_slice].reshape(field.shape)

    def set_field_in_observation(self, observation: np.ndarray, field_name: str, value: np.ndarray) -> np.ndarray:
        field_slice = self.current_player_schema.get_field_slice(field_name)
        observation = observation.copy()
        observation[field_slice] = value.flatten()
        return observation


class ObservationManagerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_builder = ObservationBuilder()

    def update_observation_field(self, field_name: str, value: Any):
        pass

    def get_observation_for_field(self, field_name: str) -> np.ndarray:
        raise NotImplementedError


def update_config_observation_size():
    schema = get_observation_schema("current_player")
    config.OBSERVATION_SIZE = schema.total_size
    return config.OBSERVATION_SIZE


def build_observation(player_id: str, player: Any, shop_vector: Optional[np.ndarray] = None,
                      players: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    builder = ObservationBuilder()
    return builder.build_observation(player_id, player, shop_vector, players)


def get_field_slice(field_name: str) -> slice:
    schema = get_observation_schema("current_player")
    return schema.get_field_slice(field_name)


def get_field_value_from_obs(observation: np.ndarray, field_name: str) -> np.ndarray:
    builder = ObservationBuilder()
    return builder.get_field_from_observation(observation, field_name)


def set_field_value_in_obs(observation: np.ndarray, field_name: str, value: np.ndarray) -> np.ndarray:
    builder = ObservationBuilder()
    return builder.set_field_in_observation(observation, field_name, value)
