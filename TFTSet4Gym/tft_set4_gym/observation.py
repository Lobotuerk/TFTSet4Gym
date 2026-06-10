import collections
import numpy as np
from . import config
from .stats import COST
from .origin_class import team_traits, game_comp_tiers
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from .observation_builder import ObservationBuilder


class Observation:
    def __init__(self):
        self.builder = ObservationBuilder()

        # Legacy attributes for backward compatibility
        self.shop_vector = None
        self.shop_mask = np.ones(5, dtype=np.int8)
        self.game_comp_vector = np.zeros(208)

        schema = get_observation_schema("current_player")
        self.dummy_observation = np.zeros(schema.total_size)

        self.cur_player_observations = collections.deque(
            maxlen=config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        )
        self.other_player_observations = {
            "player_" + str(player_id): np.zeros((7, 28, 122))
            for player_id in range(config.NUM_PLAYERS)
        }
        self.turn_since_update = 0.01

    def observation(self, player_id: str, player, action_vector=np.array([]), players=None):
        """
        Create observation using the new embedding-based schema.

        Args:
            player_id: string - the player identifier
            player: Player object - the player to observe
            action_vector: numpy array - unused, kept for compatibility
            players: dict - all players in the game (for opponent boards)

        Returns:
            Dictionary with 'tensor' and 'action_mask' keys
        """
        obs_dict = self.builder.build_observation(
            player_id, player, self.shop_vector, players
        )

        game_state_tensor = obs_dict["tensor"]

        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append(game_state_tensor)

        self.cur_player_observations.append(game_state_tensor)

        cur_player_tensor_observation = []
        for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL):
            tensor = self.cur_player_observations[i]
            cur_player_tensor_observation.append(tensor)

        final_tensor = np.asarray(cur_player_tensor_observation).flatten()

        self.turn_since_update += 0.01

        return {
            "tensor": final_tensor,
            "action_mask": obs_dict["action_mask"]
        }

    def get_field_value(self, field_name: str, observation=None):
        """Get the value of a specific field from an observation."""
        if observation is None:
            observation = self.dummy_observation
        return self.builder.get_field_from_observation(observation, field_name)

    def set_field_value(self, field_name: str, value, observation=None):
        """Set the value of a specific field in an observation."""
        if observation is None:
            observation = self.dummy_observation
        return self.builder.set_field_in_observation(observation, field_name, value)

    def generate_other_player_vectors(self, cur_player, players):
        """
        Generate opponent board vectors from the perspective of the current player.
        Each opponent board is encoded with the same per-slot embedding structure.
        """
        for player_id in players:
            other_player = players[player_id]
            if other_player != cur_player and other_player is not None:
                opp_idx = player_id
                hex_idx = 0
                for x in range(7):
                    for y in range(4):
                        unit = other_player.board[x][y]
                        if unit:
                            # Build per-slot encoding for opponent unit
                            from .observation_builder import (
                                _build_per_slot_encoding,
                                EMBEDDING_REGISTRY,
                            )
                            self.other_player_observations[player_id][opp_idx, hex_idx] = (
                                _build_per_slot_encoding(
                                    unit,
                                    EMBEDDING_REGISTRY.trait_embeddings,
                                    EMBEDDING_REGISTRY.origin_embeddings,
                                )
                            )
                        hex_idx += 1
        self.turn_since_update = 0

    def generate_game_comps_vector(self):
        """Generates the vector for a comp tier for a given player."""
        output = np.zeros(208)
        for i in range(len(game_comp_tiers)):
            tiers = np.array(list(game_comp_tiers[i].values()))
            tierMax = np.max(tiers)
            if tierMax != 0:
                tiers = tiers / tierMax
            output[i * 26: i * 26 + 26] = tiers
        self.game_comp_vector = output

    def generate_shop_vector(self, shop, player):
        """
        Generates the shop vector and information for the shop mask.
        """
        output_array = np.zeros(59)
        shop_chosen = False
        chosen_shop_index = -1
        chosen_shop = ''
        shop_costs = np.zeros((5, 1))
        shop_counts = np.zeros((58, 1))
        shop_elems = np.zeros((5, 1))
        for x in range(0, len(shop)):
            if shop[x] != " ":
                chosen = 0
                if shop[x].endswith('_c'):
                    chosen_shop_index = x
                    chosen_shop = shop[x]
                    c_shop = shop[x].split('_')
                    shop[x] = c_shop[0]
                    chosen = 1
                    shop_chosen = c_shop[1]
                    if COST[shop[x]] == 1:
                        shop_costs[x] = 3
                    else:
                        shop_costs[x] = 3 * COST[shop[x]] - 1
                else:
                    shop_costs[x] = COST[shop[x]]
                c_index = list(COST.keys()).index(shop[x])
                shop_elems[x] = c_index - 1
                shop_counts[c_index - 1] += 1
                self.shop_mask[x] = 1
            else:
                shop_elems[x] = -1
                self.shop_mask[x] = 0

            self.shop_mask[x] = 0

        if shop_chosen:
            shop[chosen_shop_index] = chosen_shop

        player.shop_costs = shop_costs
        player.shop_elems = shop_elems

        for idx, cost in enumerate(player.shop_costs):
            if player.gold < cost or cost == 0:
                self.shop_mask[idx] = 0
            elif player.gold >= cost:
                self.shop_mask[idx] = 1

        if player.bench_full():
            self.shop_mask = np.zeros(5)

        output_array[:58] = shop_counts.flatten()
        if chosen_shop != '':
            output_array[58] = list(COST.keys()).index(chosen_shop.split('_')[0]) - 1

        self.shop_vector = output_array
        player.shop_mask = self.shop_mask
