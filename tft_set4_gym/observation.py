import collections
import numpy as np
from . import config
from .stats import COST
from .origin_class import team_traits, game_comp_tiers
from . import utils as utils
from .observation_schema import get_observation_schema, OBSERVATION_REGISTRY
from .observation_builder import ObservationBuilder

'''
Includes the vector of the shop, bench, board, and item list.
Add a vector for each player composition makeup at the start of the round.
action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
'''
class Observation:
    def __init__(self):
        # Enhanced observation system with schema integration
        self.builder = ObservationBuilder()
        
        # Legacy attributes for backward compatibility
        self.shop_vector = None
        self.shop_mask = np.ones(5, dtype=np.int8)
        self.game_comp_vector = np.zeros(208)
        
        # Use schema to determine observation size
        schema = get_observation_schema("current_player")
        self.dummy_observation = np.zeros(schema.total_size)
        
        self.cur_player_observations = collections.deque(maxlen=config.OBSERVATION_TIME_STEPS *
                                                         config.OBSERVATION_TIME_STEP_INTERVAL)
        self.other_player_observations = {"player_" + str(player_id): np.zeros((26, 6, 10))
                                          for player_id in range(config.NUM_PLAYERS)}
        self.turn_since_update = 0.01

    """
    Description - Creates an observation for a given player.
    Inputs      - player_id: string
                    The player_id for the given player, used when adding other players observations
                  player: Player object
                    The player to get all of the observation vectors from
                  action_vector: numpy array
                    The next action format to use if using a 1d action space.
    Outputs     - A dictionary with a tensor field (input to the representation network) and a mask for legal actions
    """
    def observation(self, player_id: str, player, action_vector=np.array([])):
        """
        Create observation using the modern schema system.
        Maintains backward compatibility with existing interface.
        """
        # Build the main observation using the schema
        obs_dict = self.builder.build_observation(player_id, player, self.shop_vector)
        
        # Handle time-stepped observations (existing logic)
        game_state_tensor = obs_dict["tensor"]
        
        # Time-stepping logic (keep existing behavior)
        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append(game_state_tensor)
        
        self.cur_player_observations.append(game_state_tensor)
        
        # Sample time-stepped observations
        cur_player_tensor_observation = []
        for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL):
            tensor = self.cur_player_observations[i]
            cur_player_tensor_observation.append(tensor)
        
        final_tensor = np.asarray(cur_player_tensor_observation).flatten()
        
        # Update time tracking
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

    """
    Description - Generates the other players observation from the perspective of the current player.
                  This is the same as looking at each other board individually in a game.
    Inputs      - cur_player: Player object
                    Player whose perspective it is from
                  players: List of Player objects
                    All players in the game.
    """
    def generate_other_player_vectors(self, cur_player, players):
        for player_id in players:
            other_player = players[player_id]
            if other_player != cur_player:
                other_player_vector = np.zeros((26,6,10))
                if other_player:
                    other_player_vector = other_player.player_public_vector
                # other_player.board_vector,
                # other_player.bench_vector,
                # other_player.chosen_vector,
                # other_player.item_vector,
                self.other_player_observations[player_id] = other_player_vector
        self.turn_since_update = 0

    """
    Description - Generates the vector for a comp tier for a given player. This is equal to the game compositions bar 
                  on the left in TFT. 
    """
    # TODO: Add other player's compositions to the list of other player's vectors.
    def generate_game_comps_vector(self):
        output = np.zeros(208)
        for i in range(len(game_comp_tiers)):
            tiers = np.array(list(game_comp_tiers[i].values()))
            tierMax = np.max(tiers)
            if tierMax != 0:
                tiers = tiers / tierMax
            output[i * 26: i * 26 + 26] = tiers
        self.game_comp_vector = output

    '''
    Description - Generates the shop vector and information for the shop mask. This is a binary encoding of the champ
                  costs list for each shop location. 0s if there is no shop option available.
    Inputs      - shop: List of strings
                    shop to transform into a vector
                  player: Player Object
                    player who the shop belongs to.
    '''
    def generate_shop_vector(self, shop, player):
        # each champion has 6 bit for the name, 1 bit for the chosen.
        # 5 of them makes it 35.
        output_array = np.zeros((62, 4, 7))
        shop_chosen = False
        chosen_shop_index = -1
        chosen_shop = ''
        shop_costs = np.zeros((5, 1))
        shop_counts = np.zeros((58,1))
        shop_elems = np.zeros((5, 1))
        for x in range(0, len(shop)):
            if shop[x] != " ":
                chosen = 0
                if shop[x].endswith("_c"):
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
                shop_elems[x] = c_index-1
                shop_counts[c_index-1] += 1
                self.shop_mask[x] = 1
            else:
                shop_elems[x] = -1
                self.shop_mask[x] = 0

            # Input chosen mechanics once I go back and update the chosen mechanics.
            self.shop_mask[x] = 0
        if shop_chosen:
            # if shop_chosen == 'the':
            #     shop_chosen = 'the_boss'
            # c_index = list(team_traits.keys()).index(shop_chosen)
            # # This should update the item name section of the vector
            # for z in range(5, 0, -1):
            #     if c_index > 2 * z:
            #         # output_array[45 - z] = 1
            #         c_index -= 2 * z
            shop[chosen_shop_index] = chosen_shop

        player.shop_costs = shop_costs
        player.shop_elems = shop_elems

        # print(player.player_num, " has ", player.shop_elems, " = ", shop)

        for idx, cost in enumerate(player.shop_costs):
            if player.gold < cost or cost == 0:
                self.shop_mask[idx] = 0
            elif player.gold >= cost:
                self.shop_mask[idx] = 1

        if player.bench_full():
            self.shop_mask = np.zeros(5)

        for n, _ in enumerate(shop_counts):
            output_array[n] = np.ones((4,7)) * shop_counts[n]
        if chosen_shop != '':
            output_array[58] = np.ones((4,7)) * (list(COST.keys()).index(chosen_shop.split('_')[0])-1)

        self.shop_vector = output_array
        player.shop_mask = self.shop_mask
