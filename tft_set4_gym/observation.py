import collections
import numpy as np
import config
from Simulator.stats import COST
from Simulator.origin_class import team_traits, game_comp_tiers
import Simulator.utils as utils

'''
Includes the vector of the shop, bench, board, and item list.
Add a vector for each player composition makeup at the start of the round.
action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
'''
class Observation:
    def __init__(self):
        self.shop_vector = None
        self.shop_mask = np.ones(5, dtype=np.int8)
        self.game_comp_vector = np.zeros(208)
        self.dummy_observation = np.zeros(config.OBSERVATION_SIZE)
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
    def observation(self, player_id, player, action_vector=np.array([])):
        # Fetch the shop vector and game comp vector
        shop_vector = self.shop_vector
        game_state_vector = self.game_comp_vector
        # Concatenate all vector based player information
        game_state_tensor = np.concatenate((player.player_public_vector, player.player_private_vector + shop_vector))
        # player.bench_vector,
        # player.chosen_vector,
        # player.item_vector,
        # game_state_vector,
        # action_vector,
        # np.expand_dims(self.turn_since_update, axis=-1)

        # Initially fill the queue with duplicates of first observation
        # we can still sample when there aren't enough time steps yet
        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append(game_state_tensor)

        # Enqueue the latest observation and pop the oldest (performed automatically by deque with maxLen configured)
        self.cur_player_observations.append(game_state_tensor)

        # # sample every N time steps at M intervals, where maxLen of queue = M*N
        # cur_player_observation = np.array([self.cur_player_observations[i]
        #                               for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL)]).flatten()

        cur_player_tensor_observation = []
        for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL):
            tensor = self.cur_player_observations[i]
            cur_player_tensor_observation.append(tensor)
        cur_player_tensor_observation = np.asarray(cur_player_tensor_observation).flatten()
        # cur_player_tensor_observation = cur_player_tensor_observation

        # Fetch other player data
        # other_player_tensor_observation_list = []
        # for k, v in self.other_player_observations.items():
        #     if k != player_id:
        #         other_player_tensor_observation_list.append(v)
        # # other_player_tensor_observation = np.array(other_player_tensor_observation_list).flatten()
        # other_player_tensor_observation = np.concatenate(other_player_tensor_observation_list)

        # Gather all vectors into one place
        # total_tensor_observation = np.concatenate((cur_player_tensor_observation, other_player_tensor_observation))
        total_tensor_observation = cur_player_tensor_observation

        # Create a simple action mask - for now allow all actions
        # TODO: Implement proper action masking based on game state
        action_mask = np.ones(54, dtype=np.int8)

        # Used to help the model know how outdated it's information on other players is.
        # Also helps with ensuring that two observations with the same board and bench are not equal.
        self.turn_since_update += 0.01
        return {"tensor": total_tensor_observation, "action_mask": action_mask}

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
