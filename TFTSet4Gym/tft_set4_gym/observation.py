import collections
import numpy as np
from . import config
from .observation_schema import get_observation_schema
from .observation_builder import ObservationBuilder

'''
Includes the vector of the shop, bench, board, and item list.
Add a vector for each player composition makeup at the start of the round.
action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
'''
class Observation:
    def __init__(self):
        self.builder = ObservationBuilder()

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
        obs_dict = self.builder.build_observation(player_id, player)
        
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

    def build_full_observation(self, player_id: str, player, players: dict) -> dict:
        """Build observation including opponent data using the new schema.

        Returns a dict with 'tensor' and 'action_mask' keys, where the tensor
        includes opponent_boards and opponent_info fields.
        """
        return self.builder.build_full_observation(player_id, player, players)


