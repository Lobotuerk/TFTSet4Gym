from . import config
import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict, Tuple
from . import pool
from .player import Player as player_class
from .step_function import Step_Function
from .game_round import Game_Round
from .observation import Observation
from .observation_schema import OBSERVATION_REGISTRY
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import AgentSelector
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env(rank):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer pettingzoo documentation.
    """
    local_env = TFT_Simulator(env_config=None, rank=rank)

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env


from pettingzoo.utils.env import ParallelEnv


class TFTParallelWrapper(ParallelEnv):
    """Custom parallel wrapper that ensures agents and num_agents are always accessible."""
    
    def __init__(self, aec_env_fn, rank=0):
        from pettingzoo.utils.conversions import aec_to_parallel_wrapper
        self.aec_env = aec_env_fn(rank)
        self.par_env = aec_to_parallel_wrapper(self.aec_env)
        self._possible_agents = self.aec_env.possible_agents
        
        # Initialize required ParallelEnv attributes
        self.agents = self._get_current_agents()
    
    def _get_current_agents(self):
        """Get current agents. Always accessible."""
        try:
            if hasattr(self.aec_env, 'agents') and self.aec_env.agents is not None:
                # Return only live agents (those not terminated)
                live_agents = []
                for agent in self.aec_env.agents:
                    if not getattr(self.aec_env, 'terminations', {}).get(agent, False):
                        live_agents.append(agent)
                return live_agents
        except Exception as e:
            print(f"Error getting current agents: {e}")
        return self._possible_agents[:]
    
    def reset(self, seed=None, options=None):
        """Reset and update agents."""
        result = self.par_env.reset(seed=seed, options=options)
        self.agents = self._get_current_agents()
        return result
    
    def step(self, actions):
        """Step and update agents."""
        result = self.par_env.step(actions)
        self.agents = self._get_current_agents()
        return result
    
    @property 
    def num_agents(self):
        """Get number of current agents."""
        return len(self.agents)
    
    def state(self):
        """Return the global state from the underlying AEC environment."""
        return self.aec_env.state()
    
    def action_space(self, agent):
        """Return action space for the given agent."""
        return self.aec_env.action_space(agent)
    
    def observation_space(self, agent):
        """Return observation space for the given agent."""
        return self.aec_env.observation_space(agent)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the parallel environment."""
        return getattr(self.par_env, name)


def parallel_env(rank=0):
    """Create a parallel environment with proper agents/num_agents support."""
    return TFTParallelWrapper(env, rank)


class TFT_Simulator(AECEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self, env_config, rank):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}
        self.rank = rank

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function, rank)
        self.actions_taken = 0
        self.actions_taken_this_turn = 0
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        self.step_function.generate_shop_vectors(self.PLAYERS)

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.kill_list = []
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = AgentSelector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        # For MuZero
        # self.observation_spaces: Dict = dict(
        #     zip(self.agents,
        #         [Box(low=(-5.0), high=5.0, shape=(config.NUM_PLAYERS, config.OBSERVATION_SIZE,),
        #              dtype=np.float32) for _ in self.possible_agents])
        # )

        # For PPO - Use schema-based observation spaces
        schema_config = OBSERVATION_REGISTRY.get_combined_schema_config()
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Dict({
                        "tensor": Box(**schema_config["tensor"]),
                        "action_mask": Box(**schema_config["action_mask"])
                    }) for _ in self.agents
                ],
            )
        )

        # For MuZero
        # self.action_spaces = {agent: MultiDiscrete([config.ACTION_DIM for _ in range(config.NUM_PLAYERS)])
        #                       for agent in self.agents}

        # For PPO
        self.action_spaces = {agent: MultiDiscrete(config.ACTION_DIM)
                              for agent in self.agents}
        super().__init__()
    
    def get_observation_field(self, agent: str, field_name: str):
        """Get a specific field value from an agent's observation."""
        if agent not in self.observations or "tensor" not in self.observations[agent]:
            return None
            
        game_obs = self.game_observations[agent]
        return game_obs.get_field_value(field_name, self.observations[agent]["tensor"])
    
    def update_observation_schema(self, new_schema_name: str):
        """Dynamically update the observation schema for experimentation."""
        if new_schema_name in OBSERVATION_REGISTRY._schemas:
            # Update all observation objects to use new schema
            for game_obs in self.game_observations.values():
                from .observation_builder import ObservationBuilder
                game_obs.builder = ObservationBuilder()  # Recreate with new schema
            
            # Update observation spaces
            schema_config = OBSERVATION_REGISTRY.get_combined_schema_config()
            self.observation_spaces = dict(
                zip(
                    self.agents,
                    [
                        Dict({
                            "tensor": Box(**schema_config["tensor"]),
                            "action_mask": Box(**schema_config["action_mask"])
                        }) for _ in self.agents
                    ],
                )
            )
            print(f"Updated observation schema to: {new_schema_name}")
        else:
            print(f"Schema '{new_schema_name}' not found")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        return self.action_spaces[agent]

    def check_dead(self):
        num_alive = 0
        for key, player in self.PLAYERS.items():
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.game_round.NUM_DEAD = self.NUM_DEAD
                    self.pool_obj.return_hero(player)
                    self.kill_list.append(key)
                else:
                    num_alive += 1
        return num_alive

    def observe(self, player_id):
        return self.observations[player_id]

    def reset(self, seed=None, options=None):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.NUM_DEAD = 0
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function, self.rank)
        self.actions_taken = 0
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        self.step_function.generate_shop_vectors(self.PLAYERS)

        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        for agent in self.agents:
            self.PLAYERS[agent].turns_for_combat = config.ACTIONS_PER_TURN

        self.observations = {agent: self.game_observations[agent].observation(
            agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector) for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        super().__init__()
        return self.observations

    def render(self):
        ...

    def state(self):
        """
        Returns the global state of the environment as a numpy array.
        For TFT, this represents the combined state of all players.
        """
        if not hasattr(self, 'observations') or not self.observations:
            # Return empty state if no observations are available
            return np.array([])
        
        # Concatenate all player observations into a global state
        # Since observations are complex (dict with tensor and mask),
        # we'll use the tensor part and flatten it
        state_components = []
        for agent in sorted(self.possible_agents):  # Use sorted for consistent ordering
            if agent in self.observations and 'tensor' in self.observations[agent]:
                tensor_obs = self.observations[agent]['tensor']
                # Flatten the tensor observation
                flattened = np.asarray(tensor_obs).flatten()
                state_components.append(flattened)
        
        if state_components:
            return np.concatenate(state_components)
        else:
            # Fallback: return a simple state representation
            return np.array([len(self.agents), self.NUM_DEAD])

    def close(self):
        self.reset()

    def step(self, action):
        # step for dead agents
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return
        action = np.asarray(action)
        self.step_function.batch_2d_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                   self.agent_selection, self.game_observations)

        # if we don't use this line, rewards will compound per step
        # (e.g. if player 1 gets reward in step 1, he will get rewards in steps 2-8)
        self._clear_rewards()
        self.infos[self.agent_selection] = {"state_empty": self.PLAYERS[self.agent_selection].state_empty()}

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        for agent in self.agents:
            self.PLAYERS[agent].turns_for_combat = config.ACTIONS_PER_TURN - self.actions_taken
            self.observations[agent] = self.game_observations[agent].observation(
                agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector)

        # Also called in many environments but the line above this does the same thing but better
        # self._accumulate_rewards()
        if self._agent_selector.is_last():
            self.actions_taken += 1

            # If at the end of the turn
            if self.actions_taken >= config.ACTIONS_PER_TURN:
                # Take a game action and reset actions taken
                self.actions_taken = 0
                self.game_round.play_game_round()

                # Check if the game is over
                if self.check_dead() <= 1 or self.game_round.current_round > 48:
                    # Anyone left alive (should only be 1 player unless time limit) wins the game
                    for player_id in self.agents:
                        if self.PLAYERS[player_id] and self.PLAYERS[player_id].health > 0:
                            self.rewards[player_id] = 250
                            self._cumulative_rewards[player_id] = self.rewards[player_id]
                            self.PLAYERS[player_id] = None  # Without this the reward is reset

                    self.terminations = {a: True for a in self.agents}

                self.infos = {a: {"state_empty": False} for a in self.agents}

                _live_agents = self.agents[:]
                for k in self.kill_list:
                    self.terminations[k] = True
                    self.rewards[k] = (8 - len(_live_agents)) * 25
                    _live_agents.remove(k)
                    self._cumulative_rewards[k] = self.rewards[k]
                    self.PLAYERS[k] = None
                    self.game_round.update_players(self.PLAYERS)

                if len(self.kill_list) > 0:
                    self._agent_selector.reinit(_live_agents)
                    # Update the main agents list to reflect eliminated players
                    self.agents = _live_agents[:]
                self.kill_list = []

                if not all(self.terminations.values()):
                    self.game_round.start_round()

                    for agent in _live_agents:
                        self.PLAYERS[agent].turns_for_combat = config.ACTIONS_PER_TURN - self.actions_taken
                        self.observations[agent] = self.game_observations[agent].observation(
                            agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector)

            for player_id in self.PLAYERS:
                if self.PLAYERS[player_id]:
                    self.rewards[player_id] = self.PLAYERS[player_id].reward
                    self._cumulative_rewards[player_id] = self.rewards[player_id]

        # I think this if statement is needed in case all the agents die to the same minion round. a little sad.
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        # Probably not needed but doesn't hurt?
        self._deads_step_first()
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
