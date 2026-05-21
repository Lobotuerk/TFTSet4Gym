from . import config
import functools
import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict
from . import pool
from .player import Player as player_class
from .step_function import Step_Function
from .game_round import Game_Round
from .observation import Observation
from .observation_schema import OBSERVATION_REGISTRY
from pettingzoo.utils.env import ParallelEnv


def parallel_env(rank=0):
    """Create a parallel environment."""
    return TFT_Simulator(env_config=None, rank=rank)


class TFT_Simulator(ParallelEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0", "render_modes": []}

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
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        self.step_function.generate_shop_vectors(self.PLAYERS)

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.kill_list = []
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {"state_empty": False} for agent in self.possible_agents}
        self.observations = {agent: {} for agent in self.possible_agents}

        # For PPO - Use schema-based observation spaces
        schema_config = OBSERVATION_REGISTRY.get_combined_schema_config()
        self.observation_spaces = dict(
            zip(
                self.possible_agents,
                [
                    Dict({
                        "tensor": Box(**schema_config["tensor"]),
                        "action_mask": Box(**schema_config["action_mask"])
                    }) for _ in self.possible_agents
                ],
            )
        )

        # For PPO
        self.action_spaces = {agent: MultiDiscrete(config.ACTION_DIM)
                              for agent in self.possible_agents}
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
                    self.possible_agents,
                    [
                        Dict({
                            "tensor": Box(**schema_config["tensor"]),
                            "action_mask": Box(**schema_config["action_mask"])
                        }) for _ in self.possible_agents
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

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        for agent in self.agents:
            self.PLAYERS[agent].turns_for_combat = config.ACTIONS_PER_TURN

        self.observations = {agent: self.game_observations[agent].observation(
            agent, self.PLAYERS[agent]) for agent in self.agents}

        return self.observations, self.infos

    def render(self):
        """
        Renders the environment. 
        Currently placeholder to satisfy PettingZoo API.
        """
        return None

    def state(self):
        """
        Returns the global state of the environment as a numpy array.
        """
        if not hasattr(self, 'observations') or not self.observations:
            return np.array([])
        
        state_components = []
        for agent in sorted(self.possible_agents):
            if agent in self.observations and 'tensor' in self.observations[agent]:
                tensor_obs = self.observations[agent]['tensor']
                flattened = np.asarray(tensor_obs).flatten()
                state_components.append(flattened)
        
        if state_components:
            return np.concatenate(state_components)
        else:
            return np.array([len(self.agents), self.NUM_DEAD])

    def close(self):
        pass

    def step(self, actions):
        """
        Step function for parallel environment.
        actions: a dictionary of actions for each agent
        """
        # Keep track of agents that were active at the start of the step
        # These agents must be in the return dictionaries
        active_this_step = self.agents[:]
        
        # Clear rewards from previous step
        self.rewards = {agent: 0 for agent in active_this_step}
        
        # Apply actions for all active agents
        for agent, action in actions.items():
            if agent in self.agents and not self.terminations.get(agent, False):
                action = np.asarray(action)
                self.step_function.batch_2d_controller(action, self.PLAYERS[agent], self.PLAYERS,
                                                           agent, self.game_observations)
                self.infos[agent] = {"state_empty": self.PLAYERS[agent].state_empty()}

        self.actions_taken += 1

        # If we have reached the end of the turn for all agents
        if self.actions_taken >= config.ACTIONS_PER_TURN:
            self.actions_taken = 0
            self.game_round.play_game_round()

            # Check if the game is over
            if self.check_dead() <= 1 or self.game_round.current_round > 48:
                for agent in self.agents:
                    if self.PLAYERS[agent] and self.PLAYERS[agent].health > 0:
                        self.rewards[agent] += 250
                        self.PLAYERS[agent] = None

                for agent in self.agents:
                    self.terminations[agent] = True
                self.agents = []

            if self.agents:
                self.infos = {a: {"state_empty": False} for a in self.agents}

                # Handle player elimination
                _live_agents = self.agents[:]
                for k in self.kill_list:
                    self.terminations[k] = True
                    self.rewards[k] += (8 - len(_live_agents)) * 25
                    if k in _live_agents:
                        _live_agents.remove(k)
                    self.PLAYERS[k] = None
                
                self.game_round.update_players(self.PLAYERS)
                self.agents = _live_agents[:]
                self.kill_list = []

                if not all(self.terminations.values()) and self.agents:
                    self.game_round.start_round()

        # Update observations and rewards for all agents
        for agent in self.possible_agents:
            if agent in self.PLAYERS and self.PLAYERS[agent]:
                self.PLAYERS[agent].turns_for_combat = config.ACTIONS_PER_TURN - self.actions_taken
                self.observations[agent] = self.game_observations[agent].observation(
                    agent, self.PLAYERS[agent])
                # Add player-specific rewards accumulated during step/round
                self.rewards[agent] += self.PLAYERS[agent].reward
                # Reset the reward in the player object so it's not double-counted next step
                self.PLAYERS[agent].reward = 0

        # PettingZoo ParallelEnv expects only entries for current agents
        # except when they just terminated.
        
        out_obs = {a: self.observations[a] for a in active_this_step}
        out_rewards = {a: self.rewards[a] for a in active_this_step}
        out_terminations = {a: self.terminations.get(a, False) for a in active_this_step}
        out_truncations = {a: self.truncations.get(a, False) for a in active_this_step}
        out_infos = {a: self.infos[a] for a in active_this_step}

        return out_obs, out_rewards, out_terminations, out_truncations, out_infos

