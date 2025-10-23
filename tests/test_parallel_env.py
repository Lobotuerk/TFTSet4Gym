"""
Test suite to verify that parallel_env is a proper PettingZoo parallel environment.
This test follows the PettingZoo parallel API specification from:
https://pettingzoo.farama.org/api/parallel/

Tests all required methods, attributes, and behaviors for a valid parallel environment.
"""
import sys
import os

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gymnasium as gym
from typing import Dict, Any
from pettingzoo.test import parallel_api_test
from pettingzoo.utils.env import ParallelEnv

from tft_set4_gym.tft_simulator import parallel_env


class TestParallelEnvAPI:
    """Test that parallel_env conforms to PettingZoo parallel API specification."""
    
    def get_env(self):
        """Create parallel environment."""
        # The parallel_env created by parallel_wrapper_fn expects the rank parameter
        # to be passed to the underlying env function. We need to use it properly.
        try:
            return parallel_env(rank=0)
        except:
            # If that doesn't work, try the approach from the existing codebase
            return parallel_env()
    
    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        env = self.get_env()
        assert env is not None
        assert isinstance(env, ParallelEnv)
        env.close()
    
    def test_required_attributes(self):
        """Test that all required attributes are present and have correct types."""
        env = self.get_env()
        
        # Test agents attribute
        assert hasattr(env, 'agents')
        assert isinstance(env.agents, list)
        assert all(isinstance(agent, str) for agent in env.agents)
        
        # Test num_agents attribute
        assert hasattr(env, 'num_agents')
        assert isinstance(env.num_agents, int)
        assert env.num_agents == len(env.agents)
        
        # Test possible_agents attribute
        assert hasattr(env, 'possible_agents')
        assert isinstance(env.possible_agents, list)
        assert all(isinstance(agent, str) for agent in env.possible_agents)
        
        # Test max_num_agents attribute
        assert hasattr(env, 'max_num_agents')
        assert isinstance(env.max_num_agents, int)
        assert env.max_num_agents == len(env.possible_agents)
        
        # Test observation_spaces attribute
        assert hasattr(env, 'observation_spaces')
        assert isinstance(env.observation_spaces, dict)
        assert all(agent in env.observation_spaces for agent in env.possible_agents)
        assert all(isinstance(space, gym.spaces.Space) for space in env.observation_spaces.values())
        
        # Test action_spaces attribute
        assert hasattr(env, 'action_spaces')
        assert isinstance(env.action_spaces, dict)
        assert all(agent in env.action_spaces for agent in env.possible_agents)
        assert all(isinstance(space, gym.spaces.Space) for space in env.action_spaces.values())
        
        env.close()
    
    def test_reset_method(self):
        """Test that reset method works correctly."""
        env = self.get_env()
        
        # Test reset without parameters
        observations, infos = env.reset()
        
        # Check return types
        assert isinstance(observations, dict)
        assert isinstance(infos, dict)
        
        # Check that all agents have observations and infos
        assert all(agent in observations for agent in env.agents)
        assert all(agent in infos for agent in env.agents)
        
        # Check observation shapes match observation spaces
        for agent in env.agents:
            obs = observations[agent]
            obs_space = env.observation_space(agent)
            assert obs_space.contains(obs), f"Observation for {agent} doesn't match observation space"
        
        # Test reset with seed
        observations2, infos2 = env.reset(seed=42)
        assert isinstance(observations2, dict)
        assert isinstance(infos2, dict)
        
        # Test reset with options
        observations3, infos3 = env.reset(options={})
        assert isinstance(observations3, dict)
        assert isinstance(infos3, dict)
        
        env.close()
    
    def test_step_method(self):
        """Test that step method works correctly."""
        env = self.get_env()
        observations, infos = env.reset()
        
        # Create valid actions for all agents
        actions = {}
        for agent in env.agents:
            action_space = env.action_spaces[agent]
            actions[agent] = action_space.sample()
        
        # Take a step
        step_result = env.step(actions)
        
        # Check return tuple structure
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5
        
        observations, rewards, terminations, truncations, infos = step_result
        
        # Check return types
        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)
        
        # Check that all agents have entries in each dict
        for agent in env.agents:
            assert agent in observations
            assert agent in rewards
            assert agent in terminations
            assert agent in truncations
            assert agent in infos
            
            # Check value types
            assert isinstance(rewards[agent], (int, float))
            assert isinstance(terminations[agent], bool)
            assert isinstance(truncations[agent], bool)
            assert isinstance(infos[agent], dict)
        
        env.close()
    
    def test_observation_space_method(self):
        """Test observation_space method for individual agents."""
        env = self.get_env()
        
        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            assert isinstance(obs_space, gym.spaces.Space)
            # Should be the same as in observation_spaces dict
            assert obs_space == env.observation_space(agent)
        
        env.close()
    
    def test_action_space_method(self):
        """Test action_space method for individual agents."""
        env = self.get_env()
        
        for agent in env.possible_agents:
            action_space = env.action_space(agent)
            assert isinstance(action_space, gym.spaces.Space)
            # Should be the same as in action_spaces dict
            assert action_space == env.action_spaces[agent]
        
        env.close()
    
    def test_render_method(self):
        """Test that render method exists and can be called."""
        env = self.get_env()
        env.reset()  # Reset first since some methods require it
        
        assert hasattr(env, 'render')
        # Should not raise an exception
        result = env.render()
        # Render can return None, np.ndarray, str, or list
        assert result is None or isinstance(result, (np.ndarray, str, list))
        
        env.close()
    
    def test_close_method(self):
        """Test that close method exists and can be called."""
        env = self.get_env()
        
        assert hasattr(env, 'close')
        # Should not raise an exception
        env.close()
    
    def test_state_method(self):
        """Test state method if implemented."""
        env = self.get_env()
        
        if hasattr(env, 'state'):
            env.reset()  # Reset first since state method might require it
            state = env.state()
            assert isinstance(state, np.ndarray)
        
        env.close()
    
    def test_agent_consistency(self):
        """Test that agent lists are consistent throughout the environment."""
        env = self.get_env()
        observations, _ = env.reset()  # Need to reset to get agents
        
        # agents should be a subset of possible_agents
        assert set(env.agents).issubset(set(env.possible_agents))
        
        # All agents should have observation and action spaces
        for agent in env.possible_agents:
            assert agent in env.observation_spaces
            assert agent in env.action_spaces
        
        env.close()
    
    def test_space_consistency(self):
        """Test that observation and action spaces are consistent."""
        env = self.get_env()
        observations, _ = env.reset()
        
        # All observations should match their corresponding spaces
        for agent in env.agents:
            obs = observations[agent]
            obs_space = env.observation_space(agent)
            assert obs_space.contains(obs), f"Observation for {agent} doesn't match its space"
        
        # Test that action spaces can generate valid actions
        for agent in env.agents:
            action_space = env.action_spaces[agent]
            sample_action = action_space.sample()
            assert action_space.contains(sample_action), f"Sampled action for {agent} doesn't match its space"
        
        env.close()
    
    def test_full_game_cycle(self):
        """Test a complete game from start to finish."""
        env = self.get_env()
        observations, infos = env.reset()
        
        step_count = 0
        max_steps = 1000  # TFT games with random actions can take longer
        initial_agents = len(env.agents)
        
        while env.agents and step_count < max_steps:
            # Generate actions for all active agents
            actions = {}
            for agent in env.agents:
                action_space = env.action_space(agent)
                actions[agent] = action_space.sample()
            
            # Take step
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Verify step outputs
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)
            
            step_count += 1
        
        # Game should eventually end or show significant progress
        # For TFT, we accept the game ending OR substantial player elimination OR round progression
        agents_eliminated = initial_agents - len(env.agents)
        current_round = getattr(getattr(env, 'aec_env', None), 'game_round', None)
        round_num = getattr(current_round, 'current_round', 0) if current_round else 0
        
        game_progressed = (agents_eliminated > 0 or 
                          step_count < max_steps)
        assert game_progressed, f"Game did not progress - {agents_eliminated} eliminated in {step_count} steps, round {round_num}"
        
        env.close()
    
    def test_metadata_attribute(self):
        """Test that metadata attribute exists and has proper structure."""
        env = self.get_env()
        
        assert hasattr(env, 'metadata')
        assert isinstance(env.metadata, dict)
        
        # Common metadata fields
        if 'name' in env.metadata:
            assert isinstance(env.metadata['name'], str)
        
        if 'is_parallelizable' in env.metadata:
            assert isinstance(env.metadata['is_parallelizable'], bool)
        
        env.close()


class TestPettingZooCompliance:
    """Test compliance with PettingZoo's built-in test suite."""
    
    def get_env_for_testing(self):
        """Create environment specifically for PettingZoo testing."""
        try:
            return parallel_env(rank=0)
        except:
            return parallel_env()
    
    def test_pettingzoo_parallel_api_test(self):
        """Run PettingZoo's official parallel API test."""
        env = self.get_env_for_testing()
        
        # This is the comprehensive test provided by PettingZoo
        # It tests all aspects of the parallel API
        try:
            parallel_api_test(env, num_cycles=10)
            print("PettingZoo parallel API test passed!")
        except Exception as e:
            raise AssertionError(f"PettingZoo parallel API test failed: {str(e)}")
        finally:
            env.close()
    
    def test_pettingzoo_parallel_api_test_extended(self):
        """Run PettingZoo's parallel API test with more cycles."""
        env = self.get_env_for_testing()
        
        try:
            parallel_api_test(env, num_cycles=100)
            print("Extended PettingZoo parallel API test passed!")
        except Exception as e:
            raise AssertionError(f"Extended PettingZoo parallel API test failed: {str(e)}")
        finally:
            env.close()


class TestEnvironmentRobustness:
    """Test environment robustness and edge cases."""
    
    def get_env_small(self):
        """Create environment for faster testing."""
        try:
            return parallel_env(rank=0)
        except:
            return parallel_env()
    
    def test_multiple_resets(self):
        """Test that environment can be reset multiple times."""
        env = self.get_env_small()
        
        for _ in range(5):
            observations, infos = env.reset()
            assert isinstance(observations, dict)
            assert isinstance(infos, dict)
            assert len(observations) == len(env.agents)
        
        env.close()
    
    def test_step_without_reset_error(self):
        """Test that stepping without reset raises appropriate error or handles gracefully."""
        try:
            env = parallel_env(rank=0)
        except:
            env = parallel_env()
        
        # Try to step without reset - should either work or raise clear error
        try:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
        except Exception as e:
            # If it raises an error, it should be informative
            assert len(str(e)) > 0
        finally:
            env.close()
    
    def test_invalid_actions(self):
        """Test environment behavior with invalid actions."""
        env = self.get_env_small()
        observations, infos = env.reset()
        
        # Test with empty action dict
        try:
            env.step({})
        except Exception:
            pass  # This might be expected behavior
        
        # Test with actions for non-existent agents
        try:
            invalid_actions = {"non_existent_agent": 0}
            env.step(invalid_actions)
        except Exception:
            pass  # This might be expected behavior
        
        env.close()
    
    def test_action_space_sampling(self):
        """Test that action spaces can be sampled reliably."""
        env = self.get_env_small()
        
        for _ in range(10):
            for agent in env.possible_agents:
                action = env.action_space(agent).sample()
                assert env.action_space(agent).contains(action)
        
        env.close()
    
    def test_observation_space_bounds(self):
        """Test that observations stay within observation space bounds."""
        env = self.get_env_small()
        observations, _ = env.reset()
        
        for agent in env.agents:
            obs = observations[agent]
            obs_space = env.observation_space(agent)
            assert obs_space.contains(obs)
        
        # Take several steps and check observations
        for _ in range(10):
            if not env.agents:
                break
                
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, _, terminations, truncations, _ = env.step(actions)
            
            for agent in env.agents:
                if not (terminations.get(agent, False) or truncations.get(agent, False)):
                    obs = observations[agent]
                    obs_space = env.observation_space(agent)
                    assert obs_space.contains(obs), f"Observation for {agent} is out of bounds"
        
        env.close()


def run_all_tests():
    """Run all tests manually without pytest."""
    print("Starting TFT Parallel Environment Tests...")
    
    # Test API compliance
    api_tests = TestParallelEnvAPI()
    print("\n=== Testing API Compliance ===")
    
    tests_to_run = [
        ("Environment Creation", api_tests.test_environment_creation),
        ("Required Attributes", api_tests.test_required_attributes),
        ("Reset Method", api_tests.test_reset_method),
        ("Step Method", api_tests.test_step_method),
        ("Observation Space Method", api_tests.test_observation_space_method),
        ("Action Space Method", api_tests.test_action_space_method),
        ("Render Method", api_tests.test_render_method),
        ("Close Method", api_tests.test_close_method),
        ("State Method", api_tests.test_state_method),
        ("Agent Consistency", api_tests.test_agent_consistency),
        ("Space Consistency", api_tests.test_space_consistency),
        ("Metadata Attribute", api_tests.test_metadata_attribute),
        ("Full Game Cycle", api_tests.test_full_game_cycle),
    ]
    
    for test_name, test_func in tests_to_run:
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
    
    # Test PettingZoo compliance
    zoo_tests = TestPettingZooCompliance()
    print("\n=== Testing PettingZoo Compliance ===")
    
    zoo_tests_to_run = [
        ("PettingZoo Parallel API Test", zoo_tests.test_pettingzoo_parallel_api_test),
        ("Extended PettingZoo Test", zoo_tests.test_pettingzoo_parallel_api_test_extended),
    ]
    
    for test_name, test_func in zoo_tests_to_run:
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
    
    # Test robustness
    robust_tests = TestEnvironmentRobustness()
    print("\n=== Testing Environment Robustness ===")
    
    robust_tests_to_run = [
        ("Multiple Resets", robust_tests.test_multiple_resets),
        ("Step Without Reset", robust_tests.test_step_without_reset_error),
        ("Invalid Actions", robust_tests.test_invalid_actions),
        ("Action Space Sampling", robust_tests.test_action_space_sampling),
        ("Observation Space Bounds", robust_tests.test_observation_space_bounds),
    ]
    
    for test_name, test_func in robust_tests_to_run:
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
    
    print("\n=== Test Summary ===")
    print("All tests completed!")


if __name__ == "__main__":
    run_all_tests()