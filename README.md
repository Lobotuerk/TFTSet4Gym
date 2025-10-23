# TFT Set 4 Gymnasium Environment

A PettingZoo-compatible environment for Teamfight Tactics Set 4, providing a complete simulation of TFT mechanics for reinforcement learning research.

## Features

- **Complete TFT Set 4 Simulation**: All champions, items, and synergies from TFT Set 4
- **PettingZoo Compatible**: Standard multi-agent RL environment interface
- **Gymnasium Integration**: Compatible with modern RL libraries
- **Combat Simulation**: Detailed combat mechanics and interactions
- **Champion Abilities**: Full implementation of champion abilities and effects
- **Item System**: Complete item crafting and effect system
- **Multi-agent Support**: 8-player games with proper elimination mechanics

## Installation

### From Source
```bash
git clone https://github.com/Lobotuerk/TFT-Set4-Gym.git
cd TFT-Set4-Gym
pip install -e .
```

### From PyPI (coming soon)
```bash
pip install tft-set4-gym
```

## Quick Start

### Basic Usage
```python
from tft_set4_gym import parallel_env

# Create environment
env = parallel_env()

# Reset environment
observations, infos = env.reset()

# Game loop
while env.agents:
    # Sample random actions for all agents
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Step environment
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Remove terminated/truncated agents
    env.agents = [agent for agent in env.agents 
                  if not (terminations[agent] or truncations[agent])]

env.close()
```

### With Stable Baselines 3
```python
from tft_set4_gym import TFTSingleAgentWrapper
from stable_baselines3 import PPO

# Create single-agent wrapper for SB3
env = TFTSingleAgentWrapper()

# Create and train model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

## Environment Details

### Observation Space
```python
observation_space = Dict({
    'tensor': Box(0.0, 55.0, (5152,), float64),    # Game state vector
    'action_mask': Box(0, 1, (54,), int8)          # Valid action mask
})
```

### Action Space
```python
action_space = MultiDiscrete([7, 37, 10])
# [action_type, target/item_id, position]
```

### Rewards
- **Placement-based**: Higher placement = higher reward
- **Winner**: 250 points
- **Elimination order**: (8 - placement) × 25 points
- **Example**: 1st place = 250, 2nd place = 175, ..., 8th place = 25

## Game Mechanics

### Champions
- **58 unique champions** from TFT Set 4
- **Star levels**: 1⭐, 2⭐, 3⭐ upgrades
- **Abilities**: Unique champion abilities with mana system
- **Origins & Classes**: Synergy bonuses (Warlord, Mystic, etc.)

### Items
- **Component items**: Basic items that can be combined
- **Completed items**: Powerful items with unique effects
- **Item crafting**: Combine components to create completed items
- **Spatial items**: Items that affect positioning

### Combat
- **Auto-chess combat**: Champions fight automatically
- **Positioning matters**: Frontline, backline, corner positioning
- **Damage calculation**: Complex damage, armor, and resistance system
- **Crowd control**: Stuns, fears, charms, and other effects

### Economy
- **Gold management**: Income, interest, and spending decisions
- **Shop system**: Rolling for champions, costs based on level
- **Experience**: Leveling up increases shop pool and board size

## Advanced Usage

### Custom Configurations
```python
from tft_set4_gym import parallel_env

# Custom game settings
env = parallel_env(
    num_players=6,          # 6-player game instead of 8
    max_rounds=30,          # Shorter games
    debug_mode=True         # Enable debug logging
)
```

### Environment Wrappers
```python
from tft_set4_gym.wrappers import (
    RewardShapingWrapper,
    ActionMaskingWrapper,
    ObservationWrapper
)

env = parallel_env()
env = RewardShapingWrapper(env)      # Add intermediate rewards
env = ActionMaskingWrapper(env)      # Enforce action masking
env = ObservationWrapper(env)        # Custom observation format
```

## API Reference

### Core Environment
- `parallel_env()`: Create parallel multi-agent environment
- `TFTSingleAgentWrapper`: Single-agent wrapper for SB3 compatibility

### Key Classes
- `TFT_Simulator`: Main game simulation engine
- `Player`: Player state and actions
- `Champion`: Individual champion with abilities
- `Game_Round`: Round management and progression
- `Observation`: Environment observation generation

### Utilities
- `utils.py`: Helper functions for game state analysis
- `config.py`: Configuration constants and settings

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black tft_set4_gym/
isort tft_set4_gym/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Compatibility

- **Python**: 3.8+
- **PettingZoo**: 1.24+
- **Gymnasium**: 0.29+
- **NumPy**: 1.21+

## Performance

- **Environment speed**: ~50-200 FPS depending on hardware
- **Memory usage**: ~200MB per environment
- **Vectorization**: Supports multiple parallel environments

## Research Applications

This environment is designed for:
- **Multi-agent reinforcement learning** research
- **Game AI** development
- **Strategic decision making** studies
- **Curriculum learning** experiments
- **Meta-learning** research

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{tft_set4_gym,
  title={TFT Set 4 Gymnasium Environment},
  author={Lobotuerk},
  url={https://github.com/Lobotuerk/TFT-Set4-Gym},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Riot Games** for creating Teamfight Tactics
- **PettingZoo** team for the multi-agent RL framework
- **Gymnasium** team for the RL environment standards
- **TFT community** for game mechanics documentation

## Support

- **Issues**: [GitHub Issues](https://github.com/Lobotuerk/TFT-Set4-Gym/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lobotuerk/TFT-Set4-Gym/discussions)
- **Documentation**: [Wiki](https://github.com/Lobotuerk/TFT-Set4-Gym/wiki)