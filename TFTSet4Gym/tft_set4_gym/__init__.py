from .tft_simulator import TFT_Simulator, parallel_env
from .game_recorder import GameRecorder, load_trajectory, replay_trajectory

__all__ = ['TFT_Simulator', 'parallel_env', 'GameRecorder', 'load_trajectory', 'replay_trajectory']

# Initialize observation schema and update config
try:
    from .config import update_observation_size_from_schema
    update_observation_size_from_schema()
except ImportError:
    pass  # Fallback if schema system not available

# register(
#     id="TFT_Set4-v0", 
#     entry_point="TFTSet4Gym.tft_set4_gym.tft_simulator:TFT_Simulator"
# )
