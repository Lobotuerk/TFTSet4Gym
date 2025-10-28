from .tft_simulator import TFT_Simulator, parallel_env, env

__all__ = ['TFT_Simulator', 'parallel_env', 'env']

# Initialize observation schema and update config
try:
    from .config import update_observation_size_from_schema
    update_observation_size_from_schema()
except ImportError:
    pass  # Fallback if schema system not available

# register(
#     id="TFT_Set4-v0", 
#     entry_point="tft_set4_gym.tft_simulator:TFT_Simulator"
# )
