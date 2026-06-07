import json
import os
import time
from collections import defaultdict


class GameRecorder:
    """Wraps a PettingZoo parallel environment and records the trajectory.

    Records the initial seed and every action taken at each step so the
    game can be deterministically replayed later.
    """

    def __init__(self, env):
        self.env = env
        self.seed = None
        self.steps = []

    def reset(self, seed=None, options=None):
        self.seed = seed
        self.steps = []
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        self.steps.append({
            "step_index": len(self.steps),
            "actions": {
                agent: action.tolist() if hasattr(action, "tolist") else list(action)
                for agent, action in actions.items()
            },
        })
        return self.env.step(actions)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def save_trajectory(self, filepath):
        trajectory = {
            "metadata": {
                "game_id": f"tft_trajectory_{time.strftime('%Y%m%d_%H%M%S')}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                "num_agents": len(getattr(self.env, 'possible_agents', [])),
                "num_steps": len(self.steps),
                "actions_per_turn": self.env.actions_taken + 1 if hasattr(self.env, 'actions_taken') else 10,
            },
            "seed": self.seed,
            "steps": self.steps,
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(trajectory, f, indent=2)
        return filepath


def load_trajectory(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def replay_trajectory(filepath, env_factory=None):
    from .tft_simulator import parallel_env as default_env_factory

    trajectory = load_trajectory(filepath)
    env_factory = env_factory or default_env_factory

    env = env_factory()
    observations, infos = env.reset(seed=trajectory["seed"])

    actions_taken = []
    for step_data in trajectory["steps"]:
        actions = step_data["actions"]
        if env.agents:
            observations, rewards, terminations, truncations, infos = env.step(actions)
            actions_taken.append(step_data)

    env.close()
    return observations, infos, actions_taken
