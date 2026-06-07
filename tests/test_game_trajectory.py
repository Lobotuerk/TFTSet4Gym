import json
import os
import tempfile
import numpy as np
import pytest

from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env
from TFTSet4Gym.tft_set4_gym.game_recorder import GameRecorder, load_trajectory, replay_trajectory


@pytest.mark.unit
def test_trajectory_save_and_replay_produces_identical_final_observation():
    num_steps = 5

    recorded_actions = []
    recorded_env = parallel_env()
    recorder = GameRecorder(recorded_env)
    obs1, _ = recorder.reset(seed=42)

    for step_i in range(num_steps):
        agents = recorder.env.agents[:]
        if not agents:
            break
        actions = {agent: recorder.env.action_space(agent).sample() for agent in agents}
        recorded_actions.append(actions)
        obs1, _, _, _, _ = recorder.step(actions)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        recorder.save_trajectory(tmp_path)

        loaded = load_trajectory(tmp_path)
        assert loaded["seed"] == 42
        assert len(loaded["steps"]) == len(recorded_actions)

        replay_env = parallel_env()
        replay_obs, _ = replay_env.reset(seed=loaded["seed"])

        for step_data in loaded["steps"]:
            if not replay_env.agents:
                break
            step_actions = step_data["actions"]
            replay_obs, _, _, _, _ = replay_env.step(step_actions)

        replay_env.close()

        assert set(obs1.keys()) == set(replay_obs.keys())
        for agent in obs1:
            assert np.array_equal(obs1[agent]["tensor"], replay_obs[agent]["tensor"]), \
                f"Observation mismatch for {agent}"
            assert np.array_equal(obs1[agent]["action_mask"], replay_obs[agent]["action_mask"]), \
                f"Action mask mismatch for {agent}"

    finally:
        os.unlink(tmp_path)


@pytest.mark.unit
def test_trajectory_records_seed_and_actions():
    env = parallel_env()
    recorder = GameRecorder(env)
    recorder.reset(seed=12345)

    agents = recorder.env.agents[:]
    actions = {agent: recorder.env.action_space(agent).sample() for agent in agents}
    recorder.step(actions)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        recorder.save_trajectory(tmp_path)
        loaded = load_trajectory(tmp_path)
        assert loaded["seed"] == 12345
        assert len(loaded["steps"]) == 1
        assert set(loaded["steps"][0]["actions"].keys()) == set(actions.keys())
    finally:
        os.unlink(tmp_path)


@pytest.mark.unit
def test_replay_trajectory_function():
    env = parallel_env()
    recorder = GameRecorder(env)
    recorder.reset(seed=99)

    for _ in range(3):
        agents = recorder.env.agents[:]
        if not agents:
            break
        actions = {agent: recorder.env.action_space(agent).sample() for agent in agents}
        recorder.step(actions)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        recorder.save_trajectory(tmp_path)
        final_obs, infos, steps = replay_trajectory(tmp_path)
        assert isinstance(final_obs, dict)
        assert isinstance(infos, dict)
    finally:
        os.unlink(tmp_path)
