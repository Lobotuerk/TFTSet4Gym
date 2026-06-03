import pytest
import time
import sys
import os

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env, RuntimeStats
import TFTSet4Gym.tft_set4_gym.config as config


def test_runtime_stats_standalone():
    """Test the standalone RuntimeStats class for metric accumulation and dictionary conversion."""
    stats = RuntimeStats()
    assert stats.step_count == 0
    assert stats.step_total == 0.0
    assert stats.turn_count == 0
    assert stats.turn_total == 0.0
    assert stats.combat_count == 0
    assert stats.combat_total == 0.0
    assert stats.game_start_time == 0.0

    stats.record_step(0.12)
    stats.record_step(0.08)
    assert stats.step_count == 2
    assert pytest.approx(stats.step_total) == 0.20

    stats.record_turn(1.5)
    assert stats.turn_count == 1
    assert stats.turn_total == 1.5

    stats.record_combat(0.4)
    stats.record_combat(0.6)
    assert stats.combat_count == 2
    assert pytest.approx(stats.combat_total) == 1.0

    # Set start time to measure game duration
    stats.game_start_time = time.perf_counter() - 3.0
    res = stats.to_dict()

    assert res["num_steps"] == 2
    assert pytest.approx(res["step_time_total_s"]) == 0.20
    assert pytest.approx(res["step_time_avg_s"]) == 0.10

    assert res["num_turns"] == 1
    assert pytest.approx(res["turn_time_total_s"]) == 1.5
    assert pytest.approx(res["turn_time_avg_s"]) == 1.5

    assert res["num_combats"] == 2
    assert pytest.approx(res["combat_time_total_s"]) == 1.0
    assert pytest.approx(res["combat_time_avg_s"]) == 0.5

    assert res["game_duration_s"] >= 3.0


def test_simulator_runtime_stats_initialization():
    """Test that runtime_stats is properly initialized on env creation and reset."""
    env = parallel_env()
    try:
        assert hasattr(env, "runtime_stats")
        assert isinstance(env.runtime_stats, RuntimeStats)
        assert env.runtime_stats.game_start_time > 0

        first_start = env.runtime_stats.game_start_time
        time.sleep(0.01)

        env.reset()
        assert env.runtime_stats.game_start_time > first_start
    finally:
        env.close()


def test_simulator_runtime_stats_stepping():
    """Test that runtime_stats counts and times step executions."""
    env = parallel_env()
    try:
        env.reset()
        assert env.runtime_stats.step_count == 0

        # Run 5 steps
        for _ in range(5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)

        assert env.runtime_stats.step_count == 5
        assert env.runtime_stats.step_total > 0.0
    finally:
        env.close()


def test_simulator_runtime_stats_turn_and_combat():
    """Test that a turn boundary registers turn and combat timings."""
    env = parallel_env()
    try:
        env.reset()
        assert env.runtime_stats.turn_count == 0
        assert env.runtime_stats.combat_count == 0

        # Complete exactly one full turn (config.ACTIONS_PER_TURN steps)
        for _ in range(config.ACTIONS_PER_TURN):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)

        # A turn has completed, so the turn count should be 1
        assert env.runtime_stats.turn_count == 1
        assert env.runtime_stats.turn_total > 0.0

        # Combats might run depending on whether it's a combat round.
        # But we should at least check that the values are computed correctly
        stats_dict = env.runtime_stats.to_dict()
        assert "turn_time_avg_s" in stats_dict
        assert "combat_time_avg_s" in stats_dict
    finally:
        env.close()


def test_simulator_runtime_stats_termination():
    """Test that on agent termination, runtime stats are embedded in the info dictionary."""
    env = parallel_env()
    try:
        env.reset()
        # Manually force termination of all agents to see if the infos dictionary is updated
        # By completing steps until some or all terminate, or forcing game over.
        # Let's simulate a situation where we check_dead or manually trigger termination.
        # Or we can simply call check_dead and set agent health to 0.
        for agent in env.agents:
            env.PLAYERS[agent].health = 0

        # Running another step or a full turn to trigger elimination.
        # Alternatively, we can verify that the step() code path for termination contains the expected keys.
        # Let's do a step with all dead players.
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Let's check if they are terminated
        # Since we set health to 0, at the end of the turn they will be killed.
        # Let's run steps until the end of the turn.
        remaining_steps = config.ACTIONS_PER_TURN - env.actions_taken
        for _ in range(remaining_steps):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check if anyone is terminated and has the runtime stats in their info dict
        terminated_any = False
        for agent, term in terminations.items():
            if term:
                terminated_any = True
                assert agent in infos
                assert "runtime_stats" in infos[agent]
                stats = infos[agent]["runtime_stats"]
                assert "game_duration_s" in stats
                assert "num_steps" in stats
                assert "num_turns" in stats
                assert "num_combats" in stats

        assert terminated_any, "At least some agents should have terminated"
    finally:
        env.close()
