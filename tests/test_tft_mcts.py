import pytest
from TFTSet4Gym.tft_set4_gym.models.tft_mcts import (
    TFTMove,
    TFTState,
    REROLL_ACTION,
    LEVEL_ACTION,
)


class TestTFTMove:
    def test_move_equality(self):
        move1 = TFTMove(shop_index=0)
        move2 = TFTMove(target_1=0)
        assert move1 == move2

        move3 = TFTMove(board_index=1)
        move4 = TFTMove(target_2=1)
        assert move3 == move4

        move5 = TFTMove(sell_index=2)
        move6 = TFTMove(target_1=2)
        assert move5 == move6


class TestTFTState:
    def test_state_creation_from_env(self):
        class MockEnv:
            def _get_obs(self):
                return {"board": "test"}

        env = MockEnv()
        state = TFTState(env)
        assert state.observation is not None

    def test_state_actions_to_try(self):
        class MockEnv:
            def _get_obs(self):
                return {}

        env = MockEnv()
        state = TFTState(env)
        actions = state.actions_to_try()
        action_types = [a.action_type for a in actions]
        assert REROLL_ACTION in action_types
        assert LEVEL_ACTION in action_types
