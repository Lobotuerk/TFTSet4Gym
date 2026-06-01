REROLL_ACTION = 4
LEVEL_ACTION = 5


class TFTMove:
    REVERSE_MAP = {
        'shop_index': 'target_1',
        'board_index': 'target_2',
        'sell_index': 'target_1',
        'reroll': 'action_type',
        'level': 'action_type',
    }

    def __init__(self, action_type=None, target_1=None, target_2=None, **kwargs):
        self.action_type = action_type
        self.target_1 = target_1
        self.target_2 = target_2

        for key, value in kwargs.items():
            canonical = self.REVERSE_MAP.get(key, key)
            setattr(self, canonical, value)

    def __eq__(self, other):
        if not isinstance(other, TFTMove):
            return NotImplemented
        return (self.action_type == other.action_type
                and self.target_1 == other.target_1
                and self.target_2 == other.target_2)

    def __hash__(self):
        return hash((self.action_type, self.target_1, self.target_2))

    def __repr__(self):
        return (f"TFTMove(action_type={self.action_type}, "
                f"target_1={self.target_1}, target_2={self.target_2})")


class TFTState:
    def __init__(self, env=None):
        self.env = env
        self.observation = None
        if env is not None:
            self.observation = env._get_obs() if hasattr(env, '_get_obs') else None

    def actions_to_try(self):
        return [
            TFTMove(action_type=REROLL_ACTION),
            TFTMove(action_type=LEVEL_ACTION),
        ]
