import threading


class _CombatState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.kennen_hits = []
        self.lulu_targeted = []
        self.morgana_MR_list = []
        self.riven_counter = []
        self.riven_identifier_list = []
        self.vi_armor_list = []
        self.yone_list = []
        self.yone_checking = False
        self.jhin_shots = []
        self.kalista_targets = []
        self.vayne_targets = []
        self.zed_counter = []


_state = threading.local()


def get_state():
    try:
        return _state.value
    except AttributeError:
        _state.value = _CombatState()
        return _state.value
