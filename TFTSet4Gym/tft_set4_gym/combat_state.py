import threading


class CombatState:
    def __init__(self):
        self.reset()

    def reset(self):
        # From TFT-121 (champion & champion_functions globals)
        self.blue = []
        self.red = []
        self.que = []
        self.log = []
        self.MILLISECONDS = 0
        self.damage_dealt = []
        self.damage_dealt_teams = {'blue': 0, 'red': 0}
        self.galio_spawned = {'blue': False, 'red': False}

        # From ability.py & active.py globals
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
        _state.value = CombatState()
        return _state.value


def reset():
    _state.value = CombatState()
