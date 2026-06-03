import threading


class CombatState:
    def __init__(self):
        self.blue = []
        self.red = []
        self.que = []
        self.log = []
        self.MILLISECONDS = 0
        self.damage_dealt = []
        self.damage_dealt_teams = {'blue': 0, 'red': 0}
        self.galio_spawned = {'blue': False, 'red': False}


_local = threading.local()


def get_state():
    try:
        return _local.state
    except AttributeError:
        _local.state = CombatState()
        return _local.state


def reset():
    _local.state = CombatState()
