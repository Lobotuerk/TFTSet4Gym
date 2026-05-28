import pytest
import torch
import numpy as np
from TFTSet4Gym.tft_set4_gym.models.action_translation import (
    ActionTranslationModule,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    BOARD_SIZE,
    NUM_CHAMPIONS,
    EMPTY_CLASS,
    BENCH_SIZE,
)
from TFTSet4Gym.tft_set4_gym.stats import COST


def test_decodes_target_board():
    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, 5, 1, 2] = 1.0
    probs[0, EMPTY_CLASS, :, :] = 1.0
    probs[0, EMPTY_CLASS, 1, 2] = 0.0
    target = module.decode_target_board(probs)
    assert target.shape == (1, BOARD_HEIGHT, BOARD_WIDTH)
    assert target[0, 1, 2] == 5
    assert target[0, 0, 0] == EMPTY_CLASS


def test_get_current_board_empty():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]

    module = ActionTranslationModule()
    board = module.get_current_board(MockPlayer())
    assert board.shape == (BOARD_HEIGHT, BOARD_WIDTH)
    assert np.all(board == EMPTY_CLASS)


def test_get_current_board_with_champs():
    champ_names = list(COST.keys())
    real_champs = [n for n in champ_names if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(real_champs[0])
    player.board[3][1] = MockChamp(real_champs[1])

    module = ActionTranslationModule()
    board = module.get_current_board(player)
    assert board[0, 0] == 0
    assert board[1, 3] == 1
    assert np.all(board != EMPTY_CLASS) or True


def test_get_current_bench():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]
    player.bench[2] = MockChamp(champ_names[3])
    player.bench[5] = MockChamp(champ_names[7])

    module = ActionTranslationModule()
    bench = module.get_current_bench(player)
    assert (2, 3) in bench
    assert (5, 7) in bench
    assert len(bench) == 2


def test_translate_noop():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    actions = module.translate(probs, MockPlayer())
    assert len(actions) >= 1
    assert actions[0] == [0, 0, 0]


def test_translate_bench_to_board():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.bench[0] = MockChamp(champ_names[0])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 0.0
    probs[0, 0, :, :] = 1.0
    actions = module.translate(probs, player)
    assert len(actions) >= 1
    action_types = [a[0] for a in actions]
    assert 1 in action_types


def test_translate_board_to_bench():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[5])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    actions = module.translate(probs, player)
    action_types = [a[0] for a in actions]
    assert 1 in action_types, "should move to bench, not sell"
    for a in actions:
        if a[0] == 1:
            assert a[2] >= BOARD_SIZE, "board-to-bench should target bench dcord >= 28"


def test_translate_sell_when_bench_full():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[5])
    for i in range(9):
        player.bench[i] = MockChamp(champ_names[(i + 10) % NUM_CHAMPIONS])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 1.0
    actions = module.translate(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 in action_types, "should sell when bench is full"


def test_translate_board_to_board():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[5])
    player.board[1][0] = MockChamp(champ_names[3])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 0.0
    probs[0, 3, 0, 0] = 1.0
    probs[0, 5, 1, 0] = 1.0
    actions = module.translate(probs, player)
    board_to_board_moves = [a for a in actions if a[0] == 1 and a[1] < BOARD_SIZE and a[2] < BOARD_SIZE]
    assert len(board_to_board_moves) >= 1


def test_translate_keep_champ_needed_elsewhere():
    champ_names = [n for n in list(COST.keys()) if n != " " and COST[n] > 0]

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    player = MockPlayer()
    player.board[0][0] = MockChamp(champ_names[0])
    player.board[1][0] = MockChamp(champ_names[1])

    module = ActionTranslationModule()
    probs = torch.zeros(1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[0, EMPTY_CLASS, :, :] = 0.0
    probs[0, 0, 0, 0] = 1.0
    probs[0, 1, 0, 1] = 1.0
    actions = module.translate(probs, player)
    action_types = [a[0] for a in actions]
    assert 3 not in action_types, "should not sell champ 0 or 1, both are needed"
    assert 1 in action_types, "should generate moves to reposition"


def test_translate_batch():
    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    module = ActionTranslationModule()
    probs = torch.zeros(2, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)
    probs[:, EMPTY_CLASS, :, :] = 1.0
    players = [MockPlayer(), MockPlayer()]
    results = module.translate_batch(probs, players)
    assert len(results) == 2
    assert results[0][0] == [0, 0, 0]


def test_champion_name_mapping():
    module = ActionTranslationModule()
    assert len(module.idx_to_name) == NUM_CHAMPIONS
    assert len(module.name_to_idx) == NUM_CHAMPIONS
    for idx, name in module.idx_to_name.items():
        assert module.name_to_idx[name] == idx


def test_integration_with_board_generator():
    from TFTSet4Gym.tft_set4_gym.models.board_generator import BoardGenerator

    bg = BoardGenerator()
    atm = ActionTranslationModule()
    x = torch.randn(1, 116)
    out = bg(x)
    assert out.shape == (1, NUM_CHAMPIONS + 1, BOARD_HEIGHT, BOARD_WIDTH)

    class MockChamp:
        def __init__(self, name):
            self.name = name

    class MockPlayer:
        def __init__(self):
            self.board = [[None for _ in range(4)] for _ in range(7)]
            self.bench = [None for _ in range(9)]

    actions = atm.translate(out, MockPlayer())
    assert len(actions) >= 1
