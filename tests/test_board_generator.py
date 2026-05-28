import pytest
import torch
import numpy as np
from TFTSet4Gym.tft_set4_gym.models.board_generator import (
    BoardGenerator,
    NUM_CHAMPIONS,
    NUM_CLASSES,
    BOARD_HEIGHT,
    BOARD_WIDTH,
)


def test_output_shape():
    model = BoardGenerator()
    batch_size = 4
    x = torch.randn(batch_size, 116)
    out = model(x)
    assert out.shape == (batch_size, NUM_CLASSES, BOARD_HEIGHT, BOARD_WIDTH)


def test_softmax_validity():
    model = BoardGenerator()
    x = torch.randn(1, 116)
    out = model(x)
    for h in range(BOARD_HEIGHT):
        for w in range(BOARD_WIDTH):
            probs = out[0, :, h, w]
            assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
            assert torch.all(probs >= 0.0)
            assert torch.all(probs <= 1.0)


def test_champion_availability_integration():
    from TFTSet4Gym.tft_set4_gym.champion_availability import (
        encode_champion_availability,
        NUM_CHAMPIONS as N_CHAMPS,
    )
    from TFTSet4Gym.tft_set4_gym.observation_builder import ObservationBuilder
    from TFTSet4Gym.tft_set4_gym.player import Player
    from TFTSet4Gym.tft_set4_gym.stats import COST
    from TFTSet4Gym.tft_set4_gym.observation_schema import update_observation_size_in_config

    update_observation_size_in_config()

    class MockPool:
        def update_pool(self, name, amount):
            pass

    class MockChampion:
        def __init__(self, name, stars=1, chosen=False):
            self.name = name
            self.stars = stars
            self.chosen = chosen

    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    aatrox = MockChampion("aatrox", stars=2, chosen=False)
    player.board[0][0] = aatrox

    obs_dict = builder.build_observation("player_0", player)
    avail = encode_champion_availability(obs_dict["tensor"])

    model = BoardGenerator()
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(avail).float().unsqueeze(0)
        out = model(inp)

    assert out.shape == (1, N_CHAMPS + 1, 4, 7)


def test_gradient_flow():
    model = BoardGenerator()
    x = torch.randn(2, 116)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.any(param.grad != 0.0), f"Parameter {name} has zero gradient"


def test_different_batch_sizes():
    model = BoardGenerator()
    for batch_size in [1, 2, 8, 16]:
        x = torch.randn(batch_size, 116)
        out = model(x)
        assert out.shape == (batch_size, NUM_CLASSES, BOARD_HEIGHT, BOARD_WIDTH)
