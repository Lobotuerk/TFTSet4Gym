import pytest
import numpy as np
from TFTSet4Gym.tft_set4_gym.observation_builder import ObservationBuilder, ObservationManagerMixin
from TFTSet4Gym.tft_set4_gym.player import Player, encoded_list
from TFTSet4Gym.tft_set4_gym.stats import COST
from TFTSet4Gym.tft_set4_gym.observation_builder import COST_INDEX, ITEM_BUILD_INDEX
from unittest.mock import MagicMock


class MockPool:
    def __init__(self):
        pass
    def update_pool(self, name, amount):
        pass


class MockChampion:
    def __init__(self, name, stars=1, chosen=False, items=None, origin=None):
        self.name = name
        self.stars = stars
        self.chosen = chosen
        self.items = items or []
        self.origin = origin or []


@pytest.mark.unit
@pytest.mark.observation
def test_observation_builder_init():
    builder = ObservationBuilder()
    assert builder.current_player_schema is not None
    assert builder.action_mask_schema is not None


@pytest.mark.unit
@pytest.mark.observation
def test_build_observation_empty_player():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    player.health = 100
    player.level = 1
    player.gold = 0

    obs = builder.build_observation("player_0", player)

    assert "tensor" in obs
    assert "action_mask" in obs
    assert obs["tensor"].shape == (builder.current_player_schema.total_size,)

    # Check player_state field
    player_state_slice = builder.current_player_schema.get_field_slice("player_state")
    player_state = obs["tensor"][player_state_slice].reshape((7,))
    assert player_state[0] == 100.0  # health


@pytest.mark.unit
@pytest.mark.observation
def test_build_observation_with_champions():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    aatrox = MockChampion('aatrox', stars=2, chosen=False, origin=['cultist', 'vanguard'])
    player.board[0][0] = aatrox

    ahri = MockChampion('ahri', stars=1, chosen=True, origin=['spirit', 'mage'])
    player.bench[0] = ahri

    obs = builder.build_observation("player_0", player)

    # Verify board field
    board_slice = builder.current_player_schema.get_field_slice("board")
    board = obs["tensor"][board_slice].reshape((28, 122))

    # aatrox at slot 0 (x=0, y=0)
    aatrox_slot = board[0]
    # First 32 dims should be the champion index
    aatrox_idx = COST_INDEX["aatrox"] - 1
    assert aatrox_slot[0] == aatrox_idx

    # Star level at index 120
    assert aatrox_slot[120] == 2.0

    # Chosen flag at index 121
    assert aatrox_slot[121] == 0.0

    # Verify bench_champions field
    bench_slice = builder.current_player_schema.get_field_slice("bench_champions")
    bench = obs["tensor"][bench_slice].reshape((9, 122))

    ahri_slot = bench[0]
    ahri_idx = COST_INDEX["ahri"] - 1
    assert ahri_slot[0] == ahri_idx
    assert ahri_slot[120] == 1.0  # star level
    assert ahri_slot[121] == 1.0  # chosen flag


@pytest.mark.unit
@pytest.mark.observation
def test_full_board_and_bench():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    champions = list(COST.keys())[1:29]
    for i in range(7):
        for j in range(4):
            idx = i * 4 + j
            player.board[i][j] = MockChampion(champions[idx], stars=(idx % 3) + 1, origin=['cultist'])

    bench_champs = list(COST.keys())[29:38]
    for i in range(9):
        player.bench[i] = MockChampion(bench_champs[i])

    obs = builder.build_observation("player_0", player)

    assert obs["tensor"].shape == (builder.current_player_schema.total_size,)


@pytest.mark.unit
@pytest.mark.observation
def test_shop_vector():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    # Create mock shop vector
    shop_vector = np.zeros(59)
    # Put aatrox in shop slot 0
    aatrox_idx = COST_INDEX["aatrox"] - 1
    player.shop_elems = np.array([aatrox_idx, -1, -1, -1, -1])
    shop_vector[58] = 0.0  # chosen index

    obs = builder.build_observation("player_0", player, shop_vector=shop_vector)

    shop_slice = builder.current_player_schema.get_field_slice("shop")
    shop = obs["tensor"][shop_slice].reshape((5, 32))
    assert shop[0, 0] == aatrox_idx

    shop_chosen_slice = builder.current_player_schema.get_field_slice("shop_chosen")
    shop_chosen = obs["tensor"][shop_chosen_slice]
    assert shop_chosen[0] == 0.0


@pytest.mark.unit
@pytest.mark.observation
def test_get_set_field():
    builder = ObservationBuilder()
    obs = np.zeros(builder.current_player_schema.total_size)

    # Test setting and getting player_state
    gold_val = np.ones((7,)) * 50
    obs = builder.set_field_in_observation(obs, "player_state", gold_val)

    retrieved = builder.get_field_from_observation(obs, "player_state")
    assert np.allclose(retrieved, gold_val)

    # Test setting and getting board
    board_val = np.random.rand(28, 122)
    obs = builder.set_field_in_observation(obs, "board", board_val)

    retrieved_board = builder.get_field_from_observation(obs, "board")
    assert np.allclose(retrieved_board, board_val)


@pytest.mark.unit
@pytest.mark.observation
def test_edge_case_unrecognized_champion():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    player.board = "not a list of encoded_lists"

    obs = builder.build_observation("player_0", player)
    assert "tensor" in obs

    board_slice = builder.current_player_schema.get_field_slice("board")
    assert np.all(obs["tensor"][board_slice] == 0)


@pytest.mark.unit
@pytest.mark.observation
def test_action_mask():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    player.shop_mask = np.array([1, 0, 1, 0, 1], dtype=np.int8)

    obs = builder.build_observation("player_0", player)

    assert "action_mask" in obs
    assert np.array_equal(obs["action_mask"][:5], player.shop_mask)
    assert np.all(obs["action_mask"][5:] == 1)


@pytest.mark.unit
@pytest.mark.observation
def test_observation_manager_mixin():
    class TestPlayer(ObservationManagerMixin):
        def __init__(self):
            super().__init__()
            self.gold = 10

        def get_observation_for_field(self, field_name):
            if field_name == "player_state":
                return np.ones((7,)) * self.gold
            return np.zeros((7,))

    player = TestPlayer()
    assert hasattr(player, "_obs_builder")
    assert isinstance(player._obs_builder, ObservationBuilder)

    state_obs = player.get_observation_for_field("player_state")
    assert np.all(state_obs == 10)

    with pytest.raises(NotImplementedError):
        class IncompletePlayer(ObservationManagerMixin):
            pass
        p = IncompletePlayer()
        p.get_observation_for_field("player_state")


@pytest.mark.unit
@pytest.mark.observation
def test_convenience_functions():
    from TFTSet4Gym.tft_set4_gym.observation_builder import build_observation, get_field_slice as obs_get_field_slice, \
        get_field_value_from_obs, set_field_value_in_obs, update_config_observation_size
    from TFTSet4Gym.tft_set4_gym import config

    pool = MockPool()
    player = Player(pool, 0)
    obs_dict = build_observation("player_0", player)
    assert "tensor" in obs_dict

    s = obs_get_field_slice("player_state")
    assert isinstance(s, slice)

    original_size = config.OBSERVATION_SIZE
    new_size = update_config_observation_size()
    assert new_size > 0
    assert config.OBSERVATION_SIZE == new_size

    obs = obs_dict["tensor"]
    gold_val = np.ones((7,)) * 75
    obs = set_field_value_in_obs(obs, "player_state", gold_val)
    retrieved = get_field_value_from_obs(obs, "player_state")
    assert np.allclose(retrieved, gold_val)


@pytest.mark.unit
@pytest.mark.observation
def test_schema_total_size():
    """Verify the total observation size matches the spec.

    The spec lists 28,946 total which includes the action_mask (54 dims).
    The current_player schema alone is 28,892.
    """
    builder = ObservationBuilder()
    current_player_size = 28892
    action_mask_size = 54
    expected_total = current_player_size + action_mask_size

    assert builder.current_player_schema.total_size == current_player_size
    assert builder.action_mask_schema.total_size == action_mask_size
    assert builder.current_player_schema.total_size + builder.action_mask_schema.total_size == expected_total


@pytest.mark.unit
@pytest.mark.observation
def test_bench_items():
    """Test bench items encoding."""
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    # Add items to item bench
    player.item_bench[0] = 'bloodthirster'
    player.item_bench[1] = 'bf_sword'

    obs = builder.build_observation("player_0", player)

    bench_items_slice = builder.current_player_schema.get_field_slice("bench_items")
    bench_items = obs["tensor"][bench_items_slice].reshape((10, 24))

    # First slot should have bloodthirster index
    from TFTSet4Gym.tft_set4_gym.item_stats import item_builds, uncraftable_items
    expected_idx = len(uncraftable_items) + ITEM_BUILD_INDEX['bloodthirster']
    assert bench_items[0, 0] == expected_idx


@pytest.mark.unit
@pytest.mark.observation
def test_team_traits():
    """Test team traits encoding."""
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    player.team_tiers = {'assassin': 2, 'mage': 1}
    player.team_composition = {'assassin': 3, 'mage': 2}

    obs = builder.build_observation("player_0", player)

    team_traits_slice = builder.current_player_schema.get_field_slice("team_traits")
    team_traits = obs["tensor"][team_traits_slice]

    # assassin is at index 11 in TRAIT_LIST, tier 2 of max 3 = 0.667
    assassin_idx = 11
    assert abs(team_traits[assassin_idx] - 2.0/3.0) < 0.01


@pytest.mark.unit
@pytest.mark.observation
def test_opponent_boards():
    """Test opponent board building."""
    builder = ObservationBuilder()
    pool = MockPool()

    # Create current player
    player = Player(pool, 0)
    player.board = [encoded_list(4, lambda x: None) for _ in range(7)]

    # Create opponent
    opp = Player(pool, 1)
    opp.board = [encoded_list(4, lambda x: None) for _ in range(7)]
    opp.board[0][0] = MockChampion('aatrox', stars=2, origin=['cultist', 'vanguard'])
    opp.health = 80
    opp.gold = 15
    opp.level = 3
    opp.win_streak = 2

    players = {"player_0": player, "player_1": opp}

    obs_dict = builder.build_full_observation("player_0", player, players)
    obs = obs_dict["tensor"]

    opp_board_slice = builder.current_player_schema.get_field_slice("opponent_boards")
    opp_board = obs[opp_board_slice].reshape((7, 28, 122))

    # Opponent at index 0 should have aatrox at slot 0
    assert opp_board[0, 0, 0] == COST_INDEX["aatrox"] - 1

    opp_info_slice = builder.current_player_schema.get_field_slice("opponent_info")
    opp_info = obs[opp_info_slice].reshape((7, 4))

    assert opp_info[0, 0] == 80.0  # health
    assert opp_info[0, 1] == 15.0  # gold
    assert opp_info[0, 2] == 3.0   # level
    assert opp_info[0, 3] == 2.0   # streak


@pytest.mark.unit
@pytest.mark.observation
def test_empty_slot_handling():
    """Test that empty board/bench slots produce zero vectors."""
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    # Leave board and bench empty
    obs = builder.build_observation("player_0", player)

    board_slice = builder.current_player_schema.get_field_slice("board")
    board = obs["tensor"][board_slice].reshape((28, 122))
    assert np.all(board == 0)

    bench_slice = builder.current_player_schema.get_field_slice("bench_champions")
    bench = obs["tensor"][bench_slice].reshape((9, 122))
    assert np.all(bench == 0)
