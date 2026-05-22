import pytest
import numpy as np
from TFTSet4Gym.tft_set4_gym.observation_builder import ObservationBuilder, ObservationManagerMixin
from TFTSet4Gym.tft_set4_gym.player import Player, encoded_list, encode_champ_object
from TFTSet4Gym.tft_set4_gym.stats import COST
from unittest.mock import MagicMock

class MockPool:
    def __init__(self):
        pass
    def update_pool(self, name, amount):
        pass

class MockChampion:
    def __init__(self, name, stars=1, chosen=False):
        self.name = name
        self.stars = stars
        self.chosen = chosen

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
    
    # Initialize basic player state
    player.health = 100
    player.level = 1
    player.gold = 0
    
    obs = builder.build_observation("player_0", player)
    
    assert "tensor" in obs
    assert "action_mask" in obs
    assert obs["tensor"].shape == (builder.current_player_schema.total_size,)
    
    # Check that health is set correctly in the observation
    health_slice = builder.current_player_schema.get_field_slice("health")
    # Health is broadcased to (1, 4, 7) then flattened
    expected_health = np.ones((1, 4, 7)) * 100
    assert np.allclose(obs["tensor"][health_slice], expected_health.flatten())

@pytest.mark.unit
@pytest.mark.observation
def test_build_observation_with_champions():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)
    
    # Add a champion to the board
    # board is list of 7 encoded_list(4, encode_champ_object)
    # let's put 'aatrox' at (0, 0)
    aatrox = MockChampion('aatrox', stars=2, chosen=False)
    player.board[0][0] = aatrox
    
    # Add a champion to the bench
    ahri = MockChampion('ahri', stars=1, chosen=True)
    player.bench[0] = ahri
    
    obs = builder.build_observation("player_0", player)
    
    # Verify board champions
    board_champ_slice = builder.current_player_schema.get_field_slice("board_champions")
    board_champs = obs["tensor"][board_champ_slice].reshape((58, 4, 7))
    
    # aatrox index in COST
    aatrox_idx = list(COST.keys()).index('aatrox') - 1
    assert board_champs[aatrox_idx, 0, 0] == 1.0
    
    # Verify board stars
    board_star_slice = builder.current_player_schema.get_field_slice("board_stars")
    board_stars = obs["tensor"][board_star_slice].reshape((1, 4, 7))
    assert board_stars[0, 0, 0] == 2.0
    
    # Verify bench
    bench_champ_slice = builder.current_player_schema.get_field_slice("bench_champions")
    bench_champs = obs["tensor"][bench_champ_slice].reshape((58, 4, 7))
    ahri_idx = list(COST.keys()).index('ahri') - 1
    # ahri is on bench[0]
    # ObservationBuilder sums bench champions and broadcasts to (4, 7)
    assert np.all(bench_champs[ahri_idx] == 1.0)

@pytest.mark.unit
@pytest.mark.observation
def test_full_board_and_bench():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)
    
    # Fill board
    champions = list(COST.keys())[1:29] # Get 28 champion names
    for i in range(7):
        for j in range(4):
            idx = i * 4 + j
            player.board[i][j] = MockChampion(champions[idx], stars=(idx % 3) + 1)
            
    # Fill bench
    bench_champs = list(COST.keys())[29:38] # Get 9 more
    for i in range(9):
        player.bench[i] = MockChampion(bench_champs[i])
        
    obs = builder.build_observation("player_0", player)
    
    assert obs["tensor"].shape == (builder.current_player_schema.total_size,)
    # Just ensure no errors occurred during building

@pytest.mark.unit
@pytest.mark.observation
def test_shop_vector():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)
    
    # Create a mock shop vector (62, 4, 7)
    shop_vector = np.random.rand(62, 4, 7)
    
    obs = builder.build_observation("player_0", player, shop_vector=shop_vector)
    
    shop_champ_slice = builder.current_player_schema.get_field_slice("shop_champions")
    shop_champs = obs["tensor"][shop_champ_slice].reshape((58, 4, 7))
    assert np.allclose(shop_champs, shop_vector[0:58])
    
    shop_chosen_slice = builder.current_player_schema.get_field_slice("shop_chosen")
    shop_chosen = obs["tensor"][shop_chosen_slice].reshape((1, 4, 7))
    assert np.allclose(shop_chosen, shop_vector[58:59])

@pytest.mark.unit
@pytest.mark.observation
def test_get_set_field():
    builder = ObservationBuilder()
    obs = np.zeros(builder.current_player_schema.total_size)
    
    # Test setting and getting gold
    gold_val = np.ones((1, 4, 7)) * 50
    obs = builder.set_field_in_observation(obs, "gold", gold_val)
    
    retrieved_gold = builder.get_field_from_observation(obs, "gold")
    assert np.allclose(retrieved_gold, gold_val)
    
    # Test setting and getting board champions
    board_champs_val = np.random.rand(58, 4, 7)
    obs = builder.set_field_in_observation(obs, "board_champions", board_champs_val)
    
    retrieved_board = builder.get_field_from_observation(obs, "board_champions")
    assert np.allclose(retrieved_board, board_champs_val)

@pytest.mark.unit
@pytest.mark.observation
def test_edge_case_unrecognized_champion():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)
    
    player.board = "not a list of encoded_lists"
    
    # This should trigger the try-except in _build_tensor_observation and print a warning
    # but still return an observation (though board fields will be zero)
    obs = builder.build_observation("player_0", player)
    assert "tensor" in obs
    
    board_champ_slice = builder.current_player_schema.get_field_slice("board_champions")
    assert np.all(obs["tensor"][board_champ_slice] == 0)

@pytest.mark.unit
@pytest.mark.observation
def test_action_mask():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)
    
    player.shop_mask = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    
    obs = builder.build_observation("player_0", player)
    
    assert "action_mask" in obs
    # action_mask schema is (54,)
    # first 5 should match shop_mask
    assert np.array_equal(obs["action_mask"][:5], player.shop_mask)
    assert np.all(obs["action_mask"][5:] == 1) # default is 1

@pytest.mark.unit
@pytest.mark.observation
def test_observation_manager_mixin():
    class TestPlayer(ObservationManagerMixin):
        def __init__(self):
            super().__init__()
            self.gold = 10
            
        def get_observation_for_field(self, field_name):
            if field_name == "gold":
                return np.ones((1, 4, 7)) * self.gold
            return np.zeros((1, 4, 7))

    player = TestPlayer()
    assert hasattr(player, "_obs_builder")
    assert isinstance(player._obs_builder, ObservationBuilder)
    
    gold_obs = player.get_observation_for_field("gold")
    assert np.all(gold_obs == 10)
    
    with pytest.raises(NotImplementedError):
        class IncompletePlayer(ObservationManagerMixin):
            pass
        p = IncompletePlayer()
        p.get_observation_for_field("gold")

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
    
    s = obs_get_field_slice("gold")
    assert isinstance(s, slice)
    
    # Test update_config_observation_size
    original_size = config.OBSERVATION_SIZE
    new_size = update_config_observation_size()
    assert new_size > 0
    assert config.OBSERVATION_SIZE == new_size
    
    # Test get/set convenience
    obs = obs_dict["tensor"]
    gold_val = np.ones((1, 4, 7)) * 75
    obs = set_field_value_in_obs(obs, "gold", gold_val)
    retrieved = get_field_value_from_obs(obs, "gold")
    assert np.allclose(retrieved, gold_val)
