import pytest
import numpy as np
from TFTSet4Gym.tft_set4_gym.observation_builder import ObservationBuilder, ObservationManagerMixin, encode_item_id
from TFTSet4Gym.tft_set4_gym.player import Player, encoded_list, encode_champ_object
from TFTSet4Gym.tft_set4_gym.stats import COST
from TFTSet4Gym.tft_set4_gym.item_stats import uncraftable_items, item_builds

class MockPool:
    def __init__(self):
        pass
    def update_pool(self, name, amount):
        pass

class MockChampion:
    def __init__(self, name, stars=1, chosen=False, items=None):
        self.name = name
        self.stars = stars
        self.chosen = chosen
        self.items = items or []

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
    expected_health = np.array([100.0])
    assert np.allclose(obs["tensor"][health_slice], expected_health)

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
    
    # Verify bench (2D: champion_type x bench_slot)
    bench_champ_slice = builder.current_player_schema.get_field_slice("bench_champions")
    bench_champs = obs["tensor"][bench_champ_slice].reshape((58, 9))
    ahri_idx = list(COST.keys()).index('ahri') - 1
    # ahri is on bench[0], slot 0
    assert bench_champs[ahri_idx, 0] == 1.0

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
    
    # Create a mock shop vector (59,): [0:58] champion counts, [58] chosen index
    shop_vector = np.random.rand(59)
    
    obs = builder.build_observation("player_0", player, shop_vector=shop_vector)
    
    shop_champ_slice = builder.current_player_schema.get_field_slice("shop_champions")
    shop_champs = obs["tensor"][shop_champ_slice]
    assert np.allclose(shop_champs, shop_vector[0:58])
    
    shop_chosen_slice = builder.current_player_schema.get_field_slice("shop_chosen")
    shop_chosen = obs["tensor"][shop_chosen_slice]
    assert np.allclose(shop_chosen, shop_vector[58:59])

@pytest.mark.unit
@pytest.mark.observation
def test_get_set_field():
    builder = ObservationBuilder()
    obs = np.zeros(builder.current_player_schema.total_size)
    
    # Test setting and getting gold
    gold_val = np.ones((1,)) * 50
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
                return np.ones((1,)) * self.gold
            return np.zeros((1,))

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
    gold_val = np.ones((1,)) * 75
    obs = set_field_value_in_obs(obs, "gold", gold_val)
    retrieved = get_field_value_from_obs(obs, "gold")
    assert np.allclose(retrieved, gold_val)


@pytest.mark.unit
@pytest.mark.observation
def test_board_items_field():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    aatrox = MockChampion('aatrox', stars=2, chosen=False, items=['bf_sword', 'chain_vest'])
    player.board[0][0] = aatrox

    obs = builder.build_observation("player_0", player)

    board_items = builder.get_field_from_observation(obs["tensor"], "board_items")

    assert board_items.shape == (3, 4, 7)
    bf_sword_id = encode_item_id('bf_sword')
    chain_vest_id = encode_item_id('chain_vest')
    assert board_items[0, 0, 0] == bf_sword_id
    assert board_items[1, 0, 0] == chain_vest_id
    assert board_items[2, 0, 0] == 0.0


@pytest.mark.unit
@pytest.mark.observation
def test_bench_slot_wise():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    ahri = MockChampion('ahri', stars=3, chosen=False, items=['rabadons_deathcap'])
    player.bench[2] = ahri

    obs = builder.build_observation("player_0", player)

    bench_champs = builder.get_field_from_observation(obs["tensor"], "bench_champions")
    assert bench_champs.shape == (58, 9)

    ahri_idx = list(COST.keys()).index('ahri') - 1
    assert bench_champs[ahri_idx, 2] == 1.0

    bench_stars = builder.get_field_from_observation(obs["tensor"], "bench_stars")
    assert bench_stars.shape == (1, 9)
    assert bench_stars[0, 2] == 3.0

    bench_items = builder.get_field_from_observation(obs["tensor"], "bench_items")
    assert bench_items.shape == (3, 9)
    rabadons_id = encode_item_id('rabadons_deathcap')
    assert bench_items[0, 2] == rabadons_id


@pytest.mark.unit
@pytest.mark.observation
def test_item_bench_field():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    player.add_to_item_bench('bf_sword')
    player.add_to_item_bench('chain_vest')

    obs = builder.build_observation("player_0", player)

    item_bench = builder.get_field_from_observation(obs["tensor"], "item_bench")
    assert item_bench.shape == (10,)
    bf_sword_id = encode_item_id('bf_sword')
    chain_vest_id = encode_item_id('chain_vest')
    assert item_bench[0] == bf_sword_id
    assert item_bench[1] == chain_vest_id
    assert item_bench[2] == 0.0


@pytest.mark.unit
@pytest.mark.observation
def test_shop_locked_field():
    builder = ObservationBuilder()
    pool = MockPool()
    player = Player(pool, 0)

    player.shop_locked = True
    obs = builder.build_observation("player_0", player)
    shop_locked = builder.get_field_from_observation(obs["tensor"], "shop_locked")
    assert shop_locked.shape == (1,)
    assert shop_locked[0] == 1.0

    player.shop_locked = False
    obs = builder.build_observation("player_0", player)
    shop_locked = builder.get_field_from_observation(obs["tensor"], "shop_locked")
    assert shop_locked[0] == 0.0


@pytest.mark.unit
@pytest.mark.observation
def test_opponents_fields():
    builder = ObservationBuilder()
    pool = MockPool()

    class MockOpponent:
        def __init__(self, player_num, health, level, gold):
            self.player_num = player_num
            self.health = health
            self.level = level
            self.gold = gold

    main_player = Player(pool, 0)
    main_player.health = 100
    main_player.gold = 50

    opponents = {
        0: main_player,
        1: MockOpponent(1, 80, 7, 30),
        2: MockOpponent(2, 60, 6, 20),
        3: MockOpponent(3, 40, 5, 10),
    }

    obs = builder.build_observation("player_0", main_player, all_players=opponents)

    opp_health = builder.get_field_from_observation(obs["tensor"], "opponents_health")
    opp_level = builder.get_field_from_observation(obs["tensor"], "opponents_level")
    opp_gold = builder.get_field_from_observation(obs["tensor"], "opponents_gold")

    assert opp_health.shape == (7,)
    assert opp_level.shape == (7,)
    assert opp_gold.shape == (7,)

    assert opp_health[0] == 80
    assert opp_level[0] == 7
    assert opp_gold[0] == 30
    assert opp_health[1] == 60
    assert opp_gold[1] == 20
    assert np.all(opp_health[3:] == 0)


@pytest.mark.unit
@pytest.mark.observation
def test_encode_item_id():
    bf_sword_id = encode_item_id('bf_sword')
    assert bf_sword_id > 0

    unknown_id = encode_item_id('nonexistent_item')
    assert unknown_id == 0.0

    first_uncraftable = list(uncraftable_items)[0]
    first_id = encode_item_id(first_uncraftable)
    assert first_id == 1.0

    first_craftable = list(item_builds.keys())[0]
    craftable_id = encode_item_id(first_craftable)
    assert craftable_id > len(uncraftable_items) + 0.0
