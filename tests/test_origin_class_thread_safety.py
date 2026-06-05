"""
Unit tests for thread safety and concurrent reassignment protection in origin_class helpers.
"""

import pytest
import sys
import os
import threading
import time

# Add package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.combat_state import get_state
from TFTSet4Gym.tft_set4_gym.champion import champion, reset_global_variables
from TFTSet4Gym.tft_set4_gym import origin_class


def test_duelist_and_shade_helper_reassignment_thread_safety():
    """
    Test that duelist_helper and shade_helper do not raise IndexError
    if reset_global_variables() is called concurrently, reassigning the
    module-level lists to [] mid-execution.
    """
    reset_global_variables()
    state = get_state()

    champ = champion("zilean", team="blue", y=0, x=0, stars=1, itemlist=[])
    state.blue.append(champ)

    # Set up amounts to get tier > 0
    origin_class.amounts['duelist']['blue'] = 2
    origin_class.amounts['shade']['blue'] = 2

    # Verify initial behavior of duelist_helper
    origin_class.duelist_helper(champ)
    assert len(origin_class.duelist_helper_list) == 1
    assert origin_class.duelist_helper_list[0] == [champ, 1]

    # Clear and set up the custom intercept list for duelist
    origin_class.duelist_helper_list = []

    class InterceptListDuelist(list):
        def __iter__(self):
            # Mid-execution (at start of iteration), simulate concurrent reassignment
            origin_class.duelist_helper_list = []
            return super().__iter__()

    intercept_dh = InterceptListDuelist([[champ, 1]])
    origin_class.duelist_helper_list = intercept_dh

    # This should call duelist_helper and should NOT raise IndexError
    origin_class.duelist_helper(champ)

    # Verify that the value in our local reference (intercept_dh) was incremented
    assert intercept_dh[0][1] == 2
    # Verify that the module level list was reassigned to a new empty list by our intercept
    assert origin_class.duelist_helper_list == []


    # Verify initial behavior of shade_helper
    reset_global_variables()
    origin_class.amounts['shade']['blue'] = 2

    origin_class.shade_helper(champ)
    assert len(origin_class.shade_helper_list) == 1
    assert origin_class.shade_helper_list[0] == [champ, 1]

    # Clear and set up the custom intercept list for shade
    class InterceptListShade(list):
        def __iter__(self):
            # Mid-execution (at start of iteration), simulate concurrent reassignment
            origin_class.shade_helper_list = []
            return super().__iter__()

    intercept_sh = InterceptListShade([[champ, 1]])
    origin_class.shade_helper_list = intercept_sh

    # This should call shade_helper and should NOT raise IndexError
    origin_class.shade_helper(champ)

    # Verify that the value in our local reference (intercept_sh) was incremented
    assert intercept_sh[0][1] == 2
    # Verify that the module level list was reassigned to a new empty list by our intercept
    assert origin_class.shade_helper_list == []


def test_origin_class_helpers_multi_threaded_concurrency():
    """
    Stress test duelist_helper and shade_helper concurrently with
    reset_global_variables() running in other threads.
    """
    reset_global_variables()
    state = get_state()
    champ = champion("zilean", team="blue", y=0, x=0, stars=1, itemlist=[])
    state.blue.append(champ)

    origin_class.amounts['duelist']['blue'] = 2
    origin_class.amounts['shade']['blue'] = 2

    stop_event = threading.Event()
    exceptions = []

    def run_helpers():
        while not stop_event.is_set():
            try:
                origin_class.duelist_helper(champ)
                origin_class.shade_helper(champ)
            except Exception as e:
                exceptions.append(e)
                stop_event.set()

    def run_resets():
        while not stop_event.is_set():
            reset_global_variables()
            # Restore tier amounts after reset, because reset clears them
            origin_class.amounts['duelist']['blue'] = 2
            origin_class.amounts['shade']['blue'] = 2
            time.sleep(0.001)

    threads = []
    # Start multiple threads running helpers
    for _ in range(4):
        t = threading.Thread(target=run_helpers)
        t.start()
        threads.append(t)

    # Start a thread running resets
    t_reset = threading.Thread(target=run_resets)
    t_reset.start()
    threads.append(t_reset)

    # Let the stress test run for 0.5 seconds
    time.sleep(0.5)
    stop_event.set()

    for t in threads:
        t.join()

    # If any exception was caught in the helper threads, raise it to fail the test
    if exceptions:
        raise exceptions[0]
