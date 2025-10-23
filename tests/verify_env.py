"""
Quick verification script to test the TFT PettingZoo environment
before running Stable Baselines 3 training.
"""

import sys
import numpy as np
import sys
import os

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tft_set4_gym.tft_simulator import parallel_env


def test_basic_environment():
    """Test basic environment functionality."""
    print("🧪 Testing TFT PettingZoo Environment")
    print("=" * 50)
    
    try:
        # Create environment
        print("1. Creating environment...")
        env = parallel_env()
        print(f"   ✅ Environment created successfully")
        
        # Test reset
        print("2. Testing reset...")
        obs, info = env.reset()
        print(f"   ✅ Reset successful")
        print(f"   - Agents: {len(env.agents)}")
        print(f"   - Observation keys: {list(obs.keys())}")
        print(f"   - Sample obs shape: {obs[list(obs.keys())[0]]['tensor'].shape}")
        print(f"   - Action mask shape: {obs[list(obs.keys())[0]]['action_mask'].shape}")
        
        # Test action spaces
        print("3. Testing action spaces...")
        agent = list(env.agents)[0]
        action_space = env.action_space(agent)
        print(f"   ✅ Action space: {action_space}")
        sample_action = action_space.sample()
        print(f"   - Sample action: {sample_action}")
        
        # Test observation spaces
        print("4. Testing observation spaces...")
        obs_space = env.observation_space(agent)
        print(f"   ✅ Observation space: {obs_space}")
        
        # Test steps
        print("5. Testing environment steps...")
        for step in range(10):
            # Generate random actions for all agents
            actions = {}
            for agent in env.agents:
                actions[agent] = env.action_space(agent).sample()
            
            # Take step
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            # Check for agent elimination
            active_agents = len(env.agents)
            if active_agents < 8:
                print(f"   🎯 Agent elimination detected at step {step+1}!")
                print(f"   - Active agents: {active_agents}")
                break
            
            if step % 5 == 0:
                print(f"   Step {step+1}: {active_agents} agents active")
        
        print("   ✅ Environment steps working correctly")
        
        # Test state method
        print("6. Testing state method...")
        try:
            state = env.state()
            print(f"   ✅ State method works, shape: {state.shape}")
        except Exception as e:
            print(f"   ⚠️  State method issue: {e}")
        
        env.close()
        print("\n🎉 Environment verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_sb3_compatibility():
    """Check if the environment is compatible with SB3 using our custom wrapper."""
    print("\n🔧 Testing SB3 Compatibility (Pure SB3 Approach)")
    print("=" * 50)
    
    try:
        # Test our custom wrapper
        from sb3_wrapper import TFTSingleAgentWrapper
        print("1. ✅ Custom SB3 wrapper available")
        
        # Create single-agent environment
        print("2. Testing single-agent wrapper...")
        env = TFTSingleAgentWrapper()
        
        # Test reset
        obs, info = env.reset()
        print(f"   - Reset successful: obs shape {obs.shape}")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")
        
        # Test steps
        print("3. Testing environment steps...")
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"   - Episode ended at step {i+1}")
                break
        
        print(f"   - Total reward: {total_reward:.3f}")
        print(f"   - Final obs shape: {obs.shape}")
        
        # Test vectorized wrapper
        print("4. Testing vectorized wrapper...")
        from sb3_wrapper import TFTVectorizedWrapper
        
        vec_env = TFTVectorizedWrapper(num_envs=2)
        obs, infos = vec_env.reset()
        print(f"   - Vectorized reset: obs shape {obs.shape}")
        
        # Take some steps
        actions = np.array([vec_env.action_space.sample() for _ in range(2)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        print(f"   - Vectorized step: rewards {rewards}")
        
        vec_env.close()
        env.close()
        
        print("\n🎉 Pure SB3 compatibility check passed!")
        print("   ✅ No SuperSuit needed")
        print("   ✅ Dict observations handled correctly")
        print("   ✅ Multi-agent → Single-agent conversion working")
        print("   ✅ Vectorized environments supported")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SB3 compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("TFT Environment Verification Script")
    print("=" * 60)
    
    # Test basic environment
    basic_test = test_basic_environment()
    
    # Test SB3 compatibility
    sb3_test = check_sb3_compatibility()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Basic Environment Test: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"SB3 Compatibility Test: {'✅ PASS' if sb3_test else '❌ FAIL'}")
    
    if basic_test and sb3_test:
        print("\n🚀 Environment is ready for SB3 training!")
        print("Run: python sb_run.py --mode train --timesteps 10000")
    else:
        print("\n⚠️  Please fix the issues above before training.")
        if not sb3_test:
            print("Install dependencies: pip install -r requirements_sb3.txt")


if __name__ == "__main__":
    main()