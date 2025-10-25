"""
Benchmark tests for TFT Set 4 Gym environment.
Measures performance metrics like initialization, reset, step times, and memory usage.
"""

import pytest
import time
import sys
import os
import gc
import psutil
import numpy as np
from typing import Dict, List, Tuple
from statistics import mean, median, stdev

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tft_set4_gym.tft_simulator import parallel_env


class EnvironmentBenchmark:
    """Benchmark suite for TFT Set 4 Gym environment."""
    
    def __init__(self, num_iterations: int = 100, num_steps_per_episode: int = 50):
        """
        Initialize benchmark.
        
        Args:
            num_iterations: Number of iterations for timing tests
            num_steps_per_episode: Number of steps to run per episode in step timing tests
        """
        self.num_iterations = num_iterations
        self.num_steps_per_episode = num_steps_per_episode
        self.results = {}
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_environment_creation(self) -> Dict[str, float]:
        """Benchmark environment creation time."""
        print("🏗️  Benchmarking environment creation...")
        
        creation_times = []
        
        for i in range(self.num_iterations):
            if i % 20 == 0:
                print(f"   Progress: {i}/{self.num_iterations}")
                
            gc.collect()  # Clean up memory before timing
            start_time = time.perf_counter()
            
            env = parallel_env()
            
            end_time = time.perf_counter()
            creation_times.append(end_time - start_time)
            
            # Clean up
            env.close()
            del env
            gc.collect()
        
        return {
            'mean': mean(creation_times),
            'median': median(creation_times),
            'std': stdev(creation_times) if len(creation_times) > 1 else 0,
            'min': min(creation_times),
            'max': max(creation_times),
            'total_iterations': len(creation_times)
        }
    
    def benchmark_environment_reset(self) -> Dict[str, float]:
        """Benchmark environment reset time."""
        print("🔄 Benchmarking environment reset...")
        
        env = parallel_env()
        reset_times = []
        
        for i in range(self.num_iterations):
            if i % 20 == 0:
                print(f"   Progress: {i}/{self.num_iterations}")
                
            gc.collect()
            start_time = time.perf_counter()
            
            observations, infos = env.reset()
            
            end_time = time.perf_counter()
            reset_times.append(end_time - start_time)
        
        env.close()
        
        return {
            'mean': mean(reset_times),
            'median': median(reset_times),
            'std': stdev(reset_times) if len(reset_times) > 1 else 0,
            'min': min(reset_times),
            'max': max(reset_times),
            'total_iterations': len(reset_times)
        }
    
    def benchmark_environment_step(self) -> Dict[str, float]:
        """Benchmark environment step time."""
        print("👟 Benchmarking environment steps...")
        
        env = parallel_env()
        step_times = []
        
        # Run multiple episodes
        episodes = max(1, self.num_iterations // self.num_steps_per_episode)
        
        for episode in range(episodes):
            if episode % 5 == 0:
                print(f"   Episode: {episode}/{episodes}")
                
            observations, infos = env.reset()
            
            for step in range(self.num_steps_per_episode):
                if not env.agents:
                    break
                    
                # Create random actions for all agents
                actions = {}
                for agent in env.agents:
                    actions[agent] = env.action_space(agent).sample()
                
                gc.collect()
                start_time = time.perf_counter()
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                end_time = time.perf_counter()
                step_times.append(end_time - start_time)
                
                # Break if game ended
                if all(terminations.values()) or all(truncations.values()):
                    break
        
        env.close()
        
        return {
            'mean': mean(step_times),
            'median': median(step_times),
            'std': stdev(step_times) if len(step_times) > 1 else 0,
            'min': min(step_times),
            'max': max(step_times),
            'total_steps': len(step_times),
            'episodes': episodes
        }
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage during environment lifecycle."""
        print("🧠 Benchmarking memory usage...")
        
        # Baseline memory
        gc.collect()
        baseline_memory = self.get_memory_usage()
        
        # Memory after environment creation
        env = parallel_env()
        after_creation_memory = self.get_memory_usage()
        
        # Memory after reset
        observations, infos = env.reset()
        after_reset_memory = self.get_memory_usage()
        
        # Memory during steps
        memory_during_steps = []
        rewards = terminations = truncations = None  # Initialize for cleanup
        
        for step in range(20):  # Sample 20 steps
            if not env.agents:
                break
                
            actions = {}
            for agent in env.agents:
                actions[agent] = env.action_space(agent).sample()
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            memory_during_steps.append(self.get_memory_usage())
            
            if all(terminations.values()) or all(truncations.values()):
                break
        
        # Memory after cleanup
        env.close()
        del env
        try:
            del observations, rewards, terminations, truncations, infos
        except NameError:
            # Some variables might not be defined if no steps were taken
            pass
        gc.collect()
        after_cleanup_memory = self.get_memory_usage()
        
        return {
            'baseline_mb': baseline_memory,
            'after_creation_mb': after_creation_memory,
            'after_reset_mb': after_reset_memory,
            'during_steps_mean_mb': mean(memory_during_steps) if memory_during_steps else 0,
            'during_steps_max_mb': max(memory_during_steps) if memory_during_steps else 0,
            'after_cleanup_mb': after_cleanup_memory,
            'creation_overhead_mb': after_creation_memory - baseline_memory,
            'reset_overhead_mb': after_reset_memory - after_creation_memory,
            'cleanup_recovery_mb': (after_reset_memory - after_cleanup_memory) if after_cleanup_memory < after_reset_memory else 0
        }
    
    def benchmark_observation_processing(self) -> Dict[str, float]:
        """Benchmark observation processing time."""
        print("👁️  Benchmarking observation processing...")
        
        env = parallel_env()
        observations, infos = env.reset()
        
        processing_times = []
        
        for i in range(self.num_iterations):
            if i % 20 == 0:
                print(f"   Progress: {i}/{self.num_iterations}")
            
            # Get sample observation
            sample_agent = list(observations.keys())[0]
            sample_obs = observations[sample_agent]
            
            start_time = time.perf_counter()
            
            # Simulate typical observation processing
            tensor_flat = sample_obs['tensor'].flatten()
            action_mask_flat = sample_obs['action_mask'].flatten()
            combined = np.concatenate([tensor_flat, action_mask_flat])
            
            end_time = time.perf_counter()
            processing_times.append(end_time - start_time)
        
        env.close()
        
        return {
            'mean': mean(processing_times),
            'median': median(processing_times),
            'std': stdev(processing_times) if len(processing_times) > 1 else 0,
            'min': min(processing_times),
            'max': max(processing_times),
            'total_iterations': len(processing_times)
        }
    
    def run_full_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks and return comprehensive results."""
        print("🚀 Starting TFT Set 4 Gym Environment Benchmark Suite")
        print(f"📊 Configuration: {self.num_iterations} iterations, {self.num_steps_per_episode} steps per episode")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.results['environment_creation'] = self.benchmark_environment_creation()
        self.results['environment_reset'] = self.benchmark_environment_reset()
        self.results['environment_step'] = self.benchmark_environment_step()
        self.results['memory_usage'] = self.benchmark_memory_usage()
        self.results['observation_processing'] = self.benchmark_observation_processing()
        
        total_time = time.time() - start_time
        self.results['benchmark_meta'] = {
            'total_benchmark_time': total_time,
            'num_iterations': self.num_iterations,
            'num_steps_per_episode': self.num_steps_per_episode
        }
        
        print("=" * 60)
        print(f"✅ Benchmark completed in {total_time:.2f} seconds")
        
        return self.results
    
    def print_results(self):
        """Print formatted benchmark results."""
        if not self.results:
            print("❌ No benchmark results available. Run benchmark first.")
            return
        
        print("\n" + "=" * 80)
        print("🏆 TFT SET 4 GYM ENVIRONMENT BENCHMARK RESULTS")
        print("=" * 80)
        
        # Environment Creation
        creation = self.results['environment_creation']
        print(f"\n🏗️  ENVIRONMENT CREATION (n={creation['total_iterations']})")
        print(f"   Mean: {creation['mean']*1000:.2f}ms")
        print(f"   Median: {creation['median']*1000:.2f}ms")
        print(f"   Std Dev: {creation['std']*1000:.2f}ms")
        print(f"   Range: {creation['min']*1000:.2f}ms - {creation['max']*1000:.2f}ms")
        
        # Environment Reset
        reset = self.results['environment_reset']
        print(f"\n🔄 ENVIRONMENT RESET (n={reset['total_iterations']})")
        print(f"   Mean: {reset['mean']*1000:.2f}ms")
        print(f"   Median: {reset['median']*1000:.2f}ms")
        print(f"   Std Dev: {reset['std']*1000:.2f}ms")
        print(f"   Range: {reset['min']*1000:.2f}ms - {reset['max']*1000:.2f}ms")
        
        # Environment Step
        step = self.results['environment_step']
        print(f"\n👟 ENVIRONMENT STEP (n={step['total_steps']} steps, {step['episodes']} episodes)")
        print(f"   Mean: {step['mean']*1000:.2f}ms")
        print(f"   Median: {step['median']*1000:.2f}ms")
        print(f"   Std Dev: {step['std']*1000:.2f}ms")
        print(f"   Range: {step['min']*1000:.2f}ms - {step['max']*1000:.2f}ms")
        print(f"   Steps/second: {1/step['mean']:.1f}")
        
        # Memory Usage
        memory = self.results['memory_usage']
        print(f"\n🧠 MEMORY USAGE")
        print(f"   Baseline: {memory['baseline_mb']:.1f}MB")
        print(f"   After creation: {memory['after_creation_mb']:.1f}MB (+{memory['creation_overhead_mb']:.1f}MB)")
        print(f"   After reset: {memory['after_reset_mb']:.1f}MB (+{memory['reset_overhead_mb']:.1f}MB)")
        print(f"   During steps (avg): {memory['during_steps_mean_mb']:.1f}MB")
        print(f"   During steps (max): {memory['during_steps_max_mb']:.1f}MB")
        print(f"   After cleanup: {memory['after_cleanup_mb']:.1f}MB")
        if memory['cleanup_recovery_mb'] > 0:
            print(f"   Memory recovered: {memory['cleanup_recovery_mb']:.1f}MB")
        
        # Observation Processing
        obs = self.results['observation_processing']
        print(f"\n👁️  OBSERVATION PROCESSING (n={obs['total_iterations']})")
        print(f"   Mean: {obs['mean']*1000000:.1f}μs")
        print(f"   Median: {obs['median']*1000000:.1f}μs")
        print(f"   Std Dev: {obs['std']*1000000:.1f}μs")
        print(f"   Range: {obs['min']*1000000:.1f}μs - {obs['max']*1000000:.1f}μs")
        
        # Performance Summary
        print(f"\n⚡ PERFORMANCE SUMMARY")
        print(f"   Environment creation rate: {1/creation['mean']:.1f} envs/second")
        print(f"   Environment reset rate: {1/reset['mean']:.1f} resets/second")
        print(f"   Environment step rate: {1/step['mean']:.1f} steps/second")
        print(f"   Memory overhead: {memory['creation_overhead_mb']:.1f}MB per environment")
        
        meta = self.results['benchmark_meta']
        print(f"\n📊 BENCHMARK INFO")
        print(f"   Total benchmark time: {meta['total_benchmark_time']:.1f}s")
        print(f"   Configuration: {meta['num_iterations']} iterations, {meta['num_steps_per_episode']} steps/episode")
        
        print("=" * 80)


@pytest.mark.benchmark
@pytest.mark.slow
def test_quick_benchmark():
    """Run a quick benchmark with fewer iterations."""
    print("🚀 Running Quick Benchmark...")
    benchmark = EnvironmentBenchmark(num_iterations=10, num_steps_per_episode=20)
    benchmark.run_full_benchmark()
    benchmark.print_results()


@pytest.mark.benchmark
@pytest.mark.slow  
def test_comprehensive_benchmark():
    """Run a comprehensive benchmark with more iterations."""
    print("🚀 Running Comprehensive Benchmark...")
    benchmark = EnvironmentBenchmark(num_iterations=100, num_steps_per_episode=50)
    benchmark.run_full_benchmark()
    benchmark.print_results()


def run_quick_benchmark():
    """Legacy function - Run a quick benchmark with fewer iterations."""
    test_quick_benchmark()


def run_comprehensive_benchmark():
    """Legacy function - Run a comprehensive benchmark with more iterations."""
    test_comprehensive_benchmark()


if __name__ == "__main__":
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("❌ psutil not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    # Run benchmark based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        run_comprehensive_benchmark()
    else:
        run_quick_benchmark()
    
    print("\n🎉 Benchmark complete!")