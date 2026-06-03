"""
Benchmark tests for TFT Set 4 Gym environment.
Measures performance metrics like initialization, reset, step times, and memory usage.
"""

import pytest
import time
import sys
import os
import gc
import json
import datetime
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional
from statistics import mean, median, stdev

# Add the package root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TFTSet4Gym.tft_set4_gym.tft_simulator import parallel_env

# Default path for benchmark results file
BENCHMARK_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')


class MockAgent:
    """A mock agent that simulates computational load for benchmarking agent overhead."""

    def __init__(self, compute_time: float = 0.0, matrix_size: int = 100):
        self.compute_time = compute_time
        self.matrix_size = matrix_size

    def predict(self, observation) -> Dict:
        """
        Simulate agent inference by sleeping (if compute_time > 0) and
        performing a small matrix multiplication, then sampling a random action.
        """
        if self.compute_time > 0:
            time.sleep(self.compute_time)
        if self.matrix_size > 0:
            a = np.random.randn(self.matrix_size, self.matrix_size)
            b = np.random.randn(self.matrix_size, self.matrix_size)
            _ = a @ b
        return None


class EnvironmentBenchmark:
    """Benchmark suite for TFT Set 4 Gym environment."""
    
    def __init__(self, num_iterations: int = 100, num_steps_per_episode: int = 50,
                 agent_policy: Optional[object] = None):
        """
        Initialize benchmark.
        
        Args:
            num_iterations: Number of iterations for timing tests
            num_steps_per_episode: Number of steps to run per episode in step timing tests
            agent_policy: Optional agent object with a `predict(observation)` method.
                          If None, defaults to random action sampling.
        """
        self.num_iterations = num_iterations
        self.num_steps_per_episode = num_steps_per_episode
        self.agent_policy = agent_policy
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
        """Benchmark environment step time, with optional agent policy inference timing."""
        using_agent = self.agent_policy is not None
        if using_agent:
            print("👟 Benchmarking environment steps (with agent policy)...")
        else:
            print("👟 Benchmarking environment steps...")
        
        env = parallel_env()
        step_times = []
        inference_times = []
        env_step_times = []
        
        # Run multiple episodes
        episodes = max(1, self.num_iterations // self.num_steps_per_episode)
        
        for episode in range(episodes):
            if episode % 5 == 0:
                print(f"   Episode: {episode}/{episodes}")
                
            observations, infos = env.reset()
            
            for step in range(self.num_steps_per_episode):
                if not env.agents:
                    break
                    
                # Create actions for all agents
                actions = {}
                for agent in env.agents:
                    if using_agent:
                        inf_start = time.perf_counter()
                        self.agent_policy.predict(observations[agent])
                        inf_end = time.perf_counter()
                        inference_times.append(inf_end - inf_start)
                        action = env.action_space(agent).sample()
                        actions[agent] = action
                    else:
                        actions[agent] = env.action_space(agent).sample()
                
                gc.collect()
                step_start = time.perf_counter()
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                step_end = time.perf_counter()
                step_times.append(step_end - step_start)
                env_step_times.append(step_end - step_start)
                
                # Break if game ended
                if all(terminations.values()) or all(truncations.values()):
                    break
        
        env.close()
        
        result = {
            'mean': mean(step_times),
            'median': median(step_times),
            'std': stdev(step_times) if len(step_times) > 1 else 0,
            'min': min(step_times),
            'max': max(step_times),
            'total_steps': len(step_times),
            'episodes': episodes,
            'env_step_time_mean': mean(env_step_times),
        }

        if using_agent and inference_times:
            agent_inf_mean = mean(inference_times)
            total_per_step = agent_inf_mean + result['env_step_time_mean']
            result['agent_inference_time_mean'] = agent_inf_mean
            result['agent_overhead_percent'] = (
                (agent_inf_mean / total_per_step) * 100 if total_per_step > 0 else 0
            )

        return result
    
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
    
    def benchmark_full_episode(self) -> Dict[str, float]:
        """Run a full game episode and capture timing metrics from RuntimeStats."""
        using_agent = self.agent_policy is not None
        if using_agent:
            print("🎮 Benchmarking full game episode (with agent policy)...")
        else:
            print("🎮 Benchmarking full game episode...")

        env = parallel_env()
        observations, infos = env.reset()

        episode_start = time.perf_counter()
        step_count = 0
        inference_times = []
        env_step_times = []

        while env.agents:
            actions = {}
            for agent in env.agents:
                if using_agent:
                    inf_start = time.perf_counter()
                    self.agent_policy.predict(observations[agent])
                    inf_end = time.perf_counter()
                    inference_times.append(inf_end - inf_start)
                actions[agent] = env.action_space(agent).sample()

            step_start = time.perf_counter()
            observations, rewards, terminations, truncations, infos = env.step(actions)
            step_end = time.perf_counter()

            env_step_times.append(step_end - step_start)
            step_count += 1

            if all(terminations.values()) or all(truncations.values()):
                break

        episode_time = time.perf_counter() - episode_start

        # Collect runtime stats from the last surviving agent
        stats = None
        for agent, info in infos.items():
            if "runtime_stats" in info:
                stats = info["runtime_stats"]
                break

        env.close()

        result = {
            "episode_time_s": episode_time,
            "steps": step_count,
            "env_step_time_mean": mean(env_step_times) if env_step_times else 0,
        }

        if using_agent and inference_times:
            agent_inf_mean = mean(inference_times)
            total_per_step = agent_inf_mean + result["env_step_time_mean"]
            result["agent_inference_time_mean"] = agent_inf_mean
            result["agent_overhead_percent"] = (
                (agent_inf_mean / total_per_step) * 100 if total_per_step > 0 else 0
            )

        if stats:
            result.update({
                "turns": stats["num_turns"],
                "combats": stats["num_combats"],
                "turn_time_avg_ms": stats["turn_time_avg_s"] * 1000,
                "turn_time_total_s": stats["turn_time_total_s"],
                "combat_time_avg_ms": stats["combat_time_avg_s"] * 1000,
                "combat_time_total_s": stats["combat_time_total_s"],
                "step_time_avg_ms": stats["step_time_avg_s"] * 1000,
            })

        return result

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
        self.results['full_episode'] = self.benchmark_full_episode()

        total_time = time.time() - start_time
        self.results['benchmark_meta'] = {
            'total_benchmark_time': total_time,
            'num_iterations': self.num_iterations,
            'num_steps_per_episode': self.num_steps_per_episode,
            'timestamp': datetime.datetime.now().isoformat()
        }

        print("=" * 60)
        print(f"✅ Benchmark completed in {total_time:.2f} seconds")

        # Export results to JSON
        self._export_results()

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
        if 'agent_inference_time_mean' in step:
            print(f"   Agent inference time (avg): {step['agent_inference_time_mean']*1000:.2f}ms")
            print(f"   Env step time (avg): {step['env_step_time_mean']*1000:.2f}ms")
            print(f"   Agent overhead: {step['agent_overhead_percent']:.1f}%")
        
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
        
        # Full Episode
        if 'full_episode' in self.results:
            ep = self.results['full_episode']
            print(f"\n🎮 FULL EPISODE")
            print(f"   Episode time: {ep['episode_time_s']:.2f}s")
            print(f"   Steps: {ep['steps']}")
            if 'turns' in ep:
                print(f"   Turns: {ep['turns']}")
                print(f"   Turn time (avg): {ep['turn_time_avg_ms']:.2f}ms")
            if 'combats' in ep:
                print(f"   Combats: {ep['combats']}")
                print(f"   Combat time (avg): {ep['combat_time_avg_ms']:.2f}ms")
            if 'step_time_avg_ms' in ep:
                print(f"   Step time (avg): {ep['step_time_avg_ms']:.2f}ms")
            if 'agent_inference_time_mean' in ep:
                print(f"   Agent inference time (avg): {ep['agent_inference_time_mean']*1000:.2f}ms")
                print(f"   Agent overhead: {ep['agent_overhead_percent']:.1f}%")

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

    def _export_results(self, path: Optional[str] = None):
        """Export benchmark results to a JSON file."""
        if path is None:
            path = BENCHMARK_RESULTS_PATH
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📁 Results exported to {path}")

    @staticmethod
    def load_results(path: str) -> Dict:
        """Load benchmark results from a JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def compare_results(new: Dict, baseline: Dict) -> Dict:
        """Compare two benchmark result dicts and return deltas."""
        deltas = {}
        sections = [k for k in new if k != 'benchmark_meta']
        for section in sections:
            if section not in baseline:
                continue
            deltas[section] = {}
            for key in new[section]:
                if key not in baseline[section]:
                    continue
                old_val = baseline[section][key]
                new_val = new[section][key]
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)) and old_val != 0:
                    delta = new_val - old_val
                    delta_pct = (delta / abs(old_val)) * 100
                    deltas[section][key] = {
                        'baseline': old_val,
                        'current': new_val,
                        'delta': delta,
                        'delta_pct': delta_pct
                    }
                elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    deltas[section][key] = {
                        'baseline': old_val,
                        'current': new_val,
                        'delta': new_val - old_val,
                        'delta_pct': 0.0
                    }
        return deltas

    @staticmethod
    def print_comparison(deltas: Dict):
        """Print formatted comparison results."""
        if not deltas:
            print("No comparable sections found between runs.")
            return

        print("\n" + "=" * 90)
        print("📊 BENCHMARK COMPARISON — REGRESSIONS / IMPROVEMENTS")
        print("=" * 90)
        has_regression = False
        has_improvement = False

        for section, metrics in deltas.items():
            print(f"\n  [{section.replace('_', ' ').title()}]")
            for key, data in metrics.items():
                delta = data['delta']
                delta_pct = data['delta_pct']
                direction = "🔴 WORSE" if delta > 0 else "🟢 BETTER"
                if delta > 0:
                    has_regression = True
                else:
                    has_improvement = True
                print(f"    {key:30s}  {data['baseline']:>12.4f}  →  {data['current']:<12.4f}  "
                      f"({delta:+.4f}, {delta_pct:+.2f}%)  {direction}")

        print("=" * 90)
        if has_regression:
            print("⚠️  Some metrics show regressions.")
        if has_improvement:
            print("✅  Some metrics show improvements.")
        if not has_regression and not has_improvement:
            print("✅  No significant changes detected.")

    @staticmethod
    def get_scoring_keys() -> Dict[str, str]:
        """Return the metric keys that indicate 'lower is better' (performance metrics)."""
        return {
            'environment_creation': ['mean', 'median', 'std', 'min', 'max'],
            'environment_reset': ['mean', 'median', 'std', 'min', 'max'],
            'environment_step': ['mean', 'median', 'std', 'min', 'max'],
            'observation_processing': ['mean', 'median', 'std', 'min', 'max'],
            'memory_usage': ['after_creation_mb', 'after_reset_mb', 'during_steps_mean_mb',
                             'during_steps_max_mb', 'creation_overhead_mb', 'reset_overhead_mb'],
            'full_episode': ['episode_time_s', 'turn_time_avg_ms', 'combat_time_avg_ms', 'step_time_avg_ms',
                             'agent_inference_time_mean', 'agent_overhead_percent']
        }


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


@pytest.mark.benchmark
def test_benchmark_with_agent_overhead():
    """Run benchmark with a mock agent to measure agent overhead."""
    print("🚀 Running Agent Overhead Benchmark...")
    mock_agent = MockAgent(compute_time=0.001, matrix_size=50)
    benchmark = EnvironmentBenchmark(
        num_iterations=5, num_steps_per_episode=10, agent_policy=mock_agent
    )
    benchmark.run_full_benchmark()
    benchmark.print_results()

    step = benchmark.results['environment_step']
    assert 'agent_inference_time_mean' in step
    assert 'agent_overhead_percent' in step
    assert step['agent_inference_time_mean'] > 0
    assert step['agent_overhead_percent'] > 0

    ep = benchmark.results['full_episode']
    assert 'agent_inference_time_mean' in ep
    assert 'agent_overhead_percent' in ep
    assert ep['agent_inference_time_mean'] > 0


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

    # Parse arguments
    compare_path = None
    save_path = None
    run_mode = "quick"
    use_agent = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--compare" and i + 1 < len(args):
            compare_path = args[i + 1]
            i += 2
        elif args[i] == "--save-path" and i + 1 < len(args):
            save_path = args[i + 1]
            i += 2
        elif args[i] == "--agent":
            use_agent = True
            i += 1
        elif args[i] == "comprehensive":
            run_mode = "comprehensive"
            i += 1
        else:
            i += 1

    # If comparing, load baseline and skip running benchmarks
    if compare_path:
        if not os.path.exists(compare_path):
            print(f"❌ Baseline file not found: {compare_path}")
            sys.exit(1)
        if not os.path.exists(save_path or BENCHMARK_RESULTS_PATH):
            print(f"❌ New results not found. Run a benchmark first to generate results.")
            sys.exit(1)

        baseline = EnvironmentBenchmark.load_results(compare_path)
        new_results = EnvironmentBenchmark.load_results(save_path or BENCHMARK_RESULTS_PATH)
        deltas = EnvironmentBenchmark.compare_results(new_results, baseline)
        EnvironmentBenchmark.print_comparison(deltas)
        sys.exit(0)

    # Run benchmark
    agent_policy = MockAgent(compute_time=0.001, matrix_size=50) if use_agent else None
    if run_mode == "comprehensive":
        benchmark = EnvironmentBenchmark(
            num_iterations=100, num_steps_per_episode=50, agent_policy=agent_policy
        )
    else:
        benchmark = EnvironmentBenchmark(
            num_iterations=10, num_steps_per_episode=20, agent_policy=agent_policy
        )

    benchmark.run_full_benchmark()
    benchmark.print_results()

    print("\n🎉 Benchmark complete!")