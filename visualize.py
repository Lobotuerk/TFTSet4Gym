#!/usr/bin/env python3
"""Terminal-based trajectory visualizer for TFTSet4Gym.

Usage:
    python visualize.py <trajectory.json>
"""

import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TFTSet4Gym.tft_set4_gym import parallel_env


def load_trajectory(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def star_str(stars):
    return "\u2605" * stars


def fmt_champion(champ):
    if champ is None:
        return "       "
    name = champ.name[:7].ljust(7)
    stars = star_str(champ.stars)
    chosen = "C" if champ.chosen else " "
    items = ",".join(champ.items) if champ.items else ""
    return f"{name}{stars}{chosen}"


def render_player(player_id, player, shop):
    lines = []
    sep = "-" * 60
    lines.append(f"  {player_id}  HP:{player.health:>3}  Lv:{player.level}  "
                 f"Gold:{player.gold:>2}  Units:{player.num_units_in_play}/{player.max_units}")
    lines.append(f"  Streaks: W{player.win_streak} L{-player.loss_streak}")
    lines.append(sep)
    lines.append("  Board (7x4):")
    for y in range(4):
        row = "  "
        for x in range(7):
            champ = player.board[x][y]
            row += "[" + fmt_champion(champ) + "] "
        lines.append(row)
    lines.append("  Bench:")
    bench_row = "  "
    for i in range(9):
        champ = player.bench[i]
        bench_row += "[" + fmt_champion(champ) + "] "
    lines.append(bench_row)
    lines.append("  Items:")
    item_row = "  "
    for i in range(10):
        item = player.item_bench[i]
        if item:
            item_row += f"[{str(item):>16}] "
        else:
            item_row += "[                ] "
    lines.append(item_row)
    if shop:
        shop_names = []
        for s in shop:
            shop_names.append(s if s != " " else "---")
        lines.append(f"  Shop: {', '.join(shop_names)}")
    lines.append("")
    return "\n".join(lines)


def render(env, step_idx, num_steps):
    header = f"=== Step {step_idx + 1} / {num_steps} ==="
    print(header)
    print("=" * len(header))

    shops = getattr(env.step_function, "shops", {})

    for player_id in env.possible_agents:
        player = env.PLAYERS.get(player_id)
        if player is None:
            continue
        shop = shops.get(player_id, [])
        alive = player_id in env.agents
        status = "ALIVE" if (alive and player.health > 0) else "DEAD"
        print(f"[{status}] ", end="")
        print(render_player(player_id, player, shop))


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <trajectory.json>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    trajectory = load_trajectory(filepath)

    seed = trajectory["seed"]
    steps = trajectory["steps"]
    metadata = trajectory.get("metadata", {})

    print(f"Loaded trajectory: {metadata.get('game_id', 'unknown')}")
    print(f"  Seed: {seed}")
    print(f"  Steps: {len(steps)}")
    print(f"  Players: {metadata.get('num_agents', '?')}")
    print()

    env = parallel_env()
    env.reset(seed=seed)

    delay = 0.8

    for i, step_data in enumerate(steps):
        actions = step_data["actions"]
        cls = "\n" * 2
        print(cls)
        render(env, i, len(steps))
        print(f"  Actions this step: {len(actions)}")
        if env.agents:
            env.step(actions)
        time.sleep(delay)

    print("\n" + "=" * 60)
    print("  FINAL STATE")
    print("=" * 60)
    render(env, len(steps), len(steps))
    env.close()
    print("Replay complete.")


if __name__ == "__main__":
    main()
