import torch
import numpy as np
from ..stats import COST

BOARD_HEIGHT = 4
BOARD_WIDTH = 7
BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH
NUM_CHAMPIONS = 58
EMPTY_CLASS = NUM_CHAMPIONS
BENCH_SIZE = 9


class ActionTranslationModule:
    def __init__(self):
        champ_names = list(COST.keys())
        self.idx_to_name = {}
        for i, name in enumerate(champ_names):
            if name != " " and COST[name] > 0:
                idx = i - 1
                self.idx_to_name[idx] = name
        self.name_to_idx = {v: k for k, v in self.idx_to_name.items()}

    def decode_target_board(self, board_probs: torch.Tensor) -> np.ndarray:
        target = board_probs.argmax(dim=1).cpu().numpy()
        return target

    def get_current_board(self, player) -> np.ndarray:
        current = np.full((BOARD_HEIGHT, BOARD_WIDTH), EMPTY_CLASS, dtype=int)
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                champ = player.board[x][y]
                if champ and champ.name in self.name_to_idx:
                    current[y, x] = self.name_to_idx[champ.name]
        return current

    def get_current_bench(self, player) -> list:
        bench = []
        for i in range(BENCH_SIZE):
            champ = player.bench[i]
            if champ and champ.name in self.name_to_idx:
                bench.append((i, self.name_to_idx[champ.name]))
        return bench

    def _find_bench_vacancy(self, player, used_positions: set) -> int:
        for i in range(BENCH_SIZE):
            if i not in used_positions and player.bench[i] is None:
                return i
        return None

    def translate(self, board_probs: torch.Tensor, player, shop_slots: list = None) -> list:
        target_board = self.decode_target_board(board_probs)[0]
        current_board = self.get_current_board(player)
        bench = self.get_current_bench(player)

        actions = []
        used_from_bench = set()
        placed_on_board = set()

        bench_inventory = {}
        for bench_pos, champ_idx in bench:
            bench_inventory.setdefault(champ_idx, []).append(bench_pos)

        target_positions = {}
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                t = target_board[y, x]
                if t != EMPTY_CLASS:
                    target_positions.setdefault(int(t), []).append((x, y))

        with_pending_board = set()
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                target_champ = int(target_board[y, x])
                current_champ = int(current_board[y, x])
                if target_champ == EMPTY_CLASS or current_champ == target_champ:
                    continue
                if current_champ == EMPTY_CLASS:
                    if target_champ in bench_inventory and bench_inventory[target_champ]:
                        bench_pos = bench_inventory[target_champ].pop(0)
                        used_from_bench.add(bench_pos)
                        dcord = y * BOARD_WIDTH + x
                        bench_dcord = bench_pos + BOARD_SIZE
                        actions.append([1, bench_dcord, dcord])
                        placed_on_board.add((x, y))

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if (x, y) in placed_on_board:
                    continue
                target_champ = int(target_board[y, x])
                current_champ = int(current_board[y, x])

                if target_champ == EMPTY_CLASS and current_champ != EMPTY_CLASS:
                    dcord = y * BOARD_WIDTH + x
                    actions.append([3, dcord, 0])

                elif target_champ != EMPTY_CLASS and current_champ != target_champ:
                    if target_champ in bench_inventory and bench_inventory[target_champ]:
                        bench_pos = bench_inventory[target_champ].pop(0)
                        used_from_bench.add(bench_pos)
                        dcord = y * BOARD_WIDTH + x
                        bench_dcord = bench_pos + BOARD_SIZE
                        actions.append([1, bench_dcord, dcord])
                        placed_on_board.add((x, y))

        remaining_bench = [(pos, idx) for pos, idx in bench if pos not in used_from_bench]
        for bench_pos, champ_idx in remaining_bench:
            for y in range(BOARD_HEIGHT):
                for x in range(BOARD_WIDTH):
                    if (x, y) in placed_on_board:
                        continue
                    if int(target_board[y, x]) == champ_idx:
                        dcord = y * BOARD_WIDTH + x
                        bench_dcord = bench_pos + BOARD_SIZE
                        actions.append([1, bench_dcord, dcord])
                        placed_on_board.add((x, y))
                        break
                else:
                    continue
                break

        if not actions:
            actions.append([0, 0, 0])

        return actions

    def translate_batch(self, board_probs: torch.Tensor, players: list, shop_slots: list = None) -> list:
        batch_size = board_probs.shape[0]
        results = []
        for i in range(batch_size):
            if i < len(players):
                probs = board_probs[i : i + 1]
                results.append(self.translate(probs, players[i], shop_slots))
            else:
                results.append([[0, 0, 0]])
        return results
