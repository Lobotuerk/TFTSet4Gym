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

    def translate(self, board_probs: torch.Tensor, player, shop_slots: list = None) -> list:
        target_board = self.decode_target_board(board_probs)[0]

        current_board = self.get_current_board(player).copy()
        bench_state = [None] * BENCH_SIZE
        for pos, idx in self.get_current_bench(player):
            bench_state[pos] = idx

        actions = []

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                target_c = int(target_board[y, x])
                current_c = int(current_board[y, x])

                if target_c == EMPTY_CLASS:
                    continue
                if current_c == target_c:
                    continue

                src_is_bench = None
                src_pos = None

                for bp, bi in enumerate(bench_state):
                    if bi == target_c:
                        src_is_bench = True
                        src_pos = bp
                        break

                if src_is_bench is None:
                    for by in range(BOARD_HEIGHT):
                        for bx in range(BOARD_WIDTH):
                            if by == y and bx == x:
                                continue
                            if int(current_board[by, bx]) == target_c:
                                src_is_bench = False
                                src_pos = by * BOARD_WIDTH + bx
                                break
                        if src_is_bench is not None:
                            break

                if src_is_bench is None:
                    continue

                dcord = y * BOARD_WIDTH + x
                current_c = int(current_board[y, x])

                if src_is_bench:
                    bench_dcord = src_pos + BOARD_SIZE
                    actions.append([1, bench_dcord, dcord])
                    bench_state[src_pos] = None
                    if current_c != EMPTY_CLASS:
                        bench_state[src_pos] = current_c
                    current_board[y, x] = target_c
                else:
                    src_y, src_x = divmod(src_pos, BOARD_WIDTH)
                    actions.append([1, src_pos, dcord])
                    current_board[y, x] = target_c
                    current_board[src_y, src_x] = current_c if current_c != EMPTY_CLASS else EMPTY_CLASS

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                current_c = int(current_board[y, x])
                target_c = int(target_board[y, x])

                if current_c == EMPTY_CLASS:
                    continue
                if target_c != EMPTY_CLASS:
                    continue

                needed = False
                for ty in range(BOARD_HEIGHT):
                    for tx in range(BOARD_WIDTH):
                        if (int(target_board[ty, tx]) == current_c
                                and int(current_board[ty, tx]) != current_c):
                            needed = True
                            break
                    if needed:
                        break

                if needed:
                    continue

                dcord = y * BOARD_WIDTH + x
                bench_vacancy = None
                for i in range(BENCH_SIZE):
                    if bench_state[i] is None:
                        bench_vacancy = i
                        break

                if bench_vacancy is not None:
                    bench_dcord = bench_vacancy + BOARD_SIZE
                    actions.append([1, dcord, bench_dcord])
                    bench_state[bench_vacancy] = current_c
                    current_board[y, x] = EMPTY_CLASS
                else:
                    actions.append([3, dcord, 0])
                    current_board[y, x] = EMPTY_CLASS

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
