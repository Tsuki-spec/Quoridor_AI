"""
author: cms
date: 2026/11/10
File introduction: Model-free MCTS and iterative-deepening Alpha-Beta for Quoridor.
"""

import math
import random
import time
from collections import deque
from functools import lru_cache
from game import BOARD_SIZE, PAWN_ACTIONS, WALL_SIZE

WIN = 10000.0
DIFFICULTIES = {
    "easy": {"label": "入门", "seconds": 0.15, "simulations": 48, "depth": 1, "walls": 3, "rollout": 4},
    "normal": {"label": "标准", "seconds": 0.6, "simulations": 180, "depth": 3, "walls": 6, "rollout": 6},
    "hard": {"label": "困难", "seconds": 1.8, "simulations": 600, "depth": 5, "walls": 10, "rollout": 8},
    "expert": {"label": "挑战", "seconds": 4.0, "simulations": 1600, "depth": 7, "walls": 14, "rollout": 10},
}
ALGORITHMS = {
    "alpha_beta": {"label": "Alpha-Beta", "description": "最短路径评估与对抗搜索，无需模型即可对战。"},
    "mcts": {"label": "MCTS", "description": "通过多次模拟比较走法，无需模型即可对战。"},
    "alphazero": {"label": "AlphaZero", "description": "使用已训练的神经网络引导搜索，需要先载入权重。"},
}


@lru_cache(maxsize = 2048)
def board_paths(horizontal, vertical):  # Cache geometry by wall layout.
    blocked = set()
    for row, col in horizontal:
        for offset in (0, 1):
            start = row * BOARD_SIZE + col + offset
            blocked.add((start, start + BOARD_SIZE))
    for row, col in vertical:
        for offset in (0, 1):
            start = (row + offset) * BOARD_SIZE + col
            blocked.add((start, start + 1))
    neighbors = []
    for square in range(PAWN_ACTIONS):
        row, col = divmod(square, BOARD_SIZE)
        adjacent = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            target = nr * BOARD_SIZE + nc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and (min(square, target), max(square, target)) not in blocked:
                adjacent.append(target)
        neighbors.append(tuple(adjacent))
    distances = []
    for goal_row in (BOARD_SIZE - 1, 0):
        distance = [PAWN_ACTIONS] * PAWN_ACTIONS
        queue = deque(goal_row * BOARD_SIZE + col for col in range(BOARD_SIZE))
        for square in queue:
            distance[square] = 0
        while queue:
            square = queue.popleft()
            for target in neighbors[square]:
                if distance[target] == PAWN_ACTIONS:
                    distance[target] = distance[square] + 1
                    queue.append(target)
        distances.append(tuple(distance))
    return tuple(neighbors), tuple(distances)


def paths(game):
    return board_paths(frozenset(game.walls[0]), frozenset(game.walls[1]))


def pawn_actions(game):
    return sorted(row * BOARD_SIZE + col for row, col in game.pawn_moves())


def winning_actions(game):
    goal = BOARD_SIZE - 1 if game.current_player == 0 else 0
    return [action for action in pawn_actions(game) if action // BOARD_SIZE == goal]


def evaluate(game, player):  # Score from the requested player's perspective.
    if game.winner is not None:
        return WIN if game.winner == player else -WIN
    if winning_actions(game):
        return (WIN - 100) if game.current_player == player else -(WIN - 100)
    _, distances = paths(game)
    lengths = []
    for side in (0, 1):
        square = game.pawns[side][0] * BOARD_SIZE + game.pawns[side][1]
        length = distances[side][square]
        # Include currently available jumps.
        if sum(abs(a - b) for a, b in zip(game.pawns[0], game.pawns[1])) == 1:
            position = game.copy()
            position.current_player = side
            length = min(1 + distances[side][move] for move in pawn_actions(position))
        lengths.append(length)
    other = 1 - player
    score = 10.0 * (lengths[other] - lengths[player])
    score += 0.7 * (game.walls_left[player] - game.walls_left[other])
    return score + (0.5 if game.current_player == player else -0.5)


class SearchTimeout(Exception):
    pass


class SearchBase:
    def __init__(self, difficulty = "normal", seed = None):
        if difficulty not in DIFFICULTIES:
            raise ValueError("未知难度")
        self.config = DIFFICULTIES[difficulty]
        self.random = random.Random(seed)
        self.stats = {}

    def prepare(self, game, position_counts):
        if game.winner is not None:
            raise ValueError("终局没有合法动作")
        self.started = time.perf_counter()
        self.deadline = self.started + self.config["seconds"]
        self.root_player = game.current_player
        self.history = dict(position_counts or {})
        self.history[game.position_key()] = max(1, self.history.get(game.position_key(), 0))
        self.pawn_history = {}
        for key, count in self.history.items():
            pawn_key = (key[0][self.root_player], key[1], key[2], key[3])
            self.pawn_history[pawn_key] = self.pawn_history.get(pawn_key, 0) + count
        self.move_cache = {}
        self.nodes = 0
        self.stats = {"depth": 0, "simulations": 0}

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

    def repetition_cost(self, game, action):
        following = game.copy()
        following.play(action, validate = False)
        key = following.position_key()
        count = self.history.get(key, 0)
        if action < PAWN_ACTIONS:
            count = max(count, self.pawn_history.get((key[0][self.root_player], key[1], key[2], key[3]), 0))
        return min(18.0, 3.0 * count)

    def wall_candidates(self, game, full):
        if full:
            return range(PAWN_ACTIONS, PAWN_ACTIONS + 2 * WALL_SIZE * WALL_SIZE)
        neighbors, distances = paths(game)
        anchors = set()
        for player in (0, 1):
            square = game.pawns[player][0] * BOARD_SIZE + game.pawns[player][1]
            while 0 < distances[player][square] < PAWN_ACTIONS:
                target = min(neighbors[square], key = lambda cell: (distances[player][cell], abs(cell % BOARD_SIZE - 4)))
                row, col = divmod(square, BOARD_SIZE)
                nr, nc = divmod(target, BOARD_SIZE)
                if row != nr:
                    candidates = ((0, min(row, nr), col), (0, min(row, nr), col - 1))
                else:
                    candidates = ((1, row, min(col, nc)), (1, row - 1, min(col, nc)))
                for direction, wr, wc in candidates:
                    if 0 <= wr < WALL_SIZE and 0 <= wc < WALL_SIZE:
                        anchors.add(PAWN_ACTIONS + direction * WALL_SIZE * WALL_SIZE + wr * WALL_SIZE + wc)
                square = target
        return sorted(anchors)

    def candidates(self, game, full = False):  # Keep all pawn moves and the best candidate walls.
        key = (game.position_key(), full)
        if key in self.move_cache:
            return self.move_cache[key]
        player = game.current_player
        pawns, walls = [], []
        for action in pawn_actions(game):
            following = game.copy()
            following.play(action, validate = False)
            pawns.append((evaluate(following, player), action))
        if game.walls_left[player]:
            for action in self.wall_candidates(game, full):
                direction, anchor = divmod(action - PAWN_ACTIONS, WALL_SIZE * WALL_SIZE)
                row, col = divmod(anchor, WALL_SIZE)
                if game.can_place_wall(direction, row, col):
                    following = game.copy()
                    following.play(action, validate = False)
                    walls.append((evaluate(following, player), action))
        walls.sort(reverse = True)
        ranked = sorted(pawns + walls[:self.config["walls"]], reverse = True)
        result = [action for _, action in ranked]
        self.move_cache[key] = result
        return result

    def finish(self, action):
        self.stats.update(nodes = self.nodes, seconds = round(time.perf_counter() - self.started, 3))
        return int(action)


class AlphaBetaAI(SearchBase):
    def negamax(self, game, depth, alpha, beta, seen, ply):  # Child scores use the opponent's perspective.
        self.check_time()
        self.nodes += 1
        if game.winner is not None:
            return -WIN + ply  # The player to move has lost.
        if depth == 0:
            return evaluate(game, game.current_player)
        best = -math.inf
        for action in self.candidates(game):
            following = game.copy()
            following.play(action, validate = False)
            key = following.position_key()
            if key in seen:
                value = 0.0  # Treat search cycles as draws.
            else:
                seen.add(key)
                try:
                    value = -self.negamax(following, depth - 1, -beta, -alpha, seen, ply + 1)
                finally:
                    seen.remove(key)
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def search(self, game, position_counts = None):
        self.prepare(game, position_counts)
        wins = winning_actions(game)
        if wins:
            return self.finish(wins[0])
        actions = self.candidates(game, full = True)
        actions.sort(key = lambda move: self.root_score(game, move), reverse = True)
        best_action = actions[0]
        for depth in range(1, self.config["depth"] + 1):
            best, chosen = -math.inf, best_action
            try:
                for action in actions:
                    self.check_time()
                    following = game.copy()
                    following.play(action, validate = False)
                    cost = self.repetition_cost(game, action)
                    # Adjust the window for the root repetition cost.
                    value = -self.negamax(following, depth - 1, -math.inf, -best - cost,
                                          {game.position_key(), following.position_key()}, 1) - cost
                    if value > best:
                        best, chosen = value, action
                self.check_time()
            except SearchTimeout:
                break  # Keep the last completed iteration.
            best_action = chosen
            self.stats["depth"] = depth
            actions.remove(chosen)
            actions.insert(0, chosen)
            if best >= WIN - 100:
                break
        return self.finish(best_action)

    def root_score(self, game, action):
        following = game.copy()
        following.play(action, validate = False)
        return evaluate(following, game.current_player) - self.repetition_cost(game, action)


def bounded_value(score):
    if abs(score) >= WIN - 100:
        return math.copysign(1.0, score)
    return score / (60.0 + abs(score))


class RolloutNode:
    def __init__(self, game, player, prior = 1.0):
        self.game = game
        self.visits = 0
        self.value_sum = 0.0  # Root player's perspective.
        self.static = bounded_value(evaluate(game, player))
        self.estimate = self.static
        self.prior = prior
        self.children = []
        self.untried = None


class HeuristicMCTS(SearchBase):
    def prepare(self, game, position_counts):
        super().prepare(game, position_counts)
        self.ordered_cache = {}

    def ordered_moves(self, game, full = False):
        key = (game.position_key(), full)
        if key in self.ordered_cache:
            return self.ordered_cache[key]
        player = game.current_player
        ranked = []
        for action in self.candidates(game, full):
            following = game.copy()
            following.play(action, validate = False)
            ranked.append((evaluate(following, player), action))
        # Never allow a one-move loss when a legal defense exists.
        safe = [item for item in ranked if item[0] > -WIN + 100]
        ranked = safe or ranked
        ranked.sort(reverse = True)
        weights = [math.exp(max(-30.0, (score - ranked[0][0]) / 12.0)) for score, _ in ranked]
        total = sum(weights)
        result = [(action, weight / total) for (_, action), weight in zip(ranked, weights)]
        self.ordered_cache[key] = result
        return result

    def rollout(self, game, seen):
        for _ in range(self.config["rollout"]):
            if game.winner is not None or time.perf_counter() >= self.deadline:
                break
            moves = self.ordered_moves(game)
            following_moves = []
            for action, prior in moves:
                following = game.copy()
                following.play(action, validate = False)
                if following.position_key() not in seen or following.winner is not None:
                    following_moves.append((action, following))
            if not following_moves:
                return 0.0
            # Randomize equally good moves, rather than arbitrary top-three moves.
            best = evaluate(following_moves[0][1], game.current_player)
            tied = [item for item in following_moves if evaluate(item[1], game.current_player) >= best - 0.01]
            action, game = self.random.choice(tied)
            seen.add(game.position_key())
        return bounded_value(evaluate(game, self.root_player))

    def search(self, game, position_counts = None):
        self.prepare(game, position_counts)
        wins = winning_actions(game)
        if wins:
            return self.finish(wins[0])
        root = RolloutNode(game.copy(), self.root_player)
        root.untried = self.ordered_moves(game, full = True).copy()
        costs = {}
        for action, _ in root.untried:
            following = game.copy()
            following.play(action, validate = False)
            score = evaluate(following, self.root_player)
            costs[action] = bounded_value(score) - bounded_value(score - self.repetition_cost(game, action))
        root.untried.sort(key = lambda item: self.root_score(game, item[0], costs), reverse = True)
        fallback = root.untried[0][0]
        for _ in range(self.config["simulations"]):
            if time.perf_counter() >= self.deadline:
                break
            node, branch = root, [root]
            seen = {game.position_key()}
            cycle = False
            while node.game.winner is None:
                if node.untried is None:
                    node.untried = self.ordered_moves(node.game).copy()
                width = 2 + int(math.sqrt(node.visits))
                if node.untried and len(node.children) < width:
                    action, prior = node.untried.pop(0)
                    following = node.game.copy()
                    following.play(action, validate = False)
                    child = RolloutNode(following, self.root_player, prior)
                    node.children.append((action, child))
                    node = child
                    branch.append(node)
                    key = following.position_key()
                    cycle = key in seen
                    seen.add(key)
                    break
                sign = 1 if node.game.current_player == self.root_player else -1
                def uct(item):
                    action, child = item
                    mean = child.value_sum / child.visits
                    estimate = 0.65 * mean + 0.35 * child.estimate
                    exploration = 0.65 * math.sqrt(math.log(node.visits + 1) / child.visits)
                    bias = 0.5 * child.prior / (1 + child.visits)
                    return sign * estimate + exploration + bias - (costs[action] if node is root else 0.0)
                _, node = max(node.children, key = uct)
                branch.append(node)
                key = node.game.position_key()
                if key in seen:
                    cycle = True
                    break
                seen.add(key)
            value = 0.0 if cycle else self.rollout(node.game.copy(), seen)
            for visited in reversed(branch):
                visited.visits += 1
                visited.value_sum += value
                if visited.children and visited.game.winner is None:
                    sign = 1 if visited.game.current_player == self.root_player else -1
                    visited.estimate = 0.75 * sign * max(sign * child.estimate for _, child in visited.children) + 0.25 * visited.static
            self.nodes += 1
            self.stats["simulations"] += 1
            self.stats["depth"] = max(self.stats["depth"], len(branch) - 1)
        if not root.children:
            return self.finish(fallback)
        action, _ = max(root.children, key = lambda item: (
            0.65 * item[1].value_sum / item[1].visits + 0.35 * item[1].estimate - costs[item[0]], item[1].visits))
        return self.finish(action)

    def root_score(self, game, action, costs):
        following = game.copy()
        following.play(action, validate = False)
        return bounded_value(evaluate(following, game.current_player)) - costs[action]


def choose_action(game, algorithm = "alpha_beta", difficulty = "normal", position_counts = None):
    engine = {"alpha_beta": AlphaBetaAI, "mcts": HeuristicMCTS}[algorithm](difficulty)
    action = engine.search(game, position_counts)
    return action, {"algorithm": algorithm, "difficulty": difficulty, **engine.stats}
