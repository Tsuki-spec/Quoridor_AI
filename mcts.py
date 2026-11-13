"""
author: cms
date: 2026/10/11
File introduction: Monte Carlo Tree Search guided by a policy-value network.
"""

import math
import numpy as np
from game import ACTION_SIZE, BOARD_SIZE, PAWN_ACTIONS, WALL_SIZE
from search import WIN, board_paths, bounded_value, evaluate


class Node:
    def __init__(self, prior = 0.0):
        self.prior = float(prior)
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.evaluation = 0.0
        self.ranked = []
        self.solved = None

    def mean_value(self):  # Value for the player to move.
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, network, simulations = 50, c_puct = 1.5, prior_mix = 0.05, repetition_penalty = 0.2, heuristic_mix = 0.15):
        if not np.isfinite(prior_mix) or not 0 <= prior_mix <= 1:
            raise ValueError("prior_mix 必须在 [0, 1] 范围内")
        if not np.isfinite(repetition_penalty) or repetition_penalty < 0:
            raise ValueError("repetition_penalty 必须是有限非负数")
        if not np.isfinite(heuristic_mix) or not 0 <= heuristic_mix <= 1:
            raise ValueError("heuristic_mix 必须在 [0, 1] 范围内")
        if not isinstance(simulations, int) or simulations <= 0 or not np.isfinite(c_puct) or c_puct < 0:
            raise ValueError("simulations 必须是正整数，c_puct 必须是有限非负数")
        self.heuristic_mix = heuristic_mix
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.prior_mix = prior_mix
        self.repetition_penalty = repetition_penalty
        self.evaluations = {}

    @staticmethod
    def legal_actions(game):  # Reuse distance maps for wall legality and leaf evaluation.
        if game.winner is not None:
            return []
        actions = sorted(row * BOARD_SIZE + col for row, col in game.pawn_moves())
        if game.walls_left[game.current_player] <= 0:
            return actions
        horizontal, vertical = (frozenset(walls) for walls in game.walls)
        first, second = (row * BOARD_SIZE + col for row, col in game.pawns)
        for direction in (0, 1):
            walls = horizontal if direction == 0 else vertical
            for row in range(WALL_SIZE):
                for col in range(WALL_SIZE):
                    anchor = (row, col)
                    if anchor in horizontal or anchor in vertical:
                        continue
                    adjacent = ((row, col - 1), (row, col + 1)) if direction == 0 else ((row - 1, col), (row + 1, col))
                    if any(point in walls for point in adjacent):
                        continue
                    h = horizontal | {anchor} if direction == 0 else horizontal
                    v = vertical | {anchor} if direction == 1 else vertical
                    _, distances = board_paths(h, v)
                    if distances[0][first] < PAWN_ACTIONS and distances[1][second] < PAWN_ACTIONS:
                        actions.append(PAWN_ACTIONS + direction * WALL_SIZE * WALL_SIZE + row * WALL_SIZE + col)
        return actions

    @staticmethod
    def balanced_prior(actions, noise = False):
        groups = [[action for action in actions if action < PAWN_ACTIONS],
                  [action for action in actions if action >= PAWN_ACTIONS]]
        groups = [group for group in groups if group]
        probabilities = {}
        for group in groups:
            weights = np.random.dirichlet([0.3] * len(group)) if noise else np.full(len(group), 1.0 / len(group))
            for action, weight in zip(group, weights):
                probabilities[action] = float(weight) / len(groups)
        return probabilities

    def expand(self, node, game):
        if game.winner is not None:
            # The player to move has lost.
            node.solved = -1.0
            return -1.0
        key = game.position_key()
        if key in self.evaluations:
            priors, node.evaluation, node.solved = self.evaluations[key]
            node.children = {action: Node(prior) for action, prior in priors}
            node.ranked = sorted(node.children.items(), key = lambda item: item[1].prior, reverse = True)
            return node.evaluation
        actions = self.legal_actions(game)
        logits, value = self.network.predict(game.state())
        estimates = []
        for action in actions:
            following = game.copy()
            following.play(action, validate = False)
            estimates.append(evaluate(following, game.current_player))
        safe = [index for index, score in enumerate(estimates) if score > -WIN + 100]
        wins = [index for index, score in enumerate(estimates) if score >= WIN]
        keep = wins or safe or list(range(len(actions)))
        actions = [actions[index] for index in keep]
        estimates = np.asarray([estimates[index] for index in keep], dtype = np.float64)
        scores = np.asarray(logits[actions], dtype = np.float64)
        scores -= np.max(scores)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities)
        guidance = np.exp(np.maximum(-30.0, (estimates - estimates.max()) / 12.0))
        guidance /= guidance.sum()
        probabilities = (1 - self.heuristic_mix) * probabilities + self.heuristic_mix * guidance
        for action, probability in zip(actions, probabilities):
            node.children[action] = Node(probability)
        if wins:
            node.evaluation = node.solved = 1.0
        elif not safe:
            node.evaluation = node.solved = -1.0
        else:
            node.evaluation = float((1 - self.heuristic_mix) * value + self.heuristic_mix * bounded_value(evaluate(game, game.current_player)))
        self.evaluations[key] = (tuple(zip(actions, probabilities)), node.evaluation, node.solved)
        node.ranked = sorted(node.children.items(), key = lambda item: item[1].prior, reverse = True)
        return node.evaluation

    def search(self, game, training = False, temperature = 1.0, position_counts = None):
        if game.winner is not None:
            raise ValueError("终局没有合法动作")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须是有限非负数")
        goal_row = BOARD_SIZE - 1 if game.current_player == 0 else 0
        winning_moves = [row * BOARD_SIZE + col for row, col in sorted(game.pawn_moves()) if row == goal_row]
        if winning_moves:
            probabilities = np.zeros(ACTION_SIZE, dtype = np.float32)
            probabilities[winning_moves] = 1.0 / len(winning_moves) if temperature > 0 else 0.0
            if temperature == 0:
                probabilities[winning_moves[0]] = 1.0
            return probabilities
        self.evaluations = {}  # Weights may change between searches during training.
        root = Node()
        self.expand(root, game)
        actions = list(root.children)
        balanced = self.balanced_prior(actions)
        for action in actions:
            root.children[action].prior = (1 - self.prior_mix) * root.children[action].prior + self.prior_mix * balanced[action]
        if training and len(root.children) > 1:
            noise = self.balanced_prior(actions, noise = True)
            for action in actions:
                root.children[action].prior = 0.75 * root.children[action].prior + 0.25 * noise[action]
        root.ranked = sorted(root.children.items(), key = lambda item: item[1].prior, reverse = True)
        history = dict(position_counts or {})
        root_key = game.position_key()
        history[root_key] = max(1, history.get(root_key, 0))
        pawn_visits = {}
        player = game.current_player
        for key, count in history.items():
            # Also count pawn repetition while the opponent moves.
            pawn_key = (key[0][player], key[1], key[2], key[3])
            pawn_visits[pawn_key] = pawn_visits.get(pawn_key, 0) + count
        repeat_costs = {}
        for action in actions:
            following = game.copy()
            following.play(action, validate = False)
            key = following.position_key()
            repeats = history.get(key, 0)
            if action < PAWN_ACTIONS:
                pawn_key = (key[0][player], key[1], key[2], key[3])
                repeats = max(repeats, pawn_visits.get(pawn_key, 0))
            score = evaluate(following, player)
            repeat_costs[action] = self.repetition_penalty * 6 * repeats / (6 + abs(score) / 10)
        for _ in range(self.simulations):
            if root.solved is not None:
                break
            position = game.copy()
            node = root
            path = [root]
            seen = {root_key}
            cycle = False
            while node.children and node.solved is None:
                total = sum(child.visits for child in node.children.values())
                scale = math.sqrt(total + 1)
                # Child values use the opponent's perspective.
                width = 8 + int(2 * math.sqrt(node.visits))
                candidates = [item for index, item in enumerate(node.ranked) if index < width or item[0] < PAWN_ACTIONS]
                wins = [item for item in candidates if item[1].solved == -1]
                unknown = [item for item in candidates if item[1].solved != 1]
                candidates = wins or unknown or candidates
                action, child = max(
                    candidates,
                    key = lambda item: (
                        (-item[1].mean_value() if item[1].visits else node.evaluation - 0.15)
                        + self.c_puct * item[1].prior * scale / (1 + item[1].visits)
                        - (repeat_costs[item[0]] if node is root else 0.0)
                    )
                )
                position.play(action, validate = False)
                node = child
                path.append(node)
                key = position.position_key()
                if key in seen:
                    cycle = True
                    break
                seen.add(key)
            # Evaluate search cycles as draws.
            value = node.solved if node.solved is not None else 0.0 if cycle else self.expand(node, position)
            for visited in reversed(path):
                if visited.solved is None and visited.children:
                    if any(child.solved == -1 for child in visited.children.values()):
                        visited.solved = 1.0
                    elif all(child.solved == 1 for child in visited.children.values()):
                        visited.solved = -1.0
                if visited.solved is not None:
                    value = visited.solved
                visited.visits += 1
                visited.value_sum += value
                # Switch perspective at each parent.
                value = -value
        counts = np.zeros(ACTION_SIZE, dtype = np.float32)
        wins = {action for action, child in root.children.items() if child.solved == -1}
        safe = {action for action, child in root.children.items() if child.solved != 1}
        keep = wins or safe or set(root.children)
        for action, child in root.children.items():
            if action in keep:
                counts[action] = child.visits
        if not counts.any():
            for action, child in root.children.items():
                if action in keep:
                    counts[action] = child.prior
        if temperature == 0:
            probabilities = np.zeros(ACTION_SIZE, dtype = np.float32)
            probabilities[int(np.argmax(counts))] = 1.0
            return probabilities
        # Log-space scaling avoids overflow at low temperatures.
        probabilities = np.zeros(ACTION_SIZE, dtype = np.float64)
        positive = counts > 0
        scores = np.log(counts[positive].astype(np.float64))
        scores = (scores - np.max(scores)) / temperature
        probabilities[positive] = np.exp(scores)
        return (probabilities / probabilities.sum()).astype(np.float32)
