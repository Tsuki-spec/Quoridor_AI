"""Search regressions from the recorded maze and adversarial network outputs."""

import json
import unittest
import os
import random
from unittest.mock import patch
from pathlib import Path
import numpy as np
from game import Quoridor
import train
from mcts import MCTS, Node
from search import WIN, HeuristicMCTS, bounded_value, evaluate, winning_actions


class FixedNetwork:
    def __init__(self, preferred = 0, value = 0.0):
        self.preferred = preferred
        self.value = value
        self.calls = 0

    def predict(self, state):
        self.calls += 1
        logits = np.zeros(209, dtype = np.float32)
        logits[self.preferred] = 30.0
        return logits, self.value

    def predict_batch(self, states):
        predictions = [self.predict(state) for state in states]
        return np.asarray([row[0] for row in predictions]), np.asarray([row[1] for row in predictions])


class MCTSRegressionTests(unittest.TestCase):
    def test_tactical_value_matches_exhaustive_pawn_minimax(self):
        for side in (0, 1):
            for pawns in ([(0, 4), (8, 4)], [(6, 4), (2, 4)], [(3, 4), (4, 4)],
                          [(4, 5), (4, 4)], [(7, 4), (3, 1)]):
                game = Quoridor()
                game.pawns = pawns.copy()
                game.current_player = side
                game.walls_left = [0, 0]
                values = []
                for action in game.legal_actions():
                    following = game.copy()
                    following.play(action)
                    if following.winner is not None:
                        values.append(WIN)
                        continue
                    replies = []
                    for reply in following.legal_actions():
                        leaf = following.copy()
                        leaf.play(reply)
                        replies.append(evaluate(leaf, side))
                    values.append(min(replies))
                search = MCTS(FixedNetwork(), simulations = 2)
                self.assertEqual(search.tactical(game, game.legal_actions()), max(values))

    def test_tactical_candidates_preserve_all_pawn_moves_and_legality(self):
        game = Quoridor()
        randomizer = random.Random(73)
        search = MCTS(FixedNetwork(), simulations = 2)
        for _ in range(30):
            legal = game.legal_actions()
            short = set(search.short_actions(game))
            pawns = {row * 9 + col for row, col in game.pawn_moves()}
            self.assertTrue(pawns.issubset(short))
            self.assertTrue(short.issubset(legal))
            game.play(randomizer.choice(legal))
            if game.winner is not None:
                break

    def test_tactical_cache_does_not_blend_the_value_twice(self):
        game = Quoridor()
        network = FixedNetwork(value = -0.9)
        search = MCTS(network, simulations = 2)
        first, cached = Node(), Node()
        search.expand(first, game)
        search.expand(cached, game)
        self.assertEqual(network.calls, 1)
        self.assertEqual(first.evaluation, cached.evaluation)
        self.assertIsNone(first.solved)
        self.assertIsNone(cached.solved)

    def test_tactical_evaluation_can_be_disabled(self):
        game = Quoridor()
        search = MCTS(FixedNetwork(value = 0.25), simulations = 2, tactical_mix = 0)
        with patch.object(search, "tactical", side_effect = AssertionError("tactical called")):
            value = search.expand(Node(), game)
        expected = 0.85 * 0.25 + 0.15 * bounded_value(evaluate(game, game.current_player))
        self.assertAlmostEqual(value, expected)

    def test_three_ply_win_overrides_a_confident_losing_prior(self):
        for side, backward, winning in ((0, 49, 67), (1, 31, 13)):
            game = Quoridor()
            game.pawns = [(6, 4), (2, 4)]
            game.walls_left = [0, 0]
            game.current_player = side
            before = game.position_key()
            for temperature in (0, 0.1, 1, 4):
                search = MCTS(FixedNetwork(backward, -0.9), simulations = 8)
                policy = search.search(game, temperature = temperature)
                self.assertEqual(float(policy[winning]), 1)
                self.assertEqual(float(policy[backward]), 0)
            self.assertEqual(game.position_key(), before)

    def test_network_confidence_is_not_a_proof(self):
        game = Quoridor()
        search = MCTS(FixedNetwork(value = 1.0), simulations = 2)
        first, cached = Node(), Node()
        search.expand(first, game)
        search.expand(cached, game)
        self.assertIsNone(first.solved)
        self.assertIsNone(cached.solved)

    def test_cached_wall_checks_match_the_rules(self):
        randomizer = random.Random(991)
        for _ in range(3):
            game = Quoridor()
            for _ in range(40):
                before = game.position_key()
                actions = game.legal_actions()
                self.assertEqual(MCTS.legal_actions(game), actions)
                self.assertEqual(game.position_key(), before)
                if not actions:
                    break
                game.play(randomizer.choice(actions))
            self.assertEqual(MCTS.legal_actions(game), game.legal_actions())

    def test_cached_wall_checks_preserve_search_visits(self):
        game = Quoridor()
        settings = {"simulations": 64}
        with patch.object(MCTS, "legal_actions", side_effect = lambda state: state.legal_actions()):
            reference = MCTS(FixedNetwork(13, -0.25), **settings).search(game)
        actual = MCTS(FixedNetwork(13, -0.25), **settings).search(game)
        np.testing.assert_array_equal(actual, reference)

    def test_recorded_winning_maze_keeps_advancing(self):
        fixture = json.loads(Path(__file__).with_name("maze_record.json").read_text())
        game, history = Quoridor(), {}
        for turn, action in enumerate(fixture["actions"], 1):
            key = game.position_key()
            history[key] = history.get(key, 0) + 1
            if turn in fixture["regression_turns"]:
                before = game.position_key()
                distance = game.shortest_path_length(1)
                engine = HeuristicMCTS("easy", seed = 42)
                chosen = engine.search(game, history)
                self.assertEqual(game.position_key(), before)
                following = game.copy()
                following.play(chosen)
                self.assertLess(following.shortest_path_length(1), distance, f"turn {turn}")
            game.play(action)

    def test_large_advantages_remain_distinguishable(self):
        self.assertGreater(bounded_value(180), bounded_value(160))
        self.assertGreater(bounded_value(-160), bounded_value(-180))
        self.assertAlmostEqual(bounded_value(180), -bounded_value(-180))
        self.assertLess(bounded_value(180), 1)

    def test_network_cannot_override_immediate_defense(self):
        for side in (0, 1):
            game = Quoridor()
            game.current_player = side
            game.pawns = [(3, 4), (1, 7)] if side == 0 else [(7, 7), (5, 4)]
            preferred = min(row * 9 + col for row, col in game.pawn_moves())
            engine = MCTS(FixedNetwork(preferred, 0.99), simulations = 8)
            root = Node()
            engine.expand(root, game)
            self.assertNotIn(preferred, root.children)
            policy = engine.search(game, temperature = 0)
            game.play(int(np.argmax(policy)))
            self.assertFalse(winning_actions(game))

    def test_proven_win_overrides_a_wrong_network_value(self):
        game = Quoridor()
        game.pawns = [(7, 4), (4, 7)]
        root = Node()
        value = MCTS(FixedNetwork(value = -0.99), simulations = 2).expand(root, game)
        self.assertEqual(value, 1)
        self.assertEqual(root.solved, 1)
        self.assertTrue(all(action // 9 == 8 for action in root.children))

    def test_forced_loss_still_returns_a_legal_move(self):
        game = Quoridor()
        game.pawns = [(3, 4), (1, 7)]
        game.walls_left = [0, 0]
        engine = MCTS(FixedNetwork(value = 0.99), simulations = 8)
        root = Node()
        self.assertEqual(engine.expand(root, game), -1)
        self.assertEqual(root.solved, -1)
        policy = engine.search(game, temperature = 0)
        self.assertIn(int(np.argmax(policy)), game.legal_actions())

    def test_temperature_noise_and_history_preserve_inputs(self):
        game = Quoridor()
        game.walls_left = [0, 0]
        history = {game.position_key(): 3}
        before = game.position_key()
        np.random.seed(19)
        engine = MCTS(FixedNetwork(13), simulations = 12)
        for temperature in (0, 0.001, 1, 4):
            policy = engine.search(game, training = True, temperature = temperature, position_counts = history)
            self.assertTrue(np.isfinite(policy).all())
            self.assertAlmostEqual(float(policy.sum()), 1, places = 6)
            self.assertTrue(set(np.flatnonzero(policy)).issubset(game.legal_actions()))
        self.assertEqual(game.position_key(), before)
        self.assertEqual(history, {before: 3})

    def test_cached_evaluations_do_not_share_tree_nodes_or_stale_weights(self):
        game = Quoridor()
        game.walls_left = [0, 0]
        network = FixedNetwork(13)
        engine = MCTS(network, simulations = 4)
        first, second = Node(), Node()
        engine.expand(first, game)
        engine.expand(second, game)
        self.assertEqual(network.calls, 1)
        first.children[13].visits = 99
        self.assertEqual(second.children[13].visits, 0)
        engine.search(game)
        calls = network.calls
        engine.search(game)
        self.assertGreater(network.calls, calls)

    def test_parallel_self_play_uses_the_new_search(self):
        network = FixedNetwork(13)
        with patch.multiple(train, GAMES = 2, SELF_PLAY_WORKERS = 2, MAX_MOVES = 4,
                            SIMULATIONS = 2, HEURISTIC_MIX = 0.15, TACTICAL_MIX = 0.75):
            batches = list(train.game_batches(network, None))
        games, stats = batches[0]
        self.assertEqual(len(games), 2)
        self.assertGreater(stats["requests"], 0)
        self.assertEqual(len({game[2]["worker_pid"] for game in games}), 2)
        for samples, winner, info in games:
            self.assertNotEqual(info["worker_pid"], os.getpid())
            self.assertEqual(info["tactical_mix"], 0.75)
            self.assertEqual(len(samples), 4)
            for state, policy, value in samples:
                self.assertEqual(state.shape, (9, 9, 7))
                self.assertAlmostEqual(float(policy.sum()), 1, places = 6)
                self.assertTrue(np.isfinite(policy).all())
                self.assertTrue(np.isfinite(value))

    def test_invalid_search_settings(self):
        for settings in ({"simulations": 0}, {"simulations": 1.5}, {"c_puct": -1},
                         {"c_puct": float("nan")}, {"heuristic_mix": -0.1}, {"heuristic_mix": 1.1},
                         {"tactical_mix": -0.1}, {"tactical_mix": 1.1}, {"tactical_mix": float("nan")}):
            with self.assertRaises(ValueError):
                MCTS(FixedNetwork(), **settings)


if __name__ == "__main__":
    unittest.main()
