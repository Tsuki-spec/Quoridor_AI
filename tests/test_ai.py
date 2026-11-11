"""Run with python3 -m unittest discover -s tests -v; no model or GPU required."""

import math
import random
import sys
import unittest
import numpy as np
from game import Quoridor
from ai.search import AlphaBetaAI, HeuristicMCTS, DIFFICULTIES, evaluate, paths, winning_actions
import app


class SearchTests(unittest.TestCase):
    def test_distance_cache_matches_rules(self):
        randomizer = random.Random(17)
        game = Quoridor()
        for _ in range(24):
            neighbors, distances = paths(game)
            for square, adjacent in enumerate(neighbors):
                origin = divmod(square, 9)
                expected = {row * 9 + col for row, col in
                            ((origin[0] + 1, origin[1]), (origin[0] - 1, origin[1]),
                             (origin[0], origin[1] + 1), (origin[0], origin[1] - 1))
                            if game.is_crossable(origin, (row, col))}
                self.assertEqual(set(adjacent), expected)
            for side in (0, 1):
                square = game.pawns[side][0] * 9 + game.pawns[side][1]
                self.assertEqual(distances[side][square], game.shortest_path_length(side))
            self.assertAlmostEqual(evaluate(game, 0), -evaluate(game, 1))
            if game.winner is not None:
                break
            game.play(randomizer.choice(game.legal_actions()))

    def test_both_players_take_goals_and_jumps(self):
        for engine_class in (AlphaBetaAI, HeuristicMCTS):
            for side in (0, 1):
                for jump in (False, True):
                    with self.subTest(engine = engine_class.__name__, side = side, jump = jump):
                        game = Quoridor()
                        game.current_player = side
                        if side == 0:
                            game.pawns = [(6 if jump else 7, 4), (7 if jump else 2, 4)]
                        else:
                            game.pawns = [(1 if jump else 6, 4), (2 if jump else 1, 4)]
                        before = game.position_key()
                        action = engine_class("easy", seed = 4).search(game)
                        self.assertEqual(game.position_key(), before)
                        game.play(action)
                        self.assertEqual(game.winner, side)

    def test_both_algorithms_block_immediate_loss(self):
        for engine_class in (AlphaBetaAI, HeuristicMCTS):
            for side in (0, 1):
                game = Quoridor()
                game.current_player = side
                game.pawns = [(3, 4), (1, 7)] if side == 0 else [(7, 7), (5, 4)]
                action = engine_class("normal", seed = 42).search(game)
                self.assertGreaterEqual(action, 81)
                game.play(action)
                self.assertFalse(winning_actions(game))

    def test_no_wall_game_finishes_without_oscillating(self):
        for engine_class in (AlphaBetaAI, HeuristicMCTS):
            game = Quoridor()
            game.walls_left = [0, 0]
            engine = engine_class("normal", seed = 42)
            history = {}
            for step in range(40):
                key = game.position_key()
                history[key] = history.get(key, 0) + 1
                side = game.current_player
                distance = game.shortest_path_length(side)
                game.play(engine.search(game, history))
                if step < 4:  # Before contact, neither pawn needs a sidestep to avoid a jump.
                    self.assertLess(game.shortest_path_length(side), distance)
                if game.winner is not None:
                    break
            self.assertIsNotNone(game.winner)
            self.assertLessEqual(max(history.values()), 3)

    def test_history_changes_preferences_without_mutating_rules(self):
        game = Quoridor()
        game.walls_left = [0, 0]
        repeated = game.copy()
        repeated.play(3)
        history = {repeated.position_key(): 5}
        for engine_class in (AlphaBetaAI, HeuristicMCTS):
            engine = engine_class("easy", seed = 1)
            before = game.position_key()
            original_history = history.copy()
            action = engine.search(game, history)
            self.assertIn(action, game.legal_actions())
            self.assertGreater(engine.repetition_cost(game, 3), 0)
            self.assertEqual(game.position_key(), before)
            self.assertEqual(history, original_history)

    def test_time_limit_keeps_legal_fallback(self):
        for engine_class in (AlphaBetaAI, HeuristicMCTS):
            game = Quoridor()
            engine = engine_class("expert", seed = 2)
            engine.config = {**engine.config, "seconds": 0.05}
            action = engine.search(game)
            self.assertIn(action, game.legal_actions())
            self.assertLess(engine.stats["seconds"], 0.35)
            if engine_class is HeuristicMCTS:
                self.assertGreater(engine.stats["simulations"], 0)

    def test_alpha_beta_matches_exhaustive_minimax(self):
        # Without walls the candidate set is complete, so pruning must preserve minimax values.
        def minimax(game, depth, seen):
            if game.winner is not None or depth == 0:
                return evaluate(game, game.current_player)
            best = -math.inf
            for row, col in game.pawn_moves():
                following = game.copy()
                following.play(row * 9 + col)
                key = following.position_key()
                value = 0 if key in seen else -minimax(following, depth - 1, seen | {key})
                best = max(best, value)
            return best
        for pawns in ([(0, 4), (8, 4)], [(2, 2), (6, 6)], [(3, 3), (5, 5)]):
            for side in (0, 1):
                game = Quoridor()
                game.pawns = pawns
                game.current_player = side
                game.walls_left = [0, 0]
                engine = AlphaBetaAI("normal")
                action = engine.search(game)
                following = game.copy()
                following.play(action)
                chosen = -minimax(following, 2, {game.position_key(), following.position_key()})
                self.assertEqual(chosen, minimax(game, 3, {game.position_key()}))
                self.assertEqual(engine.stats["depth"], 3)

    def test_finished_game_is_rejected(self):
        game = Quoridor()
        game.winner = 0
        for engine_class in (AlphaBetaAI, HeuristicMCTS):
            with self.assertRaises(ValueError):
                engine_class().search(game)


class WebAPITests(unittest.TestCase):
    def setUp(self):
        self.model = app.ACTIVE_MODEL
        app.ACTIVE_MODEL = None

    def tearDown(self):
        app.ACTIVE_MODEL = self.model
        app.SESSIONS.clear()

    def test_classical_algorithms_both_hands_and_undo(self):
        for algorithm in ("alpha_beta", "mcts"):
            for human in (0, 1):
                state = app.api_post("/api/new", {"mode": "ai", "algorithm": algorithm,
                                                "difficulty": "easy", "human_player": human})
                if human == 0:
                    state = app.api_post("/api/move", {"id": state["id"], "action": 13})
                state = app.api_post("/api/ai", {"id": state["id"]})
                self.assertTrue(state["human_turn"])
                self.assertIsNone(state["model"])
                self.assertEqual(state["ai_stats"]["algorithm"], algorithm)
                if human == 0:
                    state = app.api_post("/api/undo", {"id": state["id"]})
                    self.assertEqual(state["move_count"], 0)
                    self.assertIsNone(state["ai_stats"])

    def test_all_levels_and_option_validation(self):
        for algorithm in ("alpha_beta", "mcts"):
            for level in DIFFICULTIES:
                state = app.api_post("/api/new", {"mode": "ai", "algorithm": algorithm, "difficulty": level})
                self.assertEqual(state["difficulty"], level)
        for payload in ({"algorithm": "unknown"}, {"difficulty": "unknown"}, {"algorithm": []}, {"difficulty": {}}):
            with self.assertRaises(app.APIError):
                app.api_post("/api/new", {"mode": "ai", **payload})
        with self.assertRaisesRegex(app.APIError, "AlphaZero"):
            app.api_post("/api/new", {"mode": "ai", "algorithm": "alphazero"})

    def test_existing_alphazero_and_pvp_still_work(self):
        class FakeNetwork:
            def predict(self, state):
                return np.zeros(209), 0.0
        app.ACTIVE_MODEL = {"network": FakeNetwork(), "name": "test.weights.h5",
                            "residual_blocks": 4, "channels": 64, "parameters": 346144}
        state = app.api_post("/api/new", {"mode": "ai", "human_player": 1, "simulations": 2})
        self.assertEqual(state["algorithm"], "alphazero")
        state = app.api_post("/api/ai", {"id": state["id"]})
        self.assertTrue(state["human_turn"])
        self.assertEqual(state["ai_stats"]["simulations"], 2)
        state = app.api_post("/api/new", {"mode": "pvp"})
        state = app.api_post("/api/move", {"id": state["id"], "action": 13})
        state = app.api_post("/api/undo", {"id": state["id"]})
        self.assertEqual(state["move_count"], 0)
        self.assertNotIn("tensorflow", sys.modules)


if __name__ == "__main__":
    unittest.main()
