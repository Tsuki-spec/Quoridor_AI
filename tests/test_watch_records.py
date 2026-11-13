"""Watch-mode and JSON-record regression tests; no model or GPU required."""

import copy
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import numpy as np
import app

GOAL_MOVES = [13, 67, 22, 58, 31, 49, 40, 31, 49, 22, 58, 13, 67, 4]


class WatchRecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix = "quoridor-record-test-")
        self.folder = Path(self.directory.name)
        self.record_patch = patch.object(app, "RECORD_DIR", self.folder)
        self.record_patch.start()
        self.model = app.ACTIVE_MODEL
        app.ACTIVE_MODEL = None

    def tearDown(self):
        app.SESSIONS.clear()
        app.ACTIVE_MODEL = self.model
        self.record_patch.stop()
        self.directory.cleanup()

    def new_watch(self, **options):
        return app.api_post("/api/new", {"mode": "watch", "players": [
            {"algorithm": "alpha_beta", "difficulty": "easy"},
            {"algorithm": "mcts", "difficulty": "easy"}], **options})

    def finished_pvp(self):
        state = app.api_post("/api/new", {"mode": "pvp"})
        for action in GOAL_MOVES:
            state = app.api_post("/api/move", {"id": state["id"], "action": action})
        return state

    def test_watch_agents_alternate_and_human_moves_are_rejected(self):
        state = self.new_watch()
        self.assertEqual([a["algorithm"] for a in state["players"]], ["alpha_beta", "mcts"])
        self.assertFalse(state["human_turn"])
        with self.assertRaises(app.APIError):
            app.api_post("/api/move", {"id": state["id"], "action": 13})
        for _ in range(4):
            state = app.api_post("/api/ai", {"id": state["id"]})
        self.assertEqual([move["player"] for move in state["moves"]], [0, 1, 0, 1])
        self.assertEqual([move["search"]["algorithm"] for move in state["moves"]], ["alpha_beta", "mcts"] * 2)
        state = app.api_post("/api/undo", {"id": state["id"]})
        self.assertEqual(state["move_count"], 3)

    def test_watch_cap_autosaves_and_undo_reopens_game(self):
        def oscillate(game, *args):
            side = game.current_player
            action = game.pawns[side][0] * 9 + (3 if game.pawns[side][1] == 4 else 4)
            return action, {"algorithm": "alpha_beta", "depth": 1}
        state = self.new_watch(max_moves = 20)
        with patch.object(app, "choose_action", side_effect = oscillate):
            for _ in range(20):
                state = app.api_post("/api/ai", {"id": state["id"]})
        self.assertTrue(state["finished"])
        self.assertIsNone(state["winner"])
        self.assertEqual(state["end_reason"], "move_limit")
        self.assertTrue(state["record_saved"])
        saved = self.folder / state["saved_record"]
        record = json.loads(saved.read_text())
        replay = app.replay_record(record)
        self.assertEqual(len(replay["frames"]), 21)
        self.assertEqual(replay["record"]["end_reason"], "move_limit")
        self.assertEqual(app.saved_records()[0]["end_reason"], "move_limit")
        invalid = copy.deepcopy(record)
        invalid["max_moves"] = 21
        with self.assertRaises(app.APIError):
            app.replay_record(invalid)
        with self.assertRaises(app.APIError):
            app.api_post("/api/ai", {"id": state["id"]})
        state = app.api_post("/api/undo", {"id": state["id"]})
        self.assertFalse(state["finished"])
        self.assertFalse(state["record_saved"])
        self.assertEqual(state["move_count"], 19)
        app.api_post("/api/record/save", {"id": state["id"]})
        self.assertEqual(len(list(self.folder.glob("*.json"))), 1)
        self.assertEqual(len(json.loads(saved.read_text())["moves"]), 19)

    def test_goal_autosave_and_roundtrip_replay(self):
        state = self.finished_pvp()
        self.assertEqual(state["winner"], 1)
        self.assertTrue(state["record_saved"])
        record = json.loads((self.folder / state["saved_record"]).read_text())
        result = app.api_post("/api/record/import", {"record": record})
        self.assertEqual(len(result["frames"]), len(GOAL_MOVES) + 1)
        self.assertEqual(result["frames"][0]["pawns"], [(0, 4), (8, 4)])
        self.assertEqual(result["frames"][-1]["winner"], 1)
        self.assertEqual(result["record"]["moves"][-1]["notation"], "E1")
        self.assertEqual(app.get_match(state["id"]).game.winner, 1)

    def test_manual_save_preserves_dirty_state_across_a_cycle(self):
        state = app.api_post("/api/new", {"mode": "pvp"})
        saved = app.api_post("/api/record/save", {"id": state["id"]})
        self.assertTrue(saved["state"]["record_saved"])
        for action in (3, 75, 4, 76):
            state = app.api_post("/api/move", {"id": state["id"], "action": action})
        self.assertEqual(state["pawns"], [(0, 4), (8, 4)])
        self.assertFalse(state["record_saved"])
        self.assertEqual(len(json.loads((self.folder / saved["filename"]).read_text())["moves"]), 0)

    def test_record_rejects_illegal_moves_and_forged_results(self):
        state = self.finished_pvp()
        original = json.loads((self.folder / state["saved_record"]).read_text())
        mutations = []
        for field, value in (("winner", 0), ("version", 2), ("moves", [True])):
            record = copy.deepcopy(original)
            record[field] = value
            mutations.append(record)
        for field, value in (("action", True), ("action", 80), ("player", 1), ("seconds", float("nan"))):
            record = copy.deepcopy(original)
            record["moves"][0][field] = value
            mutations.append(record)
        record = copy.deepcopy(original)
        record["moves"].append({"player": 0, "action": 76})
        mutations.append(record)
        for record in mutations:
            with self.assertRaises(app.APIError):
                app.replay_record(record)

    def test_neural_players_can_use_different_models_without_changing_active(self):
        class FakeNetwork:
            def predict(self, state):
                return np.zeros(209), 0.0
        def read(path, blocks, channels):
            return {"network": FakeNetwork(), "name": path.name, "residual_blocks": blocks,
                    "channels": channels, "parameters": 346144, "sha256": "a" * 64}
        files = {"a-b4-c64.h5": Path("a-b4-c64.h5"), "b-b4-c64.h5": Path("b-b4-c64.h5")}
        sentinel = read(files["a-b4-c64.h5"], 4, 64)
        app.ACTIVE_MODEL = sentinel
        with patch.object(app, "model_files", return_value = files), patch.object(app, "read_model", side_effect = read) as load:
            state = self.new_watch(players = [
                {"algorithm": "alphazero", "model_path": name, "simulations": 2} for name in files])
            self.assertEqual(load.call_count, 2)
            self.assertEqual([p["model"]["name"] for p in state["players"]], list(files))
            for _ in range(2):
                state = app.api_post("/api/ai", {"id": state["id"]})
            self.assertEqual(state["moves"][-1]["search"]["tactical_mix"], app.TACTICAL_MIX)
            self.assertIs(app.ACTIVE_MODEL, sentinel)
            same = [{"algorithm": "alphazero", "model_path": "a-b4-c64.h5"}] * 2
            load.reset_mock()
            self.new_watch(players = same)
            self.assertEqual(load.call_count, 1)

    def test_invalid_watch_settings(self):
        for options in ({"players": []}, {"players": [None, {}]}, {"max_moves": True}, {"max_moves": 19},
                        {"players": [{"algorithm": []}, {}]}, {"players": [{"algorithm": "alphazero"}, {}]}):
            with self.assertRaises(app.APIError):
                self.new_watch(**options)

    def test_replay_never_loads_a_model(self):
        state = self.finished_pvp()
        record = json.loads((self.folder / state["saved_record"]).read_text())
        record["players"][0] = {"kind": "ai", "algorithm": "alphazero", "difficulty": "easy",
                                 "model": {"name": "../../private.weights.h5", "sha256": "b" * 64}}
        with patch.object(app, "read_model") as load:
            replay = app.replay_record(record)
            load.assert_not_called()
        self.assertEqual(replay["record"]["players"][0]["model"]["sha256"], "b" * 64)
        self.assertNotIn("tensorflow", sys.modules)

    def test_saved_record_http_routes(self):
        state = self.finished_pvp()
        class QuietHandler(app.Handler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        thread = threading.Thread(target = server.serve_forever, daemon = True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def get(path):
            try:
                response = opener.open(f"http://127.0.0.1:{server.server_port}" + path)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                return response.status, json.load(response)
        try:
            code, data = get("/api/records")
            self.assertEqual(code, 200)
            self.assertEqual(len(data["records"]), 1)
            code, replay = get("/api/record?" + urllib.parse.urlencode({"name": state["saved_record"]}))
            self.assertEqual(code, 200)
            self.assertEqual(replay["frames"][-1]["winner"], 1)
            self.assertEqual(get("/api/record?name=../../app.py")[0], 400)
            (self.folder / state["saved_record"]).write_text("not json")
            self.assertEqual(get("/api/record?name=" + state["saved_record"])[0], 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout = 3)


if __name__ == "__main__":
    unittest.main()
