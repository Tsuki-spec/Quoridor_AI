"""
author: cms
date: 2026/10/11
File introduction: Local web interface for human-versus-AI and two-player Quoridor games.
"""

import json
import hashlib
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import numpy as np
from game import ACTION_SIZE, Quoridor
from mcts import MCTS
from search import ALGORITHMS, DIFFICULTIES, choose_action
from train import CHANNELS, C_PUCT, HEURISTIC_MIX, REPETITION_PENALTY, RESIDUAL_BLOCKS, ROOT_PRIOR_MIX

HOST = "127.0.0.1"
PORT = 8000
DEFAULT_SIMULATIONS = 128
DEFAULT_ALGORITHM = "alpha_beta"
DEFAULT_DIFFICULTY = "normal"
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "models"
RECORD_DIR = BASE_DIR / "records"
MAX_RECORD_BYTES = 2 * 1024 * 1024
DEFAULT_WATCH_MOVES = 400
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MODEL_LOCK = threading.RLock()
SESSION_LOCK = threading.RLock()
SESSIONS = {}
ACTIVE_MODEL = None
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class APIError(Exception):
    def __init__(self, message, status = 400):
        super().__init__(message)
        self.status = status


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise APIError(f"{name}必须是 {minimum} 至 {maximum} 之间的整数。")
    return value


def model_files():
    files = {}
    for directory in (BASE_DIR, UPLOAD_DIR, UPLOAD_DIR / "archive"):
        if directory.exists():
            for path in sorted(directory.glob("*.h5")):
                if path.is_file() and not path.name.startswith("~$") and path.resolve().parent == directory.resolve():
                    files[path.relative_to(BASE_DIR).as_posix()] = path
    return files


def model_info(model):
    if model is None:
        return None
    info = {key: model[key] for key in ("name", "residual_blocks", "channels", "parameters")}
    if "sha256" in model:
        info["sha256"] = model["sha256"]
    return info


def read_model(path, blocks, channels):
    integer(blocks, "残差块数", 1, 40)
    integer(channels, "通道数", 1, 512)
    with MODEL_LOCK:
        import tensorflow as tf
        from policy_value_net import PolicyValueNet
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        network = PolicyValueNet(residual_blocks = blocks, channels = channels)
        try:
            network.load(path)
            logits, value = network.predict(Quoridor().state())
        except (ValueError, OSError) as error:
            raise APIError(f"模型载入失败：{error}") from error
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise APIError("模型输出含有 NaN 或无穷大，请检查训练权重。")
        return {
            "network": network, "name": path.relative_to(BASE_DIR).as_posix(),
            "residual_blocks": blocks, "channels": channels, "parameters": network.model.count_params(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }


def load_model(path, blocks, channels):
    global ACTIVE_MODEL
    with MODEL_LOCK:
        ACTIVE_MODEL = read_model(path, blocks, channels)
        return model_info(ACTIVE_MODEL)


def agent_options(data, cache):
    if not isinstance(data, dict):
        raise APIError("AI 设置必须是对象。")
    algorithm = data.get("algorithm", DEFAULT_ALGORITHM)
    difficulty = data.get("difficulty", DEFAULT_DIFFICULTY)
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
        raise APIError("请选择 MCTS、Alpha-Beta 或 AlphaZero。")
    if not isinstance(difficulty, str) or difficulty not in DIFFICULTIES:
        raise APIError("请选择入门、标准、困难或挑战难度。")
    budget = {"easy": 64, "normal": 128, "hard": 256, "expert": 512}[difficulty]
    simulations = integer(data.get("simulations", budget), "搜索次数", 1, 2048)
    model = None
    if algorithm == "alphazero":
        name = data.get("model_path")
        if name is not None and not isinstance(name, str):
            raise APIError("模型路径必须是字符串。")
        if name:
            files = model_files()
            if not isinstance(name, str) or name not in files:
                raise APIError("找不到所选模型。")
            architecture = re.search(r"b(\d+)-c(\d+)", name)
            blocks = data.get("residual_blocks", int(architecture[1]) if architecture else RESIDUAL_BLOCKS)
            channels = data.get("channels", int(architecture[2]) if architecture else CHANNELS)
            key = (name, integer(blocks, "残差块数", 1, 40), integer(channels, "通道数", 1, 512))
            if key not in cache:
                cache[key] = read_model(files[name], blocks, channels)
            model = cache[key]
        else:
            model = ACTIVE_MODEL
        if model is None:
            raise APIError("AlphaZero 需要先载入权重，或在观战设置中选择模型文件。")
    return {"kind": "ai", "algorithm": algorithm, "difficulty": difficulty,
            "simulations": simulations, "model": model}


def agent_info(agent):
    if agent.get("kind") != "ai":
        return {"kind": "human"}
    return {"kind": "ai", "algorithm": agent["algorithm"], "difficulty": agent["difficulty"],
            "simulations": agent["simulations"], "model": model_info(agent.get("model"))}


def action_label(action):
    if action < 81:
        return f"{chr(65 + action % 9)}{action // 9 + 1}"
    direction, anchor = divmod(action - 81, 64)
    return f"{'H' if direction == 0 else 'V'}:{chr(65 + anchor % 8)}{anchor // 8 + 1}"


class Match:
    def __init__(self, mode, human_player, simulations, model, algorithm = "alphazero", difficulty = DEFAULT_DIFFICULTY,
                 players = None, max_moves = DEFAULT_WATCH_MOVES):
        self.id = uuid.uuid4().hex
        self.game = Quoridor()
        self.mode = mode
        self.human_player = human_player
        self.simulations = simulations
        self.algorithm = algorithm
        self.difficulty = difficulty
        self.max_moves = max_moves
        self.end_reason = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.saved_record = None
        self.revision = 0
        self.saved_revision = None
        self.record_error = None
        # Each match keeps its selected model.
        self.model = model
        self.players = players or [{"kind": "human"}, {"kind": "human"}]
        if mode == "ai":
            self.players[1 - human_player] = {"kind": "ai", "algorithm": algorithm,
                                            "difficulty": difficulty, "simulations": simulations, "model": model}
        self.history = []
        self.moves = []
        self.lock = threading.RLock()

    def undo_count(self):  # Return to the previous human turn in AI mode.
        if not self.history:
            return 0
        if self.mode in ("pvp", "watch") or self.game.current_player != self.human_player:
            return 1
        return 2 if len(self.history) >= 2 else 0

    def state(self):
        with self.lock:
            game = self.game
            human_turn = not self.finished() and (self.mode == "pvp" or self.mode == "ai" and game.current_player == self.human_player)
            latest_search = next((move["search"] for move in reversed(self.moves) if move.get("search")), None)
            return {
                "id": self.id, "mode": self.mode, "human_player": self.human_player,
                "current_player": game.current_player, "winner": game.winner, "pawns": game.pawns,
                "walls": [sorted(items) for items in game.walls], "walls_left": game.walls_left,
                "legal_actions": game.legal_actions() if human_turn else [], "human_turn": human_turn,
                "can_undo": self.undo_count() > 0, "moves": self.moves.copy(), "move_count": len(self.moves),
                "model": model_info(self.model), "simulations": self.simulations,
                "algorithm": self.algorithm, "difficulty": self.difficulty, "ai_stats": latest_search,
                "players": [agent_info(agent) for agent in self.players], "max_moves": self.max_moves,
                "finished": self.finished(), "end_reason": "goal" if game.winner is not None else self.end_reason,
                "record_saved": self.saved_revision == self.revision and self.saved_record is not None,
                "saved_record": self.saved_record, "record_error": self.record_error
            }

    def finished(self):
        return self.game.winner is not None or self.end_reason is not None

    def record(self):
        return {
            "format": "quoridor-record", "version": 1, "id": self.id, "created_at": self.created_at,
            "saved_at": datetime.now(timezone.utc).isoformat(), "mode": self.mode,
            "human_player": self.human_player, "players": [agent_info(agent) for agent in self.players],
            "max_moves": self.max_moves, "winner": self.game.winner,
            "end_reason": "goal" if self.game.winner is not None else self.end_reason or "unfinished",
            "moves": [{**move, "notation": action_label(move["action"])} for move in self.moves]
        }

    def save_record(self):
        RECORD_DIR.mkdir(parents = True, exist_ok = True)
        path = RECORD_DIR / f"{self.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record(), ensure_ascii = False, indent = 2), encoding = "utf-8")
        temporary.replace(path)
        self.saved_record = path.name
        self.saved_revision = self.revision
        self.record_error = None
        return path.name

    def commit(self, action, seconds = None, search = None):
        if self.finished():
            raise APIError("对局已经结束，请重新开始。", 409)
        if action not in self.game.legal_actions():
            raise APIError("这个动作不合法：墙不能重叠、交叉或封死任何一方的通路。", 409)
        self.history.append(self.game.copy())
        self.moves.append({"player": self.game.current_player, "action": action, "seconds": seconds})
        if search is not None:
            self.moves[-1]["search"] = search
        self.game.play(action, validate = False)
        self.revision += 1
        if self.mode == "watch" and self.game.winner is None and len(self.moves) >= self.max_moves:
            self.end_reason = "move_limit"
        if self.finished():
            try:
                self.save_record()
            except OSError as error:
                self.record_error = f"自动保存失败：{error}"


def replay_record(record):
    if not isinstance(record, dict) or record.get("format") != "quoridor-record" or type(record.get("version")) is not int or record["version"] != 1:
        raise APIError("请选择版本 1 的 Quoridor JSON 棋谱。")
    moves = record.get("moves")
    if not isinstance(moves, list) or len(moves) > 2000:
        raise APIError("棋谱必须包含动作列表，最多支持 2000 步。")
    game = Quoridor()
    frames = []
    normalized = []
    def snapshot():
        return {"pawns": game.pawns.copy(), "walls": [sorted(items) for items in game.walls],
                "walls_left": game.walls_left.copy(), "current_player": game.current_player, "winner": game.winner}
    frames.append(snapshot())
    for index, move in enumerate(moves):
        if not isinstance(move, dict):
            raise APIError(f"棋谱第 {index + 1} 步格式错误。")
        action = integer(move.get("action"), "动作编号", 0, ACTION_SIZE - 1)
        player = integer(move.get("player"), "棋手", 0, 1)
        seconds = move.get("seconds")
        if seconds is not None and (type(seconds) not in (int, float) or not np.isfinite(seconds) or not 0 <= seconds <= 86400):
            raise APIError("棋谱中的思考时间无效。")
        if player != game.current_player or action not in game.legal_actions():
            raise APIError(f"棋谱第 {index + 1} 步不合法。")
        entry = {"player": player, "action": action, "seconds": seconds, "notation": action_label(action)}
        search = move.get("search")
        if isinstance(search, dict) and isinstance(search.get("algorithm"), str) and search["algorithm"] in ALGORITHMS:
            entry["search"] = {"algorithm": search["algorithm"]}
            for key in ("depth", "simulations", "nodes"):
                if type(search.get(key)) is int and 0 <= search[key] <= 100000000:
                    entry["search"][key] = search[key]
        normalized.append(entry)
        game.play(action, validate = False)
        frames.append(snapshot())
    winner = record.get("winner")
    if winner is not None and type(winner) is not int or winner != game.winner:
        raise APIError("棋谱声明的胜者与实际落子结果不一致。")
    players = []
    supplied = record.get("players", [])
    for side in (0, 1):
        value = supplied[side] if isinstance(supplied, list) and side < len(supplied) and isinstance(supplied[side], dict) else {}
        if value.get("kind") == "ai" and isinstance(value.get("algorithm"), str) and value["algorithm"] in ALGORITHMS:
            players.append({"kind": "ai", "algorithm": value["algorithm"],
                            "difficulty": value.get("difficulty") if isinstance(value.get("difficulty"), str) and value["difficulty"] in DIFFICULTIES else DEFAULT_DIFFICULTY})
            model = value.get("model")
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                metadata = {"name": model["name"][:256]}
                for key in ("residual_blocks", "channels", "parameters"):
                    if type(model.get(key)) is int and 0 < model[key] <= 1000000000:
                        metadata[key] = model[key]
                if isinstance(model.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
                    metadata["sha256"] = model["sha256"]
                players[-1]["model"] = metadata
        else:
            players.append({"kind": "human"})
    cap = record.get("max_moves", DEFAULT_WATCH_MOVES)
    capped = record.get("mode") == "watch" and record.get("end_reason") == "move_limit" and type(cap) is int and 20 <= cap <= 1000 and len(normalized) == cap
    if game.winner is None and record.get("end_reason") == "move_limit" and not capped:
        raise APIError("棋谱步数与声明的观战上限不一致。")
    if game.winner is None and record.get("end_reason") == "goal":
        raise APIError("棋谱中没有到达目标行的动作。")
    clean = {"format": "quoridor-record", "version": 1, "mode": record.get("mode") if record.get("mode") in ("ai", "pvp", "watch") else "pvp",
             "human_player": record.get("human_player") if type(record.get("human_player")) is int and record.get("human_player") in (0, 1) else 0,
             "created_at": str(record.get("created_at", ""))[:80], "players": players, "moves": normalized,
             "winner": game.winner, "max_moves": cap if type(cap) is int and 20 <= cap <= 1000 else DEFAULT_WATCH_MOVES,
             "end_reason": "goal" if game.winner is not None else "move_limit" if capped else "unfinished"}
    return {"record": clean, "frames": frames}


def saved_records():
    result = []
    if RECORD_DIR.exists():
        for path in sorted(RECORD_DIR.glob("*.json"), key = lambda item: item.stat().st_mtime, reverse = True)[:100]:
            if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name) or path.resolve().parent != RECORD_DIR.resolve() or path.stat().st_size > MAX_RECORD_BYTES:
                continue
            try:
                record = json.loads(path.read_text(encoding = "utf-8"))
                if record.get("format") == "quoridor-record":
                    result.append({"name": path.name, "mode": record.get("mode"), "moves": len(record.get("moves", [])),
                                   "winner": record.get("winner"), "end_reason": record.get("end_reason"),
                                   "saved_at": record.get("saved_at", "")})
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    return result


def get_match(identifier):
    with SESSION_LOCK:
        match = SESSIONS.get(identifier) if isinstance(identifier, str) else None
    if match is None:
        raise APIError("对局不存在或服务已重启，请重新开始。", 404)
    return match


def api_post(route, data):
    if route == "/api/record/import":
        return replay_record(data.get("record"))
    if route == "/api/model/load":
        files = model_files()
        if data.get("path") not in files:
            raise APIError("找不到该模型，请刷新模型列表或上传权重文件。")
        return {"model": load_model(files[data["path"]], data.get("residual_blocks"), data.get("channels"))}
    if route == "/api/new":
        mode = data.get("mode", "pvp")
        if mode not in ("ai", "pvp", "watch"):
            raise APIError("请选择人机对战、双人对战或 AI 观战。")
        human = integer(data.get("human_player", 0), "先后手", 0, 1)
        if mode == "watch":
            specifications = data.get("players", [{"algorithm": "alpha_beta"}, {"algorithm": "mcts"}])
            if not isinstance(specifications, list) or len(specifications) != 2:
                raise APIError("观战需要两位 AI 的设置。")
            cache = {}
            players = [agent_options(specification, cache) for specification in specifications]
            cap = integer(data.get("max_moves", DEFAULT_WATCH_MOVES), "观战步数上限", 20, 1000)
            match = Match(mode, human, DEFAULT_SIMULATIONS, None, players = players, max_moves = cap)
            with SESSION_LOCK:
                SESSIONS[match.id] = match
            return match.state()
        algorithm = data.get("algorithm", "alphazero" if ACTIVE_MODEL else DEFAULT_ALGORITHM)
        difficulty = data.get("difficulty", DEFAULT_DIFFICULTY)
        if not isinstance(algorithm, str) or algorithm not in ALGORITHMS:
            raise APIError("请选择 MCTS、Alpha-Beta 或 AlphaZero。")
        if not isinstance(difficulty, str) or difficulty not in DIFFICULTIES:
            raise APIError("请选择入门、标准、困难或挑战难度。")
        neural_budgets = {"easy": 64, "normal": 128, "hard": 256, "expert": 512}
        simulations = integer(data.get("simulations", neural_budgets[difficulty]), "搜索次数", 1, 2048)
        selected_model = ACTIVE_MODEL if mode == "ai" and algorithm == "alphazero" else None
        if mode == "ai" and algorithm == "alphazero" and selected_model is None:
            raise APIError("AlphaZero 需要先载入权重；也可以选择无需模型的 MCTS 或 Alpha-Beta。")
        match = Match(mode, human, simulations, selected_model, algorithm, difficulty)
        with SESSION_LOCK:
            SESSIONS[match.id] = match
        return match.state()
    match = get_match(data.get("id"))
    with match.lock:
        if route == "/api/record/save":
            filename = match.save_record()
            return {"filename": filename, "record": match.record(), "state": match.state()}
        elif route == "/api/move":
            if match.mode == "watch":
                raise APIError("观战模式由双方 AI 落子。", 409)
            if match.mode == "ai" and match.game.current_player != match.human_player:
                raise APIError("现在轮到 AI，请等待它完成思考。", 409)
            match.commit(integer(data.get("action"), "动作编号", 0, ACTION_SIZE - 1))
        elif route == "/api/ai":
            if match.mode not in ("ai", "watch") or match.mode == "ai" and match.game.current_player == match.human_player:
                raise APIError("当前没有需要执行的 AI 回合。", 409)
            if match.finished():
                raise APIError("对局已经结束。", 409)
            started = time.perf_counter()
            # Rebuild repetition counts from undoable history.
            visits = {}
            for position in [*match.history, match.game]:
                key = position.position_key()
                visits[key] = visits.get(key, 0) + 1
            agent = match.players[match.game.current_player]
            if agent["algorithm"] == "alphazero":
                with MODEL_LOCK:
                    policy = MCTS(agent["model"]["network"], agent["simulations"], C_PUCT,
                                  prior_mix = ROOT_PRIOR_MIX, repetition_penalty = REPETITION_PENALTY, heuristic_mix = HEURISTIC_MIX).search(
                        match.game, training = False, temperature = 0, position_counts = visits)
                action = int(np.argmax(policy))
                search = {"algorithm": "alphazero", "difficulty": agent["difficulty"], "simulations": agent["simulations"]}
            else:
                action, search = choose_action(match.game, agent["algorithm"], agent["difficulty"], visits)
            match.commit(action, round(time.perf_counter() - started, 3), search)
        elif route == "/api/undo":
            count = match.undo_count()
            if count == 0:
                raise APIError("当前没有可以悔棋的动作。", 409)
            for _ in range(count):
                match.game = match.history.pop()
                match.moves.pop()
            match.end_reason = None
            match.revision += 1
        else:
            raise APIError("接口不存在。", 404)
        return match.state()


class Handler(BaseHTTPRequestHandler):
    def respond(self, value, status = 200, content_type = "application/json; charset=utf-8"):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii = False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        address = urlsplit(self.path)
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/static/style.css": ("style.css", "text/css; charset=utf-8"),
            "/static/app.js": ("app.js", "text/javascript; charset=utf-8")
        }
        try:
            if address.path in assets:
                name, content_type = assets[address.path]
                self.respond((WEB_DIR / name).read_bytes(), content_type = content_type)
            elif address.path == "/api/status":
                self.respond({"models": list(model_files()), "model": model_info(ACTIVE_MODEL),
                              "algorithms": ALGORITHMS, "difficulties": DIFFICULTIES, "defaults": {
                             "residual_blocks": RESIDUAL_BLOCKS, "channels": CHANNELS, "simulations": DEFAULT_SIMULATIONS,
                             "algorithm": DEFAULT_ALGORITHM, "difficulty": DEFAULT_DIFFICULTY}})
            elif address.path == "/api/state":
                self.respond(get_match(parse_qs(address.query).get("id", [None])[0]).state())
            elif address.path == "/api/records":
                self.respond({"records": saved_records()})
            elif address.path == "/api/record":
                name = parse_qs(address.query).get("name", [""])[0]
                if not re.fullmatch(r"[0-9a-f]{32}\.json", name):
                    raise APIError("棋谱文件名无效。")
                path = RECORD_DIR / name
                if not path.is_file() or path.resolve().parent != RECORD_DIR.resolve():
                    raise APIError("找不到该棋谱。", 404)
                if path.stat().st_size > MAX_RECORD_BYTES:
                    raise APIError("棋谱文件超过 2 MiB。", 413)
                self.respond(replay_record(json.loads(path.read_text(encoding = "utf-8"))))
            else:
                raise APIError("页面不存在。", 404)
        except APIError as error:
            self.respond({"error": str(error)}, error.status)
        except (ValueError, TypeError) as error:
            self.respond({"error": f"棋谱格式错误：{error}"}, 400)
        except OSError as error:
            self.respond({"error": f"读取文件失败：{error}"}, 500)

    def do_POST(self):
        address = urlsplit(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = MAX_UPLOAD_BYTES if address.path == "/api/model/upload" else MAX_RECORD_BYTES if address.path == "/api/record/import" else 16384
            if not 0 < length <= limit:
                raise APIError("请求为空或文件过大；模型上传上限为 128 MiB。", 413)
            if address.path == "/api/model/upload":
                params = parse_qs(address.query)
                original = params.get("name", [""])[0]
                if not original.lower().endswith(".h5"):
                    raise APIError("请选择 .h5 或 .weights.h5 文件。")
                name = re.sub(r"[^\w.\-]", "_", original)[-140:]
                name = name[:-3] + ".h5"
                UPLOAD_DIR.mkdir(parents = True, exist_ok = True)
                path = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}-{name}"
                try:
                    path.write_bytes(self.rfile.read(length))
                    loaded = load_model(path, int(params.get("residual_blocks", ["0"])[
                                        0]), int(params.get("channels", ["0"])[0]))
                except Exception:
                    path.unlink(missing_ok = True)
                    raise
                self.respond({"model": loaded, "models": list(model_files())})
            else:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise APIError("请求需要使用 application/json。", 415)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise APIError("请求参数必须是 JSON 对象。")
                self.respond(api_post(address.path, data))
        except APIError as error:
            self.respond({"error": str(error)}, error.status)
        except (ValueError, TypeError) as error:
            self.respond({"error": f"参数格式错误：{error}"}, 400)
        except Exception as error:
            self.respond({"error": f"操作失败：{error}"}, 500)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Quoridor 网页已启动：http://{HOST}:{PORT}；按 Ctrl+C 结束。", flush = True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n网页服务已停止。", flush = True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
