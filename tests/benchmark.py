"""Paired matches with CPU search workers and batched GPU inference."""

import hashlib
import importlib.util
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from game import Quoridor
from search import AlphaBetaAI, DIFFICULTIES


class RemoteNetwork:
    def __init__(self, connection):
        self.connection = connection
        self.calls = 0

    def predict(self, state):
        self.calls += 1
        self.connection.send(("predict", state))
        return self.connection.recv()


def worker(connection, agents, openings, opponent, max_moves):
    try:
        engines = {}
        for name, agent in agents.items():
            spec = importlib.util.spec_from_file_location(name, agent["module"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            engines[name] = module.MCTS
        while True:
            task = connection.recv()
            if task is None:
                break
            name, case, side, seed = task
            np.random.seed(seed)
            game, history = Quoridor(), {}
            for action in openings[case]:
                history[game.position_key()] = history.get(game.position_key(), 0) + 1
                game.play(action)
            network = RemoteNetwork(connection)
            engine = engines[name](network, **agents[name].get("search", {}))
            rival = AlphaBetaAI(opponent, seed = seed)
            moves, statistics, backwards = [], [], 0
            started = time.perf_counter()
            while game.winner is None and len(openings[case]) + len(moves) < max_moves:
                player = game.current_player
                before = game.shortest_path_length(player)
                history[game.position_key()] = history.get(game.position_key(), 0) + 1
                if player == side:
                    action = int(np.argmax(engine.search(game, temperature = 0, position_counts = history)))
                else:
                    action = rival.search(game, history)
                    statistics.append(dict(rival.stats))
                game.play(action)
                backwards += int(player == side and action < 81 and game.shortest_path_length(player) > before)
                moves.append(action)
            connection.send(("done", {"agent": name, "case": case, "side": side, "seed": seed,
                "opening": openings[case], "moves": moves, "winner": game.winner,
                "won": game.winner == side, "opponent_stats": statistics, "backwards": backwards,
                "max_repetition": max(history.values()), "network_calls": network.calls,
                "seconds": time.perf_counter() - started}))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_arena(agents, openings, output, workers = 4, opponent = "normal", max_moves = 200, seed = 42):
    if not agents or not openings or not isinstance(workers, int) or workers <= 0:
        raise ValueError("模型、开局不能为空，进程数必须为正整数")
    for opening in openings:
        game = Quoridor()
        for action in opening:
            game.play(action)
        if game.winner is not None or len(opening) >= max_moves:
            raise ValueError("评测开局必须非终局，且短于步数上限")
    import tensorflow as tf
    from policy_value_net import PolicyValueNet
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    networks, network_keys = {}, {}
    report = {"agents": agents, "openings": openings, "opponent": {**DIFFICULTIES[opponent], "name": opponent},
              "max_moves": max_moves, "workers": workers, "seed": seed, "tf32": False,
              "tensorflow": tf.__version__, "source_sha256": {name: digest(BASE_DIR / name)
                  for name in ("game.py", "search.py", "mcts.py", "policy_value_net.py", "train.py", "app.py", "tests/benchmark.py")},
              "agent_sha256": {}, "agent_source": {name: Path(agent["module"]).read_text()
                  for name, agent in agents.items()}, "games": [], "status": "running"}
    for name, agent in agents.items():
        key = (agent["weights"], agent.get("blocks", 4), agent.get("channels", 64))
        network_keys[name] = key
        if key not in networks:
            network = PolicyValueNet(residual_blocks = key[1], channels = key[2])
            network.load(key[0])
            for size in range(1, workers + 1):
                network.predict_batch([Quoridor().state()] * size)
            networks[key] = network
        report["agent_sha256"][name] = {"weights": digest(agent["weights"]), "module": digest(agent["module"])}
    output = Path(output)
    output.parent.mkdir(parents = True, exist_ok = True)
    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii = False, indent = 2) + "\n")
        os.replace(temporary, output)
    save()
    tasks = [(name, case, side, seed) for case in range(len(openings)) for name in agents for side in (0, 1)]
    context = mp.get_context("spawn")
    active, processes, connections = {}, [], []
    started, last_update = time.perf_counter(), time.perf_counter()
    try:
        for _ in range(min(workers, len(tasks))):
            parent, child = context.Pipe()
            process = context.Process(target = worker, args = (child, agents, openings, opponent, max_moves))
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            active[parent] = tasks.pop(0)
            parent.send(active[parent])
        while active:
            ready = wait(list(active), timeout = 1)
            if ready:
                time.sleep(0.001)
                ready = wait(list(active), timeout = 0)
            pending, states = [], []
            for connection in ready:
                kind, data = connection.recv()
                if kind == "predict":
                    pending.append(connection)
                    states.append(data)
                elif kind == "done":
                    report["games"].append(data)
                    save()
                    print("DONE", data["agent"], data["case"], data["side"], data["won"], len(data["moves"]), flush = True)
                    if tasks:
                        active[connection] = tasks.pop(0)
                        connection.send(active[connection])
                    else:
                        connection.send(None)
                        del active[connection]
                else:
                    raise RuntimeError(data)
            for key, network in networks.items():
                indices = [index for index, connection in enumerate(pending) if network_keys[active[connection][0]] == key]
                if indices:
                    logits, values = network.predict_batch([states[index] for index in indices])
                    for index, policy, value in zip(indices, logits, values):
                        pending[index].send((policy, float(value)))
            if time.perf_counter() - last_update >= 30:
                print("PROGRESS", len(report["games"]), "games", round(time.perf_counter() - started), "seconds", flush = True)
                last_update = time.perf_counter()
        for process in processes:
            process.join(timeout = 5)
        for name, agent in agents.items():
            assert digest(agent["weights"]) == report["agent_sha256"][name]["weights"]
            assert digest(agent["module"]) == report["agent_sha256"][name]["module"]
        for name, checksum in report["source_sha256"].items():
            assert digest(BASE_DIR / name) == checksum, name
        report["status"] = "complete"
        report["seconds"] = time.perf_counter() - started
        save()
        return report
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout = 2)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in connections:
            connection.close()
