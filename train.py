"""
author: cms
date: 2026/10/11
File introduction: AlphaZero self-play training from the standard Quoridor opening.
"""

import json
import multiprocessing as mp
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import signal
import time
import traceback
from collections import deque
from multiprocessing.connection import wait
from pathlib import Path
import numpy as np
from game import Quoridor
from mcts import MCTS


MODE = "train"  # "train" for self-play; "play" for terminal human-versus-AI.
GAMES = 1600  # Number of new self-play games in this run.
SELF_PLAY_WORKERS = 16  # Parallel self-play processes; use 1 for serial self-play.
INFERENCE_WAIT_MS = 2.0  # Maximum wait in milliseconds for merging inference requests.
SIMULATIONS = 400  # MCTS simulations per action.
C_PUCT = 1.5  # Exploration coefficient used by training and web matches.
ROOT_PRIOR_MIX = 0.05  # Mix type-balanced exploration into the root prior, including during evaluation.
HEURISTIC_MIX = 0.15  # Path guidance for priors and leaf values; 0 disables this blend.
TACTICAL_MIX = 0.5  # Blend two-ply tactical estimates into leaf values; 0 disables this search.
REPETITION_PENALTY = 0.2  # Root cost for repeated full states or pawn positions under the same wall setup.
UPDATES = 8  # Gradient updates per generated game.
BATCH_SIZE = 64  # Training batch size; reduced while the replay buffer is small.
REPLAY_SIZE = 20000  # Maximum number of recent position samples.
MAX_MOVES = 400  # Total actions by both players; reaching the cap is a draw.
DISTANCE_VALUE_WEIGHT = 0.5  # Strength of capped-game targets; 0 disables distance scoring.
DISTANCE_VALUE_SCALE = 6.0  # Larger scales make tanh react more slowly to path-distance gaps.
CURRENT_DISTANCE_MIX = 0.25  # Capped samples blend this fraction of their own position's distance score.
DECISIVE_GAME_WEIGHT = 4.0  # Give real winning games more replay weight; each game's weight is length-normalized.
LEARNING_RATE = 0.0003  # Adam learning rate.
RESIDUAL_BLOCKS = 4  # Residual blocks in the shared trunk.
CHANNELS = 64  # Convolution channels in the shared trunk.
TEMPERATURE = 1.0  # Action-sampling temperature for the opening moves.
TEMPERATURE_MOVES = 20  # Opening temperature duration, counted in total actions.
LATE_TEMPERATURE = 0.25  # Reduce late-game wandering; MCTS targets stay at temperature 1.
SEED = 42  # Random seed; each parallel game receives its own derived seed.
MODEL_PATH = Path(__file__).resolve().parent / "models" / f"quoridor-resnet-b{RESIDUAL_BLOCKS}-c{CHANNELS}.weights.h5"
RESUME = True  # Load existing weights before training when True.
TRACE_DIR = None  # Optional JSON trace directory; None disables trace output.


def validate_config():
    if MODE not in ("train", "play"):
        raise ValueError('MODE 必须是 "train" 或 "play"')
    if any(not isinstance(value, int) or value <= 0 for value in (GAMES, SELF_PLAY_WORKERS, SIMULATIONS, UPDATES, BATCH_SIZE, REPLAY_SIZE, MAX_MOVES, RESIDUAL_BLOCKS, CHANNELS)):
        raise ValueError(
            "GAMES、SELF_PLAY_WORKERS、SIMULATIONS、UPDATES、BATCH_SIZE、REPLAY_SIZE、MAX_MOVES、RESIDUAL_BLOCKS、CHANNELS 必须是正整数")
    if not np.isfinite(INFERENCE_WAIT_MS) or INFERENCE_WAIT_MS < 0:
        raise ValueError("INFERENCE_WAIT_MS 必须是有限非负数")
    if not np.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0 or not np.isfinite(C_PUCT) or C_PUCT < 0:
        raise ValueError("LEARNING_RATE 必须是有限正数，C_PUCT 必须是有限非负数")
    if not isinstance(TEMPERATURE_MOVES, int) or TEMPERATURE_MOVES < 0 or any(not np.isfinite(value) or value < 0 for value in (TEMPERATURE, LATE_TEMPERATURE)):
        raise ValueError("TEMPERATURE_MOVES 必须是非负整数，两个温度必须是有限非负数")
    if not isinstance(SEED, int) or not 0 <= SEED < 2 ** 32:
        raise ValueError("SEED 必须是 0 到 2**32-1 之间的整数")
    validate_distance_config(DISTANCE_VALUE_WEIGHT, DISTANCE_VALUE_SCALE)
    if not np.isfinite(ROOT_PRIOR_MIX) or not 0 <= ROOT_PRIOR_MIX <= 1:
        raise ValueError("ROOT_PRIOR_MIX 必须在 [0, 1] 范围内")
    if not np.isfinite(HEURISTIC_MIX) or not 0 <= HEURISTIC_MIX <= 1:
        raise ValueError("HEURISTIC_MIX 必须在 [0, 1] 范围内")
    if not np.isfinite(TACTICAL_MIX) or not 0 <= TACTICAL_MIX <= 1:
        raise ValueError("TACTICAL_MIX 必须在 [0, 1] 范围内")
    if not np.isfinite(REPETITION_PENALTY) or REPETITION_PENALTY < 0:
        raise ValueError("REPETITION_PENALTY 必须是有限非负数")
    if not np.isfinite(CURRENT_DISTANCE_MIX) or not 0 <= CURRENT_DISTANCE_MIX <= 1:
        raise ValueError("CURRENT_DISTANCE_MIX 必须在 [0, 1] 范围内")
    if not np.isfinite(DECISIVE_GAME_WEIGHT) or DECISIVE_GAME_WEIGHT <= 0:
        raise ValueError("DECISIVE_GAME_WEIGHT 必须是有限正数")


def validate_distance_config(weight, scale):
    if not np.isfinite(weight) or not 0 <= weight < 1:
        raise ValueError("DISTANCE_VALUE_WEIGHT 必须是 [0, 1) 范围内的有限数")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("DISTANCE_VALUE_SCALE 必须是有限正数")


def position_key(game):
    return game.position_key()


def sampling_policy(policy, temperature):
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("落子温度必须是有限非负数")
    probabilities = np.zeros_like(policy, dtype = np.float64)
    if temperature == 0:
        probabilities[int(np.argmax(policy))] = 1.0
        return probabilities
    positive = np.asarray(policy) > 0
    scores = np.log(np.asarray(policy, dtype = np.float64)[positive])
    scores = (scores - np.max(scores)) / temperature
    probabilities[positive] = np.exp(scores)
    return probabilities / probabilities.sum()


def self_play(search, max_moves, temperature = 1.0, late_temperature = 1.0, temperature_moves = 20,
              return_info = False, distance_value_weight = 0.5, distance_value_scale = 6.0, current_distance_mix = 0.25):
    validate_distance_config(distance_value_weight, distance_value_scale)
    if not np.isfinite(current_distance_mix) or not 0 <= current_distance_mix <= 1:
        raise ValueError("current_distance_mix 必须在 [0, 1] 范围内")
    game = Quoridor()
    history = []
    visits = {position_key(game): 1}
    actions = []
    repeated_positions = 0
    entropies = []
    pawn_policy_masses = []
    for move_number in range(max_moves):
        if game.winner is not None:
            break
        # Sampling temperature does not change policy targets.
        policy = search.search(game, training = True, temperature = 1.0, position_counts = visits)
        player = game.current_player
        progress = 0.0
        if current_distance_mix > 0:
            own_distance = game.shortest_path_length(player)
            rival_distance = game.shortest_path_length(1 - player)
            progress = float(distance_value_weight * np.tanh((rival_distance - own_distance) / distance_value_scale))
        history.append((game.state(), policy, player, progress))
        pawn_policy_masses.append(float(policy[:81].sum()))
        positive = policy[policy > 0]
        entropies.append(float(-np.sum(positive * np.log(positive))))
        move_temperature = temperature if move_number < temperature_moves else late_temperature
        probabilities = sampling_policy(policy, move_temperature)
        action = int(np.random.choice(len(probabilities), p = probabilities))
        game.play(action, validate = False)
        actions.append(action)
        key = position_key(game)
        visits[key] = visits.get(key, 0) + 1
        repeated_positions += int(visits[key] > 1)
    # Capped games use weaker distance-based targets.
    distances = [game.shortest_path_length(player) for player in (0, 1)]
    if any(distance is None for distance in distances):
        raise ValueError("非法局面：至少一方没有到达目标行的通路")
    if game.winner is not None:
        value0 = 1.0 if game.winner == 0 else -1.0
    else:
        value0 = float(distance_value_weight * np.tanh((distances[1] - distances[0]) / distance_value_scale))
    samples = []
    for state, policy, player, progress in history:
        final_value = value0 if player == 0 else -value0
        outcome = final_value if game.winner is not None else (1 - current_distance_mix) * final_value + current_distance_mix * progress
        samples.append((state, policy, outcome))
    info = {
        "start_mode": "standard", "end_reason": "goal" if game.winner is not None else "max_moves",
        "winner": game.winner, "steps": len(actions), "repeated_positions": repeated_positions,
        "max_repetition": max(visits.values()), "final_pawns": list(game.pawns),
        "walls_left": list(game.walls_left), "final_distances": distances,
        "value_target_player0": value0,
        "value_source": "outcome" if game.winner is not None else "distance_blend" if current_distance_mix > 0 else "distance",
        "distance_value_weight": distance_value_weight, "distance_value_scale": distance_value_scale,
        "current_distance_mix": current_distance_mix,
        "pawn_move_fraction": sum(action < 81 for action in actions) / len(actions) if actions else 0.0,
        "opening_pawn_move_fraction": sum(action < 81 for action in actions[:20]) / min(len(actions), 20) if actions else 0.0,
        "opening_pawn_policy_mass": float(np.mean(pawn_policy_masses[:20])) if pawn_policy_masses else 0.0,
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0, "actions": actions
    }
    return (samples, game.winner, info) if return_info else (samples, game.winner)


class RemoteNetwork:
    def __init__(self, connection):
        self.connection = connection

    def predict(self, state):
        self.connection.send(("predict", state))
        kind, logits, value = self.connection.recv()
        if kind != "prediction":
            raise RuntimeError("收到无效的网络推理回复")
        return logits, value


def self_play_worker(connection):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            task = connection.recv()
            np.random.seed(task["seed"])
            search = MCTS(RemoteNetwork(connection), task["simulations"], task["c_puct"],
                          prior_mix = task["root_prior_mix"], repetition_penalty = task["repetition_penalty"],
                          heuristic_mix = task["heuristic_mix"], tactical_mix = task["tactical_mix"])
            started = time.perf_counter()
            samples, winner, info = self_play(
                search, task["max_moves"], task["temperature"], task["late_temperature"], task["temperature_moves"],
                return_info = True, distance_value_weight = task["distance_value_weight"],
                distance_value_scale = task["distance_value_scale"], current_distance_mix = task["current_distance_mix"])
            info.update({"episode": task["episode"], "seed": task["seed"],
                        "worker_pid": os.getpid(), "self_play_seconds": time.perf_counter() - started,
                        "tactical_mix": search.tactical_mix})
            connection.send(("result", samples, winner, info))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def game_batches(network, search):
    if SELF_PLAY_WORKERS == 1:
        for episode in range(1, GAMES + 1):
            started = time.perf_counter()
            samples, winner, info = self_play(
                search, MAX_MOVES, TEMPERATURE, LATE_TEMPERATURE, TEMPERATURE_MOVES, return_info = True,
                distance_value_weight = DISTANCE_VALUE_WEIGHT, distance_value_scale = DISTANCE_VALUE_SCALE,
                current_distance_mix = CURRENT_DISTANCE_MIX)
            info.update({"episode": episode, "seed": SEED, "worker_pid": os.getpid(),
                        "self_play_seconds": time.perf_counter() - started})
            yield [(samples, winner, info)], {"seconds": time.perf_counter() - started, "requests": 0, "batches": 0, "max_batch": 0}
        return
    # Spawn keeps CUDA state out of workers.
    context = mp.get_context("spawn")
    connections, processes = [], []
    try:
        for _ in range(min(SELF_PLAY_WORKERS, GAMES)):
            parent, child = context.Pipe()
            connections.append(parent)
            process = context.Process(target = self_play_worker, args = (child,), daemon = True)
            try:
                process.start()
                processes.append(process)
            finally:
                child.close()
        for first in range(1, GAMES + 1, len(connections)):
            count = min(len(connections), GAMES - first + 1)
            active, results = set(connections[:count]), []
            stats = {"requests": 0, "batches": 0, "max_batch": 0}
            started = time.perf_counter()
            print(f"开始并行自对弈：第 {first} 至 {first + count - 1} 局，活跃进程 {count}。", flush = True)
            for offset, connection in enumerate(connections[:count]):
                episode = first + offset
                connection.send({
                    "episode": episode, "seed": (SEED + episode) % (2 ** 32), "simulations": SIMULATIONS,
                    "c_puct": C_PUCT, "max_moves": MAX_MOVES, "temperature": TEMPERATURE,
                    "late_temperature": LATE_TEMPERATURE, "temperature_moves": TEMPERATURE_MOVES,
                    "distance_value_weight": DISTANCE_VALUE_WEIGHT, "distance_value_scale": DISTANCE_VALUE_SCALE,
                    "current_distance_mix": CURRENT_DISTANCE_MIX, "root_prior_mix": ROOT_PRIOR_MIX,
                    "repetition_penalty": REPETITION_PENALTY, "heuristic_mix": HEURISTIC_MIX,
                    "tactical_mix": TACTICAL_MIX
                })
            while active:
                ready = wait(list(active), timeout = 1.0)
                if not ready:
                    if any(not process.is_alive() for process in processes):
                        raise RuntimeError("自对弈子进程意外退出")
                    continue
                requests = []
                deadline = time.perf_counter() + INFERENCE_WAIT_MS / 1000.0
                while ready:
                    for connection in ready:
                        message = connection.recv()
                        if message[0] == "predict":
                            requests.append((connection, message[1]))
                        elif message[0] == "result":
                            results.append(message[1:])
                            active.remove(connection)
                        elif message[0] == "error":
                            raise RuntimeError(f"自对弈失败：\n{message[1]}")
                        else:
                            raise RuntimeError("收到无效的自对弈消息")
                    candidates = list(active - {connection for connection, _ in requests})
                    remaining = max(0.0, deadline - time.perf_counter())
                    ready = wait(candidates, timeout = remaining) if requests and candidates else []
                if requests:
                    logits, values = network.predict_batch([state for _, state in requests])
                    stats["requests"] += len(requests)
                    stats["batches"] += 1
                    stats["max_batch"] = max(stats["max_batch"], len(requests))
                    for index, (connection, _) in enumerate(requests):
                        connection.send(("prediction", logits[index], float(values[index])))
            stats["seconds"] = time.perf_counter() - started
            yield sorted(results, key = lambda result: result[2]["episode"]), stats
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout = 2.0)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in connections:
            connection.close()


def train(network, search):
    replay = deque(maxlen = REPLAY_SIZE)
    replay_weights = deque(maxlen = REPLAY_SIZE)
    decisive_games = 0
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents = True, exist_ok = True)
    trace_dir = None if TRACE_DIR is None else Path(TRACE_DIR)
    if trace_dir is not None:
        trace_dir.mkdir(parents = True, exist_ok = True)
    print(f"自对弈进程数：{min(SELF_PLAY_WORKERS, GAMES)}；模型仅由主进程训练和保存。", flush = True)
    batches = game_batches(network, search)
    try:
        for batch_number, (results, stats) in enumerate(batches, start = 1):
            update_started = time.perf_counter()
            for samples, winner, _ in results:
                replay.extend(samples)
                # Weight games rather than individual game lengths.
                weight = (DECISIVE_GAME_WEIGHT if winner is not None else 1.0) / len(samples)
                replay_weights.extend([weight] * len(samples))
            for samples, winner, info in results:
                episode = info["episode"]
                decisive_games += int(winner is not None)
                losses = []
                for _ in range(UPDATES):
                    size = min(BATCH_SIZE, len(replay))
                    probabilities = np.asarray(replay_weights, dtype = np.float64)
                    probabilities /= probabilities.sum()
                    indices = np.random.choice(len(replay), size = size, replace = False, p = probabilities)
                    states, policies, values = zip(*(replay[index] for index in indices))
                    losses.append(network.train_batch(states, policies, values, return_losses = True))
                network.save(model_path)
                total_loss, policy_loss, value_loss = np.mean(losses, axis = 0)
                # Only real wins and losses have targets of +/-1.
                real_outcome_fraction = sum(abs(value) == 1 for _, _, value in replay) / len(replay)
                nonzero_value_fraction = sum(value != 0 for _, _, value in replay) / len(replay)
                info.update({"batch": batch_number, "temperature": TEMPERATURE, "late_temperature": LATE_TEMPERATURE, "temperature_moves": TEMPERATURE_MOVES,
                            "simulations": SIMULATIONS, "c_puct": C_PUCT, "self_play_workers": min(SELF_PLAY_WORKERS, GAMES),
                            "elapsed_seconds": info["self_play_seconds"], "real_outcome_sample_fraction": real_outcome_fraction,
                            "nonzero_value_sample_fraction": nonzero_value_fraction, "root_prior_mix": ROOT_PRIOR_MIX,
                            "repetition_penalty": REPETITION_PENALTY, "heuristic_mix": HEURISTIC_MIX,
                            "tactical_mix": TACTICAL_MIX, "decisive_game_weight": DECISIVE_GAME_WEIGHT})
                if trace_dir is not None:
                    trace_path = trace_dir / f"game_{episode:04d}_{time.time_ns()}.json"
                    trace_path.write_text(json.dumps(info, ensure_ascii = False, indent = 2), encoding = "utf-8")
                ending = "到达目标行" if winner is not None else "达到步数上限"
                print(
                    f"第 {episode}/{GAMES} 局：{len(samples)} 步，胜者 {winner if winner is not None else '平局'}，损失 {total_loss:.4f}（策略 {policy_loss:.4f}，价值 {value_loss:.4f}），样本 {len(replay)}，真实胜负样本 {real_outcome_fraction:.1%}，非零价值样本 {nonzero_value_fraction:.1%}，真实终局局数 {decisive_games}/{episode}，结束 {ending}，最终距离 {info['final_distances']}，最终玩家0评分 {info['value_target_player0']:+.3f}，前20手走子 {info['opening_pawn_move_fraction']:.1%}（搜索走子概率 {info['opening_pawn_policy_mass']:.1%}），重复 {info['repeated_positions']} 次，单局面最多出现 {info['max_repetition']} 次，最终位置 {info['final_pawns']}，剩余墙 {info['walls_left']}，策略熵 {info['policy_entropy']:.3f}，自对弈 {info['self_play_seconds']:.1f}s", flush = True)
            update_seconds = time.perf_counter() - update_started
            average_batch = stats["requests"] / stats["batches"] if stats["batches"] else 1.0
            print(
                f"批次 {batch_number}：{len(results)} 局，自对弈墙钟 {stats['seconds']:.1f}s，更新与保存 {update_seconds:.1f}s，平均/最大推理批量 {average_batch:.2f}/{stats['max_batch'] or 1}", flush = True)
    finally:
        batches.close()


def play_human(search):
    game = Quoridor()
    visits = {position_key(game): 1}
    print("你是玩家 0，从棋盘上方向第 8 行前进；输入动作编号，输入 q 退出。")
    move_count = 0
    while move_count < MAX_MOVES:
        print(f"\n{game}\n当前玩家：{game.current_player}；剩余墙：{game.walls_left}")
        if game.winner is not None:
            print("你赢了！" if game.winner == 0 else "AI 赢了。")
            return
        if game.current_player == 0:
            legal = game.legal_actions()
            print(f"合法动作：{legal}")
            answer = input("你的动作编号：").strip()
            if answer.lower() == "q":
                return
            try:
                action = int(answer)
            except ValueError:
                print("请输入整数动作编号。")
                continue
            if action not in legal:
                print("这个动作不合法。")
                continue
        else:
            policy = search.search(game, temperature = 0, position_counts = visits)
            action = int(np.argmax(policy))
            print(f"AI 动作：{action}")
        game.play(action, validate = False)
        key = position_key(game)
        visits[key] = visits.get(key, 0) + 1
        move_count += 1
    if game.winner is not None:
        print("你赢了！" if game.winner == 0 else "AI 赢了。")
    else:
        print("达到最大步数，本局记为平局。")


def main():
    validate_config()
    model_path = Path(MODEL_PATH)
    if MODE == "play" and not model_path.exists():
        raise FileNotFoundError(f"找不到权重文件：{model_path}；请先训练或修改 MODEL_PATH")
    # Only the main process imports TensorFlow.
    import tensorflow as tf
    from policy_value_net import PolicyValueNet
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    network = PolicyValueNet(LEARNING_RATE, residual_blocks = RESIDUAL_BLOCKS, channels = CHANNELS)
    print(f"残差网络：{RESIDUAL_BLOCKS} 个块，{CHANNELS} 个通道，{network.model.count_params():,} 个参数。", flush = True)
    if model_path.exists() and (MODE == "play" or RESUME):
        network.load(model_path)
        print(f"已载入权重：{model_path}")
    else:
        print(f"从随机权重开始训练，保存路径：{model_path}")
    search = MCTS(network, simulations = SIMULATIONS, c_puct = C_PUCT,
                  prior_mix = ROOT_PRIOR_MIX, repetition_penalty = REPETITION_PENALTY,
                  heuristic_mix = HEURISTIC_MIX, tactical_mix = TACTICAL_MIX)
    if MODE == "train":
        train(network, search)
    else:
        play_human(search)


if __name__ == "__main__":
    main()
