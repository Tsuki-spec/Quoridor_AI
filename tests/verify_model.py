"""Check real weights, route features, legacy migration and one training update."""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from game import Quoridor
from policy_value_net import PolicyValueNet

MODEL_PATH = BASE_DIR / "models/quoridor-resnet-b4-c64.weights.h5"
LEGACY_PATH = BASE_DIR / "models/archive/quoridor-strength-v3-b4-c64.weights.h5"


def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    fixture = json.loads((BASE_DIR / "tests/maze_record.json").read_text())
    game, games, states = Quoridor(), [], []
    for action in fixture["actions"]:
        games.append(game.copy())
        states.append(game.state())
        game.play(action, validate = False)
    states = np.asarray(states)
    network = PolicyValueNet()
    network.load(MODEL_PATH)
    encoded = network.encode_states(states)
    assert encoded.shape == (len(states), 9, 9, 9)
    for position, state in zip(games, encoded):
        for channel, side in ((7, position.current_player), (8, 1 - position.current_player)):
            row, col = position.pawns[side]
            assert np.isclose(state[row, col, channel] * 9, position.shortest_path_length(side))
    assert np.array_equal(network.encode_states(encoded), encoded)
    logits, values = network.predict_batch(states)
    assert np.isfinite(logits).all() and np.isfinite(values).all()
    one_logits, one_value = network.predict(states[0])
    assert np.allclose(one_logits, logits[0], atol = 1e-4)
    assert np.isclose(one_value, values[0], atol = 1e-5)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.weights.h5"
        network.save(path)
        restored = PolicyValueNet()
        restored.load(path)
        a, b = restored.predict_batch(states)
        assert np.allclose(a, logits) and np.allclose(b, values)
        policies = np.zeros((8, 209), dtype = np.float32)
        for index, action in enumerate(fixture["actions"][:8]):
            policies[index, action] = 1
        losses = restored.train_batch(states[:8], policies, np.zeros(8), return_losses = True)
        assert np.isfinite(losses).all()
        restored.save(path)
        continued = PolicyValueNet()
        continued.load(path)
        a, b = continued.predict_batch(states[:8])
        assert np.isfinite(a).all() and np.isfinite(b).all()

    legacy = PolicyValueNet(distance_features = False)
    legacy.load(LEGACY_PATH)
    converted = PolicyValueNet()
    converted.load(LEGACY_PATH)
    a, b = legacy.predict_batch(states)
    c, d = converted.predict_batch(states)
    assert np.allclose(a, c, atol = 1e-4) and np.allclose(b, d, atol = 1e-5)
    print(f"通过：{len(states)} 个迷宫局面、距离图、单个/批量推理、旧权重迁移、保存载入与续训。")


if __name__ == "__main__":
    main()
