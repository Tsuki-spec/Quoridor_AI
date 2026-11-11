"""Run both colors against the unchanged standard Alpha-Beta and save replayable records."""

import hashlib
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import app

ALGORITHM = "alphazero"  # "mcts" needs no weights or TensorFlow.
DIFFICULTY = "normal"
SIMULATIONS = 128
MODEL_PATH = "models/quoridor-resnet-b4-c64.weights.h5"
OPENING = []  # Optional legal action sequence before the AI turns.
MAX_MOVES = 200
REPORT_PATH = BASE_DIR / "records" / "evaluation.json"


def main():
    candidate = {"algorithm": ALGORITHM, "difficulty": DIFFICULTY, "simulations": SIMULATIONS}
    if ALGORITHM == "alphazero":
        candidate["model_path"] = MODEL_PATH
    opponent = {"algorithm": "alpha_beta", "difficulty": "normal"}
    results = []
    for side in (0, 1):
        players = [opponent.copy(), opponent.copy()]
        players[side] = candidate.copy()
        state = app.api_post("/api/new", {"mode": "watch", "players": players, "max_moves": MAX_MOVES})
        match = app.get_match(state["id"])
        for action in OPENING:
            match.commit(action)
        state = match.state()
        while not state["finished"]:
            state = app.api_post("/api/ai", {"id": state["id"]})
            if state["move_count"] % 10 == 0:
                print(f"执{'先' if side == 0 else '后'}：{state['move_count']} 步", flush = True)
        results.append({"candidate_side": side, "winner": state["winner"],
                        "won": state["winner"] == side, "moves": state["move_count"],
                        "record": state["saved_record"]})
        print(results[-1], flush = True)
    report = {"candidate": candidate, "opponent": opponent, "max_moves": MAX_MOVES,
              "opening": OPENING, "results": results, "wins": sum(row["won"] for row in results),
              "source_sha256": {name: hashlib.sha256((BASE_DIR / name).read_bytes()).hexdigest()
                                for name in ("search.py", "mcts.py", "train.py")}}
    REPORT_PATH.parent.mkdir(parents = True, exist_ok = True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    print(f"结果：{REPORT_PATH}；棋谱：{app.RECORD_DIR}", flush = True)


if __name__ == "__main__":
    main()
