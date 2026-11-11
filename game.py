"""
author: cms
date: 2026/10/11
File introduction: Quoridor's game rules, action encoding and input planes for the policy-value network.

Note:
- Walls are stored by the top-left corner of the 2x2 squares they occupy.
- Actions 0-80 move a pawn; 81-144 place horizontal walls; 145-208 place vertical walls.
- Player 0 moves toward row 8; player 1 moves toward row 0.
- The seven input planes describe both pawns, both wall directions, remaining walls and the player to move.
- Repeated positions do not end a game; train.py applies the self-play move limit.
"""

from collections import deque
import numpy as np

BOARD_SIZE = 9  # Standard 9x9 board.
WALL_SIZE = 8  # Eight rows and columns of wall anchors.
WALL_LIMIT = 10  # Each player starts with ten walls.
PAWN_ACTIONS = BOARD_SIZE * BOARD_SIZE  # Pawn actions encode destination squares.
ACTION_SIZE = PAWN_ACTIONS + 2 * WALL_SIZE * WALL_SIZE  # 81 pawn actions and 128 wall actions.


class Quoridor:
    def __init__(self):  # Initialize the standard opening.
        self.pawns = [(0, BOARD_SIZE // 2), (BOARD_SIZE - 1, BOARD_SIZE // 2)]
        self.walls = [set(), set()]
        self.walls_left = [WALL_LIMIT, WALL_LIMIT]
        self.current_player = 0
        self.winner = None

    def copy(self):  # Copy the full state without sharing mutable lists or wall sets.
        other = Quoridor()
        other.pawns = self.pawns.copy()
        other.walls = [items.copy() for items in self.walls]
        other.walls_left = self.walls_left.copy()
        other.current_player = self.current_player
        other.winner = self.winner
        return other

    def position_key(self):  # Include the player to move and wall resources when detecting a repeated state.
        return (tuple(self.pawns), frozenset(self.walls[0]), frozenset(self.walls[1]),
                tuple(self.walls_left), self.current_player, self.winner)

    @staticmethod
    def inside(row, col):  # Check whether a square lies inside the board.
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def is_crossable(self, start, end):  # Check whether adjacent squares are connected by an open edge.
        row, col = start
        next_row, next_col = end
        if not self.inside(next_row, next_col):
            return False
        if abs(row - next_row) + abs(col - next_col) != 1:
            return False
        if row != next_row:
            wall_row = min(row, next_row)
            return (wall_row, col) not in self.walls[0] and (wall_row, col - 1) not in self.walls[0]
        wall_col = min(col, next_col)
        return (row, wall_col) not in self.walls[1] and (row - 1, wall_col) not in self.walls[1]

    def search_legal_pawn_move(self):  # Find normal moves, straight jumps and allowed diagonal jumps.
        own = self.pawns[self.current_player]
        opponent = self.pawns[1 - self.current_player]
        moves = set()
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (own[0] + delta_row, own[1] + delta_col)
            if not self.is_crossable(own, neighbor):
                continue
            if neighbor != opponent:
                moves.add(neighbor)
                continue
            behind = (opponent[0] + delta_row, opponent[1] + delta_col)
            if self.is_crossable(opponent, behind):
                moves.add(behind)
                continue
            # Diagonal jumps are available only when a straight jump is blocked.
            for side_row, side_col in ((-delta_col, delta_row), (delta_col, -delta_row)):
                diagonal = (opponent[0] + side_row, opponent[1] + side_col)
                if self.is_crossable(opponent, diagonal):
                    moves.add(diagonal)
        return moves

    def shortest_path_length(self, player):  # Count open edges to the goal, ignoring pawns and future walls.
        start = self.pawns[player]
        goal_row = BOARD_SIZE - 1 if player == 0 else 0
        queue = deque([(start[0], start[1], 0)])
        seen = {start}
        while queue:
            row, col, distance = queue.popleft()
            if row == goal_row:
                return distance
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (row + delta_row, col + delta_col)
                if neighbor not in seen and self.is_crossable((row, col), neighbor):
                    seen.add(neighbor)
                    queue.append((neighbor[0], neighbor[1], distance + 1))
        return None

    def has_path(self, player):  # A legal wall must preserve a path for both players.
        return self.shortest_path_length(player) is not None

    def can_place_wall(self, direction, row, col):  # Reject overlaps, crossings and walls that block every path.
        if self.walls_left[self.current_player] == 0:
            return False
        if direction not in (0, 1) or not (0 <= row < WALL_SIZE and 0 <= col < WALL_SIZE):
            return False
        if (row, col) in self.walls[0] or (row, col) in self.walls[1]:
            return False
        if direction == 0 and ((row, col - 1) in self.walls[0] or (row, col + 1) in self.walls[0]):
            return False
        if direction == 1 and ((row - 1, col) in self.walls[1] or (row + 1, col) in self.walls[1]):
            return False
        # Place the candidate temporarily, check both paths, then restore the board.
        self.walls[direction].add((row, col))
        allowed = self.has_path(0) and self.has_path(1)
        self.walls[direction].remove((row, col))
        return allowed

    def legal_actions(self):  # Encode all legal pawn destinations and wall placements.
        if self.winner is not None:
            return []
        actions = [row * BOARD_SIZE + col for row, col in sorted(self.search_legal_pawn_move())]
        if self.walls_left[self.current_player] > 0:
            for direction in (0, 1):
                for row in range(WALL_SIZE):
                    for col in range(WALL_SIZE):
                        if self.can_place_wall(direction, row, col):
                            actions.append(PAWN_ACTIONS + direction * WALL_SIZE * WALL_SIZE + row * WALL_SIZE + col)
        return actions

    def play(self, action, validate = True):  # Apply a legal action and switch the player, including after a win.
        if validate and action not in self.legal_actions():
            raise ValueError(f"非法动作: {action}")
        player = self.current_player
        if action < PAWN_ACTIONS:
            self.pawns[player] = divmod(action, BOARD_SIZE)
            if self.pawns[player][0] == (BOARD_SIZE - 1 if player == 0 else 0):
                self.winner = player
        else:
            wall_action = action - PAWN_ACTIONS
            direction, position = divmod(wall_action, WALL_SIZE * WALL_SIZE)
            self.walls[direction].add(divmod(position, WALL_SIZE))
            self.walls_left[player] -= 1
        self.current_player = 1 - player

    def state(self):  # Encode the position as seven 9x9 planes for the player to move.
        planes = np.zeros((BOARD_SIZE, BOARD_SIZE, 7), dtype = np.float32)
        own = self.current_player
        own_row, own_col = self.pawns[own]
        rival_row, rival_col = self.pawns[1 - own]
        planes[own_row, own_col, 0] = 1.0
        planes[rival_row, rival_col, 1] = 1.0
        for direction in (0, 1):
            for row, col in self.walls[direction]:
                planes[row, col, 2 + direction] = 1.0
        planes[:, :, 4] = self.walls_left[own] / WALL_LIMIT
        planes[:, :, 5] = self.walls_left[1 - own] / WALL_LIMIT
        planes[:, :, 6] = float(own == 0)
        return planes

    def __str__(self):  # Display pawns and the edges blocked by walls.
        lines = []
        for row in range(BOARD_SIZE):
            cells = []
            for col in range(BOARD_SIZE):
                piece = "0" if self.pawns[0] == (row, col) else "1" if self.pawns[1] == (row, col) else "."
                cells.append(piece)
                if col < BOARD_SIZE - 1:
                    cells.append("|" if not self.is_crossable((row, col), (row, col + 1)) else " ")
            lines.append("".join(cells))
            if row < BOARD_SIZE - 1:
                lines.append(" ".join("-" if not self.is_crossable((row, col),
                             (row + 1, col)) else " " for col in range(BOARD_SIZE)))
        return "\n".join(lines)

    def open_edge(self, start, end):
        # Keep the Quoridor edge-check interface available.
        return self.is_crossable(start, end)

    def pawn_moves(self):
        # Keep the Quoridor pawn-move interface available.
        return self.search_legal_pawn_move()
