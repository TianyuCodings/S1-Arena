"""Full chess games against Stockfish 19 at skill level 0, one decision per move of the model.

The state is what the static chess leaves show (FEN, side to move, last move) plus the game so far
in standard notation; the question lists every legal move in standard algebraic notation (SAN),
alphabetically, each with a plain description, as in game/chess_best_move. The model plays White
in even seeds and Black in odd seeds. A move with no alternative is played automatically.

Stockfish's own strength limit (Skill Level / UCI_Elo) draws its random numbers from a generator
seeded with the clock, so its games cannot be replayed. The opponent therefore applies the same
rule outside the engine: a 4-line MultiPV search of a fixed 100,000 nodes, the candidate moves as
they stood after the depth-1 iteration (the depth Stockfish picks at for skill level 0), and
Stockfish's selection formula (search.cpp, Skill::pick_best) with its xorshift64* generator seeded
from the episode seed. The same seed and the same model moves give the same game.

A game ends by checkmate, stalemate, insufficient material, threefold repetition or the
fifty-move rule. A game still running after the model's 150th move is adjudicated by a
full-strength Stockfish search of 1,000,000 nodes: a side ahead by at least 3 pawns (300 cp) or
with a forced mate wins; anything else is a draw.

The Stockfish 19 binary (GPL-3.0) is downloaded from the official release on first use into
.cache/stockfish/ and checked against its SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from ...env import ROOT
from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment

SF_TAG = "sf_19"
SF_RELEASE = f"https://github.com/official-stockfish/Stockfish/releases/download/{SF_TAG}/"
_MAC = ("stockfish-macos-universal.tar.gz", "a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77")
_LINUX_ARM = ("stockfish-linux-arm64-universal.tar.gz", "fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01")
SF_ASSETS = {  # (platform.system(), platform.machine()) -> (release asset, SHA-256 shown on the release page)
    ("Darwin", "arm64"): _MAC,
    ("Darwin", "x86_64"): _MAC,
    ("Linux", "x86_64"): ("stockfish-linux-x86-64-universal.tar.gz", "9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"),
    ("Linux", "aarch64"): _LINUX_ARM,
    ("Linux", "arm64"): _LINUX_ARM,
    ("Windows", "AMD64"): ("stockfish-windows-x86-64-universal.zip", "3c8bf1f9ea66a09350a40df4f632288285ac206d99f33ab5842c408fc30b48a7"),
    ("Windows", "ARM64"): ("stockfish-windows-arm64-universal.zip", "8372ad3f0d7276deb2c70f801f541ec7db463219fc6d9c7592864e542aa4f401"),
}
SF_DIR = ROOT / ".cache" / "stockfish" / SF_TAG
_download_lock = threading.Lock()


def stockfish_path() -> Path:
    """The Stockfish 19 binary for this machine, downloaded and verified on first use."""
    key = (platform.system(), platform.machine())
    if key not in SF_ASSETS:
        raise RuntimeError(f"no official Stockfish 19 build known for {key}")
    asset, digest = SF_ASSETS[key]
    binary = SF_DIR / ("stockfish.exe" if key[0] == "Windows" else "stockfish")
    with _download_lock:
        if binary.exists():
            return binary
        SF_DIR.mkdir(parents=True, exist_ok=True)
        archive = SF_DIR / asset
        part = archive.with_name(asset + ".part")
        request = urllib.request.Request(SF_RELEASE + asset, headers={"User-Agent": "s1arena"})
        with urllib.request.urlopen(request, timeout=600) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out)
        sha = hashlib.sha256(part.read_bytes()).hexdigest()
        if sha != digest:
            part.unlink()
            raise RuntimeError(f"{asset}: SHA-256 {sha} does not match the release digest {digest}")
        part.rename(archive)
        tmp = binary.with_name(binary.name + ".part")
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                member = max((m for m in z.infolist() if Path(m.filename).name.startswith("stockfish-")), key=lambda m: m.file_size)
                with z.open(member) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            with tarfile.open(archive) as t:
                member = max((m for m in t.getmembers() if m.isfile() and Path(m.name).name.startswith("stockfish-")), key=lambda m: m.size)
                with t.extractfile(member) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out)
        tmp.chmod(0o755)
        tmp.rename(binary)
        archive.unlink()
        return binary


class StockfishSkill:
    """Stockfish's skill-level move choice (Skill::pick_best in search.cpp of Stockfish 19) with a seeded generator.

    Values are converted from the UCI centipawns back to Stockfish's internal units with the engine's own
    win-rate normalization (uci.cpp, win_rate_params), so the formula runs on the numbers the engine uses.
    """

    PAWN_VALUE = 208  # types.h
    VALUE_MATE = 32000
    MASK = (1 << 64) - 1

    def __init__(self, engine, level: int, nodes: int, seed: int) -> None:
        self.engine, self.level, self.nodes = engine, level, nodes
        state = int.from_bytes(hashlib.sha256(f"chess_play:{seed}".encode()).digest()[:8], "little")
        self.state = state or 1  # xorshift64* needs a non-zero state

    def _rand(self) -> int:
        """PRNG::rand<unsigned>() of misc.h: xorshift64*, low 32 bits."""
        s = self.state
        s ^= s >> 12
        s = (s ^ (s << 25)) & self.MASK
        s ^= s >> 27
        self.state = s
        return ((s * 2685821657736338717) & self.MASK) & 0xFFFFFFFF

    @staticmethod
    def _normalizer(board) -> float:
        import chess

        material = sum(len(board.pieces(piece, color)) * worth for piece, worth in
                       ((chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9))
                       for color in (chess.WHITE, chess.BLACK))
        m = min(max(material, 17), 78) / 58.0
        return ((-142.72052667 * m + 372.35176398) * m - 340.71073572) * m + 415.23490212

    def _internal(self, score, board) -> int:
        s = score.relative
        if s.is_mate():
            n = s.mate()
            return self.VALUE_MATE - (2 * n - 1) if n > 0 else -self.VALUE_MATE - 2 * n
        return round(s.score() * self._normalizer(board) / 100)

    def move(self, board):
        import chess.engine

        pick_depth = 1 + self.level
        lines: dict[int, dict] = {}
        with self.engine.analysis(board, chess.engine.Limit(nodes=self.nodes), multipv=4) as analysis:
            for info in analysis:
                if info.get("depth") == pick_depth and info.get("pv") and "score" in info:
                    lines[info.get("multipv", 1)] = info
        if not lines:
            raise RuntimeError(f"Stockfish reported no depth-{pick_depth} lines for {board.fen()}")
        scored = [(lines[k]["pv"][0], self._internal(lines[k]["score"], board)) for k in sorted(lines)]
        top, low = max(v for _, v in scored), min(v for _, v in scored)
        delta = min(top - low, self.PAWN_VALUE)
        weakness = 120 - 2 * self.level
        best, best_value = scored[0][0], -self.VALUE_MATE - 1
        for move, value in scored:
            push = int(weakness * (top - value) + delta * (self._rand() % weakness)) // 128
            if value + push >= best_value:
                best, best_value = move, value + push
        return best


PIECES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}  # chess.PAWN .. chess.KING


def describe_move(board, move) -> str:
    """'knight from g1 to f3', 'pawn from e7 to e8, promoting to a queen', 'king castles kingside' (as in game/chess_best_move)."""
    import chess

    if board.is_castling(move):
        return "king castles kingside" if chess.square_file(move.to_square) == 6 else "king castles queenside"
    text = f"{PIECES[board.piece_type_at(move.from_square)]} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
    if board.is_en_passant(move):
        text += ", capturing en passant"
    elif board.is_capture(move):
        text += f", capturing the {PIECES[board.piece_type_at(move.to_square)]}"
    if move.promotion:
        text += f", promoting to a {PIECES[move.promotion]}"
    return text


@register_env
class ChessVsStockfish(Environment):
    name = "game/chess_play/stockfish_level0"
    leaf = "game/chess_play"
    source = "Stockfish 19 at skill level 0 (the weakest official setting), move choice re-implemented with a seeded generator"
    url = "https://github.com/official-stockfish/Stockfish/releases/tag/sf_19"
    license = "GPL-3.0 (Stockfish 19 binary, downloaded at run time, not redistributed); GPL-3.0-or-later (python-chess)"
    tier = "C"
    criteria_source = ("Newly written game description (rules as in the FIDE Laws of Chess); the per-move question reuses the "
                       "Lichess puzzle prompt of game/chess_best_move ('Find the best move for white.')")
    notes = ("Seeds 0-19: the model plays White in even seeds and Black in odd seeds; the seed also drives the opponent's "
             "random choices, so games differ even for a deterministic model. Options: every legal move in SAN (with the usual "
             "+ and # marks), alphabetical, as in game/chess_best_move. Opponent: Stockfish 19, 1 thread, 16 MB hash, "
             "100,000 nodes per move, skill-level-0 choice among its 4 best depth-1 moves (Stockfish's formula, seeded). "
             "Forced moves (only one legal move) are played automatically and are not decisions. Draws: stalemate, insufficient "
             "material, threefold repetition, fifty-move rule. Cap: 150 moves by the model, then adjudication by a full-strength "
             "1,000,000-node search (at least +3.00 or a forced mate wins, otherwise a draw). Score 1 / 0.5 / 0 from the "
             "model's side; success = win.")
    mode = "paused"
    episodes = 20
    max_decisions = 150
    score_name = "result"
    skill_level = 0
    nodes = 100_000
    adjudication_nodes = 1_000_000
    adjudication_cp = 300

    def reset(self, seed: int) -> None:
        import chess
        import chess.engine

        self.close()
        self.engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path()))
        self.engine.configure({"Threads": 1, "Hash": 16})
        self.opponent = StockfishSkill(self.engine, self.skill_level, self.nodes, seed)
        self.board = chess.Board()
        self.me = chess.WHITE if seed % 2 == 0 else chess.BLACK
        self.model_moves = self.forced_moves = 0
        self.adjudication: dict[str, Any] | None = None
        if self.board.turn != self.me:
            self._reply()

    def _side(self, color) -> str:
        return "White" if color else "Black"

    def task_text(self) -> str:
        return (f"You are playing a game of chess as {self._side(self.me)} against a computer opponent, from the standard starting "
                "position, under the FIDE Laws of Chess. You win by checkmating the opponent's king. The game is drawn by "
                "stalemate, by insufficient material, when the same position occurs for the third time, or after 50 moves by each "
                "side without a capture or a pawn move. If the game is still going after your 150th move, a full-strength engine "
                "evaluates the final position: a side ahead by at least 3 pawns, or with a forced mate, wins; otherwise the game is "
                "a draw. Your result: 1 for a win, 0.5 for a draw, 0 for a loss.")

    def movetext(self) -> str:
        import chess

        board = chess.Board()
        parts = []
        for move in self.board.move_stack:
            if board.turn == chess.WHITE:
                parts.append(f"{board.fullmove_number}. {board.san(move)}")
            else:
                parts.append(board.san(move) if parts else f"{board.fullmove_number}... {board.san(move)}")
            board.push(move)
        return " ".join(parts)

    def observe(self) -> DecisionRequest:
        board = self.board
        legal = list(board.legal_moves)
        sans = {board.san(m): m for m in legal}
        self.options = {san: sans[san] for san in sorted(sans)}
        if board.move_stack:
            last = board.pop()
            last_text = f"{self._side(board.turn)} played {board.san(last)}"
            board.push(last)
        else:
            last_text = "none: the game starts with your move"
        side = self._side(board.turn)
        state = {"task": self.task_text(), "fen": board.fen(), "side_to_move": side, "moves": self.movetext() or "none yet",
                 "last_move": last_text}
        question = Question(type="choice", instructions=f"Find the best move for {side.lower()}.",
                            criteria={san: describe_move(board, move) for san, move in self.options.items()})
        return DecisionRequest(state=state, questions={"move": question})

    def act(self, choice: dict[str, str]) -> None:
        self.board.push(self.options[choice["move"]])
        self.model_moves += 1
        self._reply()

    def _reply(self) -> None:
        """The opponent's move, then any forced moves of the model (and the replies to them)."""
        while self._termination() is None and self.model_moves < self.max_decisions:
            if self.board.turn != self.me:
                self.board.push(self.opponent.move(self.board))
                continue
            if self.board.legal_moves.count() != 1:
                return
            self.board.push(next(iter(self.board.legal_moves)))
            self.model_moves += 1
            self.forced_moves += 1

    def _termination(self) -> str | None:
        board = self.board
        if board.is_checkmate():
            return "checkmate"
        if board.is_stalemate():
            return "stalemate"
        if board.is_insufficient_material():
            return "insufficient material"
        if board.is_repetition(3):
            return "threefold repetition"
        if board.is_fifty_moves():
            return "fifty-move rule"
        return None

    def done(self) -> bool:
        return self._termination() is not None or self.model_moves >= self.max_decisions

    def _adjudicate(self):
        """Winner of an unfinished game by a full-strength search: None for a draw."""
        import chess.engine

        info = self.engine.analyse(self.board, chess.engine.Limit(nodes=self.adjudication_nodes))
        score = info["score"].white()
        cp = score.score(mate_score=100_000)
        self.adjudication = {"eval_white": str(score), "nodes": self.adjudication_nodes}
        if score.is_mate() or abs(cp) >= self.adjudication_cp:
            return cp > 0  # chess.WHITE is True
        return None

    def outcome(self) -> dict[str, Any]:
        termination = self._termination()
        if termination == "checkmate":
            winner = not self.board.turn  # the side that just moved
        elif termination is not None:
            winner = None
        else:
            termination = "adjudicated"
            winner = self._adjudicate()
        score = 0.5 if winner is None else 1.0 if winner == self.me else 0.0
        result = "1/2-1/2" if winner is None else "1-0" if winner else "0-1"
        out = {"success": score == 1.0, "score": score, "result": result, "termination": termination,
               "model_color": self._side(self.me).lower(), "model_moves": self.model_moves, "forced_moves": self.forced_moves,
               "plies": len(self.board.move_stack), "moves": self.movetext(), "final_fen": self.board.fen(),
               "engine": self.engine.id.get("name", "")}
        if self.adjudication:
            out["adjudication"] = self.adjudication
        return out

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            try:
                engine.quit()
            except Exception:  # the engine may already be gone
                pass
            self.engine = None
