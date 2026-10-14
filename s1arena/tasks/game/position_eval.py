"""Leaf: position evaluation (局面评估).

Three kinds of target. ChessBench: the win probability that Stockfish 16 assigns to the side
to move, in five bands; this is the engine's estimate, not the probability of the real
outcome. Lichess games: whether White actually went on to win a rated game from the shown
position (for calibration against real outcomes). Connect Four: the exact game-theoretic
value of the position under perfect play, from John Tromp's solved opening database.
"""

from __future__ import annotations

import functools
import hashlib
import re
import zipfile

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch
from ._chess import START_FEN, Position, side_name
from .chess_moves import CHESSBENCH, CHESSBENCH_LICENSE, bag_records, decode_record

LEAF = "game/position_eval"

WIN_LEVELS = [
    "Losing: the side to move has a win probability below 20%",
    "Worse: win probability from 20% to 40%",
    "Roughly equal: win probability from 40% to 60%",
    "Better: win probability from 60% to 80%",
    "Winning: win probability of 80% or more",
]


def win_level(p: float) -> int:
    return min(4, int(p * 5))


@register_task
class ChessBenchStateValue(TaskAdapter):
    name = f"{LEAF}/chessbench"
    leaf = LEAF
    source = "ChessBench test positions with Stockfish 16 state values (Ruoss et al., NeurIPS 2024)"
    url = "https://github.com/google-deepmind/searchless_chess"
    license = CHESSBENCH_LICENSE
    tier = "B"
    criteria_source = "Adapted from ChessBench's value targets (uniform win-probability buckets; five instead of 128)"
    notes = ("Positions from the Lichess games that ChessBench holds out as its test set; the value is Stockfish 16's evaluation "
             "(0.05 s per position) converted to the side to move's win probability with the Lichess formula. It is the "
             "engine's estimate, not the probability of the actual result. Positions without a legal move are left out.")

    def load(self):
        playable = {decode_record(r, "ss")[0] for r in bag_records(fetch(CHESSBENCH + "test/behavioral_cloning_data.bag", "chessbench_test_behavioral_cloning_data.bag"))}
        rows = []
        for record in bag_records(fetch(CHESSBENCH + "test/state_value_data.bag", "chessbench_test_state_value_data.bag")):
            fen, p = decode_record(record, "sf")
            if fen in playable:
                rows.append({"fen": fen, "p": round(p, 6), "level": win_level(p)})
        return rows

    def label_of(self, record):
        return str(record["level"])

    def to_item(self, record, rng):
        pos = Position.from_fen(record["fen"])
        question = Question(type="score", criteria=WIN_LEVELS, instructions=(
            "How good is this chess position for the side to move? Rate the side to move's winning chances as a strong chess "
            "engine (Stockfish 16) evaluates them; the levels are bands of the engine's win probability for the side to move."))
        state = {"fen": pos.fen(), "side_to_move": side_name(pos.turn)}
        return self.item(state, {"evaluation": question}, {"evaluation": Gold(label=str(record["level"]))}, win_probability=record["p"])


# ---------- Lichess rated games: the real result ----------

GAMES_REPO, GAMES_REV = "Lichess/standard-chess-games", "de4e636eddf568a9394cc01fb0b9e1da04a6babf"
GAMES_FILE = "data/year=2024/month=06/train-00000-of-00379.parquet"  # about 236,000 games played on 1 June 2024
EVENTS = {"Rated Blitz game": "blitz", "Rated Rapid game": "rapid", "Rated Classical game": "classical"}
POOL = 20_000  # hash-ordered subset of the eligible games
TOKEN = re.compile(r"\{[^}]*\}|\d+\.(?:\.\.)?|[^\s{}]+")
CLOCK = re.compile(r"%clk (\d+):(\d+):(\d+)")


def parse_movetext(text: str) -> tuple[list[str], list[int | None]]:
    """SAN moves and the clock (seconds left after each move) of a Lichess movetext."""
    moves, clocks = [], []
    for token in TOKEN.findall(text):
        if token.startswith("{"):
            found = CLOCK.search(token)
            if found and moves:
                h, m, s = map(int, found.groups())
                clocks[-1] = 3600 * h + 60 * m + s
        elif re.fullmatch(r"\d+\.(\.\.)?", token):
            continue
        elif token in ("1-0", "0-1", "1/2-1/2", "*"):
            break
        else:
            moves.append(token)
            clocks.append(None)
    return moves, clocks


def clock_text(seconds: int) -> str:
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def time_control_text(tc: str) -> str:
    base, inc = (int(x) for x in tc.split("+"))
    minutes = f"{base // 60} minutes" if base % 60 == 0 else f"{base} seconds"
    return f"{tc} ({minutes} per player plus {inc} seconds per move)"


@register_task
class LichessGameOutcome(TaskAdapter):
    name = f"{LEAF}/lichess_games"
    leaf = LEAF
    source = "Lichess rated standard games of 1 June 2024, first shard of the June 2024 export (lichess.org)"
    url = "https://huggingface.co/datasets/Lichess/standard-chess-games"
    license = "CC0 1.0"
    tier = "C"
    criteria_source = "Written; gold is the recorded result of the game"
    notes = ("Rated blitz, rapid and classical games between human players (no bots) that ended normally or on time, at least "
             "30 plies long. One position per game, drawn uniformly between ply 20 and two plies before the end; the state "
             "gives both ratings (Lichess Glicko-2) and both clocks. A draw counts as 'White does not win'. A real outcome "
             "is a single draw from the position's distribution, so this source measures calibration rather than accuracy.")

    def load(self):
        load_env()
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(GAMES_REPO, GAMES_FILE, repo_type="dataset", revision=GAMES_REV)
        table = pq.read_table(path, columns=["Event", "Site", "Result", "WhiteTitle", "BlackTitle", "WhiteElo", "BlackElo", "Termination", "TimeControl", "movetext"])
        keep = pc.and_(pc.is_in(table["Event"], value_set=pa.array(list(EVENTS))),
                       pc.is_in(table["Termination"], value_set=pa.array(["Normal", "Time forfeit"])))
        table = table.filter(keep)
        rows = table.to_pylist()
        rows = [r for r in rows if "BOT" not in (r["WhiteTitle"], r["BlackTitle"]) and r["Result"] in ("1-0", "0-1", "1/2-1/2")
                and "%clk" in r["movetext"]]
        rows.sort(key=lambda r: hashlib.sha256(r["Site"].encode()).hexdigest())
        out = []
        for r in rows:
            moves, clocks = parse_movetext(r["movetext"])
            if len(moves) < 30 or any(c is None for c in clocks):
                continue
            out.append({"id": r["Site"].rsplit("/", 1)[-1], "time_class": EVENTS[r["Event"]], "result": r["Result"],
                        "white_elo": r["WhiteElo"], "black_elo": r["BlackElo"], "time_control": r["TimeControl"],
                        "moves": moves, "clocks": clocks})
            if len(out) >= POOL:
                break
        return out

    def label_of(self, record):
        return f"{record['result'] == '1-0'}|{record['time_class']}"

    def to_item(self, record, rng):
        ply = rng.randint(20, len(record["moves"]) - 2)  # number of moves already played
        pos = Position.from_fen(START_FEN)
        for san in record["moves"][:ply]:
            pos = pos.push(pos.parse_san(san))
        clocks = record["clocks"][:ply]
        white_clock, black_clock = clocks[-2 if ply % 2 == 0 else -1], clocks[-1 if ply % 2 == 0 else -2]
        state = {"fen": pos.fen(), "side_to_move": side_name(pos.turn), "white_rating": str(record["white_elo"]),
                 "black_rating": str(record["black_elo"]), "time_control": time_control_text(record["time_control"]),
                 "white_clock": clock_text(white_clock), "black_clock": clock_text(black_clock)}
        question = Question(type="boolean", instructions=(
            "This position comes from a rated online game between two human players on Lichess; the ratings and the time left "
            "on each player's clock are given. The game continues from here. Will White win the game? A draw or a Black win "
            "counts as no."))
        gold = Gold(label="true" if record["result"] == "1-0" else "false")
        return self.item(state, {"white_wins": question}, {"white_wins": gold}, original_id=record["id"], result=record["result"],
                         time_class=record["time_class"], ply=ply)


# ---------- Connect Four: solved 8-ply positions ----------

CONNECT4_URL = "https://archive.ics.uci.edu/static/public/26/connect+4.zip"
C4_LEVELS = [
    "X loses: O can force a win",
    "Draw: neither player can force a win",
    "X wins: X can force a win",
]
C4_LEVEL = {"loss": "0", "draw": "1", "win": "2"}


def unlzw(data: bytes) -> bytes:
    """Decompress Unix 'compress' (.Z) data: LZW with codes of 9 to 16 bits, read in groups of eight codes."""
    if data[:2] != b"\x1f\x9d":
        raise ValueError("not a .Z file")
    maxbits, block = data[2] & 0x1F, bool(data[2] & 0x80)
    table = [bytes([k]) for k in range(256)] + ([b""] if block else [])
    n, bitpos, seg_start, total = 9, 24, 24, len(data) * 8
    out, prev = bytearray(), None
    padded = data + b"\x00\x00\x00\x00"
    while bitpos + n <= total:
        if len(table) >= (1 << n) and n < maxbits:  # the next entry needs a wider code: skip to the next group of codes
            rem = (bitpos - seg_start) % (n * 8)
            bitpos += (n * 8 - rem) if rem else 0
            seg_start, n = bitpos, n + 1
            if bitpos + n > total:
                break
        chunk = int.from_bytes(padded[bitpos >> 3: (bitpos >> 3) + 4], "little")
        code = (chunk >> (bitpos & 7)) & ((1 << n) - 1)
        bitpos += n
        if block and code == 256:  # clear the table
            rem = (bitpos - seg_start) % (n * 8)
            bitpos += (n * 8 - rem) if rem else 0
            seg_start, n, prev = bitpos, 9, None
            del table[257:]
            continue
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]
            if len(table) < (1 << maxbits):
                table.append(prev + entry[:1])
        elif code == len(table):
            entry = prev + prev[:1]
            table.append(entry)
        else:
            raise ValueError("corrupt .Z data")
        out += entry
        prev = entry
    return bytes(out)


@functools.lru_cache(maxsize=1)
def connect4_rows() -> tuple[tuple[str, ...], ...]:
    with zipfile.ZipFile(fetch(CONNECT4_URL, "uci_connect4.zip")) as z:
        text = unlzw(z.read("connect-4.data.Z")).decode()
    return tuple(tuple(line.split(",")) for line in text.split("\n") if line.strip())


def connect4_board(cells: tuple[str, ...]) -> str:
    """Six rows, top row first; the dataset lists squares column by column from a1 (bottom) to g6."""
    mark = {"x": "X", "o": "O", "b": "."}
    lines = []
    for row in range(5, -1, -1):
        lines.append(f"{row + 1} " + " ".join(mark[cells[col * 6 + row]] for col in range(7)))
    lines.append("  a b c d e f g")
    return "\n".join(lines)


@register_task
class Connect4Value(TaskAdapter):
    name = f"{LEAF}/connect4"
    leaf = LEAF
    source = "Connect-4 opening database: all 8-ply positions with their game-theoretic value (John Tromp, UCI repository)"
    url = "https://archive.ics.uci.edu/dataset/26/connect+4"
    license = "CC BY 4.0"
    tier = "C"
    criteria_source = "Written; levels are the database's three classes (win, draw, loss for the first player)"
    notes = ("Every legal position after 8 plies in which no one has won and the next move is not forced, labeled by an "
             "exhaustive solver with the result under perfect play for X, who moved first and is to move. The natural class "
             "mix is kept (66% wins, 25% losses, 10% draws).")

    def load(self):
        return [{"cells": row[:42], "value": row[42]} for row in connect4_rows()]

    def label_of(self, record):
        return record["value"]

    def to_item(self, record, rng):
        question = Question(type="score", criteria=C4_LEVELS, instructions=(
            "Connect Four on the standard board of 7 columns and 6 rows: players take turns dropping a disc into a column, "
            "where it falls to the lowest empty square, and the first to line up four discs horizontally, vertically or "
            "diagonally wins. X moved first and it is X's turn. What is the result of this position if both players play "
            "perfectly from here?"))
        state = {"board": connect4_board(record["cells"]), "to_move": "X"}
        return self.item(state, {"perfect_play_result": question}, {"perfect_play_result": Gold(label=C4_LEVEL[record["value"]])})
