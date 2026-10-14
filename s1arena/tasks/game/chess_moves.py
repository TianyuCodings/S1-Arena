"""Leaf: chess, best move (国际象棋：最佳着法).

One choice question whose options are all legal moves of the position (at most 218) in
standard algebraic notation, listed in alphabetical order. Lichess puzzles give the "only
move" found by Stockfish when the puzzle was generated; ChessBench gives Stockfish 16's
value of every legal move, and only positions where the best move is clearly ahead are kept.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import struct

from ...env import load_env
from ...registry import register_task
from ...schema import Gold, Question
from ..base import TaskAdapter
from ..common import fetch
from ._chess import Position, describe_move, side_name

LEAF = "game/chess_best_move"

PUZZLES_REPO, PUZZLES_REV = "Lichess/chess-puzzles", "479ea9bc9f681385f5adb23fa27a96c2dc8ae599"  # September 2026 export
PUZZLES_FILE = "data/train-00000-of-00003.parquet"  # first of three shards: about 2 million puzzles
BANDS = ((0, "rating below 1000"), (1000, "rating 1000-1399"), (1400, "rating 1400-1799"),
         (1800, "rating 1800-2199"), (2200, "rating 2200-2599"), (2600, "rating 2600 and above"))
PER_BAND = 600  # candidates checked per band (a fixed, hash-ordered subset)

CHESSBENCH = "https://storage.googleapis.com/searchless_chess/data/"
CHESSBENCH_LICENSE = "CC BY 4.0 (ChessBench annotations); positions from the Lichess database (CC0)"


def band_of(rating: int) -> str:
    return [name for low, name in BANDS if rating >= low][-1]


def _order(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def move_options(pos: Position) -> tuple[dict[str, str], dict[str, str]]:
    """All legal moves as {SAN: description}, alphabetically, and a UCI -> SAN map."""
    legal = pos.legal_moves()
    sans = {m.uci(): pos.san(m, legal) for m in legal}
    by_san = {sans[m.uci()]: describe_move(pos, m) for m in legal}
    return {k: by_san[k] for k in sorted(by_san)}, sans


def best_move_question(pos: Position, options: dict[str, str]) -> Question:
    return Question(type="choice", instructions=f"Find the best move for {side_name(pos.turn).lower()}.", criteria=options)


def gives_mate(pos: Position, move) -> bool:
    after = pos.push(move)
    return after.in_check() and not after.legal_moves()


@register_task
class LichessPuzzles(TaskAdapter):
    name = f"{LEAF}/lichess_puzzles"
    leaf = LEAF
    source = "Lichess puzzle database, first shard of the September 2026 export (lichess.org)"
    url = "https://huggingface.co/datasets/Lichess/chess-puzzles"
    license = "CC0 1.0"
    tier = "A"
    criteria_source = "Lichess puzzle prompt ('Find the best move for white.') verbatim; options are all legal moves in SAN"
    sampling = "balanced"
    notes = ("The position is shown after the opponent's move that starts the puzzle (given as the last move); the gold is the "
             "first move of the solution, which Lichess checked with Stockfish to be the only move that keeps the advantage "
             "(or mates). Mate-in-one puzzles with more than one mating move are left out. Only puzzles with a settled rating "
             "(rating deviation at most 90, at least 200 plays, popularity at least 70). Equal shares of six rating bands.")

    def load(self):
        load_env()
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(PUZZLES_REPO, PUZZLES_FILE, repo_type="dataset", revision=PUZZLES_REV)
        table = pq.read_table(path, columns=["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes"])
        keep = pc.and_(pc.and_(pc.less_equal(table["RatingDeviation"], 90), pc.greater_equal(table["NbPlays"], 200)),
                       pc.greater_equal(table["Popularity"], 70))
        table = table.filter(keep)
        ids, ratings = table["PuzzleId"].to_pylist(), table["Rating"].to_pylist()
        by_band = collections.defaultdict(list)
        for i, (pid, rating) in enumerate(zip(ids, ratings)):
            by_band[band_of(rating)].append((_order(pid), i))
        rows = []
        for band, members in sorted(by_band.items()):
            for _, i in sorted(members)[: PER_BAND * 2]:
                row = table.slice(i, 1).to_pylist()[0]
                record = self._check(row, band)
                if record:
                    rows.append(record)
                if sum(r["band"] == band for r in rows) >= PER_BAND:
                    break
        return rows

    @staticmethod
    def _check(row: dict, band: str) -> dict | None:
        moves = row["Moves"].split()
        start = Position.from_fen(row["FEN"])
        try:
            pos = start.push(start.parse_uci(moves[0]))
            legal = pos.legal_moves()
            solution = pos.parse_uci(moves[1], legal)
        except ValueError:
            return None
        if "mateIn1" in row["Themes"] and sum(gives_mate(pos, m) for m in legal) != 1:
            return None  # any mate solves a mate-in-one puzzle, so the gold would not be unique
        return {"id": row["PuzzleId"], "fen": row["FEN"], "moves": moves[:2], "rating": row["Rating"], "band": band,
                "themes": sorted(row["Themes"])}

    def label_of(self, record):
        return record["band"]

    def to_item(self, record, rng):
        start = Position.from_fen(record["fen"])
        opponent = start.parse_uci(record["moves"][0])
        last_move = start.san(opponent)
        pos = start.push(opponent)
        options, sans = move_options(pos)
        gold = sans[record["moves"][1]]
        state = {"fen": pos.fen(), "side_to_move": side_name(pos.turn), "last_move": f"{side_name(start.turn)} played {last_move}"}
        return self.item(state, {"best_move": best_move_question(pos, options)}, {"best_move": Gold(label=gold)},
                         original_id=record["id"], rating=record["rating"], band=record["band"], themes=record["themes"])


# ---------- ChessBench (Ruoss et al., 2024): Stockfish 16 action values of every legal move ----------

def bag_records(path) -> list[bytes]:
    """Records of a ChessBench .bag file: the data, then one little-endian uint64 end offset per record."""
    data = path.read_bytes()
    (index_start,) = struct.unpack("<Q", data[-8:])
    count = (len(data) - index_start) // 8
    ends = struct.unpack(f"<{count}q", data[index_start: index_start + 8 * count])
    out, start = [], 0
    for end in ends:
        out.append(data[start:end])
        start = end
    return out


def decode_record(record: bytes, kinds: str) -> list:
    """Apache Beam TupleCoder of StrUtf8 ('s') and big-endian double ('f') fields; the last string is not length-prefixed."""
    out, i = [], 0
    for j, kind in enumerate(kinds):
        if kind == "f":
            out.append(struct.unpack(">d", record[i: i + 8])[0])
            i += 8
        elif j == len(kinds) - 1:
            out.append(record[i:].decode())
            i = len(record)
        else:
            n, shift = 0, 0
            while True:
                byte = record[i]
                i += 1
                n |= (byte & 0x7F) << shift
                shift += 7
                if not byte & 0x80:
                    break
            out.append(record[i: i + n].decode())
            i += n
    if i != len(record):
        raise ValueError("unexpected record length")
    return out


@functools.lru_cache(maxsize=1)
def chessbench_action_values() -> dict[str, dict[str, float]]:
    """FEN -> {UCI move: win probability for the side to move after that move}, test split."""
    values: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for record in bag_records(fetch(CHESSBENCH + "test/action_value_data.bag", "chessbench_test_action_value_data.bag")):
        fen, move, p = decode_record(record, "ssf")
        values[fen][move] = p
    return dict(values)


def phase_of(pos: Position) -> str:
    worth = {"n": 3, "b": 3, "r": 5, "q": 9}
    material = sum(worth.get((p or "").lower(), 0) for p in pos.board)
    if pos.fullmove <= 12:
        return "opening"
    return "endgame" if material <= 26 else "middlegame"


@register_task
class ChessBenchBestMove(TaskAdapter):
    name = f"{LEAF}/chessbench"
    leaf = LEAF
    source = "ChessBench test positions with Stockfish 16 action values (Ruoss et al., NeurIPS 2024)"
    url = "https://github.com/google-deepmind/searchless_chess"
    license = CHESSBENCH_LICENSE
    tier = "B"
    criteria_source = "Lichess puzzle prompt ('Find the best move for white.'), applied to ChessBench positions; options are all legal moves in SAN"
    MARGIN = 0.2
    notes = ("Positions from the Lichess games that ChessBench holds out as its test set. Every legal move was scored by Stockfish 16 "
             "(0.05 s per move) as the side's win probability after the move; the gold is the top move, and only positions "
             "where it leads the second-best move by at least 0.2 are kept (about 5% of positions), so engine noise does "
             "not decide the answer.")

    def load(self):
        rows = []
        for fen, values in chessbench_action_values().items():
            if len(values) < 2:
                continue
            ranked = sorted(values.items(), key=lambda kv: -kv[1])
            if ranked[0][1] - ranked[1][1] < self.MARGIN:
                continue
            pos = Position.from_fen(fen)
            rows.append({"fen": fen, "best": ranked[0][0], "best_value": round(ranked[0][1], 4),
                         "margin": round(ranked[0][1] - ranked[1][1], 4), "phase": phase_of(pos), "moves": len(values)})
        return rows

    def label_of(self, record):
        return record["phase"]

    def to_item(self, record, rng):
        pos = Position.from_fen(record["fen"])
        options, sans = move_options(pos)
        if len(sans) != record["moves"] or record["best"] not in sans:
            return None  # the move lists disagree (never observed)
        state = {"fen": pos.fen(), "side_to_move": side_name(pos.turn)}
        return self.item(state, {"best_move": best_move_question(pos, options)}, {"best_move": Gold(label=sans[record["best"]])},
                         phase=record["phase"], best_win_probability=record["best_value"], margin=record["margin"])
