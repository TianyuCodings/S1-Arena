"""A small, dependency-free chess rules module: FEN, legal moves, SAN, applying moves.

Only what the adapters need: parse a FEN, list the legal moves, write and read moves in
UCI and SAN, and apply a move. Checked against perft counts and python-chess (see the
scratch checker); the adapters never import python-chess.
"""

from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefgh"
KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
ROOK = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP = ((1, 1), (1, -1), (-1, 1), (-1, -1))
PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def square(name: str) -> int:
    return FILES.index(name[0]) + 8 * (int(name[1]) - 1)


def square_name(s: int) -> str:
    return FILES[s % 8] + str(s // 8 + 1)


def _step(s: int, df: int, dr: int) -> int | None:
    f, r = s % 8 + df, s // 8 + dr
    return f + 8 * r if 0 <= f < 8 and 0 <= r < 8 else None


def color_of(piece: str) -> str:
    return "w" if piece.isupper() else "b"


@dataclass(frozen=True)
class Move:
    src: int
    dst: int
    promo: str | None = None  # 'q', 'r', 'b' or 'n'

    def uci(self) -> str:
        return square_name(self.src) + square_name(self.dst) + (self.promo or "")

    @staticmethod
    def from_uci(text: str) -> "Move":
        return Move(square(text[:2]), square(text[2:4]), text[4] if len(text) > 4 else None)


@dataclass(frozen=True)
class Position:
    board: tuple  # 64 entries, a1 = 0, h8 = 63; 'P'..'K' white, 'p'..'k' black, None empty
    turn: str  # 'w' or 'b'
    castling: str  # subset of 'KQkq', '' for none
    ep: int | None  # en passant target square after a double pawn push
    halfmove: int = 0
    fullmove: int = 1

    # ---------- FEN ----------

    @staticmethod
    def from_fen(fen: str) -> "Position":
        parts = fen.split()
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError(f"bad FEN: {fen}")
        board: list = [None] * 64
        for i, row in enumerate(rows):
            rank, f = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    board[f + 8 * rank] = ch
                    f += 1
            if f != 8:
                raise ValueError(f"bad FEN row: {row}")
        turn = parts[1] if len(parts) > 1 else "w"
        castling = "" if len(parts) < 3 or parts[2] == "-" else parts[2]
        ep = None if len(parts) < 4 or parts[3] == "-" else square(parts[3])
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        return Position(tuple(board), turn, castling, ep, halfmove, fullmove)

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for f in range(8):
                p = self.board[f + 8 * rank]
                if p is None:
                    empty += 1
                else:
                    row += (str(empty) if empty else "") + p
                    empty = 0
            rows.append(row + (str(empty) if empty else ""))
        ep = square_name(self.ep) if self.ep is not None and self._ep_capturable() else "-"
        return f"{'/'.join(rows)} {self.turn} {self.castling or '-'} {ep} {self.halfmove} {self.fullmove}"

    def _ep_capturable(self) -> bool:
        """Show the en passant square only when a legal en passant capture exists (python-chess's default)."""
        return any(m.dst == self.ep and self.board[m.src] in "Pp" for m in self.legal_moves())

    # ---------- attacks ----------

    def king_square(self, color: str) -> int | None:
        king = "K" if color == "w" else "k"
        return next((s for s, p in enumerate(self.board) if p == king), None)

    def attacked(self, s: int, by: str) -> bool:
        """Is square s attacked by any piece of color `by`?"""
        b = self.board
        pawn, knight, king = ("P", "N", "K") if by == "w" else ("p", "n", "k")
        rook_like, bishop_like = ("R", "Q") if by == "w" else ("r", "q"), ("B", "Q") if by == "w" else ("b", "q")
        dr = -1 if by == "w" else 1  # a white pawn attacks upwards, so it sits one rank below the target
        for df in (-1, 1):
            t = _step(s, df, dr)
            if t is not None and b[t] == pawn:
                return True
        for df, drr in KNIGHT:
            t = _step(s, df, drr)
            if t is not None and b[t] == knight:
                return True
        for df, drr in KING:
            t = _step(s, df, drr)
            if t is not None and b[t] == king:
                return True
        for dirs, pieces in ((ROOK, rook_like), (BISHOP, bishop_like)):
            for df, drr in dirs:
                t = _step(s, df, drr)
                while t is not None:
                    if b[t] is not None:
                        if b[t] in pieces:
                            return True
                        break
                    t = _step(t, df, drr)
        return False

    def in_check(self, color: str | None = None) -> bool:
        color = color or self.turn
        k = self.king_square(color)
        return k is not None and self.attacked(k, "b" if color == "w" else "w")

    # ---------- move generation ----------

    def pseudo_moves(self) -> list[Move]:
        b, me = self.board, self.turn
        enemy = "b" if me == "w" else "w"
        moves: list[Move] = []
        for s, p in enumerate(b):
            if p is None or color_of(p) != me:
                continue
            kind = p.lower()
            if kind == "p":
                up = 1 if me == "w" else -1
                start_rank, last_rank = (1, 7) if me == "w" else (6, 0)
                promos = ("q", "r", "b", "n")
                t = _step(s, 0, up)
                if t is not None and b[t] is None:
                    if t // 8 == last_rank:
                        moves += [Move(s, t, q) for q in promos]
                    else:
                        moves.append(Move(s, t))
                        t2 = _step(t, 0, up)
                        if s // 8 == start_rank and t2 is not None and b[t2] is None:
                            moves.append(Move(s, t2))
                for df in (-1, 1):
                    t = _step(s, df, up)
                    if t is None:
                        continue
                    ep_ok = t == self.ep and b[t] is None and t // 8 == (5 if me == "w" else 2)
                    if (b[t] is not None and color_of(b[t]) == enemy) or ep_ok:
                        if t // 8 == last_rank:
                            moves += [Move(s, t, q) for q in promos]
                        else:
                            moves.append(Move(s, t))
            elif kind in "nk":
                for df, dr in KNIGHT if kind == "n" else KING:
                    t = _step(s, df, dr)
                    if t is not None and (b[t] is None or color_of(b[t]) == enemy):
                        moves.append(Move(s, t))
            else:
                dirs = ROOK if kind == "r" else BISHOP if kind == "b" else ROOK + BISHOP
                for df, dr in dirs:
                    t = _step(s, df, dr)
                    while t is not None:
                        if b[t] is None:
                            moves.append(Move(s, t))
                        else:
                            if color_of(b[t]) == enemy:
                                moves.append(Move(s, t))
                            break
                        t = _step(t, df, dr)
        moves += self._castling_moves()
        return moves

    def _castling_moves(self) -> list[Move]:
        b, me = self.board, self.turn
        enemy = "b" if me == "w" else "w"
        rank = 0 if me == "w" else 7
        king, rook = ("K", "R") if me == "w" else ("k", "r")
        e = 4 + 8 * rank
        out = []
        if b[e] != king or self.attacked(e, enemy):
            return out
        rights = self.castling
        if (("K" if me == "w" else "k") in rights and b[7 + 8 * rank] == rook
                and b[5 + 8 * rank] is None and b[6 + 8 * rank] is None
                and not self.attacked(5 + 8 * rank, enemy) and not self.attacked(6 + 8 * rank, enemy)):
            out.append(Move(e, 6 + 8 * rank))
        if (("Q" if me == "w" else "q") in rights and b[0 + 8 * rank] == rook
                and b[1 + 8 * rank] is None and b[2 + 8 * rank] is None and b[3 + 8 * rank] is None
                and not self.attacked(3 + 8 * rank, enemy) and not self.attacked(2 + 8 * rank, enemy)):
            out.append(Move(e, 2 + 8 * rank))
        return out

    def is_castling(self, m: Move) -> bool:
        return self.board[m.src] in ("K", "k") and abs(m.dst - m.src) == 2

    def is_capture(self, m: Move) -> bool:
        return self.board[m.dst] is not None or (self.board[m.src] in ("P", "p") and m.dst == self.ep and m.dst % 8 != m.src % 8)

    def push(self, m: Move) -> "Position":
        """The position after move m (assumed pseudo-legal)."""
        b = list(self.board)
        p = b[m.src]
        me = self.turn
        capture = self.is_capture(m)
        if p in ("P", "p") and m.dst == self.ep and m.src % 8 != m.dst % 8 and b[m.dst] is None:
            b[m.dst + (-8 if me == "w" else 8)] = None  # en passant capture
        b[m.dst] = (m.promo.upper() if me == "w" else m.promo) if m.promo else p
        b[m.src] = None
        if self.is_castling(m):
            rank = m.src // 8
            if m.dst % 8 == 6:
                b[5 + 8 * rank], b[7 + 8 * rank] = b[7 + 8 * rank], None
            else:
                b[3 + 8 * rank], b[0 + 8 * rank] = b[0 + 8 * rank], None
        rights = self.castling
        for s, lost in ((4, "KQ"), (60, "kq"), (0, "Q"), (7, "K"), (56, "q"), (63, "k")):
            if m.src == s or m.dst == s:
                rights = "".join(c for c in rights if c not in lost)
        ep = None
        if p in ("P", "p") and abs(m.dst - m.src) == 16:
            ep = (m.src + m.dst) // 2
        halfmove = 0 if p in ("P", "p") or capture else self.halfmove + 1
        fullmove = self.fullmove + (1 if me == "b" else 0)
        return Position(tuple(b), "b" if me == "w" else "w", rights, ep, halfmove, fullmove)

    def legal_moves(self) -> list[Move]:
        me = self.turn
        return [m for m in self.pseudo_moves() if not self.push(m).in_check(me)]

    def is_checkmate(self) -> bool:
        return self.in_check() and not self.legal_moves()

    # ---------- notation ----------

    def san(self, m: Move, legal: list[Move] | None = None) -> str:
        legal = legal if legal is not None else self.legal_moves()
        p = self.board[m.src]
        kind = p.lower()
        if self.is_castling(m):
            text = "O-O" if m.dst % 8 == 6 else "O-O-O"
        elif kind == "p":
            text = (FILES[m.src % 8] + "x" if self.is_capture(m) else "") + square_name(m.dst)
            if m.promo:
                text += "=" + m.promo.upper()
        else:
            others = [o for o in legal if o.dst == m.dst and o.src != m.src and self.board[o.src] == p]
            dis = ""
            if others:
                if all(o.src % 8 != m.src % 8 for o in others):
                    dis = FILES[m.src % 8]
                elif all(o.src // 8 != m.src // 8 for o in others):
                    dis = str(m.src // 8 + 1)
                else:
                    dis = square_name(m.src)
            text = kind.upper() + dis + ("x" if self.is_capture(m) else "") + square_name(m.dst)
        after = self.push(m)
        if after.in_check():
            text += "#" if not after.legal_moves() else "+"
        return text

    def parse_san(self, text: str, legal: list[Move] | None = None) -> Move:
        """Read a move in SAN ('Nbd7', 'exd6', 'e8=Q+', 'O-O'), matching it against the legal moves."""
        legal = legal if legal is not None else self.legal_moves()
        t = _strip(text)
        if t in ("O-O", "O-O-O"):
            found = [m for m in legal if self.is_castling(m) and (m.dst % 8 == 6) == (t == "O-O")]
        else:
            promo = None
            if "=" in t:
                t, promo = t.split("=")
                promo = promo.lower()
            elif t[-1] in "QRBN" and t[0].islower():  # 'e8Q'
                t, promo = t[:-1], t[-1].lower()
            dst, rest = square(t[-2:]), t[:-2].replace("x", "").replace("-", "")
            piece = rest[0] if rest and rest[0] in "NBRQK" else "P"
            dis = rest[1:] if piece != "P" else rest
            found = []
            for m in legal:
                p = self.board[m.src]
                if m.dst != dst or p.upper() != piece or m.promo != promo or self.is_castling(m):
                    continue
                src = square_name(m.src)
                if all(c in src for c in dis):
                    found.append(m)
        if len(found) != 1:
            raise ValueError(f"cannot read move {text!r} in {self.fen()}")
        return found[0]

    def parse_uci(self, text: str, legal: list[Move] | None = None) -> Move:
        legal = legal if legal is not None else self.legal_moves()
        m = Move.from_uci(text)
        if m not in legal:
            raise ValueError(f"illegal move {text} in {self.fen()}")
        return m


def _strip(san: str) -> str:
    return san.replace("0-0-0", "O-O-O").replace("0-0", "O-O").rstrip("+#!?")


def describe_move(pos: Position, m: Move) -> str:
    """'Knight from g1 to f3', 'pawn from e7 to e8, promoting to a queen', 'castle kingside'."""
    if pos.is_castling(m):
        return "king castles kingside" if m.dst % 8 == 6 else "king castles queenside"
    piece = PIECE_NAMES[pos.board[m.src].lower()]
    text = f"{piece} from {square_name(m.src)} to {square_name(m.dst)}"
    if pos.is_capture(m):
        captured = pos.board[m.dst]
        text += f", capturing the {PIECE_NAMES[captured.lower()]}" if captured else ", capturing en passant"
    if m.promo:
        text += f", promoting to a {PIECE_NAMES[m.promo]}"
    return text


def board_diagram(pos: Position) -> str:
    """Eight lines, rank 8 first; '.' marks an empty square."""
    lines = []
    for rank in range(7, -1, -1):
        row = " ".join(pos.board[f + 8 * rank] or "." for f in range(8))
        lines.append(f"{rank + 1} {row}")
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def side_name(turn: str) -> str:
    return "White" if turn == "w" else "Black"


def replay(pos: Position, moves) -> Position:
    for m in moves:
        pos = pos.push(m)
    return pos


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
