"""Cruciverb-IT generative tasks: public prompts, final-answer parsing and scoring.

Task 2 grid metrics are adapted from the official scorer at revision
0cee7892cf4f192032c537f77944fcde2fbbe3a3:
https://huggingface.co/datasets/cruciverb-it/evalita2026/blob/0cee7892cf4f192032c537f77944fcde2fbbe3a3/task_2/task_2_scorer.py
The arithmetic is preserved; normalization adds casefold to Unidecode.
Task 1 uses corrected MRR@10 (see this task's README section).
No prompt generator or prediction parser reads gold answers.
"""

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Optional


# Final response boundaries. Configure native reasoning separation in the backend.
def _final_answer(value):
    if isinstance(value, Mapping):
        # Never promote raw_output or reasoning if a final response is missing.
        value = value.get("final_answer", "")
    if not isinstance(value, str):
        return ""
    text = value.strip()
    # GPT-OSS/Harmony style explicit final channel.
    final_marker = re.compile(r"<\|channel\|>final(?:<\|message\|>|\s*)")
    matches = list(final_marker.finditer(text))
    if matches:
        text = text[matches[-1].end() :]
    elif "<|channel|>" in text or "<|message|>" in text:
        return ""
    # Gemma's native thinking field closes with <channel|>.
    while "<|channel>thought\n" in text:
        start = text.index("<|channel>thought\n")
        end = text.find("<channel|>", start)
        if end < 0:
            return ""
        text = text[:start] + text[end + len("<channel|>") :]
    if "<|channel>" in text or "<channel|>" in text:
        return ""
    for tag in ["think", "analysis"]:
        opening = re.compile(r"<\s*" + tag + r"\s*>", re.IGNORECASE)
        closing = re.compile(r"<\s*/\s*" + tag + r"\s*>", re.IGNORECASE)
        text = re.sub(
            r"<\s*" + tag + r"\s*>.*?<\s*/\s*" + tag + r"\s*>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if opening.search(text):
            return ""
        endings = list(closing.finditer(text))
        if endings:
            text = text[endings[-1].end() :]
    text = re.sub(
        r"^(?:<\|im_start\|>assistant\s*|<｜Assistant｜>|<\|assistant\|>)", "", text
    )
    for marker in [
        "</s>",
        "<|im_end|>",
        "<|endoftext|>",
        "<｜end▁of▁sentence｜>",
        "<|eot_id|>",
        "<turn|>",
        "<|return|>",
        "<|end|>",
    ]:
        text = text.split(marker, 1)[0]
    if any(
        marker in text
        for marker in ["<|im_start|>", "<|start|>", "<|channel|>", "<|user|>"]
    ):
        return ""
    return text.strip()


def _first_result(results):
    if isinstance(results, (str, bytes)):
        raise TypeError(
            "results must be the harness sequence of generations, not a string"
        )
    return _final_answer(results[0]) if results else ""


# Public grid geometry and conventional numbering.
@dataclass(frozen=True)
class Entry:
    clue: str
    row: int
    col: int
    direction: str
    length: int
    number: int

    @property
    def answer_id(self):
        return ("O" if self.direction == "A" else "V") + str(self.number)

    @property
    def positions(self):
        dr, dc = (0, 1) if self.direction == "A" else (1, 0)
        return tuple((self.row + i * dr, self.col + i * dc) for i in range(self.length))


@dataclass(frozen=True)
class Puzzle:
    grid: tuple[tuple[str, ...], ...]
    entries: tuple[Entry, ...]


def prepare(item: dict) -> Puzzle:
    """Legge ESCLUSIVAMENTE empty_grid e i campi pubblici delle definizioni."""
    grid = item.get("empty_grid")
    if (
        not isinstance(grid, list)
        or not grid
        or not isinstance(grid[0], list)
        or not grid[0]
    ):
        raise ValueError("empty_grid deve essere una matrice non vuota.")
    height, width = len(grid), len(grid[0])
    if any(
        not isinstance(row, list)
        or len(row) != width
        or any(cell not in (" ", ".") for cell in row)
        for row in grid
    ):
        raise ValueError("empty_grid deve essere rettangolare, con soli spazi e punti.")
    clues = item.get("clues")
    if not isinstance(clues, list) or not clues:
        raise ValueError("Definizioni mancanti.")
    visible = []
    for clue in clues:
        if not isinstance(clue, dict) or any(
            type(clue.get(k)) is not int for k in ("row", "col", "length")
        ):
            raise ValueError("Coordinate o lunghezza non valide.")
        row, col, length = clue["row"], clue["col"], clue["length"]
        direction, text = clue.get("direction"), clue.get("clue")
        if direction not in ("A", "D") or length < 1:
            raise ValueError("Direzione o lunghezza non valida.")
        if (
            not isinstance(text, str)
            or not text.strip()
            or "\n" in text
            or "\r" in text
        ):
            raise ValueError("La definizione deve essere un testo su una sola riga.")
        dr, dc = (0, 1) if direction == "A" else (1, 0)
        for i in range(length):
            r, c = row + i * dr, col + i * dc
            if not (0 <= r < height and 0 <= c < width) or grid[r][c] == ".":
                raise ValueError(
                    "Una definizione esce dalla griglia o attraversa una casella nera."
                )
        visible.append((text, row, col, direction, length))
    starts = sorted({(c[1], c[2]) for c in visible})
    numbers = {position: index for index, position in enumerate(starts, 1)}
    entries = [
        Entry(text, row, col, direction, length, numbers[(row, col)])
        for text, row, col, direction, length in visible
    ]
    if len({entry.answer_id for entry in entries}) != len(entries):
        raise ValueError("Definizioni duplicate nella stessa posizione e direzione.")
    entries.sort(key=lambda entry: (entry.direction != "A", entry.number))
    return Puzzle(tuple(tuple(row) for row in grid), tuple(entries))


def render_grid(puzzle: Puzzle, numbered: bool = False) -> str:
    rows, cols = len(puzzle.grid), len(puzzle.grid[0])
    if numbered:
        starts = {(entry.row, entry.col): entry.number for entry in puzzle.entries}
        cw = len(str(max(starts.values())))
        return "\n".join(
            " ".join(
                ("#" if cell == "." else str(starts.get((r, c), "_"))).rjust(cw)
                for c, cell in enumerate(row)
            )
            for r, row in enumerate(puzzle.grid)
        )
    rw, cw = len(str(rows - 1)), len(str(cols - 1))
    lines = [" " * (rw + 3) + " ".join(str(col).rjust(cw) for col in range(cols))]
    for index, row in enumerate(puzzle.grid):
        cells = " ".join(("#" if cell == "." else "_").rjust(cw) for cell in row)
        lines.append(str(index).rjust(rw) + "   " + cells)
    return "\n".join(lines)


def render_clues(puzzle: Puzzle, direction: str, coordinates: bool) -> str:
    lines = []
    for entry in puzzle.entries:
        if entry.direction != direction:
            continue
        text = f"{entry.number}. {entry.clue}"
        if coordinates:
            text += (
                f" | inizio_parola=({entry.row},{entry.col})"
                f" | lunghezza_parola={entry.length}"
            )
        lines.append(text)
    return "\n".join(lines)


# Task 1 candidates and ranking.
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


_ENUM = re.compile(r"^\s*(?:\d+[.)]|\(\d+\))\s*")


_BULLET = re.compile(r"^\s*[-*+•]\s+")


_CUE = re.compile(
    r"\b(?:(?:la|le|mia|mie|the|my)\s+)?"
    r"(?:rispost[ae]|soluzion[ei]|candidat[oi]|answers?|solutions?)"
    r"(?:\s+(?:final[ei]|definitiv[ae]|corrett[ae]|possibili|propost[ae]|"
    r"pi[uù]\s+probabil[ei]|che\s+propongo))*\s*"
    r"(?:[:=]|\b(?:è|é|e|sono|is|are|potrebbe\s+essere)\b)\s*",
    re.IGNORECASE,
)


_PROPOSAL = re.compile(
    r"\b(?:propongo|suggerisco|scelgo)\s+(?:come\s+risposta\s+)?", re.IGNORECASE
)


_NEGATIVE_START = re.compile(
    r"^(?:non\b|no\b|not\b|escludo\b|scarto\b|nessun[ao]?\b|"
    r"non\s+saprei\b|nessuna\s+risposta\b)",
    re.IGNORECASE,
)


_REJECTED = re.compile(
    r"\b(?:scartat[ao]|sbagliat[ao]|errat[ao]|esclus[ao]|incorrect|rejected)\b|"
    r"\bnon\s+(?:è|e|era|sarebbe)\s+(?:la\s+)?"
    r"(?:risposta|soluzione|corrett[ao]|valid[ao])\b|"
    r"\b(?:not\s+(?:correct|the\s+answer)|wrong\s+answer)\b",
    re.IGNORECASE,
)


_EXPLANATION = re.compile(
    r"\s+(?:perch[eé]|poich[eé]|in\s+quanto|che|significa|indica|"
    r"because|which|means|è|is)\b",
    re.IGNORECASE,
)


_CONTROL = re.compile(
    r"<\s*/?\s*think\s*>|<\|channel>|<channel\|>|"
    r"<\|(?:im_start|im_end|endoftext|assistant|user|observation)\|>|"
    r"<｜(?:Assistant|User|end▁of▁sentence)｜>|<turn\|>|</s>",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _unidecode():
    try:
        from unidecode import unidecode
    except ImportError as error:
        raise RuntimeError(
            "Install the scorer dependency: pip install Unidecode==1.4.0"
        ) from error
    return unidecode


def _normalize(value: str) -> str:
    # Preserve answer-internal whitespace and punctuation, as in the old scorer.
    return _unidecode()(value).casefold()


def _unwrap(text: str) -> str:
    text = text.strip()
    pairs = [
        ("**", "**"),
        ("__", "__"),
        ("`", "`"),
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("*", "*"),
        ("_", "_"),
    ]
    changed = True
    while changed and text:
        changed = False
        for opening, closing in pairs:
            if (
                len(text) > len(opening) + len(closing)
                and text.startswith(opening)
                and text.endswith(closing)
            ):
                text = text[len(opening) : -len(closing)].strip()
                changed = True
                break
    return text


def _cue_body(line: str) -> Optional[str]:
    # Negative mentions are not promoted to a proposed answer. Restrict the
    # check to the current clause, so "Non è EURO. La risposta è LIRA" works.
    for match in sorted(
        [*_CUE.finditer(line), *_PROPOSAL.finditer(line)], key=lambda m: m.start()
    ):
        prefix = re.split(r"[.!?;]", line[: match.start()])[-1]
        if re.search(r"\b(?:non|not|escludo|scarto)\b", prefix, re.IGNORECASE):
            continue
        return line[match.end() :].strip()
    return None


def _split_items(text: str) -> list[str]:
    """Split separators outside quoted candidates and parenthetical notes."""
    parts, start, depth, quote = [], 0, 0, None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
        elif char in '"`“‘' or (
            char == "'" and (index == 0 or not text[index - 1].isalnum())
        ):
            quote = {"“": "”", "‘": "’"}.get(char, char)
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif depth == 0:
            width = 0
            if char in ",;":
                width = 1
            elif char.isspace():
                match = re.match(
                    r"\s+(?:/|oppure|o|or)\s+", text[index:], re.IGNORECASE
                )
                if match:
                    width = match.end()
            if width:
                parts.append(text[start:index].strip())
                index += width
                start = index
                continue
        index += 1
    parts.append(text[start:].strip())
    return parts


def _candidate(text: str, *, explicit: bool = False) -> Optional[str]:
    text = _ENUM.sub("", text.strip())
    text = _BULLET.sub("", text)
    body = _cue_body(text)
    if body is not None:
        text, explicit = body, True
    # Strip only surrounding presentation punctuation, never internal letters.
    original = text.rstrip(".!? ")
    text = _unwrap(original)
    marked = original != text
    if not text:
        return None
    # Words such as NON, NO or ERRATA can themselves be crossword answers.
    # Negation/rejection checks concern phrases, not these isolated tokens.
    if not _WORD.fullmatch(text) and (
        _NEGATIVE_START.search(text) or _REJECTED.search(text)
    ):
        return None
    if explicit:
        text = re.sub(
            r"^(?:(?:la\s+parola|il\s+termine|la|il|lo)\s+|l['’])",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"^(?:probabilmente|forse|verosimilmente)\s+", "", text, flags=re.IGNORECASE
        )
    # "Moneta cipriota: LIRA" also marks an answer, without requiring the
    # literal label "Risposta:". Prefer the left side if it is itself a word.
    if ":" in text:
        head, tail = text.split(":", 1)
        if not _WORD.fullmatch(_unwrap(head.strip())):
            return _candidate(tail, explicit=True)
    # Answer before its parenthetical note, colon or dash explanation.
    text = re.split(r"\s+[—–-]\s+|\s*:\s*", text, maxsplit=1)[0].strip()
    text = re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()
    text = _EXPLANATION.split(text, maxsplit=1)[0].strip()
    before = text.rstrip(".!? ")
    text = _unwrap(before)
    marked = marked or before != text
    if _WORD.fullmatch(text):
        return text
    # Quoted/highlighted multiword candidates are explicit strings; a prose
    # sentence without such a boundary is not a multiword answer by default.
    if marked and re.fullmatch(
        r"[^\W_]+(?:['’\-][^\W_]+)*(?:\s+[^\W_]+(?:['’\-][^\W_]+)*)*", text
    ):
        return text
    return None


def _literal(text: str) -> Optional[list[str]]:
    # This parser only accepts a flat list/tuple of strings; no eval, dict
    # values, nested objects, or filtering of malformed elements.
    if not text.lstrip().startswith(("[", "(")) or len(text) > 100_000:
        return None
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return [item.strip() for item in value]
    return None


def _payload(line: str, *, explicit: bool = False) -> Optional[tuple[list[str], str]]:
    kind = (
        "numbered"
        if _ENUM.match(line)
        else "bullet"
        if _BULLET.match(line)
        else "single"
    )
    text = _BULLET.sub("", _ENUM.sub("", line.strip()))
    body = _cue_body(text)
    if body is not None:
        # A whole-line label can introduce a literal list. Labels inside a
        # comma list are handled per element instead, keeping earlier ranks.
        prefix = text[: len(text) - len(body)] if body else text
        if len(_split_items(prefix)) == 1:
            text, explicit = body, True
    if not text.strip():
        return None
    literal = _literal(text)
    if literal is not None:
        return literal, "list"
    parts = _split_items(text)
    candidates = []
    for part in parts:
        # Empty explicit slots retain their rank. Don't remove duplicates or
        # a bad-length candidate before a correct later candidate.
        if not part:
            candidates.append("")
            continue
        candidate = _candidate(part, explicit=explicit or kind != "single")
        if candidate is None:
            # An explanatory tail after an answer is harmless. If a preceding
            # slot is not recognizable, do not promote a later word to rank 1.
            if not candidates:
                return None
            break
        candidates.append(candidate)
    return (candidates, "list" if len(parts) > 1 else kind) if candidates else None


def _block(lines: list[str], start: int, body: Optional[str] = None) -> list[str]:
    explicit = body is not None
    index = start
    line = body if explicit else lines[index]
    # A heading may precede a blank line or a fenced list.
    while not line.strip() or line.lstrip().startswith("```"):
        index += 1
        if index >= len(lines):
            return []
        line = lines[index]
    literal_lines = [line]
    if line.lstrip().startswith(("[", "(")) and not _ENUM.match(line):
        # Flat Python/JSON lists may span several lines. Stop at the first
        # complete literal; ignore later prose or a different list.
        for end in range(index, len(lines)):
            if end > index:
                literal_lines.append(lines[end])
            literal = _literal("\n".join(literal_lines))
            if literal is not None:
                return literal
            if lines[end].lstrip().startswith("```"):
                return []
        return []
    first = _payload(line, explicit=explicit)
    if first is None:
        return []
    candidates, kind = first
    if kind == "list":
        return candidates
    for following in lines[index + 1 :]:
        if not following.strip() or following.lstrip().startswith("```"):
            break
        item = _payload(following)
        if (
            item is None
            or item[1] != kind
            or (kind == "single" and _cue_body(following) is not None)
        ):
            break
        candidates.extend(item[0])
    return candidates


def _parse_task1_candidates(final_answer: Any) -> list[str]:
    """Extract the first answer block, giving explicit answer labels priority.

    Input must be final content, already separated by the native model/API
    adapter. Native control tokens are refused, including unfinished thinking.
    Accepted: singleton, comma/semicolon/slash/alternative lists, numbered or
    bullet lists, one word per line, flat Python/JSON string lists, Markdown
    wrappers, answer labels and common trailing explanations. Parsing is
    deterministic and gold-independent. Repeated blocks are not concatenated.
    """
    if (
        not isinstance(final_answer, str)
        or not final_answer.strip()
        or _CONTROL.search(final_answer)
    ):
        return []
    lines = final_answer.strip().splitlines()
    # A label is stronger evidence than incidental short prose elsewhere.
    for index, line in enumerate(lines):
        body = _cue_body(line)
        if body is not None:
            if _payload(line) is not None:
                # Keep numbered/bullet prefixes and earlier candidates on the
                # same line instead of restarting a list at its answer label.
                start = index
                kind = _payload(line)[1]
                if kind in {"numbered", "bullet"}:
                    while start > 0:
                        previous = _payload(lines[start - 1])
                        if previous is None or previous[1] != kind:
                            break
                        start -= 1
                return _block(lines, start)
            return _block(lines, index, body)
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            continue
        if _payload(line) is not None or line.lstrip().startswith(("[", "(")):
            return _block(lines, index)
    return []


def task1_gold_rank(doc: Mapping[str, Any], candidates: Sequence[str]) -> Optional[int]:
    """First 1-based gold rank in the ENTIRE extracted list (may exceed 10)."""
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not all(isinstance(item, str) for item in candidates)
    ):
        raise TypeError("candidates must be an ordered sequence of strings")
    gold = doc["answer"]
    if not isinstance(gold, str) or not gold:
        raise ValueError("doc['answer'] must be a nonempty string")
    gold_normalized = _normalize(gold)
    return next(
        (
            rank
            for rank, candidate in enumerate(candidates, 1)
            if _normalize(candidate) == gold_normalized
        ),
        None,
    )


def score_task1_candidates(
    doc: Mapping[str, Any], candidates: Sequence[str]
) -> dict[str, float]:
    rank = task1_gold_rank(doc, candidates)
    in_top_ten = rank is not None and rank <= 10
    return {
        "acc@1": float(rank == 1),
        "acc@10": float(in_top_ten),
        "mrr": 1.0 / rank if in_top_ten else 0.0,
    }


# Task 2 grid and numbered-answer parsing.
@dataclass
class ParsedOutput:
    grid: Optional[list[list[Optional[str]]]]
    answers: dict[str, str]
    source: str
    issues: list[str]
    has_answer_list: bool = False

    def to_dict(self):
        return asdict(self)


def final_text(raw: Optional[str]) -> str:
    """Non utilizzare mai come risposta un blocco reasoning incompleto."""
    if not isinstance(raw, str):
        return ""
    text = raw
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            return ""
    elif "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    while "<|channel>thought\n" in text:
        start = text.index("<|channel>thought\n")
        end = text.find("<channel|>", start)
        if end < 0:
            return ""
        text = text[:start] + text[end + len("<channel|>") :]
    if "<|channel>" in text or "<channel|>" in text:
        return ""  # Un canale non riconosciuto va separato dal codec nativo.
    for marker in ("<turn|>", "<|im_end|>", "<|endoftext|>", "<|end|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def canonical_grid(value, puzzle: Puzzle):
    if not isinstance(value, list) or len(value) != len(puzzle.grid):
        return None
    result = []
    starts = {(e.row, e.col): str(e.number) for e in puzzle.entries}
    for r, (row, expected) in enumerate(zip(value, puzzle.grid)):
        if not isinstance(row, list) or len(row) != len(expected):
            return None
        converted = []
        for c, cell in enumerate(row):
            marker = starts.get((r, c))
            if (
                marker is not None
                and type(cell) in (str, int)
                and str(cell) == marker
                or cell is None
            ):
                converted.append(None)
            elif isinstance(cell, str) and len(cell) == 1:
                converted.append(
                    "." if cell == "#" else None if cell in ("_", "?", " ") else cell
                )
            else:
                return None
        result.append(converted)
    return result


def _list_literals(text):
    """Limiti sintattici delle liste; parentesi nelle stringhe ignorate."""
    index = 0
    while index < len(text):
        if text[index] != "[":
            index += 1
            continue
        start, depth, quote, triple, escaped = index, 0, None, False, False
        while index < len(text):
            char = text[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif triple and text.startswith(quote * 3, index):
                    quote = None
                    triple = False
                    index += 3
                    continue
                elif not triple and char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
                triple = text.startswith(char * 3, index)
                if triple:
                    index += 3
                    continue
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1], start, index + 1
                    index += 1
                    break
            index += 1
        else:
            return  # Lista troncata: nessuna riparazione automatica.


def _row_values(line: str, width: int):
    text = line.strip()
    if not text or text.startswith("```"):
        return None
    tokens = (
        [v.strip() for v in text.strip("|").split("|")] if "|" in text else text.split()
    )
    tokens = [v for v in tokens if v != ""]
    if tokens == [str(i) for i in range(width)]:
        return None  # Intestazione delle colonne, non caselle.
    row_number = None
    if len(tokens) == width + 1 and tokens[0].isdigit():
        row_number, tokens = int(tokens[0]), tokens[1:]
    if len(tokens) == 2 and tokens[0].isdigit() and len(tokens[1]) == width:
        row_number, tokens = int(tokens[0]), list(tokens[1])
    elif len(tokens) == 1 and len(tokens[0]) == width:
        tokens = list(tokens[0])
    if len(tokens) != width:
        return None
    result = []
    for cell in tokens:
        if cell in ("None", "null"):
            result.append(None)
        else:
            cell = cell.strip("'\"`")
            if len(cell) != 1 and not cell.isdigit():
                return None
            result.append(
                "." if cell == "#" else None if cell in ("_", "?", " ") else cell
            )
    return result, row_number


def _grids(text: str, puzzle: Puzzle):
    candidates, excluded_lines = [], set()
    for literal, start, end in _list_literals(text):
        try:
            value = ast.literal_eval(literal)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            try:
                value = json.loads(literal)
            except (ValueError, TypeError, RecursionError):
                continue
        grid = canonical_grid(value, puzzle)
        if grid is not None:
            candidates.append(grid)
            excluded_lines.update(
                range(text.count("\n", 0, start), text.count("\n", 0, end) + 1)
            )
    buffer, positions = [], []
    height, width = len(puzzle.grid), len(puzzle.grid[0])
    for line_index, line in enumerate(text.splitlines()):
        if line_index in excluded_lines:
            buffer, positions = [], []
            continue
        parsed = _row_values(line, width)
        if parsed is None:
            buffer, positions = [], []
            continue
        cells, label = parsed
        if label is not None and label != len(buffer):
            buffer, positions = [], []
            if label != 0:
                continue
        buffer.append(cells)
        positions.append(line_index)
        if len(buffer) == height:
            grid = canonical_grid(buffer, puzzle)
            if grid is not None:
                candidates.append(grid)
                excluded_lines.update(positions)
            buffer, positions = [], []
    distinct = {}
    for grid in candidates:
        distinct.setdefault(json.dumps(grid, ensure_ascii=False), grid)
    return list(distinct.values()), excluded_lines


_SECTION = re.compile(
    r"^\s*(?:[-+*•–—]\s*)?(?:#{1,6}\s*)?"
    r"(orizzontali|orizzontale|verticali|verticale|across|down)\s*:?\s*$",
    re.IGNORECASE,
)


_PAIR = re.compile(
    r"^\s*(?P<prefix>[OVAD])?\s*(?P<number>[1-9][0-9]*)\s*(?P<suffix>[OVAD])?"
    r"(?:\s*[.\):=\-]\s*|\s+)(?P<value>.+?)\s*$",
    re.IGNORECASE,
)


def _answer(value: str, clue_text: str):
    text = value.strip()
    suffix = (
        text[len(clue_text) :]
        if text.casefold().startswith(clue_text.casefold())
        else ""
    )
    if suffix.lstrip().startswith(
        (":", ";", "=", "–", "—", "-", "(lunghezza", "(linghezza")
    ):
        text = suffix.lstrip(" :;=–—-")
        text = re.sub(r"^\((?:lunghezza|linghezza)_parola=\d+\)\s*", "", text)
        text = text.lstrip(" :;=–—-")
    text = re.sub(
        r"^(?:(?:la\s+)?(?:risposta|soluzione|answer)\s*(?::|=|è)\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Il primo termine in una risposta annotata, non una ricerca della gold.
    text = re.split(r"\s+(?:[-–—]|\()", text, maxsplit=1)[0]
    text = text.strip(" \t\"'`*.,;:!")
    if text and all(char.isalpha() or char in ("?", "_") for char in text):
        return text
    return None


def _answers(text, puzzle, excluded_lines):
    mapping = {entry.answer_id: entry for entry in puzzle.entries}
    by_number = {}
    for entry in puzzle.entries:
        by_number.setdefault(entry.number, []).append(entry.answer_id)
    collected, issues = {}, []
    direction = None
    for line_index, raw_line in enumerate(text.splitlines()):
        if line_index in excluded_lines:
            continue
        line = raw_line.replace("**", "").replace("`", "").strip()
        section = _SECTION.fullmatch(line)
        if section:
            direction = (
                "A"
                if section.group(1).lower() in ("orizzontali", "orizzontale", "across")
                else "D"
            )
            continue
        if line.startswith("|"):
            cells = [v.strip() for v in line.strip("|").split("|")]
            if len(cells) == 2:
                line = cells[0] + ": " + cells[1]
        for segment in line.split(";"):
            match = _PAIR.fullmatch(segment)
            if not match:
                continue
            explicit = match.group("prefix") or match.group("suffix")
            if match.group("prefix") and match.group("suffix"):
                continue
            number = int(match.group("number"))
            if explicit:
                current_direction = "A" if explicit.upper() in ("O", "A") else "D"
            else:
                current_direction = direction
            if current_direction:
                identifier = ("O" if current_direction == "A" else "V") + str(number)
            else:
                options = by_number.get(number, [])
                if len(options) != 1:
                    issues.append(
                        f"Numero {number} senza direzione, ambiguo o sconosciuto."
                    )
                    continue
                identifier = options[0]
            if identifier not in mapping:
                issues.append(f"ID sconosciuto: {identifier}.")
                continue
            answer = _answer(match.group("value"), mapping[identifier].clue)
            if answer is not None:
                collected.setdefault(identifier, []).append(answer)
    answers = {}
    for identifier, candidates in collected.items():
        # Ripetere la stessa risposta non cambia il risultato. Correzioni
        # discordanti restano ambigue; non scegliere la risposta usando gold.
        distinct = {_normalize(value): value for value in candidates}
        if len(distinct) == 1:
            answers[identifier] = candidates[0]
        else:
            issues.append(f"Risposte discordanti per {identifier}.")
    return answers, bool(collected), issues


def _insert_answers(puzzle, answers):
    grid = [["." if cell == "." else None for cell in row] for row in puzzle.grid]
    conflicts, issues = set(), []
    for entry in puzzle.entries:
        answer = answers.get(entry.answer_id)
        if answer is None:
            continue
        if len(answer) != entry.length:
            issues.append(
                f"{entry.answer_id}: lunghezza {len(answer)}, attesa {entry.length}; non inserita."
            )
            continue
        for (row, col), letter in zip(entry.positions, answer):
            if letter in ("?", "_") or (row, col) in conflicts:
                continue
            current = grid[row][col]
            if current is None:
                grid[row][col] = letter
            elif _normalize(current) != _normalize(letter):
                grid[row][col] = None
                conflicts.add((row, col))
    if conflicts:
        issues.append(f"{len(conflicts)} incroci discordanti, lasciati None.")
    return grid, issues


def parse_output(raw_output: Optional[str], doc: dict) -> ParsedOutput:
    puzzle = prepare(doc)
    text = final_text(raw_output)
    if not text or text == "None":
        return ParsedOutput(None, {}, "no_solution", [])
    grids, excluded = _grids(text, puzzle)
    answers, has_list, issues = _answers(text, puzzle, excluded)
    if len(grids) > 1:
        return ParsedOutput(
            None,
            answers,
            "ambiguous_grids",
            issues + ["Più griglie differenti nel finale."],
            has_list,
        )
    if grids:
        # Una griglia esplicita è il risultato primario. Non migliorarla con
        # le risposte elencate a parte; queste servono solo alla WA diretta.
        return ParsedOutput(grids[0], answers, "grid", issues, has_list)
    if has_list:
        grid, insertion_issues = _insert_answers(puzzle, answers)
        return ParsedOutput(grid, answers, "answers", issues + insertion_issues, True)
    return ParsedOutput(None, {}, "unparsed", issues)


def parse_prediction(raw_output: Optional[str], doc: dict):
    return parse_output(raw_output, doc).grid


# Official Task 2 grid metrics, embedded to keep this utility self-contained.
def _validate_same_length(pred: list[Any], gold: list[Any], clues: list[Any]) -> None:
    n1, n2, n3 = len(pred), len(gold), len(clues)
    if not (n1 == n2 == n3):
        raise AssertionError(
            f"Files must have SAME number of items: pred={n1}, gold={n2}, clues={n3}. "
            "Ensure rows align by example."
        )


def _validate_grid(grid: list[list[Optional[str]]], name: str) -> tuple[int, int]:
    if not isinstance(grid, list) or len(grid) == 0:
        raise AssertionError(f"{name}: must be a non-empty list of rows.")
    n_cols = None
    for r, row in enumerate(grid):
        if not isinstance(row, list) or len(row) == 0:
            raise AssertionError(
                f"{name}: row {r} must be a non-empty list of characters."
            )
        if n_cols is None:
            n_cols = len(row)
        elif len(row) != n_cols:
            raise AssertionError(
                f"{name}: row {r} has {len(row)} columns but expected {n_cols} (ragged grid)."
            )
        for c, ch in enumerate(row):
            # gold grid can contain '.' (black square) or an uppercase/lowercase letter (string of length 1)
            # pred grid can contain None or a single character; we unify checks here and allow '.' too.
            if ch is not None and not (isinstance(ch, str) and len(ch) == 1):
                raise AssertionError(
                    f"{name}: cell ({r},{c}) must be None or a single-character string, got: {ch!r}"
                )
    return (len(grid), n_cols)


def _validate_pair_shapes(
    solved: list[list[Optional[str]]], gold: list[list[Optional[str]]], idx: int
) -> tuple[int, int]:
    sr, sc = _validate_grid(solved, f"pred_grid[{idx}]")
    gr, gc = _validate_grid(gold, f"gold_grid[{idx}]")
    if (sr, sc) != (gr, gc):
        raise AssertionError(
            f"Grid shape mismatch at example {idx}: pred={sr}x{sc} vs gold={gr}x{gc}."
        )
    return sr, sc


def _validate_clues_for_grid(
    clues: list[dict[str, Any]], rows: int, cols: int, idx: int
) -> None:
    if not isinstance(clues, list) or len(clues) == 0:
        raise AssertionError(f"clues[{idx}]: must be a non-empty list.")

    errors = []
    for j, clue in enumerate(clues):
        if not isinstance(clue, dict):
            errors.append(
                f"[{idx}] clue[{j}] must be a dict, got {type(clue).__name__}"
            )
            if len(errors) >= 20:
                break
            continue
        for k in ("row", "col", "target", "direction"):
            if k not in clue:
                errors.append(f"[{idx}] clue[{j}] missing key: {k}")
        if "row" in clue and (
            not isinstance(clue["row"], int) or not (0 <= clue["row"] < rows)
        ):
            errors.append(
                f"[{idx}] clue[{j}].row out of bounds: {clue.get('row')}, rows={rows}"
            )
        if "col" in clue and (
            not isinstance(clue["col"], int) or not (0 <= clue["col"] < cols)
        ):
            errors.append(
                f"[{idx}] clue[{j}].col out of bounds: {clue.get('col')}, cols={cols}"
            )
        if "direction" in clue and clue["direction"] not in ("A", "D"):
            errors.append(
                f"[{idx}] clue[{j}].direction must be 'A' or 'D', got {clue.get('direction')!r}"
            )
        if "target" in clue and not (
            isinstance(clue["target"], str) and len(clue["target"]) >= 1
        ):
            errors.append(
                f"[{idx}] clue[{j}].target must be a non-empty string, got {clue.get('target')!r}"
            )

        # Span bounds check if we can:
        if not errors and all(k in clue for k in ("row", "col", "target", "direction")):
            L = len(clue["target"])
            r0, c0 = clue["row"], clue["col"]
            if clue["direction"] == "A" and not (c0 + L - 1 < cols):
                errors.append(
                    f"[{idx}] clue[{j}] spans outside grid horizontally: start=({r0},{c0}), len={L}, cols={cols}"
                )
            if clue["direction"] == "D" and not (r0 + L - 1 < rows):
                errors.append(
                    f"[{idx}] clue[{j}] spans outside grid vertically: start=({r0},{c0}), len={L}, rows={rows}"
                )

        if len(errors) >= 20:
            break

    if errors:
        raise AssertionError(
            "Clue validation errors (showing up to 20):\n" + "\n".join(errors[:20])
        )


def _compute_char_accuracy(
    solved: list[list[Optional[str]]], gold: list[list[Optional[str]]]
) -> float:
    """
    Character accuracy over non-dot cells of the gold grid.
    A solved None never matches a gold letter.
    Dots '.' are skipped (black squares).
    """
    matches, total = 0, 0
    for r in range(len(gold)):
        for c in range(len(gold[0])):
            gold_ch = gold[r][c]
            if gold_ch == ".":  # skip black squares
                continue
            total += 1
            solved_ch = solved[r][c]
            gold_ch = _normalize(gold_ch)
            if solved_ch is not None:
                solved_ch = _normalize(solved_ch)
            matches += int(solved_ch == gold_ch)
    if total == 0:
        # Edge case: a grid with only dots; define accuracy as 1.0 (trivially correct)
        return 1.0
    return matches / total


def _extract_word(
    grid: list[list[Optional[str]]], row: int, col: int, length: int, direction: str
) -> str:
    chars = []
    for i in range(length):
        r = row + (i if direction == "D" else 0)
        c = col + (i if direction == "A" else 0)
        ch = grid[r][c]
        if ch is not None:
            chars.append(ch)
        # If None, we skip adding anything (equivalent to empty), so the word will be shorter;
        # this will not equal the target unless the target also had missing chars, which is not allowed.
    return "".join(chars)


def _compute_word_accuracy_for_grid(
    solved: list[list[Optional[str]]], clues: list[dict[str, Any]]
) -> float:
    correct = 0
    for clue in clues:
        r, c = clue["row"], clue["col"]
        L = len(clue["target"])
        d = clue["direction"]
        pred_word = _extract_word(solved, r, c, L, d)
        pred_word = _normalize(pred_word)
        gold_target = _normalize(clue["target"])
        correct += int(pred_word == gold_target)
    return correct / len(clues) if clues else 1.0


def score_crosswords(
    pred_grids: list[list[list[Optional[str]]]],
    gold_grids: list[list[list[Optional[str]]]],
    clues_list: list[list[dict[str, Any]]],
    verbose: bool = False,
) -> dict[str, float]:
    _validate_same_length(pred_grids, gold_grids, clues_list)

    char_accs: list[float] = []
    word_accs: list[float] = []

    for i, (solved, gold, clues) in enumerate(zip(pred_grids, gold_grids, clues_list)):
        # Allow None as an indicator of "no solution found"
        if solved is None:
            if verbose:
                print(f"[{i}] No solution found. char_acc=0.0, word_acc=0.0")
            char_accs.append(0.0)
            word_accs.append(0.0)
            continue

        rows, cols = _validate_pair_shapes(solved, gold, i)
        _validate_clues_for_grid(clues, rows, cols, i)

        char_acc = _compute_char_accuracy(solved, gold)
        word_acc = _compute_word_accuracy_for_grid(solved, clues)

        if verbose:
            print(f"[{i}] char_acc={char_acc:.4f}  word_acc={word_acc:.4f}")

        char_accs.append(char_acc)
        word_accs.append(word_acc)

    full_match_accuracy = (
        (sum(1 for a in char_accs if abs(a - 1.0) < 1e-12) / len(char_accs))
        if char_accs
        else 0.0
    )
    results = {
        "char_acc": (sum(char_accs) / len(char_accs)) if char_accs else 0.0,
        "word_acc": (sum(word_accs) / len(word_accs)) if word_accs else 0.0,
        "full_match_accuracy": full_match_accuracy,
        "num_examples": len(char_accs),
    }
    return results


# Additional direct word accuracy.
def direct_word_accuracy(doc: dict, parsed: ParsedOutput) -> float:
    """Parole autonome esatte / tutte le definizioni del cruciverba.

    Con una sola griglia, non esiste un elenco autonomo: coincide con la WA
    originale sulla griglia. Se è presente un elenco, le risposte mancanti,
    discordanti o errate valgono zero, anche se gli incroci le completerebbero.
    """
    if not parsed.has_answer_list:
        return score_crosswords([parsed.grid], [doc["gold_grid"]], [doc["clues"]])[
            "word_acc"
        ]
    entries = {
        (entry.row, entry.col, entry.direction): entry for entry in prepare(doc).entries
    }
    correct = 0
    for clue in doc["clues"]:
        entry = entries[(clue["row"], clue["col"], clue["direction"])]
        answer = parsed.answers.get(entry.answer_id)
        correct += int(
            answer is not None and _normalize(answer) == _normalize(clue["target"])
        )
    return correct / len(doc["clues"]) if doc["clues"] else 1.0


def score_parsed(doc: dict, parsed: ParsedOutput) -> dict[str, float]:
    result = score_crosswords([parsed.grid], [doc["gold_grid"]], [doc["clues"]])
    result.pop("num_examples")
    return {"word_acc_direct": direct_word_accuracy(doc, parsed), **result}


# Exact approved Task 2 prompt skeletons.
_TEMPLATES = [
    "Completa il seguente schema, rispondendo alle seguenti definizioni.\n\nSCHEMA\n\nLegenda: _ = casella bianca dove inserire una lettera della parola; # = casella nera dove non è possibile inserire nulla.\nAnche un numero rappresenta una casella bianca: indica la casella di partenza di una o più parole, le cui definizioni riportano quel numero. Il numero va dunque sostituito con la prima lettera della soluzione.\n\n{grid}\n\nDEFINIZIONI\n\nORIZZONTALI\n{horizontal_clues}\n\nVERTICALI\n{vertical_clues}\n\nSe non riesci a formattare lo schema, puoi restituire le singole risposte associate ai rispettivi numeri; in entrambi i casi, non aggiungere altro",
    "Devi risolvere un cruciverba.\nCompleta il seguente schema, rispondendo alle seguenti definizioni.\n\nSCHEMA\n\nLegenda: _ = casella bianca dove inserire una lettera della parola; # = casella nera dove non è possibile inserire nulla.\nAnche un numero rappresenta una casella bianca: indica la casella di partenza di una o più parole, le cui definizioni riportano quel numero. Il numero va dunque sostituito con la prima lettera della soluzione.\n\n{grid}\n\nDEFINIZIONI\n\nORIZZONTALI\n{horizontal_clues}\n\nVERTICALI\n{vertical_clues}\n\nSe non riesci a formattare lo schema, puoi restituire le singole risposte associate ai rispettivi numeri; in entrambi i casi, non aggiungere altro",
]


# Harness entry points.
def public_task2(doc):
    return {
        "empty_grid": doc["empty_grid"],
        "clues": [
            {key: clue[key] for key in ["clue", "row", "col", "direction", "length"]}
            for clue in doc["clues"]
        ],
    }


def _prompt(doc, index):
    puzzle = prepare(public_task2(doc))
    return _TEMPLATES[index].format(
        grid=render_grid(puzzle, numbered=True),
        horizontal_clues=render_clues(puzzle, "A", False),
        vertical_clues=render_clues(puzzle, "D", False),
    )


def cruciverb_task2_prompt1(doc):
    return _prompt(doc, 0)


def cruciverb_task2_prompt2(doc):
    return _prompt(doc, 1)


def parse_task1_candidates(value):
    return _parse_task1_candidates(_final_answer(value))


def parse_task2_output(value, doc):
    return parse_output(_final_answer(value), public_task2(doc))


def cruciverb_task1_doc_to_target(doc):
    return doc.get("fewshot_target") or doc["answer"]


def cruciverb_task1_process_results(doc, results):
    candidates = _parse_task1_candidates(_first_result(results))
    return score_task1_candidates(doc, candidates)


def cruciverb_task2_doc_to_target(doc):
    """Format a gold demonstration as numbered answers followed by its grid."""
    puzzle = prepare(public_task2(doc))
    gold = doc["gold_grid"]
    lines = []
    for direction, heading in [("A", "ORIZZONTALI"), ("D", "VERTICALI")]:
        if lines:
            lines.append("")
        lines.append(heading)
        for entry in puzzle.entries:
            if entry.direction == direction:
                answer = "".join(str(gold[r][c]) for r, c in entry.positions)
                lines.append(f"{entry.number}. {answer}")
    lines.extend(["", "SCHEMA COMPILATO"])
    lines.extend(
        " ".join("#" if cell == "." else str(cell) for cell in row) for row in gold
    )
    return "\n".join(lines)


def cruciverb_task2_process_results(doc, results):
    parsed = parse_output(_first_result(results), public_task2(doc))
    return score_parsed(doc, parsed)
