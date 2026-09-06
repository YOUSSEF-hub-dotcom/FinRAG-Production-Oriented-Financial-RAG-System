"""
Incremental JSON extractor used for real SSE token streaming.

The generation LLM emits a structured JSON document (ConsolidatedFinancialAnswer)
token by token. A naive pass-through would leak internal fields
(``internal_thought``, ``extracted_raw_data``, ``sources``) to the user.

``StreamingAnswerExtractor`` walks the raw JSON characters as they arrive and
pulls out ONLY the decoded string value of the top-level ``answer`` key,
emitting it in whitespace-aligned fragments so the frontend can render a live
token stream without exposing internal JSON. The full raw document is still
accumulated for the final parse / audit / cache steps.
"""

from __future__ import annotations

from typing import List

_ANSWER_KEY = "answer"

# Simple JSON escape -> decoded char map (all two-character escapes).
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class StreamingAnswerExtractor:
    """Incrementally decode the top-level ``answer`` string from a JSON stream."""

    def __init__(self) -> None:
        self._raw: List[str] = []            # full raw JSON document
        self._answer_chars: List[str] = []   # fully decoded answer content
        self._last_emit: int = 0             # decode index of next un-emitted char

        self._depth = 0
        self._in_string = False
        self._is_key = False
        self._key_buf: List[str] = []
        self._last_key: str = ""             # most recent depth-1 object key
        self._value_pos = False              # next string is a value (after ':')
        self._in_answer = False              # currently decoding answer value
        self._answer_done = False            # answer value string finished
        self._root_closed = False            # outermost JSON document closed

        # escape decoding within the current string
        self._esc = 0                        # 0 none, 1 backslash, 2..=unicode hex
        self._uni: List[str] = []

    # -- public API -----------------------------------------------------------

    @property
    def raw_text(self) -> str:
        return "".join(self._raw)

    @property
    def answer(self) -> str:
        return "".join(self._answer_chars)

    @property
    def answer_found(self) -> bool:
        return self._answer_done or self._in_answer or bool(self._answer_chars)

    def feed(self, chunk: str) -> str:
        """Consume a raw chunk; return newly-available decoded answer text.

        Returned fragments are broken on whitespace boundaries so the frontend's
        space-aware join does not insert artifacts mid-word. The final
        ``flush()`` releases any trailing text.
        """
        for ch in chunk:
            self._raw.append(ch)
            self._consume(ch)
        return self._release_ready()

    def flush(self) -> str:
        """Emit any trailing answer characters not yet released (stream end)."""
        remaining = "".join(self._answer_chars[self._last_emit:])
        self._last_emit = len(self._answer_chars)
        return remaining

    # -- private core ---------------------------------------------------------

    def _consume(self, ch: str) -> None:
        if self._in_string:
            self._consume_in_string(ch)
            return

        if ch == '"':
            self._open_string()
        elif ch in "[{":
            self._depth += 1
            if ch == "[":
                # inside an array, a subsequent string is a value
                self._value_pos = True
            else:
                # inside an object, a subsequent string is a key
                self._value_pos = False
        elif ch in "]}":
            self._depth = max(0, self._depth - 1)
            if self._depth <= 0:
                self._root_closed = True
        elif ch == ":":
            self._value_pos = True
        elif ch == ",":
            if self._depth == 1:
                self._value_pos = False
        # structural whitespace needs no handling outside strings

    def _open_string(self) -> None:
        self._in_string = True
        self._esc = 0
        # Only depth-1 strings participate in key/answer bookkeeping.
        if self._depth != 1:
            self._is_key = False
            self._in_answer = False
            return
        if self._value_pos:
            # value string: is it the answer field?
            self._is_key = False
            self._value_pos = False
            self._in_answer = self._last_key == _ANSWER_KEY
        else:
            # object member key at depth 1
            self._is_key = True
            self._key_buf = []
            self._in_answer = False

    def _consume_in_string(self, ch: str) -> None:
        if self._esc == 0:
            if ch == "\\":
                self._esc = 1
                return
            if ch == '"':
                self._close_string()
                return
            self._emit(ch)
            return
        if self._esc == 1:
            if ch == "u":
                self._esc = 2
                self._uni = []
                return
            self._esc = 0
            self._emit(_SIMPLE_ESCAPES.get(ch, ch))
            return
        # unicode escape: up to 4 hex digits
        self._uni.append(ch)
        if len(self._uni) == 4:
            self._esc = 0
            try:
                self._emit(chr(int("".join(self._uni), 16)))
            except ValueError:
                self._emit("\ufffd")

    def _close_string(self) -> None:
        self._in_string = False
        self._esc = 0
        if self._depth == 1 and self._is_key:
            self._last_key = "".join(self._key_buf)
            self._is_key = False
        elif self._in_answer:
            self._in_answer = False
            self._answer_done = True

    def _emit(self, ch: str) -> None:
        if self._depth != 1:
            return
        if self._is_key:
            self._key_buf.append(ch)
        elif self._in_answer:
            self._answer_chars.append(ch)

    def _release_ready(self) -> str:
        """Return answer chars since the last release, breaking on whitespace."""
        pool = self._answer_chars
        if self._last_emit >= len(pool):
            return ""
        # If the document or the answer field is finished, release everything.
        if self._answer_done or self._root_closed:
            seg = "".join(pool[self._last_emit:])
            self._last_emit = len(pool)
            return seg
        # Release only complete words (ending at a whitespace char) to keep the
        # frontend's space-aware join artifact-free.
        end = None
        for i in range(len(pool) - 1, self._last_emit - 1, -1):
            if pool[i].isspace():
                end = i
                break
        if end is None:
            return ""
        seg = "".join(pool[self._last_emit : end + 1])
        self._last_emit = end + 1
        return seg
