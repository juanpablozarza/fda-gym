"""Draw a training PNG from W&B history rows. Fetching the API is optional.

The trainer logs under the ``train/`` prefix (Hugging Face ``rewrite_logs``).
Reward is the shaped training reward. Strict is ``rewards/env/mean``.
Memorization gap is known minus unseen. Flags are 0 ok, 1 warn, 2 stop and review.
"""

from __future__ import annotations

import itertools
import struct
import zlib
from pathlib import Path

# 5x7 glyphs, leftmost bit is bit 4. Missing characters are skipped.
_GLYPHS = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "-": (0, 0, 0, 0b11111, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 0b01100, 0b01100),
    "/": (0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0),
    ":": (0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
}

_REWARD_KEYS = ("train/reward", "reward")
_STRICT_KEYS = ("train/rewards/env/mean", "rewards/env/mean", "eval/strict")
_GAP_KEYS = ("train/memorization/gap/strict", "memorization/gap/strict")
_FLAG_KEYS = {
    "gap": ("train/memorization/flags/gap", "memorization/flags/gap"),
    "no look": ("train/memorization/flags/answered_without_looking", "memorization/flags/answered_without_looking"),
    "swap": ("train/memorization/flags/swapped_facts", "memorization/flags/swapped_facts"),
    "level": ("train/memorization/flags/level", "memorization/flags/level"),
}
_STEP_KEYS = ("train/global_step", "_step")

_BG = (248, 246, 242)
_INK = (32, 32, 32)
_GRID = (220, 214, 204)
_AMBER = (196, 122, 12)
_BLUE = (30, 90, 150)
_RED = (168, 56, 48)
_FLAG_COLORS = {
    "gap": (168, 56, 48),
    "no look": (30, 90, 150),
    "swap": (90, 120, 50),
    "level": (120, 70, 150),
}


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _step_of(row: dict) -> int | None:
    for key in _STEP_KEYS:
        value = _number(row.get(key))
        if value is not None:
            return int(value)
    return None


def _first_number(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _collapse(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Keep the last value logged at each step so a resumed run does not draw twice."""
    ordered: dict[int, float] = {}
    for step, value in points:
        ordered[step] = value
    return sorted(ordered.items())


def chart_series(rows: list[dict]) -> dict:
    """Pull reward, strict, memorization gap, and flags out of W&B history rows."""
    reward: list[tuple[int, float]] = []
    strict: list[tuple[int, float]] = []
    gap: list[tuple[int, float]] = []
    flags = {name: [] for name in _FLAG_KEYS}
    for row in rows:
        if not isinstance(row, dict):
            continue
        step = _step_of(row)
        if step is None:
            continue
        value = _first_number(row, _REWARD_KEYS)
        if value is not None:
            reward.append((step, value))
        value = _first_number(row, _STRICT_KEYS)
        if value is not None:
            strict.append((step, value))
        value = _first_number(row, _GAP_KEYS)
        if value is not None:
            gap.append((step, value))
        for name, keys in _FLAG_KEYS.items():
            value = _first_number(row, keys)
            if value is not None:
                flags[name].append((step, value))
    return {
        "reward": _collapse(reward),
        "strict": _collapse(strict),
        "gap": _collapse(gap),
        "flags": {name: _collapse(points) for name, points in flags.items()},
    }


def _png(width: int, height: int, rgb: bytearray) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride : (y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    )


class _Canvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buf = bytearray(_BG * (width * height))

    def _set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.buf[i : i + 3] = bytes(color)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self._set(xx, yy, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self._set(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * err
            if doubled > -dy:
                err -= dy
                x0 += sx
            if doubled < dx:
                err += dx
                y0 += sy

    def text(self, x: int, y: int, message: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for char in message.upper():
            glyph = _GLYPHS.get(char)
            if glyph is None:
                cursor += 4 * scale
                continue
            for row, bits in enumerate(glyph):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        self.rect(cursor + col * scale, y + row * scale, scale, scale, color)
            cursor += 6 * scale


def _map_points(points: list[tuple[int, float]], box: tuple[int, int, int, int], y_min: float, y_max: float):
    x, y, w, h = box
    if not points:
        return []
    steps = [step for step, _value in points]
    lo, hi = min(steps), max(steps)
    span = hi - lo or 1
    y_span = y_max - y_min or 1.0
    mapped = []
    for step, value in points:
        px = x + round((step - lo) / span * (w - 1))
        py = y + h - 1 - round((value - y_min) / y_span * (h - 1))
        mapped.append((px, py))
    return mapped


def _panel(canvas: _Canvas, box: tuple[int, int, int, int], title: str, series: list, color, y_fixed=None) -> None:
    x, y, w, h = box
    canvas.rect(x, y, w, h, (255, 255, 255))
    canvas.text(x + 8, y + 8, title, _INK, scale=2)
    plot = (x + 36, y + 36, w - 48, h - 56)
    px, py, pw, ph = plot
    canvas.line(px, py, px, py + ph, _INK)
    canvas.line(px, py + ph, px + pw, py + ph, _INK)
    if not series:
        canvas.text(px + 8, py + ph // 2, "NO POINTS", _GRID, scale=2)
        return
    values = [value for _step, value in series]
    if y_fixed is None:
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.08 or 0.05
        y_min, y_max = lo - pad, hi + pad
    else:
        y_min, y_max = y_fixed
    for tick, label in ((y_min, f"{y_min:.2f}"), (y_max, f"{y_max:.2f}")):
        yy = py + ph - 1 - round((tick - y_min) / (y_max - y_min or 1) * (ph - 1))
        canvas.line(px, yy, px + pw, yy, _GRID)
        canvas.text(x + 2, max(y + 28, yy - 6), label[:5], _INK, scale=1)
    mapped = _map_points(series, plot, y_min, y_max)
    for start, end in itertools.pairwise(mapped):
        canvas.line(start[0], start[1], end[0], end[1], color)
    for point in mapped:
        canvas.rect(point[0] - 1, point[1] - 1, 3, 3, color)
    canvas.text(px, py + ph + 6, f"STEP {series[0][0]}-{series[-1][0]}", _INK, scale=1)


def render_chart(series: dict, path: Path, *, title: str) -> None:
    """Write one PNG with reward, strict, memorization gap, and flags."""
    width, height = 1100, 760
    canvas = _Canvas(width, height)
    canvas.text(24, 16, title[:42], _INK, scale=3)
    canvas.text(24, 48, "0 OK   1 WARN   2 STOP AND REVIEW", _INK, scale=1)
    panels = [
        (24, 78, 520, 310, "REWARD", series.get("reward") or [], _AMBER, None),
        (560, 78, 516, 310, "STRICT", series.get("strict") or [], _BLUE, None),
        (24, 408, 520, 320, "MEM GAP", series.get("gap") or [], _RED, None),
    ]
    for left, top, wide, tall, label, points, color, fixed in panels:
        _panel(canvas, (left, top, wide, tall), label, points, color, fixed)
    flag_box = (560, 408, 516, 320)
    canvas.rect(*flag_box, (255, 255, 255))
    canvas.text(flag_box[0] + 8, flag_box[1] + 8, "FLAGS", _INK, scale=2)
    flags = series.get("flags") or {}
    any_flags = any(flags.get(name) for name in _FLAG_KEYS)
    plot = (flag_box[0] + 36, flag_box[1] + 36, flag_box[2] - 48, flag_box[3] - 78)
    canvas.line(plot[0], plot[1], plot[0], plot[1] + plot[3], _INK)
    canvas.line(plot[0], plot[1] + plot[3], plot[0] + plot[2], plot[1] + plot[3], _INK)
    if not any_flags:
        canvas.text(plot[0] + 8, plot[1] + plot[3] // 2, "NO POINTS", _GRID, scale=2)
    else:
        for name, color in _FLAG_COLORS.items():
            points = flags.get(name) or []
            mapped = _map_points(points, plot, 0.0, 2.0)
            for start, end in itertools.pairwise(mapped):
                canvas.line(*start, *end, color)
            for point in mapped:
                canvas.rect(point[0] - 1, point[1] - 1, 3, 3, color)
        legend_x = flag_box[0] + 8
        for index, (name, color) in enumerate(_FLAG_COLORS.items()):
            canvas.text(legend_x + index * 120, flag_box[1] + flag_box[3] - 28, name, color, scale=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png(width, height, canvas.buf))


def fetch_history(run_id: str, *, project: str, entity: str | None) -> list[dict]:
    """Read one run's history. The key stays in the environment; it is not printed."""
    import wandb

    api = wandb.Api()
    owner = entity or api.default_entity
    if not owner:
        raise RuntimeError("W&B entity is unset. Pass --entity or set WANDB_ENTITY.")
    run = api.run(f"{owner}/{project}/{run_id}")
    keys = [
        *_STEP_KEYS,
        *_REWARD_KEYS,
        *_STRICT_KEYS,
        *_GAP_KEYS,
        *(key for pair in _FLAG_KEYS.values() for key in pair),
    ]
    return [dict(row) for row in run.scan_history(keys=keys)]
