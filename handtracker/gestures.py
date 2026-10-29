"""Распознавание жестов: поза пальцев -> текстовая подпись -> совпадение с паттерном.

Подпись одной руки:  Right:01000
    сторона, затем 5 бит в порядке: большой, указательный, средний, безымянный, мизинец.
    1 - палец разогнут, 0 - согнут.

Паттерн (то, что пишется в конфиге):
    Right:01000                точное совпадение
    *:01???                    любая рука; ? - "не важно"
    Left:11111 Right:01000     обе руки одновременно (части через пробел)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .models import FINGER_CHAINS, Finger, Hand, Landmark

FINGER_ORDER = (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

_PART_RE = re.compile(r"^(Left|Right|\*):[01?]{5}$", re.IGNORECASE)


@dataclass
class FingerThresholds:
    """Пороги определения "палец разогнут". Подстройте под свою камеру и руку."""

    # Четыре пальца: |кончик - запястье| / |PIP - запястье| больше порога -> разогнут.
    extend_ratio: float = 1.1
    # Большой: кончик дальше от основания мизинца, чем его IP-сустав, в такое число раз.
    thumb_tip_ratio: float = 1.05
    # Большой: |кончик - основание указательного| / размер ладони должно быть больше.
    thumb_spread: float = 0.6


class GestureClassifier(ABC):
    """Превращает руку в текстовый токен. Подмените, чтобы распознавать по-другому."""

    @abstractmethod
    def token(self, hand: Hand) -> str: ...

    def tokens(self, hands: Sequence[Hand]) -> list[str]:
        return sorted(self.token(hand) for hand in hands)


class FingerStateClassifier(GestureClassifier):
    """Разогнут/согнут для каждого пальца. Не зависит от поворота руки в кадре."""

    def __init__(self, thresholds: FingerThresholds | None = None) -> None:
        self.thresholds = thresholds or FingerThresholds()

    def finger_states(self, hand: Hand) -> tuple[bool, ...]:
        t = self.thresholds
        palm = max(hand.palm_size, 1e-6)

        thumb_out = (
            hand.distance(Landmark.THUMB_TIP, Landmark.PINKY_MCP)
            > hand.distance(Landmark.THUMB_IP, Landmark.PINKY_MCP) * t.thumb_tip_ratio
            and hand.distance(Landmark.THUMB_TIP, Landmark.INDEX_MCP) / palm > t.thumb_spread
        )
        states = [thumb_out]
        for finger in FINGER_ORDER[1:]:
            _, pip, _, tip = FINGER_CHAINS[finger]
            states.append(
                hand.distance(tip, Landmark.WRIST)
                > hand.distance(pip, Landmark.WRIST) * t.extend_ratio
            )
        return tuple(states)

    def token(self, hand: Hand) -> str:
        bits = "".join("1" if s else "0" for s in self.finger_states(hand))
        return f"{hand.side.value}:{bits}"


def describe_token(token: str) -> str:
    """Человекочитаемое описание пальцев: 'index, middle', 'fist', 'open palm'."""
    _, _, bits = token.partition(":")
    up = [name for name, bit in zip(FINGER_NAMES, bits) if bit == "1"]
    if not up:
        return "fist"
    if len(up) == len(FINGER_NAMES):
        return "open palm"
    return ", ".join(up)


def validate_gesture(gesture: str) -> str:
    """Проверяет паттерн и приводит к каноничному виду. Бросает ValueError."""
    parts = gesture.split()
    if not parts:
        raise ValueError("Пустой жест")
    normalized = []
    for part in parts:
        if not _PART_RE.match(part):
            raise ValueError(
                f"Неверный жест {part!r}: ожидается Left|Right|*:ххххх, где х - 0, 1 или ?"
            )
        side, bits = part.split(":")
        normalized.append(f"{'*' if side == '*' else side.capitalize()}:{bits}")
    return " ".join(normalized)


def token_matches(pattern: str, token: str) -> bool:
    p_side, _, p_bits = pattern.partition(":")
    t_side, _, t_bits = token.partition(":")
    if p_side != "*" and p_side.lower() != t_side.lower():
        return False
    return len(p_bits) == len(t_bits) and all(p in ("?", t) for p, t in zip(p_bits, t_bits))


def match_hands(gesture: str, tokens: Sequence[str]) -> list[int] | None:
    """Индексы токенов (рук), которые сработали на жест, или None, если жест не выполнен.

    Каждая часть паттерна должна совпасть хотя бы с одной рукой; лишние руки игнорируются.
    """
    parts = gesture.split()
    if not parts:
        return None
    hit: set[int] = set()
    for part in parts:
        indices = [i for i, tok in enumerate(tokens) if token_matches(part, tok)]
        if not indices:
            return None
        hit.update(indices)
    return sorted(hit)


def gesture_matches(gesture: str, tokens: Sequence[str]) -> bool:
    return match_hands(gesture, tokens) is not None
