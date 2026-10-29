"""Доменные модели: никакого OpenCV/MediaPipe, только данные и топология руки."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import NamedTuple

import numpy as np


class Side(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"

    @property
    def opposite(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


class Finger(Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"


class Landmark(IntEnum):
    """21 ключевая точка руки (индексы совпадают с MediaPipe)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Цепочки точек каждого пальца: от основания к кончику.
FINGER_CHAINS: dict[Finger, tuple[Landmark, ...]] = {
    Finger.THUMB: (Landmark.THUMB_CMC, Landmark.THUMB_MCP, Landmark.THUMB_IP, Landmark.THUMB_TIP),
    Finger.INDEX: (Landmark.INDEX_MCP, Landmark.INDEX_PIP, Landmark.INDEX_DIP, Landmark.INDEX_TIP),
    Finger.MIDDLE: (Landmark.MIDDLE_MCP, Landmark.MIDDLE_PIP, Landmark.MIDDLE_DIP, Landmark.MIDDLE_TIP),
    Finger.RING: (Landmark.RING_MCP, Landmark.RING_PIP, Landmark.RING_DIP, Landmark.RING_TIP),
    Finger.PINKY: (Landmark.PINKY_MCP, Landmark.PINKY_PIP, Landmark.PINKY_DIP, Landmark.PINKY_TIP),
}

FINGER_OF: dict[Landmark, Finger] = {
    lm: finger for finger, chain in FINGER_CHAINS.items() for lm in chain
}

FINGERTIPS: tuple[Landmark, ...] = tuple(chain[-1] for chain in FINGER_CHAINS.values())


class Bone(NamedTuple):
    start: Landmark
    end: Landmark
    finger: Finger | None  # None - кости ладони


def _build_bones() -> tuple[Bone, ...]:
    bones: list[Bone] = []
    for finger, chain in FINGER_CHAINS.items():
        bones += [Bone(a, b, finger) for a, b in zip(chain, chain[1:])]
    for finger in (Finger.THUMB, Finger.INDEX, Finger.PINKY):
        bones.append(Bone(Landmark.WRIST, FINGER_CHAINS[finger][0], finger))
    knuckles = [FINGER_CHAINS[f][0] for f in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)]
    bones += [Bone(a, b, None) for a, b in zip(knuckles, knuckles[1:])]
    return tuple(bones)


HAND_BONES: tuple[Bone, ...] = _build_bones()


@dataclass(frozen=True, eq=False)
class Hand:
    """Одна обнаруженная рука.

    landmarks: массив (21, 3) с нормализованными x, y (0..1 относительно кадра) и z.
    """

    side: Side
    score: float
    landmarks: np.ndarray
    aspect: float = 1.0  # ширина/высота кадра: нужно, чтобы расстояния не искажались

    def __getitem__(self, lm: Landmark) -> np.ndarray:
        return self.landmarks[int(lm)]

    def to_pixels(self, width: int, height: int) -> np.ndarray:
        """Координаты точек в пикселях, массив (21, 2) типа int."""
        return (self.landmarks[:, :2] * (width, height)).astype(int)

    def distance(self, a: Landmark, b: Landmark) -> float:
        """Расстояние между точками (по x, y) с поправкой на пропорции кадра."""
        dx, dy = self[a][:2] - self[b][:2]
        return float(np.hypot(dx * self.aspect, dy))

    @property
    def palm_size(self) -> float:
        """Масштаб руки: запястье - основание среднего пальца. Удобно для нормализации."""
        return self.distance(Landmark.WRIST, Landmark.MIDDLE_MCP)
