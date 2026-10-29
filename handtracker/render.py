"""Рендереры: каждый рисует одну "слойку" поверх кадра.

Чтобы добавить свою визуализацию - наследуйтесь от Renderer и реализуйте render().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import cv2
import numpy as np

from .models import FINGER_OF, FINGERTIPS, HAND_BONES, Finger, Hand, Landmark, Side

Color = tuple[int, int, int]  # BGR
# Функция "рука -> цвет всей кисти (BGR) или None": так жест перекрашивает руку целиком.
ColorOverride = Callable[[Hand], Color | None]

PALM_COLOR: Color = (200, 200, 200)
FINGER_COLORS: dict[Finger, Color] = {
    Finger.THUMB: (0, 165, 255),
    Finger.INDEX: (0, 255, 0),
    Finger.MIDDLE: (255, 200, 0),
    Finger.RING: (255, 0, 200),
    Finger.PINKY: (0, 0, 255),
}
SIDE_COLORS: dict[Side, Color] = {
    Side.LEFT: (255, 180, 0),
    Side.RIGHT: (0, 220, 255),
}


def put_text(frame: np.ndarray, text: str, org: tuple[int, int], color: Color = (255, 255, 255),
             scale: float = 0.5, thickness: int = 1) -> None:
    """Текст с тёмной обводкой, читается на любом фоне."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


class Renderer(ABC):
    enabled: bool = True

    def __call__(self, frame: np.ndarray, hands: Sequence[Hand]) -> None:
        if self.enabled:
            self.render(frame, hands)

    def toggle(self) -> None:
        self.enabled = not self.enabled

    @abstractmethod
    def render(self, frame: np.ndarray, hands: Sequence[Hand]) -> None: ...


class SkeletonRenderer(Renderer):
    """Все кости руки."""

    def __init__(self, thickness: int = 3, override: ColorOverride | None = None) -> None:
        self.thickness = thickness
        self.override = override

    def render(self, frame, hands):
        h, w = frame.shape[:2]
        for hand in hands:
            pts = hand.to_pixels(w, h).tolist()
            forced = self.override(hand) if self.override else None
            for bone in HAND_BONES:
                color = forced or FINGER_COLORS.get(bone.finger, PALM_COLOR)
                cv2.line(frame, tuple(pts[bone.start]), tuple(pts[bone.end]),
                         color, self.thickness, cv2.LINE_AA)


class LandmarkRenderer(Renderer):
    """Все 21 точка, опционально с номерами."""

    def __init__(self, show_ids: bool = True, override: ColorOverride | None = None) -> None:
        self.show_ids = show_ids
        self.override = override

    def toggle_ids(self) -> None:
        self.show_ids = not self.show_ids

    def render(self, frame, hands):
        h, w = frame.shape[:2]
        for hand in hands:
            pts = hand.to_pixels(w, h).tolist()
            forced = self.override(hand) if self.override else None
            rim = (0, 0, 0) if forced else (255, 255, 255)  # при перекраске белая кайма не нужна
            for lm in Landmark:
                x, y = pts[lm]
                color = forced or FINGER_COLORS.get(FINGER_OF.get(lm), PALM_COLOR)
                radius = 8 if lm in FINGERTIPS else 6
                cv2.circle(frame, (x, y), radius, rim, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius - 2, color, -1, cv2.LINE_AA)
                if self.show_ids:
                    put_text(frame, str(int(lm)), (x + 8, y - 8), scale=0.45)


class HandLabelRenderer(Renderer):
    """Подпись Left/Right и уверенность у запястья."""

    def render(self, frame, hands):
        h, w = frame.shape[:2]
        for hand in hands:
            x, y = hand.to_pixels(w, h)[Landmark.WRIST].tolist()
            put_text(frame, f"{hand.side.value} {hand.score:.2f}", (x - 30, y + 30),
                     SIDE_COLORS[hand.side], scale=0.8, thickness=2)


class PinchRenderer(Renderer):
    """Пример расширения: подсвечивает щипок (большой + указательный)."""

    def __init__(self, threshold: float = 0.25) -> None:
        self.threshold = threshold  # доля от размера ладони

    def render(self, frame, hands):
        h, w = frame.shape[:2]
        for hand in hands:
            ratio = hand.distance(Landmark.THUMB_TIP, Landmark.INDEX_TIP) / max(hand.palm_size, 1e-6)
            if ratio < self.threshold:
                pts = hand.to_pixels(w, h).tolist()
                a, b = tuple(pts[Landmark.THUMB_TIP]), tuple(pts[Landmark.INDEX_TIP])
                cv2.line(frame, a, b, (0, 255, 255), 5, cv2.LINE_AA)
                put_text(frame, "PINCH", (a[0] + 12, a[1] + 24), (0, 255, 255), 0.8, 2)


class FpsRenderer(Renderer):
    def __init__(self) -> None:
        self._prev = time.perf_counter()

    def render(self, frame, hands):
        now = time.perf_counter()
        fps = 1.0 / max(now - self._prev, 1e-6)
        self._prev = now
        put_text(frame, f"FPS: {fps:.0f}  Hands: {len(hands)}", (10, 30),
                 (0, 255, 0), scale=0.8, thickness=2)
