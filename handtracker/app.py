"""Пайплайн приложения: источник кадров -> детектор -> рендереры -> окно."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from .detector import HandDetector
from .processing import Processor
from .render import Renderer, put_text


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Следующий BGR-кадр или None, если поток закончился."""

    def release(self) -> None:
        pass


class CameraSource(FrameSource):
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру {index}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class VideoFileSource(CameraSource):
    """Видеофайл читается тем же VideoCapture."""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть файл {path}")


@dataclass
class KeyBinding:
    key: str
    description: str
    action: Callable[[], None]


class HandTrackerApp:
    QUIT_KEYS = (27,)  # ESC

    def __init__(
        self,
        source: FrameSource,
        detector: HandDetector,
        renderers: Sequence[Renderer],
        processors: Sequence[Processor] = (),
        *,
        mirror: bool = True,
        window: str = "Hand Tracker",
    ) -> None:
        self.source = source
        self.detector = detector
        self.renderers = list(renderers)
        self.processors = list(processors)
        self.mirror = mirror
        self.window = window
        self._bindings: dict[int, KeyBinding] = {}

    def toggle_mirror(self) -> None:
        self.mirror = not self.mirror

    def bind(self, key: str, description: str, action: Callable[[], None]) -> None:
        self._bindings[ord(key)] = KeyBinding(key, description, action)

    def _draw_hints(self, frame: np.ndarray) -> None:
        parts = ["esc: exit"] + [f"{b.key}: {b.description}" for b in self._bindings.values()]
        put_text(frame, " | ".join(parts), (10, frame.shape[0] - 12), scale=0.5)

    def run(self) -> None:
        start = time.monotonic()
        try:
            while True:
                frame = self.source.read()
                if frame is None:
                    break
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.monotonic()
                hands = self.detector.detect(frame, int((now - start) * 1000))
                for processor in self.processors:
                    processor.process(hands, now)
                for renderer in self.renderers:
                    renderer(frame, hands)
                self._draw_hints(frame)

                cv2.imshow(self.window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 0xFF:  # клавиша не нажата
                    continue
                if any(p.handle_key(key) for p in self.processors):
                    continue
                if key in self.QUIT_KEYS:
                    break
                binding = self._bindings.get(key)
                if binding:
                    binding.action()
        finally:
            for processor in self.processors:
                processor.close()
            self.source.release()
            self.detector.close()
            cv2.destroyAllWindows()
