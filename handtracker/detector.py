"""Детекторы рук. Приложение зависит только от абстракции HandDetector."""

from __future__ import annotations

import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .models import Hand, Side

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hand_landmarker.task"
)


class HandDetector(ABC):
    """Интерфейс детектора: BGR-кадр на входе, список Hand на выходе."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]: ...

    def close(self) -> None:
        pass

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class MediaPipeConfig:
    num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_path: str = DEFAULT_MODEL_PATH
    model_url: str = MODEL_URL
    # MediaPipe считает, что кадр зеркальный (селфи), и по-своему называет Left/Right.
    # Если подписи перепутаны - переключите флаг (в приложении клавиша "h").
    swap_handedness: bool = True


def _ensure_model(path: str, url: str) -> None:
    if not os.path.exists(path):
        print(f"Скачиваю модель в {path} ...")
        urllib.request.urlretrieve(url, path)


class MediaPipeHandDetector(HandDetector):
    def __init__(self, config: MediaPipeConfig | None = None) -> None:
        self.config = config or MediaPipeConfig()
        _ensure_model(self.config.model_path, self.config.model_url)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.config.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def toggle_swap(self) -> None:
        self.config.swap_handedness = not self.config.swap_handedness

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        # MediaPipe требует строго возрастающие таймстемпы.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        frame_h, frame_w = frame_bgr.shape[:2]
        aspect = frame_w / frame_h
        hands: list[Hand] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            category = handedness[0]
            side = Side(category.category_name)
            if self.config.swap_handedness:
                side = side.opposite
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            hands.append(Hand(side=side, score=category.score, landmarks=points, aspect=aspect))
        return hands

    def close(self) -> None:
        self._landmarker.close()
