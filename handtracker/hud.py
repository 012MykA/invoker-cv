"""Текстовая панель с жестами и статусом (цветом рук занимаются Skeleton/Landmark рендереры)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .gestures import describe_token
from .keys import LogKeyEmitter
from .models import Hand
from .processing import GestureEngine, GestureRecorder
from .render import Renderer, put_text


class GestureHudRenderer(Renderer):
    """Панель слева: статус эмуляции, подписи жестов рук, активные бинды, статус записи."""

    def __init__(self, engine: GestureEngine, recorder: GestureRecorder) -> None:
        self.engine = engine
        self.recorder = recorder

    def render(self, frame: np.ndarray, hands: Sequence[Hand]) -> None:
        y = 58
        dry = " (dry-run)" if isinstance(self.engine.emitter, LogKeyEmitter) else ""
        on = self.engine.enabled
        put_text(frame, f"KEYS: {'ON' if on else 'OFF'}{dry}", (10, y),
                 (0, 255, 0) if on else (0, 0, 255), 0.6, 2)
        y += 24

        for token in self.engine.tokens:  # именно эту строку копируйте в конфиг
            put_text(frame, f"{token}  [{describe_token(token)}]", (10, y), (255, 255, 255), 0.55)
            y += 22

        for binding in self.engine.active_bindings:
            color = binding.bgr or (0, 255, 0)
            put_text(frame, f"> {binding.name or binding.gesture}: {binding.key} [{binding.mode}]",
                     (10, y), color, 0.55, 2)
            y += 22

        for line in self.recorder.status_lines:
            put_text(frame, line, (10, y), (0, 255, 255), 0.6, 2)
            y += 24
