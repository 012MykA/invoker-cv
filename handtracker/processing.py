"""Обработчики результата детекции: движок биндов и запись жеста с камеры.

Процессор получает список Hand каждый кадр. Чтобы добавить своё поведение
(логирование, OSC, мышь) - наследуйтесь от Processor.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

from .config import Binding, Config
from .gestures import GestureClassifier, match_hands, validate_gesture
from .keys import CANCELLED, KeyCapture, KeyEmitter, key_name_from_cv2
from .models import Hand

ESC = 27

# Цвета (RGB hex), которые автоматически получают записанные жесты.
AUTO_COLORS = ("#ff8800", "#00c8ff", "#a020f0", "#ff3366", "#33dd66", "#ffd400")

Bgr = tuple[int, int, int]


class Processor(ABC):
    @abstractmethod
    def process(self, hands: Sequence[Hand], now: float) -> None: ...

    def handle_key(self, key: int) -> bool:
        """Вернуть True, если клавиша "съедена" и приложение не должно её обрабатывать."""
        return False

    def close(self) -> None:
        pass


@dataclass
class _BindingState:
    binding: Binding
    on_frames: int = 0
    off_frames: int = 0
    active: bool = False
    held: bool = False
    last_fire: float = -math.inf
    next_repeat: float = 0.0
    matched_hands: list[Hand] = field(default_factory=list)


class GestureEngine(Processor):
    """Сопоставляет жесты с биндами и эмулирует клавиши (с дебаунсом).

    Режимы бинда:
        press    - нажать клавишу в момент, когда жест появился
        release  - нажать клавишу в момент, когда жест пропал
        hold     - удерживать клавишу, пока показан жест
        repeat   - нажимать с интервалом repeat_ms, пока показан жест
    """

    def __init__(self, config: Config, classifier: GestureClassifier, emitter: KeyEmitter) -> None:
        self.config = config
        self.classifier = classifier
        self.emitter = emitter
        self.enabled = config.settings.enabled_on_start
        self.tokens: list[str] = []
        self._states: list[_BindingState] = []
        self._colors: dict[int, Bgr] = {}
        for binding in config.bindings:
            self._register(binding)

    def _register(self, binding: Binding) -> None:
        binding.gesture = validate_gesture(binding.gesture)
        self.emitter.validate(binding.key)
        self._states.append(_BindingState(binding))

    # --- публичный API ---------------------------------------------------------------

    @property
    def active_bindings(self) -> list[Binding]:
        return [s.binding for s in self._states if s.active]

    def color_of(self, hand: Hand) -> Bgr | None:
        """Цвет, в который надо красить эту руку (BGR), если на ней сейчас показан жест с цветом."""
        return self._colors.get(id(hand))

    def add_binding(self, binding: Binding) -> None:
        self._register(binding)
        self.config.bindings.append(binding)
        self.config.save()

    def toggle(self) -> None:
        self.set_enabled(not self.enabled)

    def set_enabled(self, value: bool) -> None:
        self.enabled = value
        if not value:
            self._release_all()

    # --- Processor -------------------------------------------------------------------

    def process(self, hands: Sequence[Hand], now: float) -> None:
        settings = self.config.settings
        pairs = [(hand, self.classifier.token(hand)) for hand in hands]
        flat = [token for _, token in pairs]
        self.tokens = sorted(flat)
        self._colors = {}

        for st in self._states:
            indices = match_hands(st.binding.gesture, flat)
            st.matched_hands = [pairs[i][0] for i in indices] if indices is not None else []

            if indices is not None:
                st.on_frames += 1
                st.off_frames = 0
                color = st.binding.bgr
                if color is not None:  # при нескольких совпадениях побеждает последний бинд
                    for hand in st.matched_hands:
                        self._colors[id(hand)] = color
            else:
                st.off_frames += 1
                st.on_frames = 0

            if not st.active:
                if st.on_frames >= settings.stable_frames:
                    st.active = True
                    self._activate(st, now)
            elif st.off_frames >= settings.release_frames:
                st.active = False
                self._deactivate(st)
            elif (
                st.binding.mode == "repeat"
                and st.off_frames == 0
                and now >= st.next_repeat
            ):
                self._repeat(st, now)

    def close(self) -> None:
        self._release_all()
        self.emitter.close()

    # --- внутреннее ------------------------------------------------------------------

    def _activate(self, st: _BindingState, now: float) -> None:
        if not self.enabled:
            return
        b = st.binding
        if b.mode == "hold":
            self.emitter.press(b.key)
            st.held = True
        elif b.mode == "repeat":
            self._repeat(st, now)
        elif b.mode == "press":
            if now - st.last_fire >= self.config.settings.cooldown_ms / 1000:
                self.emitter.tap(b.key)
                st.last_fire = now

    def _repeat(self, st: _BindingState, now: float) -> None:
        if self.enabled:
            self.emitter.tap(st.binding.key)
        st.next_repeat = now + st.binding.repeat_ms / 1000

    def _deactivate(self, st: _BindingState) -> None:
        self._release_held(st)
        if st.binding.mode == "release" and self.enabled:
            self.emitter.tap(st.binding.key)

    def _release_held(self, st: _BindingState) -> None:
        if st.held:
            self.emitter.release(st.binding.key)
            st.held = False

    def _release_all(self) -> None:
        for st in self._states:
            self._release_held(st)


class _Phase(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    CAPTURING = auto()
    WAIT_KEY = auto()


class GestureRecorder(Processor):
    """Запись жеста с камеры: b -> пауза -> запись -> нажать клавишу/комбинацию для бинда.

    С capture (pynput) ловится любая клавиша и комбинация (ctrl+s) в любом окне.
    Без него - только одиночные клавиши, нажатые в окне OpenCV.
    """

    def __init__(
        self,
        config: Config,
        classifier: GestureClassifier,
        engine: GestureEngine,
        capture: KeyCapture | None = None,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.engine = engine
        self.capture = capture
        self.phase = _Phase.IDLE
        self.captured: str | None = None
        self._t0: float | None = None
        self._now = 0.0
        self._samples: list[str] = []
        self._message = ""
        self._message_until = 0.0

    # --- публичный API ---------------------------------------------------------------

    def start(self) -> None:
        if self.phase is _Phase.IDLE:
            self.phase = _Phase.COUNTDOWN
            self._t0 = None
            self._message = ""

    @property
    def status_lines(self) -> list[str]:
        """Текст для HUD (только ASCII: cv2.putText не умеет кириллицу)."""
        lines: list[str] = []
        settings = self.config.settings
        if self.phase is _Phase.COUNTDOWN:
            elapsed = 0.0 if self._t0 is None else self._now - self._t0
            lines.append(f"BIND: show the gesture... {max(0.0, settings.capture_delay_s - elapsed):.1f}s (ESC cancel)")
        elif self.phase is _Phase.CAPTURING:
            lines.append("BIND: hold still, recording... (ESC cancel)")
        elif self.phase is _Phase.WAIT_KEY:
            lines.append(f"BIND: captured {self.captured}")
            if self.capture is not None:
                lines.append("Press any key or combo to bind (ESC cancel)")
            else:
                lines.append("Press a key in this window (ESC cancel)")
        if self._message and self._now < self._message_until:
            lines.append(self._message)
        return lines

    # --- Processor -------------------------------------------------------------------

    def process(self, hands: Sequence[Hand], now: float) -> None:
        self._now = now
        if self.phase is _Phase.IDLE:
            return
        if self.phase is _Phase.WAIT_KEY:
            if self.capture is not None:
                result = self.capture.poll()
                if result == CANCELLED:
                    self._reset("Cancelled")
                elif result:
                    self._bind(result)
            return

        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0
        settings = self.config.settings

        if self.phase is _Phase.COUNTDOWN:
            if elapsed >= settings.capture_delay_s:
                self.phase = _Phase.CAPTURING
                self._t0 = now
                self._samples = []
        elif self.phase is _Phase.CAPTURING:
            self._samples.append(" ".join(self.classifier.tokens(hands)))
            if elapsed >= settings.capture_duration_s:
                self._finish()

    def handle_key(self, key: int) -> bool:
        if self.phase is _Phase.IDLE:
            return False
        if key == ESC:
            self._reset("Cancelled")
            return True
        if self.phase is not _Phase.WAIT_KEY:
            return False
        if self.capture is not None:
            return True  # клавишу ловит глобальный слушатель; окно на неё не реагирует

        name = key_name_from_cv2(key)
        if name is None:
            self._say("This key can't be captured here - edit the JSON config")
            return True
        self._bind(name)
        return True

    def close(self) -> None:
        if self.capture is not None:
            self.capture.stop()

    # --- внутреннее ------------------------------------------------------------------

    def _bind(self, key_name: str) -> None:
        index = len(self.config.bindings)
        binding = Binding(
            gesture=self.captured,
            key=key_name,
            mode="press",
            name=f"gesture_{index + 1}",
            color=AUTO_COLORS[index % len(AUTO_COLORS)],
        )
        try:
            self.engine.add_binding(binding)
        except ValueError as exc:
            print(f"[gesture] не удалось привязать: {exc}")
            self._reset("Can't bind this key, edit the JSON config")
            return
        print(f"[gesture] bound {binding.gesture} -> {binding.key} (saved to {self.config.path})")
        self._reset(f"Bound {binding.gesture} -> {binding.key}")

    def _say(self, text: str, seconds: float = 4.0) -> None:
        self._message = text
        self._message_until = self._now + seconds

    def _reset(self, message: str = "") -> None:
        if self.capture is not None:
            self.capture.stop()
        self.phase = _Phase.IDLE
        self._t0 = None
        self._samples = []
        if message:
            self._say(message)

    def _finish(self) -> None:
        seen = [s for s in self._samples if s]
        if len(seen) < max(5, int(0.6 * len(self._samples))):
            self._reset("No hand detected, try again")
            return
        signature, count = Counter(seen).most_common(1)[0]
        if count / len(seen) < 0.6:
            self._reset("Gesture is unstable, hold still and try again")
            return
        self.captured = validate_gesture(signature)
        print(f"[gesture] captured: {self.captured}")
        self.phase = _Phase.WAIT_KEY
        self._t0 = None
        if self.capture is not None:
            self.capture.start()
