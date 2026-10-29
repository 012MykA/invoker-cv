"""JSON-конфиг: настройки и список биндов "жест -> клавиша (+ цвет подсветки)"."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .gestures import FingerThresholds

MODES = ("press", "release", "hold", "repeat")


def parse_color(value) -> tuple[int, int, int] | None:
    """'#ff8800' или [255, 136, 0] (RGB) -> BGR для OpenCV. None/'' -> без цвета."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str):
            text = value.strip().lstrip("#")
            if len(text) != 6:
                raise ValueError
            r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            r, g, b = (int(v) for v in value)
            if not all(0 <= c <= 255 for c in (r, g, b)):
                raise ValueError
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f"Неверный цвет {value!r}: ожидается '#rrggbb' или [r, g, b]") from None
    return (b, g, r)


def _build(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Binding:
    gesture: str                       # "Right:01000", "*:01???", "Left:11111 Right:01000"
    key: str                           # любая клавиша: "q", "space", "f5", "up", "ctrl+shift+s", "vk65"
    mode: str = "press"                # press - нажать клавишу, когда жест появился; release - когда пропал
                                       # (ещё: hold - держать, пока жест; repeat - повторять)
    name: str = ""
    repeat_ms: int = 150               # интервал для mode = repeat
    color: str | list[int] | None = None   # цвет, в который красится вся кисть: "#ff0000" или [255, 0, 0]

    def __post_init__(self) -> None:
        if self.mode == "tap":  # старое название режима
            self.mode = "press"
        if self.mode not in MODES:
            raise ValueError(f"Бинд {self.name or self.gesture!r}: mode должен быть одним из {MODES}")
        parse_color(self.color)

    @property
    def bgr(self) -> tuple[int, int, int] | None:
        return parse_color(self.color)


@dataclass
class Settings:
    stable_frames: int = 5             # сколько кадров подряд жест должен держаться
    release_frames: int = 4            # сколько кадров подряд без жеста, чтобы считать отпущенным
    cooldown_ms: int = 400             # минимальный интервал между срабатываниями mode = press
    dry_run: bool = False              # не нажимать клавиши, только печатать в консоль
    enabled_on_start: bool = True      # сразу эмулировать нажатия (переключается клавишей k)
    mirror: bool = True                # зеркалить картинку с камеры
    swap_handedness: bool = True       # поменять Left/Right (см. detector.py)
    capture_delay_s: float = 0.1       # пауза перед записью жеста (клавиша b)
    capture_duration_s: float = 0.1    # сколько секунд записывать жест
    thresholds: FingerThresholds = field(default_factory=FingerThresholds)


@dataclass
class Config:
    path: Path
    settings: Settings = field(default_factory=Settings)
    bindings: list[Binding] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            config = cls(path=path)
            config.save()
            print(f"[config] создан {path}")
            return config

        data = json.loads(path.read_text(encoding="utf-8"))
        raw_settings = dict(data.get("settings", {}))
        thresholds = _build(FingerThresholds, raw_settings.pop("thresholds", {}))
        settings = _build(Settings, raw_settings)
        settings.thresholds = thresholds
        bindings = [_build(Binding, item) for item in data.get("bindings", [])]
        return cls(path=path, settings=settings, bindings=bindings)

    def save(self) -> None:
        data = {
            "settings": asdict(self.settings),
            "bindings": [asdict(b) for b in self.bindings],
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
