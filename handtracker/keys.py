"""Эмуляция и перехват клавиш. Остальной код зависит только от абстракций KeyEmitter и KeyCapture.

Имя клавиши в конфиге - любое из:
    буква/цифра/символ:     q   1   ;   ,
    именованная клавиша:    space enter tab esc shift ctrl alt up down left right
                            f1..f24 home end page_up page_down insert delete caps_lock
                            print_screen scroll_lock num_lock pause menu cmd
                            media_play_pause media_volume_up ... (всё из pynput.keyboard.Key)
    левая/правая:           shift_l shift_r ctrl_l ctrl_r alt_l alt_r
    цифровой блок (Windows): numpad0..numpad9 numpad_add numpad_subtract numpad_multiply
                            numpad_divide numpad_decimal
    любой виртуальный код:  vk65  (число - vk-код клавиши в вашей ОС)
    комбинация:             ctrl+shift+s   alt+f4
"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod

# Удобные псевдонимы -> имена клавиш pynput или одиночные символы.
ALIASES = {
    "escape": "esc",
    "return": "enter",
    "control": "ctrl",
    "win": "cmd",
    "windows": "cmd",
    "super": "cmd",
    "pageup": "page_up",
    "pagedown": "page_down",
    "del": "delete",
    "ins": "insert",
    "capslock": "caps_lock",
    "numlock": "num_lock",
    "scrolllock": "scroll_lock",
    "prtsc": "print_screen",
    "printscreen": "print_screen",
    "apps": "menu",
    "arrow_up": "up",
    "arrow_down": "down",
    "arrow_left": "left",
    "arrow_right": "right",
    "left_shift": "shift_l", "lshift": "shift_l",
    "right_shift": "shift_r", "rshift": "shift_r",
    "left_ctrl": "ctrl_l", "lctrl": "ctrl_l",
    "right_ctrl": "ctrl_r", "rctrl": "ctrl_r",
    "left_alt": "alt_l", "lalt": "alt_l",
    "right_alt": "alt_r", "ralt": "alt_r",
    # символы, которые неудобно писать в комбинации
    "plus": "+", "minus": "-", "equals": "=", "comma": ",", "period": ".",
    "slash": "/", "backslash": "\\", "semicolon": ";", "quote": "'",
    "grave": "`", "lbracket": "[", "rbracket": "]",
}

# Виртуальные коды цифрового блока Windows.
NUMPAD_VK = {f"numpad{i}": 0x60 + i for i in range(10)}
NUMPAD_VK.update({
    "numpad_multiply": 0x6A, "numpad_add": 0x6B, "numpad_subtract": 0x6D,
    "numpad_decimal": 0x6E, "numpad_divide": 0x6F,
})

_NUMPAD_BY_VK = {vk: name for name, vk in NUMPAD_VK.items()}
_VK_RE = re.compile(r"^vk(\d+)$")


def _is_windows() -> bool:
    return sys.platform == "win32"

CANCELLED = "<cancelled>"  # результат KeyCapture.poll(), если запись отменена (ESC)


def key_name_from_cv2(code: int) -> str | None:
    """Код cv2.waitKey -> имя клавиши для конфига. None, если клавишу так не записать."""
    special = {32: "space", 13: "enter", 9: "tab", 8: "backspace", ord("+"): "plus"}
    if code in special:
        return special[code]
    if 33 <= code <= 126:
        return chr(code).lower()
    return None


# ---------------------------------------------------------------------------------------
# Эмуляция нажатий
# ---------------------------------------------------------------------------------------

class KeyEmitter(ABC):
    """Комбинация - строка вида 'space', 'a', 'ctrl+shift+s', 'f5'."""

    @abstractmethod
    def press(self, combo: str) -> None: ...

    @abstractmethod
    def release(self, combo: str) -> None: ...

    def tap(self, combo: str) -> None:
        self.press(combo)
        self.release(combo)

    def validate(self, combo: str) -> None:
        """Бросает ValueError, если комбинацию нельзя распознать."""

    def close(self) -> None:
        pass


class LogKeyEmitter(KeyEmitter):
    """Ничего не нажимает, только печатает. Для проверки конфига."""

    def press(self, combo: str) -> None:
        print(f"[keys] press   {combo}")

    def release(self, combo: str) -> None:
        print(f"[keys] release {combo}")

    def tap(self, combo: str) -> None:
        print(f"[keys] tap     {combo}")


class PynputKeyEmitter(KeyEmitter):
    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key, KeyCode

        self._kb = Controller()
        self._keys = Key.__members__
        self._KeyCode = KeyCode

    def _parse(self, combo: str) -> list:
        parsed = []
        for part in combo.split("+") if combo != "+" else ["+"]:
            name = part.strip().lower()
            if not name:
                raise ValueError(f"Пустая клавиша в комбинации {combo!r}")
            name = ALIASES.get(name, name)
            vk = _VK_RE.match(name)
            if vk:
                parsed.append(self._KeyCode.from_vk(int(vk.group(1))))
            elif name in NUMPAD_VK:
                parsed.append(self._KeyCode.from_vk(NUMPAD_VK[name]))
            elif len(name) == 1:
                parsed.append(name)
            elif name in self._keys:
                parsed.append(self._keys[name])
            else:
                raise ValueError(f"Неизвестная клавиша {part!r} в комбинации {combo!r}")
        return parsed

    def press(self, combo: str) -> None:
        for key in self._parse(combo):
            self._kb.press(key)

    def release(self, combo: str) -> None:
        for key in reversed(self._parse(combo)):
            self._kb.release(key)

    def validate(self, combo: str) -> None:
        self._parse(combo)


def make_emitter(dry_run: bool = False) -> KeyEmitter:
    if dry_run:
        return LogKeyEmitter()
    try:
        return PynputKeyEmitter()
    except Exception as exc:  # нет pynput или нет доступа к вводу
        print(f"[keys] pynput недоступен ({exc}); работаю в режиме dry-run. pip install pynput")
        return LogKeyEmitter()


# ---------------------------------------------------------------------------------------
# Перехват клавиши (для записи бинда): ловит ЛЮБУЮ клавишу и комбинации, даже без фокуса окна
# ---------------------------------------------------------------------------------------

_CANON = {
    "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt_l": "alt", "alt_r": "alt",
    "shift_l": "shift", "shift_r": "shift",
    "cmd_l": "cmd", "cmd_r": "cmd",
}


def canonical_key_name(key) -> str:
    """Событие pynput (Key или KeyCode) -> имя клавиши для конфига. Работает по атрибутам."""
    if hasattr(key, "char"):  # KeyCode: обычная клавиша
        char = getattr(key, "char", None)
        vk = getattr(key, "vk", None)
        if char == " ":
            return "space"
        if char == "+":
            return "plus"
        if char and char.isprintable():
            return char.lower()
        # С зажатым Ctrl pynput отдаёт управляющий символ - восстанавливаем по vk-коду.
        # На Windows 0x61-0x7A это цифровой блок и F-клавиши, а не буквы, поэтому диапазоны разные.
        if vk is not None:
            if _is_windows():
                if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
                    return chr(vk).lower()
                if vk in _NUMPAD_BY_VK:
                    return _NUMPAD_BY_VK[vk]
            elif 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A or 0x61 <= vk <= 0x7A:
                return chr(vk).lower()
        return f"vk{vk}" if vk is not None else "unknown"
    name = getattr(key, "name", None) or str(key)
    return _CANON.get(name, name)


class ComboBuilder:
    """Из потока нажатий/отпусканий собирает комбинацию: ctrl+shift+s или одиночный модификатор."""

    MODIFIERS = ("ctrl", "alt", "shift", "cmd")

    def __init__(self) -> None:
        self._mods: list[str] = []

    def press(self, name: str) -> str | None:
        if name in self.MODIFIERS:
            if name not in self._mods:
                self._mods.append(name)
            return None
        return "+".join([*self._mods, name])

    def release(self, name: str) -> str | None:
        # Модификатор отпущен, а обычной клавиши так и не было - значит, бинд на сами модификаторы.
        if name in self.MODIFIERS and self._mods:
            return "+".join(self._mods)
        return None


class KeyCapture(ABC):
    @abstractmethod
    def start(self) -> None:
        """Начать ждать следующую клавишу/комбинацию."""

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def poll(self) -> str | None:
        """Имя клавиши/комбинации, CANCELLED (нажат ESC) или None, если ещё ничего нет."""


class PynputKeyCapture(KeyCapture):
    def __init__(self) -> None:
        from pynput import keyboard

        self._keyboard = keyboard
        self._listener = None
        self._builder = ComboBuilder()
        self._result: str | None = None

    def start(self) -> None:
        self.stop()
        self._builder = ComboBuilder()
        self._result = None
        self._listener = self._keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def poll(self) -> str | None:
        return self._result

    def _on_press(self, key):
        self._handle(canonical_key_name(key), pressed=True)
        return self._result is None  # False останавливает слушатель

    def _on_release(self, key):
        self._handle(canonical_key_name(key), pressed=False)
        return self._result is None

    def _handle(self, name: str, pressed: bool) -> None:
        if self._result is not None:
            return
        if name == "esc":
            self._result = CANCELLED
            return
        result = self._builder.press(name) if pressed else self._builder.release(name)
        if result:
            self._result = result


def make_key_capture() -> KeyCapture | None:
    """None, если pynput недоступен: тогда запись клавиши идёт через окно OpenCV."""
    try:
        return PynputKeyCapture()
    except Exception:
        return None
