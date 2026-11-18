from __future__ import annotations

import shutil
import subprocess


def copy_text(text: str) -> None:
    try:
        import pyperclip

        pyperclip.copy(text)
        return
    except Exception:
        pass

    if _tk_copy(text):
        return

    commands = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        subprocess.run(command, input=text, text=True, check=True)
        return

    raise RuntimeError("No clipboard backend found. Install pyperclip, wl-copy, xclip, or xsel.")


def paste_text() -> str:
    try:
        import pyperclip

        return pyperclip.paste()
    except Exception:
        pass

    text = _tk_paste()
    if text is not None:
        return text

    commands = [
        ["wl-paste"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, text=True, check=True, capture_output=True)
        return result.stdout

    raise RuntimeError("No clipboard backend found. Install pyperclip, wl-paste, xclip, or xsel.")


def _tk_copy(text: str) -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception:
        return False


def _tk_paste() -> str | None:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return text
    except Exception:
        return None
