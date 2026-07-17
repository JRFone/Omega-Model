from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "Launch Omega FISH Model.bat"
ICON = ROOT / "assets" / "omega_fish.ico"


def powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_shortcut() -> Path:
    if not LAUNCHER.exists():
        raise FileNotFoundError(f"Launcher not found: {LAUNCHER}")
    desktop_query = "[Environment]::GetFolderPath('Desktop')"
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", desktop_query],
        check=True,
        capture_output=True,
        text=True,
    )
    desktop = Path(completed.stdout.strip())
    shortcut = desktop / "Omega FISH Model.lnk"
    icon_line = f"$shortcut.IconLocation = {powershell_literal(str(ICON))};" if ICON.exists() else ""
    script = (
        "$shell = New-Object -ComObject WScript.Shell;"
        f"$shortcut = $shell.CreateShortcut({powershell_literal(str(shortcut))});"
        "$shortcut.TargetPath = $env:ComSpec;"
        f"$shortcut.Arguments = '/c " + '""' + str(LAUNCHER) + '""' + "';"
        f"$shortcut.WorkingDirectory = {powershell_literal(str(ROOT))};"
        f"{icon_line}"
        "$shortcut.Description = 'Launch Omega FISH Model';"
        "$shortcut.Save();"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], check=True)
    if not shortcut.exists():
        raise RuntimeError(f"Windows did not create the shortcut: {shortcut}")
    return shortcut


if __name__ == "__main__":
    print(create_shortcut())
