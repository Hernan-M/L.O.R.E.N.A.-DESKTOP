import subprocess
import time
import sys
import json

def get_windows_camera_names() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        t0 = time.monotonic()
        # Fast query via PowerShell CIM / WMI
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' } | Select-Object -ExpandProperty Caption"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
        names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        print(f"Query took {time.monotonic() - t0:.3f}s: {names}")
        return names
    except Exception as exc:
        print("Error getting camera names:", exc)
        return []

names = get_windows_camera_names()
print("Names:", names)
