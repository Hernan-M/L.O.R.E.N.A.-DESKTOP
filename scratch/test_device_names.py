import sys
import ctypes
from ctypes import wintypes

def get_camera_names():
    # Test getting camera names on Windows
    names = []
    if sys.platform != "win32":
        return names
    try:
        import subprocess
        # PowerShell query for video capture devices
        cmd = ["powershell", "-NoProfile", "-Command", 
               "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' } | Select-Object -ExpandProperty Name"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    except Exception as e:
        print("Error:", e)
    return names

print("Cameras from WMI:", get_camera_names())
