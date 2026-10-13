import subprocess
import sys

def get_pnp_camera_names() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        # Fast query via powershell
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
        if res.returncode == 0:
            return [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    except Exception:
        pass
    return []

print("PnP names:", get_pnp_camera_names())
