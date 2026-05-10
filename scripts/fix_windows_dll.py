"""Download and install Visual C++ Redistributable 2022 to fix torch DLL error."""
import urllib.request
import subprocess
import os

url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
dest = os.path.join(os.environ.get("TEMP", "."), "vc_redist.x64.exe")

print("Downloading Visual C++ Redistributable 2022...")
urllib.request.urlretrieve(url, dest)
print(f"Saved to {dest}")

print("Installing (this may take 30 seconds)...")
result = subprocess.run([dest, "/install", "/quiet", "/norestart"])

if result.returncode in (0, 3010):
    print("\nInstalled successfully!")
    print(">>> REBOOT the machine now, then run: python -c \"import torch; print(torch.__version__)\"")
else:
    print(f"\nInstaller exited with code {result.returncode}")
    print("Try running the file manually:", dest)
