from pathlib import Path
from datetime import datetime
import shutil
import subprocess
from app.config.settings import BACKUP_DIR

def backup_file(file_path):
    src = Path(file_path)
    if not src.exists():
        return False

    Path(BACKUP_DIR).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = Path(BACKUP_DIR) / f"{src.name}.{timestamp}.bak"
    shutil.copy2(src, dst)
    return str(dst)

def read_file(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def write_file(file_path, content):
    backup_file(file_path)
    Path(file_path).write_text(content, encoding="utf-8")
    return True

def compile_check(file_path):
    result = subprocess.run(
        ["python3", "-m", "py_compile", file_path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr

def restart():
    import os, stat, subprocess
    from pathlib import Path
    script = Path("restart.sh")
    if not script.exists(): raise FileNotFoundError("Script not found")
    if script.is_symlink(): raise PermissionError("Symlink blocked")
    if not os.access(script, os.X_OK): script.chmod(script.stat().st_mode | stat.S_IEXEC)
    subprocess.Popen(["./restart.sh"], shell=False)

def search_internet(query: str) -> str:
    return web_search_tool.run(query)
