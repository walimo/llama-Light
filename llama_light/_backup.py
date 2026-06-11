"""Config backup and restore — safety net for config changes."""
import glob
import os
import shutil
from .config import CONFIG_DIR, GLOBAL_CONFIG, ensure_dirs

def backup() -> str | None:
    ensure_dirs()
    if not os.path.exists(GLOBAL_CONFIG):
        return None
    ts = __import__("datetime").datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = os.path.join(CONFIG_DIR, f"config.backup.{ts}.json")
    shutil.copy2(GLOBAL_CONFIG, backup_path)
    backups = sorted(glob.glob(os.path.join(CONFIG_DIR, "config.backup.*.json")))
    for old in backups[:-10]:
        try:
            os.remove(old)
        except OSError:
            pass
    return backup_path

def restore(backup_path: str | None = None) -> None:
    if backup_path:
        dest = GLOBAL_CONFIG
    else:
        backups = sorted(glob.glob(os.path.join(CONFIG_DIR, "config.backup.*.json")))
        if not backups:
            print("[backup] no backups found")
            return
        backup_path = backups[-1]
        dest = GLOBAL_CONFIG
    shutil.copy2(backup_path, dest)
    print(f"[backup] restored from {backup_path}")

def list_backups() -> list[tuple[str, int, float]]:
    backups = sorted(glob.glob(os.path.join(CONFIG_DIR, "config.backup.*.json")))
    result = []
    for p in backups:
        try:
            s = os.stat(p)
            result.append((p, s.st_size, s.st_mtime))
        except OSError:
            continue
    return result
