# llama_light/registry.py
import datetime
import json
import os
import tempfile
from typing import Dict, List, Optional

from .config import HF_CACHE_DIR, REGISTRY_FILE, ensure_dirs


def _load() -> Dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"models": {}, "snapshots": {}}

def _save(data: Dict) -> None:
    ensure_dirs()
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(REGISTRY_FILE))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, REGISTRY_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise


def scan_hf_cache() -> int:
    data  = _load()
    known = {m["local_path"] for m in data["models"].values()}
    added = 0

    if not os.path.isdir(HF_CACHE_DIR):
        return 0

    for root, _, files in os.walk(HF_CACHE_DIR):
        for fname in files:
            if not fname.endswith(".gguf"):
                continue
            fpath = os.path.join(root, fname)
            if fpath in known:
                continue
            try:
                size_gb = os.path.getsize(fpath) / 1024**3
                base    = os.path.splitext(fname)[0][:50]
                name    = base
                suffix  = 1
                while name in data["models"]:
                    name = f"{base}_{suffix}"; suffix += 1
                data["models"][name] = {
                    "name":          name,
                    "hf_repo":       "auto-detected",
                    "hf_file":       fname,
                    "local_path":    fpath,
                    "size_gb":       round(size_gb, 2),
                    "registered_at": datetime.datetime.now().isoformat(),
                }
                known.add(fpath)
                added += 1
            except Exception:
                continue

    # prune entries whose files have been deleted
    data["models"] = {
        k: v for k, v in data["models"].items()
        if os.path.exists(v.get("local_path", ""))
    }
    _save(data)
    return added


def register(name: str, local_path: str,
             hf_repo: str = "", hf_file: str = "") -> None:
    data    = _load()
    size_gb = os.path.getsize(local_path) / 1024**3 if os.path.exists(local_path) else 0
    data["models"][name] = {
        "name":          name,
        "hf_repo":       hf_repo,
        "hf_file":       hf_file,
        "local_path":    local_path,
        "size_gb":       round(size_gb, 2),
        "registered_at": datetime.datetime.now().isoformat(),
    }
    _save(data)


def find(query: str) -> Optional[Dict]:
    """Triple-layer: registry exact → registry fuzzy → HF cache → direct path."""
    data = _load()
    q    = query.lower()

    # exact
    if query in data["models"]:
        return data["models"][query]
    # fuzzy name / filename
    for name, info in data["models"].items():
        fname = os.path.basename(info["local_path"])
        if q in name.lower() or q in fname.lower():
            return info
    # HF cache scan
    if os.path.isdir(HF_CACHE_DIR):
        for root, _, files in os.walk(HF_CACHE_DIR):
            for fname in files:
                if fname.endswith(".gguf") and q in fname.lower():
                    fpath = os.path.join(root, fname)
                    return {
                        "name":       os.path.splitext(fname)[0],
                        "hf_repo":    "auto-detected",
                        "hf_file":    fname,
                        "local_path": fpath,
                        "size_gb":    round(os.path.getsize(fpath) / 1024**3, 2),
                    }
    # direct path
    if os.path.isfile(query):
        return {
            "name":       os.path.basename(query),
            "hf_repo":    "",
            "hf_file":    os.path.basename(query),
            "local_path": os.path.abspath(query),
            "size_gb":    round(os.path.getsize(query) / 1024**3, 2),
        }
    return None


def list_models() -> List[Dict]:
    return sorted(_load()["models"].values(), key=lambda m: m["name"])


def delete_model(name: str) -> Optional[str]:
    data = _load()
    if name in data["models"]:
        path = data["models"].pop(name)["local_path"]
        _save(data)
        return path
    return None
