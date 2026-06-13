# llama_light/model_manager.py
import os
import shutil
from typing import List, Optional, Tuple

from huggingface_hub import hf_hub_download

from .config import CACHE_ROOT, ensure_dirs
from .registry import find, scan_hf_cache


def _model_dir(model_id: str) -> str:
    return os.path.join(CACHE_ROOT, model_id)

def _model_path(model_id: str, filename: str) -> str:
    return os.path.join(_model_dir(model_id), filename)


def resolve_model(query: str) -> Optional[str]:
    """Triple-layer model lookup → absolute path or None."""
    scan_hf_cache()
    info = find(query)
    return info["local_path"] if info else None


def pull(repo_id: str, filename: str,
         model_id: Optional[str] = None,
         revision: Optional[str] = None) -> str:
    ensure_dirs()
    mid = model_id or repo_id.split("/")[-1]
    os.makedirs(_model_dir(mid), exist_ok=True)

    dest = _model_path(mid, filename)
    if os.path.exists(dest):
        print(f"[pull] already cached → {dest}")
        return dest

    print(f"[pull] downloading {repo_id}/{filename} ...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=_model_dir(mid),
            revision=revision,
        )
        print(f"[pull] saved → {path}")
        return path
    except Exception as e:
        raise IOError(f"Download failed: {e}")


def ls() -> List[Tuple[str, str, float]]:
    """Returns (model_id, filename, size_gb) for every cached .gguf."""
    ensure_dirs()
    results = []
    if not os.path.isdir(CACHE_ROOT):
        return results
    for model_id in sorted(os.listdir(CACHE_ROOT)):
        d = os.path.join(CACHE_ROOT, model_id)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".gguf"):
                fpath = os.path.join(d, fname)
                results.append((model_id, fname, os.path.getsize(fpath) / 1024**3))
    return results


def rm(model_id: str, filename: Optional[str] = None) -> None:
    if filename:
        path = _model_path(model_id, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not cached: {model_id}/{filename}")
        os.remove(path)
        print(f"[rm] removed {path}")
    else:
        d = _model_dir(model_id)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Not cached: {model_id}")
        shutil.rmtree(d)
        print(f"[rm] removed {d}")
