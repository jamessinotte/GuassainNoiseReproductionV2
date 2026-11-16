"""Small, explicit persistence and provenance helpers."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace a complete record atomically; never append an incomplete trial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def json_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def code_identity():
    paths = sorted((ROOT / "repro").glob("*.py"))
    paths += sorted((ROOT / "assets").glob("*.json"))
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}


def environment():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    try:
        import torch
        result.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      torch_threads=torch.get_num_threads())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return result

