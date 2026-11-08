import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 4
SFT_EPOCH_UPDATES = 275
SFT_FINAL_UPDATE = 550


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path, algorithm="sha256", *, git_blob=False):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        if git_blob:
            h.update(f"blob {Path(path).stat().st_size}\0".encode())
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, writer):
    """Publish a complete file in the same directory, or leave the prior one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".pending-", delete=False) as out:
        temporary = Path(out.name)
        try:
            writer(out)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            out.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, lambda out: out.write(canonical(value) + b"\n"))


def exact_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Different frozen contract: {Path(path).name}; use a new work directory")
    else:
        atomic_json(path, value)


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for line, raw in enumerate(f, 1):
            if not raw.strip():
                raise ValueError(f"Unexpected blank row {line} in {Path(path).name}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Unexpected row type at {line} in {Path(path).name}")
            yield value


def relative_file(root, name):
    root = Path(root).resolve()
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Expected a safe relative file")
    leaf = root / path
    if leaf.is_symlink() or not leaf.is_file() or not leaf.resolve().is_relative_to(root):
        raise ValueError(f"Missing, linked or out-of-tree file: {path.name}")
    return leaf


@contextmanager
def job_lock(work, name):
    # A process lock (not a stale PID file) releases when its owner exits.
    # This also permits CPU fixture tests on Windows; training requires Linux.
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with (work / f".{name}.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError(f"{name} is already running in this work directory") from None
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError(f"{name} is already running in this work directory") from None
        yield


def source_lock():
    return read_json(ROOT / "sources.json")


def verify_checkout(name):
    entry = source_lock()[name]
    path = ROOT / ".deps" / ("llamafactory" if name == "trainer" else "genevolve")
    if not (path / ".git").exists():
        raise FileNotFoundError("Run scripts/bootstrap.sh to fetch the pinned upstream sources")
    git = lambda *args: subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("rev-parse", "HEAD") != entry["revision"] or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Pinned {name} checkout is different or modified")
    return path


def check_model_files(path):
    """Validate HF model-index/shard layout without loading weights."""
    import re
    import struct
    path = Path(path)
    cfg = read_json(relative_file(path, "config.json"))
    if cfg.get("model_type") != "qwen3_vl" or cfg.get("text_config", {}).get("hidden_size") != 4096:
        raise ValueError("Expected the Qwen3-VL-8B architecture")
    # Processor and tokenizer are needed for both training continuation and serving.
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        relative_file(path, name)
    if not any((path / name).is_file() for name in ("chat_template.jinja", "chat_template.json")):
        raise ValueError("Missing the Qwen3-VL chat template")
    index_path = path / "model.safetensors.index.json"
    weights = read_json(index_path)["weight_map"] if index_path.is_file() else {}
    shards = sorted(set(weights.values())) if weights else ["model.safetensors"]
    expected = {shard: set() for shard in shards}
    for key, shard in weights.items():
        expected[shard].add(key)
    for name in shards:
        if not re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", name):
            raise ValueError("Unexpected weight shard name")
        shard = relative_file(path, name)
        with shard.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 0 < size <= (100 << 20):
                raise ValueError("Invalid safetensors header")
            header = json.loads(f.read(size))
        offsets = sorted(tuple(t["data_offsets"]) for k, t in header.items() if k != "__metadata__")
        keys = set(header) - {"__metadata__"}
        if not keys or (weights and keys != expected[name]):
            raise ValueError("Invalid tensor map")
        end = 0
        for a, b in offsets:
            if a != end or b < a:
                raise ValueError("Invalid tensor offsets")
            end = b
        if shard.stat().st_size != 8 + size + end:
            raise ValueError("Truncated or inconsistent weight shard")
    return shards


def file_manifest(root, paths):
    result = {}
    for p in sorted(paths, key=lambda p: p.as_posix()):
        name = p.relative_to(root).as_posix()
        actual = relative_file(root, name)
        result[name] = {"bytes": actual.stat().st_size, "sha256": file_sha(actual)}
    return result


def verify_manifest(root, manifest):
    for name, spec in manifest.items():
        path = relative_file(root, name)
        if path.stat().st_size != spec["bytes"] or file_sha(path) != spec["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
