from pathlib import Path
import pickletools
import re
from zipfile import ZipFile

from .common import (
    SFT_EPOCH_UPDATES, SFT_FINAL_UPDATE, WORLD_SIZE, check_model_files, digest,
    exact_json, file_manifest, read_json, relative_file, verify_manifest,
)

def metadata(archive):
    member = next((m for m in archive.infolist() if m.filename.endswith("/data.pkl")), None)
    if member is None or member.file_size > (64 << 20):
        raise ValueError("Invalid PyTorch metadata archive")
    return archive.read(member)


def has_scheduler_state(raw):
    # Inspect pickle opcodes only, never execute/unpickle arbitrary contents.
    found = False
    for opcode, arg, _ in pickletools.genops(raw):
        if found:
            if opcode.name in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                continue
            return opcode.name != "NONE"
        found = arg == "lr_scheduler"
    return False


def inspect_checkpoint(path, expected_step):
    """Fail closed on partial, model-only, or wrong-world-size saves."""
    path = Path(path)
    if (path.is_symlink() or path.name != f"checkpoint-{expected_step}"
            or expected_step not in (SFT_EPOCH_UPDATES, SFT_FINAL_UPDATE)):
        raise ValueError("Unexpected checkpoint path")
    state = read_json(relative_file(path, "trainer_state.json"))
    if (state.get("global_step") != expected_step or state.get("max_steps") != SFT_FINAL_UPDATE
            or abs(float(state.get("epoch", -1)) - expected_step / SFT_EPOCH_UPDATES) > 1e-5):
        raise ValueError("Trainer state does not match this schedule")
    shard_names = check_model_files(path)
    if not relative_file(path, "training_args.bin").stat().st_size:
        raise ValueError("Missing training arguments")
    for rank in range(WORLD_SIZE):
        if not relative_file(path, f"rng_state_{rank}.pth").stat().st_size:
            raise ValueError("Missing rank RNG state")

    tag = f"global_step{expected_step}"
    if relative_file(path, "latest").read_text(encoding="utf-8").strip() != tag:
        raise ValueError("Wrong DeepSpeed checkpoint tag")
    state_dir = path / tag
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise ValueError("Missing DeepSpeed state directory")
    model_ranks, optimizer_ranks = set(), set()
    scheduler_present = False
    if (path / "scheduler.pt").is_file():
        with ZipFile(relative_file(path, "scheduler.pt")) as archive:
            scheduler_present = bool(metadata(archive))
    for p in state_dir.iterdir():
        match = re.fullmatch(r"(?:bf16_)?zero_pp_rank_(\d+)_mp_rank_00_(model|optim)_states\.pt", p.name)
        if not match:
            continue
        f = relative_file(path, p.relative_to(path).as_posix())
        rank = int(match[1])
        with ZipFile(f) as archive:
            contents = metadata(archive)
            if match[2] == "model":
                if rank in model_ranks:
                    raise ValueError("Duplicate DeepSpeed model rank")
                model_ranks.add(rank)
                scheduler_present |= has_scheduler_state(contents)
            else:
                if rank in optimizer_ranks:
                    raise ValueError("Duplicate DeepSpeed optimizer rank")
                optimizer_ranks.add(rank)
    required_ranks = set(range(WORLD_SIZE))
    if model_ranks != required_ranks or optimizer_ranks != required_ranks or not scheduler_present:
        raise ValueError("Missing ZeRO-3 model/optimizer ranks or scheduler")
    paths = [p for p in path.rglob("*") if p.is_file() or p.is_symlink()]
    return {"step": expected_step, "world_size": WORLD_SIZE,
            "weights": shard_names, "files": file_manifest(path, paths)}


def attest(work, step, run_contract):
    ckpt = work / "checkpoints" / f"checkpoint-{step}"
    receipt = {"run_contract_sha256": digest(run_contract), **inspect_checkpoint(ckpt, step)}
    exact_json(work / "integrity" / (ckpt.name + ".json"), receipt)
    return ckpt, receipt


def final_checkpoint(work):
    path = work / "checkpoints" / f"checkpoint-{SFT_FINAL_UPDATE}"
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Missing or linked final checkpoint")
    run = read_json(work / "run_contract.json")
    receipt = read_json(work / "integrity" / (path.name + ".json"))
    if (receipt["step"] != SFT_FINAL_UPDATE or receipt["world_size"] != WORLD_SIZE
            or receipt["run_contract_sha256"] != digest(run)):
        raise ValueError("Final checkpoint is not attested for this Stage-1 run")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(receipt["files"]):
        raise ValueError("Final checkpoint file set changed")
    verify_manifest(path, receipt["files"])
    return path, receipt, run
