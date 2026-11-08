import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from zipfile import BadZipFile

from .checkpoint import attest
from .common import (
    ROOT, WORLD_SIZE, SFT_EPOCH_UPDATES, SFT_FINAL_UPDATE, atomic_json, digest,
    exact_json, file_sha, job_lock, read_json, source_lock, verify_checkout,
)
from .data import SFT_DIR, verify_prepared


def config(work, resume=None):
    args = read_json(ROOT / "configs/sft.json")
    batch = args["per_device_train_batch_size"] * args["gradient_accumulation_steps"] * WORLD_SIZE
    if batch != 32 or source_lock()["dataset"]["train_rows"] // batch != SFT_EPOCH_UPDATES:
        raise ValueError("The paper's effective batch and updates must be preserved")
    args.update({
        "model_name_or_path": str(work / "base"),
        "dataset_dir": str(work / "data" / SFT_DIR),
        "media_dir": str(work / "data" / SFT_DIR),
        "output_dir": str(work / "checkpoints"),
        "deepspeed": str(ROOT / "configs/zero3.json"),
        "overwrite_output_dir": False,
    })
    if resume is not None:
        args["resume_from_checkpoint"] = str(resume)
    return args


def check_gpus():
    if sys.platform != "linux":
        raise RuntimeError("The four-GPU runtime requires Linux")
    # Exit the probe process before launching training so it owns no GPU memory.
    subprocess.run([sys.executable, "-c", """
import torch
if torch.cuda.device_count() != 4:
    raise SystemExit('Expose exactly the four allocated A100 GPUs')
for i in range(4):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    if 'A100' not in p.name or total < 76*2**30 or free < 72*2**30:
        raise SystemExit(f'GPU {i} must be an idle A100 80 GB')
"""], check=True)


def check_storage(work, resume_step):
    if shutil.disk_usage(work).free < (300 if resume_step == 0 else 150) * 2**30:
        raise RuntimeError("Not enough work-volume space for complete ZeRO-3 checkpoints")
    host = re.search(r"^MemAvailable:\s+(\d+)\s+kB", Path("/proc/meminfo").read_text(), re.M)
    available = int(host[1]) * 1024 if host else 0
    cap = Path("/sys/fs/cgroup/memory.max")
    used = Path("/sys/fs/cgroup/memory.current")
    if cap.is_file() and used.is_file() and cap.read_text().strip() != "max":
        available = min(available, int(cap.read_text()) - int(used.read_text()))
    if available < 160 * 2**30:
        raise RuntimeError("CPU AdamW offload needs at least 160 GiB available system RAM")


def run_training(work, *, resume):
    with job_lock(work, "gpu"):
        verify_checkout("trainer")
        prepared = verify_prepared(work)
        scientific = {
            "sources": source_lock(),
            "world_size": WORLD_SIZE,
            "global_batch_size": 32,
            "final_update": SFT_FINAL_UPDATE,
            "dataset_receipt_sha256": digest(prepared),
            "sft_config": read_json(ROOT / "configs/sft.json"),
            "deepspeed_config": read_json(ROOT / "configs/zero3.json"),
            "launcher_sha256": file_sha(__file__),
            "token_loss_entry_sha256": file_sha(Path(__file__).with_name("train_entry.py")),
        }
        exact_json(work / "run_contract.json", scientific)
        out = work / "checkpoints"
        final = out / f"checkpoint-{SFT_FINAL_UPDATE}"
        epoch1 = out / f"checkpoint-{SFT_EPOCH_UPDATES}"
        if final.exists():
            try:
                # Successful re-entry only creates a missing receipt.
                return attest(work, SFT_FINAL_UPDATE, scientific)
            except (OSError, ValueError, KeyError, BadZipFile) as error:
                if (not resume or final.is_symlink()
                        or (work / "integrity" / (final.name + ".json")).exists()):
                    raise
                # Preserve interrupted output as evidence; only explicit resume
                # from an intact epoch-1 checkpoint can move it out of the way.
                attest(work, SFT_EPOCH_UPDATES, scientific)
                folder = work / "interrupted"
                folder.mkdir(exist_ok=True)
                if folder.is_symlink():
                    raise ValueError("Interrupted checkpoint path must not be a symlink") from error
                final.rename(folder / f"{final.name}-{time.time_ns()}")
        start = None
        if resume:
            if not epoch1.is_dir():
                raise ValueError("No complete epoch-1 checkpoint to resume")
            start, _ = attest(work, SFT_EPOCH_UPDATES, scientific)
        elif out.exists() and any(out.iterdir()):
            raise ValueError("Existing run output; use --resume or a new work directory")
        check_gpus()
        check_storage(work, SFT_EPOCH_UPDATES if start else 0)
        from .sft_smoke import check_sample
        sft = work / "data" / SFT_DIR
        sample = check_sample(work / "base", sft / "train.jsonl", sft)
        atomic_json(work / "sft_preflight.json", sample)
        print("Native multimodal SFT preflight:", sample, flush=True)
        args_file = work / ("sft-resume.json" if start else "sft-launch.json")
        atomic_json(args_file, config(work, start))
        env = dict(os.environ)
        env.pop("MASTER_PORT", None)
        # Use this environment's torchrun, including on the distributed ranks.
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", "--nproc-per-node=4", "--module", "ge_repro.train_entry",
                        str(args_file)], check=True, env=env, cwd=ROOT)
        attest(work, SFT_EPOCH_UPDATES, scientific)
        return attest(work, SFT_FINAL_UPDATE, scientific)
