#!/usr/bin/env python3
"""Pinned GenEvolve Stage-1 recipe and 5,000 on-policy trajectory collection."""
import argparse
import json
from pathlib import Path

from ge_repro.common import SFT_EPOCH_UPDATES, SFT_FINAL_UPDATE, job_lock, source_lock
from ge_repro.data import prepare
from ge_repro.train import config, run_training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work"), help="Dedicated writable work directory")
    sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--verify-only", action="store_true")
    train = sub.add_parser("sft")
    train.add_argument("--resume", action="store_true", help="Resume the attested epoch-1 checkpoint")
    train.add_argument("--dry-run", action="store_true")
    rollout = sub.add_parser("collect")
    rollout.add_argument("--allow-live-search", action="store_true")
    rollout.add_argument("--workers", type=int, default=16)
    rollout.add_argument("--search-rps", type=float, default=4)
    rollout.add_argument("--port", type=int, default=18080)
    rollout.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    work = args.work.resolve()
    if args.mode == "prepare":
        with job_lock(work, "gpu"):
            value = prepare(work, download=not args.verify_only)
    elif args.mode == "sft":
        if args.dry_run:
            value = {"world_size": 4, "global_batch_size": 32,
                     "save_updates": [SFT_EPOCH_UPDATES, SFT_FINAL_UPDATE], "sft": config(work)}
        else:
            ckpt, receipt = run_training(work, resume=args.resume)
            value = {"checkpoint": str(ckpt), "step": receipt["step"],
                     "integrity_files": len(receipt["files"])}
    else:
        if args.workers < 1 or not 0 < args.search_rps <= 1000 or not 1024 <= args.port <= 65535:
            p.error("Invalid collection concurrency, search rate or port")
        if args.dry_run:
            value = {"checkpoint": "checkpoint-550", "data_parallel": 4, "tensor_parallel": 1,
                     "prompt_split": "GenEvolve-Data-RL/train.jsonl", **source_lock()["rollout"]}
        else:
            from ge_repro.serve import collect
            value = collect(work, port=args.port, workers=args.workers, search_rps=args.search_rps,
                            allow_live_search=args.allow_live_search)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
