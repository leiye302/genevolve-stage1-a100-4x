from collections import Counter
from pathlib import Path
import re

from .common import (
    atomic_json, check_model_files, digest, exact_json, file_manifest, file_sha,
    relative_file, rows, source_lock,
)

SFT_DIR = "GenEvolve-Data-SFT"
RL_DIR = "GenEvolve-Data-RL"
SYSTEM_SHA = "6147c9217f39c63d66f0545fee3dc206801c5b82f9309615a6ae9225c7b9cfe1"
IMAGES = 50291


def sft_registry():
    return {
        "genevolve_" + split: {
            "file_name": split + ".jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role", "content_tag": "content", "user_tag": "user",
                "assistant_tag": "assistant", "system_tag": "system",
            },
        }
        for split in ("train", "eval")
    }


def read_prompts(path, *, official=True):
    if official and file_sha(path, "sha1", git_blob=True) != source_lock()["dataset"]["prompt_git_blob"]:
        raise ValueError("Self-evolution train split does not match the pinned Git blob")
    pool = []
    for row in rows(path):
        prompt, ident = row["prompt"], str(row["id"])
        if not isinstance(prompt, str) or not prompt.strip() or not ident:
            raise ValueError("Empty request in the source prompt pool")
        # Deliberate allowlist. GT image, answers, and metadata never reach the policy.
        pool.append({"source_id": ident, "prompt": prompt})
    if len({r["source_id"] for r in pool}) != len(pool) or len({r["prompt"] for r in pool}) != len(pool):
        raise ValueError("Duplicate tasks in the rollout source")
    if official and len(pool) != source_lock()["dataset"]["prompt_rows"]:
        raise ValueError("Wrong self-evolution training pool size")
    return pool


def verify_metadata(train, evaluation, prompts):
    lock = source_lock()["dataset"]
    train_hash, eval_hash = file_sha(train), file_sha(evaluation, "sha1", git_blob=True)
    if train_hash != lock["train_sha256"] or eval_hash != lock["eval_git_blob"]:
        raise ValueError("SFT splits are not the pinned authors' files")
    system = None
    seen_images = set()
    seen_prompts = set()
    stats = {}
    for split, path in (("train", train), ("eval", evaluation)):
        count = turns = max_images = 0
        for row in rows(path):
            m, images = row["messages"], row["images"]
            if [x["role"] for x in m] != ["system"] + ["user", "assistant"] * ((len(m)-1)//2):
                raise ValueError("Invalid role alternation in SFT source")
            if not isinstance(images, list) or any(
                not re.fullmatch(r"images/traj_\d+/IMG_\d+\.jpg", p) for p in images
            ):
                raise ValueError("Unexpected SFT image asset path")
            if system is None:
                system = m[0]["content"]
                import hashlib
                if hashlib.sha256(system.encode("utf-8")).hexdigest() != SYSTEM_SHA:
                    raise ValueError("Different SFT system prompt")
            if m[0]["content"] != system:
                raise ValueError("Inconsistent SFT system prompt")
            if sum(x["content"].count("<image>") for x in m if x["role"] == "user") != len(images):
                raise ValueError("SFT visual placeholders do not match ordered image files")
            if any("<image>" in x["content"] for x in m if x["role"] != "user"):
                raise ValueError("Unexpected image placeholder outside user observations")
            if not m[-1]["content"].strip() or "<answer>" not in m[-1]["content"]:
                raise ValueError("SFT trajectory lacks a final program")
            seen_images.update(images)
            seen_prompts.add(m[1]["content"])
            count += 1
            turns += (len(m)-1)//2
            max_images = max(max_images, len(images))
        if count != lock[split + "_rows"]:
            raise ValueError(f"Unexpected SFT {split} size")
        stats[split] = {"rows": count, "assistant_turns": turns, "max_images": max_images}
    if len(seen_images) != IMAGES:
        raise ValueError("Unexpected count of unique training/eval image files")
    pool = read_prompts(prompts)
    if any(r["prompt"] in seen_prompts for r in pool):
        raise ValueError("Unexpected identical request in SFT and rollout pools")
    return {
        "train_sha256": train_hash, "eval_git_blob": eval_hash, "prompt_sha256": file_sha(prompts),
        "system_prompt_sha256": SYSTEM_SHA, "splits": stats, "unique_images": len(seen_images),
        "rollout_pool_rows": len(pool),
    }, sorted(seen_images), system


def verify_prepared(work):
    data = work / "data"
    sft = data / SFT_DIR
    metadata, images, system = verify_metadata(sft / "train.jsonl", sft / "eval.jsonl", data / RL_DIR / "train.jsonl")
    # Only relative references are passed to LLaMA-Factory. Verify their bytes
    # before training and at resume so a repopulated cache cannot change samples.
    image_sha = digest([(name, file_sha(relative_file(sft, name))) for name in images])
    model = work / "base"
    check_model_files(model)
    model_files = [p for p in model.iterdir() if p.is_file() and p.name != ".gitignore"]
    receipt = {
        **metadata,
        "dataset": source_lock()["dataset"],
        "model": source_lock()["model"],
        "image_inventory_sha256": image_sha,
        "base_files": file_manifest(model, model_files),
    }
    exact_json(work / "prepared.json", receipt)
    exact_json(sft / "dataset_info.json", sft_registry())
    exact_json(work / "sft_system_prompt.json", {"sha256": SYSTEM_SHA, "text": system})
    return receipt


def prepare(work, *, download):
    if download:
        from huggingface_hub import snapshot_download
        lock = source_lock()
        model, dataset = lock["model"], lock["dataset"]
        snapshot_download(model["id"], revision=model["revision"], local_dir=work / "base",
                          allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "LICENSE"])
        snapshot_download(
            dataset["id"], revision=dataset["revision"], repo_type="dataset",
            local_dir=work / "data", max_workers=16, allow_patterns=[
                SFT_DIR + "/train.jsonl", SFT_DIR + "/eval.jsonl", SFT_DIR + "/images/**", RL_DIR + "/train.jsonl",
            ],
        )
    return verify_prepared(work)
