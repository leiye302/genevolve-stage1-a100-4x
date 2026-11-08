"""Reproducible slot ledger and native GenEvolve protocol with visual observations."""
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import random
import time

from .common import (
    atomic_json, atomic_write, canonical, digest, exact_json, file_sha, read_json,
    relative_file, source_lock,
)
from .tools import InfrastructureError


def make_slots(pool, count, seed):
    """Give each official training prompt one draw before repeating any."""
    if not pool or count < 1 or count > 2 * len(pool):
        raise ValueError("The plan needs one or two draws from the fixed prompt pool")
    rng = random.Random(seed)
    slots = []
    for draw in range(2):
        indexes = list(range(len(pool)))
        rng.shuffle(indexes)
        for i in indexes[: count-len(slots)]:
            index = len(slots)
            slots.append({
                "slot": index, "draw": draw, "seed": rng.randrange(2**31),
                "source_id": pool[i]["source_id"], "prompt": pool[i]["prompt"],
            })
        if len(slots) == count:
            return slots
    raise AssertionError("Plan size mismatch")


def validate_asset(out, path):
    import re
    if not re.fullmatch(r"images/[0-9a-f]{64}\.jpg", path):
        raise ValueError("Invalid trajectory image name")
    return relative_file(out, path)


def multimodal_messages(messages, images, out, cache):
    """Replace only native user observation placeholders, in recorded order."""
    files = iter(images)
    result = []
    for message in messages:
        role, text = message["role"], message["content"]
        if role != "user" or "<image>" not in text:
            result.append(message)
            continue
        parts = []
        for i, piece in enumerate(text.split("<image>")):
            if i:
                name = next(files, None)
                if name is None:
                    raise ValueError("Image placeholder without an asset")
                if name not in cache:
                    cache[name] = "data:image/jpeg;base64," + base64.b64encode(
                        validate_asset(out, name).read_bytes()).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": cache[name]}})
            if piece:
                parts.append({"type": "text", "text": piece})
        result.append({"role": role, "content": parts})
    if next(files, None) is not None:
        raise ValueError("Image asset without an observation placeholder")
    return result


def final_program(payload, registry, max_refs):
    if not isinstance(payload, dict) or not isinstance(payload.get("gen_prompt"), str):
        return None, "invalid_answer"
    prompt = payload["gen_prompt"].strip()
    refs = payload.get("reference_images")
    if not prompt or not isinstance(refs, list) or len(refs) > max_refs:
        return None, "invalid_answer"
    selected = []
    for ref in refs:
        ident = ref.get("img_id") if isinstance(ref, dict) else None
        if not isinstance(ident, str) or not (record := registry.lookup(ident)):
            return None, "unresolved_reference"
        note = ref.get("note")
        if not isinstance(note, str) or not note.strip():
            return None, "invalid_reference_note"
        selected.append({**record, "note": note})
    ids = [r["img_id"] for r in selected]
    if ids != sorted(set(ids)):
        return None, "duplicate_or_unsorted_references"
    return {"gen_prompt": prompt, "reference_images": selected}, "answer"


def run_slot(slot, *, client, served, system, toolset, out, sampling, native, prompt_tokens=None):
    """Native parsers, tool ordering, final-step override; no forced successful sample."""
    registry = native.ImageIdManager()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": slot["prompt"]}]
    attachments, steps, cache = [], [], {}
    total, termination, program = 0, "max_rounds_reached", None
    if prompt_tokens and prompt_tokens(messages) > sampling["max_initial_prompt_tokens"]:
        raise InfrastructureError("initial_prompt_exceeds_budget")
    for i in range(sampling["max_rounds"]):
        if i == sampling["max_rounds"]-1:
            messages.append({"role": "user", "content": native.FINAL_STEP_MESSAGE})
        budget = min(sampling["max_tokens_per_round"], sampling["max_generated_tokens"]-total)
        if budget <= 0:
            termination = "response_budget_reached"
            break
        try:
            completion = client.chat.completions.create(
                model=served, messages=multimodal_messages(messages, attachments, out, cache),
                max_tokens=budget, temperature=sampling["temperature"], top_p=sampling["top_p"],
                seed=(slot["seed"] + i) % 2**31, extra_body={"top_k": sampling["top_k"]},
            )
            choice = completion.choices[0]
            usage = int(completion.usage.completion_tokens)
            if not 0 <= usage <= budget:
                raise ValueError("Invalid completion usage")
            text = (choice.message.content or "").strip()
        except Exception as error:
            if (steps and getattr(error, "status_code", None) == 400
                    and "maximum context length" in str(error).lower()):
                termination = "context_budget_reached"
                break
            # Never persist vendor exception bodies or request headers.
            raise InfrastructureError(f"actor_request_failed_{type(error).__name__}") from None
        total += usage
        messages.append({"role": "assistant", "content": text})
        step = {"round": i + 1, "completion_tokens": usage, "finish_reason": choice.finish_reason}
        steps.append(step)
        answer = native._parse_answer(text)
        if answer is not None:
            program, termination = final_program(answer, registry, sampling["max_reference_images"])
            break
        if i == sampling["max_rounds"]-1:
            termination = "no_final_answer"
            break
        if total >= sampling["max_generated_tokens"]:
            termination = "response_budget_reached"
            break
        try:
            tool = native._parse_tool_call(text)
        except (TypeError, AttributeError):
            tool = None
        if tool is None:
            observation = "Format error: output one <tool_call>{...}</tool_call> or <answer>{...}</answer>."
            step["tool_status"] = "invalid_action"
        else:
            name, args = tool
            observation, images, status = toolset.execute(name, args, registry, sampling["max_search_results"])
            attachments.extend(images)
            step.update({"tool": name, "arguments": args, "tool_status": status, "images": images})
        messages.append({"role": "user", "content": f"<tool_response>\n{observation}\n</tool_response>"})
    if not steps:
        raise InfrastructureError("no_actor_generation")
    return {
        **slot, "messages": messages, "images": attachments, "steps": steps,
        "generated_tokens": total, "termination": termination, "program": program,
    }


def load_done(out, slots, plan_hash):
    items = {}
    image_checked = set()
    for path in sorted((out / "episodes").glob("*.json")):
        row = read_json(path)
        slot = row.get("slot")
        if (not isinstance(slot, int) or not 0 <= slot < len(slots) or
                path.name != f"{slot:05d}.json" or slot in items or row.get("plan_sha256") != plan_hash
                or any(row.get(k) != v for k, v in slots[slot].items()) or not row.get("steps")):
            raise ValueError("Mixed, invalid, or duplicate rollout ledger entries")
        if sum(m["content"].count("<image>") for m in row["messages"] if m["role"] == "user") != len(row["images"]):
            raise ValueError("Stored trajectory has inconsistent visual observations")
        for name in row["images"]:
            if name not in image_checked:
                asset = validate_asset(out, name)
                if file_sha(asset) != asset.stem:
                    raise ValueError("Stored rollout visual asset checksum differs")
                image_checked.add(name)
        items[slot] = row
    return items


def collect_slots(out, identity, slots, generate, *, workers):
    out = Path(out)
    plan = {"identity": identity, "slots": slots}
    plan_hash = digest(plan)
    exact_json(out / "plan.json", plan)
    items = load_done(out, slots, plan_hash)
    missing = iter(s for s in slots if s["slot"] not in items)
    stop = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        def submit():
            slot = next(missing, None)
            if slot is not None:
                pending[executor.submit(generate, slot)] = slot["slot"]
        for _ in range(workers):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                slot = pending.pop(f)
                try:
                    row = f.result()
                except Exception as error:
                    category = str(error) if isinstance(error, InfrastructureError) else type(error).__name__
                    atomic_json(out / "errors" / f"{slot:05d}-{time.time_ns()}.json",
                                {"slot": slot, "plan_sha256": plan_hash, "reason": category})
                    stop = category
                    continue
                if any(row.get(k) != v for k, v in slots[slot].items()):
                    raise ValueError("Generator returned a foreign rollout slot")
                value = {**row, "plan_sha256": plan_hash}
                atomic_json(out / "episodes" / f"{slot:05d}.json", value)
                items[slot] = value
                if len(items) % 25 == 0:
                    print(f"Saved {len(items)}/{len(slots)} actual trajectories", flush=True)
            if stop is None:
                for _ in done:
                    submit()
    if stop is not None:
        raise InfrastructureError(f"Collection paused after {len(items)} durable trajectories: {stop}")
    if len(items) != len(slots):
        raise ValueError("Incomplete collection")
    items = load_done(out, slots, plan_hash)
    if len(items) != len(slots):
        raise ValueError("The durable slot set is incomplete")
    path = out / "trajectories.jsonl"
    def write_archive(f):
        for i in range(len(slots)):
            f.write(canonical(items[i]) + b"\n")
    atomic_write(path, write_archive)
    receipt = {
        "count": len(items), "plan_sha256": plan_hash, "source": identity,
        "distinct_prompts": len({r["source_id"] for r in items.values()}),
        "valid_programs": sum(r["program"] is not None for r in items.values()),
        "terminal_counts": dict(Counter(r["termination"] for r in items.values())),
        "visual_assets": len({name for r in items.values() for name in r["images"]}),
        "trajectories_sha256": file_sha(path),
    }
    atomic_json(out / "manifest.json", receipt)
    return receipt
