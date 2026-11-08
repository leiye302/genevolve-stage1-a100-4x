from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .checkpoint import final_checkpoint
from .common import digest, exact_json, file_sha, job_lock, read_json, source_lock, verify_checkout
from .data import RL_DIR, SYSTEM_SHA, read_prompts
from .rollout import collect_slots, load_done, make_slots, run_slot
from .tools import LiveTools
from .train import check_gpus


def command(checkpoint, served, port, max_len):
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(checkpoint), "--served-model-name", served,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", "1", "--data-parallel-size", "4",
        "--dtype", "bfloat16", "--max-model-len", str(max_len),
        "--gpu-memory-utilization", "0.86", "--max-num-seqs", "8",
        "--limit-mm-per-prompt", '{"image":50,"video":0}',
        "--generation-config", "vllm", "--disable-log-requests",
    ]


@contextmanager
def server(work, checkpoint, alias, port, max_len):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    env = {k: v for k, v in os.environ.items() if k not in
           ("SERPER_API_KEY", "SERPER_BASE_URL", "OPENAI_API_KEY", "HF_TOKEN")}
    env["VLLM_USE_V1"] = "1"
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        with (work / "serve.log").open("ab") as log:
            process = subprocess.Popen(command(checkpoint, alias, port, max_len),
                cwd=work, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            base = f"http://127.0.0.1:{port}/v1"
            until = time.monotonic() + 600
            import json
            while time.monotonic() < until:
                if process.poll() is not None:
                    raise RuntimeError("Local vLLM did not start; see work/serve.log")
                try:
                    with local_http.open(base + "/models", timeout=3) as response:
                        models = json.load(response)["data"]
                    if not any(m.get("id") == alias and Path(m.get("root", "")).resolve() == checkpoint
                               for m in models):
                        raise RuntimeError("Serving endpoint is not using the attested SFT checkpoint")
                    break
                except (OSError, TimeoutError):
                    time.sleep(3)
            else:
                raise TimeoutError("Local vLLM readiness timed out; see work/serve.log")
            yield base
    finally:
        if process is not None:
            # Only the private process group created above, never other vLLM jobs.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        signal.signal(signal.SIGTERM, old_handler)


def collect(work, *, port, workers, search_rps, allow_live_search):
    with job_lock(work, "gpu"):
        checkpoint, receipt, run = final_checkpoint(work)
        lock = source_lock()
        if run["sources"] != lock:
            raise ValueError("The SFT checkpoint was not produced with this repository's recipe lock")
        source = work / "data" / RL_DIR / "train.jsonl"
        pool = read_prompts(source)
        system = read_json(work / "sft_system_prompt.json")["text"]
        if hashlib.sha256(system.encode("utf-8")).hexdigest() != SYSTEM_SHA:
            raise ValueError("The SFT-compatible system prompt changed")
        s = lock["rollout"]
        slots = make_slots(pool, s["count"], s["seed"])
        out = work / "rollouts"
        identity = {
            "stage1_update": receipt["step"],
            "checkpoint_sha256": digest(receipt),
            "sft_run_sha256": digest(run),
            "prompt_source": lock["dataset"]["id"],
            "dataset_revision": lock["dataset"]["revision"],
            "prompt_split": RL_DIR + "/train.jsonl",
            "prompt_sha256": file_sha(source),
            "system_prompt_sha256": SYSTEM_SHA,
            "agent_revision": lock["agent"]["revision"],
            "sampling": s,
            "collector_sha256": digest({name: file_sha(Path(__file__).with_name(name)) for name in
                                         ("serve.py", "rollout.py", "tools.py")}),
        }
        plan = {"identity": identity, "slots": slots}
        exact_json(out / "plan.json", plan)
        if len(load_done(out, slots, digest(plan))) == len(slots):
            return collect_slots(out, identity, slots, None, workers=workers)
        if not allow_live_search:
            raise ValueError("Collection uses live search quota; explicitly pass --allow-live-search")
        agent = verify_checkout("agent")
        sys.path.insert(0, str(agent))
        import httpx
        from openai import OpenAI
        from transformers import AutoTokenizer
        from genevolve import agent as native
        if not Path(native.__file__).resolve().is_relative_to(agent):
            raise ValueError("Unexpected GenEvolve import path")
        check_gpus()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        def prompt_tokens(messages):
            return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        # Test the entire fixed input pool before incurring search requests.
        for p in pool:
            if prompt_tokens([{"role": "system", "content": system},
                              {"role": "user", "content": p["prompt"]}]) > s["max_initial_prompt_tokens"]:
                raise ValueError("A planned initial prompt exceeds the published 6144-token budget")
        tools = LiveTools(out, search_rps=search_rps)
        alias = "genevolve-sft550-" + digest(receipt)[:12]
        try:
            tools.preflight()
            with server(work, checkpoint, alias, port, s["max_model_len"]) as base:
                with OpenAI(base_url=base, api_key="EMPTY", timeout=180, max_retries=1,
                            http_client=httpx.Client(trust_env=False)) as client:
                    def generate(slot):
                        return run_slot(slot, client=client, served=alias, system=system, toolset=tools,
                                        out=out, sampling=s, native=native)
                    return collect_slots(out, identity, slots, generate, workers=workers)
        finally:
            tools.close()
