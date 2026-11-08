import hashlib
import json
from pathlib import Path
import pickle
import re
import struct
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from ge_repro.checkpoint import attest, final_checkpoint, inspect_checkpoint
from ge_repro.common import SFT_FINAL_UPDATE, atomic_json, file_sha, relative_file, source_lock
from ge_repro.data import read_prompts, sft_registry
from ge_repro.rollout import collect_slots, make_slots, run_slot
from ge_repro.serve import command
from ge_repro.tools import InfrastructureError
from ge_repro.train import config


class ImageIds:
    def __init__(self):
        self.items = {}

    def lookup(self, key):
        return self.items.get(key)


def parse(text, tag):
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.S)
    return json.loads(match[1]) if match else None


native = SimpleNamespace(
    ImageIdManager=ImageIds, FINAL_STEP_MESSAGE="End with a final answer.",
    _parse_answer=lambda x: parse(x, "answer"),
    _parse_tool_call=lambda x: (y["name"], y["arguments"]) if (y := parse(x, "tool_call")) else None,
)


def fixture_checkpoint(work, step=550):
    path = work / "checkpoints" / f"checkpoint-{step}"
    path.mkdir(parents=True)
    atomic_json(path / "config.json", {"model_type": "qwen3_vl", "text_config": {"hidden_size": 4096}})
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"):
        atomic_json(path / name, {})
    header = json.dumps({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x01")
    atomic_json(path / "trainer_state.json", {"global_step": step, "max_steps": 550, "epoch": step/275})
    (path / "training_args.bin").write_bytes(b"test")
    (path / "latest").write_text(f"global_step{step}", encoding="utf-8")
    z = path / f"global_step{step}"
    z.mkdir()
    for i in range(4):
        (path / f"rng_state_{i}.pth").write_bytes(b"fixture")
        for kind in ("model", "optim"):
            name = f"bf16_zero_pp_rank_{i}_mp_rank_00_{kind}_states.pt"
            with ZipFile(z / name, "w") as f:
                f.writestr("archive/data.pkl", pickle.dumps({"lr_scheduler": {"last_epoch": step}}))
    return path


class ContractTests(unittest.TestCase):
    def test_exact_paper_sft_and_four_cards(self):
        args = config(Path("scratch"))
        self.assertEqual(args["gradient_accumulation_steps"] * args["per_device_train_batch_size"] * 4, 32)
        self.assertEqual(8800 // 32 * args["num_train_epochs"], SFT_FINAL_UPDATE)
        self.assertEqual((args["cutoff_len"], args["learning_rate"], args["warmup_ratio"]), (32768, 1e-5, .02))
        self.assertEqual(args["finetuning_type"], "full")
        self.assertTrue(args["freeze_vision_tower"] and args["freeze_multi_modal_projector"])
        self.assertFalse(args["train_on_prompt"] or args["mask_history"] or args["save_only_model"])
        self.assertEqual(sft_registry()["genevolve_eval"]["file_name"], "eval.jsonl")
        serve = command(Path("scratch/checkpoint-550"), "fixture", 18080, 65536)
        self.assertEqual(serve[serve.index("--data-parallel-size") + 1], "4")
        self.assertEqual(serve[serve.index("--host") + 1], "127.0.0.1")

    def test_exact_5000_slots_no_holdout_or_ground_truth(self):
        pool = [{"source_id": str(i), "prompt": "Question " + str(i)} for i in range(2575)]
        x = make_slots(pool, 5000, 2026)
        self.assertEqual(x, make_slots(pool, 5000, 2026))
        self.assertEqual(len({r["source_id"] for r in x[:2575]}), 2575)
        self.assertEqual(len({r["source_id"] for r in x[2575:]}), 2425)
        self.assertEqual([r["slot"] for r in x], list(range(5000)))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            path.write_text('{"id":1,"prompt":"request","gt_image":"secret.png","meta":{"target":"hidden"}}\n')
            self.assertEqual(read_prompts(path, official=False), [{"source_id": "1", "prompt": "request"}])

    def test_checkpoint_integrity_and_full_resume_files(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            cp = fixture_checkpoint(work)
            run = {"fixture": True}
            atomic_json(work / "run_contract.json", run)
            attest(work, 550, run)
            self.assertEqual(final_checkpoint(work)[0], cp)
            (cp / "rng_state_2.pth").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                final_checkpoint(work)
            (cp / "global_step550" / "bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt").unlink()
            with self.assertRaisesRegex(ValueError, "model/optimizer"):
                inspect_checkpoint(cp, 550)

    def test_no_model_only_or_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            cp = fixture_checkpoint(Path(td))
            with (cp / "model.safetensors").open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "inconsistent weight shard"):
                inspect_checkpoint(cp, 550)
            with self.assertRaisesRegex(ValueError, "safe relative"):
                relative_file(cp, "../outside")

    def test_pixels_really_enter_the_next_actor_request(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            image = b"fixture image bytes"
            name = "images/" + hashlib.sha256(image).hexdigest() + ".jpg"
            path = out / name
            path.parent.mkdir()
            path.write_bytes(image)
            slot = {"slot": 0, "source_id": "1", "prompt": "Draw a tower", "draw": 0, "seed": 10}
            first = '<think>Find the shape.</think><tool_call>{"name":"image_search","arguments":{"query":"tower"}}</tool_call>'
            final = '<think>Use the reference.</think><answer>{"gen_prompt":"Draw the first reference image","reference_images":[{"img_id":"IMG_001","note":"shape"}]}</answer>'
            calls = []
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(usage=SimpleNamespace(completion_tokens=10),
                    choices=[SimpleNamespace(finish_reason="stop",
                             message=SimpleNamespace(content=[first, final][len(calls)-1]))])
            def execute(tool, args, reg, k):
                self.assertEqual((tool, args["query"]), ("image_search", "tower"))
                reg.items["IMG_001"] = {"img_id": "IMG_001", "local_path": name}
                return "IMG_001: <image>\n  title: tower", [name], "ok"
            row = run_slot(slot, client=SimpleNamespace(chat=SimpleNamespace(
                           completions=SimpleNamespace(create=create))), served="stage1",
                           system="system", toolset=SimpleNamespace(execute=execute), out=out,
                           sampling=source_lock()["rollout"], native=native)
            self.assertEqual(calls[0]["messages"][1]["content"], "Draw a tower")
            parts = calls[1]["messages"][-1]["content"]
            self.assertEqual([p["type"] for p in parts], ["text", "image_url", "text"])
            self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(row["termination"], "answer")
            self.assertEqual(row["images"], [name])
            self.assertEqual(row["messages"][2]["content"], first)
            self.assertEqual(row["program"]["reference_images"][0]["local_path"], name)

    def test_transport_pause_then_exact_resume_keeps_model_failures(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            pool = [{"source_id": str(i), "prompt": f"task {i}"} for i in range(5)]
            slots = make_slots(pool, 8, 2026)
            identity = {"fixed_checkpoint": "550"}
            first_calls, second_calls = [], []
            def generated(slot):
                # Genuine generation may terminate without a valid program.
                return {**slot, "messages": [{"role": "assistant", "content": "unparseable"}],
                        "steps": [{"round": 1}], "images": [], "program": None, "termination": "no_final_answer"}
            def pause(slot):
                first_calls.append(slot["slot"])
                if slot["slot"] == 2:
                    raise InfrastructureError("search_service_failure")
                return generated(slot)
            with self.assertRaisesRegex(InfrastructureError, "paused after 2 durable"):
                collect_slots(out, identity, slots, pause, workers=1)
            self.assertFalse((out / "manifest.json").exists())
            self.assertEqual(len(list((out / "episodes").glob("*.json"))), 2)
            self.assertEqual(len(list((out / "errors").glob("*.json"))), 1)
            def resume(slot):
                second_calls.append(slot["slot"])
                return generated(slot)
            r = collect_slots(out, identity, slots, resume, workers=2)
            self.assertEqual(first_calls, [0, 1, 2])
            self.assertEqual(sorted(second_calls), list(range(2, 8)))
            self.assertEqual(r["count"], 8)
            self.assertEqual(r["terminal_counts"], {"no_final_answer": 8})
            self.assertEqual(r["trajectories_sha256"], file_sha(out / "trajectories.jsonl"))
            with self.assertRaisesRegex(ValueError, "Different frozen contract"):
                collect_slots(out, {"different": True}, slots, resume, workers=1)


if __name__ == "__main__":
    unittest.main()
