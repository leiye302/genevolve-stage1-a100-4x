# GenEvolve Stage-1 SFT and 5,000 rollouts on 4× A100 80 GB

Standalone reproduction runner for [GenEvolve](https://arxiv.org/html/2605.21605v2). It trains the supervised cold start from **Qwen3-VL-8B-Instruct**, saves complete ZeRO-3 checkpoints, and samples trajectories from that run's **checkpoint-550**. It does not use the published, post-RL MeiGen-AI/GenEvolve weights as an initialization. No GRPO, Visual Experience Distillation, image renderer, image judge, or proprietary teacher runs here.

## What is reproduced

The [paper's Table 9](https://arxiv.org/html/2605.21605v2#A3.T9) and [official data](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench) specify the recipe. The [official agent release](https://github.com/MeiGen-AI/GenEvolve) does not provide the original SFT launcher; this repository reconstructs it using pinned LLaMA-Factory. All upstream revisions and SFT/rollout split identities are in [sources.json](sources.json).

| Setting | This runner |
| --- | --- |
| Backbone | Qwen3-VL-8B-Instruct, pinned public base revision |
| Data | Authors' exact 8,800 train / 200 eval; 50,291 original reference JPEGs |
| Objective | Full language-policy SFT. Freeze vision tower and multimodal projector. Mask system, user, tool observations and image tokens. Supervise all assistant turns, including explicit reasoning, tool calls and final JSON. |
| Context | 32,768 cutoff; no packing or completion-only data conversion; unchanged multimodal trajectories |
| Optimizer | AdamW, LR 1e-5, decay 1e-6, cosine, 2% warmup |
| Hardware adaptation | 4 GPUs × microbatch 2 × accumulation 4 = global 32; paper: 16 × 2 × 1 |
| Runtime | BF16, FA2, activation checkpointing, stock LLaMA-Factory ZeRO-3 with no CPU/NVMe offload |
| Schedule | 2 epochs × 275 optimizer updates = 550; held-out SFT loss and full checkpoint after each epoch |

The paper writes its assistant-token objective as $\mathcal{L}=-\sum_{i,t}m_{i,t}\log p_\theta(y_{i,t}\mid h_{i,t})/\sum_{i,t}m_{i,t}$. The LLaMA-Factory template named **qwen3_vl_nothink** uses the Instruct chat framing; it does **not** delete explicit assistant reasoning from the data. The released SFT rows already use user-role tool observations containing their native wrapper and image placeholders. We do not relabel these as assistant messages, regenerate the teachers, or train on GT images.

The exact package revisions, seed and image pixel bounds were not all specified by the authors. We use pinned public packages, seed 42 and the pinned trainer's default image bounds (1,024 to 589,824 pixels). The training entry uses the pinned LLaMA-Factory trainer and its normal causal loss, with no custom loss, reduction hook or optimizer override. The ZeRO-3 config matches the pinned upstream example. We preserve the per-device microbatch and increase only accumulation to preserve the effective batch on four cards. This does not imply bit-for-bit identity to the authors' 16-GPU run.

## Setup and run

Requirements: Linux x86-64, Python 3.11, **four allocated A100 80 GB cards**, CUDA 12.8 toolkit and a compatible NVIDIA driver, GCC/G++, git, and outbound access to GitHub/Hugging Face. Use a local SSD/NVMe volume with **400 GiB free before preparation**. The two complete ZeRO-3 checkpoints include FP32 optimizer states and are much larger than model-only exports. Do not put checkpoints in a Git repository. Full setup fetches about 7.4 GB of SFT data plus the base weights, and separately installs the training and serving environments.

~~~bash
git clone https://github.com/leiye302/genevolve-stage1-a100-4x.git
cd genevolve-stage1-a100-4x
# Only expose the four GPUs actually assigned to this job.
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/bootstrap.sh

# Check the contract without CUDA, model downloads, or a paid API call.
.venv-train/bin/python run.py sft --dry-run
.venv-rollout/bin/python run.py collect --dry-run

# Replace ./work with a writable volume that has sufficient free space.
.venv-train/bin/python run.py --work ./work prepare
.venv-train/bin/python run.py --work ./work sft

# Configure the live Serper API credential privately. Do not commit credentials.
read -rsp 'Serper API key: ' SERPER_API_KEY; echo
export SERPER_API_KEY
.venv-rollout/bin/python run.py --work ./work collect --allow-live-search
~~~

To run all three stages in order after setting the key, run:

~~~bash
GENEVOLVE_ALLOW_LIVE_SEARCH=1 bash scripts/run_all.sh ./work
~~~

The two search endpoints are checked before collection. Live collection incurs search requests and image downloads; no text-model API is required. Set an explicit trusted HTTPS **SERPER_BASE_URL** only for a compatible gateway. The key is not passed in process arguments or written to the run outputs. **--workers** (16 default) and **--search-rps** (4 default) control concurrency and search request rate; **--port** (18080) is the local vLLM port. The four GPUs are used for SFT first and for four vLLM data-parallel replicas afterwards, never both at once.

## Checkpoints and recovery

- **work/checkpoints/checkpoint-275** and **checkpoint-550** are full, same-world-size continuation checkpoints, not adapter-only exports. They include HF weights/tokenizer/processor, trainer state, RNG files for all four ranks, DeepSpeed model and optimizer partitions, and scheduler state.
- **work/integrity/checkpoint-*.json** records SHA-256 and byte size for each file. Collection requires the attested final checkpoint and checks the full file set; the vLLM service also reports the exact checkpoint path under a digest-derived model alias. The base, SFT data/asset inventory and effective recipe have a separate **prepared.json** / **run_contract.json** receipt.
- To resume SFT after epoch 1, use the same work directory, pins and four GPUs: **run.py --work ./work sft --resume** in the training environment. An interrupted, unattested final save is preserved under **work/interrupted/** only after the epoch-1 checkpoint is checked. A previously attested final checkpoint with changed bytes is an error, not overwritten. Do not point this launcher at a different model or edit an existing run contract.
- After a collection interruption, re-run the same **collect** command. Complete atomic per-slot records and their images are checked and not sampled again. Failed transport attempts are logged separately without vendor error bodies. No paid requests are made just to re-verify an already complete collection.

## Rollout contract and outputs

The 2,575 distinct prompts are taken only from the authors' **GenEvolve-Data-RL/train.jsonl**, with no evaluation split or GT-image path sent to the Actor. Fixed seed 2026 schedules one draw per prompt, then a second draw for 2,425 prompts. Each slot records its own seed and source ID. This covers the pool and produces exactly 5,000 attempts; it is not the paper's six-per-question GRPO group formation.

The collection uses temperature 0.7, top-p 0.95, top-k disabled, at most 11 Actor calls, initial prompt cap 6,144, total generated-token cap 30,000, and up to 2 selected references. Per-call cap 4,096 comes from the public runtime. We use the exact system prompt common to the released SFT rows so this is a rollout of the cold-start policy under its supervised protocol, not the public post-RL inference prompt. Native **search**, **image_search**, **query_knowledge**, and the upstream parser/final-step instruction stay in the loop. The public image tool's metadata-only response is supplemented with real image bytes in the same **&lt;image&gt;** positions as SFT. It asks for up to 5 search results, bounds downloads and saves the exact JPEGs delivered to the Actor. Live search results can drift even with a fixed model seed.

- **work/rollouts/plan.json**: frozen 5,000 slot IDs, prompts, seeds and checkpoint/recipe identity.
- **work/rollouts/episodes/**: one atomically written episode per slot. Includes assistant output and reasoning as generated, tool calls and observations, ordered image assets, program, and terminal status.
- **work/rollouts/images/**: content-addressed JPEGs supplied to the Actor, referenced by relative path.
- **work/rollouts/trajectories.jsonl** and **manifest.json**: ordered final archive, checksum, valid-program count and failure counts. A malformed model answer or model budget exhaustion is an actual sampled trajectory and is retained with its status. A search/API/serving infrastructure failure is not a completed trajectory; repair it and resume.
- The rollout ends at the prompt-reference program. It saves actual retrieved images, **not** 5,000 rendered final images. It neither renders the final program nor scores image quality.

## CPU verification

~~~bash
python -m unittest discover -s tests -v
python run.py sft --dry-run
python run.py collect --dry-run
bash -n scripts/bootstrap.sh && bash -n scripts/run_all.sh
~~~

These checks require no GPU, no private files and no API. They check the schedule, checkpoint layout and SHA, image-to-message wiring, exact rollout plan and pause/resume behavior. They are not a claim of completed four-GPU training, live search, or generated-image evaluation. Upstream model/data/tools retain their own licenses and terms; the wrapper does not redistribute their weights, reference images, or other artifacts.
