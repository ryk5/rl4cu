"""
SFT training job on Modal.

Runs VERL's multi-turn SFT trainer to fine-tune Qwen2.5-Coder-32B-Instruct
with LoRA on the generated CUDA kernel trajectories.

Usage (from repo root):
    # upload data first, then train
    modal run modal_app/sft_train.py

    # or with custom args
    modal run modal_app/sft_train.py::train \
        --data-path data/sft/verl_sft_train.parquet \
        --val-data-path data/sft/verl_sft_val.parquet \
        --run-name my_run
"""

import pathlib
import modal

# ── image ─────────────────────────────────────────────────────────────────────
# VERL requires torch + flash-attn + a recent transformers.
# We pin flash-attn to a pre-built wheel to avoid long compile times.

VERL_IMAGE = (
    # official VERL base image: torch 2.6, flash-attn 2.7.4, CUDA 12.4, cuDNN 9.8
    modal.Image.from_registry(
        "verlai/verl:base-verl0.4-cu124-cudnn9.8-torch2.6-fa2.7.4",
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y -qq git",
        "git clone --branch v0.6.0 --depth 1 https://github.com/volcengine/verl /opt/verl && pip install -e /opt/verl && python -c 'import verl.trainer.sft_trainer; print(\"verl OK\")'",
    )
    .pip_install(
        "pandas",
        "pyarrow",
        "wandb",
    )
)

app = modal.App("rl4cu-sft")

# persistent volume for model weights + checkpoints
volume = modal.Volume.from_name("rl4cu-weights", create_if_missing=True)
VOLUME_MOUNT = "/vol"
CHECKPOINTS_DIR = f"{VOLUME_MOUNT}/sft_checkpoints"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"

# ── config defaults ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_RUN_NAME = "qwen25coder32b_sft_lora"

# LoRA config — rank 16, target q/k/v/o projections
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# training
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1       # 32B model, 4×H100 80GB
GRAD_ACCUM_STEPS = 4            # effective batch = 4 × 4 = 16
MAX_SEQ_LEN = 8192
WARMUP_STEPS = 50


@app.function(
    image=VERL_IMAGE,
    gpu="H100:4",
    timeout=60 * 60 * 6,   # 6 hours
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    data_path: str = "data/sft/verl_sft_train.parquet",
    val_data_path: str = "data/sft/verl_sft_val.parquet",
    run_name: str = DEFAULT_RUN_NAME,
    lora_rank: int = LORA_RANK,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
):
    """
    Run VERL SFT with LoRA on 4×H100.

    Data files are uploaded from local via Modal's add_local_file at call time.
    Model weights are downloaded from HuggingFace and cached on the volume.
    """
    import os
    import subprocess

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_dir = f"{CHECKPOINTS_DIR}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # VERL v0.6.0 SFT uses torchrun for multi-GPU FSDP
    # Config keys from: verl/trainer/config/sft_trainer_engine.yaml @ v0.6.0
    # model comes from model@model: hf_model -> field is model.path
    # optim comes from optim@optim: fsdp -> field is optim.lr, optim.lr_warmup_steps_ratio
    # Must run from /opt/verl so Hydra finds the config/ directory
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        "-m", "verl.trainer.sft_trainer",
        # model — field is 'path' inside hf_model config
        f"model.path={MODEL_ID}",
        f"model.enable_gradient_checkpointing=True",
        # LoRA
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={lora_rank * 2}",
        f"model.target_modules=all-linear",
        # data (from sft_trainer_engine.yaml)
        f"data.train_files={data_path}",
        f"data.val_files={val_data_path}",
        f"data.messages_key=messages",
        f"data.max_length={MAX_SEQ_LEN}",
        f"data.truncation=right",
        f"data.use_dynamic_bsz=True",
        f"data.max_token_len_per_gpu={MAX_SEQ_LEN}",
        f"data.micro_batch_size_per_gpu={PER_DEVICE_BATCH_SIZE}",
        f"data.train_batch_size=8",
        # training
        f"trainer.total_epochs={num_epochs}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.project_name=rl4cu",
        f"trainer.experiment_name={run_name}",
        f'trainer.logger=["console"]',
        # optimizer (from optim/fsdp.yaml — uses lr_warmup_steps_ratio)
        f"optim.lr={learning_rate}",
        f"optim.lr_warmup_steps_ratio=0.05",
        # checkpoint
        "checkpoint.save_contents=[model,optimizer,extra]",
    ]

    print("launching VERL SFT:")
    print(" ".join(cmd))
    print()

    env = os.environ.copy()
    # ensure verl source is on path for all torchrun child processes
    env["PYTHONPATH"] = "/opt/verl:" + env.get("PYTHONPATH", "")

    # run from /opt/verl so Hydra finds the config/ directory
    result = subprocess.run(cmd, check=False, env=env, cwd="/opt/verl")
    if result.returncode != 0:
        raise RuntimeError(f"SFT training failed with exit code {result.returncode}")

    print(f"\ntraining complete. checkpoints at: {output_dir}")
    volume.commit()
    return output_dir


@app.function(
    image=VERL_IMAGE,
    gpu="H100",
    timeout=60 * 30,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_lora(checkpoint_dir: str, output_name: str = "merged"):
    """
    Merge LoRA adapter weights into the base model.
    Saves the merged model to the volume for use in RL training.
    """
    import os
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    os.environ["HF_HOME"] = HF_CACHE_DIR

    merged_dir = f"{CHECKPOINTS_DIR}/{output_name}"
    os.makedirs(merged_dir, exist_ok=True)

    print(f"loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"loading LoRA adapter from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base, checkpoint_dir)

    print("merging...")
    model = model.merge_and_unload()

    print(f"saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    volume.commit()
    print(f"done. merged model at: {merged_dir}")
    return merged_dir


@app.local_entrypoint()
def main(
    data_path: str = "data/sft/verl_sft_train.parquet",
    val_data_path: str = "data/sft/verl_sft_val.parquet",
    run_name: str = DEFAULT_RUN_NAME,
):
    import pathlib

    # check local files exist before uploading
    for p in [data_path, val_data_path]:
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/format_sft_dataset.py first."
            )

    # upload parquet files to the Modal volume so training can read them
    vol_data_dir = f"{VOLUME_MOUNT}/sft_data"
    vol_train = f"{vol_data_dir}/verl_sft_train.parquet"
    vol_val   = f"{vol_data_dir}/verl_sft_val.parquet"

    print("uploading SFT data to Modal volume...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(data_path, "/sft_data/verl_sft_train.parquet")
        batch.put_file(val_data_path, "/sft_data/verl_sft_val.parquet")
    print("upload complete.")

    print(f"starting SFT run: {run_name}")
    print(f"  train: {vol_train}")
    print(f"  val:   {vol_val}")
    print(f"  model: {MODEL_ID}")
    print(f"  lora:  rank={LORA_RANK}, targets={LORA_TARGET_MODULES}")
    print()

    output_dir = train.remote(
        data_path=vol_train,
        val_data_path=vol_val,
        run_name=run_name,
    )
    print(f"SFT complete. checkpoints at: {output_dir}")
