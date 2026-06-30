#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# pi_setup.sh — one-shot setup for rl4cu RL training on a PrimeIntellect node
#
# Run once after SSHing in:
#   bash pi_setup.sh
#
# What it does:
#   1. Install system deps (git)
#   2. Clone VERL v0.6.0 and install editable
#   3. Install vLLM 0.8.5 (matches cuda 12.4 + torch 2.6 base image)
#   4. Clone this repo and install it
#   5. Download Qwen2.5-Coder-32B-Instruct to ~/hf_cache
#   6. Upload the rollout parquet from local machine (scp'd separately)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"   # set via: export HF_TOKEN=hf_xxx
MODAL_TOKEN_ID="${MODAL_TOKEN_ID:-}"
MODAL_TOKEN_SECRET="${MODAL_TOKEN_SECRET:-}"

echo "=== [1/6] system deps ==="
apt-get update -qq && apt-get install -y -qq git tmux htop nvtop

echo "=== [2/6] VERL v0.6.0 ==="
if [ ! -d /opt/verl ]; then
  git clone --branch v0.6.0 --depth 1 https://github.com/volcengine/verl /opt/verl
fi
pip install -q -e /opt/verl
python -c "import verl.trainer.main_ppo; print('verl OK')"

echo "=== [3/6] vLLM 0.8.5 ==="
VLLM_USE_V1=0 pip install -q "vllm==0.8.5"
VLLM_USE_V1=0 python -c "import vllm; print('vllm OK:', vllm.__version__)"

echo "=== [4/6] rl4cu repo ==="
cd /root
if [ ! -d rl4cu ]; then
  git clone https://github.com/rkim/rl4cu.git || {
    echo "NOTE: no public repo — copy manually with rsync/scp"
  }
fi
if [ -d /root/rl4cu ]; then
  pip install -q --no-deps -e /root/rl4cu/kernelbench 2>/dev/null || true
  pip install -q -e /root/rl4cu
fi
pip install -q pandas pyarrow wandb

echo "=== [5/6] Modal CLI (for reward eval) ==="
pip install -q modal
if [ -n "$MODAL_TOKEN_ID" ] && [ -n "$MODAL_TOKEN_SECRET" ]; then
  python -m modal token set \
    --token-id "$MODAL_TOKEN_ID" \
    --token-secret "$MODAL_TOKEN_SECRET"
  echo "Modal token configured"
else
  echo "WARNING: MODAL_TOKEN_ID/SECRET not set — reward evals will fail"
  echo "Run: python -m modal token set --token-id <id> --token-secret <secret>"
fi

echo "=== [6/6] Download Qwen2.5-Coder-32B-Instruct ==="
mkdir -p ~/hf_cache
export HF_HOME=~/hf_cache
if [ -n "$HF_TOKEN" ]; then
  python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    local_dir="/root/hf_cache/Qwen2.5-Coder-32B-Instruct",
    token=__import__("os").environ["HF_TOKEN"],
)
print("Model downloaded to ~/hf_cache/Qwen2.5-Coder-32B-Instruct")
PYEOF
else
  echo "WARNING: HF_TOKEN not set — model download skipped"
  echo "Run: export HF_TOKEN=hf_xxx && python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-32B-Instruct', local_dir='/root/hf_cache/Qwen2.5-Coder-32B-Instruct', token=__import__(\\\"os\\\").environ['HF_TOKEN'])\""
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. scp the rollout parquet:  scp -P 1234 /path/to/rollout_data_l1.parquet root@<IP>:~/rollout_data_l1.parquet"
echo "  2. scp the reward + trloo scripts (done by pi_launch.sh)"
echo "  3. Run:  bash /root/rl4cu/scripts/pi_launch.sh"
