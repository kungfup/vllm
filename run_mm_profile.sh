#!/usr/bin/env bash
# ----------------------------------------------
# run_mm_profile.sh
# 一键启动 Qwen2.5-VL-32B 服务并抓取多模态 profiler Trace。
# 1. 按需修改下方 CONFIG 区域的模型路径 / GPU / 并发参数等。
# 2. 运行：  bash run_mm_profile.sh
# 3. Trace 输出在 $VLLM_TORCH_PROFILER_DIR，可用 TensorBoard 查看。
# ----------------------------------------------

set -euo pipefail

#################################
# --------- CONFIG -------------
#################################
MODEL_PATH="/mnt/bn/vllmlfdata/yzh/model/Qwen/Qwen2.5-VL-32B-Instruct/"
PORT=25000
CUDA_VISIBLE_DEVICES="0,1"
PP_SIZE=2
TP_SIZE=1
CTX_LEN=32768
PREFILL=32768
MAX_RUNNING_REQ=8
MEM_STATIC=0.9
QUANT="fp8"
MM_BACKEND="fa3"
CHAT_TEMPLATE="qwen2-vl"

# Profiler & benchmark
TRACE_DIR="/tmp/vllm_trace"
IMAGE_PATH="apple.jpg"       # 测试图片
NUM_ITERS=100                 # 请求总数
CONCURRENCY=20                # 并发度
MAX_TOKENS=256                # 生成上限
#################################

export VLLM_TORCH_PROFILER_DIR="$TRACE_DIR"
mkdir -p "$TRACE_DIR"

LOG_FILE="launch_${PORT}.log"
CMD="unset NCCL_TOPO_FILE; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --port $PORT \
  --pipeline-parallel-size $PP_SIZE \
  --tensor-parallel-size $TP_SIZE \
  --max-model-len $CTX_LEN \
  --enable-chunked-prefill \
  --max-num-seqs $MAX_RUNNING_REQ \
  --gpu-memory-utilization $MEM_STATIC \
  --chat-template $CHAT_TEMPLATE \
  --trust-remote-code \
  --disable-mm-preprocessor-cache \
  --quantization $QUANT \
  --mm-attention-backend $MM_BACKEND > $LOG_FILE 2>&1 &"

echo "Launching server… (log: $LOG_FILE)"
eval "$CMD"
SERVER_PID=$!

# ---------- wait until port open ----------
function wait_port() {
  local host=$1; local port=$2; local max_try=60; local i=0
  while ! nc -z "$host" "$port"; do
    i=$((i+1))
    if [[ $i -ge $max_try ]]; then
      echo "Port $port not ready after $max_try tries." >&2; exit 1;
    fi
    echo "Waiting for $host:$port … ($i)"; sleep 2;
  done
}

wait_port 127.0.0.1 "$PORT"
echo "Server is ready. Starting benchmark…"

python3 -m vllm.benchmarks.mm_profile_latency \
  --image-path "$IMAGE_PATH" \
  --api-url "http://127.0.0.1:$PORT" \
  --model "$(basename "$MODEL_PATH")" \
  --num-iters "$NUM_ITERS" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --out-json "profile_result.json"

echo "Benchmark finished. Trace saved to $TRACE_DIR"

echo "Stopping server (pid=$SERVER_PID)…"
kill "$SERVER_PID" || true

# ---------- GPU 占位 ----------
echo "Running GPU occupant script…"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python auto_qps/gpu_occ.py || true
echo "gpu_occ.py finished (or not found)." 