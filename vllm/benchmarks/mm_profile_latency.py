#!/usr/bin/env python3
"""mm_profile_latency.py

使用 OpenAI Chat Completions 接口，对已经运行的 vLLM 服务进行多模态
（文本 + 单张图片）基准测试，自动控制 torch.profiler 的启停。

示例：
    python3 -m vllm.benchmarks.mm_profile_latency \
      --image-path apple.jpg \
      --api-url http://127.0.0.1:25000 \
      --model Qwen2.5-VL-32B-Instruct \
      --num-iters 10 \
      --out-json result.json

在运行脚本前，请确保：
1. 服务通过 sglang.launch_server 启动，监听 /v1/chat/completions，且
   export VLLM_TORCH_PROFILER_DIR 指向可写目录。
2. 模型支持多模态输入（如 Qwen-VL, Qwen2-VL, CogVLM 等）。
"""
from __future__ import annotations

# fmt: off
import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import List

import aiohttp
import numpy as np
from PIL import Image

# 使用 Qwen 官方工具处理图片
from qwen_vl_utils import fetch_image  # type: ignore
# fmt: on

# -------------------------- 工具函数 --------------------------

def encode_image_to_base64(img_path: str | Path, resize: int | None = None) -> str:
    """用 fetch_image 打开并标准化图片，再转 base64 字符串。"""
    img_path = Path(img_path)
    if not img_path.is_file():
        raise FileNotFoundError(img_path)

    # 让 fetch_image 做统一的缩放 / 归一化处理
    img = fetch_image({
        "type": "image",
        "image": str(img_path.resolve()),  # 直接传路径
        "min_pixels": 28 * 28 * 4,
        "max_pixels": 512 * 512 if resize is None else resize * resize,
    })

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_payload(prompt: str, img_b64: str, model: str, max_tokens: int) -> dict:
    """组装 OpenAI ChatCompletions JSON 负载。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }


# -------------------------- 主流程 --------------------------

async def _single_request(session: aiohttp.ClientSession, endpoint: str, payload: dict, headers: dict, timeout: int) -> float:
    """发送一次请求并返回耗时（毫秒）。"""
    start = time.perf_counter()
    async with session.post(endpoint, json=payload, headers=headers, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            sys.stderr.write(f"Request failed: {resp.status} {text}\n")
    return (time.perf_counter() - start) * 1000


async def run_benchmark(
    api_url: str,
    model: str,
    img_path: str | Path,
    prompt: str,
    num_iters: int,
    max_tokens: int,
    concurrency: int,
    timeout: int = 300,
) -> List[float]:
    """并发执行 num_iters 次请求，返回毫秒级延迟列表。"""

    img_b64 = encode_image_to_base64(img_path)
    payload = build_payload(prompt, img_b64, model, max_tokens)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer None",
    }
    endpoint = api_url.rstrip("/") + "/v1/chat/completions"

    latencies: List[float] = []

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def bound_req():
            async with sem:
                lat = await _single_request(session, endpoint, payload, headers, timeout)
                latencies.append(lat)

        tasks = [asyncio.create_task(bound_req()) for _ in range(num_iters)]
        await asyncio.gather(*tasks)

    return latencies


def control_profiler(api_url: str, action: str) -> None:
    """调用 /start_profile 或 /stop_profile 控制 torch.profiler."""
    assert action in {"start", "stop"}
    url = api_url.rstrip("/") + f"/{action}_profile"
    resp = requests.post(url, timeout=10)
    if resp.status_code != 200:
        sys.stderr.write(f"Failed to {action} profiler: {resp.status_code} {resp.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser("vLLM multimodal latency profiler")
    parser.add_argument("--image-path", required=True, help="输入测试图片路径")
    parser.add_argument(
        "--api-url", default="http://127.0.0.1:25000", help="vLLM API Server 基址"
    )
    parser.add_argument("--model", required=True, help="OpenAI 请求里的 model 字段")
    parser.add_argument("--num-iters", type=int, default=10, help="请求总次数")
    parser.add_argument(
        "--prompt",
        default="Please describe this image in detail.",
        help="用户文本提示 (会放在图片前面)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="生成最大 token 数"
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="结果保存路径 (可选)"
    )
    parser.add_argument("--concurrency", type=int, default=10, help="并发请求数 (同时在飞) ")

    args = parser.parse_args()

    print("Starting torch.profiler on server ...")
    control_profiler(args.api_url, "start")

    print(f"Running {args.num_iters} requests ...")
    latencies = asyncio.run(
        run_benchmark(
            api_url=args.api_url,
            model=args.model,
            img_path=args.image_path,
            prompt=args.prompt,
            num_iters=args.num_iters,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    )

    print("Stopping torch.profiler on server ...")
    control_profiler(args.api_url, "stop")

    lat_arr = np.array(latencies)
    avg = float(lat_arr.mean())
    ptiles = {p: float(np.percentile(lat_arr, p)) for p in [50, 90, 95, 99]}

    print(f"Avg latency: {avg:.2f} ms")
    for p, v in ptiles.items():
        print(f"p{p}: {v:.2f} ms")

    if args.out_json:
        out = {
            "latencies_ms": latencies,
            "avg_ms": avg,
            "percentiles_ms": ptiles,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main() 