import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import random
import sys
import time
import traceback  # for error formatting
import itertools
from concurrent import futures
from queue import Queue
# from random import randint, random
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import aiohttp
import numpy as np
import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm
from qwen_vl_utils import fetch_image
import io
import json
import socket
import subprocess
import signal

# os.environ["MAX_CASE_ITEMS_CNT"] = "30"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

AIOHTTP_TIMEOUT = 10000

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0

async def send_request(session, url, data, headers):
    async with session.post(url=url, json=data, headers=headers) as response:
        return await response.json()


async def async_request_openai_chat_completions(
    prompts,
    model_name,
    api_url_list,
    api_key,
    max_out_len=1000,
    multi_modal_content=None,
):
    api_url = random.choice(api_url_list)
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession() as session:
        tasks = []
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        st = time.perf_counter()
        for prompt in prompts:
            content = [{"type": "text", "text": prompt}]
            if multi_modal_content:
                content += multi_modal_content
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
                "max_tokens": max_out_len,
                "stream": False,
                "top_k": 1,
                "top_p":0.001,
                "ignore_eos":True
            }
            tasks.append(send_request(session, api_url, payload, headers))
        responses = await asyncio.gather(*tasks)
        timestamp = time.perf_counter()
        prompt_tokens = 0
        generate_tokens = 0
        for o in responses:
            try:
                prompt_tokens += o["usage"]["prompt_tokens"]
                generate_tokens += o["usage"]["completion_tokens"]
            except Exception as e:
                logging.info("bad stat")
                continue
        prompt_tokens = prompt_tokens / len(prompts)
        generate_tokens = generate_tokens / len(prompts)
        latency = (timestamp - st) * 1000
        # logging.info(f"prompt_tokens: {latency}")
        return latency, prompt_tokens, generate_tokens


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    low_bound = request_rate - request_rate * 0.2
    high_bound = request_rate + request_rate * 0.2
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # interval = np.random.uniform(1.0 / high_bound, 1.0 / low_bound, size=[1])[0]
        # interval = np.random.uniform(1.0 / request_rate-0.2, 1.0 / request_rate+0.2, size=[1])[0]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def benchmark(
    input_requests, request_rate, model_name, api_url_list, api_key, max_out_len, profile=False
):
    base_url = 'http://127.0.0.1:25000'
    if profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=base_url + "/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompts, mm_content = request
        tasks.append(
            asyncio.create_task(
                async_request_openai_chat_completions(
                    prompts=prompts,
                    model_name=model_name,
                    api_url_list=api_url_list,
                    api_key=api_key,
                    max_out_len=max_out_len,
                    multi_modal_content=mm_content,
                )
            )
        )
    outputs: List[float] = await asyncio.gather(*tasks)
    if profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    return outputs

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def resize_image_to_base64(imgpath):
    base64_image = encode_image(imgpath)
    img = fetch_image({"type": "image",
                        "image": f"data:image/jpeg;base64,{base64_image}",
                        "min_pixels": 28 * 28 * 4,  # 最小 4 个 image patch
                        "max_pixels": 512 * 512})
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_bytes = img_bytes.getvalue()
    img_base64 = base64.b64encode(image_bytes)
    # 将Base64字节数据转换为字符串
    img_base64_str = img_base64.decode('utf-8')
    return img_base64_str

def get_inputs(input_len=4096, num_samples=300):
    # TODO 需要check decoding 长度是不是满足要求
    image_path = 'apple.jpg'
    input_requests = []
    image_tokens = 326
    # num_images = min(input_len // image_tokens, 20)
    num_images = 1
    space_tokens = input_len - num_images * image_tokens -123
    question = "A "* space_tokens +"please describe thoses image one by one in details, "+"<image>" * num_images
    is_multi = num_images > 1
    mm_content = []
    base64_image = resize_image_to_base64(image_path)
    url = f"data:image/jpeg;base64,{base64_image}"
    for image_path in range(num_images):
        mm_content_tmp = {
            "type": "image_url",
            "image_url": {"url": url},
            # "modalities": "multi-images" if is_multi else "image",
        }
        mm_content.append(mm_content_tmp)
    input_requests.append([[question], mm_content])
    input_requests = input_requests * num_samples
    return input_requests

async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession() as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def check_port_health(host, port, max_retries=3, retry_interval=5):
    """
    检查端口是否可用，支持重试机制
    :param host: 目标主机
    :param port: 目标端口
    :param max_retries: 最大重试次数
    :param retry_interval: 重试间隔(秒)
    :return: bool 是否健康
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except Exception as e:
            print(f"端口检查异常: {str(e)}")
        
        retry_count += 1
        if retry_count < max_retries:
            print(f"端口 {port} 不可达，第 {retry_count} 次重试...")
            time.sleep(retry_interval)
    
    return False


def main(
    request_rate,
    model_name,
    api_url_list,
    api_key,
    max_out_len,
    input_len=4096,
    num_samples=200,
    profile=False,
    service_port: int = 8000,
):
    # 等待对应服务端口就绪
    while not check_port_health("127.0.0.1", service_port):
        print("端口不可用，等待10秒后重试...")
        time.sleep(10)
    print("request_rate", request_rate)
    print("max_out_len", max_out_len)
    print("input_len", input_len)
    print("num_samples", num_samples)


    input_requests = get_inputs(num_samples=num_samples, input_len=input_len)

    input_requests = input_requests
    time_start = time.perf_counter()
    benchmark_result = asyncio.run(
        benchmark(
            input_requests=input_requests[:10],
            request_rate=request_rate,
            model_name=model_name,
            api_url_list=api_url_list,
            api_key=api_key,
            max_out_len=max_out_len,
        )
    )
    time_end = time.perf_counter()
    true_qps = len(input_requests) / (time_end - time_start)
    print(f"Total time: {(time_end - time_start) * 1000}")
    print(f"True QPS: {true_qps}")
    benchmark_prompt_tokens = [i[1] for i in benchmark_result]
    benchmark_generate_tokens = [i[2] for i in benchmark_result]
    print(f"Total prompt tokens: {np.sum(benchmark_prompt_tokens)}")
    print(f"Total generate tokens: {np.sum(benchmark_generate_tokens)}")
    print(f"Avg prompt tokens: {np.mean(benchmark_prompt_tokens)}")
    print(f"Avg generate tokens: {np.mean(benchmark_generate_tokens)}")
    benchmark_latency = [i[0] for i in benchmark_result]
    latencies = np.array(benchmark_latency)
    percentages = [10, 25, 50, 75, 90, 95, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} ms")
    avg_latency = np.mean(latencies)
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} ms")
        
    time_start = time.perf_counter()

    benchmark_result = asyncio.run(
        benchmark(
            input_requests=input_requests,
            request_rate=request_rate,
            model_name=model_name,
            api_url_list=api_url_list,
            api_key=api_key,
            max_out_len=max_out_len,
            profile=profile
        )
    )
    time_end = time.perf_counter()
    result_dict = {}
    true_qps = len(input_requests) / (time_end - time_start)
    print(f"Total time: {(time_end - time_start) * 1000}")
    print(f"True QPS: {true_qps}")
    benchmark_prompt_tokens = [i[1] for i in benchmark_result]
    benchmark_generate_tokens = [i[2] for i in benchmark_result]
    print(f"Total prompt tokens: {np.sum(benchmark_prompt_tokens)}")
    print(f"Total generate tokens: {np.sum(benchmark_generate_tokens)}")
    print(f"Avg prompt tokens: {np.mean(benchmark_prompt_tokens)}")
    print(f"Avg generate tokens: {np.mean(benchmark_generate_tokens)}")
    benchmark_latency = [i[0] for i in benchmark_result]
    latencies = np.array(benchmark_latency)
    percentages = [10, 25, 50, 75, 90, 95, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} ms")
    avg_latency = np.mean(latencies)
    result_dict['true_qps'] = true_qps
    result_dict['avg_latency'] = avg_latency
    result_dict['avg_prompt_tokens'] = np.mean(benchmark_prompt_tokens)
    result_dict['avg_generate_tokens'] = np.mean(benchmark_generate_tokens)
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} ms")
        result_dict[f'{percentage}%_latency'] = percentile
    return result_dict


def build_cmd(model_path: str, serv_cfg: dict) -> str:
    """根据 services_config 拼接启动 SGLang / vLLM 服务器的命令字符串。

    参数示例::
        serv_cfg = {
            "port": 25000,
            "pp_size": 2,
            "tp_size": 1,
            "quantization": "fp8",
            "kv_cache_dtype": "None",
            "mem-fraction-static": 0.70,
            "context-length": 32768,
            "max-prefill-tokens": 32768,
            "max-running-requests": 8,
            "CUDA_VISIBLE_DEVICES": [0,1]
        }
    """

    # 1. GPU 可见性
    gpu_list = serv_cfg.get("CUDA_VISIBLE_DEVICES", [])
    cuda_env = ""
    if gpu_list:
        cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_list))} "

    # 2. 基本必需参数
    cmd_parts = [
        # 使用 vLLM 的 OpenAI 兼容服务器
        "nohup python3 -m vllm.entrypoints.openai.api_server",
        # 必需参数
        f"--model {model_path}",
        f"--port {serv_cfg.get('port', 26000)}",
        f"--pipeline-parallel-size {serv_cfg.get('pp_size', 1)}",
        f"--tensor-parallel-size {serv_cfg.get('tp_size', 1)}",
        f"--max-model-len {serv_cfg.get('context-length', 32768)}",

        # Scheduler / KV cache
        "--enable-chunked-prefill",  # 显式开启 Chunked Prefill
        f"--max-num-seqs {serv_cfg.get('max-running-requests', 8)}",
        f"--gpu-memory-utilization {serv_cfg.get('mem-fraction-static', 0.8)}",

        # 模型相关
        "--chat-template qwen2-vl",
        "--trust-remote-code",
        "--disable-mm-preprocessor-cache",
    ]

    # 3. 可选量化 / kv-cache dtype
    q = serv_cfg.get("quantization", None)
    if q and str(q).lower() != "none":
        cmd_parts += ["--quantization", str(q)]

    kv = serv_cfg.get("kv_cache_dtype", None)
    if kv and str(kv).lower() != "none":
        cmd_parts += ["--kv-cache-dtype", str(kv)]

    # 4. split (FA3/num?)
    split_v = serv_cfg.get("split", None)
    # 若传入的是列表，仅取首元素；若仍为 None 或 'none'，则忽略
    if isinstance(split_v, (list, tuple)):
        split_v = split_v[0] if split_v else None
    if split_v is not None and str(split_v).lower() != "none":
        cmd_parts += ["--split", str(split_v)]

    # 5. mm-attention-backend
    mm_backend = serv_cfg.get("mm-attention-backend", None)
    if mm_backend:
        cmd_parts += ["--mm-attention-backend", str(mm_backend)]

    log_file = serv_cfg.get("log_file_path", "nohup.out")
    # 将 NCCL_TOPO_FILE 置空，避免 NCCL 拓扑文件导致的潜在错误
    cmd = (
        "unset NCCL_TOPO_FILE; "
        + cuda_env
        + " ".join(cmd_parts)
        + f" > {log_file} 2>&1 &"
    )
    return cmd

# ---------- helper utils ----------


def wait_port_ready(host: str, port: int, timeout: int = 120):
    """Block until TCP port is open or timeout (s)."""
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(2)
    return False


def kill_gpu_processes(gpus: list[int]):
    """Kill sglang processes running on specified GPU indices only."""
    if not gpus:
        return
    gpu_pat = "|".join(str(g) for g in gpus)
    cmd = (
        "nvidia-smi | grep 'python' | grep -E ' "
        f"(:?{gpu_pat}) "
        "' | awk '{print $5}'"
    )
    try:
        pids = subprocess.check_output(cmd, shell=True).decode().strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    except subprocess.CalledProcessError:
        pass


def launch_service(model_path: str, serv_cfg: dict):
    """Kill old procs on GPUs then start new server and wait port ready."""
    gpus = serv_cfg.get("CUDA_VISIBLE_DEVICES", [])
    kill_gpu_processes(gpus)
    cmd = build_cmd(model_path, serv_cfg)
    banner = (
        "=" * 55
        + f"\nLaunching service with config: port={serv_cfg['port']}, pp={serv_cfg['pp_size']}, tp={serv_cfg['tp_size']}, "
          f"split={serv_cfg.get('split')}, mem={serv_cfg.get('mem-fraction-static')}, "
          f"cl={serv_cfg.get('context-length')}, prefill={serv_cfg.get('max-prefill-tokens')}\n"
        + f"Results will be saved to: {save_path_global}\n"
        + f"Service log file: {serv_cfg.get('log_file_path')}\n"
        + f"Service command file: {serv_cfg.get('cmd_file_path')}\n"
        + "=" * 55
    )
    logging.info(banner)
    logging.info(f"Command: {cmd}")
    subprocess.Popen(cmd, shell=True)
    port = serv_cfg.get("port", 8000)
    if not wait_port_ready("127.0.0.1", port, timeout=180):
        logging.error("Port %s not ready after timeout", port)
        return False
    return True


# ---------------- GPU OCC fallback -----------------


def run_gpu_occ(gpus: list[int]):
    """Run gpu_occ.py on the specified GPU list in foreground."""
    if not gpus:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    logging.info(
        f"Launch gpu_occ.py on GPUs {env['CUDA_VISIBLE_DEVICES']} as fallback occupant."
    )
    subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "gpu_occ.py")],
        env=env,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto QPS VLLM script")
    parser.add_argument("--config", type=str, default="config_vllm.json", help="Path to config json")
    args = parser.parse_args()

    with open(args.config,'r') as r:
        config_dict = json.load(r)
    services_config = config_dict['services_config']
    data_config = config_dict['data_config']
    model_path = config_dict['model_path']

    # ----- 提取各参数列表（如仅给单值，也统一转成 list） -----
    def ensure_list(val, default):
        if val is None:
            return [default]
        return val if isinstance(val, list) else [val]

    quant_list   = ensure_list(services_config.get('quantization'), [None])
    kv_list      = ensure_list(services_config.get('kv_cache_dtype'), [None])
    pp_list      = ensure_list(services_config.get('pp_size'), [1])
    tp_list      = ensure_list(services_config.get('tp_size'), [1])
    split_list   = ensure_list(services_config.get('split'), [None])
    mem_list     = ensure_list(services_config.get('mem-fraction-static'), [None])
    ctx_list     = ensure_list(services_config.get('context-length'), [32768])
    prefill_list = ensure_list(services_config.get('max-prefill-tokens'), [32768])
    mm_backend_list = ensure_list(services_config.get('mm-attention-backend'), [None])
    port_list    = ensure_list(services_config.get('port'), [8000])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir  = os.path.join(base_dir, 'results_vllm')
    logs_dir     = os.path.join(base_dir, 'service_logs')
    cmds_dir     = os.path.join(base_dir, 'service_cmds')
    qps_logs_dir = os.path.join(base_dir, 'qps_log')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(cmds_dir, exist_ok=True)
    os.makedirs(qps_logs_dir, exist_ok=True)

    gpus_for_service = services_config.get('CUDA_VISIBLE_DEVICES', [])

    try:
        # 遍历所有组合
        for (quantization, kv_cache_dtype, pp_size, tp_size, split_value, mem_fraction_static, \
             ctx_len, prefill_tok, mm_backend, port) in itertools.product(
                quant_list, kv_list, pp_list, tp_list, split_list, mem_list, ctx_list, prefill_list, mm_backend_list, port_list):

            # -------- 生成文件名片段 --------
            filename_parts = [f"port-{port}", f"pp-{pp_size}", f"tp-{tp_size}"]
            if split_value is not None and str(split_value).lower() != 'none':
                filename_parts.append(f"split-{split_value}")
            if mem_fraction_static is not None and str(mem_fraction_static).lower() != 'none':
                filename_parts.append(f"mem-{mem_fraction_static}")
            filename_parts.append(f"cl-{ctx_len}")
            filename_parts.append(f"prefill-{prefill_tok}")

            save_path = os.path.join(results_dir, f"result_{'_'.join(filename_parts)}.json")
            log_file_path = os.path.join(logs_dir,   f"servlog_{'_'.join(filename_parts)}.log")
            cmd_file_path = os.path.join(cmds_dir,   f"cmd_{'_'.join(filename_parts)}.sh")
            qps_log_path = os.path.join(qps_logs_dir, f"qps_{'_'.join(filename_parts)}.log")

            # 全局路径供 banner 使用
            global save_path_global
            save_path_global = save_path

            mrr_raw = services_config.get("max-running-requests", 8)
            max_running_req = mrr_raw[0] if isinstance(mrr_raw, list) else mrr_raw

            serv_cfg_full = {
                "port": port,
                "pp_size": pp_size,
                "tp_size": tp_size,
                "quantization": quantization,
                "kv_cache_dtype": kv_cache_dtype,
                "mem-fraction-static": mem_fraction_static,
                "context-length": ctx_len,
                "max-prefill-tokens": prefill_tok,
                "max-running-requests": max_running_req,
                "CUDA_VISIBLE_DEVICES": gpus_for_service,
                "split": split_value,
                "mm-attention-backend": mm_backend,
                "log_file_path": log_file_path,
                "cmd_file_path": cmd_file_path,
            }

            # 写命令文件（若不存在或内容变化）
            cmd_str_to_save = build_cmd(model_path, serv_cfg_full)
            if (not os.path.exists(cmd_file_path)) or (open(cmd_file_path).read().strip() != cmd_str_to_save):
                with open(cmd_file_path, 'w') as f:
                    f.write(cmd_str_to_save + '\n')
                try:
                    os.chmod(cmd_file_path, 0o750)
                except Exception:
                    pass

            # 为当前组合添加单独的 QPS 日志 handler
            qps_file_handler = logging.FileHandler(qps_log_path)
            qps_file_handler.setLevel(logging.INFO)
            qps_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(qps_file_handler)

            # 启动 / 重启服务
            if not launch_service(model_path, serv_cfg_full):
                logging.error("Failed to launch service for current configuration, skip.")
                continue

            api_url_list = [f"http://127.0.0.1:{port}/v1/chat/completions"]

            # 压测循环（保持原逻辑不变）
            qps_gap = data_config['qps_gap']
            for idx, input_len in enumerate(data_config['input_len']):
                running_qps = data_config['qps_start'][idx]
                prev_qps = running_qps
                for num_samples in data_config['num_samples']:
                    for max_out_len in data_config['max_out_len']:
                        flag=True
                        hit_top=False
                        try:
                            while flag:
                                state_info = main(
                                    request_rate=running_qps,
                                    model_name=model_path,
                                    api_url_list=api_url_list,
                                    api_key="None",
                                    max_out_len=max_out_len,
                                    input_len=input_len,
                                    num_samples=200,
                                    profile=False,
                                    service_port=port
                                )
                                true_qps = state_info['true_qps']
                                diff = abs(true_qps - running_qps)/running_qps
                                if diff>0.05 and running_qps>=true_qps:
                                    if prev_qps==running_qps:
                                        flag=False
                                    elif running_qps>prev_qps:
                                        running_qps-=0.03
                                    hit_top=True
                                else:
                                    output_payload={
                                        'params':{
                                            'quantization':quantization,
                                            'kv_cache_dtype':kv_cache_dtype,
                                            'pp_size':pp_size,
                                            'tp_size':tp_size,
                                            'split':split_value,
                                            'mem_fraction_static':mem_fraction_static,
                                            'context-length':ctx_len,
                                            'max-prefill-tokens':prefill_tok,
                                            'input_len':input_len,
                                            'num_samples':num_samples,
                                            'max_out_len':max_out_len,
                                            'running_qps':running_qps
                                        },
                                        'metrics':{
                                            'qps':running_qps,
                                            'true_qps':true_qps,
                                            'avg_latency':state_info['avg_latency'],
                                        }
                                    }
                                    with open(save_path,'a') as f:
                                        f.write(json.dumps(output_payload)+"\n")
                                    prev_qps=running_qps
                                    if hit_top:
                                        flag=False
                                    else:
                                        running_qps+=qps_gap
                        except Exception as e:
                            logging.error("Exception during benchmark: %s", e)
                            # Assume service crashed, restart and reset qps
                            launch_service(model_path, serv_cfg_full)
                            running_qps = data_config['qps_start'][idx]
                            prev_qps = running_qps
                            continue

            # 组合结束后移除文件 handler，避免下一个组合重复写同一文件
            logging.getLogger().removeHandler(qps_file_handler)

    finally:
        # 全部组合完成后，占位 GPU
        run_gpu_occ(gpus_for_service)

    logging.info("All test runs finished.")

