#!/usr/bin/env python3
"""
一体化服务：同时运行 llama-server（路由模式）和 sd-cli HTTP 服务。
收到 /generate 请求时，自动卸载 LLM 模型并调用 sd-cli.exe 生成图像。
"""

import subprocess
import sys
import os
import json
import signal
import atexit
import time
from typing import Optional, Dict, Any
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==================== 默认配置 ====================
DEFAULT_CONFIG = {
    "sd_cli_exe": "sd-cli.exe",  # sd-cli.exe 路径（可绝对路径）
    "llama_server_cmd": [  # llama-server 启动命令（必须使用路由模式）
        "llama-server",
        "--host",
        "0.0.0.0",
        "--port",
        "11434",
        "--models-preset",
        "config.ini",
    ],
    "llama_server_url": "http://localhost:11434",
    "sd_defaults": {  # sd-cli 默认参数（可被请求覆盖）
        "diffusion_model": r"",
        "llm": r"",
        "vae": r"",
        "lora_dir": r"",
        "negative_prompt": "错误的肢体，错误的关节",
        "cfg_scale": 1.0,
        "steps": 9,
        "height": 1920,
        "width": 1440,
        "offload_to_cpu": False,
        "diffusion_fa": True,
        "verbose": False,
    },
    "http_port": 8000,  # sd 服务的端口
}


# ==================== 工具函数 ====================
def load_config(config_path: str = "sd_config.json") -> Dict[str, Any]:
    """加载配置文件，若不存在则创建默认配置"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"已创建默认配置文件: {config_path}")
        return DEFAULT_CONFIG.copy()


def unload_llm_model(server_url: str, model_name: Optional[str] = None) -> bool:
    """通过 llama-server 的 /models/unload 接口卸载模型"""
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 提取模型列表（兼容两种可能的格式）
        if isinstance(data, dict) and "data" in data:
            models_list = data["data"]
        elif isinstance(data, list):
            models_list = data
        else:
            print("无法解析 /models 响应格式", file=sys.stderr)
            return False

        # 找出所有已加载的模型（status.value == 'loaded'）
        loaded_models = [
            m
            for m in models_list
            if isinstance(m, dict) and m.get("status", {}).get("value") == "loaded"
        ]

        if not loaded_models:
            print("没有已加载的模型，无需卸载", file=sys.stderr)
            return True

        # 确定要卸载的目标模型 id
        target = model_name if model_name else loaded_models[0]["id"]
        # 检查目标是否在已加载列表中
        if not any(m["id"] == target for m in loaded_models):
            print(f"模型 {target} 未加载，跳过卸载", file=sys.stderr)
            return True

        unload_resp = requests.post(
            f"{server_url.rstrip('/')}/models/unload",
            json={"model": target},
            timeout=10,
        )
        unload_resp.raise_for_status()
        print(f"已卸载模型: {target}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"卸载模型失败: {e}", file=sys.stderr)
        return False


def run_sd_cli(
    cfg: Dict[str, Any], cli_args: Dict[str, Any]
) -> subprocess.CompletedProcess:
    """调用 sd-cli.exe 生成图像"""
    defaults = cfg["sd_defaults"]
    params = {**defaults, **cli_args}

    cmd = [
        cfg["sd_cli_exe"],
        "--diffusion-model",
        params["diffusion_model"],
        "--vae",
        params["vae"],
        "--llm",
        params["llm"],
        "--output",
        params.get("output", "output.png"),
        "-p",
        params["prompt"],
        "-n",
        params.get("negative_prompt", ""),
        "--lora-model-dir",
        params.get("lora_dir", ""),
        "--steps",
        str(params.get("steps", 9)),
        "--cfg-scale",
        str(params.get("cfg_scale", 1.0)),
        "-H",
        str(params.get("height", 1920)),
        "-W",
        str(params.get("width", 1440)),
        "--vae-tiling",
    ]

    if params.get("verbose"):
        cmd.append("-v")
    if params.get("offload_to_cpu"):
        cmd.append("--offload-to-cpu")
    if params.get("diffusion_fa"):
        cmd.append("--diffusion-fa")

    capture = params.get("verbose", False)
    return subprocess.run(cmd, check=True, capture_output=capture, text=True)


# ==================== HTTP 服务 ====================
app = FastAPI(title="sd-cli service with llama.cpp unload")
server_config: Dict[str, Any] = {}


class GenerateRequest(BaseModel):
    prompt: str
    output: str = "output.png"
    steps: Optional[int] = None  # 0-10，实际 sd-cli 可能支持任意正整数
    height: Optional[int] = None
    width: Optional[int] = None
    negative_prompt: str = ""


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    """生成图像，调用前自动卸载 LLM 模型"""
    global server_config
    # 1. 卸载 LLM 模型（释放显存）
    if not unload_llm_model(server_config["llama_server_url"]):
        print("警告：卸载 LLM 模型失败，继续生成图像...", file=sys.stderr)

    # 2. 构建 sd-cli 参数（合并默认值）
    defaults = server_config["sd_defaults"]
    cli_args = {
        "prompt": req.prompt,
        "output": req.output,
        "steps": req.steps if req.steps is not None else defaults.get("steps", 9),
        "height": (
            req.height if req.height is not None else defaults.get("height", 1920)
        ),
        "width": req.width if req.width is not None else defaults.get("width", 1440),
        "negative_prompt": defaults.get("negative_prompt", "") + req.negative_prompt,
        "cfg_scale": defaults.get("cfg_scale", 1.0),
        "verbose": defaults.get("verbose", False),
        "offload_to_cpu": defaults.get("offload_to_cpu", False),
        "diffusion_fa": defaults.get("diffusion_fa", True),
        "diffusion_model": defaults["diffusion_model"],
        "llm": defaults["llm"],
        "vae": defaults["vae"],
        "lora_dir": defaults.get("lora_dir", ""),
    }

    try:
        result = run_sd_cli(server_config, cli_args)
        return {
            "status": "success",
            "output": cli_args["output"],
            "stdout": result.stdout if cli_args["verbose"] else None,
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"sd-cli failed: {e.stderr}")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ==================== llama-server 管理 ====================
llama_process: Optional[subprocess.Popen] = None


def start_llama_server(config: Dict[str, Any]):
    """启动 llama-server 子进程（后台运行）"""
    global llama_process
    cmd = config["llama_server_cmd"]
    server_url = config["llama_server_url"]
    try:
        # 使用 Popen 使其后台运行
        llama_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        # 等待服务就绪
        for _ in range(30):
            try:
                requests.get(f"{server_url}/health", timeout=1)
                print(f"llama-server 已就绪,{server_url}")
                return
            except requests.RequestException:
                time.sleep(1)
        print("警告：llama-server 启动超时", file=sys.stderr)
    except Exception as e:
        print(f"启动 llama-server 失败: {e}", file=sys.stderr)
        raise


def shutdown_llama_server():
    """关闭 llama-server 子进程"""
    global llama_process
    if llama_process and llama_process.poll() is None:
        print("正在关闭 llama-server...")
        if os.name == "nt":
            llama_process.terminate()
        else:
            llama_process.send_signal(signal.SIGTERM)
        try:
            llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_process.kill()
        print("llama-server 已关闭")


# ==================== 主入口 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="一体化服务：llama-server + sd-cli HTTP"
    )
    parser.add_argument("--config", default="sd_config.json", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    global server_config
    server_config = config

    # 启动 llama-server 子进程
    start_llama_server(config)
    atexit.register(shutdown_llama_server)

    # 启动 HTTP 服务（当前进程）
    print(f"sd-cli HTTP 服务启动在 http://0.0.0.0:{config['http_port']}")
    uvicorn.run(app, host="0.0.0.0", port=config["http_port"])


if __name__ == "__main__":
    main()
