"""自动重试的断点续传下载：应对代理对大文件不稳定的情况。

反复调用 snapshot_download（huggingface_hub 会从 .incomplete 续传），
每轮带超时；直到全部文件就绪或达到最大轮数。
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

REPO = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
MAX_TRIES = int(os.environ.get("MAX_TRIES", "200"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


def main():
    for i in range(1, MAX_TRIES + 1):
        try:
            p = snapshot_download(
                REPO,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*", "*.model"],
                max_workers=1,
            )
            print(f"DONE -> {p}", flush=True)
            return
        except Exception as e:
            print(f"[try {i}] 失败，续传重试: {repr(e)[:160]}", flush=True)
            time.sleep(3)
    print("EXHAUSTED: 达到最大重试轮数仍未完成", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
