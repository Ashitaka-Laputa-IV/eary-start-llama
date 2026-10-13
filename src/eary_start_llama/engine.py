"""进程层：真正启动 / 停止 `llama serve`，处理日志与来源解析。

职责：
  - 把 "user/model[:quant]" 解析到本地 HF 缓存里的真实 gguf 路径
  - 启动子进程（独立会话 + 重定向日志到文件），等待端口就绪
  - 停止（先 SIGTERM，超时再 SIGKILL，按进程组）
  - 读取 / 跟随日志
  - 从 `llama serve -cl` 发现缓存里的模型（用于首次导入清单）
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import state
from .config import LOG_DIR, Model

LLAMA_BIN = os.environ.get("ESL_LLAMA_BIN") or shutil.which("llama") or "llama"

HF_HUB = Path(
    os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
)

DEFAULT_PORT = 9931


class StartError(RuntimeError):
    """启动失败，附带日志末尾用于排查。"""

    def __init__(self, message: str, tail: str = "") -> None:
        super().__init__(message)
        self.tail = tail


# --------------------------------------------------------------------------- #
# 来源解析
# --------------------------------------------------------------------------- #
def resolve_source_path(source: str) -> Path | None:
    """把 source 映射到本地 gguf 路径；找不到返回 None。

    - 以 .gguf 结尾或本身是存在的路径 -> 直接当本地文件
    - 否则当作 HF 仓库 "user/model[:quant]" -> 在 HF 缓存里找匹配的 *.gguf
    """
    expanded = Path(source).expanduser()
    if source.endswith(".gguf") or expanded.exists():
        return expanded if expanded.exists() else None

    repo, _, quant = source.partition(":")
    base = HF_HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("*/*.gguf"))
    if quant:
        for cand in candidates:
            if quant.lower() in cand.name.lower():
                return cand
    return candidates[0] if candidates else None


def describe(model: Model) -> tuple[str, str]:
    """返回 (quant, 人类可读体积) 用于展示。"""
    _, _, quant = model.source.partition(":")
    size = "-"
    path = resolve_source_path(model.source)
    if path is not None and path.exists():
        size = _human_size(path.stat().st_size)
    return (quant or "-", size)


def derive_name(repo: str) -> str:
    """从一个 HF 仓库名猜一个短名，例如 bge-m3-Q8_0-GGUF -> bge-m3。"""
    stem = repo.split("/")[-1]
    stem = re.sub(r"[-_]?gguf$", "", stem, flags=re.I)
    stem = re.sub(r"-(q\d[\w.]*)$", "", stem, flags=re.I)
    return stem.lower() or repo.split("/")[-1].lower()


def discover() -> dict[str, Model]:
    """运行 `llama serve -cl`，把缓存里的模型导入成初始清单。"""
    try:
        proc = subprocess.run(
            [LLAMA_BIN, "serve", "-cl"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    models: dict[str, Model] = {}
    for line in proc.stdout.splitlines():
        match = re.match(r"\s*\d+\.\s+(\S+)\s*$", line)
        if not match:
            continue
        spec = match.group(1)  # "user/model:QUANT"
        repo, _, quant = spec.partition(":")
        name = _unique_name(derive_name(repo), models)
        source = repo + (f":{quant}" if quant else "")
        models[name] = Model(name=name, source=source)
    return models


# --------------------------------------------------------------------------- #
# 启动 / 停止
# --------------------------------------------------------------------------- #
def build_command(model: Model, port: int) -> list[str]:
    """组装真实的 `llama serve` 命令。

    日志改用进程 stdout 重定向（等价于 --log-file，但能连带捕获崩溃信息）。
    """
    cmd = [
        LLAMA_BIN,
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    local = resolve_source_path(model.source)
    if local is not None:
        cmd += ["-m", str(local)]
    else:
        # 本地没有就交给 llama 自己去 HF 拉取
        cmd += ["-hf", model.source]

    if model.args:
        cmd += shlex.split(model.args)
    return cmd


def start(
    model: Model,
    *,
    port: int | None = None,
    timeout: float = 180.0,
) -> state.Runtime:
    """启动模型服务，等待端口就绪；失败抛 StartError。"""
    used_ports = {rt.port for rt in state.live().values()}
    chosen = _free_port(port or model.port, used_ports)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"{model.name}.log"
    cmd = build_command(model, chosen)

    log = open(logfile, "wb")  # 每次启动覆盖旧日志
    log.write(("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n").encode())
    log.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        log.close()
        raise StartError(f"无法执行 {LLAMA_BIN}: {exc}") from exc

    if not _wait_ready(proc, chosen, timeout):
        _kill(proc.pid, signal.SIGKILL)
        log.close()
        raise StartError(f"{model.name} 启动失败或超时", tail=tail(logfile))

    log.close()
    runtime = state.Runtime(
        name=model.name,
        pid=proc.pid,
        port=chosen,
        started_at=time.time(),
        logfile=str(logfile),
    )
    state.set_running(runtime)
    return runtime


def stop(name: str, *, timeout: float = 15.0) -> bool:
    """停止模型服务；返回是否真的停掉了正在运行的进程。"""
    runtime = state.live().get(name)
    if runtime is None:
        state.clear(name)
        return False

    if state.is_alive(runtime.pid):
        _kill(runtime.pid, signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline and state.is_alive(runtime.pid):
            time.sleep(0.2)
        if state.is_alive(runtime.pid):
            _kill(runtime.pid, signal.SIGKILL)

    state.clear(name)
    return True


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def tail(logfile: str | Path, lines: int = 15) -> str:
    try:
        content = Path(logfile).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])


def read_log(logfile: str | Path) -> list[str]:
    try:
        content = Path(logfile).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return content.splitlines()


def follow_log(logfile: str | Path):
    """生成器：先吐已有内容，再持续跟随追加（类似 tail -f）。"""
    path = Path(logfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                yield line.rstrip("\n")
            else:
                time.sleep(0.3)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _unique_name(name: str, models: dict[str, Model]) -> str:
    if name not in models:
        return name
    i = 2
    while f"{name}-{i}" in models:
        i += 1
    return f"{name}-{i}"


def _free_port(preferred: int | None, used: set[int]) -> int:
    port = preferred or DEFAULT_PORT
    while port in used or not _port_free(port):
        port += 1
    return port


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 与 llama server 行为对齐：允许复用处于 TIME_WAIT 的端口，
        # 否则频繁启停会把可用端口误判为占用，导致端口号一路漂移。
        # 对真正处于 LISTEN 的端口，bind 仍会因 EADDRINUSE 失败。
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_ready(proc: subprocess.Popen, port: int, timeout: float) -> bool:
    """轮询直到服务就绪、进程退出或超时。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:  # 进程已退出 = 启动失败
            return False
        if _healthy(port):
            return True
        time.sleep(0.3)
    return False


def _healthy(port: int) -> bool:
    """优先用 /health；不支持该端点时退化为 TCP 可连判定。"""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        # 404 = 该版本没有 /health 端点 -> 退化为端口探测
        if exc.code == 404:
            return _port_open(port)
        return False  # 503 等 = 仍在加载
    except Exception:
        return False


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _kill(pid: int, sig: int) -> None:
    """按进程组发信号；失败则退回单进程。"""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} TiB"
