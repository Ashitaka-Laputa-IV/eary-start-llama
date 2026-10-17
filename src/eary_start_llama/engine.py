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


_DESCRIBE_CACHE: dict[str, tuple[str, str]] = {}


def describe(model: Model) -> tuple[str, str]:
    """返回 (quant, 人类可读体积) 用于展示。

    按 source 缓存结果：界面每 2 秒刷新一次，模型文件不会变，
    避免反复遍历 HF 缓存目录拖慢刷新。
    """
    cached = _DESCRIBE_CACHE.get(model.source)
    if cached is not None:
        return cached
    _, _, quant = model.source.partition(":")
    size = "-"
    path = resolve_source_path(model.source)
    if path is not None and path.exists():
        size = _human_size(path.stat().st_size)
    result = (quant or "-", size)
    _DESCRIBE_CACHE[model.source] = result
    return result


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
        # 启动失败/中途崩溃：确保杀干净并回收子进程，否则会留下僵尸进程，
        # 崩溃实例占用的内存/显存也无法及时释放。
        _terminate(proc)
        log.close()
        state.clear_if_match(model.name, proc.pid)
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


def _terminate(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """杀掉进程并 wait 回收，避免僵尸进程与资源泄漏。"""
    _kill(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    # 仍未退出：再补一刀（可能刚派生出子进程）
    _kill(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


# --------------------------------------------------------------------------- #
# 进程扫描与状态对齐
# --------------------------------------------------------------------------- #
def _proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _is_llama_serve(args: list[str]) -> bool:
    if not args:
        return False
    return "llama" in os.path.basename(args[0]) and "serve" in args


def _parse_llama_args(args: list[str]) -> tuple[str | None, int | None, bool]:
    """从命令行解析 (模型引用, 端口, 是否 HF 仓库)。"""
    model_ref: str | None = None
    port: int | None = None
    is_hf = False
    i = 1
    while i < len(args):
        arg = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None
        if arg in ("-m", "--model") and nxt:
            model_ref, is_hf = nxt, False
            i += 2
            continue
        if arg in ("-hf", "--hf-repo", "-hfr") and nxt:
            model_ref, is_hf = nxt, True
            i += 2
            continue
        if arg == "--port" and nxt:
            try:
                port = int(nxt)
            except ValueError:
                pass
            i += 2
            continue
        if arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        i += 1
    return model_ref, port, is_hf


def _proc_started_at(pid: int) -> float:
    """从 /proc/<pid>/stat 推算进程启动时间戳。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        rest = stat.rsplit(")", 1)[-1].split()
        start_ticks = int(rest[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return time.time() - (uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, IndexError, ValueError):
        return time.time()


def scan_running(models: dict[str, Model]) -> list[state.Runtime]:
    """扫描系统中真实的 `llama serve` 进程，重建运行项列表。

    用于对齐 state.json：机器重启、手动启动、状态文件丢失时，
    也能重新认出正在运行的服务。
    """
    if not Path("/proc").is_dir():
        return []

    by_path: dict[str, str] = {}
    by_source: dict[str, str] = {}
    for name, model in models.items():
        by_source[model.source] = name
        path = resolve_source_path(model.source)
        if path is not None:
            by_path[str(path)] = name
            try:
                by_path[str(path.resolve())] = name
            except OSError:
                pass

    found: list[state.Runtime] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        args = _proc_cmdline(pid)
        if not _is_llama_serve(args):
            continue

        ref, port, is_hf = _parse_llama_args(args)
        if ref is None or port is None:
            continue

        if is_hf:
            name = by_source.get(ref)
        else:
            name = by_path.get(ref)
            if name is None:
                try:
                    name = by_path.get(str(Path(ref).resolve()))
                except OSError:
                    pass
        if name is None:
            continue

        found.append(
            state.Runtime(
                name=name,
                pid=pid,
                port=port,
                started_at=_proc_started_at(pid),
                logfile=str(LOG_DIR / f"{name}.log"),
            )
        )
    return found


def reconcile(models: dict[str, Model]) -> dict[str, state.Runtime]:
    """让 state 与真实进程对齐，返回对齐后的运行项。

    - 发现运行中但未登记的 -> 补登记
    - 登记了但进程已不是 llama 的（含 pid 被复用）-> 清理
    """
    found = {rt.name: rt for rt in scan_running(models)}

    known = state.load()
    for rt in found.values():
        current = known.get(rt.name)
        if current is None or current.pid != rt.pid:
            state.set_running(rt)

    for name, rt in known.items():
        if name in found:
            continue
        if not _is_llama_serve(_proc_cmdline(rt.pid)):
            state.clear_if_match(name, rt.pid)

    return state.live()


def _human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} TiB"
