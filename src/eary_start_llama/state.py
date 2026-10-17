"""状态层：运行状态（state.json）。

这份文件是"程序自己写的现在是什么状态"：谁在跑、PID、端口、启动时间。

两个关键点：
  1. 不能只信这份 json：每次读取都会用 `os.kill(pid, 0)` 校验进程存活，
     死掉的记录自动清理，避免"假 running"。
  2. 并发安全：同时启动/停止多个模型时会并发读改写，这里用
     "线程锁 + 文件锁 + 原子替换" 保证记录不会互相覆盖、文件不会写坏。
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from .config import STATE_DIR, STATE_FILE

try:  # Windows 没有 fcntl，退化为仅进程内线程锁
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

_LOCK_FILE = STATE_DIR / "state.lock"
_THREAD_LOCK = threading.RLock()


@dataclass
class Runtime:
    name: str
    pid: int
    port: int
    started_at: float
    logfile: str

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def uptime(self) -> str:
        return _humanize(time.time() - self.started_at)


def is_alive(pid: int) -> bool:
    """进程是否存活（信号 0 探测）。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但不属于当前用户
        return True
    return True


@contextmanager
def _exclusive():
    """进程内线程锁 + 跨进程文件锁，保证"读-改-写"整体原子。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK:
        if fcntl is None:
            yield
            return
        with open(_LOCK_FILE, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read() -> dict[str, Runtime]:
    if not STATE_FILE.exists():
        return {}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    runtimes: dict[str, Runtime] = {}
    for name, item in raw.items():
        if not isinstance(item, dict):
            continue
        try:
            runtimes[name] = Runtime(**item)
        except TypeError:
            continue
    return runtimes


def _write(runtimes: dict[str, Runtime]) -> None:
    """原子写入：先写临时文件再 os.replace，读者永远不会读到半个文件。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {name: asdict(rt) for name, rt in runtimes.items()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, STATE_FILE)


def load() -> dict[str, Runtime]:
    """读取原始状态，不做存活校验。"""
    with _exclusive():
        return _read()


def save(runtimes: dict[str, Runtime]) -> None:
    with _exclusive():
        _write(runtimes)


def live() -> dict[str, Runtime]:
    """返回真正活着的运行项，并顺手清理掉死掉的记录。"""
    with _exclusive():
        runtimes = _read()
        alive = {name: rt for name, rt in runtimes.items() if is_alive(rt.pid)}
        if alive.keys() != runtimes.keys():
            _write(alive)
        return alive


def set_running(rt: Runtime) -> None:
    with _exclusive():
        runtimes = _read()
        runtimes[rt.name] = rt
        _write(runtimes)


def clear(name: str) -> None:
    with _exclusive():
        runtimes = _read()
        if name in runtimes:
            del runtimes[name]
            _write(runtimes)


def clear_if_match(name: str, pid: int) -> None:
    """仅当记录仍指向该 pid 时才删除。

    用于"启动失败后清理"：并发场景下，若该模型已被另一次成功启动覆盖，
    就不能误删新记录。
    """
    with _exclusive():
        runtimes = _read()
        current = runtimes.get(name)
        if current is not None and current.pid == pid:
            del runtimes[name]
            _write(runtimes)


def _humanize(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours, total = divmod(total, 3600)
    minutes, seconds = divmod(total, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"
