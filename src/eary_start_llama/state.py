"""状态层：运行状态（state.json）。

这份文件是"程序自己写的现在是什么状态"：谁在跑、PID、端口、启动时间。

关键点：**不能只信这份 json**。每次读取都会用 `os.kill(pid, 0)`
校验进程是否真的活着，死掉的记录会被自动清理，避免出现"假 running"。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

from .config import STATE_DIR, STATE_FILE


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


def load() -> dict[str, Runtime]:
    """读取原始状态，不做存活校验。"""
    if not STATE_FILE.exists():
        return {}
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        name: Runtime(**item)
        for name, item in raw.items()
        if isinstance(item, dict)
    }


def save(runtimes: dict[str, Runtime]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {name: asdict(rt) for name, rt in runtimes.items()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def live() -> dict[str, Runtime]:
    """返回真正活着的运行项，并顺手清理掉死掉的记录。"""
    runtimes = load()
    alive = {name: rt for name, rt in runtimes.items() if is_alive(rt.pid)}
    if alive.keys() != runtimes.keys():
        save(alive)
    return alive


def set_running(rt: Runtime) -> None:
    runtimes = load()
    runtimes[rt.name] = rt
    save(runtimes)


def clear(name: str) -> None:
    runtimes = load()
    if name in runtimes:
        del runtimes[name]
        save(runtimes)


def _humanize(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours, total = divmod(total, 3600)
    minutes, seconds = divmod(total, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"
